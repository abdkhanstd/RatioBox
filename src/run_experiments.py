"""Rolling-origin out-of-time evaluation of breach forecasters.

For test year Y and horizon h, models are trained on decision quarters whose target quarter ends at
least one full quarter before Y starts (so every training label was filed before the first test decision),
and they forecast every decision quarter in Y. Predictions are written to results/preds/.
"""
import os, sys, time, argparse, json, tracemalloc
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from screens import RATIOS, RULEBOOKS, UNSEEN, tau_vec, OFF
import models as M

YEARS = list(range(2019, 2026))
HORIZONS = [1, 4]
NEAREST = {'MSCI-entry': 'MSCI', 'MSCI-rec70': 'FTSE', 'Debt-25': 'SC', 'Cash-40': 'SC'}
SEEDS = [int(s) for s in os.environ.get('RB_SEEDS', '0,1,2').split(',')]
# RB_STRICT_FILING=1 also drops training rows whose target balance sheet was first filed after the first test
# decision of the window (late filings, 0.05% of training labels; see filing_audit.py). Off for the main results.
STRICT_FILING = os.environ.get('RB_STRICT_FILING', '0') == '1'


def filing_dates():
    P = pd.read_parquet('data/panel.parquet', columns=['cik', 'q', 'avail'])
    P['qi'] = P.q.str[:4].astype(int) * 4 + P.q.str[-1].astype(int) - 1
    return P.drop_duplicates(['cik', 'qi']).set_index(['cik', 'qi']).avail


def split(X, h, Y, avail=None):
    fut = X[[f'{k}_f{h}' for k in RATIOS]].notna().all(axis=1)
    tr = X[(X.qi + h <= Y * 4 - 2) & fut & (X.qi >= 2012 * 4)]
    te = X[(X.qi >= Y * 4) & (X.qi <= Y * 4 + 3) & fut]
    if avail is not None:
        first = avail.reindex(pd.MultiIndex.from_arrays([te.cik, te.qi])).min()
        tgt = avail.reindex(pd.MultiIndex.from_arrays([tr.cik, tr.qi + h])).values
        tr = tr[~(tgt > first)]
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def rows_for(D, h, tau):
    Rt = D[RATIOS].values; Rf = D[[f'{k}_f{h}' for k in RATIOS]].values
    ok, fail = M.label(Rt, Rf, tau)
    return D[ok].reset_index(drop=True), fail[ok].astype(int)


def stack_published(D, h):
    parts, ys, taus = [], [], []
    for name, rb in RULEBOOKS.items():
        d, y = rows_for(D, h, tau_vec(rb))
        parts.append(d); ys.append(y); taus.append(np.tile(tau_vec(rb), (len(d), 1)))
    return pd.concat(parts, ignore_index=True), np.concatenate(ys), np.vstack(taus)


class Pooled:
    """wrap a per-threshold model so it is trained on the stacked published rulebooks."""
    def __init__(self, base): self.base = base
    def fit(self, D, y, T):
        self.base.fit(D, y, T); return self
    def predict(self, D, tau): return self.base.predict(D, np.tile(np.asarray(tau, float), (len(D), 1)))


def run_job(args):
    h, Y, group, out_dir = args
    M.GBM_PARAMS['num_threads'] = int(os.environ.get('GBM_THREADS', '8'))
    os.makedirs(out_dir, exist_ok=True)
    X = pd.read_parquet('data/tasks.parquet')
    tr, te = split(X, h, Y, filing_dates() if STRICT_FILING else None)
    Rf_tr = tr[[f'{k}_f{h}' for k in RATIOS]].values
    screens = dict(RULEBOOKS); screens.update(UNSEEN)
    recs, cost = [], []

    def emit(method, seed, model, scr_names, kind='native'):
        for s in scr_names:
            tau = tau_vec(screens[s])
            d, y = rows_for(te, h, tau)
            if len(d) == 0:
                continue
            t0 = time.time(); p = model(s).predict(d, tau) if callable(model) else model.predict(d, tau); dt = time.time() - t0
            recs.append(pd.DataFrame(dict(cik=d.cik.values, qi=d.qi.values, h=h, Y=Y, screen=s, method=method, seed=seed,
                                          kind=kind, p=np.clip(p, 1e-6, 1 - 1e-6), y=y)))
            cost.append(dict(method=method, seed=seed, h=h, Y=Y, screen=s, phase='predict', sec=dt, n=len(d)))

    def timed_fit(method, seed, f):
        tracemalloc.start(); t0 = time.time(); m = f(); dt = time.time() - t0
        peak = tracemalloc.get_traced_memory()[1] / 2 ** 20; tracemalloc.stop()
        cost.append(dict(method=method, seed=seed, h=h, Y=Y, screen='all', phase='fit', sec=dt, peak_mb=peak))
        return m

    pub = list(RULEBOOKS); uns = list(UNSEEN)
    if group == 'simple':
        # per-rulebook simple baselines
        for name in pub:
            d, y = rows_for(tr, h, tau_vec(RULEBOOKS[name]))
            T = tau_vec(RULEBOOKS[name])
            m = timed_fit('clim', 0, lambda: M.Clim().fit(d, y, T)); emit('clim', 0, m, [name])
            m = timed_fit('logit', 0, lambda: M.LogitModel(M.cols_full).fit(d, y, T)); emit('logit', 0, m, [name])
        # threshold-aware simple baselines trained on the stacked published rulebooks
        Ds, ys, Ts = stack_published(tr, h)
        m = timed_fit('margin', 0, lambda: M.LogitModel(M.cols_margin).fit(Ds, ys, Ts)); emit('margin', 0, Pooled(m), pub + uns)
        m = timed_fit('proximity', 0, lambda: M.LogitModel(M.cols_proximity).fit(Ds, ys, Ts)); emit('proximity', 0, Pooled(m), pub + uns)
        mc = timed_fit('clim', 0, lambda: M.Clim().fit(Ds, ys, Ts)); emit('clim', 0, mc, uns, kind='pooled')
    elif group.startswith('gbm'):
        feats = M.CURRENT_ONLY if group == 'gbm_cur' else M.BASE_FEATS
        tag = 'gbm' if group == 'gbm' else 'gbm-cur'
        for seed in SEEDS:
            fitted = {}
            for name in pub:
                d, y = rows_for(tr, h, tau_vec(RULEBOOKS[name]))
                fitted[name] = timed_fit(tag, seed, lambda: M.GBMDirect(feats, seed).fit(d, y, tau_vec(RULEBOOKS[name])))
                emit(tag, seed, fitted[name], [name])
            if group == 'gbm':
                emit('gbm-nearest', seed, lambda s: fitted[NEAREST[s]], uns, kind='transfer')
                # refit reference: a rulebook GBM relabelled and refitted on the unseen screen's own labels
                for s in uns:
                    d, y = rows_for(tr, h, tau_vec(UNSEEN[s]))
                    m = timed_fit('gbm-oracle', seed, lambda: M.GBMDirect(feats, seed).fit(d, y, tau_vec(UNSEEN[s])))
                    emit('gbm-oracle', seed, m, [s], kind='refit')
                # pooled direct model: stacked published rulebooks, no augmentation, no monotonicity
                Ds, ys, Ts = stack_published(tr, h)
                m = timed_fit('gbm-pooled', seed, lambda: M.GBMDirect(feats, seed).fit(Ds, ys, Ts))
                emit('gbm-pooled', seed, Pooled(m), pub + uns)
    elif group.startswith('surface'):
        cfg = {'surface': dict(), 'surface_nomono': dict(monotone=False), 'surface_noaug': dict(draws=0),
               'surface_cur': dict(feats=M.CURRENT_ONLY), 'surface_d8': dict(draws=8),
               'surface_indep': dict(uniform_active=False)}[group]
        tag = group.replace('_', '-')
        for seed in SEEDS:
            m = timed_fit(tag, seed, lambda: M.Surface(seed=seed, **cfg).fit(tr, Rf_tr))
            emit(tag, seed, m, pub + uns)
    elif group.startswith('qcop') or group == 'qpoint':
        feats = M.CURRENT_ONLY if group == 'qcop_cur' else M.BASE_FEATS
        tag = {'qcop': 'qcop', 'qcop_cur': 'qcop-cur', 'qpoint': 'qcop-point', 'qcop_earlier': 'qcop-earlier'}[group]
        cls = M.PointCopula if group == 'qpoint' else M.QuantCopula
        # earlier configuration (ablation): 300 rounds, 31 leaves, flat tails and 4,000 Monte Carlo draws
        kw = dict(rounds=300, leaves=31) if group == 'qcop_earlier' else {}
        keep = 'legacy' if group == 'qcop_earlier' else 'full'
        for seed in SEEDS:
            m = timed_fit(tag, seed, lambda: cls(feats=feats, seed=seed, **kw).fit(tr, Rf_tr))
            for s in pub + uns:
                tau = tau_vec(screens[s]); d, y = rows_for(te, h, tau)
                t0 = time.time(); V = m.predict_variants(d, tau, legacy=group == 'qcop_earlier'); dt = time.time() - t0
                cost.append(dict(method=tag, seed=seed, h=h, Y=Y, screen=s, phase='predict', sec=dt, n=len(d)))
                for vname, p in V.items():
                    if group != 'qcop' and vname != keep:
                        continue
                    name = tag if vname in ('full', 'legacy') else f'{tag}-{vname}'
                    recs.append(pd.DataFrame(dict(cik=d.cik.values, qi=d.qi.values, h=h, Y=Y, screen=s, method=name, seed=seed,
                                                  kind='native', p=np.clip(p, 1e-6, 1 - 1e-6), y=y)))
            if group == 'qcop' and seed == SEEDS[0]:
                np.save(f'{out_dir}/corr_h{h}_Y{Y}.npy', m.corr)
    os.makedirs(out_dir, exist_ok=True)
    if recs:
        pd.concat(recs).to_parquet(f'{out_dir}/h{h}_Y{Y}_{group}.parquet', index=False)
    pd.DataFrame(cost).to_csv(f'{out_dir}/cost_h{h}_Y{Y}_{group}.csv', index=False)
    return f'done h{h} Y{Y} {group} n_tr={len(tr)} n_te={len(te)}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', default='simple,gbm,surface,qcop')
    ap.add_argument('--years', default=','.join(map(str, YEARS)))
    ap.add_argument('--horizons', default='1,4')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', default='results/preds')
    a = ap.parse_args()
    jobs = [(int(h), int(Y), g, a.out) for g in a.groups.split(',') for h in a.horizons.split(',') for Y in a.years.split(',')]
    jobs = [j for j in jobs if not os.path.exists(f'{a.out}/h{j[0]}_Y{j[1]}_{j[2]}.parquet')]
    print('jobs', len(jobs), flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(run_job, jobs):
            print(r, flush=True)


if __name__ == '__main__':
    main()
