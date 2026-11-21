"""Training and scoring cost of each method on one rolling-origin window (h = 4, test year 2025).

Every method runs in a fresh process with 16 threads, so its peak resident memory is measured alone.
Scoring time covers all eight screens of the test year.
"""
import os, sys, time, json, subprocess, argparse
import numpy as np, pandas as pd

METHODS = ['margin', 'proximity', 'logit', 'gbm', 'gbm-pooled', 'surface', 'qcop']


def one(method, h=4, Y=2025, threads=16):
    import psutil
    os.environ['GBM_THREADS'] = str(threads)
    import models as M, run_experiments as RE
    from screens import RULEBOOKS, UNSEEN, tau_vec
    M.GBM_PARAMS['num_threads'] = threads
    X = pd.read_parquet('data/tasks.parquet')
    tr, te = RE.split(X, h, Y)
    Rf = tr[[f'{k}_f{h}' for k in M.RATIOS]].values
    screens = dict(RULEBOOKS); screens.update(UNSEEN)
    t0 = time.time(); n_models = 1
    if method in ('margin', 'proximity'):
        Ds, ys, Ts = RE.stack_published(tr, h)
        m = M.LogitModel(M.cols_margin if method == 'margin' else M.cols_proximity).fit(Ds, ys, Ts); m = RE.Pooled(m)
        fitted = {s: m for s in screens}
    elif method in ('logit', 'gbm'):
        fitted = {}
        for name in RULEBOOKS:
            d, y = RE.rows_for(tr, h, tau_vec(RULEBOOKS[name]))
            fitted[name] = (M.LogitModel(M.cols_full) if method == 'logit' else M.GBMDirect(M.BASE_FEATS, 0)).fit(d, y, tau_vec(RULEBOOKS[name]))
        for s in UNSEEN:
            fitted[s] = fitted[RE.NEAREST[s]]
        n_models = len(RULEBOOKS)
    elif method == 'gbm-pooled':
        Ds, ys, Ts = RE.stack_published(tr, h)
        m = RE.Pooled(M.GBMDirect(M.BASE_FEATS, 0).fit(Ds, ys, Ts)); fitted = {s: m for s in screens}
    elif method == 'surface':
        m = M.Surface(seed=0).fit(tr, Rf); fitted = {s: m for s in screens}
    elif method == 'qcop':
        m = M.QuantCopula(seed=0).fit(tr, Rf); fitted = {s: m for s in screens}; n_models = 3 * len(M.QLEVELS)
    fit_s = time.time() - t0
    t0 = time.time(); n = 0
    for s, rb in screens.items():
        d, y = RE.rows_for(te, h, tau_vec(rb)); fitted[s].predict(d, tau_vec(rb)); n += len(d)
    pred_s = time.time() - t0
    peak = psutil.Process().memory_info().peak_wset / 2 ** 20
    return dict(method=method, fit_s=fit_s, score_s=pred_s, scored=n, score_ms_per_1k=pred_s / n * 1e6,
                peak_mb=peak, n_train=len(tr), models=n_models)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--method'); ap.add_argument('--out', default='results/cost.csv')
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if a.method:
        print(json.dumps(one(a.method)))
    else:
        rows = []
        for m in METHODS:
            r = subprocess.run([sys.executable, __file__, '--method', m], capture_output=True, text=True)
            line = [l for l in r.stdout.splitlines() if l.startswith('{')]
            if line:
                rows.append(json.loads(line[-1])); print(rows[-1], flush=True)
            else:
                print('failed', m, r.stderr[-2000:])
        pd.DataFrame(rows).to_csv(a.out, index=False)
