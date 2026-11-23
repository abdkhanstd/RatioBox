"""Robustness reanalyses of the saved forecasts (no retraining).

1. tag changes: breaches that coincide with a change of tag status (missing debt, cash or receivables tags) between
   the decision quarter and the target quarter, and metrics without those decisions;
2. listed firms: metrics and the admission analysis for firms listed on NYSE or Nasdaq;
3. attrition: share of decisions without a filed balance sheet h quarters later, and a bound on the breach rate;
4. quarter-block bootstrap of the macro AUC difference (decision quarters resampled instead of firms);
5. size of coherence violations;
6. quarter-block bootstrap on the subsets of item 1 and 2 (one year ahead).
All metrics use seed-averaged forecasts. Outputs: results/robust_*.csv. --parts selects items (default all).
"""
import glob, os, argparse, numpy as np, pandas as pd
from evaluate import metrics, GROUPS, boot_p
from screens import RATIOS

MAIN = ['margin', 'proximity', 'logit', 'gbm', 'gbm-nearest', 'gbm-pooled', 'surface', 'gbm-oracle', 'qcop']


def load(pred_dir, pattern='h*_Y*_*.parquet'):
    P = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(pred_dir, pattern))], ignore_index=True)
    return P.groupby(['h', 'Y', 'screen', 'method', 'cik', 'qi'], as_index=False).agg(p=('p', 'mean'), y=('y', 'first'))


def macro(A, methods, tag):
    rows = []
    for (h, m), g in A[A.method.isin(methods)].groupby(['h', 'method']):
        for gname, scr in GROUPS.items():
            ms = [metrics(s.y.values, s.p.values) for _, s in g[g.screen.isin(scr)].groupby('screen') if s.y.nunique() == 2]
            if len(ms) == len(scr):
                rows.append(dict(subset=tag, h=h, group=gname, method=m, n=int(g[g.screen.isin(scr)].shape[0]),
                                 **{k: np.mean([x[k] for x in ms]) for k in ['auc', 'brier', 'ece', 'logloss']}))
    return rows


def tag_status(T, P):
    """tag flags at the decision quarter and at the target quarter of every decision."""
    P = P.copy()
    P['qi'] = P.q.str[:4].astype(int) * 4 + P.q.str[-1].astype(int) - 1
    F = P[['cik', 'qi', 'debt_tagged', 'ar_missing', 'cash_missing']].drop_duplicates(['cik', 'qi'])
    return F.set_index(['cik', 'qi']).astype(float)


def quarter_boot(A, a, b, scr, h, reps, rng):
    """macro AUC difference a minus b with decision quarters resampled (time-clustered bootstrap)."""
    blocks = []
    for s in scr:
        g = A[(A.h == h) & (A.screen == s)]
        j = g[g.method == a].set_index(['cik', 'qi'])[['p', 'y']].join(g[g.method == b].set_index(['cik', 'qi'])[['p']], rsuffix='_b', how='inner')
        blocks.append(j)
    qs = np.unique(np.concatenate([j.index.get_level_values(1).values for j in blocks]))
    pos = {q: i for i, q in enumerate(qs)}
    idx = [np.array([pos[q] for q in j.index.get_level_values(1)]) for j in blocks]
    def diff(w):
        d = []
        for j, ix in zip(blocks, idx):
            ww = None if w is None else w[ix].astype(float)
            d.append(metrics(j.y.values, j.p.values, ww)['auc'] - metrics(j.y.values, j.p_b.values, ww)['auc'])
        return np.mean(d)
    d0 = diff(None)
    D = np.array([diff(rng.multinomial(len(qs), np.full(len(qs), 1 / len(qs)))) for _ in range(reps)])
    return d0, np.percentile(D, 2.5), np.percentile(D, 97.5), boot_p(D), len(qs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', default='results/preds'); ap.add_argument('--legacy', default='results/preds_legacy')
    ap.add_argument('--out', default='results'); ap.add_argument('--reps', type=int, default=500)
    ap.add_argument('--parts', default='subsets,attrition,qboot,coherence,subboot')
    a = ap.parse_args()
    parts = set(a.parts.split(','))
    T = pd.read_parquet('data/tasks.parquet')
    A = load(a.preds)
    A = A[A.method.isin(MAIN)]
    rows = macro(A, MAIN, 'all') if 'subsets' in parts else []

    # 1. tag changes between the decision and the target quarter
    F = tag_status(T, pd.read_parquet('data/panel.parquet'))
    for h in [1, 4]:
        sel = A.h == h
        now = F.reindex(pd.MultiIndex.from_arrays([A.loc[sel, 'cik'], A.loc[sel, 'qi']])).values
        fut = F.reindex(pd.MultiIndex.from_arrays([A.loc[sel, 'cik'], A.loc[sel, 'qi'] + h])).values
        A.loc[sel, 'tagchg'] = (np.nan_to_num(now, nan=-1) != np.nan_to_num(fut, nan=-1)).any(axis=1)
    tc = []
    for (h, s), g in A[A.method == 'qcop'].groupby(['h', 'screen']):
        tc.append(dict(h=h, screen=s, n=len(g), breaches=int(g.y.sum()), share_decisions=g.tagchg.mean(),
                       share_breaches=g[g.y == 1].tagchg.mean()))
    ex = T.drop_duplicates('cik').set_index('cik').exchanges.fillna('')
    listed = ex[ex.str.contains('NYSE|Nasdaq', regex=True)].index
    AL = A[A.cik.isin(listed)]
    if 'subsets' in parts:
        pd.DataFrame(tc).to_csv(f'{a.out}/robust_tagchange.csv', index=False)
        rows += macro(A[~A.tagchg.astype(bool)], MAIN, 'no tag change')
        # 2. firms listed on NYSE or Nasdaq (current listing from the submissions archive)
        rows += macro(AL, MAIN, 'listed')
        pd.DataFrame(rows).to_csv(f'{a.out}/robust_subsets.csv', index=False)
    # admission analysis on listed firms (MSCI, h = 4, same rule as decision.py)
    D = AL[(AL.h == 4) & (AL.screen == 'MSCI')].merge(T[['cik', 'qi'] + RATIOS], on=['cik', 'qi'])
    D['entry_ok'] = np.all(D[RATIOS].values < 0.30, axis=1)
    adm = []
    for m in ['gbm', 'surface', 'qcop']:
        g = D[D.method == m]; rule, fc, nq = [], [], 0
        for qi, gq in g.groupby('qi'):
            k = int(gq.entry_ok.sum()); rule.append(gq[gq.entry_ok].y.values); fc.append(gq.nsmallest(k, 'p').y.values); nq += 1
        r, f = np.concatenate(rule).mean(), np.concatenate(fc).mean()
        adm.append(dict(method=m, quarters=nq, admitted=len(np.concatenate(fc)), fail_rule=r, fail_forecast=f, diff=f - r, rel=(r - f) / r))
    pd.DataFrame(adm).to_csv(f'{a.out}/robust_admission_listed.csv', index=False)

    # 3. attrition: eligible test decisions without a filed balance sheet h quarters later
    att = []
    for h in ([1, 4] if 'attrition' in parts else []):
        for Y in range(2019, 2026):
            d = T[(T.qi >= Y * 4) & (T.qi <= Y * 4 + 3)]
            d = d[d.qi + h <= 2026 * 4 + 1]   # targets after 2026Q2 may not be filed yet at the archive date
            miss = d[[f'{k}_f{h}' for k in RATIOS]].isna().any(axis=1)
            ok = np.all(d[RATIOS].values < 1 / 3, axis=1)
            fut = d.loc[ok & ~miss, [f'{k}_f{h}' for k in RATIOS]].values
            br = (~np.all(fut < 1 / 3, axis=1)).mean()
            att.append(dict(h=h, Y=Y, decisions=len(d), attrition=miss.mean(), msci_pass=int(ok.sum()),
                            msci_attrition=miss[ok].mean(), msci_breach=br,
                            msci_breach_bound=(br * (~miss[ok]).sum() + miss[ok].sum()) / ok.sum()))
    if att:
        pd.DataFrame(att).to_csv(f'{a.out}/robust_attrition.csv', index=False)

    # 4. quarter-block bootstrap of macro AUC differences
    rng = np.random.default_rng(3)
    qb = []
    for h in ([1, 4] if 'qboot' in parts else []):
        for gname, base in [('published', 'gbm'), ('published', 'surface'), ('unseen', 'surface'), ('unseen', 'gbm-oracle'), ('unseen', 'gbm-nearest')]:
            d0, lo, hi, p, nq = quarter_boot(A, 'qcop', base, GROUPS[gname], h, a.reps, rng)
            qb.append(dict(h=h, group=gname, baseline=base, d_auc=d0, lo=lo, hi=hi, p=p, quarters=nq))
    if qb:
        pd.DataFrame(qb).to_csv(f'{a.out}/robust_quarterboot.csv', index=False)

    # 5. size of coherence violations; 'refit' pairs a rulebook GBM on the published screens with the refit GBM
    #    on the unseen screens, the comparator a fund would use after relabelling and refitting
    if 'coherence' in parts:
        pairs = [('MSCI', 'FTSE'), ('MSCI', 'MSCI-M'), ('MSCI-M', 'FTSE'), ('MSCI-entry', 'MSCI'), ('MSCI', 'MSCI-rec70')]
        C = A[~A.method.isin(['gbm-oracle'])].copy(); C['method'] = C.method.replace({'gbm-nearest': 'gbm'})
        R = pd.concat([A[A.method == 'gbm'], A[A.method == 'gbm-oracle']]).assign(method='refit')
        C = pd.concat([C, R], ignore_index=True)
        cm = []
        for (h, m), g in C.groupby(['h', 'method']):
            piv = g.pivot_table(index=['cik', 'qi'], columns='screen', values='p')
            for s_, l in pairs:
                if m == 'refit' and 'MSCI-entry' not in (s_, l) and 'MSCI-rec70' not in (s_, l):
                    continue   # published pairs of 'refit' are those of the rulebook GBM
                if s_ in piv and l in piv:
                    d = piv[[s_, l]].dropna(); inv = np.maximum(0, d[l] - d[s_])
                    cm.append(dict(h=h, method=m, strict=s_, loose=l, n=len(d), viol=(inv > 1e-3).mean(),
                                   mean_inv=inv[inv > 1e-3].mean() if (inv > 1e-3).any() else 0.0,
                                   share_gt01=(inv > 0.01).mean(), share_gt05=(inv > 0.05).mean()))
        pd.DataFrame(cm).to_csv(f'{a.out}/robust_coherence_size.csv', index=False)

    # 6. quarter-block bootstrap on the subsets, one year ahead
    if 'subboot' in parts:
        sb = []
        for tag, S in [('no tag change', A[~A.tagchg.astype(bool)]), ('listed', AL)]:
            for gname, base in [('published', 'gbm'), ('unseen', 'surface'), ('unseen', 'gbm-oracle')]:
                d0, lo, hi, p, nq = quarter_boot(S, 'qcop', base, GROUPS[gname], 4, a.reps, rng)
                sb.append(dict(subset=tag, h=4, group=gname, baseline=base, d_auc=d0, lo=lo, hi=hi, p=p, quarters=nq))
        pd.DataFrame(sb).to_csv(f'{a.out}/robust_subset_boot.csv', index=False)
    print('done')


if __name__ == '__main__':
    main()
