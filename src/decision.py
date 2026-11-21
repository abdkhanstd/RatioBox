"""Decision analysis: which screen-passing stocks should an Islamic index admit at a quarterly review?

At each review quarter t of the test years, the candidates are the firms that pass the MSCI-style screen
(all three ratios below 33.33%). The MSCI entry rule admits only candidates with all ratios below 30%.
We compare it with ranking rules that admit the same number of candidates in the same quarter:
lowest forecast probability of failing the screen one year later (h = 4), or largest minimum margin.
Outcomes: share of admitted stocks that fail the 33.33% screen at the review one year later (t+4), and share that
fail it at any of the next four quarterly reviews (admissions with all four later quarters filed).
"""
import numpy as np, pandas as pd, glob, argparse
from screens import RATIOS, RULEBOOKS, UNSEEN, tau_vec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', default='results/preds')
    ap.add_argument('--tasks', default='data/tasks.parquet')
    ap.add_argument('--out', default='results')
    ap.add_argument('--reps', type=int, default=1000)
    a = ap.parse_args()
    X = pd.read_parquet(a.tasks)
    P = pd.concat([pd.read_parquet(f) for f in glob.glob(f'{a.preds}/h4_Y*_*.parquet')], ignore_index=True)
    P = P[P.screen == 'MSCI'].groupby(['method', 'cik', 'qi'], as_index=False).agg(p=('p', 'mean'), y=('y', 'first'))
    tau = tau_vec(RULEBOOKS['MSCI'])
    X = X[['cik', 'qi'] + RATIOS + [f'{k}_f{h}' for k in RATIOS for h in [1, 2, 3, 4]]]
    X = X.merge(P[['cik', 'qi']].drop_duplicates(), on=['cik', 'qi'])
    Rt = X[RATIOS].values
    X['entry_ok'] = np.all(Rt < 0.30, axis=1)
    X['m_min'] = (tau - Rt).min(axis=1)
    fails = []
    for h in [1, 2, 3, 4]:
        Rf = X[[f'{k}_f{h}' for k in RATIOS]].values
        fails.append(np.where(np.isnan(Rf).any(axis=1), np.nan, (~np.all(Rf < tau, axis=1)).astype(float)))
    X['fail4'] = fails[3]
    F = np.column_stack(fails)
    X['fail_any'] = np.where(np.isnan(F).all(axis=1), np.nan, np.nanmax(np.nan_to_num(F, nan=0), axis=1))
    X['complete'] = ~np.isnan(F).any(axis=1)

    rows, curves, boots = [], [], []
    methods = [m for m in P.method.unique()]
    rng = np.random.default_rng(5)
    for m in methods:
        pm = P[P.method == m][['cik', 'qi', 'p']]
        D = X.merge(pm, on=['cik', 'qi'], how='inner')
        sel_rule, sel_fc, sel_mg = [], [], []
        per_q = []
        for qi, g in D.groupby('qi'):
            k = int(g.entry_ok.sum())
            s_r, s_f, s_m = g[g.entry_ok], g.nsmallest(k, 'p'), g.nlargest(k, 'm_min')
            sel_rule.append(s_r); sel_fc.append(s_f); sel_mg.append(s_m)
            cr, cf = s_r[s_r.complete], s_f[s_f.complete]
            per_q.append((len(s_r), s_r.fail4.sum(), s_f.fail4.sum(), s_m.fail4.sum(),
                          len(cr), cr.fail_any.sum(), len(cf), cf.fail_any.sum()))
        for name, S in [('MSCI 30% entry rule', sel_rule), (f'forecast:{m}', sel_fc), ('largest margin', sel_mg)]:
            S = pd.concat(S)
            rows.append(dict(method=m, policy=name, admitted=len(S), fail4=S.fail4.mean(),
                             fail_any=S.loc[S.complete, 'fail_any'].mean(), n_complete=int(S.complete.sum())))
        # review-quarter bootstrap of the exit-rate difference (forecast ranking minus entry rule)
        Q = np.array(per_q, float)
        d0 = (Q[:, 2].sum() - Q[:, 1].sum()) / Q[:, 0].sum()
        ds = []
        for _ in range(2000):
            ix = rng.integers(0, len(Q), len(Q)); q = Q[ix]
            ds.append((q[:, 2].sum() - q[:, 1].sum()) / q[:, 0].sum())
        ds = np.array(ds)
        # the same for failing at any of the next four quarterly reviews (decisions with all four reviews observed)
        any_rate = lambda q: (q[:, 7].sum() / q[:, 6].sum(), q[:, 5].sum() / q[:, 4].sum())
        fa, ra = any_rate(Q); da0 = fa - ra; das = []
        rng2 = np.random.default_rng(6)
        for _ in range(2000):
            q = Q[rng2.integers(0, len(Q), len(Q))]; f_, r_ = any_rate(q); das.append(f_ - r_)
        das = np.array(das)
        boots.append(dict(method=m, diff_vs_rule=d0, lo=np.percentile(ds, 2.5), hi=np.percentile(ds, 97.5),
                          p=min(1.0, 2 * min((ds <= 0).mean(), (ds >= 0).mean())), quarters=len(Q),
                          rel_reduction=-d0 / (Q[:, 1].sum() / Q[:, 0].sum()),
                          any_diff=da0, any_lo=np.percentile(das, 2.5), any_hi=np.percentile(das, 97.5),
                          any_p=min(1.0, 2 * min((das <= 0).mean(), (das >= 0).mean())), any_rel=-da0 / ra, rule_any=ra))
        # coverage-risk curve: admit the fraction c of candidates with the lowest forecast (per quarter)
        for c in np.linspace(0.3, 1.0, 71):
            S = pd.concat([g.nsmallest(int(round(c * len(g))), 'p') for _, g in D.groupby('qi')])
            curves.append(dict(method=m, coverage=c, fail4=S.fail4.mean(), fail_any=S.loc[S.complete, 'fail_any'].mean()))
            S = pd.concat([g.nlargest(int(round(c * len(g))), 'm_min') for _, g in D.groupby('qi')])
            curves.append(dict(method='margin-rank', coverage=c, fail4=S.fail4.mean(), fail_any=S.loc[S.complete, 'fail_any'].mean()))
        rule_cov = D.entry_ok.mean()
        curves.append(dict(method='MSCI 30% entry rule', coverage=rule_cov, fail4=D[D.entry_ok].fail4.mean(),
                           fail_any=D[D.entry_ok & D.complete].fail_any.mean()))
    R = pd.DataFrame(rows).drop_duplicates(['policy', 'admitted', 'fail4'])
    R.to_csv(f'{a.out}/decision.csv', index=False)
    B = pd.DataFrame(boots)
    # dual view: admitted share at the entry rule's one-year failure rate (interpolated along each ranking curve)
    Cv = pd.DataFrame(curves).drop_duplicates(['method', 'coverage'])
    rule = Cv[Cv.method == 'MSCI 30% entry rule'].iloc[0]
    eq = []
    for m, g in Cv[Cv.method != 'MSCI 30% entry rule'].groupby('method'):
        g = g.sort_values('coverage')
        cov = np.interp(rule.fail4, g.fail4.values, g.coverage.values) if g.fail4.is_monotonic_increasing else             g[g.fail4 <= rule.fail4].coverage.max()
        eq.append(dict(method=m, coverage_at_rule_fail=cov, rule_coverage=rule.coverage, rule_fail4=rule.fail4))
    B = B.merge(pd.DataFrame(eq), on='method', how='outer')
    B.to_csv(f'{a.out}/decision_boot.csv', index=False)
    print(B.round(4).to_string())
    pd.DataFrame(curves).drop_duplicates(['method', 'coverage']).to_csv(f'{a.out}/decision_curves.csv', index=False)
    print(R.to_string())


if __name__ == '__main__':
    main()
