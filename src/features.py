"""Decision-time features and future ratios for every eligible firm-quarter.

Decision time = release of the quarter-t balance sheet. Features use only quarters <= t (first-reported
values). Targets are the ratios of quarter t+h as first reported (h = 1 and 4 quarters ahead).
Output: data/tasks.parquet
"""
import numpy as np, pandas as pd, argparse
from screens import RATIOS, RULEBOOKS, tau_vec

STRUCT = ['r_liab', 'r_curliab', 'r_curassets', 'r_equity', 'r_retained', 'r_ppe', 'r_goodwill', 'r_intang',
          'r_inventory', 'r_oplease', 'r_ltd_current']
HORIZONS = [1, 4]
MIN_TA = 10e6


def build(panel):
    P = panel.copy()
    P['qi'] = P.q.apply(lambda s: int(s[:4]) * 4 + int(s[-1]) - 1)
    P = P.sort_values(['cik', 'qi'])
    base = P[['cik', 'qi', 'avail'] + RATIOS + ['log_ta']].copy()
    X = P.copy()
    for L in [1, 2, 3, 4, 8]:
        lag = base.copy(); lag['qi'] = lag.qi + L
        lag = lag.rename(columns={c: f'{c}_l{L}' for c in RATIOS + ['log_ta']}).rename(columns={'avail': 'avail_lag'})
        X = X.merge(lag, on=['cik', 'qi'], how='left')
        late = X.avail_lag > X.avail   # a lagged value first filed after the decision date is not usable
        X.loc[late, [f'{c}_l{L}' for c in RATIOS + ['log_ta']]] = np.nan
        X = X.drop(columns='avail_lag')
    for c in RATIOS + ['log_ta']:
        X[f'{c}_d1'] = X[c] - X[f'{c}_l1']
        X[f'{c}_d4'] = X[c] - X[f'{c}_l4']
        X[f'{c}_d8'] = X[c] - X[f'{c}_l8']
    for c in RATIOS:
        lags = X[[c, f'{c}_l1', f'{c}_l2', f'{c}_l3', f'{c}_l4']]
        X[f'{c}_max4'] = lags.max(axis=1)
        X[f'{c}_sd4'] = lags.diff(axis=1).std(axis=1)
    # number of quarters of history observed up to t
    X['hist'] = X.groupby('cik').cumcount()
    X['fq'] = X.qi % 4
    X['sic2'] = (X.sic // 100).fillna(-1).astype(int)
    # status under the published rulebooks at t
    Rt = X[RATIOS].values
    for name, rb in RULEBOOKS.items():
        X[f'pass_{name}'] = np.all(Rt < tau_vec(rb), axis=1)
    X['n_fail_rb'] = sum((~X[f'pass_{n}']).astype(int) for n in RULEBOOKS)
    # targets
    for h in [1, 2, 3, 4]:
        fut = base[['cik', 'qi'] + RATIOS].copy(); fut['qi'] = fut.qi - h
        fut = fut.rename(columns={c: f'{c}_f{h}' for c in RATIOS})
        X = X.merge(fut, on=['cik', 'qi'], how='left')
    elig = (~X.sic_prohibited) & (X.TA >= MIN_TA) & (X.entity_type.fillna('operating') == 'operating')
    X = X[elig & (X.qi >= 2011 * 4 + 3)].copy()
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', default='data/panel.parquet')
    ap.add_argument('--out', default='data/tasks.parquet')
    a = ap.parse_args()
    X = build(pd.read_parquet(a.panel))
    X.to_parquet(a.out, index=False)
    print('tasks', X.shape, 'firms', X.cik.nunique())
    for h in HORIZONS:
        print('h', h, 'with target', X[f'r_debt_f{h}'].notna().sum())


if __name__ == '__main__':
    main()
