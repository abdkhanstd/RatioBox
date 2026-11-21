"""Empirical coverage of the per-ratio forecast intervals of RatioBox on the rolling-origin test sets.

Uses the released seed-0 models (weights/rolling/h{h}_Y{Y}, src/export_models.py). For every test decision and ratio the
central 50% and 90% intervals of the forecast change are read from the sorted quantiles, and coverage is the share of
realised changes inside them. Output: results/interval_coverage.csv.
"""
import os, numpy as np, pandas as pd
import run_experiments as RE
from score import load
from screens import RATIOS


def main():
    X = pd.read_parquet(os.path.join('data', 'tasks.parquet'))
    rows = []
    for h in RE.HORIZONS:
        for Y in RE.YEARS:
            path = os.path.join('weights', 'rolling', f'h{h}_Y{Y}')
            if not os.path.exists(os.path.join(path, 'meta.json')):
                continue
            m, meta = load(path)
            tr, te = RE.split(X, h, Y)
            lv = np.array(meta['levels'])
            for j, k in enumerate(RATIOS):
                d = te[te[k].notna() & te[f'{k}_f{h}'].notna()]
                Q = m._quantiles(d[m.feats], j)
                dy = (d[f'{k}_f{h}'] - d[k]).values
                for nom, (a, b) in [(0.5, (0.25, 0.75)), (0.9, (0.05, 0.95))]:
                    lo = np.array([np.interp(a, lv, q) for q in Q]); hi = np.array([np.interp(b, lv, q) for q in Q])
                    rows.append(dict(h=h, Y=Y, ratio=k, nominal=nom, n=len(d), coverage=float(((dy >= lo) & (dy <= hi)).mean()),
                                     width=float(np.median(hi - lo))))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join('results', 'interval_coverage.csv'), index=False)
    print(R.groupby(['h', 'ratio', 'nominal']).apply(lambda g: np.average(g.coverage, weights=g.n)).round(3).to_string())


if __name__ == '__main__':
    main()
