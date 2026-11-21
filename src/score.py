"""Score firm-quarters with saved RatioBox weights under any rulebook.

Example:
  python src/score.py --weights weights/final/h4 --tasks data/tasks.parquet --quarter 2026Q2 \
         --debt 0.3333 --cash 0.3333 --rec 0.49 --out scores.csv
A threshold of 0 (or omitted) switches that ratio screen off.
"""
import os, sys, json, argparse, glob, numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from screens import RATIOS, OFF
import models as M


def load(path):
    meta = json.load(open(os.path.join(path, 'meta.json')))
    m = M.QuantCopula(feats=meta['features'], seed=meta['seed'], n_mc=meta['n_mc'])
    m.models = {}
    for k in RATIOS:
        for a in meta['levels']:
            m.models[(k, a)] = lgb.Booster(model_file=os.path.join(path, f'{k}_q{a:.3f}.txt'))
    m.corr = np.load(os.path.join(path, 'copula_corr.npy'))
    m.Zmc = np.random.default_rng(m.seed).multivariate_normal(np.zeros(3), m.corr, size=m.n_mc)
    return m, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True); ap.add_argument('--tasks', default='data/tasks.parquet')
    ap.add_argument('--quarter', required=True)
    for k in ['debt', 'cash', 'rec']:
        ap.add_argument(f'--{k}', type=float, default=0.0)
    ap.add_argument('--out', default='scores.csv')
    a = ap.parse_args()
    m, meta = load(a.weights)
    X = pd.read_parquet(a.tasks)
    D = X[X.q == a.quarter].reset_index(drop=True)
    tau = np.array([v if v > 0 else OFF for v in [a.debt, a.cash, a.rec]])
    D = D[np.all(D[RATIOS].values < tau, axis=1)].reset_index(drop=True)
    p = m.predict(D, tau)
    out = D[['cik', 'name', 'q'] + RATIOS].copy(); out['breach_prob'] = p
    out.sort_values('breach_prob', ascending=False).to_csv(a.out, index=False)
    print(f"scored {len(out)} passing firm-quarters, horizon {meta['h']} quarters -> {a.out}")


if __name__ == '__main__':
    main()
