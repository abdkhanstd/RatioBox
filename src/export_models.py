"""Save fitted RatioBox models (LightGBM boosters, copula correlation, metadata).

weights/rolling/h{h}_Y{Y}/  : seed-0 model of every rolling-origin window used in the paper
weights/final/h{h}/         : model trained on every label available in the archive, for scoring new filings
"""
import os, json, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models as M, run_experiments as RE
from screens import RATIOS


def save(m, path, meta):
    os.makedirs(path, exist_ok=True)
    for (k, a), b in m.models.items():
        b.save_model(os.path.join(path, f'{k}_q{a:.3f}.txt'))
    np.save(os.path.join(path, 'copula_corr.npy'), m.corr)
    meta.update(levels=[float(a) for a in M.QLEVELS], features=m.feats, ratios=RATIOS, n_mc=m.n_mc, seed=m.seed)
    json.dump(meta, open(os.path.join(path, 'meta.json'), 'w'), indent=1)


def main():
    os.environ.setdefault('GBM_THREADS', '48')
    M.GBM_PARAMS['num_threads'] = int(os.environ['GBM_THREADS'])
    X = pd.read_parquet('data/tasks.parquet')
    # the 2024 windows first: they feed the forecast-interval figure
    order = [(h, 2024) for h in RE.HORIZONS] + [(h, Y) for h in RE.HORIZONS for Y in RE.YEARS if Y != 2024]
    for h, Y in order:
        p = f'weights/rolling/h{h}_Y{Y}'
        if os.path.exists(os.path.join(p, 'meta.json')):
            continue
        tr, te = RE.split(X, h, Y)
        m = M.QuantCopula(seed=0).fit(tr, tr[[f'{k}_f{h}' for k in RATIOS]].values)
        save(m, p, dict(h=h, test_year=Y, n_train=len(tr), train_last_decision_q=int(tr.qi.max())))
        print('saved', p, flush=True)
    for h in RE.HORIZONS:
        p = f'weights/final/h{h}'
        if not os.path.exists(os.path.join(p, 'meta.json')):
            fut = X[[f'{k}_f{h}' for k in RATIOS]].notna().all(axis=1)
            tr = X[fut & (X.qi >= 2012 * 4)].reset_index(drop=True)
            m = M.QuantCopula(seed=0).fit(tr, tr[[f'{k}_f{h}' for k in RATIOS]].values)
            save(m, p, dict(h=h, n_train=len(tr), train_last_decision_q=int(tr.qi.max())))
            print('saved', p, flush=True)


if __name__ == '__main__':
    main()
