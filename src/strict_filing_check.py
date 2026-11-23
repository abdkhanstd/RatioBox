"""Effect of the late-filed training labels.

Compares seed-0 forecasts of RatioBox and the label-trained GBMs from the main run (results/preds) with a refit that
drops every training row whose target balance sheet was first filed after the first test decision of its window
(RB_STRICT_FILING=1 RB_SEEDS=0 python src/run_experiments.py --groups qcop,gbm --out results/preds_strict).
Output: results/robust_strict_filing.csv with macro AUC and Brier per horizon and screen group for both runs.
"""
import glob, numpy as np, pandas as pd
from evaluate import metrics, GROUPS

METHODS = {'published': ['qcop', 'gbm'], 'unseen': ['qcop', 'gbm-nearest', 'gbm-oracle']}


def load(d, seed=0):
    P = pd.concat([pd.read_parquet(f) for f in glob.glob(f'{d}/h*_Y*_qcop.parquet') + glob.glob(f'{d}/h*_Y*_gbm.parquet')],
                  ignore_index=True)
    return P[P.seed == seed]


def macro(P):
    rows = []
    for h in sorted(P.h.unique()):
        for g, scr in GROUPS.items():
            for m in METHODS[g]:
                vals = [metrics(x.y.values, x.p.values) for _, x in P[(P.h == h) & (P.method == m) & P.screen.isin(scr)].groupby('screen')]
                if len(vals) == len(scr):
                    rows.append(dict(h=h, group=g, method=m, auc=np.mean([v['auc'] for v in vals]), brier=np.mean([v['brier'] for v in vals])))
    return pd.DataFrame(rows)


def main():
    A, B = load('results/preds'), load('results/preds_strict')
    keys = ['h', 'Y', 'screen', 'method']
    common = A[keys].drop_duplicates().merge(B[keys].drop_duplicates())
    A, B = A.merge(common), B.merge(common)
    R = macro(A).merge(macro(B), on=['h', 'group', 'method'], suffixes=('_main', '_strict'))
    R['d_auc'] = R.auc_strict - R.auc_main
    R['d_brier'] = R.brier_strict - R.brier_main
    R.to_csv('results/robust_strict_filing.csv', index=False)
    print(R.round(4).to_string())


if __name__ == '__main__':
    main()
