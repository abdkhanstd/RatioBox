"""Filing-lag audit of the rolling-origin windows.

Training uses decisions whose target quarter ends at least one full quarter before the test year. This script
counts the training labels whose target balance sheet (total assets) was first filed after the first test
decision of the window, i.e. late filings that were not yet public when the test window opened.
Outputs: results/filing_lag_audit.csv (one row per horizon and test year) and results/threshold_ties.csv.
"""
import numpy as np, pandas as pd
from screens import RATIOS, RULEBOOKS, UNSEEN, OFF
from run_experiments import YEARS, HORIZONS


def main():
    X = pd.read_parquet('data/tasks.parquet')
    Pn = pd.read_parquet('data/panel.parquet', columns=['cik', 'q', 'avail'])
    Pn['qi'] = Pn.q.str[:4].astype(int) * 4 + Pn.q.str[-1].astype(int) - 1
    av = Pn.drop_duplicates(['cik', 'qi']).set_index(['cik', 'qi']).avail
    rows = []
    for h in HORIZONS:
        fut = X[[f'{k}_f{h}' for k in RATIOS]].notna().all(axis=1)
        tgt = pd.Series(av.reindex(pd.MultiIndex.from_arrays([X.cik, X.qi + h])).values, index=X.index)
        dec = pd.Series(av.reindex(pd.MultiIndex.from_arrays([X.cik, X.qi])).values, index=X.index)
        for Y in YEARS:
            tr = (X.qi + h <= Y * 4 - 2) & fut & (X.qi >= 2012 * 4)      # same windows as run_experiments.split
            te = (X.qi >= Y * 4) & (X.qi <= Y * 4 + 3) & fut
            first = dec[te].min(); t = tgt[tr]
            rows.append(dict(h=h, Y=Y, n_train=int(tr.sum()), first_test_decision=first.date(),
                             n_after_first_test=int((t > first).sum()), share_after_first_test=float((t > first).mean())))
    R = pd.DataFrame(rows)
    R.to_csv('results/filing_lag_audit.csv', index=False)
    print(R.to_string(), '\npooled share', R.n_after_first_test.sum() / R.n_train.sum())
    # ties: a screened ratio exactly at one of its thresholds, now or at a target quarter (labels count a tie as a breach)
    scr = list(RULEBOOKS.values()) + list(UNSEEN.values()); tie = np.zeros(len(X), bool)
    for k in RATIOS:
        for c in [k] + [f'{k}_f{h}' for h in HORIZONS]:
            for t in {rb[k] for rb in scr if rb[k] < OFF}:
                tie |= np.isclose(X[c].values, t, rtol=0, atol=1e-12)
    pd.DataFrame([dict(decisions=len(X), with_tie=int(tie.sum()))]).to_csv('results/threshold_ties.csv', index=False)
    print('decisions with a ratio exactly at a threshold:', int(tie.sum()), 'of', len(X))


if __name__ == '__main__':
    main()
