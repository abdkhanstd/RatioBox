"""Re-create the earlier RatioBox configuration (group qcop_earlier) on one window and compare it with the saved
forecasts in results/preds (default: one quarter ahead, test year 2022). Output: results/check_qcop_earlier.csv."""
import os, sys, argparse, tempfile, subprocess, pandas as pd
from evaluate import metrics


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--h', default='1'); ap.add_argument('--year', default='2022')
    a = ap.parse_args()
    tmp = tempfile.mkdtemp()
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'run_experiments.py'), '--groups', 'qcop_earlier',
                    '--horizons', a.h, '--years', a.year, '--workers', '1', '--out', tmp], check=True)
    f = f'h{a.h}_Y{a.year}_qcop_earlier.parquet'
    new, old = pd.read_parquet(os.path.join(tmp, f)), pd.read_parquet(os.path.join('results', 'preds', f))
    rows = []
    for s in sorted(new.screen.unique()):
        gn = new[new.screen == s].groupby(['cik', 'qi']).agg(p=('p', 'mean'), y=('y', 'first'))
        go = old[old.screen == s].groupby(['cik', 'qi']).agg(p=('p', 'mean'), y=('y', 'first'))
        mn, mo = metrics(gn.y.values, gn.p.values), metrics(go.y.values, go.p.values)
        rows.append(dict(h=a.h, Y=a.year, screen=s, auc_rerun=mn['auc'], auc_saved=mo['auc'], brier_rerun=mn['brier'], brier_saved=mo['brier']))
    R = pd.DataFrame(rows); R.to_csv(os.path.join('results', 'check_qcop_earlier.csv'), index=False)
    print(R.round(4).to_string())


if __name__ == '__main__':
    main()
