"""Metrics, firm-clustered bootstrap intervals, paired tests and coherence checks for the forecasts."""
import glob, os, numpy as np, pandas as pd, argparse
from sklearn.metrics import roc_auc_score, average_precision_score
from concurrent.futures import ProcessPoolExecutor

PROPOSED = 'qcop'   # RatioBox


def boot_p(d):
    """two-sided bootstrap p-value of a difference with the finite-sample correction (1 + count) / (B + 1)."""
    d = np.asarray(d); B = len(d)
    return min(1.0, 2 * (1 + min((d <= 0).sum(), (d >= 0).sum())) / (B + 1))


def ece(y, p, bins=10, w=None):
    w = np.ones_like(p) if w is None else w
    order = np.argsort(p); cw = np.cumsum(w[order]) / w.sum()
    b = np.minimum((cw * bins).astype(int), bins - 1)
    e = 0.0
    for k in range(bins):
        s = order[b == k]
        if len(s):
            e += w[s].sum() / w.sum() * abs(np.average(y[s], weights=w[s]) - np.average(p[s], weights=w[s]))
    return e


def metrics(y, p, w=None):
    w = np.ones_like(p) if w is None else w
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return dict(auc=roc_auc_score(y, p, sample_weight=w), ap=average_precision_score(y, p, sample_weight=w),
                brier=np.average((p - y) ** 2, weights=w),
                logloss=-np.average(y * np.log(pc) + (1 - y) * np.log(1 - pc), weights=w), ece=ece(y, p, w=w))


class FastMetrics:
    """metrics(y, p, w) for one fixed (y, p) and many bootstrap weights: the same definitions as metrics(), with the
    sorting done once (ties of p are grouped exactly as in roc_auc_score and average_precision_score)."""

    def __init__(self, y, p, bins=10):
        self.y, self.p, self.bins = np.asarray(y, float), np.asarray(p, float), bins
        u, self.inv = np.unique(self.p, return_inverse=True)
        self.nu = len(u)
        self.order = np.argsort(self.p)
        self.yo, self.po = self.y[self.order], self.p[self.order]
        pc = np.clip(self.p, 1e-6, 1 - 1e-6)
        self.sq = (self.p - self.y) ** 2
        self.ll = -(self.y * np.log(pc) + (1 - self.y) * np.log(1 - pc))

    def __call__(self, w):
        w = np.asarray(w, float); W = w.sum()
        pos = np.bincount(self.inv, w * self.y, self.nu); neg = np.bincount(self.inv, w * (1 - self.y), self.nu)
        P, N = pos.sum(), neg.sum()
        auc = (np.dot(pos, np.cumsum(neg) - neg) + 0.5 * np.dot(pos, neg)) / (P * N)
        tp, fp = np.cumsum(pos[::-1]), np.cumsum(neg[::-1])          # thresholds from the highest score down
        prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        ap = np.dot(np.diff(np.concatenate([[0.0], tp / P])), prec)
        wo = w[self.order]; cw = np.cumsum(wo) / W
        b = np.minimum((cw * self.bins).astype(int), self.bins - 1)
        sw = np.bincount(b, wo, self.bins); sy = np.bincount(b, wo * self.yo, self.bins); sp = np.bincount(b, wo * self.po, self.bins)
        k = sw > 0
        ece = float(np.sum(sw[k] / W * np.abs(sy[k] / sw[k] - sp[k] / sw[k])))
        return dict(auc=auc, ap=ap, brier=np.dot(w, self.sq) / W, logloss=np.dot(w, self.ll) / W, ece=ece)


def load(pred_dir):
    fs = glob.glob(f'{pred_dir}/h*_Y*_*.parquet')
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


def seed_table(P):
    """metric per seed (pooled over test years), then mean and sd across seeds."""
    rows = []
    for (h, s, m, sd), g in P.groupby(['h', 'screen', 'method', 'seed']):
        if g.y.nunique() < 2:
            continue
        rows.append(dict(h=h, screen=s, method=m, seed=sd, n=len(g), base=g.y.mean(), **metrics(g.y.values, g.p.values)))
    T = pd.DataFrame(rows)
    agg = T.groupby(['h', 'screen', 'method']).agg(n=('n', 'first'), base=('base', 'first'), n_seeds=('seed', 'count'),
                                                   **{f'{k}': (k, 'mean') for k in ['auc', 'ap', 'brier', 'logloss', 'ece']},
                                                   **{f'{k}_sd': (k, 'std') for k in ['auc', 'ap', 'brier', 'ece']}).reset_index()
    return T, agg


def _boot(args):
    key, y, pa, pb, cl, reps, seed = args
    rng = np.random.default_rng(seed)
    uc, inv = np.unique(cl, return_inverse=True)
    out = []
    fa, fb = FastMetrics(y, pa), FastMetrics(y, pb)
    for r in range(reps):
        w = rng.multinomial(len(uc), np.full(len(uc), 1 / len(uc)))[inv].astype(float)
        if w[y == 1].sum() == 0:
            continue
        ma, mb = fa(w), fb(w)
        out.append({k: ma[k] - mb[k] for k in ma})
    D = pd.DataFrame(out)
    pt_a, pt_b = metrics(y, pa), metrics(y, pb)
    res = dict(zip(['h', 'screen', 'method'], key))
    for k in ['auc', 'ap', 'brier', 'logloss', 'ece']:
        d = D[k].values
        res[f'd_{k}'] = pt_a[k] - pt_b[k]
        res[f'd_{k}_lo'], res[f'd_{k}_hi'] = np.percentile(d, [2.5, 97.5])
        res[f'p_{k}'] = boot_p(d)
    return res


def paired(P, reps=5000, workers=24, baselines=None):
    """proposed minus baseline, seed-averaged predictions, firm-clustered bootstrap."""
    A = P.groupby(['h', 'screen', 'method', 'cik', 'qi'], as_index=False).agg(p=('p', 'mean'), y=('y', 'first'))
    jobs = []
    for (h, s), g in A.groupby(['h', 'screen']):
        base = g[g.method == PROPOSED].set_index(['cik', 'qi'])
        if base.empty:
            continue
        for m in g.method.unique():
            if m == PROPOSED or (baselines and m not in baselines):
                continue
            o = g[g.method == m].set_index(['cik', 'qi'])
            j = base.join(o, rsuffix='_b', how='inner')
            if j.y.nunique() < 2:
                continue
            jobs.append(((h, s, m), j.y.values, j.p.values, j.p_b.values, j.index.get_level_values(0).values, reps, 7))
    with ProcessPoolExecutor(workers) as ex:
        R = pd.DataFrame(list(ex.map(_boot, jobs)))
    # Holm across baselines within each horizon x screen, separately per metric
    for k in ['auc', 'brier']:
        R[f'p_{k}_holm'] = np.nan
        for (h, s), g in R.groupby(['h', 'screen']):
            pv = g[f'p_{k}'].values; order = np.argsort(pv); m = len(pv)
            adj = np.maximum.accumulate(np.minimum(1, (m - np.arange(m)) * pv[order]))
            R.loc[g.index[order], f'p_{k}_holm'] = adj
    return R


GROUPS = {'published': ['MSCI', 'MSCI-M', 'FTSE', 'SC'], 'unseen': ['MSCI-entry', 'MSCI-rec70', 'Debt-25', 'Cash-40']}


def _boot_group(args):
    """macro-average over the screens of a group, proposed minus baseline, firm-clustered bootstrap."""
    key, blocks, reps, seed = args
    rng = np.random.default_rng(seed)
    firms = np.unique(np.concatenate([b[4] for b in blocks]))
    pos = {f: i for i, f in enumerate(firms)}
    idx = [np.array([pos[c] for c in b[4]]) for b in blocks]
    fast = [(FastMetrics(b[0], b[1]), FastMetrics(b[0], b[2])) for b in blocks]
    def macro(wf):
        ma, mb = [], []
        for b, ix, (fa, fb) in zip(blocks, idx, fast):
            if wf is None:
                ma.append(metrics(b[0], b[1])); mb.append(metrics(b[0], b[2]))
            else:
                w = wf[ix].astype(float); ma.append(fa(w)); mb.append(fb(w))
        return {k: np.mean([m[k] for m in ma]) for k in ma[0]}, {k: np.mean([m[k] for m in mb]) for k in mb[0]}
    pa, pb = macro(None)
    D = []
    for r in range(reps):
        wf = rng.multinomial(len(firms), np.full(len(firms), 1 / len(firms)))
        a, b = macro(wf); D.append({k: a[k] - b[k] for k in a})
    D = pd.DataFrame(D)
    res = dict(zip(['h', 'group', 'method'], key))
    for k in ['auc', 'ap', 'brier', 'logloss', 'ece']:
        res[f'a_{k}'], res[f'b_{k}'] = pa[k], pb[k]
        res[f'd_{k}'] = pa[k] - pb[k]
        res[f'd_{k}_lo'], res[f'd_{k}_hi'] = np.percentile(D[k], [2.5, 97.5])
        res[f'p_{k}'] = boot_p(D[k].values)
    return res


def paired_groups(P, reps=5000, workers=24, exclude_years=()):
    A = P[~P.Y.isin(exclude_years)].groupby(['h', 'screen', 'method', 'cik', 'qi'], as_index=False).agg(p=('p', 'mean'), y=('y', 'first'))
    jobs = []
    for h in sorted(A.h.unique()):
        for gname, scr in GROUPS.items():
            methods = set(A[(A.h == h) & A.screen.isin(scr)].method) - {PROPOSED}
            for m in sorted(methods):
                blocks = []
                for s in scr:
                    g = A[(A.h == h) & (A.screen == s)]
                    a = g[g.method == PROPOSED].set_index(['cik', 'qi']); b = g[g.method == m].set_index(['cik', 'qi'])
                    j = a.join(b, rsuffix='_b', how='inner')
                    if len(j) and j.y.nunique() == 2:
                        blocks.append((j.y.values, j.p.values, j.p_b.values, s, j.index.get_level_values(0).values))
                if len(blocks) == len(scr):
                    jobs.append(((h, gname, m), blocks, reps, 11))
    with ProcessPoolExecutor(workers) as ex:
        R = pd.DataFrame(list(ex.map(_boot_group, jobs)))
    for k in ['auc', 'ap', 'brier', 'ece']:
        R[f'p_{k}_holm'] = np.nan
        for (h, g), gg in R.groupby(['h', 'group']):
            pv = gg[f'p_{k}'].values; order = np.argsort(pv); m = len(pv)
            adj = np.maximum.accumulate(np.minimum(1, (m - np.arange(m)) * pv[order]))
            R.loc[gg.index[order], f'p_{k}_holm'] = adj
    return R


def capture(P, fracs=(0.05, 0.10, 0.20)):
    """watch list: share of breaches among the riskiest fraction of passing firms, flagged within each decision quarter."""
    A = P.groupby(['h', 'screen', 'method', 'cik', 'qi'], as_index=False).agg(p=('p', 'mean'), y=('y', 'first'))
    A['rank'] = A.groupby(['h', 'screen', 'method', 'qi']).p.rank(ascending=False, method='first')
    A['n_q'] = A.groupby(['h', 'screen', 'method', 'qi']).p.transform('size')
    rows = []
    for (h, s, m), g in A.groupby(['h', 'screen', 'method']):
        r = dict(h=h, screen=s, method=m, breaches=int(g.y.sum()))
        for f in fracs:
            flag = g['rank'] <= np.ceil(f * g.n_q)
            r[f'capture{int(f * 100)}'] = g.y[flag].sum() / g.y.sum()
            r[f'precision{int(f * 100)}'] = g.y[flag].mean()
        rows.append(r)
    return pd.DataFrame(rows)


def coherence(P):
    """share of firm-quarters where a stricter screen gets a LOWER breach probability (impossible ordering)."""
    pairs = [('MSCI', 'FTSE'), ('MSCI', 'MSCI-M'), ('MSCI-M', 'FTSE'), ('MSCI-entry', 'MSCI'), ('MSCI', 'MSCI-rec70')]
    P = P[P.method != 'clim'].copy()
    P['method'] = P.method.replace({'gbm-nearest': 'gbm'})   # rulebook models score unseen screens via the nearest rulebook
    A = P.groupby(['h', 'screen', 'method', 'cik', 'qi'], as_index=False).agg(p=('p', 'mean'))
    rows = []
    for (h, m), g in A.groupby(['h', 'method']):
        piv = g.pivot_table(index=['cik', 'qi'], columns='screen', values='p')
        for strict, loose in pairs:
            if strict in piv and loose in piv:
                d = piv[[strict, loose]].dropna()
                if len(d):
                    rows.append(dict(h=h, method=m, strict=strict, loose=loose, n=len(d),
                                     viol=float((d[strict] < d[loose] - 1e-3).mean())))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', default='results/preds')
    ap.add_argument('--out', default='results')
    ap.add_argument('--reps', type=int, default=5000)
    a = ap.parse_args()
    P = load(a.preds)
    T, agg = seed_table(P)
    T.to_csv(f'{a.out}/metrics_by_seed.csv', index=False)
    agg.to_csv(f'{a.out}/metrics.csv', index=False)
    yr = []
    for (h, s, m, Y), g in P.groupby(['h', 'screen', 'method', 'Y']):
        if g.y.nunique() == 2:
            gg = g.groupby(['cik', 'qi'], as_index=False).agg(p=('p', 'mean'), y=('y', 'first'))
            yr.append(dict(h=h, screen=s, method=m, Y=Y, n=len(gg), base=gg.y.mean(), **metrics(gg.y.values, gg.p.values)))
    pd.DataFrame(yr).to_csv(f'{a.out}/metrics_by_year.csv', index=False)
    coherence(P).to_csv(f'{a.out}/coherence.csv', index=False)
    capture(P).to_csv(f'{a.out}/capture.csv', index=False)
    paired(P, reps=a.reps).to_csv(f'{a.out}/paired_by_screen.csv', index=False)
    paired_groups(P, reps=a.reps).to_csv(f'{a.out}/paired_groups.csv', index=False)
    paired_groups(P, reps=a.reps, exclude_years=(2022,)).to_csv(f'{a.out}/paired_groups_no2022.csv', index=False)
    print(agg.pivot_table(index=['h', 'screen'], columns='method', values='auc').round(3).to_string())


if __name__ == '__main__':
    main()
