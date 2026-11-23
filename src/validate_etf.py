"""External validation against real constituent changes of an Islamic index fund.

The Wahed FTSE USA Shariah ETF (HLAL) tracks the FTSE USA Shariah Index and reports its full holdings every quarter
(Form N-PORT, src/download_etf.py). At each holdings date t we take the held firms, the latest decision quarter whose
balance sheet was filed by t, and the seed-averaged one-quarter and one-year FTSE forecasts of every method from the
rolling-origin test predictions (results/preds). The outcome is the removal of the firm from the fund's holdings at the
next holdings date (h = 1) or at any of the next four (h = 4). Firms without a later balance sheet (delisted or acquired)
have no test prediction and drop out, so removals here are not caused by firms that disappear.
Outputs: results/etf_validation.csv (AUC, firm-clustered bootstrap) and results/etf_labels.csv (label agreement).
"""
import glob, os, re, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from evaluate import boot_p

METHODS = ['margin', 'proximity', 'logit', 'gbm', 'gbm-pooled', 'surface', 'qcop']
PREDECESSORS = {'bunge ltd': 1144519, 'iac inc': 1800227}   # holding name prefix -> registrant with the earlier filings


def holdings():
    rows = []
    for f in glob.glob(os.path.join('data_raw', 'etf', 'nport', '*.xml')):
        s = open(f, encoding='utf-8').read()
        d = re.search(r'<repPdDate>([^<]+)</repPdDate>', s).group(1)
        for blk in re.findall(r'<invstOrSec>(.*?)</invstOrSec>', s, re.S):
            g = lambda t: (re.search(t, blk).group(1) if re.search(t, blk) else '')
            if g(r'<assetCat>([^<]*)</assetCat>') == 'EC':
                rows.append(dict(date=pd.Timestamp(d), name=g(r'<name>([^<]*)</name>'), cusip=g(r'<cusip>([^<]*)</cusip>'),
                                 ticker=g(r'<ticker value="([^"]*)"')))
    H = pd.DataFrame(rows)
    J = json.load(open(os.path.join('data_raw', 'sec', 'company_tickers_exchange.json')))
    tk = pd.DataFrame(J['data'], columns=J['fields']).drop_duplicates('ticker').set_index('ticker').cik
    F = pd.read_parquet(os.path.join('data', 'firms.parquet'))
    # name matches prefer filers that have decisions in the panel (later entries overwrite earlier ones)
    with_dec = set(pd.read_parquet(os.path.join('data', 'tasks.parquet'), columns=['cik']).cik)
    F = F.assign(has=F.cik.isin(with_dec)).sort_values('has')
    norm = lambda x: re.sub(r'\b(inc|corp|co|plc|ltd|the|cos|holdings?)\b', '', re.sub(r'[^a-z0-9 ]', ' ', x.lower().replace('&amp;', '&'))).split()
    names = {' '.join(norm(n)): c for n, c in zip(F.name.fillna(''), F.cik)}
    H['tk_cik'] = H.ticker.map(tk)
    H['nm_cik'] = H.name.map(lambda x: names.get(' '.join(norm(x))))
    # registrants that the ticker file maps to a successor without the earlier filings (re-registered issuers)
    for name, cik in PREDECESSORS.items():
        H.loc[H.name.str.lower().str.startswith(name), 'nm_cik'] = cik
    # identity of a holding across snapshots: the CUSIP issuer prefix (first six characters) when the CUSIP is valid,
    # which survives renames; otherwise the ticker
    valid = H.cusip.str.fullmatch(r'[0-9A-Z]{9}').fillna(False)
    H['key'] = np.where(valid, 'C' + H.cusip.str[:6], 'T' + H.ticker.fillna('') + H.name)
    # candidate filers of each holding: every ticker or name match seen for its issuer in any snapshot, filers with
    # decisions first (renamed or re-registered issuers then reach the registrant that filed at the holding date)
    pool = pd.concat([H[['key', 'tk_cik']].rename(columns={'tk_cik': 'cik'}), H[['key', 'nm_cik']].rename(columns={'nm_cik': 'cik'})]).dropna()
    pool['has'] = pool.cik.isin(with_dec)
    cands = pool.sort_values('has', ascending=False).groupby('key').cik.apply(lambda s: list(dict.fromkeys(int(c) for c in s)))
    H['cands'] = H.key.map(cands)
    H['cands'] = H.cands.apply(lambda c: c if isinstance(c, list) else [])
    H['cik'] = H.cands.apply(lambda c: c[0] if c else np.nan)
    return H


def boot_diff(D, a, b, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    firms = np.sort(D.cik.unique()); pos = {f: i for i, f in enumerate(firms)}; ix = D.cik.map(pos).values
    d0 = roc_auc_score(D.y, D[a]) - roc_auc_score(D.y, D[b]); ds = []
    for _ in range(reps):
        w = rng.multinomial(len(firms), np.full(len(firms), 1 / len(firms)))[ix]
        ds.append(roc_auc_score(D.y, D[a], sample_weight=w) - roc_auc_score(D.y, D[b], sample_weight=w))
    ds = np.array(ds)
    return d0, np.percentile(ds, 2.5), np.percentile(ds, 97.5), boot_p(ds)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p)
    adj = np.empty(m); adj[o] = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[o]))
    return adj


def main():
    H = holdings()
    dates = sorted(H.date.unique())
    # holdings are identified by issuer key; each key carries its candidate filers
    held = {d: set(H[H.date == d].key) for d in dates}
    cands = H.drop_duplicates('key').set_index('key').cands.to_dict()
    # filers held at each report: a holding counts as removed only if none of its candidate filers is still held,
    # so a change of CUSIP field or name does not create a removal
    held_filers = {d: {c for k in held[d] for c in cands[k]} for d in dates}
    T = pd.read_parquet(os.path.join('data', 'tasks.parquet'), columns=['cik', 'qi', 'avail', 'r_debt', 'r_cash', 'r_rec'])
    P = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join('results', 'preds', 'h*_Y*_*.parquet'))], ignore_index=True)
    P = P[(P.screen == 'FTSE') & P.method.isin(METHODS)]
    P = P.groupby(['h', 'method', 'cik', 'qi'], as_index=False).agg(p=('p', 'mean'), lab=('y', 'first'))
    Pn = pd.read_parquet(os.path.join('data', 'panel.parquet'), columns=['cik', 'sic_prohibited'])
    in_panel = set(Pn.cik); prohibited = set(Pn[Pn.sic_prohibited.astype(bool)].cik)
    out, lab, acc = [], [], []
    # windows of holdings reports after the decision: 'next' starts at the next report, 'shifted' one report later,
    # which leaves time for the target-quarter balance sheet to be filed and screened
    for h in [1, 4]:
        for window, start in [('next', 1), ('shifted', 2)]:
            rows, cnt = [], dict(no_match=0, business_screen=0, not_in_panel=0, other_no_decision=0)
            for k, d in enumerate(dates[:-(h + start - 1)]):
                ahead = dates[k + start:k + start + h]
                for key in held[d]:
                    cs = cands[key]
                    if not cs:
                        cnt['no_match'] += 1; continue
                    # the latest decision filed by the holding date over all candidate filers of the issuer
                    q = T[T.cik.isin(cs) & (T.avail <= d)]
                    if not len(q):
                        kk = 'business_screen' if any(x in prohibited for x in cs) else 'not_in_panel' if not any(x in in_panel for x in cs) else 'other_no_decision'
                        cnt[kk] += 1; continue
                    last = q.sort_values(['qi', 'avail']).iloc[-1]
                    c = int(last.cik)
                    r3 = last[['r_debt', 'r_cash', 'r_rec']].astype(float)
                    removed = any(key not in held[a] and not (set(cs) & held_filers[a]) for a in ahead)
                    worst = max(r3.r_debt, r3.r_cash) if not r3.isna().any() else np.nan
                    rows.append(dict(date=d, cik=c, qi=int(last.qi), y=int(removed),
                                     missing=bool(r3.isna().any()), r_debt=r3.r_debt,
                                     fails_now=bool(not r3.isna().any() and not (r3.r_debt < 1 / 3 and r3.r_cash < 1 / 3 and r3.r_rec < 0.5)),
                                     fails_debt=bool(r3.r_debt >= 1 / 3),
                                     in_band=bool(1 / 3 <= worst < 0.35 and r3.r_rec < 0.5)))
            D0 = pd.DataFrame(rows).sort_values(['date', 'cik']).reset_index(drop=True)
            W = P[P.h == h].pivot_table(index=['cik', 'qi'], columns='method', values='p').reset_index()
            L = P[(P.h == h) & (P.method == 'qcop')][['cik', 'qi', 'lab']]
            D = D0.merge(W, on=['cik', 'qi']).merge(L, on=['cik', 'qi']).dropna(subset=METHODS).sort_values(['date', 'cik']).reset_index(drop=True)
            held_total = sum(len(held[d]) for d in dates[:-(h + start - 1)])
            fl = D0[D0.fails_now]
            acc.append(dict(h=h, window=window, held=held_total, **cnt, missing_ratio=int(D0.missing.sum()),
                            fails_emulated=len(fl), fails_on_debt=float(fl.fails_debt.mean()), median_debt_failing=float(fl.r_debt.median()),
                            failing_in_band=float(fl.in_band.mean()),
                            no_forecast_other=len(D0) - len(fl) - int(D0.missing.sum()) - len(D), evaluated=len(D)))
            res = []
            for m in METHODS:
                r = dict(h=h, window=window, method=m, n=len(D), firms=D.cik.nunique(), removals=int(D.y.sum()),
                         auc=roc_auc_score(D.y, D[m]), auc_label_target=roc_auc_score(D.lab, D[m]))
                if m != 'qcop':
                    r['d_auc'], r['lo'], r['hi'], r['p'] = boot_diff(D, 'qcop', m)
                res.append(r)
            R_ = pd.DataFrame(res); mask = R_.method != 'qcop'
            R_.loc[mask, 'p_holm'] = holm(R_.loc[mask, 'p'].values)
            out.append(R_)
            # agreement between the screen-implied FTSE breach at the target quarter and actual removals
            lab.append(dict(h=h, window=window, held=held_total, evaluated=len(D), removal_rate=D.y.mean(), breach_rate=D.lab.mean(),
                            removal_if_breach=D[D.lab == 1].y.mean(), removal_if_pass=D[D.lab == 0].y.mean(),
                            breach_if_removed=D[D.y == 1].lab.mean(), auc_label=roc_auc_score(D.y, D.lab)))
    pd.concat(out).to_csv(os.path.join('results', 'etf_validation.csv'), index=False)
    pd.DataFrame(lab).to_csv(os.path.join('results', 'etf_labels.csv'), index=False)
    pd.DataFrame(acc).to_csv(os.path.join('results', 'etf_coverage.csv'), index=False)
    print(pd.concat(out).round(4).to_string()); print(pd.DataFrame(lab).round(4).to_string()); print(pd.DataFrame(acc).to_string())


if __name__ == '__main__':
    main()
