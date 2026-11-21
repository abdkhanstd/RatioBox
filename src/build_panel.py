"""Build the firm-quarter panel of screening components and ratios from first-reported XBRL values.

Steps
1. Instant (balance-sheet) facts are pivoted to one row per firm and fiscal period end. A value is kept
   only if it was filed no later than the filing that first reported total assets for that period, so
   each row contains what an investor could read when that balance sheet was released.
2. Period ends are mapped to calendar quarters (period end minus 15 days), one row per firm-quarter.
3. Screening components (total debt, cash, cash plus interest-bearing securities, receivables) are
   built with documented tag hierarchies. Ratios use total assets as denominator.
4. Business-activity screen: firms in prohibited industries (by SIC code) are removed.
Output: data/panel.parquet
"""
import numpy as np, pandas as pd, argparse, os

FIRST = lambda df, cols: df[[c for c in cols if c in df.columns]].bfill(axis=1).iloc[:, 0] if any(c in df.columns for c in cols) else pd.Series(np.nan, index=df.index)


def ssum(df, cols):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series(0.0, index=df.index)
    return df[cols].fillna(0).sum(axis=1)


def anyobs(df, cols):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series(False, index=df.index)
    return df[cols].notna().any(axis=1)


PROHIBITED_SIC = [(6000, 6799), (2080, 2085), (2100, 2199), (2011, 2013), (3480, 3489), (5180, 5182),
                  (5921, 5921), (7800, 7849), (7990, 7999)]


def prohibited(sic):
    if pd.isna(sic):
        return True
    return any(lo <= sic <= hi for lo, hi in PROHIBITED_SIC)


def build(facts_path, firms_path):
    f = pd.read_parquet(facts_path)
    inst = f[f.start.isna()].copy()
    inst['concept'] = inst.concept.astype(str)
    # filing that first reported total assets for each firm-period
    a = inst[inst.concept == 'Assets'][['cik', 'end', 'filed']].rename(columns={'filed': 'avail'})
    inst = inst.merge(a, on=['cik', 'end'], how='inner')
    inst = inst[inst.filed <= inst.avail]
    W = inst.pivot_table(index=['cik', 'end', 'avail'], columns='concept', values='val', aggfunc='first').reset_index()
    W['q'] = (W.end - pd.Timedelta(days=15)).dt.to_period('Q')
    W = W.sort_values(['cik', 'q', 'end']).drop_duplicates(['cik', 'q'], keep='last')

    # ---- components
    W['TA'] = W['Assets']
    cash = FIRST(W, ['CashAndCashEquivalentsAtCarryingValue', 'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents',
                     'Cash', 'CashEquivalentsAtCarryingValue'])
    sti = FIRST(W, ['ShortTermInvestments', 'MarketableSecuritiesCurrent', 'AvailableForSaleSecuritiesDebtSecuritiesCurrent',
                    'AvailableForSaleSecuritiesCurrent'])
    sti = sti.fillna(0) + (W['HeldToMaturitySecuritiesCurrent'].fillna(0) if 'HeldToMaturitySecuritiesCurrent' in W else 0)
    combo = W['CashCashEquivalentsAndShortTermInvestments'] if 'CashCashEquivalentsAndShortTermInvestments' in W else pd.Series(np.nan, index=W.index)
    W['CASH'] = cash.where(cash.notna(), combo)
    W['CASHSEC'] = np.where(cash.notna(), cash.fillna(0) + sti, combo)
    W['cash_missing'] = W['CASH'].isna().astype(int)
    W['AR'] = FIRST(W, ['AccountsReceivableNetCurrent', 'ReceivablesNetCurrent', 'AccountsNotesAndLoansReceivableNetCurrent',
                        'AccountsAndOtherReceivablesNetCurrent', 'AccountsReceivableNet'])
    W['ar_missing'] = W['AR'].isna().astype(int)

    # total interest-bearing debt (short + long term, finance leases; operating leases excluded)
    st_extra = ['ShortTermBorrowings', 'CommercialPaper', 'NotesPayableCurrent', 'LoansPayableCurrent',
                'ConvertibleNotesPayableCurrent', 'OtherShortTermBorrowings', 'BankOverdrafts', 'LineOfCredit']
    cur_a = FIRST(W, ['DebtCurrent'])
    cur_b = ssum(W, ['LongTermDebtCurrent', 'CapitalLeaseObligationsCurrent', 'FinanceLeaseLiabilityCurrent']) \
        .where(~anyobs(W, ['LongTermDebtAndCapitalLeaseObligationsCurrent']), FIRST(W, ['LongTermDebtAndCapitalLeaseObligationsCurrent'])) + ssum(W, st_extra)
    current = cur_a.where(cur_a.notna(), cur_b)
    nc_tag = FIRST(W, ['LongTermDebtNoncurrent', 'LongTermDebtAndCapitalLeaseObligations'])
    nc_sum = ssum(W, ['ConvertibleDebtNoncurrent', 'LongTermNotesPayable', 'LongTermLoansPayable', 'LongTermLineOfCredit',
                      'OtherLongTermDebtNoncurrent', 'SeniorNotes'])
    noncurrent = nc_tag.where(nc_tag.notna(), nc_sum)
    lease_nc = ssum(W, ['FinanceLeaseLiabilityNoncurrent', 'CapitalLeaseObligationsNoncurrent'])
    noncurrent = noncurrent + np.where(anyobs(W, ['LongTermDebtAndCapitalLeaseObligations']), 0, lease_nc)
    total_a = FIRST(W, ['DebtLongtermAndShorttermCombinedAmount'])
    total_b = FIRST(W, ['LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities']) + ssum(W, ['ShortTermBorrowings', 'CommercialPaper'])
    total_c = FIRST(W, ['LongTermDebt']) + ssum(W, ['ShortTermBorrowings', 'CommercialPaper'])
    split = current.fillna(0) + noncurrent.fillna(0)
    has_split = cur_a.notna() | anyobs(W, ['LongTermDebtCurrent', 'LongTermDebtAndCapitalLeaseObligationsCurrent', 'CapitalLeaseObligationsCurrent', 'FinanceLeaseLiabilityCurrent']) | anyobs(W, ['LongTermDebtNoncurrent', 'LongTermDebtAndCapitalLeaseObligations', 'ConvertibleDebtNoncurrent',
                                             'LongTermNotesPayable', 'LongTermLoansPayable', 'LongTermLineOfCredit',
                                             'OtherLongTermDebtNoncurrent', 'SeniorNotes'] + st_extra)
    debt = total_a.copy()
    debt = debt.where(debt.notna(), total_b)
    debt = debt.where(debt.notna(), np.where(has_split, split, np.nan))
    debt = debt.where(debt.notna(), total_c)
    other = ssum(W, ['SecuredDebt', 'UnsecuredDebt', 'OtherNotesPayable', 'SubordinatedDebt', 'ConvertibleSubordinatedDebt',
                     'NotesPayableRelatedPartiesClassifiedCurrent'])
    has_other = anyobs(W, ['SecuredDebt', 'UnsecuredDebt', 'OtherNotesPayable', 'SubordinatedDebt', 'ConvertibleSubordinatedDebt',
                           'NotesPayableRelatedPartiesClassifiedCurrent'])
    debt = debt.where(debt.notna(), np.where(has_other, other, np.nan))
    debt = debt.where(debt.notna(), FIRST(W, ['DebtInstrumentCarryingAmount']))
    W['debt_tagged'] = debt.notna().astype(int)
    W['DEBT'] = debt.fillna(0.0)

    # ---- ratios on total assets
    W = W[W.TA > 0].copy()
    W['r_debt'] = (W.DEBT / W.TA).clip(0, 5)
    W['r_cash'] = (W.CASHSEC / W.TA).clip(0, 1.5)
    W['r_rec'] = ((W.AR.fillna(0) + W.CASH.fillna(0)) / W.TA).clip(0, 1.5)
    # other structure features
    for c, nm in [('Liabilities', 'r_liab'), ('LiabilitiesCurrent', 'r_curliab'), ('AssetsCurrent', 'r_curassets'),
                  ('StockholdersEquity', 'r_equity'), ('RetainedEarningsAccumulatedDeficit', 'r_retained'),
                  ('PropertyPlantAndEquipmentNet', 'r_ppe'), ('Goodwill', 'r_goodwill'),
                  ('IntangibleAssetsNetExcludingGoodwill', 'r_intang'), ('InventoryNet', 'r_inventory'),
                  ('OperatingLeaseLiability', 'r_oplease')]:
        W[nm] = (W[c] / W.TA).clip(-5, 5) if c in W else np.nan
    W['r_ltd_current'] = (FIRST(W, ['LongTermDebtCurrent', 'DebtCurrent']) / W.TA).clip(0, 5)
    W['log_ta'] = np.log(W.TA)

    keep = ['cik', 'q', 'end', 'avail', 'TA', 'DEBT', 'CASH', 'CASHSEC', 'AR', 'cash_missing', 'ar_missing', 'debt_tagged',
            'r_debt', 'r_cash', 'r_rec', 'r_liab', 'r_curliab', 'r_curassets', 'r_equity', 'r_retained', 'r_ppe',
            'r_goodwill', 'r_intang', 'r_inventory', 'r_oplease', 'r_ltd_current', 'log_ta']
    P = W[keep].copy()

    firms = pd.read_parquet(firms_path)
    P = P.merge(firms[['cik', 'sic', 'entity_type', 'exchanges', 'name']], on='cik', how='left')
    P['sic_prohibited'] = P.sic.apply(prohibited)
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--facts', default='data/xbrl_facts.parquet')
    ap.add_argument('--firms', default='data/firms.parquet')
    ap.add_argument('--out', default='data/panel.parquet')
    a = ap.parse_args()
    P = build(a.facts, a.firms)
    P['q'] = P.q.astype(str)
    P.to_parquet(a.out, index=False)
    print('panel', P.shape, 'firms', P.cik.nunique())
    print(P.groupby(P.q.str[:4]).size())


if __name__ == '__main__':
    main()
