"""Extract balance-sheet and income facts from the SEC XBRL company-facts bulk archive.

For every company and concept we keep, for each period, the value as FIRST reported
(earliest filing date), so later restatements never leak into earlier decisions.
Output: data/xbrl_facts.parquet (long format).
"""
import os, sys, zipfile, orjson, argparse
import multiprocessing as mp
import pandas as pd

INSTANT = [
    # size and structure
    'Assets', 'AssetsCurrent', 'Liabilities', 'LiabilitiesCurrent', 'StockholdersEquity',
    'LiabilitiesAndStockholdersEquity', 'RetainedEarningsAccumulatedDeficit',
    'PropertyPlantAndEquipmentNet', 'Goodwill', 'IntangibleAssetsNetExcludingGoodwill', 'InventoryNet',
    'OperatingLeaseLiability',
    # cash and interest-bearing securities
    'CashAndCashEquivalentsAtCarryingValue', 'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents',
    'Cash', 'CashEquivalentsAtCarryingValue', 'CashCashEquivalentsAndShortTermInvestments',
    'ShortTermInvestments', 'MarketableSecuritiesCurrent', 'AvailableForSaleSecuritiesDebtSecuritiesCurrent',
    'AvailableForSaleSecuritiesCurrent', 'HeldToMaturitySecuritiesCurrent', 'TradingSecuritiesCurrent',
    # receivables
    'AccountsReceivableNetCurrent', 'ReceivablesNetCurrent', 'AccountsNotesAndLoansReceivableNetCurrent',
    'AccountsReceivableNet', 'AccountsAndOtherReceivablesNetCurrent',
    # interest-bearing debt
    'DebtCurrent', 'LongTermDebtCurrent', 'LongTermDebtNoncurrent', 'LongTermDebt', 'ShortTermBorrowings',
    'CommercialPaper', 'LongTermDebtAndCapitalLeaseObligations', 'LongTermDebtAndCapitalLeaseObligationsCurrent',
    'LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities', 'DebtLongtermAndShorttermCombinedAmount',
    'LineOfCredit', 'LongTermLineOfCredit', 'ConvertibleNotesPayableCurrent', 'ConvertibleDebtNoncurrent',
    'ConvertibleNotesPayable', 'NotesPayableCurrent', 'NotesPayable', 'LongTermNotesPayable', 'SeniorNotes',
    'OtherLongTermDebtNoncurrent', 'OtherShortTermBorrowings', 'LoansPayableCurrent', 'LoansPayable',
    'LongTermLoansPayable', 'BankOverdrafts', 'FinanceLeaseLiability', 'FinanceLeaseLiabilityCurrent',
    'FinanceLeaseLiabilityNoncurrent', 'CapitalLeaseObligations', 'CapitalLeaseObligationsCurrent',
    'CapitalLeaseObligationsNoncurrent', 'DebtInstrumentCarryingAmount', 'SecuredDebt', 'UnsecuredDebt', 'OtherNotesPayable',
    'SubordinatedDebt', 'ConvertibleSubordinatedDebt', 'NotesPayableRelatedPartiesClassifiedCurrent',
]
DURATION = [
    'Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'SalesRevenueNet',
    'NetIncomeLoss', 'NetCashProvidedByUsedInOperatingActivities', 'InterestExpense',
    'InvestmentIncomeInterest', 'OperatingIncomeLoss',
]
FORMS = {'10-K', '10-Q', '10-KT', '10-QT', '10-K405'}
WANT = set(INSTANT) | set(DURATION)


def parse_one(args):
    path, names = args
    out = []
    with zipfile.ZipFile(path) as z:
        for nm in names:
            try:
                d = orjson.loads(z.read(nm))
            except Exception:
                continue
            cik = int(os.path.basename(nm)[3:13])
            g = d.get('facts', {}).get('us-gaap', {})
            for concept in WANT.intersection(g.keys()):
                units = g[concept].get('units', {})
                recs = units.get('USD')
                if not recs:
                    continue
                inst = concept in INSTANT
                best = {}
                for r in recs:
                    if r.get('form') not in FORMS:
                        continue
                    if inst and 'start' in r:
                        continue
                    if (not inst) and 'start' not in r:
                        continue
                    key = (r.get('start'), r['end'])
                    if key not in best or r['filed'] < best[key]['filed']:
                        best[key] = r
                for (st, en), r in best.items():
                    out.append((cik, concept, st, en, r['filed'], r['val'], r.get('form'), r.get('fp'), r.get('fy')))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', default='data_raw/sec/companyfacts.zip')
    ap.add_argument('--out', default='data/xbrl_facts.parquet')
    ap.add_argument('--workers', type=int, default=48)
    a = ap.parse_args()
    with zipfile.ZipFile(a.zip) as z:
        names = [n for n in z.namelist() if n.endswith('.json')]
    print('companies in archive:', len(names), flush=True)
    chunks = [names[i::a.workers * 8] for i in range(a.workers * 8)]
    rows = []
    with mp.Pool(a.workers) as pool:
        for i, part in enumerate(pool.imap_unordered(parse_one, [(a.zip, c) for c in chunks])):
            rows.extend(part)
            if i % 50 == 0:
                print('chunk', i, 'rows', len(rows), flush=True)
    df = pd.DataFrame(rows, columns=['cik', 'concept', 'start', 'end', 'filed', 'val', 'form', 'fp', 'fy'])
    for c in ['start', 'end', 'filed']:
        df[c] = pd.to_datetime(df[c], errors='coerce')
    df['concept'] = df.concept.astype('category')
    df['form'] = df.form.astype('category')
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_parquet(a.out, index=False)
    print('saved', a.out, df.shape)


if __name__ == '__main__':
    main()
