"""Extract firm metadata (SIC industry, exchange, entity type) from the SEC submissions bulk archive.
Output: data/firms.parquet
"""
import zipfile, orjson, argparse, os
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', default='data_raw/sec/submissions.zip')
    ap.add_argument('--out', default='data/firms.parquet')
    a = ap.parse_args()
    rows = []
    with zipfile.ZipFile(a.zip) as z:
        for nm in z.namelist():
            if not nm.endswith('.json') or '-submissions-' in nm:
                continue
            try:
                d = orjson.loads(z.read(nm))
            except Exception:
                continue
            rows.append(dict(cik=int(d.get('cik', 0) or 0), name=d.get('name'), sic=d.get('sic'),
                             sic_desc=d.get('sicDescription'), entity_type=d.get('entityType'),
                             exchanges='|'.join([e for e in (d.get('exchanges') or []) if e]),
                             tickers='|'.join([t for t in (d.get('tickers') or []) if t]),
                             state_inc=d.get('stateOfIncorporation'), category=d.get('category')))
    df = pd.DataFrame(rows)
    df['sic'] = pd.to_numeric(df.sic, errors='coerce')
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_parquet(a.out, index=False)
    print('saved', a.out, df.shape)
    print(df.entity_type.value_counts().head())


if __name__ == '__main__':
    main()
