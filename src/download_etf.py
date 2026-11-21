"""Download the quarterly holdings (Form N-PORT) of the Wahed FTSE USA Shariah ETF (HLAL, SEC series S000065986).

The fund tracks the FTSE USA Shariah Index, whose financial screens divide by total assets like the FTSE rulebook of
the paper. The SEC asks automated clients to declare a contact in the User-Agent header:

    SEC_USER_AGENT="Your Name your.email@example.com" python src/download_etf.py
"""
import os, re, time, urllib.request

UA = os.environ.get('SEC_USER_AGENT')
OUT = os.path.join('data_raw', 'etf', 'nport')
SERIES, CIK = 'S000065986', '1683471'


def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main():
    assert UA, 'set SEC_USER_AGENT to "Name email" as the SEC requires'
    os.makedirs(OUT, exist_ok=True)
    atom = get(f'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={SERIES}&type=NPORT-P&dateb=&owner=include&count=100&output=atom').decode('utf-8', 'ignore')
    for acc in re.findall(r'<accession-number>([^<]+)</accession-number>', atom):
        f = os.path.join(OUT, acc + '.xml')
        if not os.path.exists(f):
            open(f, 'wb').write(get(f'https://www.sec.gov/Archives/edgar/data/{CIK}/{acc.replace("-", "")}/primary_doc.xml'))
            time.sleep(0.3)
    print(len(os.listdir(OUT)), 'filings')


if __name__ == '__main__':
    main()
