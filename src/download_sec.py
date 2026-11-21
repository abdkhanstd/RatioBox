"""Download the SEC EDGAR bulk archives used in the study.

The SEC fair-access policy asks automated clients to declare a contact in the User-Agent header.
Set it before running, for example:  SEC_USER_AGENT="YourName your.email@domain"  python src/download_sec.py
"""
import os, sys, urllib.request, shutil

FILES = {
    'companyfacts.zip': 'https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip',
    'submissions.zip': 'https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip',
}


def main(out='data_raw/sec'):
    ua = os.environ.get('SEC_USER_AGENT')
    if not ua:
        sys.exit('Set SEC_USER_AGENT to "Name email" as required by the SEC fair-access policy.')
    os.makedirs(out, exist_ok=True)
    for fn, url in FILES.items():
        dst = os.path.join(out, fn)
        if os.path.exists(dst):
            print('exists', dst); continue
        req = urllib.request.Request(url, headers={'User-Agent': ua})
        with urllib.request.urlopen(req) as r, open(dst, 'wb') as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        print('saved', dst)


if __name__ == '__main__':
    main()
