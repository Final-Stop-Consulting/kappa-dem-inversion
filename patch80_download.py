import sys, os, time
from pathlib import Path
import drms
import urllib.request

REPO_DIR = Path(os.environ.get('FB_REPO', Path(__file__).resolve().parent))
OUT = os.environ.get('PATCH80_OUT', str(REPO_DIR / 'Results' / 'patch80'))
os.makedirs(OUT, exist_ok=True)
CHANNELS = [94, 131, 171, 193, 211, 335]

def download_date(date_str, tai, tag):
    """date_str like 2019-12-01, tai like 12:00:00, tag like d1"""
    c = drms.Client()
    base_t = f'{date_str}T{tai}Z'
    for wv in CHANNELS:
        fpath = os.path.join(OUT, f'{tag}_{wv}.fits')
        if os.path.exists(fpath) and os.path.getsize(fpath) > 1_000_000:
            print(f'[skip] {tag} {wv} already present ({os.path.getsize(fpath)} bytes)')
            continue
        # query a small window, pick the record nearest base time with QUALITY==0
        qs = f'aia.lev1_euv_12s[{base_t}/60s][{wv}]'
        k, seg = c.query(qs, key='T_REC,WAVELNTH,EXPTIME,QUALITY', seg='image')
        if len(k) == 0:
            print(f'[FAIL] no records for {wv} at {base_t}')
            continue
        good = k[k['QUALITY'] == 0]
        if len(good) == 0:
            good = k
        idx = good.index[0]
        url = 'http://jsoc.stanford.edu' + seg['image'].iloc[idx]
        print(f'[get] {tag} {wv}: {k.loc[idx,"T_REC"]} EXP={k.loc[idx,"EXPTIME"]:.4f} Q={k.loc[idx,"QUALITY"]} -> {url}')
        t0 = time.time()
        urllib.request.urlretrieve(url, fpath)
        print(f'   saved {os.path.getsize(fpath)} bytes in {time.time()-t0:.1f}s')

if __name__ == '__main__':
    tag = sys.argv[1]
    date_str = sys.argv[2]
    tai = sys.argv[3]
    download_date(date_str, tai, tag)
