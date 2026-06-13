import sys, os, time
from pathlib import Path
os.environ.setdefault('HOME', os.path.expanduser('~'))
import numpy as np
import sunpy.map
import astropy.units as u
from aiapy.calibrate import register

REPO_DIR = Path(os.environ.get('FB_REPO', Path(__file__).resolve().parent))
OUT = os.environ.get('PATCH80_OUT', str(REPO_DIR / 'Results' / 'patch80'))
os.makedirs(OUT, exist_ok=True)

def reg_one(tag, wv):
    fpath = os.path.join(OUT, f'{tag}_{wv}.fits')
    outnpz = os.path.join(OUT, f'{tag}_{wv}_reg.npz')
    if os.path.exists(outnpz):
        print(f'[skip] {tag} {wv} registered npz exists')
        return
    t0 = time.time()
    m = sunpy.map.Map(fpath)
    exptime = float(m.meta['exptime'])
    # Register to level 1.5: scale to 0.6"/px, derotate, recenter.
    # Explicitly SKIP update_pointing and degradation correction (no network calls).
    m15 = register(m)  # register does NOT call update_pointing or correct_degradation itself
    data = np.asarray(m15.data, dtype=np.float64)
    # Normalize to DN/s per pixel
    dn_s = data / exptime
    meta = {
        'exptime': exptime,
        'cdelt1': float(m15.meta['cdelt1']),
        'cdelt2': float(m15.meta['cdelt2']),
        'crpix1': float(m15.meta['crpix1']),
        'crpix2': float(m15.meta['crpix2']),
        'rsun_obs': float(m15.meta.get('rsun_obs', m.meta.get('rsun_obs'))),
        'r_sun': float(m15.meta.get('r_sun', m.meta.get('r_sun', 0.0))),
        'shape0': data.shape[0], 'shape1': data.shape[1],
    }
    np.savez_compressed(outnpz, dn_s=dn_s.astype(np.float32),
                        **{k: meta[k] for k in meta})
    print(f'[ok] {tag} {wv}: exp={exptime:.4f} shape={data.shape} '
          f'cdelt={meta["cdelt1"]:.4f} crpix=({meta["crpix1"]:.2f},{meta["crpix2"]:.2f}) '
          f'rsun_obs={meta["rsun_obs"]:.2f} mean_dns={np.nanmean(dn_s):.3f} '
          f'in {time.time()-t0:.1f}s')

if __name__ == '__main__':
    tag = sys.argv[1]; wv = int(sys.argv[2])
    reg_one(tag, wv)
