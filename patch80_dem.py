import sys, os, json, time
from pathlib import Path
os.environ.setdefault('HOME', os.path.expanduser('~'))
REPO_DIR = Path(os.environ.get('FB_REPO', Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_DIR))
import numpy as np
import kappa_dem_pipeline as kp
from demregpy import dn2dem

OUT = os.environ.get('PATCH80_OUT', str(REPO_DIR / 'Results' / 'patch80'))
os.makedirs(OUT, exist_ok=True)
CHANNELS = [94, 131, 171, 193, 211, 335]
PIX = 0.6                 # arcsec/px (cdelt)
HALF_ARCSEC = 50.0        # half-size -> 100"x100" patch
HALF_PX = int(round(HALF_ARCSEC / PIX))   # ~83 px half -> ~166 px box
DISK_FRAC = 0.95          # patch centers within 0.95 R_sun
GRID_STEP_ARCSEC = float(os.environ.get('GRID_STEP', '100'))  # grid spacing
LO_171, HI_171 = 30.0, 500.0   # quiet-Sun gating on patch-mean 171 DN/s
N_TARGET = 80

def fwhm(x, y):
    h = y.max()/2; ab = y >= h
    if not ab.any(): return float('nan')
    f = int(np.argmax(ab)); l = len(ab)-1-int(np.argmax(ab[::-1]))
    le = x[f] if f == 0 else x[f-1]+(h-y[f-1])/(y[f]-y[f-1])*(x[f]-x[f-1])
    ri = x[l] if l+1 >= len(x) else x[l]+(h-y[l])/(y[l+1]-y[l])*(x[l+1]-x[l])
    return ri-le

def load_channels(tag):
    data = {}
    meta = None
    for wv in CHANNELS:
        d = np.load(os.path.join(OUT, f'{tag}_{wv}_reg.npz'))
        data[wv] = d['dn_s'].astype(np.float32)
        if meta is None:
            meta = {k: float(d[k]) for k in ['cdelt1','crpix1','crpix2','rsun_obs']}
    return data, meta

def build_patch_centers(meta, shape):
    """Regular grid scan, row-major, centers within DISK_FRAC*R_sun (using 171 WCS)."""
    cdelt = meta['cdelt1']
    crpix1 = meta['crpix1']; crpix2 = meta['crpix2']  # 1-based FITS pixel of disk center
    rsun_px = meta['rsun_obs'] / cdelt
    rmax_px = DISK_FRAC * rsun_px
    cx = crpix1 - 1.0   # 0-based center col (x)
    cy = crpix2 - 1.0   # 0-based center row (y)
    step_px = GRID_STEP_ARCSEC / cdelt
    ny, nx = shape
    # grid offsets so it is symmetric about center
    n_half = int(np.floor((rmax_px) / step_px))
    offs = np.arange(-n_half, n_half+1) * step_px
    centers = []
    for dy in offs:               # row-major: outer loop rows (y)
        for dx in offs:
            yy = cy + dy; xx = cx + dx
            r = np.hypot(dx, dy)
            if r > rmax_px:
                continue
            # ensure full patch fits in array
            yi, xi = int(round(yy)), int(round(xx))
            if yi-HALF_PX < 0 or yi+HALF_PX >= ny or xi-HALF_PX < 0 or xi+HALF_PX >= nx:
                continue
            centers.append((yi, xi))
    return centers, rsun_px, (cy, cx)

def patch_mean(arr, yi, xi):
    sub = arr[yi-HALF_PX:yi+HALF_PX+1, xi-HALF_PX:xi+HALF_PX+1]
    return float(np.nanmean(sub))

def main(tag):
    t0 = time.time()
    data, meta = load_channels(tag)
    shape = data[171].shape
    centers, rsun_px, (cy, cx) = build_patch_centers(meta, shape)
    print(f'[{tag}] grid step={GRID_STEP_ARCSEC}" -> {len(centers)} candidate centers; '
          f'rsun_px={rsun_px:.1f} half_px={HALF_PX}')

    # Gate on patch-mean 171 DN/s
    qualifying = []
    for (yi, xi) in centers:
        m171 = patch_mean(data[171], yi, xi)
        if LO_171 <= m171 <= HI_171:
            qualifying.append((yi, xi, m171))
    print(f'[{tag}] {len(qualifying)} patches pass 171 gate [{LO_171},{HI_171}] DN/s')

    # Deterministic selection of up to N_TARGET, evenly spaced over the qualifying list
    if len(qualifying) > N_TARGET:
        idx = np.linspace(0, len(qualifying)-1, N_TARGET).round().astype(int)
        idx = np.unique(idx)
        sel = [qualifying[i] for i in idx]
    else:
        sel = qualifying
    print(f'[{tag}] selected {len(sel)} patches')

    # DEM setup
    tlT, tm, names = kp.load_demregpy_response()
    temps = 10**np.arange(5.7, 7.2 + 0.05, 0.05)
    mlogt = np.array([np.mean([np.log10(temps[i]), np.log10(temps[i+1])])
                      for i in range(len(temps)-1)])
    # per-channel exposure for noise
    exps = np.array([meta and 0]*6, dtype=float)  # placeholder, set below
    exps = np.array([float(np.load(os.path.join(OUT, f'{tag}_{wv}_reg.npz'))['exptime'])
                     for wv in CHANNELS])

    recs = []
    for (yi, xi, m171) in sel:
        dn = np.array([patch_mean(data[wv], yi, xi) for wv in CHANNELS])
        # noise with per-channel exposure
        edn = kp.compute_aia_noise(dn, exposure_time=exps)
        dem, edem, elogt, chisq, dn_reg = dn2dem(dn, edn, tm, tlT, temps)
        dem = np.asarray(dem).ravel()
        pk = float(mlogt[int(np.argmax(dem))])
        fw = float(fwhm(mlogt, dem))
        ratio = float(dn[2] / dn[3])   # 171/193
        recs.append(dict(yi=yi, xi=xi, dn=dn.tolist(), m171=m171,
                         fwhm=fw, peak_logt=pk, ratio_171_193=ratio,
                         chisq=float(chisq)))
    fws = np.array([r['fwhm'] for r in recs])
    pks = np.array([r['peak_logt'] for r in recs])
    rts = np.array([r['ratio_171_193'] for r in recs])
    chs = np.array([r['chisq'] for r in recs])
    summ = dict(tag=tag, n_patches=len(recs),
                grid_step=GRID_STEP_ARCSEC,
                fwhm_median=float(np.nanmedian(fws)),
                fwhm_min=float(np.nanmin(fws)), fwhm_max=float(np.nanmax(fws)),
                fwhm_p05=float(np.nanpercentile(fws,5)),
                fwhm_p95=float(np.nanpercentile(fws,95)),
                peak_median=float(np.nanmedian(pks)),
                ratio_median=float(np.nanmedian(rts)),
                chisq_median=float(np.nanmedian(chs)))
    with open(os.path.join(OUT, f'{tag}_results.json'), 'w') as f:
        json.dump(dict(summary=summ, patches=recs,
                       meta=meta, exps=exps.tolist(), channels=CHANNELS), f, indent=1)
    np.savez_compressed(os.path.join(OUT, f'{tag}_results.npz'),
                        centers=np.array([(r['yi'],r['xi']) for r in recs]),
                        dn=np.array([r['dn'] for r in recs]),
                        fwhm=fws, peak_logt=pks, ratio=rts, chisq=chs, mlogt=mlogt)
    print('SUMMARY', json.dumps(summ, indent=1))
    print(f'done in {time.time()-t0:.1f}s')

if __name__ == '__main__':
    main(sys.argv[1])
