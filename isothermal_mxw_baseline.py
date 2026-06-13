#!/usr/bin/env python3
"""
isothermal_mxw_baseline.py — Algorithmic-floor measurement of demregpy FWHM (paper Section 3.4, Table 4).

Forward-models a pure isothermal Maxwellian at log T = 6.176 (T_eff = 1.5 MK)
with EM_SCALE = 1e27, no kappa modification. Runs through the same demregpy
pipeline configuration as the single-T kappa and multi-T kappa tests. Reports
recovered FWHM, peak log T, chi^2 (raw and / dof).

This is the demregpy algorithmic floor: the FWHM a single-T delta-function
input recovers due to the AIA response kernels + GSVD-selected regularization
smoothing, with no underlying physical broadening from a kappa distribution.
All physical broadening above this floor reflects either kappa-driven
charge-state redistribution or multi-T integration along the line of sight.

Used in paper Section 3.4 / Table 4 to anchor the FWHM ladder:
    Isothermal Mxw baseline:  0.174  (this script)
    kappa=3 recovered:        0.191
    kappa=2.5 recovered:      0.222
    Brooks-shape Mxw forward: 0.319 (multi-T)
    multi-T kappa=2.5:        0.305
    kappa=2 recovered:        0.353
    Real-QS distribution:     0.229-0.383 (median 0.277)

Dependencies: numpy<2.0, scipy, demregpy. Does NOT require ChiantiPy or the
per-ion checkpoint -- runs purely off the demregpy-bundled AIA temperature
response functions.

Usage:
    python isothermal_mxw_baseline.py
"""
import os
import time
import numpy as np
import scipy.io as sio
from pathlib import Path

if 'HOME' not in os.environ:
    os.environ['HOME'] = os.path.expanduser('~')

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / 'Results'
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

LOG_T_EFF = 6.176
EM_SCALE = 1.0e27
AIA_CHANNELS = [94, 131, 171, 193, 211, 335]

DEM_LOG_T_MIN = 5.7
DEM_LOG_T_MAX = 7.2
DEM_DLOGT = 0.05


def compute_aia_noise(dn, exposure=2.9):
    """Shot + read noise matching kappa_dem_pipeline.compute_aia_noise.

    Uses the AIA dn2ph (DN-per-photon) conversion so the Poisson sqrt is taken
    on photon counts, not DN counts. dn2ph = gains * chan_wvl / 3397 (the
    silicon e-h energy / photon-energy factor); shotnoise on DN/s is then
    sqrt(dn2ph * DN_total) / dn2ph / exposure_time.
    """
    gains = np.array([18.3, 17.6, 17.7, 18.3, 18.3, 17.6])
    chan_wvl = np.array([94, 131, 171, 193, 211, 335])
    dn2ph = gains * chan_wvl / 3397.0
    rdnse = np.array([1.14, 1.18, 1.15, 1.20, 1.20, 1.18])
    dn_total = dn * exposure
    shotnoise = np.sqrt(dn2ph * np.abs(dn_total)) / dn2ph / exposure
    return np.sqrt(rdnse**2 + shotnoise**2)


def interp_edge(x0, x1, y0, y1, y):
    """Linear half-max interpolation between two adjacent grid points."""
    if y1 == y0:
        return (x0 + x1) / 2
    return x0 + (y - y0) / (y1 - y0) * (x1 - x0)


def compute_fwhm(log_T_centers, dem):
    """FWHM in log T via linear half-max interpolation on a DEM curve."""
    half = dem.max() / 2
    above = dem >= half
    if not above.any():
        return float('nan')
    first = int(np.argmax(above))
    last = len(above) - 1 - int(np.argmax(above[::-1]))
    left = (interp_edge(log_T_centers[first-1], log_T_centers[first],
                        dem[first-1], dem[first], half)
            if first > 0 else log_T_centers[first])
    right = (interp_edge(log_T_centers[last], log_T_centers[last+1],
                         dem[last], dem[last+1], half)
             if last+1 < len(log_T_centers) else log_T_centers[last])
    return right - left


def main():
    log_path = RESULTS_DIR / 'isothermal_mxw_baseline_results.txt'
    log_f = open(log_path, 'w', encoding='utf-8')

    def log(*a):
        msg = ' '.join(str(x) for x in a)
        print(msg)
        log_f.write(msg + '\n')
        log_f.flush()

    log('=' * 70)
    log('Isothermal Maxwellian baseline FWHM through demregpy')
    log('=' * 70)
    log(f'SCRIPT_DIR   = {SCRIPT_DIR}')
    log(f'RESULTS_DIR  = {RESULTS_DIR}')
    log('')

    # ---- load demregpy AIA response
    import demregpy
    from demregpy import dn2dem
    trin = sio.readsav(demregpy.tresp.aia_tresp)
    tresp_logT_full = trin['logt']
    tresp_full = trin['tr'].T
    log(f'AIA response: logT [{tresp_logT_full.min()}, {tresp_logT_full.max()}], '
        f'shape {tresp_full.shape}')

    # ---- DEM temperature grid
    log_temps = np.arange(DEM_LOG_T_MIN, DEM_LOG_T_MAX + DEM_DLOGT/2, DEM_DLOGT)
    temps_K = 10**log_temps
    log(f'DEM grid: log T = {log_temps[0]:.2f}-{log_temps[-1]:.2f}, '
        f'dlogT = {DEM_DLOGT}, n = {len(log_temps)}')

    tresp_grid = np.zeros((len(log_temps), 6))
    for j in range(6):
        tresp_grid[:, j] = np.interp(log_temps, tresp_logT_full, tresp_full[:, j])

    # ---- synthesize isothermal-Maxwellian DN at log T = T_eff
    dn_mxw_iso = np.zeros(6)
    for j in range(6):
        dn_mxw_iso[j] = np.interp(LOG_T_EFF, tresp_logT_full, tresp_full[:, j])
    dn_mxw_iso *= EM_SCALE
    edn = compute_aia_noise(dn_mxw_iso)

    log(f'\nIsothermal Maxwellian (no kappa) DN/s/px at log T = {LOG_T_EFF}:')
    log(f"{'Chan':>5s} {'DN':>14s} {'edn':>14s} {'frac':>10s}")
    for j, c in enumerate(AIA_CHANNELS):
        log(f'{c:>5d} {dn_mxw_iso[j]:>14.4e} {edn[j]:>14.4e} '
            f'{edn[j]/dn_mxw_iso[j]:>10.3f}')

    # ---- demregpy inversion
    log('\nRunning demregpy DEM inversion...')
    t0 = time.time()
    dem, edem, elogt, chisq, dn_reg = dn2dem(
        dn_mxw_iso[None, :], edn[None, :], tresp_grid, log_temps, temps_K
    )
    dt = time.time() - t0

    dem = np.asarray(dem).squeeze()
    edem = np.asarray(edem).squeeze()
    elogt = np.asarray(elogt).squeeze()
    chisq = float(np.asarray(chisq).squeeze())
    dn_reg = np.asarray(dn_reg).squeeze()
    log(f'  done in {dt:.2f}s; chi^2 (raw) = {chisq:.3f}, chi^2/dof = {chisq/5:.3f}')

    log_T_centers = log_temps[:len(dem)]
    peak_idx = int(np.argmax(dem))
    peak_logT = log_T_centers[peak_idx]
    fwhm = compute_fwhm(log_T_centers, dem)

    log(f'\nIsothermal Maxwellian (no kappa) recovered DEM:')
    log(f'  peak log T = {peak_logT:.3f}')
    log(f'  T_peak     = {10**peak_logT / 1e6:.2f} MK')
    log(f'  FWHM       = {fwhm:.3f} dex (log T)')
    log(f'  chi^2 (raw) = {chisq:.3f}; chi^2/dof = {chisq/5:.3f}')

    log('\nFor comparison (paper Table 4):')
    log('  Isothermal Mxw (this script):  FWHM ~0.174 (algorithmic floor)')
    log('  kappa=3 recovered:             FWHM 0.191')
    log('  kappa=2.5 recovered:           FWHM 0.222')
    log('  Brooks-shape Mxw forward:      FWHM 0.319 (multi-T)')
    log('  Multi-T kappa=2.5:             FWHM 0.305')
    log('  kappa=2 recovered:             FWHM 0.353')
    log('  Real-QS distribution:          FWHM 0.229-0.383 (median 0.277)')

    np.savez(RESULTS_DIR / 'isothermal_mxw_baseline_results.npz',
             log_temps=log_temps, temps_K=temps_K,
             tresp_grid=tresp_grid,
             dn_mxw_iso=dn_mxw_iso, edn=edn,
             dem=dem, edem=edem, elogt=elogt, chisq=chisq, dn_reg=dn_reg,
             peak_logT=peak_logT, fwhm=fwhm)
    log(f'\nNumerical output saved to {RESULTS_DIR / "isothermal_mxw_baseline_results.npz"}')

    log_f.close()
    return {'fwhm': fwhm, 'peak_logT': peak_logT, 'chisq': chisq}


if __name__ == '__main__':
    main()
