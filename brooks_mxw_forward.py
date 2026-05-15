#!/usr/bin/env python3
"""
Brooks-shape Maxwellian forward model (paper Section 3.4, Table 4 / Table 5).

Mirror of `multi_thermal_kappa_test.py` with the kappa ion-fraction grid
replaced by the matched Maxwellian grid everywhere. Forward-models the
Brooks 2009 quiet_sun_eis.dem reference shape under pure Maxwellian
physics (no kappa correction) through the demregpy AIA pipeline:
  - DEM(T) = Brooks 2009 quiet_sun_eis.dem reference shape
  - Ion populations at each T set by Dz23 v10.1 matched Maxwellian
  - Per-ion AIA channel contributions taken from the existing per-ion
    T_eff checkpoint (kappa_dem_pipeline.py Stage 2), with the
    ion-intrinsic emissivity factored out via the Section 2.2
    factorization (paper Eq. 2):
        DN_i^Mxw(T) = sum_ions DN_i,ion^Mxw(T_eff) * f^Mxw(ion, T) / f^Mxw(ion, T_eff)

Runs the synthetic multi-T Maxwellian DN through demregpy with the same
Maxwellian response functions used by the single-T tests and the multi-T
kappa test. Reports recovered FWHM, peak log T, chi^2 (raw and / dof).

Used in paper Section 3.4 / Table 4 to anchor the FWHM ladder for the
Brooks-shape Maxwellian row (expected FWHM ~0.319):
    Isothermal Mxw baseline:           FWHM 0.174
    kappa=3 recovered:                 FWHM 0.191
    Brooks 2009 input shape (file):    FWHM 0.220
    kappa=2.5 single-T recovered:      FWHM 0.222
    Multi-T kappa=2.5 (Brooks shape):  FWHM 0.305
    Brooks-shape Mxw forward (this):   FWHM 0.319
    kappa=2 recovered:                 FWHM 0.353
    Real-QS distribution:              FWHM 0.229-0.383 (median 0.277)

Dependencies: numpy<2.0, scipy, demregpy. Reads the per-ion checkpoint at
Results/ion_contributions_checkpoint.json (produced by
kappa_dem_pipeline.py); does NOT require running ChiantiPy directly.

The factorization assumes ion-intrinsic emissivity (line strength times
wavelength response) is approximately T-independent near each ion's
formation temperature. This is the same approximation used in
`multi_thermal_kappa_test.py`; see that script's docstring for the
discussion of the 94 A high-T Fe XVIII edge case.

Usage:
    python brooks_mxw_forward.py
"""

import os, sys, json, time
import numpy as np
import scipy.io as sio
from pathlib import Path

# ---- paths (resolve relative to this script)
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / 'Data'
KAPPA_DIR = DATA_DIR / 'kappa_v10.1'
RESULTS_DIR = PROJECT_DIR / 'Results'
OUT_DIR = RESULTS_DIR
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Windows: ChiantiPy/demregpy environment fixes
if 'HOME' not in os.environ:
    os.environ['HOME'] = os.path.expanduser('~')

# ---- physical constants
T_EFF = 1.5e6
LOG_T_EFF = 6.176
N_E = 1.0e9
EM_SCALE = 1.0e27
AIA_CHANNELS = [94, 131, 171, 193, 211, 335]

DEM_LOG_T_MIN = 5.7
DEM_LOG_T_MAX = 7.2
DEM_DLOGT = 0.05

# ---- AIA noise model (matching kappa_dem_pipeline.py compute_aia_noise)
def compute_aia_noise(dn, exposure=2.9):
    """Shot + read noise matching kappa_dem_pipeline.compute_aia_noise.

    Uses the AIA dn2ph (DN-per-photon) conversion so the Poisson sqrt is taken
    on photon counts, not DN counts. dn2ph = gains * chan_wvl / 3397; shotnoise
    on DN/s is sqrt(dn2ph * DN_total) / dn2ph / exposure_time.
    """
    gains = np.array([18.3, 17.6, 17.7, 18.3, 18.3, 17.6])
    chan_wvl = np.array([94, 131, 171, 193, 211, 335])
    dn2ph = gains * chan_wvl / 3397.0
    rdnse = np.array([1.14, 1.18, 1.15, 1.20, 1.20, 1.18])
    dn_total = dn * exposure
    shotnoise = np.sqrt(dn2ph * np.abs(dn_total)) / dn2ph / exposure
    return np.sqrt(rdnse**2 + shotnoise**2)


# ---- ioneq parser (replicates parse_ioneq from pipeline / multi_thermal)
def parse_ioneq(path):
    """Parse CHIANTI .ioneq file. Returns logT array + nested dict ions[Z][stage] -> array."""
    with open(path) as f:
        lines = f.readlines()
    parts = lines[0].split()
    n_t = int(parts[0])
    temps = []
    line_idx = 1
    while len(temps) < n_t and line_idx < len(lines):
        tokens = lines[line_idx].split()
        for t in tokens:
            try:
                temps.append(float(t))
                if len(temps) >= n_t:
                    break
            except ValueError:
                pass
        line_idx += 1
    logT = np.array(temps)
    ions = {}
    while line_idx < len(lines):
        ln = lines[line_idx].strip()
        if not ln or ln.startswith('-1') or ln.startswith('%'):
            line_idx += 1
            if ln.startswith('-1') or ln.startswith('%'):
                break
            continue
        tokens = ln.split()
        if len(tokens) < 2:
            line_idx += 1
            continue
        try:
            z = int(tokens[0])
            stage = int(tokens[1])
        except ValueError:
            line_idx += 1
            continue
        fracs = []
        for tk in tokens[2:]:
            try:
                fracs.append(float(tk))
            except ValueError:
                pass
        line_idx += 1
        while len(fracs) < n_t and line_idx < len(lines):
            for tk in lines[line_idx].split():
                try:
                    fracs.append(float(tk))
                    if len(fracs) >= n_t:
                        break
                except ValueError:
                    pass
            line_idx += 1
        if len(fracs) == n_t:
            ions.setdefault(z, {})[stage] = np.array(fracs)
    return logT, ions


def interp_logT(grid_logT, frac_arr, logT):
    """Log-linear interpolation in logT, linear in fraction (returns 0 below floor)."""
    if logT < grid_logT[0] or logT > grid_logT[-1]:
        return 0.0
    return float(np.interp(logT, grid_logT, frac_arr))


def load_brooks_dem(path):
    """Read CHIANTI .dem file. Returns (logT, log10_DEM) on the file's grid."""
    logT_pts, logDEM_pts = [], []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith('%') or ln.startswith('-1'):
                if ln.startswith('-1'):
                    break
                continue
            parts = ln.split()
            if len(parts) >= 2:
                try:
                    logT_pts.append(float(parts[0]))
                    logDEM_pts.append(float(parts[1]))
                except ValueError:
                    continue
    return np.array(logT_pts), np.array(logDEM_pts)


# ---- main pipeline
def main():
    log_path = OUT_DIR / 'brooks_mxw_forward_results.txt'
    log_f = open(log_path, 'w', encoding='utf-8')
    def log(*a):
        msg = ' '.join(str(x) for x in a)
        print(msg)
        log_f.write(msg + '\n')
        log_f.flush()

    log('=' * 70)
    log('Brooks-shape Maxwellian forward model (paper Section 3.4, Table 4)')
    log('=' * 70)

    # ---- load demregpy AIA response
    import demregpy
    trin = sio.readsav(demregpy.tresp.aia_tresp)
    tresp_logT_full = trin['logt']    # shape (101,) range 4--9
    tresp_full = trin['tr'].T         # shape (101, 6) -- channels axis last
    log(f'AIA response: logT [{tresp_logT_full.min()}, {tresp_logT_full.max()}], '
        f'shape {tresp_full.shape}')

    # ---- DEM temperature grid (matches demregpy invocation in pipeline)
    log_temps = np.arange(DEM_LOG_T_MIN, DEM_LOG_T_MAX + DEM_DLOGT/2, DEM_DLOGT)
    temps_K = 10**log_temps
    log(f'DEM grid: log T = {log_temps[0]:.2f}-{log_temps[-1]:.2f}, '
        f'dlogT = {DEM_DLOGT}, n = {len(log_temps)}')

    # tresp at DEM grid
    tresp_grid = np.zeros((len(log_temps), 6))
    for j in range(6):
        tresp_grid[:, j] = np.interp(log_temps, tresp_logT_full, tresp_full[:, j])

    # ---- load Brooks DEM
    brooks_path = DATA_DIR / 'CHIANTI_11.0.2_database' / 'dem' / 'quiet_sun_eis.dem'
    brooks_logT, brooks_logDEM = load_brooks_dem(brooks_path)
    log(f'Brooks DEM: {len(brooks_logT)} knots, log T = '
        f'{brooks_logT.min()}-{brooks_logT.max()}')

    # interpolate Brooks DEM onto the full DEM grid (zero outside)
    brooks_dem = np.zeros_like(log_temps)
    in_range = (log_temps >= brooks_logT.min()) & (log_temps <= brooks_logT.max())
    brooks_dem[in_range] = 10**np.interp(log_temps[in_range], brooks_logT, brooks_logDEM)

    log(f'Brooks DEM peak: log T = {log_temps[np.argmax(brooks_dem)]:.3f}, '
        f'DEM = {brooks_dem.max():.3e}')
    log(f'Brooks integrated EM (linear T): {np.trapz(brooks_dem, temps_K):.3e}')

    # ---- load Dz23 matched Maxwellian ion fractions (no kappa side needed)
    log('\nLoading Dz23 matched Maxwellian ion fractions...')
    mxw_logT, mxw_ions = parse_ioneq(KAPPA_DIR / 'Dz23_mxw.ioneq')
    log(f'  mxw:   {sum(len(v) for v in mxw_ions.values())} (Z, stage) entries')
    log(f'  mxw logT range: {mxw_logT.min()}-{mxw_logT.max()}, n = {len(mxw_logT)}')

    # ---- load existing checkpoint (per-ion contributions at T_eff)
    log('\nLoading per-ion checkpoint at T_eff = 1.5 MK...')
    with open(RESULTS_DIR / 'ion_contributions_checkpoint.json') as f:
        saved = json.load(f)
    contrib = {}
    for chan_str, ion_dict in saved.items():
        chan = int(chan_str)
        contrib[chan] = {}
        for key_str, val in ion_dict.items():
            z, stage = map(int, key_str.split(','))
            contrib[chan][(z, stage)] = float(val)
    log(f'  channels: {sorted(contrib.keys())}')
    log(f'  total (ion, channel) entries: {sum(len(v) for v in contrib.values())}')

    # ---- factor out ion-intrinsic emissivity at T_eff (Maxwellian only)
    # g_check(chan, ion) ~= A_X * f_Mxw(ion, T_eff) * emiss_intrinsic(chan, ion)
    # emiss_intrinsic_AX(chan, ion) := g_check(chan, ion) / f_Mxw(ion, T_eff)
    # g_Mxw(chan, ion, T) ~= emiss_intrinsic_AX * f_Mxw(ion, T)
    log('\nFactoring out ion-intrinsic emissivity at T_eff...')
    f_mxw_at_Teff = {}
    for z in mxw_ions:
        for stage in mxw_ions[z]:
            f_mxw_at_Teff[(z, stage)] = interp_logT(mxw_logT, mxw_ions[z][stage], LOG_T_EFF)

    emiss_intrinsic_AX = {}
    skipped_zero_fmxw = 0
    for chan in AIA_CHANNELS:
        for (z, stage), g in contrib[chan].items():
            f0 = f_mxw_at_Teff.get((z, stage), 0.0)
            if f0 < 1e-30:
                skipped_zero_fmxw += 1
                continue
            emiss_intrinsic_AX.setdefault((z, stage), {})[chan] = g / f0
    log(f'  ions with valid intrinsic emissivity: {len(emiss_intrinsic_AX)}')
    log(f'  skipped (f_Mxw~0 at T_eff): {skipped_zero_fmxw} (ion, channel) entries')

    # ---- precompute f_Mxw(ion, T) on the DEM grid
    log('\nInterpolating Mxw ion fractions onto DEM grid...')
    f_mxw_grid = {}
    all_ions = set(emiss_intrinsic_AX.keys())
    for (z, stage) in all_ions:
        f_mxw_grid[(z, stage)] = np.array([
            interp_logT(mxw_logT, mxw_ions.get(z, {}).get(stage,
                np.zeros_like(mxw_logT)), lt) for lt in log_temps])
    log(f'  done. {len(all_ions)} ions x {len(log_temps)} temperatures.')

    # ---- build synthetic per-channel Mxw response, forward-model Brooks DEM
    log('\nBuilding synthetic Maxwellian response and DN...')
    tresp_synth_Mxw = np.zeros((len(log_temps), 6))
    for j, chan in enumerate(AIA_CHANNELS):
        for (z, stage), chan_dict in emiss_intrinsic_AX.items():
            if chan not in chan_dict:
                continue
            base = chan_dict[chan]
            tresp_synth_Mxw[:, j] += base * f_mxw_grid[(z, stage)]

    # ---- forward DN(channel) for Brooks DEM under pure Maxwellian
    # DN = integral Tresp(T) DEM(T) dT = integral Tresp(T) DEM(T) T dlnT
    dlnT = np.log(10) * DEM_DLOGT
    dn_mxw_brooks = np.array([
        np.sum(tresp_synth_Mxw[:, j] * brooks_dem * temps_K) * dlnT
        for j in range(6)
    ])
    # Cross-check: pure demregpy tresp x Brooks (Maxwellian baseline)
    dn_mxw_demreg = np.array([
        np.sum(tresp_grid[:, j] * brooks_dem * temps_K) * dlnT
        for j in range(6)
    ])

    log('\nSynthetic DN/s/px (Brooks DEM, pure Maxwellian):')
    log(f"{'Chan':>5s} {'Mxw_synth':>14s} {'Mxw_demreg':>14s} {'ratio':>10s}")
    for j, chan in enumerate(AIA_CHANNELS):
        rs_d = dn_mxw_brooks[j] / dn_mxw_demreg[j] if dn_mxw_demreg[j] else 0
        log(f'{chan:>5d} {dn_mxw_brooks[j]:>14.4e} '
            f'{dn_mxw_demreg[j]:>14.4e} {rs_d:>10.3f}')

    # The synth Mxw and demreg tresp may have different normalizations; for the
    # pure-Maxwellian forward model the demreg tresp is itself the calibrated
    # response. Use dn_mxw_demreg directly (it already absorbs the absolute
    # calibration); this is the cleanest pure-Mxw forward.
    dn_for_inversion = dn_mxw_demreg
    edn = compute_aia_noise(dn_for_inversion)

    log('\nDN for inversion (demregpy tresp x Brooks DEM):')
    log(f"{'Chan':>5s} {'DN':>14s} {'edn':>14s} {'frac_err':>10s}")
    for j, chan in enumerate(AIA_CHANNELS):
        log(f'{chan:>5d} {dn_for_inversion[j]:>14.4e} '
            f'{edn[j]:>14.4e} {edn[j]/dn_for_inversion[j]:>10.3f}')

    # ---- demregpy inversion
    log('\nRunning demregpy DEM inversion...')
    from demregpy import dn2dem
    t0 = time.time()
    dem, edem, elogt, chisq, dn_reg = dn2dem(
        dn_for_inversion[None, :],
        edn[None, :],
        tresp_grid,
        log_temps,
        temps_K
    )
    dt = time.time() - t0
    dem = np.asarray(dem).squeeze()
    edem = np.asarray(edem).squeeze()
    elogt = np.asarray(elogt).squeeze()
    chisq = float(np.asarray(chisq).squeeze())
    dn_reg = np.asarray(dn_reg).squeeze()
    log(f'  done in {dt:.1f}s; chi^2 (raw) = {chisq:.3f}, chi^2/dof = {chisq/5:.3f}; dem shape {dem.shape}')

    # bin centers (DEM grid)
    log_T_centers = (log_temps[:-1] + log_temps[1:]) / 2 if len(dem) == len(log_temps) - 1 else log_temps[:len(dem)]
    if len(log_T_centers) != len(dem):
        log_T_centers = log_temps[:len(dem)]

    # ---- shape metrics
    log('\nDEM shape metrics:')
    peak_idx = int(np.argmax(dem))
    peak_logT = log_T_centers[peak_idx]
    peak_T = 10**peak_logT
    log(f'  peak: log T = {peak_logT:.3f}, T = {peak_T/1e6:.2f} MK')

    half = dem.max() / 2
    above = dem >= half
    if above.any():
        first = np.argmax(above)
        last = len(above) - 1 - np.argmax(above[::-1])
        if first > 0:
            x0, x1 = log_T_centers[first-1], log_T_centers[first]
            y0, y1 = dem[first-1], dem[first]
            left = x0 + (half - y0) / (y1 - y0) * (x1 - x0)
        else:
            left = log_T_centers[first]
        if last < len(log_T_centers) - 1:
            x0, x1 = log_T_centers[last], log_T_centers[last+1]
            y0, y1 = dem[last], dem[last+1]
            right = x0 + (half - y0) / (y1 - y0) * (x1 - x0)
        else:
            right = log_T_centers[last]
        fwhm = right - left
        log(f'  FWHM = {fwhm:.3f} dex (log T)')
    else:
        fwhm = float('nan')
        log('  FWHM: undefined (no half-max crossing)')

    log('\nDN recovery (predicted from recovered DEM / observed):')
    for j, chan in enumerate(AIA_CHANNELS):
        rec = dn_reg[j] / dn_for_inversion[j] if dn_for_inversion[j] else 0
        log(f'  {chan:>4d} A: {rec*100:>6.1f}%')

    # ---- save
    np.savez(OUT_DIR / 'brooks_mxw_forward_results.npz',
             log_temps=log_temps, temps_K=temps_K,
             brooks_dem=brooks_dem,
             tresp_synth_Mxw=tresp_synth_Mxw,
             tresp_demreg=tresp_grid,
             dn_mxw_brooks=dn_mxw_brooks, dn_mxw_demreg=dn_mxw_demreg,
             dn_for_inversion=dn_for_inversion, edn=edn,
             dem=dem, edem=edem, elogt=elogt, chisq=chisq, dn_reg=dn_reg,
             peak_logT=peak_logT, fwhm=fwhm)

    log('\n' + '=' * 70)
    log('SUMMARY')
    log('=' * 70)
    log('Brooks-shape Maxwellian forward (no kappa) recovered DEM:')
    log(f'  peak log T = {peak_logT:.3f}, FWHM = {fwhm:.3f}, '
        f'chi^2 (raw) = {chisq:.3f}, chi^2/dof = {chisq/5:.3f}')
    log('\nFor comparison (paper Table 4):')
    log('  Single-T kappa=2.5 recovered:   FWHM 0.222')
    log('  Multi-T kappa=2.5 (Brooks):     FWHM 0.305')
    log('  Brooks-shape Mxw forward (this): FWHM ~0.319')
    log('  Real-QS distribution:           FWHM 0.229-0.383 (median 0.277)')

    log_f.close()
    return {'fwhm': fwhm, 'peak_logT': peak_logT, 'chisq': chisq}


if __name__ == '__main__':
    main()
