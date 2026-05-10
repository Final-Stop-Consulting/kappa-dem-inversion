#!/usr/bin/env python3
"""
Multi-thermal kappa test (R1 #1).

Forward-model a multi-T kappa source through the AIA pipeline:
  - DEM(T) = Brooks 2009 quiet_sun_eis.dem shape
  - Ion populations at each T set by Dz23 v10.1 kappa=2.5 ion fractions
  - Per-ion AIA channel contributions taken from the existing single-T
    checkpoint, with the ion-intrinsic emissivity factored out via the
    sec:perion factorization (eq. 2 of the paper):
        DN_i^kappa = sum_ions DN_i,ion^Mxw(T_eff) * f^kappa(T)/f^Mxw(T_eff)

Run synthetic DN through demregpy. Report FWHM, peak T, chi^2/dof.

Compare:
  - Single-T kappa  (existing) FWHM 0.222
  - Brooks-shape Mxw forward   FWHM 0.326
  - Multi-T kappa  (this test) FWHM ?
  - Real-QS distribution       FWHM 0.229--0.383 (median 0.277)
"""

import os, sys, json, time
import numpy as np
import scipy.io as sio
from pathlib import Path

# ---- paths (resolve relative to this script: Analysis/multi_thermal_kappa_test.py)
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'Data'
KAPPA_DIR = DATA_DIR / 'kappa_v10.1'
RESULTS_DIR = PROJECT_DIR / 'Analysis' / 'Results'
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
def compute_aia_noise(dn_per_channel, exposure=2.9):
    """Shot + read noise (matching pipeline)."""
    rdnse = np.array([1.14, 1.18, 1.15, 1.20, 1.20, 1.18])  # DN
    counts = dn_per_channel * exposure
    photon_noise = np.sqrt(np.maximum(counts, 0)) / exposure
    edn = np.sqrt(photon_noise**2 + rdnse**2)
    return edn


# ---- ioneq parser (replicates parse_ioneq from pipeline)
def parse_ioneq(path):
    """Parse CHIANTI .ioneq file. Returns logT array + nested dict ions[Z][stage] -> array."""
    with open(path) as f:
        lines = f.readlines()
    # First line: number of temps + number of elements
    parts = lines[0].split()
    n_t = int(parts[0])
    # Temperatures in line(s) following — read until we have n_t values
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
    # Ion fractions: each subsequent line is Z, stage, then n_t fractions (may span multiple lines)
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
        # Collect n_t fractions starting after Z, stage
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


# ---- Brooks DEM loader
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
    log_path = OUT_DIR / 'multi_thermal_kappa_test.log'
    log_f = open(log_path, 'w')
    def log(*a):
        msg = ' '.join(str(x) for x in a)
        print(msg)
        log_f.write(msg + '\n')
        log_f.flush()

    log('=' * 70)
    log('Multi-thermal kappa test (R1 #1)')
    log('=' * 70)

    # ---- load demregpy AIA response
    import demregpy
    trin = sio.readsav(demregpy.tresp.aia_tresp)
    tresp_logT_full = trin['logt']    # shape (101,) range 4--9
    tresp_full = trin['tr'].T         # shape (101, 6) — channels axis last
    log(f'AIA response: logT [{tresp_logT_full.min()}, {tresp_logT_full.max()}], '
        f'shape {tresp_full.shape}')

    # ---- DEM temperature grid (matches demregpy invocation in pipeline)
    log_temps = np.arange(DEM_LOG_T_MIN, DEM_LOG_T_MAX + DEM_DLOGT/2, DEM_DLOGT)
    temps_K = 10**log_temps
    log(f'DEM grid: log T = {log_temps[0]:.2f}–{log_temps[-1]:.2f}, '
        f'dlogT = {DEM_DLOGT}, n = {len(log_temps)}')

    # tresp at DEM grid
    tresp_grid = np.zeros((len(log_temps), 6))
    for j in range(6):
        tresp_grid[:, j] = np.interp(log_temps, tresp_logT_full, tresp_full[:, j])

    # ---- load Brooks DEM
    brooks_path = DATA_DIR / 'CHIANTI_11.0.2_database' / 'dem' / 'quiet_sun_eis.dem'
    brooks_logT, brooks_logDEM = load_brooks_dem(brooks_path)
    log(f'Brooks DEM: {len(brooks_logT)} knots, log T = '
        f'{brooks_logT.min()}–{brooks_logT.max()}')

    # interpolate Brooks DEM onto the full DEM grid (zero outside)
    brooks_dem = np.zeros_like(log_temps)
    in_range = (log_temps >= brooks_logT.min()) & (log_temps <= brooks_logT.max())
    brooks_dem[in_range] = 10**np.interp(log_temps[in_range], brooks_logT, brooks_logDEM)

    log(f'Brooks DEM peak: log T = {log_temps[np.argmax(brooks_dem)]:.3f}, '
        f'DEM = {brooks_dem.max():.3e}')
    log(f'Brooks integrated EM (linear T): {np.trapz(brooks_dem, temps_K):.3e}')
    log(f'Brooks integrated EM (∫ DEM dlnT): {np.trapz(brooks_dem * temps_K, log_temps * np.log(10)):.3e}')

    # ---- load Dz23 ion fractions
    log('\nLoading Dz23 ion fractions...')
    kappa_logT, kappa_ions = parse_ioneq(KAPPA_DIR / 'Dz23_kappa_2p5.ioneq')
    mxw_logT, mxw_ions = parse_ioneq(KAPPA_DIR / 'Dz23_mxw.ioneq')
    log(f'  kappa: {sum(len(v) for v in kappa_ions.values())} (Z, stage) entries')
    log(f'  mxw:   {sum(len(v) for v in mxw_ions.values())} (Z, stage) entries')
    log(f'  kappa logT range: {kappa_logT.min()}–{kappa_logT.max()}, n = {len(kappa_logT)}')

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

    # ---- factor out ion-intrinsic emissivity at T_eff
    # g_check(chan, ion) ~= A_X * f_Mxw(ion, T_eff) * emiss_intrinsic(chan, ion)
    # emiss_intrinsic_AX(chan, ion) := g_check(chan, ion) / f_Mxw(ion, T_eff)
    # so g_Mxw(chan, ion, T) ~= emiss_intrinsic_AX * f_Mxw(ion, T)
    # and g_kappa(chan, ion, T) ~= emiss_intrinsic_AX * f_kappa(ion, T)
    #
    # This treats ion-intrinsic emissivity (line strength × wavelength response)
    # as approximately T-independent near T_eff. Good for ions whose formation
    # T is near T_eff; less good for ions with formation T far from T_eff
    # (those contribute negligibly to AIA at T_eff anyway).

    log('\nFactoring out ion-intrinsic emissivity at T_eff...')
    f_mxw_at_Teff = {}
    f_kappa_at_Teff = {}
    for z in mxw_ions:
        for stage in mxw_ions[z]:
            f_mxw_at_Teff[(z, stage)] = interp_logT(mxw_logT, mxw_ions[z][stage], LOG_T_EFF)
    for z in kappa_ions:
        for stage in kappa_ions[z]:
            f_kappa_at_Teff[(z, stage)] = interp_logT(kappa_logT, kappa_ions[z][stage], LOG_T_EFF)

    emiss_intrinsic_AX = {}  # {(z,stage): {chan: g_check / f_Mxw_at_Teff}}
    skipped_zero_fmxw = 0
    for chan in AIA_CHANNELS:
        for (z, stage), g in contrib[chan].items():
            f0 = f_mxw_at_Teff.get((z, stage), 0.0)
            if f0 < 1e-30:
                skipped_zero_fmxw += 1
                continue
            emiss_intrinsic_AX.setdefault((z, stage), {})[chan] = g / f0
    log(f'  ions with valid intrinsic emissivity: {len(emiss_intrinsic_AX)}')
    log(f'  skipped (f_Mxw≈0 at T_eff): {skipped_zero_fmxw} (ion, channel) entries')

    # ---- precompute f_kappa(ion, T) and f_Mxw(ion, T) on the DEM grid
    log('\nInterpolating ion fractions onto DEM grid...')
    f_mxw_grid = {}
    f_kappa_grid = {}
    all_ions = set(emiss_intrinsic_AX.keys())
    for (z, stage) in all_ions:
        f_mxw_grid[(z, stage)] = np.array([
            interp_logT(mxw_logT, mxw_ions.get(z, {}).get(stage,
                np.zeros_like(mxw_logT)), lt) for lt in log_temps])
        f_kappa_grid[(z, stage)] = np.array([
            interp_logT(kappa_logT, kappa_ions.get(z, {}).get(stage,
                np.zeros_like(kappa_logT)), lt) for lt in log_temps])
    log(f'  done. {len(all_ions)} ions × {len(log_temps)} temperatures.')

    # ---- build synthetic per-channel response (Mxw) and forward DN (kappa)
    log('\nBuilding synthetic responses and DN...')

    # Per-channel synthetic Mxw response from checkpoint factorization
    tresp_synth_Mxw = np.zeros((len(log_temps), 6))
    tresp_synth_kappa = np.zeros((len(log_temps), 6))
    for j, chan in enumerate(AIA_CHANNELS):
        for (z, stage), chan_dict in emiss_intrinsic_AX.items():
            if chan not in chan_dict:
                continue
            base = chan_dict[chan]
            tresp_synth_Mxw[:, j]   += base * f_mxw_grid[(z, stage)]
            tresp_synth_kappa[:, j] += base * f_kappa_grid[(z, stage)]

    # Sanity: synth_Mxw at T_eff should equal sum of checkpoint entries
    log('\nSanity check: synth_Mxw(T_eff) vs checkpoint sum:')
    for j, chan in enumerate(AIA_CHANNELS):
        check_sum = sum(contrib[chan].values())
        idx_Teff = np.argmin(np.abs(log_temps - LOG_T_EFF))
        synth_at_Teff = tresp_synth_Mxw[idx_Teff, j]
        log(f'  {chan:>4d} Å: checkpoint Σ = {check_sum:.4e}, '
            f'synth(log T={log_temps[idx_Teff]:.3f}) = {synth_at_Teff:.4e}, '
            f'ratio = {synth_at_Teff/check_sum if check_sum else 0:.3f}')

    # Compare synth_Mxw to demregpy tresp shape (different normalization is fine,
    # just want to see if shape is reasonable)
    log('\nShape comparison synth_Mxw vs demregpy tresp (peak log T per channel):')
    for j, chan in enumerate(AIA_CHANNELS):
        peak_synth = log_temps[np.argmax(tresp_synth_Mxw[:, j])]
        peak_demreg = log_temps[np.argmax(tresp_grid[:, j])]
        log(f'  {chan:>4d} Å: synth peak = {peak_synth:.2f}, '
            f'demregpy peak = {peak_demreg:.2f}')

    # ---- forward-model: DN(channel) for Brooks DEM × κ ion fractions
    # Use the synth_kappa response (per-ion factorization)
    dlnT = np.log(10) * DEM_DLOGT
    # DN = ∫ Tresp(T) DEM(T) dT = ∫ Tresp(T) DEM(T) T dlnT
    dn_kappa_brooks = np.array([
        np.sum(tresp_synth_kappa[:, j] * brooks_dem * temps_K) * dlnT
        for j in range(6)
    ])
    dn_mxw_brooks = np.array([
        np.sum(tresp_synth_Mxw[:, j] * brooks_dem * temps_K) * dlnT
        for j in range(6)
    ])
    # Cross-check: pure demregpy tresp × Brooks (Maxwellian baseline)
    dn_mxw_demreg = np.array([
        np.sum(tresp_grid[:, j] * brooks_dem * temps_K) * dlnT
        for j in range(6)
    ])

    log('\nSynthetic DN/s/px (Brooks DEM):')
    log(f"{'Chan':>5s} {'kappa_synth':>14s} {'Mxw_synth':>14s} {'Mxw_demreg':>14s} {'k/M_synth':>10s}")
    for j, chan in enumerate(AIA_CHANNELS):
        rk_m = dn_kappa_brooks[j] / dn_mxw_brooks[j] if dn_mxw_brooks[j] else 0
        log(f'{chan:>5d} {dn_kappa_brooks[j]:>14.4e} {dn_mxw_brooks[j]:>14.4e} '
            f'{dn_mxw_demreg[j]:>14.4e} {rk_m:>10.3f}')

    # The synth Mxw and demreg tresp may have different normalizations; what
    # matters for the inversion is that we feed DN derived using a consistent
    # forward model, and invert with the demregpy response. To do this without
    # the normalization drift, use the κ/Mxw synth ratio to scale the demreg-
    # forward Mxw DNs:
    chan_ratio_kappa = dn_kappa_brooks / dn_mxw_brooks
    dn_kappa_for_inversion = dn_mxw_demreg * chan_ratio_kappa
    edn = compute_aia_noise(dn_kappa_for_inversion)

    log('\nDN for inversion (Mxw_demreg × κ/Mxw_synth_ratio):')
    log(f"{'Chan':>5s} {'DN':>14s} {'edn':>14s} {'frac_err':>10s}")
    for j, chan in enumerate(AIA_CHANNELS):
        log(f'{chan:>5d} {dn_kappa_for_inversion[j]:>14.4e} '
            f'{edn[j]:>14.4e} {edn[j]/dn_kappa_for_inversion[j]:>10.3f}')

    # ---- demregpy inversion
    log('\nRunning demregpy DEM inversion...')
    from demregpy import dn2dem
    t0 = time.time()
    dem, edem, elogt, chisq, dn_reg = dn2dem(
        dn_kappa_for_inversion[None, :],
        edn[None, :],
        tresp_grid,
        log_temps,
        temps_K
    )
    dt = time.time() - t0
    # For a single-pixel input, dem/edem/elogt have shape (30,), chisq is scalar, dn_reg has shape (6,)
    dem = np.asarray(dem).squeeze()
    edem = np.asarray(edem).squeeze()
    elogt = np.asarray(elogt).squeeze()
    chisq = float(np.asarray(chisq).squeeze())
    dn_reg = np.asarray(dn_reg).squeeze()
    log(f'  done in {dt:.1f}s; chi^2/dof = {chisq:.3f}; dem shape {dem.shape}')

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

    # FWHM by linear half-max interpolation
    half = dem.max() / 2
    above = dem >= half
    if above.any():
        first = np.argmax(above)
        last = len(above) - 1 - np.argmax(above[::-1])
        # interpolate left edge
        if first > 0:
            x0, x1 = log_T_centers[first-1], log_T_centers[first]
            y0, y1 = dem[first-1], dem[first]
            left = x0 + (half - y0) / (y1 - y0) * (x1 - x0)
        else:
            left = log_T_centers[first]
        # interpolate right edge
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
        rec = dn_reg[j] / dn_kappa_for_inversion[j] if dn_kappa_for_inversion[j] else 0
        log(f'  {chan:>4d} Å: {rec*100:>6.1f}%')

    # ---- save
    np.savez(OUT_DIR / 'multi_thermal_kappa_test.npz',
             log_temps=log_temps, temps_K=temps_K,
             brooks_dem=brooks_dem,
             tresp_synth_Mxw=tresp_synth_Mxw, tresp_synth_kappa=tresp_synth_kappa,
             tresp_demreg=tresp_grid,
             dn_kappa_brooks=dn_kappa_brooks, dn_mxw_brooks=dn_mxw_brooks,
             dn_mxw_demreg=dn_mxw_demreg, chan_ratio_kappa=chan_ratio_kappa,
             dn_kappa_for_inversion=dn_kappa_for_inversion, edn=edn,
             dem=dem, edem=edem, elogt=elogt, chisq=chisq, dn_reg=dn_reg,
             peak_logT=peak_logT, fwhm=fwhm)

    log('\n' + '=' * 70)
    log('SUMMARY')
    log('=' * 70)
    log('Multi-thermal kappa=2.5 (Brooks shape) recovered DEM:')
    log(f'  peak log T = {peak_logT:.3f}, FWHM = {fwhm:.3f}, chi2/dof = {chisq:.3f}')
    log('\nFor comparison:')
    log('  Single-T kappa=2.5 (existing):  FWHM 0.222')
    log('  Brooks-shape Mxw forward:       FWHM 0.326')
    log('  Real-QS distribution:           FWHM 0.229-0.383 (median 0.277)