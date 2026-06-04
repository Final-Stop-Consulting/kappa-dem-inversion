#!/usr/bin/env python3
"""
Reference-DEM robustness runner (paper revision, reviewer concern R4.4).

Question: are the multi-thermal degeneracy conclusions unchanged if the
Brooks 2009 reference DEM is replaced by other published quiet-Sun DEMs?

This script takes ANY quiet-Sun DEM(T) curve (two-column CHIANTI .dem format)
and, through the SAME pipeline used in the paper (multi_thermal_kappa_test.py),
forward-models two source families on that DEM shape and inverts them with
demregpy:

  (A) multi-thermal MAXWELLIAN on the DEM shape
  (B) multi-thermal KAPPA=2.5 on the DEM shape (Dz23 v10.1 ion fractions)

then reports recovered FWHM(logT), peak logT and chi^2/dof for each. The point
is to show both families land in the real-QS FWHM band (median ~0.283, range
~0.230-0.401) regardless of which reference DEM is used -- i.e. the degeneracy
is robust to the reference choice.

The forward model is NOT reinvented here: it imports the building blocks of
multi_thermal_kappa_test.py (load_brooks_dem, parse_ioneq, interp_logT,
compute_aia_noise) and reuses the per-ion checkpoint
(Results/ion_contributions_checkpoint.json) plus the Section-2.2 ion-intrinsic
factorization. The only generalization is that the DEM-file path is a parameter,
so the same forward model runs on an arbitrary .dem and we loop over a list.

Pipeline (identical to multi_thermal_kappa_test.py "published footing"):
  - demregpy AIA temperature response (aia_tresp).
  - temps grid 10**arange(5.7, 7.2+0.05, 0.05).
  - per-channel synthetic Mxw / kappa responses from the per-ion checkpoint
    factorized over Dz23 mxw / kappa=2.5 ion fractions.
  - Maxwellian forward DN = demregpy tresp x DEM (channel-integrated Mxw, A).
  - kappa forward DN = (demregpy tresp x DEM) x (synth_kappa/synth_Mxw channel
    ratio), the published normalization-drift-free construction (family B).
  - kp.compute_aia_noise for the error vector.
  - dn2dem inversion against the demregpy response grid.
  - FWHM via linear half-max interpolation on the bin-center grid.

Inputs processed:
  - Brooks 2009  quiet_sun_eis.dem   (VALIDATION GATE: Mxw 0.319, kappa 0.305)
  - CHIANTI std  quiet_sun.dem       (additional reference)
  - any *.dem the user drops in   outputs/refdem/inputs/

DEM file format expected: two whitespace-separated columns
      log10 T [K]      log10 DEM [cm^-5 K^-1]
one knot per line; lines beginning with '%' are comments; a line beginning
with '-1' terminates the data (standard CHIANTI .dem convention). The FWHM is
normalization-invariant, so an arbitrary additive offset in log DEM does not
matter; curves are clipped/interpolated to the response T-range automatically.

Usage:
    python3 refdem_robustness.py
Outputs (to outputs/refdem/):
    refdem_results.txt, refdem_results.json
"""

import os, sys, json, time
import numpy as np
import scipy.io as sio
from pathlib import Path

# ---- locate the paper repo and import the existing forward-model building blocks
REPO = Path(os.environ.get('FB_REPO', Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO))
import multi_thermal_kappa_test as mt  # noqa: E402

# reuse the EXACT building blocks (no reinvention)
load_dem          = mt.load_brooks_dem      # CHIANTI .dem reader (logT, log10 DEM)
parse_ioneq       = mt.parse_ioneq
interp_logT       = mt.interp_logT
compute_aia_noise = mt.compute_aia_noise

# reuse the same constants
LOG_T_EFF    = mt.LOG_T_EFF       # 6.176
AIA_CHANNELS = mt.AIA_CHANNELS    # [94,131,171,193,211,335]
DEM_LOG_T_MIN = mt.DEM_LOG_T_MIN  # 5.7
DEM_LOG_T_MAX = mt.DEM_LOG_T_MAX  # 7.2
DEM_DLOGT     = mt.DEM_DLOGT      # 0.05

DATA_DIR    = mt.DATA_DIR
KAPPA_DIR   = mt.KAPPA_DIR
RESULTS_DIR = mt.RESULTS_DIR

OUT_DIR    = Path(os.environ.get('FB_OUT', str(REPO / 'Results')))
INPUTS_DIR = Path(os.environ.get('REFDEM_INPUTS', str(REPO / 'Data' / 'reference_dems')))
OUT_DIR.mkdir(parents=True, exist_ok=True)
INPUTS_DIR.mkdir(parents=True, exist_ok=True)

if 'HOME' not in os.environ:
    os.environ['HOME'] = os.path.expanduser('~')

# real-QS FWHM band (paper)
QS_BAND = (0.230, 0.401)
QS_MEDIAN = 0.283


# ---- FWHM via linear half-max interpolation (spec-mandated implementation)
def fwhm(x, y):
    h = y.max() / 2
    ab = y >= h
    if not ab.any():
        return float('nan')
    f = int(np.argmax(ab))
    l = len(ab) - 1 - int(np.argmax(ab[::-1]))
    le = x[f] if f == 0 else x[f - 1] + (h - y[f - 1]) / (y[f] - y[f - 1]) * (x[f] - x[f - 1])
    ri = x[l] if l + 1 >= len(x) else x[l] + (h - y[l]) / (y[l + 1] - y[l]) * (x[l + 1] - x[l])
    return ri - le


# ============================================================================
# One-time setup: response grid + per-ion factorized synthetic responses.
# DEM-independent: built ONCE and reused for every reference DEM.
# Mirrors multi_thermal_kappa_test.main() exactly, minus the DEM-specific part.
# ============================================================================
def build_forward_model(log):
    import demregpy
    trin = sio.readsav(demregpy.tresp.aia_tresp)
    tresp_logT_full = trin['logt']        # (101,)
    tresp_full = trin['tr'].T             # (101, 6) channels last

    log_temps = np.arange(DEM_LOG_T_MIN, DEM_LOG_T_MAX + DEM_DLOGT / 2, DEM_DLOGT)
    temps_K = 10 ** log_temps

    tresp_grid = np.zeros((len(log_temps), 6))
    for j in range(6):
        tresp_grid[:, j] = np.interp(log_temps, tresp_logT_full, tresp_full[:, j])

    kappa_logT, kappa_ions = parse_ioneq(KAPPA_DIR / 'Dz23_kappa_2p5.ioneq')
    mxw_logT,   mxw_ions   = parse_ioneq(KAPPA_DIR / 'Dz23_mxw.ioneq')

    with open(RESULTS_DIR / 'ion_contributions_checkpoint.json') as f:
        saved = json.load(f)
    contrib = {}
    for chan_str, ion_dict in saved.items():
        chan = int(chan_str)
        contrib[chan] = {}
        for key_str, val in ion_dict.items():
            z, stage = map(int, key_str.split(','))
            contrib[chan][(z, stage)] = float(val)

    f_mxw_at_Teff = {}
    for z in mxw_ions:
        for stage in mxw_ions[z]:
            f_mxw_at_Teff[(z, stage)] = interp_logT(mxw_logT, mxw_ions[z][stage], LOG_T_EFF)

    emiss_intrinsic_AX = {}
    for chan in AIA_CHANNELS:
        for (z, stage), g in contrib[chan].items():
            f0 = f_mxw_at_Teff.get((z, stage), 0.0)
            if f0 < 1e-30:
                continue
            emiss_intrinsic_AX.setdefault((z, stage), {})[chan] = g / f0

    all_ions = set(emiss_intrinsic_AX.keys())
    f_mxw_grid, f_kappa_grid = {}, {}
    for (z, stage) in all_ions:
        f_mxw_grid[(z, stage)] = np.array([
            interp_logT(mxw_logT, mxw_ions.get(z, {}).get(stage, np.zeros_like(mxw_logT)), lt)
            for lt in log_temps])
        f_kappa_grid[(z, stage)] = np.array([
            interp_logT(kappa_logT, kappa_ions.get(z, {}).get(stage, np.zeros_like(kappa_logT)), lt)
            for lt in log_temps])

    tresp_synth_Mxw = np.zeros((len(log_temps), 6))
    tresp_synth_kappa = np.zeros((len(log_temps), 6))
    for j, chan in enumerate(AIA_CHANNELS):
        for (z, stage), chan_dict in emiss_intrinsic_AX.items():
            if chan not in chan_dict:
                continue
            base = chan_dict[chan]
            tresp_synth_Mxw[:, j]   += base * f_mxw_grid[(z, stage)]
            tresp_synth_kappa[:, j] += base * f_kappa_grid[(z, stage)]

    log(f'  forward model built: {len(all_ions)} ions, '
        f'DEM grid log T {log_temps[0]:.2f}-{log_temps[-1]:.2f} (n={len(log_temps)})')

    return dict(log_temps=log_temps, temps_K=temps_K, tresp_grid=tresp_grid,
                tresp_synth_Mxw=tresp_synth_Mxw, tresp_synth_kappa=tresp_synth_kappa)


# ---- load + regrid an arbitrary DEM onto the response grid (normalization-safe)
def load_dem_on_grid(path, log_temps):
    dem_logT, dem_logDEM = load_dem(path)
    if len(dem_logT) < 2:
        raise ValueError(f'{path}: fewer than 2 valid DEM knots')
    order = np.argsort(dem_logT)
    dem_logT, dem_logDEM = dem_logT[order], dem_logDEM[order]
    dem = np.zeros_like(log_temps)
    in_range = (log_temps >= dem_logT.min()) & (log_temps <= dem_logT.max())
    dem[in_range] = 10 ** np.interp(log_temps[in_range], dem_logT, dem_logDEM)
    overlap = int(in_range.sum())
    return dem, dem_logT, overlap


# ---- run both source families on one DEM through the published pipeline
def run_one_dem(name, dem_path, fm, log):
    from demregpy import dn2dem
    log_temps = fm['log_temps']
    temps_K   = fm['temps_K']
    tresp_grid = fm['tresp_grid']

    dem, dem_logT, overlap = load_dem_on_grid(dem_path, log_temps)
    if dem.max() <= 0:
        raise ValueError(f'{name}: DEM has no positive overlap with response grid')

    peak_in_logT = log_temps[int(np.argmax(dem))]
    log(f'  loaded {name}: knots log T {dem_logT.min():.2f}-{dem_logT.max():.2f}, '
        f'{overlap}/{len(log_temps)} grid bins in range, input-DEM peak log T {peak_in_logT:.3f}')

    dlnT = np.log(10) * DEM_DLOGT

    dn_mxw_demreg = np.array([
        np.sum(tresp_grid[:, j] * dem * temps_K) * dlnT for j in range(6)])

    dn_kappa_synth = np.array([
        np.sum(fm['tresp_synth_kappa'][:, j] * dem * temps_K) * dlnT for j in range(6)])
    dn_mxw_synth = np.array([
        np.sum(fm['tresp_synth_Mxw'][:, j] * dem * temps_K) * dlnT for j in range(6)])
    chan_ratio_kappa = np.where(dn_mxw_synth != 0, dn_kappa_synth / dn_mxw_synth, 0.0)
    dn_kappa_for_inversion = dn_mxw_demreg * chan_ratio_kappa

    results = {}
    for fam, dn_obs in (('mxw', dn_mxw_demreg), ('kappa', dn_kappa_for_inversion)):
        edn = compute_aia_noise(dn_obs)
        dem_r, edem, elogt, chisq, dn_reg = dn2dem(
            dn_obs[None, :], edn[None, :], tresp_grid, log_temps, temps_K)
        dem_r = np.asarray(dem_r).squeeze()
        chisq = float(np.asarray(chisq).squeeze())

        if len(dem_r) == len(log_temps) - 1:
            log_T_centers = (log_temps[:-1] + log_temps[1:]) / 2
        else:
            log_T_centers = log_temps[:len(dem_r)]

        f = fwhm(log_T_centers, dem_r)
        peak_logT = float(log_T_centers[int(np.argmax(dem_r))])
        chi2dof = chisq / 5.0
        results[fam] = dict(fwhm=round(float(f), 4),
                            peak_logT=round(peak_logT, 4),
                            chi2dof=round(chi2dof, 4),
                            chisq_raw=round(chisq, 4))
        log(f'    [{fam:>5s}] FWHM = {f:.3f}, peak log T = {peak_logT:.3f}, '
            f'chi2/dof = {chi2dof:.3f}')

    results['input_peak_logT'] = round(float(peak_in_logT), 4)
    results['grid_bins_in_range'] = overlap
    return results


def in_band(v):
    return (not np.isnan(v)) and (QS_BAND[0] <= v <= QS_BAND[1])


def main():
    txt_path = OUT_DIR / 'refdem_results.txt'
    json_path = OUT_DIR / 'refdem_results.json'
    lf = open(txt_path, 'w', encoding='utf-8')

    def log(*a):
        msg = ' '.join(str(x) for x in a)
        print(msg)
        lf.write(msg + '\n')
        lf.flush()

    log('=' * 74)
    log('Reference-DEM robustness runner (R4.4)')
    log('Same pipeline as multi_thermal_kappa_test.py, arbitrary reference DEM')
    log('=' * 74)
    log(f'Real-QS FWHM band: {QS_BAND[0]}-{QS_BAND[1]} (median {QS_MEDIAN})')

    log('\nBuilding DEM-independent forward model (response + factorized ion responses)...')
    fm = build_forward_model(log)

    dem_dir = DATA_DIR / 'CHIANTI_11.0.2_database' / 'dem'

    jobs = []
    jobs.append(('Brooks2009_quiet_sun_eis', dem_dir / 'quiet_sun_eis.dem', 'GATE'))
    jobs.append(('CHIANTI_quiet_sun',        dem_dir / 'quiet_sun.dem',     'reference'))
    dropped = sorted(INPUTS_DIR.glob('*.dem'))
    for p in dropped:
        jobs.append((p.stem, p, 'user'))

    all_results = {}
    gate_pass = None
    gate_mxw = gate_kappa = None

    for label, path, role in jobs:
        log('\n' + '-' * 74)
        log(f'[{role}] {label}')
        log(f'  path: {path}')
        if not Path(path).exists():
            log(f'  SKIP: file not found')
            all_results[label] = {'role': role, 'error': 'file not found',
                                  'path': str(path)}
            continue
        try:
            r = run_one_dem(label, path, fm, log)
        except Exception as e:
            log(f'  ERROR: {e}')
            all_results[label] = {'role': role, 'error': str(e), 'path': str(path)}
            continue
        r['role'] = role
        r['path'] = str(path)
        r['mxw_in_band'] = bool(in_band(r['mxw']['fwhm']))
        r['kappa_in_band'] = bool(in_band(r['kappa']['fwhm']))
        all_results[label] = r

        if role == 'GATE':
            gate_mxw = r['mxw']['fwhm']
            gate_kappa = r['kappa']['fwhm']
            gate_pass = (abs(gate_mxw - 0.319) <= 0.01) and (abs(gate_kappa - 0.305) <= 0.01)
            log(f'  GATE check: Mxw {gate_mxw:.3f} (target 0.319+-0.01), '
                f'kappa {gate_kappa:.3f} (target 0.305+-0.01) -> '
                f'{"PASS" if gate_pass else "FAIL"}')

    log('\n' + '=' * 74)
    log('SUMMARY TABLE')
    log('=' * 74)
    hdr = (f"{'reference DEM':<30s} {'role':<10s} "
           f"{'Mxw_FWHM':>9s} {'Mxw_pk':>7s} {'Mxw_x2':>7s} "
           f"{'kap_FWHM':>9s} {'kap_pk':>7s} {'kap_x2':>7s} {'both_in_band':>13s}")
    log(hdr)
    log('-' * len(hdr))
    for label, r in all_results.items():
        if 'error' in r:
            log(f"{label:<30s} {r.get('role',''):<10s}  ERROR: {r['error']}")
            continue
        both = r['mxw_in_band'] and r['kappa_in_band']
        log(f"{label:<30s} {r['role']:<10s} "
            f"{r['mxw']['fwhm']:>9.3f} {r['mxw']['peak_logT']:>7.3f} {r['mxw']['chi2dof']:>7.3f} "
            f"{r['kappa']['fwhm']:>9.3f} {r['kappa']['peak_logT']:>7.3f} {r['kappa']['chi2dof']:>7.3f} "
            f"{('YES' if both else 'NO'):>13s}")

    log('\n' + '=' * 74)
    log('CONCLUSION')
    log('=' * 74)
    if gate_pass is not None:
        log(f'Brooks validation gate: {"PASS" if gate_pass else "FAIL"} '
            f'(Mxw {gate_mxw:.3f} / target 0.319; kappa {gate_kappa:.3f} / target 0.305)')
    refs_in_band = [lab for lab, r in all_results.items()
                    if 'error' not in r and r['mxw_in_band'] and r['kappa_in_band']]
    refs_total = [lab for lab, r in all_results.items() if 'error' not in r]
    if len(refs_in_band) == len(refs_total) and refs_total:
        log(f'Both source families land in the real-QS band {QS_BAND} for ALL '
            f'{len(refs_total)} reference DEMs -> degeneracy robust to reference choice.')
    else:
        out = [lab for lab in refs_total if lab not in refs_in_band]
        log(f'Both-in-band for {len(refs_in_band)}/{len(refs_total)} references; '
            f'outside band: {out}')

    meta = dict(
        description='Reference-DEM robustness (R4.4); same pipeline as '
                    'multi_thermal_kappa_test.py',
        qs_band=list(QS_BAND), qs_median=QS_MEDIAN,
        gate_pass=bool(gate_pass) if gate_pass is not None else None,
        gate_mxw_fwhm=gate_mxw, gate_kappa_fwhm=gate_kappa,
        dem_grid='10**arange(5.7,7.2+0.05,0.05)',
        inputs_dir=str(INPUTS_DIR),
        n_user_dem_processed=len(dropped),
        results=all_results,
    )
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(meta, jf, indent=2)
    log(f'\nWrote {txt_path}')
    log(f'Wrote {json_path}')
    log(f'User-DEM drop folder (auto-processed): {INPUTS_DIR}  '
        f'(found {len(dropped)} *.dem this run)')

    lf.close()
    return meta


if __name__ == '__main__':
    main()
