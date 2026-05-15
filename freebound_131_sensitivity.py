#!/usr/bin/env python3
"""
freebound_131_sensitivity.py — 131 A DN sensitivity test (paper Section 2.3).

The channel-integrated free-bound treatment in `kappa_dem_pipeline.py` is known
to under-estimate the kappa free-bound enhancement in the 131 A channel, where
low-charge Fe ions (Fe VIII-X) have per-ion ion-fraction ratios of 3-7x
against a channel-integrated free-free ratio of ~1.0. A fully resolved per-ion
free-bound treatment under kappa requires KAPPA-native recombination
cross-sections that are not in the public ChiantiPy / Dz23 distribution.

This script asks the empirical version of the question: how sensitive is the
recovered DEM shape to inflating the 131 A DN? We multiply the saved
single-T kappa = 2.5 131 A DN by 2x and 3x — generous upper bounds on what a
fully resolved per-ion free-bound treatment could plausibly add — and re-run
demregpy with the same temperature response and shot+read noise model.

Result reported in paper Section 2.3 (under the dn2ph-corrected noise model):
    - Peak log T unchanged at 6.150 across nominal, 2x, 3x
    - FWHM stable to within +/- 0.005 dex
    - No secondary artifact spike at the 131 A peak formation temperature
    - chi^2/dof shifts by less than 0.1 across the 3x inflation (lines-only)
    - 131 A per-channel recovery drops ~96% -> ~47% as the pipeline absorbs the
      inflated DN by reducing per-channel recovery rather than warping the DEM

Dependencies: numpy<2.0, scipy, demregpy. Reads the kappa=2.5 DN vector from
Results/kappa_dem_inversion_results.npz (produced by kappa_dem_pipeline.py).
Does NOT require ChiantiPy or the CHIANTI database.

Usage:
    python freebound_131_sensitivity.py
"""
import os
import time
import numpy as np
from pathlib import Path

if 'HOME' not in os.environ:
    os.environ['HOME'] = os.path.expanduser('~')

import demregpy
from demregpy import dn2dem

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / 'Results'
FIGURES_DIR = SCRIPT_DIR / 'Figures'
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

AIA_CHANNELS = [94, 131, 171, 193, 211, 335]
CH_131_IDX = AIA_CHANNELS.index(131)

DEM_LOG_T_MIN = 5.7
DEM_LOG_T_MAX = 7.2
DEM_DLOGT = 0.05


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


def interp_edge(x0, x1, y0, y1, y):
    if y1 == y0:
        return (x0 + x1) / 2
    return x0 + (y - y0) / (y1 - y0) * (x1 - x0)


def compute_fwhm(log_T_centers, dem):
    """FWHM in log T via linear half-max interpolation."""
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


def run_inversion(dn, edn, tresp_grid, log_temps, temps_K):
    dem, edem, elogt, chisq, dn_reg = dn2dem(
        dn[None, :], edn[None, :], tresp_grid, log_temps, temps_K
    )
    return (
        np.asarray(dem).squeeze(),
        np.asarray(edem).squeeze(),
        np.asarray(elogt).squeeze(),
        float(np.asarray(chisq).squeeze()),
        np.asarray(dn_reg).squeeze(),
    )


def shape_report(log_f, label, dem, log_T_centers, chisq, dn, dn_reg):
    """Write a shape-stability report for one inversion case."""
    peak_idx = int(np.argmax(dem))
    peak_logT = log_T_centers[peak_idx]
    fwhm = compute_fwhm(log_T_centers, dem)
    # check for secondary local max above log T 6.5 (potential artifact spike)
    tail_mask = log_T_centers > 6.5
    secondary_str = 'none above log T = 6.5'
    if tail_mask.any():
        tail_dem = dem.copy()
        tail_dem[~tail_mask] = 0
        if tail_dem.max() > dem.max() * 0.05:
            sec_idx = int(np.argmax(tail_dem))
            sec_logT = log_T_centers[sec_idx]
            sec_rel = dem[sec_idx] / dem.max()
            secondary_str = f'log T {sec_logT:.3f}, rel. amplitude {sec_rel:.3f}'

    def log(*a):
        msg = ' '.join(str(x) for x in a)
        print(msg)
        log_f.write(msg + '\n')

    log(f'\n--- {label} ---')
    log(f'  chi^2 (raw)  = {chisq:.3f}')
    log(f'  chi^2/dof    = {chisq/5:.3f}')
    log(f'  peak log T   = {peak_logT:.3f}  (T = {10**peak_logT/1e6:.2f} MK)')
    log(f'  FWHM         = {fwhm:.3f}')
    log(f'  secondary    = {secondary_str}')
    log(f'  per-channel recovery (DN_reg / DN):')
    log(f'    {"Chan":>5} {"DN":>14} {"DN_reg":>14} {"recovery":>10}')
    for j, c in enumerate(AIA_CHANNELS):
        rec = dn_reg[j] / dn[j] * 100
        log(f'    {c:>5d} {dn[j]:>14.4e} {dn_reg[j]:>14.4e} {rec:>9.1f}%')


def main():
    log_path = RESULTS_DIR / 'freebound_131_sensitivity_results.txt'
    log_f = open(log_path, 'w', encoding='utf-8')

    def log(*a):
        msg = ' '.join(str(x) for x in a)
        print(msg)
        log_f.write(msg + '\n')
        log_f.flush()

    log('=' * 72)
    log('131 A DN sensitivity test (paper Section 2.3)')
    log('=' * 72)

    # ---- load saved nominal kappa=2.5 lines-only results
    nom_file = RESULTS_DIR / 'kappa_dem_inversion_results.npz'
    if not nom_file.exists():
        log(f'\nERROR: {nom_file} not found.')
        log('Run kappa_dem_pipeline.py first to generate the nominal kappa=2.5 inversion.')
        log_f.close()
        return

    nom = np.load(nom_file, allow_pickle=True)
    log(f'\nLoaded nominal results from {nom_file.name}')
    log(f'  kappa = {nom["kappa"]}, T_eff = {nom["T_eff"]/1e6:.2f} MK, '
        f'T_core = {nom["T_core"]/1e6:.2f} MK')
    log(f'  nominal saved chi^2/dof = {nom["chisq"]/5:.4f}')

    # ---- reconstruct demregpy invocation
    log_temps = np.arange(DEM_LOG_T_MIN, DEM_LOG_T_MAX + DEM_DLOGT/2, DEM_DLOGT)
    temps_K = 10**log_temps

    tresp_logT_full = np.asarray(nom['tresp_logT'])
    tresp_full = np.asarray(nom['tresp_matrix'])  # (101, 6)
    tresp_grid = np.zeros((len(log_temps), 6))
    for j in range(6):
        tresp_grid[:, j] = np.interp(log_temps, tresp_logT_full, tresp_full[:, j])

    # ---- nominal inversion
    dn_nom = np.asarray(nom['dn_kappa']).copy()
    edn_nom = compute_aia_noise(dn_nom)
    log(f'\nReplay nominal inversion with shot+read noise...')
    dem_nom, edem_nom, elogt_nom, chisq_nom, dn_reg_nom = run_inversion(
        dn_nom, edn_nom, tresp_grid, log_temps, temps_K
    )
    log_T_centers = log_temps[:len(dem_nom)]
    shape_report(log_f, 'NOMINAL kappa=2.5 (lines-only)',
                 dem_nom, log_T_centers, chisq_nom, dn_nom, dn_reg_nom)

    # ---- 2x 131 A
    dn_2x = dn_nom.copy()
    dn_2x[CH_131_IDX] *= 2.0
    edn_2x = compute_aia_noise(dn_2x)
    dem_2x, edem_2x, elogt_2x, chisq_2x, dn_reg_2x = run_inversion(
        dn_2x, edn_2x, tresp_grid, log_temps, temps_K
    )
    shape_report(log_f, '131 A x 2 (intermediate)',
                 dem_2x, log_T_centers, chisq_2x, dn_2x, dn_reg_2x)

    # ---- 3x 131 A
    dn_3x = dn_nom.copy()
    dn_3x[CH_131_IDX] *= 3.0
    edn_3x = compute_aia_noise(dn_3x)
    log(f'\n  131 A DN: {dn_nom[CH_131_IDX]:.4e} --> {dn_3x[CH_131_IDX]:.4e} (3x)')

    dem_3x, edem_3x, elogt_3x, chisq_3x, dn_reg_3x = run_inversion(
        dn_3x, edn_3x, tresp_grid, log_temps, temps_K
    )
    shape_report(log_f, '131 A x 3 (per-ion free-bound proxy)',
                 dem_3x, log_T_centers, chisq_3x, dn_3x, dn_reg_3x)

    # ---- summary
    log('\n' + '=' * 72)
    log('SUMMARY')
    log('=' * 72)
    log(f'{"Case":>20} {"chi2/dof":>10} {"peak logT":>11} {"FWHM":>8} {"131 rec":>9}')
    for label, dem, chisq, dn, dn_reg in [
        ('NOMINAL', dem_nom, chisq_nom, dn_nom, dn_reg_nom),
        ('131 x 2', dem_2x, chisq_2x, dn_2x, dn_reg_2x),
        ('131 x 3', dem_3x, chisq_3x, dn_3x, dn_reg_3x),
    ]:
        peak_logT = log_T_centers[int(np.argmax(dem))]
        fwhm = compute_fwhm(log_T_centers, dem)
        rec131 = dn_reg[CH_131_IDX] / dn[CH_131_IDX] * 100
        log(f'{label:>20} {chisq/5:>10.3f} {peak_logT:>11.3f} {fwhm:>8.3f} {rec131:>8.1f}%')

    # ---- save numerical output
    np.savez(RESULTS_DIR / 'freebound_131_sensitivity_results.npz',
             log_temps=log_temps,
             dn_nom=dn_nom, edn_nom=edn_nom, dem_nom=dem_nom,
             chisq_nom=chisq_nom, dn_reg_nom=dn_reg_nom,
             dn_2x=dn_2x, edn_2x=edn_2x, dem_2x=dem_2x,
             chisq_2x=chisq_2x, dn_reg_2x=dn_reg_2x,
             dn_3x=dn_3x, edn_3x=edn_3x, dem_3x=dem_3x,
             chisq_3x=chisq_3x, dn_reg_3x=dn_reg_3x)
    log(f'\nNumerical output saved to {RESULTS_DIR / "freebound_131_sensitivity_results.npz"}')

    # ---- optional figure (only if matplotlib is available)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(log_T_centers, dem_nom, 'k-',  lw=2.5, label='nominal kappa=2.5')
        ax1.plot(log_T_centers, dem_2x,  'b--', lw=2.0, label='131 A x 2')
        ax1.plot(log_T_centers, dem_3x,  'r:',  lw=2.0, label='131 A x 3')
        ax1.set_xlabel('log T (K)')
        ax1.set_ylabel(r'DEM (cm$^{-5}$ K$^{-1}$)')
        ax1.set_title('Recovered DEM (linear scale)')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(alpha=0.3)
        ax1.set_xlim(5.7, 7.2)

        ax2.semilogy(log_T_centers, np.maximum(dem_nom, 1e15), 'k-',  lw=2.5, label='nominal')
        ax2.semilogy(log_T_centers, np.maximum(dem_2x, 1e15),  'b--', lw=2.0, label='131 A x 2')
        ax2.semilogy(log_T_centers, np.maximum(dem_3x, 1e15),  'r:',  lw=2.0, label='131 A x 3')
        ax2.axvline(5.95, color='c', alpha=0.3, ls=':', label='131 A peak formation T')
        ax2.set_xlabel('log T (K)')
        ax2.set_ylabel(r'DEM (cm$^{-5}$ K$^{-1}$)')
        ax2.set_title('Recovered DEM (log scale, artifact check)')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(alpha=0.3)
        ax2.set_xlim(5.7, 7.2)
        ax2.set_ylim(1e17, 1e23)

        plt.suptitle('131 A DN sensitivity: DEM shape stability under free-bound inflation',
                     y=1.02)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'freebound_131_sensitivity.png', dpi=130, bbox_inches='tight')
        plt.savefig(FIGURES_DIR / 'freebound_131_sensitivity.pdf', bbox_inches='tight')
        log(f'Figure saved to {FIGURES_DIR / "freebound_131_sensitivity.{png,pdf}"}')
    except ImportError:
        log('\nmatplotlib not available; skipping figure generation.')

    log_f.close()


if __name__ == '__main__':
    main()
