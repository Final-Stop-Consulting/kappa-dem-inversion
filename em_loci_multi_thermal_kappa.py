#!/usr/bin/env python3
"""
em_loci_multi_thermal_kappa.py — EM-loci collapse under a multi-thermal kappa source (paper Section 3.6).

Section 3.6 of the paper reports a 1.06 dex monotonic EM-loci spread across
Fe IX-XVI when forcing a single-T kappa = 2.5 source to reproduce the
Brooks 2009 predicted radiances. Each ion is sampled at the same T_eff far
from its formation temperature; the spread combines the kappa ion-fraction
reweighting with the multi-thermal radiance the Brooks DEM actually represents.

A realistic multi-thermal kappa source (the one constructed in
`multi_thermal_kappa_test.py` for the FWHM = 0.305 compute) should collapse
this spread substantially, because each ion now samples its own formation
temperature where the kappa/Maxwellian ion-fraction ratio is close to unity.

This script computes the collapse directly. Using the same factorization
approximation Section 3.4 already uses (line emissivity tracks ion fraction
near formation T):

    R_kappa_multi(line_i) / R_Brooks(line_i)
       ~ <f_kappa(ion_i)>_DEM / <f_Mxw(ion_i)>_DEM

where <.>_DEM is the Brooks-DEM-weighted ion-fraction average. The EM that the
multi-T kappa source needs to reproduce R_Brooks(line_i) scales as the
reciprocal of this ratio; the log-spread across the 8 lines is the empirical
collapse statistic.

Result reported in paper Section 3.6:
    Single-T tilt (Table 6, tab:emrec):  1.06 dex
    Multi-T kappa residual (this script): 0.14 dex
    Ratio:                                14% of single-T amplitude

Dependencies: numpy. Reads kappa and Maxwellian ion-fraction tables from
Data/kappa_v10.1/ and the Brooks 2009 reference DEM from
Data/CHIANTI_11.0.2_database/dem/quiet_sun_eis.dem. Does NOT require
ChiantiPy, demregpy, or per-line emissivity computations.

Usage:
    python em_loci_multi_thermal_kappa.py
"""
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / 'Data'
KAPPA_DIR = DATA_DIR / 'kappa_v10.1'
RESULTS_DIR = SCRIPT_DIR / 'Results'
FIGURES_DIR = SCRIPT_DIR / 'Figures'
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

# Paper Table 6 lines (Fe IX-XVI EUV coronal lines)
LINES = [
    ('Fe IX',   26,  9, 171.07),
    ('Fe X',    26, 10, 184.54),
    ('Fe XI',   26, 11, 188.22),
    ('Fe XII',  26, 12, 195.12),
    ('Fe XIII', 26, 13, 202.04),
    ('Fe XIV',  26, 14, 264.79),
    ('Fe XV',   26, 15, 284.16),
    ('Fe XVI',  26, 16, 335.41),
]

# Paper Table 6 single-T (kappa=2.5 at T_eff=1.5 MK) reference values for reporting
PAPER_TABLE6_LOG_EM = {
    'Fe IX':   26.93,
    'Fe X':    26.58,
    'Fe XI':   26.35,
    'Fe XII':  26.19,
    'Fe XIII': 26.02,
    'Fe XIV':  25.94,
    'Fe XV':   25.93,
    'Fe XVI':  25.87,
}
PAPER_FK_FM = {
    'Fe IX':   3.10, 'Fe X':    1.73, 'Fe XI':   1.04, 'Fe XII':  0.71,
    'Fe XIII': 0.63, 'Fe XIV':  0.62, 'Fe XV':   0.71, 'Fe XVI':  1.40,
}

LOG_T_EFF = 6.176


def parse_ioneq(path):
    """Parse CHIANTI .ioneq file. Returns (logT array, ions[Z][stage] = fractions)."""
    with open(path) as f:
        lines = f.readlines()
    parts = lines[0].split()
    n_t = int(parts[0])
    temps = []
    line_idx = 1
    while len(temps) < n_t and line_idx < len(lines):
        for tk in lines[line_idx].split():
            try:
                temps.append(float(tk))
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


def load_brooks_dem(path):
    """Read CHIANTI .dem file. Returns (logT, log10_DEM) on the file's grid."""
    logT, logDEM = [], []
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
                    logT.append(float(parts[0]))
                    logDEM.append(float(parts[1]))
                except ValueError:
                    continue
    return np.array(logT), np.array(logDEM)


def main():
    log_path = RESULTS_DIR / 'em_loci_multi_thermal_kappa_results.txt'
    log_f = open(log_path, 'w', encoding='utf-8')

    def log(*a):
        msg = ' '.join(str(x) for x in a)
        print(msg)
        log_f.write(msg + '\n')
        log_f.flush()

    log('=' * 72)
    log('Multi-thermal kappa EM-loci collapse (paper Section 3.6)')
    log('=' * 72)

    kappa_path = KAPPA_DIR / 'Dz23_kappa_2p5.ioneq'
    mxw_path = KAPPA_DIR / 'Dz23_mxw.ioneq'
    brooks_path = DATA_DIR / 'CHIANTI_11.0.2_database' / 'dem' / 'quiet_sun_eis.dem'

    for p in (kappa_path, mxw_path, brooks_path):
        if not p.exists():
            log(f'\nERROR: required data file not found:\n  {p}')
            log('See README.md for the Data/ layout (CHIANTI v11 atomic database + Dz23 kappa tables).')
            log_f.close()
            return

    log('\nLoading Dz23 kappa = 2.5 and Maxwellian ion fractions...')
    klog, k_ions = parse_ioneq(kappa_path)
    mlog, m_ions = parse_ioneq(mxw_path)
    assert np.allclose(klog, mlog), 'kappa and Mxw ioneq files have different temperature grids'
    log(f'  ioneq logT range: {klog.min():.2f}-{klog.max():.2f}, n = {len(klog)}')

    log('\nLoading Brooks 2009 quiet-Sun DEM...')
    bT, blogDEM = load_brooks_dem(brooks_path)
    log(f'  Brooks DEM: {len(bT)} knots, log T = {bT.min():.2f}-{bT.max():.2f}')

    # interpolate Brooks DEM onto the ioneq logT grid (zero outside Brooks range)
    DEM_on_ioneq = np.zeros_like(klog)
    in_range = (klog >= bT.min()) & (klog <= bT.max())
    DEM_on_ioneq[in_range] = 10**np.interp(klog[in_range], bT, blogDEM)
    dlogT = np.gradient(klog)
    weight = DEM_on_ioneq * dlogT
    weight_sum = weight.sum()

    log(f'  DEM-weighted integral: {weight_sum:.3e} (unit-less in this normalization)')

    log('\n' + '-' * 72)
    log('Per-line analysis:')
    log('-' * 72)
    log(f'{"Line":>8s} {"<f_k>_DEM":>12s} {"<f_M>_DEM":>12s} '
        f'{"M/k ratio":>10s} {"log(M/k)":>10s} '
        f'{"f_k(Teff)":>11s} {"f_M(Teff)":>11s} '
        f'{"M/k @Teff":>10s}')

    ratio_multi_arr, ratio_singleT_arr = [], []
    line_labels = []

    for (label, z, stage, lam) in LINES:
        fk_T = k_ions[z][stage]
        fM_T = m_ions[z][stage]

        # DEM-weighted ion fraction (multi-T source)
        fk_DEM = np.sum(weight * fk_T) / weight_sum
        fM_DEM = np.sum(weight * fM_T) / weight_sum
        ratio_multi = fM_DEM / fk_DEM if fk_DEM > 0 else float('nan')

        # Single-T (T_eff) reference
        fk_Teff = float(np.interp(LOG_T_EFF, klog, fk_T))
        fM_Teff = float(np.interp(LOG_T_EFF, mlog, fM_T))
        ratio_singleT = fM_Teff / fk_Teff if fk_Teff > 0 else float('nan')

        ratio_multi_arr.append(ratio_multi)
        ratio_singleT_arr.append(ratio_singleT)
        line_labels.append(label)

        log(f'{label:>8s} {fk_DEM:>12.4e} {fM_DEM:>12.4e} '
            f'{ratio_multi:>10.3f} {np.log10(ratio_multi):>10.3f} '
            f'{fk_Teff:>11.4e} {fM_Teff:>11.4e} '
            f'{ratio_singleT:>10.3f}')

    log_ratio_multi = np.log10(np.array(ratio_multi_arr))
    log_ratio_singleT = np.log10(np.array(ratio_singleT_arr))

    log('\n' + '-' * 72)
    log('SPREAD COMPARISON (log10 across the 8 lines)')
    log('-' * 72)
    log(f'  Single-T at T_eff:        '
        f'log(M/k) range = [{log_ratio_singleT.min():.3f}, {log_ratio_singleT.max():.3f}]; '
        f'spread = {log_ratio_singleT.max() - log_ratio_singleT.min():.3f} dex')
    log(f'  Multi-T (DEM-weighted):   '
        f'log(M/k) range = [{log_ratio_multi.min():.3f}, {log_ratio_multi.max():.3f}]; '
        f'spread = {log_ratio_multi.max() - log_ratio_multi.min():.3f} dex')
    log(f'  (single-T value above is the f_M/f_k ratio spread, NOT the paper')
    log(f'   Table 6 EM-loci tilt of 1.060 dex; the 14% collapse statistic')
    log(f'   below is computed against the Table 6 amplitude, not this value)')

    # Sanity check: single-T M/k at T_eff against paper Table 6 f_k/f_M values
    log('\nSingle-T ratio M/k consistency check against paper Table 6 f_k/f_M:')
    log(f'{"Line":>8s} {"this M/k(Teff)":>16s} {"paper f_k/f_M":>15s} {"-> M/k":>10s}')
    for i, (label, _, _, _) in enumerate(LINES):
        paper_Mk = 1.0 / PAPER_FK_FM[label]
        log(f'{label:>8s} {ratio_singleT_arr[i]:>16.3f} {PAPER_FK_FM[label]:>15.3f} '
            f'{paper_Mk:>10.3f}')

    # Paper single-T spread (Table 6 v2 corrected, range 25.87-26.93)
    paper_logEM = np.array([PAPER_TABLE6_LOG_EM[l[0]] for l in LINES])
    paper_spread = paper_logEM.max() - paper_logEM.min()
    multi_spread = log_ratio_multi.max() - log_ratio_multi.min()

    log(f'\nPaper Table 6 single-T spread: {paper_spread:.3f} dex '
        f'(across {paper_logEM.min():.2f}-{paper_logEM.max():.2f})')
    log(f'Multi-T compute spread:        {multi_spread:.3f} dex')
    log(f'Multi-T / single-T spread:     {multi_spread/paper_spread*100:.1f}%')
    log(f'(Paper Section 3.6 reports 14% based on the v2-corrected single-T amplitude)')

    # ---- Sensitivity test (prior-audit Issue 9): Fe XVI proxy edge case.
    # Fe XVI 335.41 forms at log T ~6.4, where the Brooks DEM has minimal
    # weight (Brooks peaks at log T = 6.05). The DEM-weighted ion-fraction
    # proxy is least reliable at the line set's hot endpoint. Drop Fe XVI
    # and re-compute the multi-T spread to check whether the 14% headline
    # statistic is dominated by the proxy's weakest point.
    log('\n' + '-' * 72)
    log('Fe XVI exclusion sensitivity (proxy edge-case check)')
    log('-' * 72)
    mask_no_fe16 = np.array([l[0] != 'Fe XVI' for l in LINES])
    log_ratio_multi_no_fe16 = log_ratio_multi[mask_no_fe16]
    multi_spread_no_fe16 = (log_ratio_multi_no_fe16.max()
                             - log_ratio_multi_no_fe16.min())
    delta = abs(multi_spread_no_fe16 - multi_spread)
    log(f'  Multi-T spread (all 8 lines):       {multi_spread:.3f} dex')
    log(f'  Multi-T spread (Fe IX-XV, no XVI):  {multi_spread_no_fe16:.3f} dex')
    log(f'  Change from dropping Fe XVI:        {delta:.3f} dex')
    if delta < 0.02:
        log(f'  -> headline result is robust (delta < 0.02 dex)')
    else:
        log(f'  -> WARNING: Fe XVI dominates the spread (delta >= 0.02 dex)')

    np.savez(RESULTS_DIR / 'em_loci_multi_thermal_kappa_results.npz',
             line_labels=np.array(line_labels),
             log_ratio_multi=log_ratio_multi,
             log_ratio_singleT=log_ratio_singleT,
             paper_logEM=paper_logEM,
             ratio_multi=np.array(ratio_multi_arr),
             ratio_singleT=np.array(ratio_singleT_arr))
    log(f'\nNumerical output saved to {RESULTS_DIR / "em_loci_multi_thermal_kappa_results.npz"}')

    # ---- optional figure
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        x = np.arange(len(line_labels))
        multi_anchored = log_ratio_multi - np.median(log_ratio_multi)
        paper_anchored = paper_logEM - np.median(paper_logEM)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        ax1.plot(x, paper_anchored, 'rs-', lw=2, ms=8,
                 label=f'Single-T (paper Table 6, spread {paper_spread:.2f} dex)')
        ax1.plot(x, multi_anchored, 'g^-', lw=2.5, ms=10,
                 label=f'Multi-T kappa (this compute, spread {multi_spread:.2f} dex)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(line_labels, rotation=30)
        ax1.set_ylabel('log10 EM (anchored at median)')
        ax1.set_title('EM-loci spread: single-T tilt vs multi-T collapse')
        ax1.axhline(0, color='gray', alpha=0.4, lw=0.5)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(alpha=0.3)
        ax1.set_ylim(-0.7, 0.7)

        bars = ax2.bar(['Single-T\n(paper Table 6)', 'Multi-T kappa\n(this compute)'],
                       [paper_spread, multi_spread],
                       color=['red', 'green'], alpha=0.7, edgecolor='black')
        for bar, val in zip(bars, [paper_spread, multi_spread]):
            ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f} dex',
                     ha='center', fontsize=10, fontweight='bold')
        ax2.set_ylabel('log10 EM spread (dex)')
        ax2.set_title('Spread comparison')
        ax2.axhline(0.2, color='gray', linestyle='--', alpha=0.5,
                    label='~0.2 dex (typical multi-T QS EIS scatter)')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(alpha=0.3, axis='y')
        ax2.set_ylim(0, max(paper_spread, multi_spread) * 1.25)

        plt.suptitle('Multi-thermal kappa EM-loci collapse (paper Section 3.6)', y=1.02)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'em_loci_multi_thermal_kappa.png', dpi=130, bbox_inches='tight')
        plt.savefig(FIGURES_DIR / 'em_loci_multi_thermal_kappa.pdf', bbox_inches='tight')
        log(f'Figure saved to {FIGURES_DIR / "em_loci_multi_thermal_kappa.{png,pdf}"}')
    except ImportError:
        log('\nmatplotlib not available; skipping figure generation.')

    log_f.close()


if __name__ == '__main__':
    main()
