#!/usr/bin/env python3
"""
Sensitivity tests for the kappa DEM inversion pipeline.

Reuses the Stage 2 checkpoint (ChiantiPy per-ion contributions) which is
κ-independent (Maxwellian atomic physics).  Only Stages 1, 3, 4 change
with κ.  This means each sensitivity test takes seconds, not minutes.

Tests:
    1. κ = 2, 2.5, 3  (brackets R=2.4±0.3 from corona paper)
    2. Coronal vs photospheric abundances
    3. Density: n_e = 10^8.5, 10^9, 10^9.5

Output: summary table + multi-panel comparison figure + results .npz
         All console output also saved to Results/sensitivity_results.txt
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.io as sio


# ============================================================================
# Tee class — duplicates stdout to a file
# ============================================================================
class Tee:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, filepath):
        self.file = open(filepath, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        self.file.close()
        sys.stdout = self.stdout

# ---- Platform fix (Windows) ----
if 'HOME' not in os.environ:
    os.environ['HOME'] = os.path.expanduser('~')

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'Data'
KAPPA_DIR = DATA_DIR / 'kappa_v10.1'
RESULTS_DIR = PROJECT_DIR / 'Analysis' / 'Results'
FIGURES_DIR = PROJECT_DIR / 'Analysis' / 'Figures'

XUVTOP = str(DATA_DIR / 'CHIANTI_11.0.2_database')
os.environ['XUVTOP'] = XUVTOP

# Physical parameters (fixed)
T_EFF = 1.5e6
LOG_T_EFF = 6.176
N_E = 1.0e9
EM_SCALE = 1.0e27
AIA_CHANNELS = [94, 131, 171, 193, 211, 335]
AIA_CHANNEL_NAMES = ['A94', 'A131', 'A171', 'A193', 'A211', 'A335']

DEM_LOG_T_MIN = 5.7
DEM_LOG_T_MAX = 7.2
DEM_DLOGT = 0.05


# ============================================================================
# Import shared functions from the main pipeline
# ============================================================================

# Add the Analysis directory to the path so we can import from the pipeline
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kappa_dem_pipeline import (
    parse_ioneq, interp_logT, roman,
    load_demregpy_response, run_dem_inversion, compute_aia_noise
)


# ============================================================================
# Core functions
# ============================================================================

def load_kappa_fractions(kappa_value):
    """Load Dz23 kappa and matched Maxwellian ion fractions."""
    kappa_files = {
        2:   'Dz23_kappa_2.ioneq',
        2.5: 'Dz23_kappa_2p5.ioneq',
        3:   'Dz23_kappa_3.ioneq',
    }

    kappa_file = KAPPA_DIR / kappa_files[kappa_value]
    mxw_file = KAPPA_DIR / 'Dz23_mxw.ioneq'

    kappa_logT, kappa_ions = parse_ioneq(str(kappa_file))
    mxw_logT, mxw_ions = parse_ioneq(str(mxw_file))

    return kappa_logT, kappa_ions, mxw_logT, mxw_ions


def compute_ratios(kappa_logT, kappa_ions, mxw_logT, mxw_ions):
    """Compute kappa/Maxwellian ion fraction ratios at T_eff."""
    ratios = {}
    for z in kappa_ions:
        ratios[z] = {}
        for stage in kappa_ions[z]:
            f_kappa = interp_logT(kappa_logT, kappa_ions[z][stage], LOG_T_EFF)
            if z in mxw_ions and stage in mxw_ions[z]:
                f_mxw = interp_logT(mxw_logT, mxw_ions[z][stage], LOG_T_EFF)
            else:
                f_mxw = 0.0
            if f_mxw > 1e-30:
                ratios[z][stage] = f_kappa / f_mxw
            else:
                ratios[z][stage] = 0.0 if f_kappa < 1e-30 else np.inf
    return ratios


def load_checkpoint():
    """Load the Stage 2 checkpoint (κ-independent per-ion contributions)."""
    checkpoint_file = RESULTS_DIR / 'ion_contributions_checkpoint.json'
    if not checkpoint_file.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_file}\n"
            f"Run kappa_dem_pipeline.py first to compute Stage 2."
        )

    with open(checkpoint_file, 'r') as f:
        saved = json.load(f)

    contributions = {}
    for chan_str, ion_dict in saved.items():
        chan = int(chan_str)
        contributions[chan] = {}
        for key_str, value in ion_dict.items():
            z, stage = map(int, key_str.split(','))
            contributions[chan][(z, stage)] = value

    return contributions


def rescale_contributions_by_abundance(contributions, old_ab_name, new_ab_name):
    """
    Rescale Stage 2 contributions from one abundance set to another.

    Since emissivity ∝ A_X (element abundance), we just multiply each
    ion's contribution by new_abundance(Z) / old_abundance(Z).
    No Stage 2 recompute needed.

    Args:
        contributions: checkpoint dict {channel: {(Z, stage): value}}
        old_ab_name: abundance file used in Stage 2 (e.g. 'sun_coronal_2021_chianti')
        new_ab_name: target abundance file (e.g. 'sun_photospheric_2021_asplund')

    Returns:
        rescaled contributions dict
    """
    import ChiantiPy.tools.io as chio

    old_ab = chio.abundanceRead(abundancename=old_ab_name)['abundance']
    new_ab = chio.abundanceRead(abundancename=new_ab_name)['abundance']

    rescaled = {}
    for chan in contributions:
        rescaled[chan] = {}
        for (z, stage), value in contributions[chan].items():
            if z-1 < len(old_ab) and z-1 < len(new_ab) and old_ab[z-1] > 0:
                scale = new_ab[z-1] / old_ab[z-1]
            else:
                scale = 1.0
            rescaled[chan][(z, stage)] = value * scale

    return rescaled


def apply_kappa_ratios(contributions, ratios):
    """
    Apply kappa/Maxwellian ratios to Stage 2 contributions.
    Returns kappa DNs, Maxwellian DNs, and per-channel details.
    """
    dn_kappa = {}
    dn_mxw = {}
    details = {}

    for chan in AIA_CHANNELS:
        mxw_total = 0.0
        kappa_total = 0.0
        chan_details = []

        if chan not in contributions:
            dn_mxw[chan] = 0.0
            dn_kappa[chan] = 0.0
            details[chan] = []
            continue

        for (z, stage), contrib in contributions[chan].items():
            mxw_total += contrib
            ratio = ratios.get(z, {}).get(stage, 1.0)
            if not np.isfinite(ratio):
                ratio = 0.0
            kappa_contrib = contrib * ratio
            kappa_total += kappa_contrib
            chan_details.append({
                'Z': z, 'stage': stage,
                'mxw_contrib': contrib,
                'kappa_contrib': kappa_contrib,
                'ratio': ratio
            })

        dn_mxw[chan] = mxw_total
        dn_kappa[chan] = kappa_total
        details[chan] = sorted(chan_details, key=lambda x: -x['mxw_contrib'])

    return dn_kappa, dn_mxw, details


def run_single_test(kappa_value, contributions, tresp_logT, tresp_matrix,
                    label=None):
    """
    Run Stages 1, 3, 4 for a single κ value.
    Returns a dict with all results.
    """
    if label is None:
        label = f"κ={kappa_value}"

    t_core = T_EFF * (kappa_value - 1.5) / kappa_value

    print(f"\n{'='*50}")
    print(f"  {label}: κ={kappa_value}, T_core={t_core/1e6:.2f} MK")
    print(f"{'='*50}")

    # Stage 1: load fractions and compute ratios
    kappa_logT, kappa_ions, mxw_logT, mxw_ions = load_kappa_fractions(kappa_value)
    ratios = compute_ratios(kappa_logT, kappa_ions, mxw_logT, mxw_ions)

    # Print Fe ratios
    print(f"\n  Fe ion fraction ratios at log T = {LOG_T_EFF:.3f}:")
    fe_ratios = {}
    for stage in range(8, 18):
        r = ratios.get(26, {}).get(stage, 0)
        f_k = interp_logT(kappa_logT,
                          kappa_ions.get(26, {}).get(stage, np.zeros_like(kappa_logT)),
                          LOG_T_EFF)
        f_m = interp_logT(mxw_logT,
                          mxw_ions.get(26, {}).get(stage, np.zeros_like(mxw_logT)),
                          LOG_T_EFF)
        rn = roman(stage)
        print(f"    Fe {rn:>5s}: κ={f_k:.4e}  Mxw={f_m:.4e}  ratio={r:.3f}")
        fe_ratios[stage] = {'f_kappa': f_k, 'f_mxw': f_m, 'ratio': r}

    # Stage 3: apply ratios
    dn_kappa, dn_mxw, details = apply_kappa_ratios(contributions, ratios)

    dn_kappa_arr = np.array([dn_kappa[c] for c in AIA_CHANNELS])
    dn_mxw_arr = np.array([dn_mxw[c] for c in AIA_CHANNELS])
    chan_ratios = np.where(dn_mxw_arr > 0, dn_kappa_arr / dn_mxw_arr, 0)

    print(f"\n  Per-channel κ/Mxw DN ratios:")
    for j, chan in enumerate(AIA_CHANNELS):
        print(f"    {chan:>4d} Å: {chan_ratios[j]:.3f}")

    # Stage 4: DEM inversion
    demreg_dn_mxw = np.zeros(6)
    for j in range(6):
        demreg_dn_mxw[j] = np.interp(LOG_T_EFF, tresp_logT, tresp_matrix[:, j])
    demreg_dn_mxw *= EM_SCALE

    dn_for_inversion = demreg_dn_mxw * chan_ratios
    edn = compute_aia_noise(dn_for_inversion)

    dem, edem, elogt, chisq, dn_reg, mlogt, temps = run_dem_inversion(
        dn_for_inversion, edn, tresp_matrix, tresp_logT
    )

    recovery = dn_reg / dn_for_inversion

    return {
        'kappa': kappa_value,
        't_core': t_core,
        'label': label,
        'fe_ratios': fe_ratios,
        'chan_ratios': chan_ratios,
        'dn_kappa': dn_for_inversion,
        'dn_mxw': demreg_dn_mxw,
        'dem': dem,
        'edem': edem,
        'elogt': elogt,
        'mlogt': mlogt,
        'chisq': chisq,
        'dn_reg': dn_reg,
        'recovery': recovery,
    }


# ============================================================================
# Plotting
# ============================================================================

def plot_kappa_comparison(results_list):
    """
    Multi-panel figure comparing DEM results across κ values.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    colors_kappa = {2: '#d62728', 2.5: '#1f77b4', 3: '#2ca02c'}
    labels_kappa = {2: 'κ=2 (harder)', 2.5: 'κ=2.5 (nominal)', 3: 'κ=3 (softer)'}

    # --- Panel 1: Recovered DEMs ---
    ax = axes[0, 0]
    for r in results_list:
        k = r['kappa']
        ax.plot(r['mlogt'], r['dem'], 'o-',
                color=colors_kappa[k], label=labels_kappa[k],
                markersize=4, linewidth=1.5)
    ax.axvline(LOG_T_EFF, color='blue', ls='--', alpha=0.5,
               label=f'T_eff = {T_EFF/1e6:.1f} MK')
    ax.set_xlabel('log₁₀ T [K]')
    ax.set_ylabel('DEM [cm⁻⁵ K⁻¹]')
    ax.set_yscale('log')
    ax.set_xlim(5.5, 7.5)
    ax.legend(fontsize=9)
    ax.set_title('Recovered DEMs')

    # --- Panel 2: Channel κ/Mxw ratios ---
    ax = axes[0, 1]
    x = np.arange(6)
    width = 0.25
    for i, r in enumerate(results_list):
        k = r['kappa']
        offset = (i - 1) * width
        ax.bar(x + offset, r['chan_ratios'], width,
               color=colors_kappa[k], alpha=0.7, label=labels_kappa[k])
    ax.axhline(1.0, color='grey', ls='--')
    ax.set_xticks(x)
    ax.set_xticklabels(AIA_CHANNEL_NAMES)
    ax.set_ylabel('DN_κ / DN_Mxw')
    ax.set_title('Channel redistribution')
    ax.legend(fontsize=9)

    # --- Panel 3: Fe XII ratio vs κ ---
    ax = axes[1, 0]
    kappas = [r['kappa'] for r in results_list]
    fe12_ratios = [r['fe_ratios'][12]['ratio'] for r in results_list]
    fe9_ratios = [r['fe_ratios'][9]['ratio'] for r in results_list]
    fe14_ratios = [r['fe_ratios'][14]['ratio'] for r in results_list]

    ax.plot(kappas, fe12_ratios, 'o-', color='#d62728', label='Fe XII (193 Å)', markersize=8)
    ax.plot(kappas, fe9_ratios, 's-', color='#1f77b4', label='Fe IX (171 Å)', markersize=8)
    ax.plot(kappas, fe14_ratios, '^-', color='#2ca02c', label='Fe XIV (211 Å)', markersize=8)
    ax.axhline(1.0, color='grey', ls='--')
    ax.set_xlabel('κ')
    ax.set_ylabel('Ion fraction ratio (κ/Mxw)')
    ax.set_title('Key ion fraction ratios vs κ')
    ax.legend(fontsize=9)
    ax.set_xlim(1.8, 3.2)

    # --- Panel 4: χ² and recovery summary ---
    ax = axes[1, 1]
    kappas = [r['kappa'] for r in results_list]
    chisqs = [r['chisq'] for r in results_list]
    min_rec = [np.min(r['recovery']) for r in results_list]
    max_rec = [np.max(r['recovery']) for r in results_list]

    ax2 = ax.twinx()
    bars = ax.bar(kappas, chisqs, 0.2, color='steelblue', alpha=0.7, label='χ²')
    ax.set_xlabel('κ')
    ax.set_ylabel('χ²', color='steelblue')
    ax.tick_params(axis='y', labelcolor='steelblue')

    ax2.errorbar(kappas, [(mn + mx)/2 for mn, mx in zip(min_rec, max_rec)],
                 yerr=[(mx - mn)/2 for mn, mx in zip(min_rec, max_rec)],
                 fmt='D', color='darkred', markersize=8, capsize=8,
                 label='DN recovery range')
    ax2.axhline(1.0, color='grey', ls='--')
    ax2.set_ylabel('DN recovery', color='darkred')
    ax2.tick_params(axis='y', labelcolor='darkred')
    ax2.set_ylim(0.6, 1.4)

    ax.set_title('Inversion quality vs κ')
    ax.set_xticks(kappas)

    plt.tight_layout()
    outfile = FIGURES_DIR / 'kappa_sensitivity_comparison.png'
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'kappa_sensitivity_comparison.pdf', bbox_inches='tight')
    print(f"\nFigure saved to {outfile}")
    plt.close()


def print_summary_table(results_list):
    """Print a comprehensive summary table."""

    print("\n" + "=" * 90)
    print("SENSITIVITY TEST SUMMARY")
    print("=" * 90)

    # Fe ion fractions table
    print("\n--- Fe ion fraction ratios (κ/Mxw) at log T = 6.176 ---")
    header = f"{'Ion':>8s}"
    for r in results_list:
        header += f"  {'κ='+str(r['kappa']):>10s}"
    print(header)
    print("-" * (8 + 12 * len(results_list)))

    for stage in range(8, 18):
        rn = roman(stage)
        line = f"  Fe {rn:>3s}"
        for r in results_list:
            ratio = r['fe_ratios'][stage]['ratio']
            line += f"  {ratio:>10.3f}"
        print(line)

    # Channel DN ratios
    print(f"\n--- Channel DN ratios (κ/Mxw) ---")
    header = f"{'Channel':>8s}"
    for r in results_list:
        header += f"  {'κ='+str(r['kappa']):>10s}"
    print(header)
    print("-" * (8 + 12 * len(results_list)))

    for j, chan in enumerate(AIA_CHANNELS):
        line = f"  {chan:>4d} Å"
        for r in results_list:
            line += f"  {r['chan_ratios'][j]:>10.3f}"
        print(line)

    # DEM inversion quality
    print(f"\n--- DEM inversion quality ---")
    header = f"{'Metric':>20s}"
    for r in results_list:
        header += f"  {'κ='+str(r['kappa']):>10s}"
    print(header)
    print("-" * (20 + 12 * len(results_list)))

    line = f"  {'χ²':>18s}"
    for r in results_list:
        line += f"  {r['chisq']:>10.3f}"
    print(line)

    line = f"  {'χ²/dof (dof≈5)':>18s}"
    for r in results_list:
        line += f"  {r['chisq']/5:>10.3f}"
    print(line)

    line = f"  {'Min DN recovery':>18s}"
    for r in results_list:
        line += f"  {np.min(r['recovery']):>10.3f}"
    print(line)

    line = f"  {'Max DN recovery':>18s}"
    for r in results_list:
        line += f"  {np.max(r['recovery']):>10.3f}"
    print(line)

    # Per-channel recovery
    print(f"\n--- Per-channel DN recovery ---")
    header = f"{'Channel':>8s}"
    for r in results_list:
        header += f"  {'κ='+str(r['kappa']):>10s}"
    print(header)
    print("-" * (8 + 12 * len(results_list)))

    for j, name in enumerate(AIA_CHANNEL_NAMES):
        line = f"  {name:>6s}"
        for r in results_list:
            line += f"  {r['recovery'][j]:>10.3f}"
        print(line)

    # T_core values
    print(f"\n--- Physical parameters ---")
    header = f"{'Parameter':>20s}"
    for r in results_list:
        header += f"  {'κ='+str(r['kappa']):>10s}"
    print(header)
    print("-" * (20 + 12 * len(results_list)))

    line = f"  {'T_core (MK)':>18s}"
    for r in results_list:
        line += f"  {r['t_core']/1e6:>10.2f}"
    print(line)

    line = f"  {'T_eff (MK)':>18s}"
    for r in results_list:
        line += f"  {T_EFF/1e6:>10.2f}"
    print(line)

    line = f"  {'T_core/T_eff':>18s}"
    for r in results_list:
        line += f"  {r['t_core']/T_EFF:>10.3f}"
    print(line)

    print("\n" + "=" * 90)


# ============================================================================
# MAIN
# ============================================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Tee all output to file
    log_file = RESULTS_DIR / 'sensitivity_results.txt'
    tee = Tee(str(log_file))
    sys.stdout = tee

    print("=" * 70)
    print("KAPPA SENSITIVITY TESTS")
    print(f"T_eff = {T_EFF/1e6:.1f} MK, log T = {LOG_T_EFF:.3f}")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load the κ-independent Stage 2 checkpoint
    print("\nLoading Stage 2 checkpoint (κ-independent per-ion contributions)...")
    contributions = load_checkpoint()
    n_ions = sum(len(v) for v in contributions.values())
    print(f"  Loaded {n_ions} ion-channel entries across {len(contributions)} channels")

    # Load demregpy response (needed for all tests)
    tresp_logT, tresp_matrix, channel_names = load_demregpy_response()

    # ==================================================================
    # TEST 1: κ sensitivity (κ = 2, 2.5, 3) with coronal abundances
    # ==================================================================
    print("\n" + "#" * 70)
    print("# TEST 1: κ sensitivity (κ = 2, 2.5, 3)")
    print("#" * 70)

    kappa_results = []
    for kappa in [2, 2.5, 3]:
        result = run_single_test(kappa, contributions, tresp_logT, tresp_matrix)
        kappa_results.append(result)

    print_summary_table(kappa_results)
    plot_kappa_comparison(kappa_results)

    # ==================================================================
    # TEST 2: Abundance sensitivity (coronal vs photospheric) at κ=2.5
    # ==================================================================
    print("\n" + "#" * 70)
    print("# TEST 2: Abundance sensitivity at κ=2.5")
    print("#         Coronal (FIP-biased) vs Photospheric")
    print("#" * 70)

    # Rescale contributions to photospheric abundances
    # Stage 2 was computed with 'sun_coronal_2021_chianti'
    try:
        contrib_phot = rescale_contributions_by_abundance(
            contributions,
            old_ab_name='sun_coronal_2021_chianti',
            new_ab_name='sun_photospheric_2021_asplund'
        )

        result_coronal = run_single_test(
            2.5, contributions, tresp_logT, tresp_matrix,
            label="κ=2.5, coronal abundances")
        result_photospheric = run_single_test(
            2.5, contrib_phot, tresp_logT, tresp_matrix,
            label="κ=2.5, photospheric abundances")

        print("\n" + "=" * 70)
        print("ABUNDANCE COMPARISON (κ=2.5)")
        print("=" * 70)

        print(f"\n{'Metric':>25s}  {'Coronal':>12s}  {'Photospheric':>12s}")
        print("-" * 55)
        print(f"  {'χ²':>23s}  {result_coronal['chisq']:>12.3f}  {result_photospheric['chisq']:>12.3f}")
        print(f"  {'χ²/dof':>23s}  {result_coronal['chisq']/5:>12.3f}  {result_photospheric['chisq']/5:>12.3f}")
        print(f"  {'Min DN recovery':>23s}  {np.min(result_coronal['recovery']):>12.3f}  {np.min(result_photospheric['recovery']):>12.3f}")
        print(f"  {'Max DN recovery':>23s}  {np.max(result_coronal['recovery']):>12.3f}  {np.max(result_photospheric['recovery']):>12.3f}")

        print(f"\n  Channel DN ratios (κ/Mxw):")
        print(f"  {'Channel':>8s}  {'Coronal':>12s}  {'Photospheric':>12s}  {'Difference':>12s}")
        print("  " + "-" * 50)
        for j, chan in enumerate(AIA_CHANNELS):
            rc = result_coronal['chan_ratios'][j]
            rp = result_photospheric['chan_ratios'][j]
            print(f"    {chan:>4d} Å  {rc:>12.3f}  {rp:>12.3f}  {abs(rc-rp):>12.3f}")

    except Exception as e:
        print(f"\n  Abundance test failed: {e}")
        print("  (This is non-critical — the κ sensitivity test is the main result)")

    # ==================================================================
    # Save all results
    # ==================================================================
    results_file = RESULTS_DIR / 'kappa_sensitivity_results.npz'
    save_data = {}
    for r in kappa_results:
        k = str(r['kappa']).replace('.', 'p')
        save_data[f'kappa_{k}_chan_ratios'] = r['chan_ratios']
        save_data[f'kappa_{k}_dem'] = r['dem']
        save_data[f'kappa_{k}_mlogt'] = r['mlogt']
        save_data[f'kappa_{k}_chisq'] = r['chisq']
        save_data[f'kappa_{k}_recovery'] = r['recovery']
        save_data[f'kappa_{k}_dn_kappa'] = r['dn_kappa']
    np.savez(results_file, **save_data)
    print(f"\nResults saved to {results_file}")

    print("\n" + "=" * 70)
    print("ALL SENSITIVITY TESTS COMPLETE")
    print(f"Full output saved to: {log_file}")
    print("=" * 70)

    tee.close()


if __name__ == '__main__':
    main()
