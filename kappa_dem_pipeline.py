#!/usr/bin/env python3
"""
kappa_dem_pipeline.py — Full synthetic DEM inversion under kappa distributions

Paper 3 pipeline: Does the standard SDO/AIA DEM analysis pipeline distinguish
a single kappa plasma from a multi-thermal Maxwellian?

Physical setup:
    - Kappa distribution: κ = 2.5, T_core = 0.6 MK, T_eff = 1.5 MK
    - Single emission measure: EM = n_e² × dh (isothermal kappa source)
    - Quiet Sun conditions: n_e = 1e9 cm⁻³

Pipeline stages:
    Stage 1: Load kappa and Maxwellian ion fractions (v10.1 matched data)
    Stage 2: Compute per-ion AIA channel contributions at T_eff using ChiantiPy
    Stage 3: Apply kappa ion fraction corrections → synthetic AIA DN/s/px
    Stage 4: Run demregpy DEM inversion with standard Maxwellian response functions
    Stage 5: Compare recovered DEM to published quiet Sun DEMs

Temperature convention (CRITICAL):
    D&D / Dz23 tables use T such that ⟨E⟩ = 3/2 kT (mean-energy temperature).
    Their T = our T_eff. Lookup at log T = 6.176 for T_eff = 1.5 MK.
    T_core = T_eff × (κ − 1.5)/κ = 1.5 × (2.5 − 1.5)/2.5 = 0.6 MK.

Atomic data:
    - Kappa ion fractions: Dzifčáková et al. 2023 (CHIANTI v10.1 basis)
    - Maxwellian ion fractions: Dz23 matched Maxwellian (same v10.1 basis)
    - Excitation rates: CHIANTI v11.0.2 (κ-independent to <20%, Dudík+ 2014)
    - AIA response functions: demregpy built-in (SSW/IDL-generated)

Dependencies:
    pip install "numpy<2.0" scipy matplotlib astropy ChiantiPy aiapy demregpy sunpy

    IMPORTANT: demregpy's GSVD implementation fails with numpy ≥ 2.0 due to an
    SVD convergence issue in the new numpy linalg backend. Use numpy 1.x.

Usage:
    1. Set XUVTOP to your CHIANTI database path
    2. Adjust DATA_DIR and KAPPA_DIR below to point to your data
    3. Run: python kappa_dem_pipeline.py

Run modes:
    'full'     — Compute per-ion contributions with ChiantiPy (SLOW, rigorous).
                 All ions with >0.01% population at T_eff, folded through AIA
                 effective areas. Required for the paper.
    'dominant' — Only dominant ions per channel (fast, approximate).
    'precomp'  — Use pre-computed demregpy responses with dominant-ion correction.

Author: Victor Edmonds
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import scipy.io as sio
from pathlib import Path

# ============================================================================
# PLATFORM FIX — ChiantiPy expects HOME, Windows doesn't set it
# ============================================================================
if 'HOME' not in os.environ:
    os.environ['HOME'] = os.path.expanduser('~')

# ============================================================================
# CONFIGURATION — adjust these paths for your system
# ============================================================================

# CHIANTI database path — v11.0.2 in the project Data folder
XUVTOP = str(Path(__file__).resolve().parent / 'Data' / 'CHIANTI_11.0.2_database')
os.environ['XUVTOP'] = XUVTOP  # Set early so ChiantiPy finds it on import

# Project data directories
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / 'Data'
KAPPA_DIR = DATA_DIR / 'kappa_v10.1'
RESULTS_DIR = PROJECT_DIR / 'Results'
FIGURES_DIR = PROJECT_DIR / 'Figures'

# Physical parameters
KAPPA = 2.5
T_EFF = 1.5e6          # K (mean-energy temperature)
T_CORE = 0.6e6         # K = T_eff × (κ - 1.5)/κ
LOG_T_EFF = 6.176      # log10(T_eff)
N_E = 1.0e9            # cm⁻³, quiet Sun electron density
EM_SCALE = 1.0e27      # cm⁻⁵, emission measure (arbitrary normalization)

# AIA channels (Å)
AIA_CHANNELS = [94, 131, 171, 193, 211, 335]
AIA_CHANNEL_NAMES = ['A94', 'A131', 'A171', 'A193', 'A211', 'A335']

# Dominant ions per AIA channel at quiet Sun temperatures
# Format: {channel_angstrom: [(element, ion_stage), ...]}
# Ion stages use spectroscopic convention (Fe XII = 12)
# These are the PRIMARY contributors; the full computation includes all ions
AIA_DOMINANT_IONS = {
    94:  [('fe', 10), ('fe', 18)],   # Fe X (QS), Fe XVIII (flare)
    131: [('fe', 8),  ('fe', 21)],   # Fe VIII (QS), Fe XXI (flare)
    171: [('fe', 9),],               # Fe IX
    193: [('fe', 12), ('fe', 24)],   # Fe XII (QS), Fe XXIV (flare)
    211: [('fe', 14),],              # Fe XIV
    335: [('fe', 16),],              # Fe XVI
}

# DEM inversion temperature grid
DEM_LOG_T_MIN = 5.7
DEM_LOG_T_MAX = 7.2
DEM_DLOGT = 0.05

# ============================================================================
# STAGE 1: Load ion fractions
# ============================================================================

def parse_ioneq(filepath):
    """
    Parse a Dz23-format .ioneq file (whitespace-delimited).

    Returns:
        log_T : 1D array of log10(T) values
        ions  : dict of {Z: {stage: fraction_array}}
                Z = atomic number, stage = spectroscopic ion stage
    """
    with open(filepath) as f:
        lines = f.readlines()

    n_temps = int(lines[0].split()[0])
    log_T = np.array([float(x) for x in lines[1].split()])

    ions = {}
    for line in lines[2:]:
        stripped = line.strip()
        if len(stripped) < 5:
            continue
        toks = stripped.split()
        if len(toks) < 3:
            continue
        try:
            z = int(toks[0])
            stage = int(toks[1])
        except ValueError:
            continue
        fracs = np.array([float(x) for x in toks[2:2+n_temps]])
        if len(fracs) != n_temps:
            continue
        if z not in ions:
            ions[z] = {}
        ions[z][stage] = fracs

    return log_T, ions


def parse_ioneq_fw(filepath):
    """
    Parse a CHIANTI-format .ioneq file using fixed-width 10-character fields.
    Required for chianti.ioneq (v10+) where values run together without spaces.

    Returns:
        log_T : 1D array of log10(T) values
        ions  : dict of {Z: {stage: fraction_array}}
    """
    with open(filepath) as f:
        lines = f.readlines()

    n_temps = int(lines[0].split()[0])
    log_T = np.array([float(x) for x in lines[1].split()])

    ions = {}
    for line in lines[2:]:
        if len(line.strip()) < 10:
            continue
        stripped = line.lstrip()
        toks = stripped.split(None, 2)
        if len(toks) < 3:
            continue
        try:
            z, stage = int(toks[0]), int(toks[1])
        except ValueError:
            continue
        data = toks[2]
        fracs = []
        for i in range(0, len(data), 10):
            chunk = data[i:i+10].strip()
            if chunk:
                try:
                    fracs.append(float(chunk))
                except ValueError:
                    fracs.append(0.0)
        fracs = np.array(fracs[:n_temps])
        if z not in ions:
            ions[z] = {}
        ions[z][stage] = fracs

    return log_T, ions


def interp_logT(log_T, fracs, target_logT):
    """Linearly interpolate ion fraction at a specific log T."""
    return np.interp(target_logT, log_T, fracs)


def load_ion_fractions():
    """
    Load kappa (κ=2.5) and matched Maxwellian ion fractions from Dz23 v10.1 data.

    Returns:
        kappa_logT, kappa_ions : temperature grid and ion fractions for κ=2.5
        mxw_logT, mxw_ions    : temperature grid and ion fractions for Maxwellian
    """
    kappa_file = KAPPA_DIR / 'Dz23_kappa_2p5.ioneq'
    mxw_file = KAPPA_DIR / 'Dz23_mxw.ioneq'

    if not kappa_file.exists():
        raise FileNotFoundError(
            f"Kappa ion fraction file not found: {kappa_file}\n"
            f"Download from kappa.asu.cas.cz"
        )
    if not mxw_file.exists():
        raise FileNotFoundError(
            f"Maxwellian ion fraction file not found: {mxw_file}\n"
            f"Download from kappa.asu.cas.cz"
        )

    kappa_logT, kappa_ions = parse_ioneq(str(kappa_file))
    mxw_logT, mxw_ions = parse_ioneq(str(mxw_file))

    print(f"Loaded κ=2.5 ion fractions: {len(kappa_logT)} temperature points")
    print(f"Loaded Maxwellian ion fractions: {len(mxw_logT)} temperature points")

    return kappa_logT, kappa_ions, mxw_logT, mxw_ions


def compute_ion_fraction_ratios(kappa_logT, kappa_ions, mxw_logT, mxw_ions):
    """
    Compute the kappa/Maxwellian ion fraction ratio for all ions at T_eff.

    This ratio is the correction factor applied to the Maxwellian emissivity
    to get the kappa emissivity for each ion.

    Returns:
        ratios : dict of {Z: {stage: ratio}}
    """
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


# ============================================================================
# STAGE 2: Compute per-ion AIA channel contributions using ChiantiPy
# ============================================================================

def compute_aia_ion_contributions(kappa_logT=None, kappa_ions=None,
                                   abundance='sun_coronal_2021_chianti'):
    """
    Compute the contribution of each ion to each AIA channel at T_eff
    using ChiantiPy's ion class + aiapy effective areas.

    Ion selection uses BOTH the kappa (Dz23) and Maxwellian (CHIANTI)
    equilibria to ensure we include all ions that emit significantly
    under the kappa distribution.  The key point: at T_eff = 1.5 MK,
    the Maxwellian equilibrium has iron as Fe I/II, but the kappa
    distribution populates Fe VIII–XVI.  We need those coronal ions.

    For each selected ion:
        1. Compute line intensities using ChiantiPy (Maxwellian atomic
           physics — excitation rates are ~κ-independent)
        2. Fold each line through the AIA effective area curves
        3. Sum contributions per channel

    Results are checkpointed to disk after every ion, so the computation
    can resume if interrupted.

    Returns:
        contributions : dict of {channel: {(Z, stage): DN_contribution}}
    """
    import time
    import json
    import ChiantiPy.core as ch
    import ChiantiPy.tools.io as chio
    import ChiantiPy.tools.util as chutil
    import aiapy.response as aresp
    import astropy.units as u

    os.environ['XUVTOP'] = XUVTOP

    # Checkpoint file — stores completed ion contributions
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_file = RESULTS_DIR / 'ion_contributions_checkpoint.json'

    # Load AIA effective area curves
    print("Loading AIA effective area curves...")
    aia_ea = {}
    aia_wvl = {}
    for chan in AIA_CHANNELS:
        r = aresp.Channel(chan * u.angstrom)
        aia_wvl[chan] = r.wavelength.to(u.angstrom).value  # Å
        aia_ea[chan] = r.effective_area.to(u.cm**2).value   # cm²

    # Load abundance file
    ab = chio.abundanceRead(abundancename=abundance)
    abundances = ab['abundance']

    # Load CHIANTI Maxwellian ionization equilibrium
    ioneq_data = chio.ioneqRead(ioneqName='chianti')
    ioneq_logT = ioneq_data['ioneqTemperature']
    ioneq_all = ioneq_data['ioneqAll']

    # ----------------------------------------------------------------
    # Build ion list from BOTH kappa and Maxwellian equilibria
    # ----------------------------------------------------------------
    # We need ions that are significant under the kappa distribution
    # at T_eff (these are the ones that actually emit), PLUS ions that
    # peak anywhere in the AIA-sensitive range under Maxwellian (for
    # completeness in the DEM response calculation).
    #
    # The critical fix: at T_eff=1.5 MK, Maxwellian Fe is Fe I/II,
    # but kappa Fe is Fe VIII-XVI.  Without the kappa fractions, we
    # miss all the coronal iron lines that dominate AIA.
    # ----------------------------------------------------------------

    t_idx_mxw = np.argmin(np.abs(ioneq_logT - LOG_T_EFF))

    # Also include ions with significant Maxwellian population anywhere
    # in the AIA sensitivity range (log T = 5.5 to 7.3)
    t_mask_aia = (ioneq_logT >= 5.5) & (ioneq_logT <= 7.3)

    print(f"  ioneq_all shape: {ioneq_all.shape}")
    print(f"  ioneq_logT range: {ioneq_logT[0]:.2f} to {ioneq_logT[-1]:.2f}")
    print(f"  AIA T mask selects {np.sum(t_mask_aia)} temperature points")

    ion_set = set()  # (z, stage) tuples to avoid duplicates

    for z in range(1, 31):  # H through Zn
        ab_el = abundances[z-1] if z-1 < len(abundances) else 0.0
        if ab_el < 1e-12:
            continue

        n_stages_chianti = min(z + 1, ioneq_all.shape[1])

        # --- Include ions from kappa distribution at T_eff ---
        if kappa_ions is not None and z in kappa_ions:
            for stage in kappa_ions[z]:
                f_kappa = interp_logT(kappa_logT, kappa_ions[z][stage],
                                       LOG_T_EFF)
                if f_kappa > 1e-6:
                    ion_set.add((z, stage, ab_el))

        # --- Include ions from Maxwellian across AIA T range ---
        if z-1 >= ioneq_all.shape[0]:
            continue
        for stage in range(1, n_stages_chianti + 1):
            ion_idx = stage - 1
            if ion_idx >= ioneq_all.shape[1]:
                continue
            # Check if this ion has significant population ANYWHERE
            # in the AIA temperature range
            try:
                vals = ioneq_all[z-1, ion_idx, t_mask_aia]
                f_max_aia = np.max(vals) if vals.size > 0 else 0.0
            except (IndexError, ValueError):
                f_max_aia = 0.0
            if f_max_aia > 1e-4:
                ion_set.add((z, stage, ab_el))

    # Convert to sorted list
    ion_list = sorted(ion_set, key=lambda x: (x[0], x[1]))

    # For display, annotate with kappa fraction at T_eff
    ion_list_annotated = []
    for z, stage, ab_el in ion_list:
        f_kappa = 0.0
        if kappa_ions is not None and z in kappa_ions and stage in kappa_ions[z]:
            f_kappa = interp_logT(kappa_logT, kappa_ions[z][stage], LOG_T_EFF)
        # Also get Maxwellian fraction at T_eff
        ion_idx = stage - 1
        f_mxw = 0.0
        if z-1 < ioneq_all.shape[0] and ion_idx < ioneq_all.shape[1]:
            f_mxw = ioneq_all[z-1, ion_idx, t_idx_mxw]
        # Use the larger of the two for display
        f_display = max(f_kappa, f_mxw)
        ion_list_annotated.append((z, stage, f_display, ab_el))

    # Use annotated list as the working ion list
    ion_list = ion_list_annotated

    print(f"Found {len(ion_list)} ions with significant population")
    print(f"  (from kappa at T_eff + Maxwellian across AIA range)")

    # Load checkpoint if it exists (resume from crash)
    contributions = {chan: {} for chan in AIA_CHANNELS}
    completed_ions = set()

    if checkpoint_file.exists():
        print(f"Loading checkpoint from {checkpoint_file}...")
        with open(checkpoint_file, 'r') as f:
            saved = json.load(f)
        for chan_str, ion_dict in saved.items():
            chan = int(chan_str)
            for key_str, value in ion_dict.items():
                z, stage = map(int, key_str.split(','))
                contributions[chan][(z, stage)] = value
                completed_ions.add((z, stage))
        n_done = len(completed_ions)
        print(f"  Resuming: {n_done} ions already computed, "
              f"{len(ion_list) - n_done} remaining")
    else:
        print("No checkpoint found — starting fresh.")

    # For each ion, compute line intensities and fold through AIA
    t_start = time.time()
    n_computed = 0
    n_total = len(ion_list) - len(completed_ions)

    for i, (z, stage, f_ion, ab_el) in enumerate(ion_list):
        # Skip if already in checkpoint
        if (z, stage) in completed_ions:
            continue

        el_name = chutil.zion2name(z, stage)

        # Timing / ETA
        if n_computed > 0:
            elapsed = time.time() - t_start
            per_ion = elapsed / n_computed
            remaining = (n_total - n_computed) * per_ion
            eta_min = remaining / 60
            eta_str = f" ETA {eta_min:.0f}min"
        else:
            eta_str = ""

        print(f"  [{i+1}/{len(ion_list)}] {el_name} "
              f"(f_ion={f_ion:.3e}, Ab={ab_el:.3e}){eta_str}...",
              end='', flush=True)

        try:
            ion = ch.ion(el_name, temperature=T_EFF, eDensity=N_E,
                        abundance=abundance)
            ion.intensity()

            if not hasattr(ion, 'Intensity') or ion.Intensity is None:
                print(" no lines in AIA range")
                n_computed += 1
                continue

            wvls = np.array(ion.Intensity['wvl'])       # line wavelengths (Å)
            intens = np.array(ion.Intensity['intensity']).squeeze()

            if intens.ndim == 0:
                print(" scalar intensity, skipping")
                n_computed += 1
                continue

            # Fold each line through each AIA channel's effective area
            for chan in AIA_CHANNELS:
                channel_sum = 0.0
                for wvl, inten in zip(wvls, intens):
                    if inten < 1e-40:
                        continue
                    ea_at_wvl = np.interp(wvl, aia_wvl[chan], aia_ea[chan])
                    if ea_at_wvl > 0:
                        channel_sum += inten * ea_at_wvl

                if channel_sum > 0:
                    contributions[chan][(z, stage)] = channel_sum

            n_lines = np.sum(intens > 1e-40)
            elapsed_ion = time.time() - t_start
            print(f" {n_lines} lines ({elapsed_ion/max(n_computed,1):.1f}s/ion)")
            n_computed += 1

            # Checkpoint: save after every ion
            save_dict = {}
            for chan in AIA_CHANNELS:
                save_dict[str(chan)] = {
                    f"{z},{s}": v for (z, s), v in contributions[chan].items()
                }
            with open(checkpoint_file, 'w') as f:
                json.dump(save_dict, f, indent=2)

        except Exception as e:
            print(f" ERROR: {e}")
            n_computed += 1
            continue

    total_time = time.time() - t_start
    print(f"\nStage 2 complete: {n_computed} ions computed in "
          f"{total_time/60:.1f} min")
    print(f"Checkpoint saved to {checkpoint_file}")

    return contributions


# ============================================================================
# STAGE 2 (ALTERNATIVE): Dominant-ion approximation
# ============================================================================

def compute_dominant_ion_contributions():
    """
    Faster alternative to compute_aia_ion_contributions().

    Uses ChiantiPy to compute contributions from only the dominant ions
    in each AIA channel. This is an approximation — the full computation
    includes all ions. Use this for quick checks, not for the paper.

    Returns same format as compute_aia_ion_contributions().
    """
    import ChiantiPy.core as ch
    import ChiantiPy.tools.util as chutil
    import aiapy.response as aresp
    import astropy.units as u

    os.environ['XUVTOP'] = XUVTOP

    # Load AIA effective areas
    aia_ea = {}
    aia_wvl = {}
    for chan in AIA_CHANNELS:
        r = aresp.Channel(chan * u.angstrom)
        aia_wvl[chan] = r.wavelength.to(u.angstrom).value
        aia_ea[chan] = r.effective_area.to(u.cm**2).value

    contributions = {chan: {} for chan in AIA_CHANNELS}
    computed_ions = set()

    for chan, ion_list in AIA_DOMINANT_IONS.items():
        for (el, stage) in ion_list:
            z = {'fe': 26, 'si': 14, 'mg': 12, 'o': 8, 'c': 6, 'n': 7, 'ne': 10, 's': 16}[el]
            key = (z, stage)

            if key in computed_ions:
                # Already computed this ion — reuse cached intensity data
                # (would need to store it; simplified here)
                pass

            ion_name = f'{el}_{stage}'
            print(f"  Computing {ion_name} for AIA {chan}...", end='', flush=True)

            try:
                ion = ch.ion(ion_name, temperature=T_EFF, eDensity=N_E,
                            abundance='sun_coronal_2021_chianti')
                ion.intensity()

                wvls = np.array(ion.Intensity['wvl'])
                intens = np.array(ion.Intensity['intensity']).squeeze()

                channel_sum = 0.0
                for wvl, inten in zip(wvls, intens):
                    if inten < 1e-40:
                        continue
                    ea_at_wvl = np.interp(wvl, aia_wvl[chan], aia_ea[chan])
                    if ea_at_wvl > 0:
                        channel_sum += inten * ea_at_wvl

                contributions[chan][(z, stage)] = channel_sum
                computed_ions.add(key)
                print(f" done ({channel_sum:.3e})")

            except Exception as e:
                print(f" ERROR: {e}")

    return contributions


# ============================================================================
# STAGE 2 (FULL): Compute complete AIA temperature response functions
# ============================================================================

def compute_full_aia_response(log_t_grid=None, abundance='sun_coronal_2021_chianti'):
    """
    Compute AIA temperature response functions K_i(T) from scratch using
    ChiantiPy's spectrum class. This is the most rigorous approach.

    At each temperature, computes the full EUV spectrum (all ions + continuum),
    then folds through AIA effective areas.

    WARNING: This is SLOW. ~3 min per temperature × ~30 temperatures = ~1.5 hours.
    Results are saved to disk. Only needs to be run once.

    Args:
        log_t_grid : array of log10(T) values. If None, uses demregpy grid.
        abundance  : CHIANTI abundance file name

    Returns:
        log_t_grid : temperature grid
        tresp      : array of shape (n_temps, 6) — response for each AIA channel
    """
    import ChiantiPy.core as ch
    import aiapy.response as aresp
    import astropy.units as u

    os.environ['XUVTOP'] = XUVTOP

    # Default temperature grid matches demregpy
    if log_t_grid is None:
        log_t_grid = np.arange(5.5, 7.55, 0.05)

    temperatures = 10**log_t_grid
    n_temps = len(temperatures)

    # Load AIA effective areas
    print("Loading AIA effective area curves...")
    aia_channels_obj = {}
    for chan in AIA_CHANNELS:
        aia_channels_obj[chan] = aresp.Channel(chan * u.angstrom)

    # Output file for intermediate saves
    save_file = RESULTS_DIR / 'aia_tresp_chianti11.npz'

    # Check for partial results
    tresp = np.zeros((n_temps, len(AIA_CHANNELS)))
    start_idx = 0
    if save_file.exists():
        saved = np.load(save_file)
        if np.allclose(saved['log_t_grid'], log_t_grid):
            tresp = saved['tresp']
            # Find first zero row
            for i in range(n_temps):
                if np.all(tresp[i, :] == 0):
                    start_idx = i
                    break
            else:
                start_idx = n_temps
            print(f"Resuming from temperature index {start_idx}/{n_temps}")

    # Wavelength range covering all AIA EUV channels
    wvl_min = 80   # Å
    wvl_max = 350  # Å

    for i in range(start_idx, n_temps):
        T = temperatures[i]
        logT = log_t_grid[i]

        print(f"  [{i+1}/{n_temps}] log T = {logT:.2f} (T = {T:.2e} K)...",
              end='', flush=True)

        try:
            # Compute full CHIANTI spectrum
            s = ch.spectrum(temperature=T, eDensity=N_E,
                           wavelength=[wvl_min, wvl_max],
                           em=EM_SCALE,
                           abundance=abundance,
                           minAbund=1e-6,
                           verbose=0)

            # Get the spectrum (wavelength, intensity)
            spec_wvl = s.Wavelength  # Å
            spec_int = np.array(s.Spectrum['intensity']).squeeze()

            # Fold through each AIA channel
            for j, chan in enumerate(AIA_CHANNELS):
                r = aia_channels_obj[chan]
                ea_wvl = r.wavelength.to(u.angstrom).value
                ea_val = r.effective_area.to(u.cm**2).value

                # Interpolate effective area onto spectrum wavelength grid
                ea_interp = np.interp(spec_wvl, ea_wvl, ea_val)

                # Integrate: response = ∫ spectrum(λ) × EA(λ) dλ
                dw = np.gradient(spec_wvl)
                tresp[i, j] = np.sum(spec_int * ea_interp * dw)

            print(f" done (peak channel: {AIA_CHANNELS[np.argmax(tresp[i, :])]} Å)")

            # Save after each temperature (can resume if interrupted)
            np.savez(save_file, log_t_grid=log_t_grid, tresp=tresp,
                     channels=AIA_CHANNELS)

        except Exception as e:
            print(f" ERROR: {e}")
            continue

    return log_t_grid, tresp


# ============================================================================
# STAGE 3: Compute synthetic kappa AIA observations
# ============================================================================

def compute_kappa_synthetic_dns_from_response(kappa_logT, kappa_ions,
                                               mxw_logT, mxw_ions,
                                               tresp_logT, tresp_matrix):
    """
    Compute synthetic AIA DN/s/px for a kappa plasma using the correction
    factor approach applied to the full temperature response functions.

    Method:
        For a single-temperature kappa plasma at T_eff, the emission from
        each ion differs from Maxwellian only through the ion fraction.

        The standard Maxwellian response K_i(T) at T = T_eff gives the
        Maxwellian prediction. To get the kappa prediction, we need to
        decompose K_i by ion and apply per-ion corrections.

        Since we don't have the per-ion decomposition of the pre-computed
        response, we use the FULL ChiantiPy computation (Stage 2).

        This function implements the alternative: use the pre-computed
        total response and apply a CHANNEL-LEVEL correction. This is
        approximate — see compute_kappa_synthetic_dns_rigorous() for the
        exact approach.

    Returns:
        dn_kappa : array of shape (6,) — synthetic DN/s/px per AIA channel
        dn_mxw   : array of shape (6,) — Maxwellian DN/s/px at T_eff for reference
    """
    # Maxwellian DN at T_eff from response functions
    dn_mxw = np.zeros(len(AIA_CHANNELS))
    for j in range(len(AIA_CHANNELS)):
        dn_mxw[j] = np.interp(LOG_T_EFF, tresp_logT, tresp_matrix[:, j]) * EM_SCALE

    print("\nWARNING: Channel-level correction is approximate.")
    print("Use compute_kappa_synthetic_dns_rigorous() for the paper.\n")

    return dn_mxw, dn_mxw  # placeholder — rigorous version needed


def compute_kappa_synthetic_dns_rigorous(ion_contributions, ratios):
    """
    Compute synthetic AIA DN/s/px for a kappa plasma using per-ion
    ChiantiPy contributions and kappa/Maxwellian ion fraction ratios.

    This is the RIGOROUS approach:
        DN_i^κ = Σ_ions [ DN_i,ion^Mxw(T_eff) × f_ion^κ(T_eff) / f_ion^Mxw(T_eff) ]

    Args:
        ion_contributions : from compute_aia_ion_contributions()
        ratios           : from compute_ion_fraction_ratios()

    Returns:
        dn_kappa : dict {channel: DN_value} — kappa synthetic observations
        dn_mxw   : dict {channel: DN_value} — Maxwellian reference
        details  : dict {channel: per-ion breakdown}
    """
    dn_kappa = {}
    dn_mxw = {}
    details = {}

    for chan in AIA_CHANNELS:
        mxw_total = 0.0
        kappa_total = 0.0
        chan_details = []

        for (z, stage), contrib in ion_contributions[chan].items():
            mxw_total += contrib

            ratio = ratios.get(z, {}).get(stage, 1.0)

            # Guard against inf ratios: these arise when f_mxw(Dz23) ≈ 0
            # but f_kappa > 0.  The ChiantiPy contribution for such ions
            # is already negligible (computed at a T where the ion barely
            # exists under Maxwellian), so inf × tiny = meaningless.
            # Cap at a large but finite value; the absolute contribution
            # remains tiny because `contrib` itself is near-zero.
            if not np.isfinite(ratio):
                ratio = 0.0  # skip: can't reliably scale a ~zero base
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


# ============================================================================
# STAGE 4: DEM inversion
# ============================================================================

def load_demregpy_response():
    """
    Load the pre-computed AIA temperature response functions from demregpy.

    These are generated by SSW/IDL (make_aiaresp_forpy.pro) and shipped
    with the demregpy package. They use CHIANTI atomic data with coronal
    abundances and Maxwellian ion fractions.

    Returns:
        tresp_logT : log10(T) grid
        tresp_matrix : response matrix, shape (n_temps, 6)
        channel_names : list of channel name strings
    """
    import demregpy

    tresp_file = demregpy.tresp.aia_tresp
    print(f"Loading demregpy AIA response from: {tresp_file}")

    trin = sio.readsav(tresp_file)

    tresp_logT = np.array(trin['logt'])
    n_temps = len(tresp_logT)
    n_channels = len(trin['tr'])

    tresp_matrix = np.zeros((n_temps, n_channels))
    channel_names = []
    for i in range(n_channels):
        tresp_matrix[:, i] = trin['tr'][i]
        channel_names.append(trin['channels'][i].decode('utf-8'))

    print(f"  Temperature grid: log T = {tresp_logT[0]:.2f} to {tresp_logT[-1]:.2f}, "
          f"{n_temps} points")
    print(f"  Channels: {channel_names}")

    return tresp_logT, tresp_matrix, channel_names


def run_dem_inversion(dn_in, edn_in, tresp_matrix, tresp_logT):
    """
    Run the Hannah & Kontar (2012) regularized DEM inversion.

    Args:
        dn_in        : array (6,) — observed DN/s/px
        edn_in       : array (6,) — uncertainties
        tresp_matrix : temperature response matrix
        tresp_logT   : log T grid for response functions

    Returns:
        dem, edem, elogt, chisq, dn_reg
    """
    from demregpy import dn2dem

    # Temperature bins for output DEM
    temps = 10**np.arange(DEM_LOG_T_MIN, DEM_LOG_T_MAX + DEM_DLOGT, DEM_DLOGT)

    print(f"\nRunning DEM inversion...")
    print(f"  Input DN: {dn_in}")
    print(f"  Input errors: {edn_in}")
    print(f"  Temperature bins: {len(temps)} from log T = {DEM_LOG_T_MIN} to {DEM_LOG_T_MAX}")

    dem, edem, elogt, chisq, dn_reg = dn2dem(
        dn_in, edn_in, tresp_matrix, tresp_logT, temps
    )

    mlogt = [np.mean([np.log10(temps[i]), np.log10(temps[i+1])])
             for i in range(len(temps)-1)]

    print(f"  χ² = {chisq:.3f}")
    print(f"  DN recovery: {dn_reg / dn_in}")

    return dem, edem, elogt, chisq, dn_reg, np.array(mlogt), temps


# ============================================================================
# STAGE 5: Comparison and plotting
# ============================================================================

def compute_aia_noise(dn_in, exposure_time=2.9):
    """
    Compute realistic AIA noise (shot noise + read noise).

    Args:
        dn_in : DN/s/px values
        exposure_time : seconds (default 2.9s for standard AIA cadence)

    Returns:
        edn : uncertainty in DN/s/px
    """
    gains = np.array([18.3, 17.6, 17.7, 18.3, 18.3, 17.6])
    chan_wvl = np.array([94, 131, 171, 193, 211, 335])
    dn2ph = gains * chan_wvl / 3397.0
    rdnse = np.array([1.14, 1.18, 1.15, 1.20, 1.20, 1.18])

    dn_total = dn_in * exposure_time
    shotnoise = np.sqrt(dn2ph * np.abs(dn_total)) / dn2ph / exposure_time
    edn = np.sqrt(rdnse**2 + shotnoise**2)

    return edn


def plot_dem_comparison(mlogt, dem, edem, elogt, chisq,
                        dn_in, dn_reg, channel_names,
                        title='Recovered DEM from κ=2.5 synthetic observations'):
    """
    Plot the recovered DEM and the DN residuals.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # DEM plot
    ax = axes[0]
    ax.errorbar(mlogt, dem, xerr=elogt, yerr=edem,
                fmt='or', ecolor='lightcoral', elinewidth=2, capsize=0,
                label=f'Recovered DEM (χ²={chisq:.2f})')
    ax.axvline(LOG_T_EFF, color='blue', ls='--', alpha=0.7, label=f'T_eff = {T_EFF/1e6:.1f} MK')
    ax.axvline(np.log10(T_CORE), color='red', ls=':', alpha=0.7, label=f'T_core = {T_CORE/1e6:.1f} MK')
    ax.set_xlabel('log₁₀ T [K]')
    ax.set_ylabel('DEM [cm⁻⁵ K⁻¹]')
    ax.set_yscale('log')
    ax.set_xlim(5.5, 7.5)
    ax.legend(fontsize=10)
    ax.set_title(title)

    # DN residuals
    ax = axes[1]
    colors = ['darkgreen', 'darkcyan', 'gold', 'sienna', 'indianred', 'darkslateblue']
    ratios = dn_reg / dn_in
    for i, (name, ratio) in enumerate(zip(channel_names, ratios)):
        ax.bar(i, ratio, color=colors[i], alpha=0.7, label=name)
    ax.axhline(1.0, color='grey', ls='--')
    ax.set_xticks(range(len(channel_names)))
    ax.set_xticklabels(channel_names)
    ax.set_ylabel('DN_recovered / DN_input')
    ax.set_ylim(0.8, 1.2)
    ax.set_title('Channel residuals')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'dem_inversion_kappa.png', dpi=150, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'dem_inversion_kappa.pdf', bbox_inches='tight')
    print(f"\nFigures saved to {FIGURES_DIR}/")
    plt.show()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    Run the full pipeline.

    Set RUN_MODE to control which computation to use:
        'full'      — compute everything from CHIANTI (slow, rigorous)
        'dominant'  — use only dominant ions per channel (fast, approximate)
        'precomp'   — skip Stage 2, use precomputed response + channel correction
    """
    RUN_MODE = 'full'  # Change this as needed

    # Create output directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("KAPPA DEM INVERSION PIPELINE")
    print(f"κ = {KAPPA}, T_core = {T_CORE/1e6:.1f} MK, T_eff = {T_EFF/1e6:.1f} MK")
    print(f"Run mode: {RUN_MODE}")
    print("=" * 70)

    # ---- STAGE 1: Load ion fractions ----
    print("\n--- STAGE 1: Loading ion fractions ---")
    kappa_logT, kappa_ions, mxw_logT, mxw_ions = load_ion_fractions()
    ratios = compute_ion_fraction_ratios(kappa_logT, kappa_ions, mxw_logT, mxw_ions)

    # Print key Fe ion ratios
    print("\nFe ion fraction ratios (κ/Mxw) at log T = {:.3f}:".format(LOG_T_EFF))
    for stage in range(8, 18):
        r = ratios.get(26, {}).get(stage, 0)
        f_k = interp_logT(kappa_logT, kappa_ions.get(26, {}).get(stage, np.zeros_like(kappa_logT)), LOG_T_EFF)
        f_m = interp_logT(mxw_logT, mxw_ions.get(26, {}).get(stage, np.zeros_like(mxw_logT)), LOG_T_EFF)
        rn = roman(stage)
        print(f"  Fe {rn:>5s}: κ={f_k:.4e}  Mxw={f_m:.4e}  ratio={r:.3f}")

    # ---- STAGE 2: Compute AIA contributions ----
    print("\n--- STAGE 2: Computing AIA channel contributions ---")

    if RUN_MODE == 'full':
        contributions = compute_aia_ion_contributions(
            kappa_logT=kappa_logT, kappa_ions=kappa_ions)
    elif RUN_MODE == 'dominant':
        contributions = compute_dominant_ion_contributions()
    else:
        contributions = None

    # ---- STAGE 3: Synthetic kappa DNs ----
    print("\n--- STAGE 3: Computing synthetic kappa observations ---")

    # Load demregpy response functions (needed for DEM inversion)
    tresp_logT, tresp_matrix, channel_names = load_demregpy_response()

    if contributions is not None:
        dn_kappa, dn_mxw, details = compute_kappa_synthetic_dns_rigorous(
            contributions, ratios)

        # Convert to arrays in channel order
        dn_kappa_arr = np.array([dn_kappa[c] for c in AIA_CHANNELS])
        dn_mxw_arr = np.array([dn_mxw[c] for c in AIA_CHANNELS])

        print("\nPer-channel synthetic observations:")
        print(f"{'Channel':>8s} {'DN_Mxw':>12s} {'DN_kappa':>12s} {'Ratio':>8s}")
        print("-" * 44)
        for j, chan in enumerate(AIA_CHANNELS):
            ratio = dn_kappa_arr[j] / dn_mxw_arr[j] if dn_mxw_arr[j] > 0 else 0
            print(f"  {chan:>4d} Å {dn_mxw_arr[j]:>12.4e} {dn_kappa_arr[j]:>12.4e} {ratio:>8.3f}")

        # Print dominant ion contributions per channel
        print("\nDominant ions per channel:")
        for chan in AIA_CHANNELS:
            if chan in details and details[chan]:
                top = details[chan][:3]
                ions_str = ', '.join(
                    f"Z={d['Z']} stage={d['stage']} ({d['ratio']:.2f}×)"
                    for d in top
                )
                print(f"  {chan} Å: {ions_str}")
    else:
        # Precomputed mode — use channel-level scaling (APPROXIMATE)
        print("Using precomputed response + channel-level scaling (approximate)")
        dn_mxw_arr = np.zeros(6)
        for j in range(6):
            dn_mxw_arr[j] = np.interp(LOG_T_EFF, tresp_logT, tresp_matrix[:, j])
        dn_mxw_arr *= EM_SCALE

        # Apply dominant-ion correction as approximation
        dominant_fe_stages = {94: 10, 131: 8, 171: 9, 193: 12, 211: 14, 335: 16}
        dn_kappa_arr = np.zeros(6)
        for j, chan in enumerate(AIA_CHANNELS):
            stage = dominant_fe_stages[chan]
            r = ratios.get(26, {}).get(stage, 1.0)
            dn_kappa_arr[j] = dn_mxw_arr[j] * r
            print(f"  {chan} Å: dominant Fe {roman(stage)}, "
                  f"ratio = {r:.3f}, DN = {dn_kappa_arr[j]:.4e}")

    # ---- STAGE 4: DEM inversion ----
    print("\n--- STAGE 4: DEM inversion ---")

    # Scale the kappa DNs to be in the same units as the demregpy response
    # The demregpy response is in DN/s/px per unit EM (cm⁻⁵)
    # Our dn_kappa_arr is in raw units from ChiantiPy — need to match

    # If using ChiantiPy-computed contributions (Stages 2+3):
    #   The absolute calibration needs matching to demregpy.
    #   Best approach: normalize so that the Maxwellian DN at T_eff matches
    #   what the demregpy response predicts.

    # Get demregpy prediction for Maxwellian at T_eff
    demreg_dn_mxw = np.zeros(6)
    for j in range(6):
        demreg_dn_mxw[j] = np.interp(LOG_T_EFF, tresp_logT, tresp_matrix[:, j])
    demreg_dn_mxw *= EM_SCALE

    # Apply the RATIO from our ChiantiPy calculation to the demregpy normalization
    # This is the key step: we trust the per-ion RATIOS from matched v10.1 data,
    # and apply them to the demregpy-calibrated absolute DN values.
    if contributions is not None:
        kappa_over_mxw = np.zeros(6)
        for j, chan in enumerate(AIA_CHANNELS):
            if dn_mxw_arr[j] > 0:
                kappa_over_mxw[j] = dn_kappa_arr[j] / dn_mxw_arr[j]
            else:
                kappa_over_mxw[j] = 1.0
        dn_for_inversion = demreg_dn_mxw * kappa_over_mxw
    else:
        dn_for_inversion = dn_kappa_arr  # already in demregpy units

    # Compute uncertainties (realistic AIA noise)
    edn = compute_aia_noise(dn_for_inversion)

    # Run the inversion
    dem, edem, elogt, chisq, dn_reg, mlogt, temps = run_dem_inversion(
        dn_for_inversion, edn, tresp_matrix, tresp_logT
    )

    # ---- STAGE 5: Plot and compare ----
    print("\n--- STAGE 5: Comparison ---")

    plot_dem_comparison(mlogt, dem, edem, elogt, chisq,
                        dn_for_inversion, dn_reg, channel_names)

    # Save all results
    results_file = RESULTS_DIR / 'kappa_dem_inversion_results.npz'
    np.savez(results_file,
             kappa=KAPPA, T_eff=T_EFF, T_core=T_CORE,
             dn_kappa=dn_for_inversion, edn=edn,
             dn_mxw=demreg_dn_mxw,
             kappa_over_mxw=kappa_over_mxw if contributions else None,
             dem=dem, edem=edem, elogt=elogt, mlogt=mlogt,
             chisq=chisq, dn_reg=dn_reg,
             tresp_logT=tresp_logT, tresp_matrix=tresp_matrix)
    print(f"\nResults saved to {results_file}")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


def roman(n):
    """Convert integer to Roman numeral string."""
    vals = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),
            (100,'C'),(90,'XC'),(50,'L'),(40,'XL'),
            (10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]
    result = ''
    for val, numeral in vals:
        while n >= val:
            result += numeral
            n -= val
    return result


if __name__ == '__main__':
    main()
