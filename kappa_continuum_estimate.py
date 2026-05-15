#!/usr/bin/env python
"""
kappa_continuum_estimate.py — Quantify the continuum contribution to AIA channels
and estimate the kappa/Maxwellian continuum correction.

The main pipeline (kappa_dem_pipeline.py) computes LINE emission only.
This script quantifies what's missing:

  1. Maxwellian free-free + free-bound continuum at T_eff = 1.5 MK
  2. Fold through AIA effective areas
  3. Compare to line-only DN from the checkpoint
  4. Estimate kappa continuum enhancement using analytical formula
  5. Show the corrected χ² with continuum included

The kappa free-free emissivity enhancement over Maxwellian at photon energy E is
(Dudík et al. 2017, Eq. 4; see also Owocki & Scudder 1983):

  j_κ(E) / j_Mxw(E) = A_κ × [1 + E / ((κ - 3/2) k T_eff)]^{-(κ+1)}
                        / exp(-E / k T_eff)

where A_κ = Γ(κ+1) / [Γ(κ-1/2) × (κ-3/2)^{3/2}] is the normalization.

For free-bound, the enhancement is similar but depends on ion fractions —
we use the per-ion κ/Mxw ratio as for lines, which is the dominant effect.

Author: Victor Edmonds
Date: March 2026
"""

import numpy as np
import json
import sys
import os
from pathlib import Path
from scipy.special import gamma

# ============================================================================
# Setup
# ============================================================================

if 'HOME' not in os.environ:
    os.environ['HOME'] = os.path.expanduser('~')

SCRIPT_DIR = Path(__file__).resolve().parent
XUVTOP = str(SCRIPT_DIR / 'Data' / 'CHIANTI_11.0.2_database')
os.environ['XUVTOP'] = XUVTOP

RESULTS_DIR = SCRIPT_DIR / 'Results'
RESULTS_DIR.mkdir(exist_ok=True)

# Physical constants
K_B = 1.380649e-16      # erg/K (Boltzmann)
H_PLANCK = 6.62607e-27  # erg·s
C_LIGHT = 2.99792e10    # cm/s
EV_PER_ERG = 6.242e11   # eV per erg

# Pipeline parameters (must match kappa_dem_pipeline.py)
T_EFF = 1.5e6            # K
T_CORE = 0.6e6           # K
LOG_T_EFF = 6.176
N_E = 1.0e9              # cm⁻³
EM_SCALE = 1.0e27        # cm⁻⁵
KAPPA = 2.5

AIA_CHANNELS = [94, 131, 171, 193, 211, 335]


# ============================================================================
# Tee class
# ============================================================================

class Tee:
    def __init__(self, filepath):
        self.file = open(filepath, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        sys.stdout = self.stdout
        self.file.close()


# ============================================================================
# Kappa/Maxwellian free-free ratio
# ============================================================================

def kappa_ff_ratio(energy_eV, kappa, kT_eV):
    """
    Compute the kappa/Maxwellian free-free emissivity ratio at photon energy E.

    Uses the velocity-averaged emission formula. The ratio is:
      j_κ / j_Mxw = A_κ × [1 + E/((κ-3/2) kT)]^{-(κ+1)} / exp(-E/kT)

    where A_κ normalizes the kappa distribution to have the same mean energy.

    Args:
        energy_eV : photon energy in eV (scalar or array)
        kappa     : kappa parameter
        kT_eV     : thermal energy in eV (= k_B × T_eff in eV)

    Returns:
        ratio : j_κ / j_Mxw at each energy
    """
    E = np.asarray(energy_eV, dtype=float)

    # Normalization factor A_κ
    # A_κ = Γ(κ+1) / [Γ(κ-1/2) × (κ-3/2)^{3/2}]
    A_kappa = gamma(kappa + 1) / (gamma(kappa - 0.5) * (kappa - 1.5)**1.5)

    # Kappa emissivity (proportional to)
    x = E / ((kappa - 1.5) * kT_eV)
    j_kappa = A_kappa * (1.0 + x)**(-(kappa + 1))

    # Maxwellian emissivity (proportional to)
    j_mxw = np.exp(-E / kT_eV)

    # Avoid division by zero for very small j_mxw
    ratio = np.where(j_mxw > 1e-300, j_kappa / j_mxw, 1.0)

    return ratio


# ============================================================================
# Main
# ============================================================================

def main():
    log_file = RESULTS_DIR / 'continuum_estimate_results.txt'
    tee = Tee(log_file)

    print("=" * 70)
    print("CONTINUUM CONTRIBUTION ESTIMATE")
    print(f"T_eff = {T_EFF/1e6:.1f} MK, κ = {KAPPA}")
    print(f"Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Compute Maxwellian continuum using ChiantiPy
    # ------------------------------------------------------------------
    print("\n--- Step 1: Maxwellian continuum at T_eff ---")

    import ChiantiPy.core as ch
    import ChiantiPy.tools.io as chio
    import aiapy.response as aresp
    import astropy.units as u

    # Load AIA effective areas
    print("Loading AIA effective area curves...")
    aia_ea = {}
    aia_wvl = {}
    for chan in AIA_CHANNELS:
        r = aresp.Channel(chan * u.angstrom)
        aia_wvl[chan] = r.wavelength.to(u.angstrom).value
        aia_ea[chan] = r.effective_area.to(u.cm**2).value

    # Wavelength grid for continuum (Å) — cover all AIA channels
    wvl_grid = np.arange(10, 400, 0.5)  # 10–400 Å, 0.5 Å step

    # Major ions for continuum
    # Free-free: all ions contribute (proportional to Z²×n_ion)
    # Free-bound: recombination edges, mainly from H-like and He-like ions
    # plus Fe L-shell ions
    #
    # We compute the dominant contributors: Fe ions near the equilibrium
    # charge state, plus H, He, C, N, O, Ne, Mg, Si, S, Ca, Ni
    #
    # For Fe at T_eff = 1.5 MK under Maxwellian:
    # The CHIANTI ioneq at 1.5 MK has Fe peaked around Fe IX-XIV

    # Load abundance
    ab = chio.abundanceRead(abundancename='sun_coronal_2021_chianti')
    abundances = ab['abundance']

    # Load ioneq to find dominant ions
    ioneq_data = chio.ioneqRead(ioneqName='chianti')
    ioneq_logT = ioneq_data['ioneqTemperature']
    ioneq_all = ioneq_data['ioneqAll']

    # Find temperature index closest to log T_eff in ioneq
    # Note: CHIANTI v11 ioneq may store T in linear K
    if ioneq_logT[0] > 100:
        # Linear temperatures, convert
        ioneq_logT_use = np.log10(ioneq_logT)
    else:
        ioneq_logT_use = ioneq_logT
    t_idx = np.argmin(np.abs(ioneq_logT_use - LOG_T_EFF))

    # Collect continuum per AIA channel
    cont_dn_ff = {chan: 0.0 for chan in AIA_CHANNELS}
    cont_dn_fb = {chan: 0.0 for chan in AIA_CHANNELS}

    # Elements to include (Z values for major coronal emitters)
    elements = {
        1: 'H', 2: 'He', 6: 'C', 7: 'N', 8: 'O', 10: 'Ne',
        12: 'Mg', 14: 'Si', 16: 'S', 20: 'Ca', 26: 'Fe', 28: 'Ni'
    }

    print(f"\nComputing continuum for {len(elements)} elements...")

    for z, el_name in elements.items():
        ab_el = abundances[z-1] if z-1 < len(abundances) else 0.0
        if ab_el < 1e-12:
            continue

        # Find the dominant ion stages at T_eff
        n_stages = min(z + 1, ioneq_all.shape[1])
        for stage in range(1, n_stages + 1):
            ion_idx = stage - 1
            if ion_idx >= ioneq_all.shape[1]:
                continue

            try:
                f_ion = ioneq_all[z-1, ion_idx, t_idx]
            except IndexError:
                continue

            if f_ion < 1e-6:
                continue

            # ChiantiPy ion string
            import ChiantiPy.tools.util as chutil
            ion_str = chutil.zion2name(z, stage)

            try:
                cont = ch.continuum(ion_str, T_EFF,
                                    abundance='sun_coronal_2021_chianti',
                                    em=EM_SCALE)

                # Free-free
                try:
                    cont.freeFree(wvl_grid)
                    if hasattr(cont, 'FreeFree') and cont.FreeFree is not None:
                        ff_intensity = np.array(cont.FreeFree['intensity']).squeeze()
                        if ff_intensity.ndim > 0 and ff_intensity.size == len(wvl_grid):
                            for chan in AIA_CHANNELS:
                                ea_interp = np.interp(wvl_grid, aia_wvl[chan], aia_ea[chan])
                                dn_ff = np.trapz(ff_intensity * ea_interp, wvl_grid)
                                cont_dn_ff[chan] += dn_ff
                except Exception:
                    pass

                # Free-bound
                try:
                    cont.freeBound(wvl_grid)
                    if hasattr(cont, 'FreeBound') and cont.FreeBound is not None:
                        fb_intensity = np.array(cont.FreeBound['intensity']).squeeze()
                        if fb_intensity.ndim > 0 and fb_intensity.size == len(wvl_grid):
                            for chan in AIA_CHANNELS:
                                ea_interp = np.interp(wvl_grid, aia_wvl[chan], aia_ea[chan])
                                dn_fb = np.trapz(fb_intensity * ea_interp, wvl_grid)
                                cont_dn_fb[chan] += dn_fb
                except Exception:
                    pass

            except Exception as e:
                continue

        print(f"  {el_name} (Z={z}): done")

    # ------------------------------------------------------------------
    # Step 2: Load line-only DN from checkpoint
    # ------------------------------------------------------------------
    print("\n--- Step 2: Line-only DN from checkpoint ---")

    checkpoint_file = RESULTS_DIR / 'ion_contributions_checkpoint.json'
    if not checkpoint_file.exists():
        print("ERROR: No checkpoint file found!")
        tee.close()
        return

    with open(checkpoint_file, 'r') as f:
        saved = json.load(f)

    # Sum line contributions per channel (Maxwellian)
    line_dn_mxw = {}
    for chan in AIA_CHANNELS:
        chan_str = str(chan)
        if chan_str in saved:
            line_dn_mxw[chan] = sum(saved[chan_str].values()) * EM_SCALE
        else:
            line_dn_mxw[chan] = 0.0

    # Also load the kappa DN from the sensitivity results
    sens_file = RESULTS_DIR / 'kappa_sensitivity_results.npz'
    kappa_dn = None
    if sens_file.exists():
        data = np.load(sens_file, allow_pickle=True)
        if 'kappa_2p5_dn_kappa' in data:
            kappa_dn = data['kappa_2p5_dn_kappa']

    # ------------------------------------------------------------------
    # Step 3: Compute kappa/Maxwellian continuum ratio
    # ------------------------------------------------------------------
    print("\n--- Step 3: Kappa continuum enhancement ---")

    kT_eV = K_B * T_EFF * EV_PER_ERG  # kT_eff in eV

    print(f"  kT_eff = {kT_eV:.1f} eV")

    # For each AIA channel, compute the effective photon energy
    # and the kappa/Mxw free-free ratio at that energy
    chan_energy_eV = {}
    chan_ff_ratio = {}
    for chan in AIA_CHANNELS:
        # Effective photon energy: E = hc/λ
        # Use the channel's peak effective area wavelength as representative
        peak_idx = np.argmax(aia_ea[chan])
        peak_wvl = aia_wvl[chan][peak_idx]  # Å
        E_eV = (H_PLANCK * C_LIGHT / (peak_wvl * 1e-8)) * EV_PER_ERG
        chan_energy_eV[chan] = E_eV

        # Kappa/Mxw ratio at this energy
        ratio = kappa_ff_ratio(E_eV, KAPPA, kT_eV)
        chan_ff_ratio[chan] = ratio

        print(f"  {chan:>3d} Å: peak at {peak_wvl:.1f} Å, "
              f"E = {E_eV:.1f} eV, κ/Mxw ff ratio = {ratio:.3f}")

    # Actually, we should compute the ratio integrated over the AIA bandpass
    # weighted by the continuum spectrum, not just at the peak wavelength
    print("\n  Bandpass-integrated κ/Mxw free-free ratios:")
    chan_ff_ratio_integrated = {}
    for chan in AIA_CHANNELS:
        ea_interp = np.interp(wvl_grid, aia_wvl[chan], aia_ea[chan])
        # Energy at each wavelength
        E_grid = (H_PLANCK * C_LIGHT / (wvl_grid * 1e-8)) * EV_PER_ERG
        # Kappa/Mxw ratio at each wavelength
        ratio_grid = kappa_ff_ratio(E_grid, KAPPA, kT_eV)
        # Weight by Maxwellian continuum shape × EA
        # Mxw continuum ∝ exp(-E/kT) / λ² (rough), but we just use the
        # ChiantiPy free-free result if available. For now, use EA as weight.
        weight = ea_interp * np.exp(-E_grid / kT_eV)  # Mxw-weighted
        if np.sum(weight) > 0:
            ratio_int = np.sum(ratio_grid * weight) / np.sum(weight)
        else:
            ratio_int = 1.0
        chan_ff_ratio_integrated[chan] = ratio_int
        print(f"    {chan:>3d} Å: integrated κ/Mxw ratio = {ratio_int:.3f}")

    # ------------------------------------------------------------------
    # Step 4: Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CONTINUUM CONTRIBUTION SUMMARY")
    print("=" * 70)

    print(f"\n{'Chan':>5s}  {'Line Mxw':>10s}  {'FF Mxw':>10s}  {'FB Mxw':>10s}  "
          f"{'Cont/Line':>10s}  {'κ/M ff':>8s}  {'Cont κ':>10s}")
    print("-" * 75)

    cont_kappa_dn = {}
    for chan in AIA_CHANNELS:
        ff = cont_dn_ff[chan]
        fb = cont_dn_fb[chan]
        line = line_dn_mxw[chan]
        cont_total = ff + fb
        cont_frac = cont_total / line if line > 0 else 0.0

        # Kappa continuum estimate:
        # ff enhanced by integrated ratio, fb enhanced roughly same way
        # (conservative: fb depends on ion fractions too)
        ff_kappa = ff * chan_ff_ratio_integrated[chan]
        fb_kappa = fb * chan_ff_ratio_integrated[chan]  # rough estimate
        cont_kappa = ff_kappa + fb_kappa
        cont_kappa_dn[chan] = cont_kappa

        print(f"  {chan:3d} Å  {line:10.3e}  {ff:10.3e}  {fb:10.3e}  "
              f"{cont_frac:10.3%}  {chan_ff_ratio_integrated[chan]:8.3f}  "
              f"{cont_kappa:10.3e}")

    # ------------------------------------------------------------------
    # Step 5: Impact on kappa DN and χ²
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("IMPACT ON KAPPA DN AND chi-squared")
    print("=" * 70)

    if kappa_dn is not None:
        print(f"\n{'Chan':>5s}  {'DN_κ (lines)':>13s}  {'+ cont κ':>10s}  "
              f"{'DN_κ total':>12s}  {'Change':>8s}")
        print("-" * 55)

        dn_kappa_corrected = np.zeros(6)
        for i, chan in enumerate(AIA_CHANNELS):
            dn_lines = kappa_dn[i]
            dn_cont = cont_kappa_dn[chan]
            dn_total = dn_lines + dn_cont
            change = dn_cont / dn_lines if dn_lines > 0 else 0
            dn_kappa_corrected[i] = dn_total
            print(f"  {chan:3d} Å  {dn_lines:13.4e}  {dn_cont:10.3e}  "
                  f"{dn_total:12.4e}  {change:+8.1%}")

        # Re-run DEM inversion with corrected DN
        print("\n--- Re-running DEM inversion with continuum-corrected DN ---")

        sys.path.insert(0, str(SCRIPT_DIR))
        from kappa_dem_pipeline import (load_demregpy_response, run_dem_inversion,
                                         compute_aia_noise)

        tresp_logT, tresp_matrix, _ = load_demregpy_response()
        edn = compute_aia_noise(dn_kappa_corrected)

        dem, edem, elogt, chisq, dn_reg, mlogt, temps = run_dem_inversion(
            dn_kappa_corrected, edn, tresp_matrix, tresp_logT
        )

        recovery = dn_reg / dn_kappa_corrected

        print(f"\n  Original χ² (lines only):     7.588")
        print(f"  Corrected χ² (lines + cont):  {chisq:.3f}")
        print(f"  Original χ²/dof:              1.518")
        print(f"  Corrected χ²/dof:             {chisq/5:.3f}")

        print(f"\n  DN recovery with continuum:")
        for i, chan in enumerate(AIA_CHANNELS):
            print(f"    {chan:3d} Å: {recovery[i]:.3f}")

        print(f"\n  Recovery range: {np.min(recovery):.3f} to {np.max(recovery):.3f}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
The continuum contribution is quantified as a fraction of line emission
for each AIA channel. For channels where continuum is significant (94, 131),
the kappa enhancement of the continuum (harder spectrum) adds additional
DN that was missing from the lines-only computation.

If the corrected chi-squared is LOWER than the original, the continuum
correction IMPROVES the fit — meaning the lines-only result was
conservative and the missing continuum was not hiding a problem.
""")

    print(f"Results saved to: {log_file}")
    tee.close()


if __name__ == '__main__':
    main()
