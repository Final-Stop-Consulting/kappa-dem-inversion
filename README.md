# Kappa DEM Inversion Pipeline

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19188096.svg)](https://doi.org/10.5281/zenodo.19188096)

Forward-modeling and DEM inversion code for the paper:

> **The Quiet-Sun DEM Under Kappa: Diagnostic Degeneracy and a Closure-Corrected Coronal Energy Budget**
> V. Edmonds, Final Stop Consulting LLC (2026, in prep.)

## Overview

This pipeline forward-models SDO/AIA EUV observations from a kappa-distributed plasma and passes them through the standard `demregpy` regularized DEM inversion (Hannah & Kontar 2012), testing what the inversion returns when its load-bearing Maxwellian assumption fails. The paper has two arches:

1. **Diagnostic degeneracy at the AIA-imaging level.** By the convergence theorem of [Edmonds 2026b, Open Transport](https://doi.org/10.1515/ot-2026-0011), every electron-temperature diagnostic mediated by collisional ionization equilibrium structurally returns the effective temperature *T*<sub>eff</sub> of the underlying distribution and cannot detect departure from Maxwellian form. We verify this empirically: a single isothermal κ = 2.5 source returns a multi-thermal-looking DEM at χ²/dof = 1.00 with FWHM = 0.222 in log *T*, inside the FWHM distribution returned by the same pipeline from real quiet-Sun AIA observations (median 0.277, range 0.229–0.383 across 80 patches at solar minimum). A multi-thermal κ source (Brooks 2009 shape × κ = 2.5 ion fractions) recovers FWHM = 0.305, near the median of the real-QS distribution. The AIA pipeline does not separate single-T κ, multi-T Maxwellian, or multi-T κ on FWHM alone.

2. **Closure-corrected coronal energy budget.** The Spitzer–Härm conductivity used in the standard quiet-Sun budget evaluates a Maxwellian moment closure at the EUV-derived *T*<sub>eff</sub>, a tail-weighted moment of the κ distribution. The leading-order bulk-temperature correction substitutes *T*<sub>core</sub> = (κ−3/2)/κ · *T*<sub>eff</sub>; at κ = 2.5 the fluid conductive flux reduces by (*T*<sub>eff</sub>/*T*<sub>core</sub>)<sup>7/2</sup> ≈ 25, and the Withbroe & Noyes (1977) fluid total of 3 × 10<sup>5</sup> erg cm<sup>−2</sup> s<sup>−1</sup> reduces to ~1.5 × 10<sup>5</sup>. AW measurements (0.5–5 × 10<sup>5</sup>) span the corrected fluid budget within existing measurement uncertainty. Total-budget AW sufficiency depends on non-local kinetic transport not captured by any fluid closure (the open kinetic-transport problem of Edmonds 2026a §7).

The κ ≈ 2.5 prior comes from [Edmonds 2026a, Open Journal of Astrophysics](https://doi.org/10.33232/001c.161223) (radio–EUV–density triple intersection); the bulk-temperature correction in §4.5 applies the SOL-context derivation of Edmonds 2026b §4.2.5 to the QS case.

## Pipeline stages

1. **Ion fraction loading** — Read κ ion fractions from Dzifčáková et al. (2023) v10.1 tables and matched Maxwellian fractions on the same atomic-data basis.
2. **Per-ion AIA channel contributions** — Compute line intensities for ~165 ions using ChiantiPy at *T*<sub>eff</sub>, fold through AIA effective areas (`aiapy`), weight by κ vs. Maxwellian ion-fraction ratios.
3. **Synthetic DN construction** — Sum per-ion contributions to get κ-predicted DN per AIA channel with shot + read noise (matching the standard AIA noise model).
4. **DEM inversion** — Run `demregpy` (Hannah & Kontar 2012) regularized inversion using standard Maxwellian temperature response functions.
5. **DEM shape comparison** — Compare recovered DEM against the Brooks 2009 quiet-Sun reference DEM and the matching-pipeline real-QS distribution.

## Scripts

| Script | Description |
|---|---|
| `kappa_dem_pipeline.py` | Main 5-stage pipeline (κ = 2.5, *T*<sub>core</sub> = 0.6 MK, *T*<sub>eff</sub> = 1.5 MK). Produces the per-ion checkpoint that downstream scripts reuse. |
| `kappa_sensitivity_tests.py` | Pipeline run for κ = 2, 2.5, 3 and coronal vs. photospheric abundances. Reuses the per-ion checkpoint (κ-independent). |
| `kappa_dem_comparison.py` | DEM shape comparison (peak *T*, FWHM, slopes) of recovered DEMs against published quiet-Sun DEMs. |
| `kappa_continuum_estimate.py` | Free-free + free-bound continuum quantification per AIA channel, including the κ/Maxwellian energy-dependent reversal. |
| `multi_thermal_kappa_test.py` | Multi-thermal κ extension (paper §3.4): forward-models the Brooks 2009 quiet-Sun reference DEM populated by κ = 2.5 ion fractions across the temperature grid, runs through `demregpy`, reports recovered FWHM. Used to verify the matching-pipeline argument against multi-T sources. |

## Key results

- **Single-T κ = 2.5:** χ²/dof = 1.00 (lines + continuum), recovered FWHM = 0.222, peak log *T* = 6.18.
- **Multi-T κ = 2.5 (Brooks shape):** recovered FWHM = 0.305, peak log *T* = 5.98, χ²/dof = 0.76. Inside the real-QS distribution; the κ correction averages out under multi-T integration because each channel samples its dominant ion near formation temperature.
- **Brooks-Maxwellian forward** (no κ): recovered FWHM = 0.319, consistent with the real-QS distribution. Three source families (single-T κ, multi-T Maxwellian, multi-T κ) all land in the same recovered-FWHM band.
- **Fe XI structural crossover:** Fe XI ion-fraction ratio (κ/Mxw) stays within 5% of unity across κ = 2, 2.5, 3 (0.954, 1.040, 0.998) — the ion at which tail-driven ionization and bulk-driven recombination balance. Within-framework prediction: at different *T*<sub>eff</sub> the crossover shifts to a different ion.
- **EUV continuum reversal at AIA wavelengths:** κ free-free is *suppressed* near *E* ≈ k*T*<sub>eff</sub> (94 Å, ratio 0.785 at κ = 2.5) and *enhanced* at lower photon energies (1.005 at 131 Å, 1.802 at 335 Å), contrary to the X-ray-extrapolated expectation that κ continuum is uniformly hardened relative to Maxwellian.
- **Energy budget revision:** Fluid conductive *F*<sub>c</sub> reduces by ~25× under the leading-order bulk-temperature correction; W&N fluid total 3 × 10<sup>5</sup> → ~1.5 × 10<sup>5</sup> erg cm<sup>−2</sup> s<sup>−1</sup>, robust at the ~10% level across the κ ∈ [2, 3] prior. The total-budget impact (including non-local kinetic transport) is the open kinetic-transport problem of Edmonds 2026a §7.

## Within-framework falsifiable predictions

Inherited from Edmonds 2026a §8.4 (independent of AIA-imaging DEM analysis):

- **AR core collapse:** *T*<sub>B</sub>/*T*<sub>H</sub> ≲ 1.5 in active-region cores at meter wavelengths (vs. ~2.4 in the quiet Sun) as collisionality restores thermal equilibrium.
- **Density dependence:** Diagnostic ratio *R* should decrease systematically with electron density.
- **Topological control:** Closed-field regions should show larger *R* than open-field regions at matched density.
- **Fe X 6378 Å forbidden line:** Intensity enhancement consistent with low κ in QS observations.

Specific to the AIA-imaging analysis of this paper:

- **Fe XI crossover at non-QS *T*<sub>eff</sub>:** The crossover ion shifts predictably with *T*<sub>eff</sub>; testable against the Dzifčáková et al. (2023) ion-fraction tables across the coronal temperature range.
- **EUV continuum reversal at AIA wavelengths:** Coronagraphic continuum measurements that isolate free-free emission from line contributions provide a direct observational test.

## Data dependencies (not in this repository)

- **CHIANTI v11.0.2 atomic database** — [chiantidatabase.org](https://www.chiantidatabase.org/). Set the `XUVTOP` environment variable.
- **Kappa ion fraction tables (Dz23 v10.1)** — [Dzifčáková et al. 2023](https://doi.org/10.3847/1538-4365/acd2cf). Place in `Data/kappa_v10.1/` relative to the project root.

## Requirements

```
numpy<2.0
scipy
matplotlib
astropy
ChiantiPy>=0.15
aiapy
demregpy
sunpy
```

Install with:
```bash
pip install "numpy<2.0" scipy matplotlib astropy ChiantiPy aiapy demregpy sunpy
```

> **Note:** `demregpy`'s GSVD implementation requires `numpy < 2.0`.

## Running

```bash
# Full pipeline (slow — ChiantiPy computes per-ion contributions for ~165 ions)
python kappa_dem_pipeline.py

# Sensitivity tests across kappa values (fast — reuses checkpoint)
python kappa_sensitivity_tests.py

# DEM shape comparison
python kappa_dem_comparison.py

# Continuum quantification
python kappa_continuum_estimate.py

# Multi-thermal kappa extension (uses the per-ion T_eff checkpoint)
python multi_thermal_kappa_test.py
```

The main pipeline takes ~15–20 minutes on first run (Stage 2 ion computation). Stage-2 checkpoint is saved to `Results/ion_contributions_checkpoint.json` and reused on subsequent runs and by all downstream scripts.

## Output

- `Results/` — Numerical results (`.npz` checkpoints, `.json` per-ion contributions, `.txt` log summaries)
- `Figures/` — Publication figures (`.pdf`, `.png`)

## References

- Brooks, D. H., Warren, H. P., Williams, D. R., & Watanabe, T. 2009, *ApJ*, 705, 1522
- Dere, K. P., Del Zanna, G., Young, P. R., & Landi, E. 2023, *ApJS*, 268, 52 (CHIANTI v11)
- Dzifčáková, E., Dudík, J., & Zemanová, A. 2023, *ApJS*, 268, 52 (κ ion-fraction tables)
- Edmonds, V. 2026a, *Open Journal of Astrophysics*, 9, doi:10.33232/001c.161223 (κ ≈ 2.5 prior; falsifiable predictions)
- Edmonds, V. 2026b, *Open Transport*, 1, 20260011, doi:10.1515/ot-2026-0011 (convergence theorem; bulk-temperature correction §4.2.5)
- Hannah, I. G., & Kontar, E. P. 2012, *A&A*, 539, A146 (`demregpy`)
- Withbroe, G. L., & Noyes, R. W. 1977, *ARA&A*, 15, 363 (standard QS energy budget)

## License

MIT — see `LICENSE`. Please cite the paper if you use this code.
