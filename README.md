# Kappa DEM Inversion Pipeline

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19188096.svg)](https://doi.org/10.5281/zenodo.19188096)

Forward-modeling and DEM inversion code for the paper:

> **The Quiet-Sun DEM Under Kappa: Diagnostic Degeneracy and a Closure-Corrected Coronal Energy Budget**
> V. Edmonds, Final Stop Consulting LLC (2026, in prep.)

## Overview

This pipeline forward-models SDO/AIA EUV observations from a kappa-distributed plasma and passes them through the standard `demregpy` regularized DEM inversion (Hannah & Kontar 2012), testing what the inversion returns when its load-bearing Maxwellian assumption fails. The paper has two arches:

1. **Diagnostic degeneracy at the AIA-imaging level.** By the convergence theorem of [Edmonds 2026b, Open Transport](https://doi.org/10.1515/ot-2026-0011), every electron-temperature diagnostic mediated by collisional ionization equilibrium structurally returns the effective temperature *T*<sub>eff</sub> of the underlying distribution and cannot detect departure from Maxwellian form. We verify this empirically: a single isothermal κ = 2.5 source returns a multi-thermal-looking DEM at χ²/dof = 1.00 with FWHM = 0.222 in log *T*, just below the narrow edge of the FWHM distribution returned by the same pipeline from real quiet-Sun AIA observations (median 0.283, range 0.230–0.401 across 80 patches at solar minimum; cross-date stable to −0.0025 dex). A multi-thermal κ source (Brooks 2009 shape × κ = 2.5 ion fractions) recovers FWHM = 0.305, comfortably inside the real-QS distribution near its median. The AIA pipeline does not separate single-T κ, multi-T Maxwellian, or multi-T κ on FWHM alone.

2. **Closure-corrected coronal energy budget.** The Spitzer–Härm conductivity used in the standard quiet-Sun budget evaluates a Maxwellian moment closure at the EUV-derived *T*<sub>eff</sub>, a tail-weighted moment of the κ distribution. The leading-order bulk-temperature correction substitutes *T*<sub>core</sub> = (κ−3/2)/κ · *T*<sub>eff</sub>; at κ = 2.5 the fluid conductive flux reduces by (*T*<sub>eff</sub>/*T*<sub>core</sub>)<sup>7/2</sup> ≈ 25, and the Withbroe & Noyes (1977) fluid total of 3 × 10<sup>5</sup> erg cm<sup>−2</sup> s<sup>−1</sup> reduces to ~1.6 × 10<sup>5</sup>, robust at the ~10% level across the κ ∈ [2, 3] prior. AW measurements (0.5–5 × 10<sup>5</sup>) span the corrected fluid budget within existing measurement uncertainty. Total-budget AW sufficiency depends on non-local kinetic transport not captured by any fluid closure (the open kinetic-transport problem of Edmonds 2026a §7).

The κ ≈ 2.5 prior comes from [Edmonds 2026a, Open Journal of Astrophysics](https://doi.org/10.33232/001c.161223) (radio–EUV–density triple intersection); the bulk-temperature correction in §4.5 applies the SOL-context derivation of Edmonds 2026b §4.2.5 to the QS case.

## For reviewers — verification map

The repository is structured so that each numerical claim in the paper can be independently verified. The four most-testable claims (algorithmic FWHM floor, 131 Å free-bound shape stability, multi-T κ EM-loci collapse, multi-T κ FWHM) require only `numpy<2.0`, `scipy`, and `demregpy` from PyPI plus committed checkpoint files — no atomic-database installation needed. The full 5-stage Stage-2 pipeline (which produces the per-ion contributions) requires CHIANTI v11 installed but its checkpoint is committed so downstream verification does not require re-running it.

| Paper claim | Section | Script | Result file | Dependencies | Time |
|---|---|---|---|---|---|
| Algorithmic FWHM floor 0.174 | §3.4, Table 4 | `isothermal_mxw_baseline.py` | `Results/isothermal_mxw_baseline_results.txt` | numpy<2 + scipy + demregpy | ~10 s |
| 131 Å DN-shape stability under 3× inflation | §2.3 | `freebound_131_sensitivity.py` | `Results/freebound_131_sensitivity_results.npz` | numpy<2 + scipy + demregpy + committed κ=2.5 DN vector | ~10 s |
| Multi-T κ EM-loci collapse 0.14 dex (14% of single-T tilt) | §3.6 | `em_loci_multi_thermal_kappa.py` | `Results/em_loci_multi_thermal_kappa_results.npz` | numpy + Dz23 ion-fraction tables | ~5 s |
| Multi-T κ recovered FWHM 0.305 | §3.4 | `multi_thermal_kappa_test.py` | `Results/multi_thermal_kappa_results.npz` | numpy<2 + scipy + demregpy + Dz23 tables + committed per-ion checkpoint | ~30 s |
| Single-T κ recovered FWHM 0.222, χ²/dof = 1.00 | §3.3 | `kappa_dem_pipeline.py` | `Results/kappa_dem_inversion_results.npz` | full pipeline: ChiantiPy + CHIANTI v11 + Dz23 tables | ~15–20 min first run |
| κ sensitivity sweep (κ = 2, 2.5, 3) | §3.4 | `kappa_sensitivity_tests.py` | `Results/kappa_sensitivity_results.npz` | reuses per-ion checkpoint (committed) | ~1 min |
| Continuum κ/Mxw ratios + 131 Å free-bound | §2.3 | `kappa_continuum_estimate.py` | `Results/continuum_estimate_results.txt` | ChiantiPy + CHIANTI v11 | ~2 min |
| DEM shape vs Brooks 2009 + published QS DEMs | §3.5 | `kappa_dem_comparison.py` | `Results/dem_comparison_results.txt` | numpy + matplotlib | ~30 s |

The pipeline is implemented entirely in Python; no IDL or SolarSoft installation is required.

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
| `multi_thermal_kappa_test.py` | Multi-thermal κ extension (paper §3.4): forward-models the Brooks 2009 reference DEM populated by κ = 2.5 ion fractions across the temperature grid through `demregpy`. Reuses the per-ion checkpoint; reports recovered FWHM, peak *T*, χ²/dof. |
| `isothermal_mxw_baseline.py` | Algorithmic-floor measurement (paper §3.4 / Table 4): runs a synthetic isothermal Maxwellian DEM at log *T* = 6.176 through `demregpy` to measure the irreducible FWHM from AIA response kernels + GSVD smoothing alone. Does **not** require ChiantiPy or the per-ion checkpoint — runs purely off the `demregpy`-bundled response. |
| `freebound_131_sensitivity.py` | DEM shape stability test (paper §2.3): triples the 131 Å DN in the saved κ = 2.5 inversion to simulate a fully resolved per-ion free-bound treatment, then re-runs `demregpy`. Verifies the recovered DEM shape (peak, FWHM, secondary structure) is robust to the channel-integrated free-bound under-estimate. Requires only `Results/kappa_dem_inversion_results.npz`. |
| `em_loci_multi_thermal_kappa.py` | Multi-thermal κ EM-loci collapse (paper §3.6): computes the EM-required-per-line spread when the multi-T κ source replaces the single-T idealization, using the Dz23 ion-fraction tables and the Brooks 2009 DEM. Reports collapse from 1.06 dex (single-T tilt of Table 6) to 0.14 dex (multi-T residual, 14% of single-T amplitude). No ChiantiPy or `demregpy` required. |

## Key results

| Source | Recovered FWHM (log *T*) | Peak log *T* | χ²/dof |
|---|---:|---:|---:|
| Isothermal Maxwellian (algorithmic floor) | 0.174 | 6.15 | — |
| κ = 3 single-T | 0.191 | 6.18 | 1.02 |
| κ = 2.5 single-T (lines + continuum) | **0.222** | 6.18 | **1.00** |
| Multi-T κ = 2.5 (Brooks shape) | 0.305 | 5.98 | 0.17 |
| Brooks-shape Maxwellian (multi-T forward) | 0.319 | 5.95 | — |
| κ = 2 single-T | 0.353 | 6.03 | 3.49 |
| Real-QS distribution (80 patches, 2019-12-01, rebuilt pipeline) | 0.230–0.401 (median 0.283) | 5.975 (median) | 0.998 (median) |
| Real-QS cross-date check (80 patches, 2020-05-15) | 0.225–0.412 (median 0.281; shift −0.0025 dex) | 5.975 (median) | 1.003 (median) |

- **Convergence-theorem verification.** Three source families (single-T κ, multi-T Maxwellian, multi-T κ) all land in the same recovered-FWHM band as the real-QS pipeline-output distribution. The κ correction averages out under multi-T integration because each AIA channel samples its dominant ion near formation temperature, where the κ/Maxwellian ion-fraction ratio is close to unity.
- **Fe XI structural crossover.** Fe XI ion-fraction ratio (κ/Mxw) stays within 5% of unity across κ = 2, 2.5, 3 (0.954, 1.040, 0.998) — the ion at which tail-driven ionization and bulk-driven recombination balance. A second crossover sits between Fe XV and Fe XVI where the κ tail drives ionization further than a Maxwellian-at-*T*<sub>eff</sub> would; the resulting "U-shape" across Fe VIII–XVII is the κ signature in iron charge states.
- **EUV continuum reversal at AIA wavelengths.** κ free-free is *suppressed* near *E* ≈ k*T*<sub>eff</sub> (94 Å, ratio 0.785 at κ = 2.5) and *enhanced* at lower photon energies (1.005 at 131 Å, 1.802 at 335 Å), contrary to the X-ray-extrapolated expectation that κ continuum is uniformly hardened relative to Maxwellian.
- **EM-loci collapse under multi-thermal κ.** A multi-thermal κ source (Brooks 2009 DEM shape × Dz23 κ ion fractions) collapses the 1.06 dex single-T EM-loci tilt across Fe IX–XVI to a 0.14 dex residual (14% of single-T amplitude), below the ~0.2 dex scatter typical of multi-thermal QS EIS analyses. The single-T tilt is the predicted signature of κ ion-fraction reweighting against multi-thermal radiances, not a model of the corona.
- **Energy budget revision.** Fluid conductive *F*<sub>c</sub> reduces by ~25× under the leading-order bulk-temperature correction (*T*<sub>eff</sub>/*T*<sub>core</sub>)<sup>7/2</sup>; W&N fluid total 3 × 10<sup>5</sup> → ~1.6 × 10<sup>5</sup> erg cm<sup>−2</sup> s<sup>−1</sup>, robust at the ~10% level across the κ ∈ [2, 3] prior. AW measurements (0.5–5 × 10<sup>5</sup>) span the corrected fluid budget within existing measurement uncertainty. The total-budget impact (including non-local kinetic transport, which carries ~40% of the κ = 2.5 distribution's energy above the Shoub v⁴ collisionless threshold) is the open kinetic-transport problem of Edmonds 2026a §7.

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
- **Kappa ion fraction tables (Dz23 v10.1)** — [Dzifčáková et al. 2023, *ApJS* 269, 45](https://doi.org/10.3847/1538-4365/ad014d) (KAPPA package paper III). Place in `Data/kappa_v10.1/` relative to the project root.

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

The scripts are designed to be run from a clone of this repository on a machine with the data dependencies (CHIANTI v11 atomic database, Dz23 v10.1 kappa ion-fraction tables) installed alongside. Paths are resolved relative to each script's own location.

```bash
# Full pipeline (slow — ChiantiPy computes per-ion contributions for ~165 ions, ~15-20 min first run)
python kappa_dem_pipeline.py

# Sensitivity tests across kappa values (fast — reuses checkpoint from kappa_dem_pipeline.py)
python kappa_sensitivity_tests.py

# DEM shape comparison
python kappa_dem_comparison.py

# Continuum quantification
python kappa_continuum_estimate.py

# Multi-thermal kappa extension (uses the per-ion T_eff checkpoint)
python multi_thermal_kappa_test.py

# Algorithmic-floor baseline (no checkpoint needed — runs purely off demregpy bundled response)
python isothermal_mxw_baseline.py

# Sensitivity tests (read existing Results/ outputs; fast)
python freebound_131_sensitivity.py
python em_loci_multi_thermal_kappa.py
```

The main pipeline takes ~15–20 minutes on first run (Stage 2 ion computation via ChiantiPy). The Stage-2 checkpoint is saved to `Results/ion_contributions_checkpoint.json` and reused on subsequent runs and by all downstream scripts.

**Sensitivity scripts — minimal dependencies.**
- `isothermal_mxw_baseline.py` requires only `numpy<2.0`, `scipy`, and `demregpy` — no ChiantiPy, no CHIANTI database, no kappa tables.
- `freebound_131_sensitivity.py` requires `numpy<2.0`, `scipy`, `demregpy` and `Results/kappa_dem_inversion_results.npz` (from the main pipeline run, committed to this repo).
- `em_loci_multi_thermal_kappa.py` requires only `numpy` and the Dz23 kappa ion-fraction tables + Brooks 2009 DEM file (both data dependencies — see Data dependencies above). No ChiantiPy or `demregpy`.
- `multi_thermal_kappa_test.py` requires `Results/ion_contributions_checkpoint.json` (committed) and the Dz23 kappa ion-fraction tables.

**Cross-platform notes.** Scripts set `HOME` from `~` on import to handle Windows ChiantiPy compatibility. All paths use `pathlib.Path` and are resolved via `Path(__file__).resolve().parent.parent` for project-root awareness; no absolute paths are hardcoded.

## Output

- `Results/` — Numerical results (`.npz` checkpoints, `.json` per-ion contributions, `.txt` log summaries)
- `Figures/` — Publication figures (`.pdf`, `.png`)

## References

- Brooks, D. H., Warren, H. P., Williams, D. R., & Watanabe, T. 2009, *ApJ*, 705, 1522
- Dere, K. P., Del Zanna, G., Young, P. R., & Landi, E. 2023, *ApJS*, 268, 52 (CHIANTI v10.1)
- Dufresne, R. P., Del Zanna, G., Young, P. R., et al. 2024, *ApJ*, 974, 71 (CHIANTI v11)
- Dzifčáková, E., Dudík, J., Pavelková, J., Solarová, J., & Zemanová, A. 2023, *ApJS*, 269, 45 (κ ion-fraction tables)
- Edmonds, V. 2026a, *Open Journal of Astrophysics*, 9, doi:10.33232/001c.161223 (κ ≈ 2.5 prior; falsifiable predictions)
- Edmonds, V. 2026b, *Open Transport*, 1, 20260011, doi:10.1515/ot-2026-0011 (convergence theorem; bulk-temperature correction §4.2.5)
- Hannah, I. G., & Kontar, E. P. 2012, *A&A*, 539, A146 (`demregpy`)
- Withbroe, G. L., & Noyes, R. W. 1977, *ARA&A*, 15, 363 (standard QS energy budget)

## License

MIT — see `LICENSE`. Please cite the paper if you use this code.


## Revision analysis (TP-1080 referee response) — locked results

All computations use the published inversion footing (demregpy `aia_tresp_en.dat` full 101-point response, temps `10**arange(5.7, 7.25, 0.05)`, shot+read `compute_aia_noise`). Footing gates: lines-only χ² = 7.588 and continuum-corrected χ² = 5.020 reproduce the published values exactly. Exact reproduction of the canonical free-bound numbers requires ChiantiPy==0.15.2 (the submission-baseline version); ChiantiPy 0.16.0 reproduces the documented robustness band (131 Å spectral 1.48–1.58) with both validation gates passing.

| Script | Output (Results/) | Locked result |
|---|---|---|
| `freebound_perion_kappa.py` | `freebound_perion_kappa.{json,npz}`, `freebound_perion_records.json` | Per-ion/per-level κ-consistent free-bound (Milne reweighting at E_e = E_photon − I_edge). 131 Å FB κ/Mxw: 1.00 (draft free-free proxy) → 1.48 (spectral; ChiantiPy 0.15.2, the submission-baseline version — canonical) to 1.58 (ChiantiPy 0.16.0; version band, gates pass on both). Population folding: ≈1.49 on 0.15.2; 3.64 on 0.16.0 with f_mxw ≥ 1e−3 (Mg XII trace-ion artifact excluded; 0.15.2 never computes these bare/H-like ions). |
| `fb_lock_published_footing.py` | `fb_lock_published_footing.{json,txt}` | Corrected-DN re-inversion: χ²/dof 1.004 → 0.994 (spectral and floored-full; ChiantiPy 0.15.2 canonical — 0.999/1.004 on 0.16.0); peak log T 6.175 and FWHM 0.218 unchanged; no secondary artifact. |
| `lambda_sensitivity_sweep.py` | `lambda_sweep_results.{json,txt}` | reg_tweak {0.5, 1, 2}× across 4 source families: max ΔFWHM ≤ 0.009 dex; peak log T invariant. |
| `patch80_download.py` / `patch80_register.py` / `patch80_dem.py` | `Results/patch80/` | Rebuilt 80-patch real-QS pipeline. 2019-12-01: median FWHM 0.283, range 0.230–0.401, peak log T 5.975, 171/193 median 2.10. 2020-05-15: median 0.281. Cross-date shift −0.0025 dex (stable). |
| `backreaction_bound.py` | `backreaction_bound.json` | Static-loop self-consistency: at the observed structure the (T_eff/T_core)^{7/2} correction is exactly multiplicative (47% fluid-total reduction); the floating-apex counterfactual requires apex T_eff = 2.56 MK (observationally excluded) and still floors the reduction at 32%. |
| `refdem_robustness.py` | `refdem_results.{json,txt}` + `Data/reference_dems/*.dem` | Reference-DEM robustness across six QS DEMs: Brooks 2009 gate PASS (0.322/0.307); Vernazza & Reeves 1978 via CHIANTI (0.387/0.378); Raymond & Doyle 1981 (ApJ 247, 686, Fig. 3; = Del Zanna & Mason 2018 Fig. 57), tilt-converted per RD81 Eq. 1 (0.412/0.399; raw-axis bound 0.522/0.409; digitization validated against the primary figure and their printed Table 1 power law, rise slope 1.58 vs b = 1.65 ± 0.15); Warren & Brooks 2009 Gaussian, analytic from printed params (0.338/0.320); Landi & Feldman 2008 near-isothermal, analytic from Table 3 (0.175/0.262). Broad references: κ vs Mxw indistinguishable (Δ ≤ 0.018 dex) and in the real-QS band — the degeneracy is reference-independent. Near-isothermal reference: Mxw recovers the algorithmic floor while κ is lifted into the observed band (the single-T-probe behavior from a published curve). |

`Results/fwhm_floor_corrected.json` carries the R3.3 quadrature floor subtraction (√(FWHM² − 0.174²)).
