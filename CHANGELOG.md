# Changelog

All notable changes to this repository are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions track the GitHub release tag and the associated Zenodo version DOI.

## [2.1.1] — 2026-05-15

Patch release: reference-correctness fixes and completion of release
artifacts that were missing from v2.1.0.

### Added

- `CHANGELOG.md` — this file (intended for v2.1.0 but didn't land).
- `Figures/freebound_131_sensitivity.{png,pdf}` and
  `Figures/em_loci_multi_thermal_kappa.{png,pdf}` — sensitivity-test
  figures (the scripts shipped in v2.1.0, the figures are added here).

### Changed

- `README.md` References section: corrected the Dere et al. 2023 entry
  parenthetical from "(CHIANTI v11)" to "(CHIANTI v10.1)" — the
  *ApJS* 268, 52 paper is CHIANTI Paper XVII / v10.1. Added a new
  reference to **Dufresne et al. 2024, *ApJ* 974, 71** as the actual
  CHIANTI v11 (Paper XVIII) paper. Corrected the **Dzifčáková et al.
  2023** entry: restored Pavelková, J. and Solarová, J. to the author
  list (the previous entry only listed three of the five co-authors)
  and fixed the volume/page from the wrong *ApJS* 268, 52 (copy-paste
  from the Dere entry) to the correct **269, 45**.
- `README.md` Data dependencies section: fixed the Dzifčáková 2023 DOI
  from `10.3847/1538-4365/ac2aa7` (which resolves to KAPPA Paper II,
  Dzifčáková et al. 2021) to `10.3847/1538-4365/ad014d` (the actual
  KAPPA Paper III, 2023).
- `kappa_continuum_estimate.py`: corrected the docstring attribution
  for the free-free κ/Mxw emissivity ratio from "Dudík et al. 2017"
  to "Dudík et al. 2012" — the actual reference is Dudík, Kašparová,
  Dzifčáková, Karlický & Mackovjak 2012, *A&A* 539, A107, "The
  non-Maxwellian continuum in the X-ray, UV, and radio range."
- `CITATION.cff`: bumped `version` from "1.0.0" (errant placeholder)
  to "2.1.1" (matching the actual release tag).
- `Results/*.txt` text outputs refreshed from sandbox verification
  reruns (`continuum_estimate_results.txt`, `sensitivity_results.txt`,
  `ion_contributions_checkpoint.json`).
- `.gitignore` — minor tightening.

## [2.1.0] — 2026-05-15

Submission-grade release accompanying the paper *The Quiet-Sun DEM
Under Kappa: Diagnostic Degeneracy and a Closure-Corrected Coronal
Energy Budget* (V. Edmonds, in prep., *Transport Phenomena*).

### Added

- `freebound_131_sensitivity.py` — DEM-shape stability test under
  inflated 131 Å DN (paper §2.3). Triples the 131 Å DN in the saved
  κ = 2.5 inversion and re-runs `demregpy` to confirm the recovered
  DEM peak, FWHM, and high-T tail are robust to the
  channel-integrated free-bound under-estimate.
- `em_loci_multi_thermal_kappa.py` — multi-thermal κ EM-loci collapse
  compute (paper §3.6). Reports the residual EM-loci spread
  (0.14 dex) for the multi-T κ source built from Brooks 2009 × Dz23
  κ ion fractions, vs the 1.06 dex single-T tilt of Table 6.
- `Results/freebound_131_sensitivity_results.{npz,txt}` and
  `Results/em_loci_multi_thermal_kappa_results.{npz,txt}` — numerical
  outputs from the two new sensitivity scripts.

### Changed

- Path resolution in `kappa_dem_pipeline.py`,
  `kappa_sensitivity_tests.py`, `multi_thermal_kappa_test.py`,
  `isothermal_mxw_baseline.py` (and the XUVTOP setup in
  `kappa_continuum_estimate.py`, `kappa_dem_comparison.py`) now uses
  `Path(__file__).resolve().parent` so the scripts run cleanly from a
  fresh clone (previously assumed a wrapper layout with an `Analysis/`
  subdirectory). `Results/`, `Figures/`, and `Data/` resolve relative
  to the repository root.
- `multi_thermal_kappa_test.py` — module docstring polish;
  internal-process label in log output removed.
- `README.md` — added the two new sensitivity scripts to the script
  table; added the EM-loci collapse bullet to Key results; updated
  the W&N fluid total reference from 1.5 × 10⁵ to 1.6 × 10⁵
  erg cm⁻² s⁻¹ (consistency with paper Table 7).

### Removed

- `COMMIT_MESSAGE.txt` — pre-written commit message no longer relevant.

## [0.1.0] — 2026-03-23

Initial public release accompanying the first arXiv preprint of
Edmonds 2026a.

### Added

- `kappa_dem_pipeline.py` — main 5-stage pipeline (κ = 2.5,
  *T*<sub>core</sub> = 0.6 MK, *T*<sub>eff</sub> = 1.5 MK).
- `kappa_sensitivity_tests.py` — κ = 2, 2.5, 3 + coronal vs.
  photospheric abundance sensitivity sweep.
- `kappa_dem_comparison.py` — DEM shape comparison against published
  QS DEMs.
- `kappa_continuum_estimate.py` — free-free + free-bound continuum
  quantification.
- `multi_thermal_kappa_test.py` — multi-T κ source forward model
  (paper §3.4).
- `isothermal_mxw_baseline.py` — algorithmic-floor measurement
  (paper §3.4 / Table 4).
- `Results/`, `Figures/` — supporting outputs.
- `README.md`, `CITATION.cff`, `LICENSE`, `requirements.txt`,
  `.gitignore`.
