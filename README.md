# Kappa DEM Inversion Pipeline

Analysis code for testing whether SDO/AIA DEM inversions can detect non-Maxwellian (kappa) electron velocity distributions in the solar corona.

**Paper:** *Can standard DEM analysis detect kappa distributions? A forward-modeling test with SDO/AIA* (Edmonds, in prep.)

## Overview

This pipeline forward-models AIA EUV observations from a single kappa-distributed plasma and passes them through a standard DEM inversion to test whether the non-Maxwellian signature is detectable. The core finding is that it is not — the DEM inversion absorbs the kappa signature into a plausible multi-thermal DEM shape indistinguishable from published quiet Sun results.

### Pipeline stages

1. **Ion fraction loading** — Read kappa ion fractions from Dzifčáková et al. (2023) tables and Maxwellian fractions from the CHIANTI ionization equilibrium.
2. **Per-ion AIA channel contributions** — Compute line intensities for 280 ions using ChiantiPy at T_eff, fold through AIA effective areas (aiapy), weight by kappa vs. Maxwellian ion fraction ratios.
3. **Synthetic DN construction** — Sum per-ion contributions to get kappa-predicted DN per AIA channel with realistic Poisson + read noise.
4. **DEM inversion** — Run demregpy (Hannah & Kontar 2012) regularized inversion using standard Maxwellian temperature response functions.
5. **DEM shape comparison** — Compare recovered DEM against published quiet Sun DEMs from the CHIANTI database.

## Scripts

| Script | Description |
|--------|-------------|
| `kappa_dem_pipeline.py` | Main 5-stage pipeline. Produces synthetic kappa DN and DEM inversion for κ=2.5 (T_core=0.6 MK, T_eff=1.5 MK). |
| `kappa_sensitivity_tests.py` | Runs pipeline for κ=2, 2.5, 3 and compares coronal vs. photospheric abundances. |
| `kappa_dem_comparison.py` | Stage 5: shape comparison of recovered DEMs against published quiet Sun DEMs (Brooks+2009, Vernazza & Reeves 1978, Dupree+1973). |
| `kappa_continuum_estimate.py` | Quantifies free-free + free-bound continuum contributions to each AIA channel, including kappa/Maxwellian enhancement ratios. |

## Key results

- **κ=2.5 DEM inversion:** χ²/dof = 1.00 (with continuum correction), DN recovery within 3–37% across all 6 channels.
- **DEM width match:** Recovered FWHM = 0.222 in log T, matching Brooks et al. (2009) to within 1%.
- **Continuum reversal:** Kappa free-free emission is *suppressed* at E > kT_eff (94 Å) but *enhanced* at E < kT_eff (335 Å) — opposite to the common assumption.
- **131 Å most sensitive:** 23% continuum fraction + Fe VIII line excess makes it the most kappa-sensitive channel, but it still doesn't break the overall fit.

## Data dependencies (not included)

These are too large for the repository and must be obtained separately:

- **CHIANTI v11.0.2 atomic database** — Available from [CHIANTI](https://www.chiantidatabase.org/). Set the `XUVTOP` environment variable or edit the path in `kappa_dem_pipeline.py`.
- **Kappa ion fraction tables (v10.1)** — From [Dzifčáková et al. (2023)](https://doi.org/10.3847/1538-4365/acd2cf). Place in a `kappa_v10.1/` directory and edit `KAPPA_DIR` in `kappa_dem_pipeline.py`.

Both paths are configured near the top of `kappa_dem_pipeline.py` (lines 71–77) and can be adjusted for your system.

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

> **Note:** demregpy's GSVD implementation requires numpy < 2.0 due to a removed `np.int` alias.

## Running

```bash
# Full pipeline (slow — computes all 280 ions via ChiantiPy)
python kappa_dem_pipeline.py

# Sensitivity tests across kappa values
python kappa_sensitivity_tests.py

# DEM shape comparison (requires sensitivity results)
python kappa_dem_comparison.py

# Continuum quantification
python kappa_continuum_estimate.py
```

The main pipeline takes ~15–20 minutes on first run (Stage 2 ion computation). Results are checkpointed to `results/ion_contributions_checkpoint.json` and reused on subsequent runs.

## Output

- `Results/` — Numerical results (.npz, .json, .txt)
- `Figures/` — Publication-quality figures (.pdf, .png)

## References

- Dzifčáková, E., Dudík, J., & Zemanová, A. 2023, ApJS, 268, 52 (kappa ion fractions)
- Hannah, I. G. & Kontar, E. P. 2012, A&A, 539, A146 (demregpy)
- Brooks, D. H., Warren, H. P., Williams, D. R., & Watanabe, T. 2009, ApJ, 705, 1522
- Dere, K. P. et al. 2023, ApJS, 268, 52 (CHIANTI v11)

## License

This code is provided for reproducibility of the associated publication. Please cite the paper if you use this code.
