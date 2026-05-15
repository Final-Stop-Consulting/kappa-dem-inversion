#!/usr/bin/env python3
"""
Isothermal Maxwellian baseline FWHM through demregpy.

Forward-models a pure isothermal Maxwellian at log T = 6.176 (T_eff = 1.5 MK)
with EM_SCALE = 1e27, no kappa modification. Runs through the same demregpy
pipeline configuration as the single-T kappa and multi-T kappa tests. Reports
recovered FWHM, peak log T, chi^2/dof.

This is the demregpy algorithmic floor: the FWHM a single-T delta-function
input recovers due to the AIA response kernels + GSVD-selected regularization
smoothing, with no underlying physical broadening from a kappa distribution.
All physical broadening above this floor reflects either kappa-driven
charge-state redistribution or multi-T integration along the line of sight.

Used in paper Section 3.4 to anchor the FWHM ladder (Table 4):
    Isothermal Mxw baseline:  0.174  (this script)
    kappa=3 recovered:        0.191
    kappa=2.5 recovered:      0.222
    Brooks-shape Mxw forward: 0.319
    multi-T kappa=2.5:        0.305
    kappa=2 recovered:        0.353
    Real-QS distribution:     0.229-0.383 (median 0.277)

Dependencies: numpy<2.0, scipy, demregpy. Does NOT require ChiantiPy or the
per-ion checkpoint — runs purely off the demregpy-bundled AIA temperature
response functions.

Usage:
    python isothermal_mxw_baseline.py
"""

import os
import sys
import time
import numpy as np
import scipy.io as sio
from pathlib import Path

# Windows: demregpy / ChiantiPy environment fixes
if 'HOME' not in os.environ:
    os.environ['HOME'] = os.path.expanduser('~')

# ---- paths (resolve relative to this script: Analysis/isothermal_mxw_baseline.py)
PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / 'Results'
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# ---- physical constants
LOG_T_EFF = 6.176
EM_SCALE = 1.0e27
AIA_CHANNELS = [94, 131, 171, 193, 211, 335]

DEM_LOG_T_MIN = 5.7
DEM_LOG_T_MAX = 7.2
DEM_DLOGT = 0.05


def compute_aia_noise(dn, exposure=2.9):
    """Shot + read noise (matching kappa_dem_pipeline.compute_aia_noise)."""
    rdnse = np.array([1.14, 1.18, 1.15, 1.20, 1.20, 1.18])
    counts = dn * exposure
    photon = np.sqrt(np.maximum(counts, 0)) / exposure
    return np.sqrt(photon**2 + rdnse**2)


def interp_edge(x0, x1, y0, y1, y):
    """Linear half-max interpolation between two adjacent grid points."""
    if y1 == y0:
        return (x0 + x1) / 2
    return x0 + (y - y0) / (y1 - y0) * (x1 - x0)


def compute_fwhm(log_T_centers, dem):
    """FWHM in log T via linear half-max interpolation on a DEM curve."""
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
             if last+1 < len(log_T_center