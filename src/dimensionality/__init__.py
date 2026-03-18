"""
dimensionality
==============

Bias-corrected participation ratio estimators for measuring the global
(and local) dimensionality of neural representation manifolds.

Reference
---------
Chun, Canatar, Chung, Lee (2026).
"Estimating Dimensionality of Neural Representations from Finite Samples."
ICLR 2026.

Quick start
-----------
>>> import numpy as np
>>> from dimensionality import participation_ratio

>>> Phi = np.random.randn(200, 100)       # 200 stimuli × 100 neurons
>>> gamma = participation_ratio(Phi)      # γ_both  (bias-corrected)

Two-trial noise correction::

    Phi1 = np.random.randn(200, 100)
    Phi2 = np.random.randn(200, 100)      # same stimuli & neurons, repeat trial
    gamma = participation_ratio(Phi1, Phi2)

All four estimator variants::

    result = participation_ratio(Phi, return_all=True)
    # result['naive'], result['row'], result['col'], result['both']

Finite underlying matrix::

    from dimensionality import participation_ratio_finite
    gamma = participation_ratio_finite(Phi, R=5000, C=2000)

Subsampling sweep::

    from dimensionality import sweep_dimensionality, plot_sweep
    result = sweep_dimensionality(Phi, axis='P')   # vary P, fix Q = all columns
    fig, ax = plot_sweep(result, true_d=50)
"""

from .estimators import participation_ratio
from .finite import participation_ratio_finite
from .sweep import sweep_dimensionality
from .plot import plot_sweep

__all__ = [
    "participation_ratio",
    "participation_ratio_finite",
    "sweep_dimensionality",
    "plot_sweep",
]
__version__ = "0.1.0"
