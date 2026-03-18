# Sample-size invariant measure of dimensionality

Bias-corrected **participation ratio (PR)** estimators for measuring the dimensionality of neural representation manifolds, as introduced in:

> **Estimating Dimensionality of Neural Representations from Finite Samples**
> Chanwoo Chun\*, Abdulkadir Canatar\*, SueYeon Chung, Daniel Lee
> *ICLR 2026*

---

## Background

Given a neural activation matrix $\Phi \in \mathbb{R}^{P\times Q}$ (P stimuli × Q neurons), the participation ratio


$$ \gamma =\frac{\left(\sum_i \lambda_i \right)^2}{\sum_i \lambda_i^2} $$


is a soft count of the number of nonzero eigenvalues of the stimulus covariance $K=\frac{1}{Q}\Phi\Phi^\top$.  The naive estimator is severely biased downward when P or Q is small — it behaves approximately as a harmonic mean of P, Q, and the true $\gamma$:

$$ \mathbb{E}\left[ \frac{1}{\gamma_{\text{naive}}}\right] \approx \frac{1}{P}+ \frac{1}{Q} +\frac{1}{\gamma} $$

This package provides unbiased estimators that correct for finite P and/or Q by averaging over disjoint index sets.

| Estimator | Corrects | Use when |
|-----------|----------|----------|
| `γ_naive` | nothing | baseline reference |
| `γ_row` | row (stimulus) sampling bias | full neuron access, subsampled stimuli |
| `γ_col` | column (neuron) sampling bias | full stimulus access, subsampled neurons |
| `γ_both` | both row and column bias | subsampled neurons, subsampled stimuli|

An additional `participation_ratio_finite` estimator handles the case where Φ is a submatrix sampled without replacement from a large-but-finite R×C matrix (Appendix A.6 of the paper).

---

## Installation

```bash
pip install dimensionality
```

Or, for development:

```bash
git clone https://github.com/chun-chanwoo/dimensionality.git
cd dimensionality
pip install -e ".[dev]"
```

**Dependencies:** `numpy >= 1.24`, `opt_einsum >= 3.3`.  Python ≥ 3.9.

---

## Quick start

```python
import numpy as np
from dimensionality import participation_ratio

# Phi: P stimuli × Q neurons  (do NOT pre-center)
Phi = np.random.randn(200, 100)

# Default: γ_both (bias-corrected for both row and column subsampling)
gamma = participation_ratio(Phi)
print(gamma)
```

### All four estimators

```python
result = participation_ratio(Phi, return_all=True)
# result['naive'], result['row'], result['col'], result['both']
```

### Two-trial noise correction

When two independent repeat trials are available for the same stimuli and neurons, the cross-trial construction removes additive and multiplicative noise bias:

```python
gamma = participation_ratio(Phi1, Phi2)
```

### Return numerator and denominator separately

```python
result = participation_ratio(Phi, return_parts=True)
# result['both'], result['A'], result['B']

# Combined with return_all:
result = participation_ratio(Phi, return_all=True, return_parts=True)
# result['naive'], result['A_naive'], result['B_naive'], ...
```

### Neuron dimensionality

To estimate dimensionality along the neuron axis (centering across stimuli), transpose the matrix:

```python
gamma_neuron = participation_ratio(Phi.T)
```

---

## Finite underlying matrix

When Φ is a P×Q submatrix sampled without replacement from a finite R×C population matrix, use `participation_ratio_finite`:

```python
from dimensionality import participation_ratio_finite

gamma = participation_ratio_finite(Phi, R=5000, C=2000)

# With noise correction:
gamma = participation_ratio_finite(Phi1, R=5000, C=2000, Phi2=Phi2)

# Also return the naive estimate:
result = participation_ratio_finite(Phi, R=5000, C=2000, return_naive=True)
# result['gamma'], result['naive']
```

---

## Subsampling sweep

To assess how the estimate converges with sample size, sweep over P or Q:

```python
from dimensionality import sweep_dimensionality, plot_sweep

# Sweep over number of stimuli; keep all neurons
result = sweep_dimensionality(Phi, axis='P', n_trials=20)

# result['values']  — array of P values used
# result['naive'], result['row'], result['col'], result['both']  — mean estimates
# result['both_sem']  — standard error of the mean

fig, ax = plot_sweep(result, true_d=50)
```

To sweep over number of neurons instead:

```python
result = sweep_dimensionality(Phi, axis='Q')
```

For the finite estimator:

```python
result = sweep_dimensionality(Phi, axis='P', estimator='finite', R=5000, C=2000)
# result['naive'], result['gamma']
```

---

## Important: do not pre-center

The bias corrections rely on an algebraic three-term centering structure built into the estimator formulas.  Subtracting column means from Φ before passing it to the estimator introduces statistical dependencies between rows that break the bias correction.  **Pass the raw activation matrix directly.**

---

## API reference

### `participation_ratio(Phi, Phi2=None, *, return_all=False, return_parts=False)`

Estimate the task dimensionality (PR of the centered covariance) of Φ.

- **Phi** — raw activation matrix, shape (P, Q); P ≥ 4, Q ≥ 2.
- **Phi2** — optional second trial for noise correction.
- **return_all** — if `True`, return dict with all four estimator variants.
- **return_parts** — if `True`, include numerator A and denominator B.

Returns a scalar (`γ_both`) by default, or a dict when either flag is set.

---

### `participation_ratio_finite(Phi, R, C, Phi2=None, *, return_naive=False, return_parts=False)`

Estimate the PR of the full R×C matrix from the observed P×Q submatrix.

- **R, C** — number of rows/columns in the full underlying matrix; R ≥ P, C ≥ Q.
- **return_naive** — if `True`, also return the (uncorrected) naive estimate.
- **return_parts** — if `True`, include numerator A and denominator B.

Returns a scalar by default, or a dict when either flag is set.

---

### `sweep_dimensionality(Phi, axis='P', values=None, n_trials=20, Phi2=None, estimator='infinite', R=None, C=None, ...)`

Run a subsampling sweep.  Returns a dict with mean estimates and SEMs at each value.  See docstring for full parameter list.

---

### `plot_sweep(result, ax=None, true_d=None, title=None, figsize=(5, 4))`

Plot the output of `sweep_dimensionality`.  Returns `(fig, ax)`.

---

## Repository structure

```
src/
  dimensionality/
    __init__.py        # public API
    _core.py           # quartic einsum helper
    estimators.py      # participation_ratio
    finite.py          # participation_ratio_finite
    sweep.py           # sweep_dimensionality
    plot.py            # plot_sweep
tests/
  test_estimators.py
examples/
  demo.ipynb           # interactive walkthrough on synthetic data
  synthetic.py         # Figure 1 reproduction
```

---

## Citation

If you use this package, please cite:

```bibtex
@inproceedings{chun2026estimating,
  title     = {Estimating Dimensionality of Neural Representations from Finite Samples},
  author    = {Chun, Chanwoo and Canatar, Abdulkadir and Chung, SueYeon and Lee, Daniel},
  booktitle = {International Conference on Learning Representations},
  year      = {2026},
}
```

---

## License

MIT
