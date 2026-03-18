# Dimensionality Estimator

Python research repository implementing a bias-corrected participation ratio (PR) estimator for measuring global (and local) dimensionality of neural representation manifolds. Based on the ICLR 2026 paper:

> **Estimating Dimensionality of Neural Representations from Finite Samples**
> Chanwoo Chun\*, Abdulkadir Canatar\*, SueYeon Chung, Daniel Lee

---

## Background

The **participation ratio (PR)** γ is a soft count of nonzero eigenvalues of a covariance matrix K = (1/Q) ΦΦᵀ, where Φ ∈ ℝ^{P×Q} is the neural activation matrix (P stimuli × Q neurons):

```
γ = (Σᵢ λᵢ)² / Σᵢ λᵢ²   =   A / B
```

The naive estimator γ_naive is heavily biased when P or Q is small. The key insight: bias arises from **overlapping indices** in the sums for A and B. The fix is to average over **disjoint/unequal indices** only.

### Estimators

| Estimator    | Corrects         | Use when                                      |
|--------------|------------------|-----------------------------------------------|
| `γ_naive`    | nothing          | baseline comparison                           |
| `γ_row`      | row (stimulus) sampling bias   | full neuron access, sampled stimuli |
| `γ_col`      | column (neuron) sampling bias  | full stimulus access, sampled neurons |
| `γ_both`     | both row and column bias       | general case (recommended)          |

### Extensions

- **Noise correction**: pass two trial matrices Φ⁽¹⁾, Φ⁽²⁾; redefine v^{αβ}_{ijkl} ← Φ⁽¹⁾_{iα} Φ⁽²⁾_{jα} Φ⁽¹⁾_{kβ} Φ⁽²⁾_{lβ}. Eliminates additive/multiplicative noise bias.
- **Importance sampling**: weight samples by r(x) = ρ_X(x)/ρ_X^obs(x) and c(w) = ρ_W(w)/ρ_W^obs(w) to correct for biased sampling distributions.
- **Sparse matrices**: skip summands that include any missing entry.
- **Finite underlying matrix**: when sampling P rows/cols from a finite R×C matrix (without replacement), use the corrected estimators that require knowledge of R and C.
- **Local dimensionality**: weight samples by proximity (Mahalanobis distance with local metric), take average over all center points. Noise-resistant unlike TwoNN.

### Scaling law of γ_naive

Under uniform row/column norms:
```
E[1/γ_naive] ≈ 1/P + 1/Q + 1/γ
```
γ_naive is approximately a harmonic mean of P, Q, and γ — like parallel resistance.

---

## Implementation Notes

- Core computation uses `opt_einsum` (no JAX). Disjoint-index sums are re-expressed as linear combinations of regular sums to enable vectorization. See Sec. A.3 of the paper for the full expansions.
- **Centering is algebraic**: the three-term formula structure encodes centering (e.g. A = ⟨v_iijj⟩ − 2⟨v_iijl⟩ + ⟨v_ijlr⟩). Do **not** pre-subtract column means before calling the estimators — doing so introduces statistical dependencies that break the bias correction.
- Task dimensionality (default): pass Φ directly (rows = stimuli, columns = neurons).
- Neuron dimensionality: pass Φ.T instead.
- The ratio A/B introduces a small, unavoidable O((1/P + 1/Q)²) bias even when A and B are individually unbiased — this is negligible in practice.

---

## Package API

```python
from dimensionality import participation_ratio, participation_ratio_finite

# Single-trial, default output (γ_both, bias-corrected for both axes)
gamma = participation_ratio(Phi)                         # float

# Two-trial noise correction
gamma = participation_ratio(Phi1, Phi2)                  # float

# Return all four estimators
result = participation_ratio(Phi, return_all=True)
# result['naive'], result['row'], result['col'], result['both']

# Return numerator A and denominator B
result = participation_ratio(Phi, return_parts=True)
# result['both'], result['A'], result['B']

# Both options at once — adds A_naive/B_naive, A_row/B_row, etc.
result = participation_ratio(Phi, return_all=True, return_parts=True)

# Finite underlying matrix (R×C), submatrix Φ is P×Q
gamma = participation_ratio_finite(Phi, R=5000, C=2000)  # float
gamma = participation_ratio_finite(Phi1, R=5000, C=2000, Phi2=Phi2)  # two-trial
result = participation_ratio_finite(Phi, R=5000, C=2000, return_parts=True)
# result['gamma'], result['A'], result['B']
```

Minimum sizes: `participation_ratio` requires P ≥ 4, Q ≥ 2. `participation_ratio_finite` requires the same plus R ≥ P, C ≥ Q.

---

## Repository Structure

```
src/
  dimensionality/
    __init__.py        # exports participation_ratio, participation_ratio_finite
    _core.py           # _gett_all(pattern, A, B) — quartic einsum helper
    estimators.py      # participation_ratio (infinite underlying matrix)
    finite.py          # participation_ratio_finite (finite R×C underlying matrix)
tests/
  test_estimators.py   # API tests, bias ordering, noise correction, finite estimator
examples/
  synthetic.py         # Figure 1 reproduction (vary P or Q, all four estimators)
legacy/                # Old exploratory notebooks — ignore unless explicitly referenced
  biology/             #   Brain data experiments (Stringer, MajajHong, TVSD)
  intrinsic/           #   Local dimensionality experiments
  synthetic/           #   Synthetic data experiments
```

The `legacy/` folder contains the original research notebooks used to produce figures in the paper. **Do not modify or base new code on legacy/ unless explicitly asked.**

To run examples:
```bash
python examples/synthetic.py            # saves examples/figure1.png
python examples/synthetic.py --show     # also opens interactive window
```

To run tests:
```bash
pytest tests/
```

---

## Key Symbols

| Symbol | Meaning |
|--------|---------|
| Φ ∈ ℝ^{P×Q} | Sample activation matrix (P stimuli, Q neurons) |
| Φ^{(∞)} | True infinite underlying matrix |
| K = (1/Q)ΦΦᵀ | Sample covariance matrix |
| γ | True participation ratio (dimensionality) |
| γ_naive | Naive estimator (biased) |
| γ_both | Bias-corrected estimator (both row and column) |
| A, B | Numerator and denominator of γ (centered) |
| v^{αβ}_{ijlr} = Φ_{iα}Φ_{jα}Φ_{lβ}Φ_{rβ} | Elementary quartic tensor |
| r_{ijlr} | Column-marginalized tensor (sum over α≠β) |
| t¹–t⁵ | Five unique terms in A and B |


## Environment
- Conda environment: `dimensionality`
- Activate before running any Python: `conda activate dimensionality`