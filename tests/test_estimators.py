"""
Tests for the dimensionality estimators.

Generative model (Section 4.3 of the paper):

    X  ~ N(0, I)  shape (P, d)   — stimulus latent vectors
    W  ~ N(0, I)  shape (d, Q)   — neuron latent vectors
    Phi = X @ W / d + noise        shape (P, Q)

In the limit P, Q → ∞ the true noise-free PR equals d.

Fast tests use small matrices and many random seeds to keep wall-clock time
low.  Convergence experiments that reproduce Figure 1 live in
examples/synthetic.py.
"""

import numpy as np
import pytest
from dimensionality import participation_ratio, participation_ratio_finite


# ---------------------------------------------------------------------------
# Shared generative model
# ---------------------------------------------------------------------------

def _make_Phi(P, Q, d, sigma=0.0, rng=None):
    """Single-trial activation matrix from the paper's linear model.

    Parameters
    ----------
    P, Q : int   — number of stimuli / neurons in the observed matrix
    d    : int   — true latent dimensionality
    sigma: float — noise std (0 = noiseless)
    rng  : numpy Generator

    Returns
    -------
    Phi : ndarray, shape (P, Q)
    """
    if rng is None:
        rng = np.random.default_rng(0)
    X = rng.standard_normal((P, d))   # stimuli  in R^d
    W = rng.standard_normal((d, Q))   # neurons  in R^d  (columns are w_alpha)
    Phi = X @ W / d
    if sigma > 0:
        Phi = Phi + rng.standard_normal((P, Q)) * sigma
    return Phi


def _make_two_trials(P, Q, d, sigma, rng=None):
    """Two noisy trials sharing the same stimuli and neurons (same X, W)."""
    if rng is None:
        rng = np.random.default_rng(0)
    X = rng.standard_normal((P, d))
    W = rng.standard_normal((d, Q))
    signal = X @ W / d
    Phi1 = signal + rng.standard_normal((P, Q)) * sigma
    Phi2 = signal + rng.standard_normal((P, Q)) * sigma
    return Phi1, Phi2


# ---------------------------------------------------------------------------
# participation_ratio — API / output format
# ---------------------------------------------------------------------------

class TestParticipationRatioAPI:
    def test_returns_float_by_default(self):
        Phi = _make_Phi(20, 15, d=5)
        result = participation_ratio(Phi)
        assert isinstance(result, float)
        assert result > 0

    def test_return_all_keys(self):
        Phi = _make_Phi(20, 15, d=5)
        result = participation_ratio(Phi, return_all=True)
        assert set(result.keys()) == {"naive", "row", "col", "both"}
        for v in result.values():
            assert isinstance(v, float)
            assert v > 0

    def test_return_parts_only(self):
        Phi = _make_Phi(20, 15, d=5)
        result = participation_ratio(Phi, return_parts=True)
        assert set(result.keys()) == {"both", "A", "B"}
        assert result["both"] == pytest.approx(result["A"] / result["B"])

    def test_return_all_and_parts(self):
        Phi = _make_Phi(20, 15, d=5)
        result = participation_ratio(Phi, return_all=True, return_parts=True)
        expected_keys = {
            "naive", "row", "col", "both",
            "A_naive", "B_naive",
            "A_row",   "B_row",
            "A_col",   "B_col",
            "A_both",  "B_both",
        }
        assert set(result.keys()) == expected_keys

    def test_gamma_both_equals_default(self):
        Phi = _make_Phi(30, 20, d=5)
        assert participation_ratio(Phi) == pytest.approx(
            participation_ratio(Phi, return_all=True)["both"]
        )

    def test_two_trial(self):
        Phi1, Phi2 = _make_two_trials(20, 15, d=5, sigma=0.5)
        result = participation_ratio(Phi1, Phi2)
        assert isinstance(result, float)
        assert result > 0

    def test_two_trial_shape_mismatch_raises(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="same shape"):
            participation_ratio(
                rng.standard_normal((20, 15)),
                rng.standard_normal((20, 10)),
            )

    def test_too_few_stimuli_raises(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="P=3"):
            participation_ratio(rng.standard_normal((3, 10)))

    def test_too_few_neurons_raises(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="Q=1"):
            participation_ratio(rng.standard_normal((10, 1)))


# ---------------------------------------------------------------------------
# participation_ratio — bias ordering
# ---------------------------------------------------------------------------

class TestBiasOrdering:
    """γ_both should be closer to the true d than γ_naive when P or Q is small.

    We run multiple random seeds and check that the mean estimate from γ_both
    is systematically closer to d than the mean from γ_naive.
    """

    def _run_seeds(self, P, Q, d, n_seeds=50):
        gammas = {"naive": [], "row": [], "col": [], "both": []}
        for seed in range(n_seeds):
            Phi = _make_Phi(P, Q, d, rng=np.random.default_rng(seed))
            res = participation_ratio(Phi, return_all=True)
            for key in gammas:
                gammas[key].append(res[key])
        return {k: np.mean(v) for k, v in gammas.items()}

    def test_both_less_biased_than_naive_small_P(self):
        """Small P, large Q: row correction should dominate."""
        d = 10
        means = self._run_seeds(P=20, Q=500, d=d)
        err_naive = abs(means["naive"] - d)
        err_both  = abs(means["both"]  - d)
        assert err_both < err_naive, (
            f"γ_both mean={means['both']:.2f} should be closer to d={d} "
            f"than γ_naive mean={means['naive']:.2f}"
        )

    def test_both_less_biased_than_naive_small_Q(self):
        """Small Q, large P: col correction should dominate."""
        d = 10
        means = self._run_seeds(P=500, Q=20, d=d)
        err_naive = abs(means["naive"] - d)
        err_both  = abs(means["both"]  - d)
        assert err_both < err_naive

    def test_row_corrects_row_bias(self):
        """With small P, γ_row should be less biased than γ_naive."""
        d = 10
        means = self._run_seeds(P=20, Q=500, d=d)
        assert abs(means["row"] - d) < abs(means["naive"] - d)

    def test_col_corrects_col_bias(self):
        """With small Q, γ_col should be less biased than γ_naive."""
        d = 10
        means = self._run_seeds(P=500, Q=20, d=d)
        assert abs(means["col"] - d) < abs(means["naive"] - d)

    def test_large_PQ_all_estimators_close(self):
        """With large P and Q, all estimators should be close to d."""
        d = 5
        Phi = _make_Phi(P=2000, Q=2000, d=d, rng=np.random.default_rng(99))
        res = participation_ratio(Phi, return_all=True)
        for name, val in res.items():
            assert abs(val - d) < 0.5, (
                f"γ_{name}={val:.3f} should be near d={d} with large P, Q"
            )


# ---------------------------------------------------------------------------
# participation_ratio — noise correction
# ---------------------------------------------------------------------------

class TestNoiseCorrection:
    """Two-trial noise correction should reduce bias relative to single-trial."""

    def test_two_trial_less_biased_than_single(self):
        """With moderate noise, γ_both(Phi1, Phi2) closer to truth than γ_both(Phi1)."""
        d, sigma = 5, 1.0
        n_seeds = 30
        errs_single, errs_two = [], []

        # Approximate ground truth with noiseless large matrices
        truth_vals = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed + 1000)
            truth_vals.append(
                participation_ratio(_make_Phi(2000, 2000, d=d, sigma=0.0, rng=rng))
            )
        truth = np.mean(truth_vals)

        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            Phi1, Phi2 = _make_two_trials(P=100, Q=100, d=d, sigma=sigma, rng=rng)
            errs_single.append(abs(participation_ratio(Phi1) - truth))
            errs_two.append(abs(participation_ratio(Phi1, Phi2) - truth))

        assert np.mean(errs_two) < np.mean(errs_single), (
            f"Mean error two-trial={np.mean(errs_two):.3f} should be less than "
            f"single-trial={np.mean(errs_single):.3f}"
        )


# ---------------------------------------------------------------------------
# participation_ratio_finite — API / output format
# ---------------------------------------------------------------------------

class TestFiniteAPI:
    def test_returns_float(self):
        Phi = _make_Phi(10, 8, d=3)
        result = participation_ratio_finite(Phi, R=100, C=80)
        assert isinstance(result, float)
        assert result > 0

    def test_return_parts(self):
        Phi = _make_Phi(10, 8, d=3)
        result = participation_ratio_finite(Phi, R=100, C=80, return_parts=True)
        assert set(result.keys()) == {"gamma", "A", "B"}
        assert result["gamma"] == pytest.approx(result["A"] / result["B"])

    def test_invalid_R_raises(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="R=5"):
            participation_ratio_finite(rng.standard_normal((10, 8)), R=5, C=80)

    def test_invalid_C_raises(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="C=5"):
            participation_ratio_finite(rng.standard_normal((10, 8)), R=100, C=5)

    def test_two_trial(self):
        Phi1, Phi2 = _make_two_trials(10, 8, d=3, sigma=0.5)
        result = participation_ratio_finite(Phi1, R=100, C=80, Phi2=Phi2)
        assert isinstance(result, float)
        assert result > 0

    def test_large_RC_approaches_infinite_estimator(self):
        """When R >> P and C >> Q, finite estimator should approach infinite."""
        d = 5
        Phi = _make_Phi(P=50, Q=50, d=d, rng=np.random.default_rng(15))
        gamma_inf = participation_ratio(Phi)
        gamma_fin = participation_ratio_finite(Phi, R=100_000, C=100_000)
        assert abs(gamma_fin - gamma_inf) < 0.5, (
            f"Finite ({gamma_fin:.3f}) should be close to infinite "
            f"({gamma_inf:.3f}) when R, C >> P, Q"
        )
