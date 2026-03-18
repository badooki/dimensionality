"""
Bias-corrected participation ratio estimator — finite underlying matrix.

Estimates the PR of a large-but-finite R×C matrix G from a P×Q submatrix
sampled without replacement, with optional two-trial noise correction.

Reference: Chun et al., ICLR 2026, Section 4.2.4 and Appendix A.6.

Note on cross-trial pattern symmetry
-------------------------------------
The expansion formulas use the single-trial identity Σ v_{iiij} = Σ v_{ijjj}
(by relabeling dummy indices).  For the cross-trial redefinition
v^{αβ}_{ijkl} = A_{iα}B_{jα}A_{kβ}B_{lβ}, this identity does not hold
exactly.  The resulting error is O(σ² × geometry) and is small for moderate
noise.  For a fully rigorous cross-trial version, additional einsum patterns
would need to be computed separately; see the docstring of
``participation_ratio_finite`` for details.
"""

import numpy as np
from ._core import _gett_all


def participation_ratio_finite(
    Phi,
    R,
    C,
    Phi2=None,
    *,
    return_naive=False,
    return_parts=False,
):
    """Estimate the PR of a finite R×C matrix from a P×Q submatrix.

    When the underlying matrix G is large but finite (R × C) and the observed
    submatrix Φ is sampled *without replacement*, the standard infinite-matrix
    estimator (``participation_ratio``) is biased.  This function uses the
    finite-matrix correction formulas derived in Appendix A.6 of the paper.

    As with ``participation_ratio``, centering is handled algebraically —
    pass the raw activation matrix without pre-subtracting column means.

    Parameters
    ----------
    Phi : array_like, shape (P, Q)
        Raw observed submatrix (P rows, Q columns sampled without replacement
        from G).
    R : int
        Number of rows in the full underlying matrix G.  Must satisfy R ≥ P.
    C : int
        Number of columns in the full underlying matrix G.  Must satisfy
        C ≥ Q.
    Phi2 : array_like, shape (P, Q), optional
        Second independent trial for noise correction.  Same stimuli and
        neurons as ``Phi`` must be presented/recorded.
    return_naive : bool, default False
        If ``True``, also compute and return the naive (uncorrected) PR
        estimate alongside the finite-corrected one.  Forces dict output.
    return_parts : bool, default False
        If ``True``, return a dict with keys ``'gamma'``, ``'A'``, ``'B'``
        instead of a scalar.

    Returns
    -------
    gamma : float
        Estimated PR of G (with task-dimensionality centering).
        Returned when ``return_naive=False`` and ``return_parts=False``.
    result : dict
        Returned when ``return_naive=True`` or ``return_parts=True``.
        Always contains ``'gamma'``.
        Additional keys when ``return_naive=True``: ``'naive'``.
        Additional keys when ``return_parts=True``: ``'A'``, ``'B'``.

    Notes
    -----
    **Cross-trial noise correction approximation.**
    The cross-trial construction (Section A.4) eliminates additive and
    independent multiplicative noise at the cost of a small approximation
    error: the single-trial index-relabeling symmetry
    Σ v_{iiij} = Σ v_{ijjj} does not hold exactly for cross-trial v tensors.
    The error is O(σ² × geometry) and negligible for typical neural noise
    levels.  A fully exact two-trial finite estimator would require computing
    several additional einsum patterns separately.

    **Minimum sizes.**  Requires P ≥ 4, Q ≥ 2, R ≥ P, C ≥ Q.
    """
    Phi = np.asarray(Phi, dtype=float)
    P, Q = Phi.shape

    if P < 4:
        raise ValueError(f"P={P} is too small; need at least 4 stimuli.")
    if Q < 2:
        raise ValueError(f"Q={Q} is too small; need at least 2 neurons.")
    if R < P:
        raise ValueError(f"R={R} must be >= P={P}.")
    if C < Q:
        raise ValueError(f"C={C} must be >= Q={Q}.")

    if Phi2 is None:
        A, B = Phi, Phi
    else:
        Phi2 = np.asarray(Phi2, dtype=float)
        if Phi2.shape != Phi.shape:
            raise ValueError("Phi and Phi2 must have the same shape.")
        A, B = Phi, Phi2

    nf = np.sqrt(P * Q)
    An, Bn = A / nf, B / nf

    # ------------------------------------------------------------------
    # 1.  Quartic sums  (normalized by 1/P²Q² via nf)
    # ------------------------------------------------------------------
    t1,  t1d  = _gett_all('ijji', An, Bn)
    t2,  t2d  = _gett_all('iiii', An, Bn)
    t3,  t3d  = _gett_all('ijjj', An, Bn)
    t5,  t5d  = _gett_all('ijjl', An, Bn)
    t6,  t6d  = _gett_all('iijj', An, Bn)
    t7,  t7d  = _gett_all('iijl', An, Bn)
    t9,  t9d  = _gett_all('ijlm', An, Bn)

    # Single-column (α = β) variants
    t1s = t1 - t1d;  t2s = t2 - t2d;  t3s = t3 - t3d
    t5s = t5 - t5d;  t6s = t6 - t6d;  t7s = t7 - t7d;  t9s = t9 - t9d

    # ------------------------------------------------------------------
    # 2.  Distinct-row-index sums  (Eqs. S16–S22)
    # ------------------------------------------------------------------
    # 2-index distinct sums: Σ_{i≠j} = Σ_{ij} − Σ_i(diagonal)
    n2d_iijj = t6d - t2d;  n2s_iijj = t6s - t2s
    n2d_iiij = t3d - t2d;  n2s_iiij = t3s - t2s
    n2d_ijij = t1d - t2d;  n2s_ijij = t1s - t2s

    # 3-index distinct sums (Eqs. S20–S21)
    n3d_iijl = t7d - 2*t3d - t6d + 2*t2d
    n3s_iijl = t7s - 2*t3s - t6s + 2*t2s
    n3d_ijjl = t5d - 2*t3d - t1d + 2*t2d
    n3s_ijjl = t5s - 2*t3s - t1s + 2*t2s

    # 4-index distinct sums (Eq. S22)
    n4d = t9d - 2*(t7d + 2*t5d) + (t6d + 8*t3d + 2*t1d) - 6*t2d
    n4s = t9s - 2*(t7s + 2*t5s) + (t6s + 8*t3s + 2*t1s) - 6*t2s

    # ------------------------------------------------------------------
    # 3.  Falling factorials
    # ------------------------------------------------------------------
    Pf = [1, P, P*(P-1), P*(P-1)*(P-2), P*(P-1)*(P-2)*(P-3)]
    Rf = [1, R, R*(R-1), R*(R-1)*(R-2), R*(R-1)*(R-2)*(R-3)]
    Qd, Qs = Q*(Q-1), Q
    Cd, Cs = C*(C-1), C

    def _build_tg(a, terms_d, terms_s):
        """Assemble one t^k_g estimator (Eqs. S11–S15).

        Parameters
        ----------
        a : int
            Power of R in the overall 1/(R^a C²) prefactor (2, 3, or 4).
        terms_d : list of (row_order, value)
            Terms that sum over distinct (α ≠ β) column index pairs.
        terms_s : list of (row_order, value)
            Terms that sum over a single column index (α = β).
        """
        acc = 0.0
        for row_order, val in terms_d:
            acc += Rf[row_order] * Cd / (Pf[row_order] * Qd) * val
        for row_order, val in terms_s:
            acc += Rf[row_order] * Cs / (Pf[row_order] * Qs) * val
        return acc / (R**a * C**2)

    # ------------------------------------------------------------------
    # 4.  Assemble t¹_g … t⁵_g  (Eqs. S11–S15)
    # ------------------------------------------------------------------

    # t¹_g — pattern iijj  (Eq. S11)
    t1g = _build_tg(2,
                    [(2, n2d_iijj), (1, t2d)],
                    [(2, n2s_iijj), (1, t2s)])

    # t²_g — pattern iijl  (Eq. S12)
    t2g = _build_tg(3,
                    [(3, n3d_iijl), (2, 2*n2d_iiij + n2d_iijj), (1, t2d)],
                    [(3, n3s_iijl), (2, 2*n2s_iiij + n2s_iijj), (1, t2s)])

    # t³_g — pattern ijij  (Eq. S13)
    t3g = _build_tg(2,
                    [(2, n2d_ijij), (1, t2d)],
                    [(2, n2s_ijij), (1, t2s)])

    # t⁴_g — pattern ijjl  (Eq. S14)
    t4g = _build_tg(3,
                    [(3, n3d_ijjl), (2, 2*n2d_iiij + n2d_ijij), (1, t2d)],
                    [(3, n3s_ijjl), (2, 2*n2s_iiij + n2s_ijij), (1, t2s)])

    # t⁵_g — pattern ijlr  (Eq. S15)
    t5g = _build_tg(4,
                    [(4, n4d),  (3, 2*n3d_iijl + 4*n3d_ijjl),
                     (2, 4*n2d_iiij), (1, t2d)],
                    [(4, n4s),  (3, 2*n3s_iijl + 4*n3s_ijjl),
                     (2, 4*n2s_iiij), (1, t2s)])

    # ------------------------------------------------------------------
    # 5.  Centered PR of G:  γ = (t¹_g − 2t²_g + t⁵_g) / (t³_g − 2t⁴_g + t⁵_g)
    # ------------------------------------------------------------------
    A_finite = t1g - 2*t2g + t5g
    B_finite = t3g - 2*t4g + t5g
    gamma_finite = A_finite / B_finite

    if not return_naive and not return_parts:
        return float(gamma_finite)

    result = {"gamma": float(gamma_finite)}

    if return_naive:
        # Naive (uncorrected) PR from the observed submatrix Φ.
        # Uses the same normalised t-values already computed above.
        A_naive = t6 - (2.0 / P) * t7 + (1.0 / P) ** 2 * t9
        B_naive = t1 - (2.0 / P) * t5 + (1.0 / P) ** 2 * t9
        result["naive"] = float(A_naive / B_naive)

    if return_parts:
        result["A"] = float(A_finite)
        result["B"] = float(B_finite)

    return result
