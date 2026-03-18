"""
Bias-corrected participation ratio estimator — infinite underlying matrix.

Reference: Chun et al., "Estimating Dimensionality of Neural Representations
from Finite Samples", ICLR 2026.  See Sections 2–4 and Appendices A–D.
"""

import numpy as np
from ._core import _gett_all


def participation_ratio(
    Phi,
    Phi2=None,
    *,
    return_all=False,
    return_parts=False,
):
    """Estimate the participation ratio (PR) of a neural representation matrix.

    The PR is a soft count of nonzero eigenvalues of the stimulus covariance
    matrix K = (1/Q) Φ Φᵀ and serves as a measure of the global
    dimensionality of the neural manifold.

    The function computes the **centered (task) dimensionality**: the PR of
    the centered kernel kc(x, y), which removes the mean stimulus response
    from each neuron.  Centering is handled **algebraically** through the
    three-term formula structure — do *not* pre-subtract column means from
    ``Phi`` before calling, as doing so would break the independence
    assumptions on which the bias corrections rely.

    By default the function returns ``γ_both``, the estimator that corrects
    finite-sample bias from both row (stimulus) and column (neuron)
    subsampling (Section 4 of the paper).

    Parameters
    ----------
    Phi : array_like, shape (P, Q)
        Raw activation matrix.  Rows are stimuli, columns are neurons.
        Pass the matrix as recorded — do **not** pre-center.
    Phi2 : array_like, shape (P, Q), optional
        Second independent trial matrix recorded for the same stimuli and
        neurons as ``Phi``.  When provided, the cross-trial product
        construction is used to remove additive and multiplicative noise
        bias (Section 4.2.1).  When ``None``, a single-trial estimate is
        computed (no noise correction).
    return_all : bool, default False
        If ``True``, return a dict containing all four estimator variants:
        ``'naive'``, ``'row'``, ``'col'``, and ``'both'``.
    return_parts : bool, default False
        If ``True``, also include the numerator (A) and denominator (B) of
        each returned estimator.

    Returns
    -------
    gamma : float
        ``γ_both``, the bias-corrected dimensionality estimate.
        Returned when both ``return_all`` and ``return_parts`` are ``False``.
    result : dict
        Returned when ``return_all`` or ``return_parts`` is ``True``.

        Keys present when ``return_all=False``::

            'both'

        Keys present when ``return_all=True``::

            'naive', 'row', 'col', 'both'

        Additional keys when ``return_parts=True``:

        * ``return_all=False``: ``'A'``, ``'B'``
        * ``return_all=True``:  ``'A_naive'``, ``'B_naive'``,
          ``'A_row'``, ``'B_row'``, ``'A_col'``, ``'B_col'``,
          ``'A_both'``, ``'B_both'``

    Notes
    -----
    **Centering.**
    The estimator always computes the *task dimensionality* (stimulus-driven
    variance).  Centering is encoded in the algebraic structure of the
    formula:

    .. code-block::

        A = <v_iijj> - 2<v_iijl> + <v_ijlr>   # numerator
        B = <v_ijij> - 2<v_ijjl> + <v_ijlr>   # denominator

    Subtracting these cross-terms is mathematically equivalent to computing
    the PR of the centered covariance, without requiring any pre-processing
    of the data matrix.

    **Neuron dimensionality.**
    To estimate the neuron dimensionality instead (column centering, i.e.
    subtracting mean across neurons for each stimulus), transpose ``Phi``
    before calling: ``participation_ratio(Phi.T)``.

    **Minimum sizes.**  Requires P ≥ 4 and Q ≥ 2.

    **Single vs. two-trial.**
    With ``Phi2=None`` the same matrix is used for both trial slots, so no
    noise correction is applied.  Providing ``Phi2`` removes additive and
    independent multiplicative noise bias; correlated noise across trials
    is not removed (see Section A.4 of the paper).
    """
    Phi = np.asarray(Phi, dtype=float)
    P, Q = Phi.shape

    if P < 4:
        raise ValueError(f"P={P} is too small; need at least 4 stimuli.")
    if Q < 2:
        raise ValueError(f"Q={Q} is too small; need at least 2 neurons.")

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
    # Quartic sums  (all normalized by 1/P²Q² via the nf division above)
    # ------------------------------------------------------------------
    t1,  t1d  = _gett_all('ijji', An, Bn)   # ~ v_{ijij}  (denominator B)
    t2,  t2d  = _gett_all('iiii', An, Bn)   # v_{iiii}   (diagonal)
    t3,  t3d  = _gett_all('ijjj', An, Bn)   # ~ v_{iiij}  (single-trial symmetry)
    t5,  t5d  = _gett_all('ijjl', An, Bn)   # v_{ijjl}
    t6,  t6d  = _gett_all('iijj', An, Bn)   # v_{iijj}   (numerator A)
    t7,  t7d  = _gett_all('iijl', An, Bn)   # v_{iijl}
    t9,  t9d  = _gett_all('ijlm', An, Bn)   # v_{ijlr}

    # ------------------------------------------------------------------
    # Scalar coefficients for the row-correction formulas
    # ------------------------------------------------------------------
    f1 = P / (P - 2)
    f2 = 2.0 / (P - 2)
    f3 = 1.0 / ((P - 1) * (P - 2))
    row_factor = P / (P - 3)
    col_factor = Q / (Q - 1)

    # ------------------------------------------------------------------
    # Numerators (A) and denominators (B) for each estimator variant
    # ------------------------------------------------------------------

    # Naive — no bias correction
    A_naive = t6  - (2.0 / P) * t7  + (1.0 / P) ** 2 * t9
    B_naive = t1  - (2.0 / P) * t5  + (1.0 / P) ** 2 * t9

    # Row-corrected — disjoint row indices, all column indices
    A_row = row_factor * (
        t6
        - (2.0 / (P - 1)) * t7
        + (1.0 / (P - 2)) * (4.0 * t3 - P * t2)
        + f3 * (t9 - 4.0 * t5 + 2.0 * t1 - t6)
    )
    B_row = row_factor * (
        t1 - f1 * t2 + f2 * (2.0 * t3 - t5) + f3 * (t6 - 2.0 * t7 + t9)
    )

    # Col-corrected — all row indices, disjoint column indices
    A_col = t6d - (2.0 / P) * t7d + (1.0 / P) ** 2 * t9d
    B_col = t1d - (2.0 / P) * t5d + (1.0 / P) ** 2 * t9d

    # Both-corrected — disjoint row AND column indices  (recommended)
    A_both = row_factor * col_factor * (
        t6d
        - (2.0 / (P - 1)) * t7d
        + (1.0 / (P - 2)) * (4.0 * t3d - P * t2d)
        + f3 * (t9d - 4.0 * t5d + 2.0 * t1d - t6d)
    )
    B_both = row_factor * col_factor * (
        t1d - f1 * t2d + f2 * (2.0 * t3d - t5d) + f3 * (t6d - 2.0 * t7d + t9d)
    )

    # ------------------------------------------------------------------
    # Build output
    # ------------------------------------------------------------------
    gamma_both = A_both / B_both

    if not return_all and not return_parts:
        return float(gamma_both)

    result = {"both": float(gamma_both)}

    if return_all:
        result["naive"] = float(A_naive / B_naive)
        result["row"]   = float(A_row   / B_row)
        result["col"]   = float(A_col   / B_col)

    if return_parts:
        if return_all:
            result.update({
                "A_naive": float(A_naive), "B_naive": float(B_naive),
                "A_row":   float(A_row),   "B_row":   float(B_row),
                "A_col":   float(A_col),   "B_col":   float(B_col),
                "A_both":  float(A_both),  "B_both":  float(B_both),
            })
        else:
            result["A"] = float(A_both)
            result["B"] = float(B_both)

    return result
