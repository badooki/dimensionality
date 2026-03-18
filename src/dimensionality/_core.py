"""
Internal helpers shared by the infinite- and finite-matrix estimators.
"""

import numpy as np
import opt_einsum as oe


def _gett_all(pattern, A, B):
    """Compute two quartic sums for a given row-index pattern.

    Given P×Q matrices A (trial 1) and B (trial 2), each already divided
    by sqrt(P*Q), compute two scalars:

      pval  : (1/P²Q²) Σ_{rows(pattern), α, β}  A_{.α} B_{.α} A_{.β} B_{.β}
      pqval : (1/P²Q²) Σ_{rows(pattern), α≠β}   A_{.α} B_{.α} A_{.β} B_{.β}

    The 4-character ``pattern`` specifies which row indices are tied across
    the four factors in order (A_col_α, B_col_α, A_col_β, B_col_β).
    Repeated characters tie the corresponding row indices together.

    For example:
      ``'iijj'`` → A_{iα} B_{iα} A_{jβ} B_{jβ}   (numerator-type term)
      ``'ijji'`` → A_{iα} B_{jα} A_{jβ} B_{iβ}   (denominator-type term)
      ``'ijlm'`` → A_{iα} B_{jα} A_{lβ} B_{mβ}   (four distinct rows)

    Parameters
    ----------
    pattern : str, length 4
        Row-index pattern using characters from ``{i, j, l, m}``.
    A, B : ndarray, shape (P, Q)
        Normalized trial matrices (already divided by ``sqrt(P*Q)``).

    Returns
    -------
    pval  : float   – sum over all (α, β) column index pairs
    pqval : float   – sum over distinct (α ≠ β) column index pairs
    """
    i, j, l, m = list(pattern)
    all_cols = f"{i}a,{j}a,{l}b,{m}b->"
    same_col = f"{i}a,{j}a,{l}a,{m}a->"
    pval  = float(oe.contract(all_cols, A, B, A, B))
    pqval = pval - float(oe.contract(same_col, A, B, A, B))
    return pval, pqval
