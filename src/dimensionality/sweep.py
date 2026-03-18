"""
Subsampling sweep over P (stimuli) or Q (neurons).

For each target size, rows or columns are drawn uniformly at random without
replacement from the full data matrix.  This is repeated ``n_trials`` times
and the estimates are averaged (after computing the ratio, by default).
"""

import numpy as np
from .estimators import participation_ratio
from .finite import participation_ratio_finite


def _log_int_range(lo, hi, n=12):
    """n log-spaced integers in [lo, hi], always including hi."""
    vals = np.unique(np.round(np.geomspace(lo, hi, n)).astype(int))
    if vals[-1] != hi:
        vals = np.append(vals, hi)
    return vals


def sweep_dimensionality(
    Phi,
    axis="P",
    values=None,
    n_trials=20,
    Phi2=None,
    estimator="infinite",
    R=None,
    C=None,
    average_before_ratio=False,
    rng=None,
):
    """Estimate participation ratio at varying numbers of rows or columns.

    For each entry in ``values``, ``n_trials`` submatrices are drawn by
    sampling that many rows (``axis='P'``) or columns (``axis='Q'``) uniformly
    at random *without replacement* from ``Phi``.  The PR is computed for each
    subsample and the trials are averaged.

    Parameters
    ----------
    Phi : array_like, shape (P_max, Q_max)
        Full data matrix.  Rows are stimuli, columns are neurons.
    axis : {'P', 'Q'}
        Which dimension to sweep.
    values : array_like of int, optional
        Target sizes to evaluate.  If ``None``, ~12 log-spaced integers from
        ``min(10, axis_max)`` up to the full axis size are used.
    n_trials : int, default 20
        Number of independent random subsamples per value.
    Phi2 : array_like, shape (P_max, Q_max), optional
        Second trial for noise correction.  Subsampled with the same indices
        as ``Phi`` at each trial.
    estimator : {'infinite', 'finite'}
        Which estimator family to use.  ``'infinite'`` returns all four
        variants (naive, row, col, both).  ``'finite'`` returns naive and the
        finite-corrected estimate (gamma).
    R, C : int, optional
        Full underlying matrix dimensions; required when
        ``estimator='finite'``.
    average_before_ratio : bool, default False
        If ``False`` (default): compute γ = A/B for each subsample, then
        average the γ values across trials.
        If ``True``: average A and B separately across trials first, then
        compute γ = mean(A) / mean(B).
    rng : numpy Generator, optional
        For reproducibility.

    Returns
    -------
    result : dict
        'axis'   : 'P' or 'Q'
        'values' : 1-D int array of sweep values used
        'fixed_Q': Q_max  (present when axis='P')
        'fixed_P': P_max  (present when axis='Q')

        For ``estimator='infinite'``:
            'naive', 'row', 'col', 'both' : mean estimates  (1-D arrays)
            'naive_sem', 'row_sem', 'col_sem', 'both_sem' : SEM arrays

        For ``estimator='finite'``:
            'naive', 'gamma' : mean estimates  (1-D arrays)
            'naive_sem', 'gamma_sem' : SEM arrays

        When ``average_before_ratio=True``, the dict additionally contains
        the corresponding ``'*_ratio_of_means'`` keys computed from the
        averaged A/B.
    """
    Phi = np.asarray(Phi, dtype=float)
    P_max, Q_max = Phi.shape

    if rng is None:
        rng = np.random.default_rng()

    if axis not in ("P", "Q"):
        raise ValueError(f"axis must be 'P' or 'Q', got {axis!r}")

    if estimator not in ("infinite", "finite"):
        raise ValueError(f"estimator must be 'infinite' or 'finite', got {estimator!r}")

    if estimator == "finite" and (R is None or C is None):
        raise ValueError("R and C are required when estimator='finite'.")

    # Determine the axis being swept and enforce hard minimums
    axis_max = P_max if axis == "P" else Q_max
    hard_min = 4 if axis == "P" else 2

    if values is None:
        lo = max(hard_min, min(10, axis_max))
        values = _log_int_range(lo, axis_max)
    values = np.asarray(values, dtype=int)

    # Filter out values below the hard minimum and above axis_max
    values = values[(values >= hard_min) & (values <= axis_max)]
    if len(values) == 0:
        raise ValueError(
            f"No valid values to sweep: all entries are outside "
            f"[{hard_min}, {axis_max}]."
        )

    keys = ["naive", "row", "col", "both"] if estimator == "infinite" else ["naive", "gamma"]

    # acc[k] will be shape (n_values, n_trials) after the loop
    acc      = {k: [] for k in keys}
    acc_A    = {k: [] for k in keys}   # numerators, for average_before_ratio
    acc_B    = {k: [] for k in keys}   # denominators

    for val in values:
        trial_vals = {k: [] for k in keys}
        trial_A    = {k: [] for k in keys}
        trial_B    = {k: [] for k in keys}

        for _ in range(n_trials):
            # Random subsample without replacement
            if axis == "P":
                idx = rng.choice(P_max, size=int(val), replace=False)
                sub  = Phi[idx, :]
                sub2 = Phi2[idx, :] if Phi2 is not None else None
            else:
                idx = rng.choice(Q_max, size=int(val), replace=False)
                sub  = Phi[:, idx]
                sub2 = Phi2[:, idx] if Phi2 is not None else None

            if estimator == "finite":
                res = participation_ratio_finite(
                    sub, R=R, C=C, Phi2=sub2,
                    return_naive=True, return_parts=average_before_ratio,
                )
                for k in keys:
                    trial_vals[k].append(res[k])
                if average_before_ratio:
                    # A/B only available for the finite-corrected estimate
                    trial_A["gamma"].append(res["A"])
                    trial_B["gamma"].append(res["B"])
            else:
                res = participation_ratio(
                    sub, sub2,
                    return_all=True,
                    return_parts=average_before_ratio,
                )
                for k in keys:
                    trial_vals[k].append(res[k])
                if average_before_ratio:
                    # keys with return_all=True, return_parts=True are A_naive,
                    # A_row, A_col, A_both (and same for B)
                    for k in keys:
                        trial_A[k].append(res[f"A_{k}"])
                        trial_B[k].append(res[f"B_{k}"])

        for k in keys:
            acc[k].append(trial_vals[k])
            if average_before_ratio:
                if trial_A[k]:
                    acc_A[k].append(trial_A[k])
                    acc_B[k].append(trial_B[k])

    # Build result arrays
    result = {
        "axis":   axis,
        "values": values,
    }
    if axis == "P":
        result["fixed_Q"] = Q_max
    else:
        result["fixed_P"] = P_max

    for k in keys:
        mat = np.array(acc[k])                      # (n_values, n_trials)
        result[k]              = mat.mean(axis=1)
        result[f"{k}_sem"]     = mat.std(axis=1, ddof=1) / np.sqrt(n_trials)

    if average_before_ratio:
        for k in keys:
            if acc_A[k]:
                A_mean = np.array(acc_A[k]).mean(axis=1)  # (n_values,)
                B_mean = np.array(acc_B[k]).mean(axis=1)
                result[f"{k}_ratio_of_means"] = A_mean / B_mean

    return result
