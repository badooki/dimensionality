"""
Plotting utilities for sweep results.
"""

# Estimator display styles — colours and markers match legacy/synthetic/Linear_example.ipynb
# Labels use paper notation (no hat, no 'naive'/'row'/'col'/'both' verbatim)
_STYLES_INF = {
    "naive": dict(color="k",   marker="o",  label=r"$\gamma_{\rm naive}$"),
    "row":   dict(color="g",   marker=">",  label=r"$\gamma_{\rm row}$"),
    "col":   dict(color="g",   marker="<",  label=r"$\gamma_{\rm col}$"),
    "both":  dict(color="r",   marker="x",  label=r"$\gamma_{\rm both}$"),
}
_STYLES_FIN = {
    "naive": dict(color="k",   marker="o",  label=r"$\gamma_0$"),
    "gamma": dict(color="r",   marker="x",  label=r"$\gamma_{\rm finite}$"),
}

_LW  = 1.5   # line width
_MS  = 7     # marker size
_A   = 0.7   # line alpha
_ELW = 1.0   # error-bar line width


def plot_sweep(result, ax=None, true_d=None, title=None, figsize=(5, 4)):
    """Plot the output of :func:`sweep_dimensionality`.

    Parameters
    ----------
    result : dict
        Return value of ``sweep_dimensionality``.
    ax : matplotlib Axes, optional
        Axes to draw on.  A new figure is created if ``None``.
    true_d : float, optional
        Draw a solid blue reference line at this value.
    title : str, optional
        Axes title.
    figsize : tuple, default (5, 4)
        Figure size; ignored when ``ax`` is provided.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    values = result["values"]
    styles = _STYLES_FIN if "gamma" in result else _STYLES_INF

    for k, sty in styles.items():
        if k not in result:
            continue
        mean = result[k]
        sem_key = f"{k}_sem"

        # Error bars (±1 SEM)
        if sem_key in result:
            sem = result[sem_key]
            ax.errorbar(values, mean, yerr=sem,
                        color=sty["color"], marker="", ls="",
                        alpha=1.0, lw=_ELW, zorder=0)

        # Main line
        ax.plot(values, mean,
                color=sty["color"], marker=sty["marker"],
                ls="-", lw=_LW, ms=_MS, alpha=_A,
                fillstyle="none", zorder=1,
                label=sty["label"])

    if true_d is not None:
        ax.axhline(true_d, color="b", lw=3, alpha=0.5,
                   label=r"$\gamma$" if true_d else None, zorder=-10)

    axis = result["axis"]
    if axis == "P":
        fixed = result.get("fixed_Q")
        xlabel = r"$P$" + (f"  ($Q={fixed}$)" if fixed else "")
    else:
        fixed = result.get("fixed_P")
        xlabel = r"$Q$" + (f"  ($P={fixed}$)" if fixed else "")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(r"$\gamma$", fontsize=12)
    #ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10, framealpha=0.8)

    if title:
        ax.set_title(title, fontsize=12)

    return fig, ax
