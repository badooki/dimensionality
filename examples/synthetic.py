"""
Figure 1 reproduction — bias-corrected participation ratio estimators.

Generates two panels that mirror Figure 1 of:
    Chun, Canatar, Chung, Lee (2026).
    "Estimating Dimensionality of Neural Representations from Finite Samples."
    ICLR 2026.

Generative model  (Section 4.3):
    X ~ N(0, I_d)  shape (P, d)
    W ~ N(0, I_d)  shape (d, Q)
    Phi = X @ W / d + noise        shape (P, Q)

Panel A: vary P with Q fixed at 100.
Panel B: vary Q with P fixed at 200.

True d = 50, noise variance σ² = 0.2, 20 random seeds per (P, Q) point.

Usage
-----
    python examples/synthetic.py              # saves figure to examples/figure1.png
    python examples/synthetic.py --show       # also opens interactive window
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from dimensionality import participation_ratio


# ---------------------------------------------------------------------------
# Generative model
# ---------------------------------------------------------------------------

def _make_Phi(P, Q, d, sigma=0.0, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    X = rng.standard_normal((P, d))
    W = rng.standard_normal((d, Q))
    Phi = X @ W / d
    if sigma > 0:
        Phi = Phi + rng.standard_normal((P, Q)) * sigma
    return Phi


# ---------------------------------------------------------------------------
# Sweep helper
# ---------------------------------------------------------------------------

def sweep(P_vals, Q_vals, d, sigma, n_seeds=20):
    """Return mean ± sem of all four estimators over n_seeds realisations.

    Exactly one of P_vals or Q_vals should be a list; the other is a scalar.
    Returns a dict: key → array of shape (n_points,) for mean and sem.
    """
    keys = ["naive", "row", "col", "both"]
    results = {k: [] for k in keys}
    sems    = {k: [] for k in keys}

    is_P_sweep = hasattr(P_vals, "__len__")
    axis = P_vals if is_P_sweep else Q_vals

    for val in axis:
        P = val if is_P_sweep else P_vals
        Q = val if not is_P_sweep else Q_vals
        estimates = {k: [] for k in keys}
        for seed in range(n_seeds):
            Phi = _make_Phi(P, Q, d, sigma=sigma, rng=np.random.default_rng(seed))
            res = participation_ratio(Phi, return_all=True)
            for k in keys:
                estimates[k].append(res[k])
        for k in keys:
            arr = np.array(estimates[k])
            results[k].append(np.mean(arr))
            sems[k].append(np.std(arr, ddof=1) / np.sqrt(n_seeds))

    return {k: np.array(results[k]) for k in keys}, {k: np.array(sems[k]) for k in keys}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

STYLES = {
    "naive": dict(color="#999999", ls="--",  label=r"$\gamma_{\rm naive}$"),
    "row":   dict(color="#4DA6FF", ls="-.",  label=r"$\gamma_{\rm row}$"),
    "col":   dict(color="#FF8C42", ls=":",   label=r"$\gamma_{\rm col}$"),
    "both":  dict(color="#2CA02C", ls="-",   label=r"$\gamma_{\rm both}$"),
}


def _plot_panel(ax, axis_vals, means, sems, true_d, xlabel):
    for k, sty in STYLES.items():
        ax.plot(axis_vals, means[k], lw=1.8, **sty)
        ax.fill_between(axis_vals,
                        means[k] - sems[k],
                        means[k] + sems[k],
                        color=sty["color"], alpha=0.2)
    ax.axhline(true_d, color="black", lw=1.2, ls="-", label=f"truth $d={true_d}$", zorder=0)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(r"Estimated $\gamma$", fontsize=12)
    ax.legend(fontsize=10, framealpha=0.8)
    ax.set_ylim(bottom=0)


def main():
    parser = argparse.ArgumentParser(description="Reproduce Figure 1 from Chun et al. 2026")
    parser.add_argument("--show", action="store_true", help="Open interactive matplotlib window")
    parser.add_argument("--d",       type=int,   default=50,  help="True latent dimensionality")
    parser.add_argument("--sigma",   type=float, default=np.sqrt(0.2), help="Noise std (default √0.2)")
    parser.add_argument("--seeds",   type=int,   default=20,  help="Number of random seeds")
    parser.add_argument("--Q_fixed", type=int,   default=100, help="Fixed Q for panel A")
    parser.add_argument("--P_fixed", type=int,   default=200, help="Fixed P for panel B")
    parser.add_argument("--out", default="examples/figure1.png", help="Output path")
    args = parser.parse_args()

    d, sigma, n_seeds = args.d, args.sigma, args.seeds

    # Panel A: vary P, Q fixed
    P_vals = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500]
    print(f"Panel A: sweeping P={P_vals}, Q={args.Q_fixed}, d={d}, σ={sigma:.3f}, seeds={n_seeds}")
    means_A, sems_A = sweep(P_vals, args.Q_fixed, d, sigma, n_seeds)

    # Panel B: vary Q, P fixed
    Q_vals = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500]
    print(f"Panel B: sweeping Q={Q_vals}, P={args.P_fixed}, d={d}, σ={sigma:.3f}, seeds={n_seeds}")
    means_B, sems_B = sweep(args.P_fixed, Q_vals, d, sigma, n_seeds)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    _plot_panel(axes[0], P_vals, means_A, sems_A, d, xlabel=r"Number of stimuli $P$  ($Q$=" + str(args.Q_fixed) + ")")
    _plot_panel(axes[1], Q_vals, means_B, sems_B, d, xlabel=r"Number of neurons $Q$  ($P$=" + str(args.P_fixed) + ")")
    axes[0].set_title("A: vary stimuli", fontsize=12)
    axes[1].set_title("B: vary neurons", fontsize=12)
    fig.suptitle(rf"Bias-corrected PR estimators  ($d={d}$, $\sigma^2={sigma**2:.2f}$)", fontsize=13)
    fig.tight_layout()

    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
