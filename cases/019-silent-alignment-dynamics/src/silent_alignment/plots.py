"""Deterministic PNG/SVG figures from saved observations, without solving again."""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .io import read_json


def figures(run, output):
    run, output = Path(run), Path(output)
    out = output / "figures"
    out.mkdir(exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "case019-p0-v1", "savefig.dpi": 140})
    colors = ["#175c9d", "#bc5214", "#258650"]

    def load(name):
        with np.load(run / "raw" / (name + ".npz"), allow_pickle=False) as raw:
            return {k: raw[k] for k in raw.files}

    def save(fig, name):
        fig.savefig(out / (name + ".png"), metadata={"Software": "Case019"})
        fig.savefig(out / (name + ".svg"), metadata={"Date": None, "Creator": "Case019"})
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained", sharex=True)
    for row, (name, label) in enumerate((("B_random0_rho2", "Random R, replicate 0"),
                                        ("B_aligned_rho2", "Oracle aligned R (unique control)"))):
        r = load(name)
        t = r["times"]
        axes[row, 0].plot(t, r["right_distance"], label="Right factor distance")
        axes[row, 0].plot(t, r["left_distance"], ls="--", label="Left factor distance (t=0 NA)")
        axes[row, 0].set(ylabel="Projector distance / (2m)", ylim=(-.025, .7))
        axes[row, 1].plot(t, r["learnable_residual"], color=colors[1], label="Observed residual")
        axes[row, 1].axhline(.9, ls=":", color="0.35", label="10% visible-loss threshold")
        axes[row, 1].set(ylabel="(L - Lmin) / (L0 - Lmin)", ylim=(-.03, 1.04))
        for i in range(4):
            axes[row, 2].plot(t, r["z_singular_sorted"][:, i], label=f"Sorted singular value {i+1}")
        axes[row, 2].set(ylabel="Singular value of Z", ylim=(-.025, 1.04))
        for j, title in enumerate(("Factor subspace", "Learnable loss", "Product magnitude")):
            axes[row, j].set(title=f"{label}\n{title}", xlabel="Gradient-flow time t")
            axes[row, j].legend(fontsize=8, loc="best")
    fig.suptitle("8 x 8, m=4, rho=1e-6: direction and amplitude on the same time axis\n"
                 "Individual saved trajectories; no averaging; random seed stream 61023019/2/3/0")
    save(fig, "01_direction_loss_spectrum")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for j, rho in enumerate((1e-2, 1e-4, 1e-6)):
        for rep in (0, 1):
            r = load(f"B_random{rep}_rho{j}")
            for col, x in enumerate((r["times"], r["times"] / np.log(1 / rho))):
                label = f"rho={rho:g}, R{rep}"
                axes[0, col].plot(x, r["right_distance"], color=colors[j], ls="-" if rep == 0 else "--", label=label)
                axes[1, col].plot(x, r["learnable_residual"], color=colors[j], ls="-" if rep == 0 else "--", label=label)
    for col, xlabel in enumerate(("Gradient-flow time t", "Rescaled time t / log(1/rho)")):
        axes[0, col].set(ylabel="Right projector distance / (2m)", xlabel=xlabel)
        axes[1, col].set(ylabel="Learnable residual", xlabel=xlabel)
        axes[0, col].legend(fontsize=8)
    fig.suptitle("8 x 8 random starts: 2 paired R samples, 3 rho values\nSame target and R across rho; a finite pilot")
    save(fig, "02_rho_time_scales")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    summary = read_json(output / "summary.json")
    for i, sigma in enumerate((.5, 1.0)):
        for j, rho in enumerate((1e-2, 1e-4, 1e-6)):
            r = load(f"A_sigma{i}_rho{j}")
            t, z = r["times"], r["target_mode_signed"][:, 0]
            with np.load(run / "raw" / f"A_sigma{i}_rho{j}_scalar_reference.npz", allow_pickle=False) as a:
                approximation = a["early_approx"].copy() / sigma
                approximation[~a["early_approx_valid_domain"]] = np.nan
            axes[i, 0].plot(t, z / sigma, color=colors[j], label=f"Observed, rho={rho:g}")
            axes[i, 0].plot(t, approximation, color=colors[j], ls=":", lw=2,
                            label="Early rho*sinh / sigma (z <= 1% target)" if j == 0 else None)
            axes[i, 2].plot(t, r["right_distance"], color=colors[j], label=f"rho={rho:g}")
        fit = summary["scalar_fixed_amplitude_fits"][i]
        x = np.array([np.log(1/p["rho"]) for p in fit["points"]])
        y = np.array([p["crossing"]["time"] for p in fit["points"]])
        axes[i, 1].plot(x, y, "o-", label="Observed 10% target arrival")
        intercept = np.mean(y - fit["predicted_leading_log_slope"] * x)
        axes[i, 1].plot(x, intercept + fit["predicted_leading_log_slope"] * x, ls="--",
                        label="Predicted slope 1/(2 sigma); fitted intercept")
        axes[i, 0].set(xlabel="Gradient-flow time t", ylabel="z / sigma", title=f"Aligned scalar, sigma={sigma:g}")
        axes[i, 1].set(xlabel="log(1/rho)", ylabel="Time to z = 0.1 sigma", title="Finite-rho timing diagnostic")
        axes[i, 2].set(xlabel="Gradient-flow time t", ylabel="Right projector distance", title="Direction already aligned", ylim=(-.01, .05))
        for ax in axes[i]:
            ax.legend(fontsize=7)
    fig.suptitle("Aligned scalar controls: amplitude delay with no direction to learn\n6 primary trajectories; early approximation shown only in its declared domain")
    save(fig, "03_aligned_scalar_activation")
