"""Generate every figure from ``results/results.csv``.

Figure-specific notes, each responding to a review comment:

Figure 1  Redrawn architecture.  Each tier label appears exactly once, and no
          jump-diffusion or regime-switching element is drawn, because neither
          is implemented in this package.

Figure 3  Convergence plotted against the cumulative number of objective
          evaluations, with every curve carried forward as a right-continuous
          best-so-far step function.  A method that terminates early therefore
          appears as a flat line at its final value instead of vanishing part
          of the way along the axis, which is what made the previous version
          unreadable.

Figure 4  Legend anchored outside the axes so it cannot overlap the data or
          the caption, with distinct marker shapes in addition to colour.

Figure 5  Means annotated numerically and read from the same results file as
          the tables, so the figure and the tables cannot disagree.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.patches as mpatches           # noqa: E402
import matplotlib.pyplot as plt                 # noqa: E402

__all__ = ["write_figures", "PALETTE"]

PROPOSED = "QuOpt-Agents"
ENUMERATION = "Exhaustive enumeration"

# Colour-blind-safe palette (Okabe-Ito), consistent across all figures.
PALETTE = {
    PROPOSED: "#0072B2",
    "Classical MVO": "#CC79A7",
    "Strong Greedy": "#E69F00",
    "GA": "#009E73",
    "PSO": "#56B4E9",
    "DE": "#D55E00",
    "SA": "#666666",
    ENUMERATION: "#000000",
}
MARKERS = {PROPOSED: "o", "Classical MVO": "s", "Strong Greedy": "^",
           "GA": "D", "PSO": "v", "DE": "P", "SA": "X"}


def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "figure.constrained_layout.use": True,
        "savefig.dpi": 600,
    })


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "eps", "png"):
        fig.savefig(out_dir / f"{name}.{suffix}", format=suffix,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.pdf / .eps / .png")


# ------------------------------------------------------------------ Figure 1
def figure1_architecture(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, text, face, fontsize=8, weight="normal"):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.10,rounding_size=0.12",
            linewidth=0.9, edgecolor="#333333", facecolor=face))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, wrap=True)

    def arrow(x1, y1, x2, y2, style="-|>", dashed=False, label=None):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, linewidth=0.9,
                                    color="#333333",
                                    linestyle="--" if dashed else "-",
                                    shrinkA=1, shrinkB=1))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, label, fontsize=7,
                    ha="center", style="italic", color="#555555")

    box(0.3, 8.2, 4.3, 1.3, "Agent 1 \u2014 Data Preparation", "#D9E2F3",
        fontsize=9, weight="bold")
    box(0.3, 6.6, 2.05, 1.3,
        "Synthetic Market Generator\ngeometric Brownian motion\nfive regimes; "
        "T = 252; seed 42", "#EAF1F8", fontsize=6.5)
    box(2.55, 6.6, 2.05, 1.3,
        "Real-Data Loader\nYahoo Finance adjusted close\n16 US large-cap "
        "equities", "#EAF1F8", fontsize=6.5)

    box(0.3, 4.9, 4.3, 1.1,
        "Agent 2a \u2014 Market Analyst\nlog-returns; annualised "
        "$\\hat{\\mu}$ and $\\hat{\\Sigma}$ (factor 252)", "#E2F0D9",
        fontsize=7.5)
    box(0.3, 3.2, 4.3, 1.1,
        "Agent 2b \u2014 Risk Assessor\nportfolio return, volatility, "
        "Sharpe ratio ($r_f = 0$)", "#E2F0D9", fontsize=7.5)

    box(5.4, 3.2, 4.3, 6.3,
        "", "#FFF2CC")
    ax.text(7.55, 9.15, "Agent 2c \u2014 Portfolio Optimizer\n"
            "(three-tier hybrid)", ha="center", va="center", fontsize=9,
            fontweight="bold")
    box(5.7, 7.3, 3.7, 1.1,
        "Tier 1: multi-risk spectrum\n$q \\in Q$ (8 values)", "#FDF3D0",
        fontsize=7.5)
    box(5.7, 5.9, 3.7, 1.1,
        "Tier 2: classically simulated QAOA\n$p = 2$, 1024 shots per $q$\n"
        "(no quantum hardware)", "#FDF3D0", fontsize=7)
    box(5.7, 4.5, 3.7, 1.1,
        "Tier 3: classical local search\npairwise swaps, "
        "$\\varepsilon = 10^{-5}$", "#FDF3D0", fontsize=7.5)
    box(5.7, 3.4, 3.7, 0.85,
        "interchangeable global-search tier", "#FFFFFF", fontsize=6.5)

    box(0.3, 1.2, 9.4, 1.3,
        "Output \u2014 selected portfolio $x^*$ and out-of-sample metrics; "
        "comparison with Classical MVO, Strong Greedy, GA, PSO, DE, SA and "
        "exhaustive enumeration", "#F2F2F2", fontsize=7.5)

    arrow(1.35, 8.2, 1.35, 7.9)
    arrow(3.55, 8.2, 3.55, 7.9)
    arrow(2.45, 6.6, 2.45, 6.0)
    arrow(2.45, 4.9, 2.45, 4.3)
    arrow(4.6, 3.75, 5.4, 3.75)
    arrow(7.55, 3.2, 7.55, 2.5)
    arrow(2.45, 3.2, 2.45, 2.5)
    ax.annotate("", xy=(5.7, 7.85), xytext=(4.6, 4.05),
                arrowprops=dict(arrowstyle="-|>", linewidth=0.9,
                                color="#B45F06", linestyle="--",
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(5.05, 6.35, "risk-parameter\nfeedback", fontsize=6.5,
            style="italic", color="#B45F06", ha="center")

    _save(fig, out_dir, "figure1_architecture")


# ------------------------------------------------------------------ Figure 2
def figure2_runtime(real: pd.DataFrame, out_dir: Path) -> None:
    runtime = (real.assign(runtime_s=pd.to_numeric(real["runtime_s"],
                                                   errors="coerce"))
               .groupby("method")["runtime_s"].mean().dropna()
               .sort_values())
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    colors = [PALETTE.get(m, "#999999") for m in runtime.index]
    bars = ax.barh(range(len(runtime)), runtime.to_numpy(), color=colors,
                   edgecolor="#333333", linewidth=0.6)
    ax.set_yticks(range(len(runtime)))
    ax.set_yticklabels(runtime.index)
    ax.set_xscale("log")
    ax.set_xlabel("Mean runtime per walk-forward window (s, logarithmic)")
    for bar, value in zip(bars, runtime.to_numpy()):
        ax.text(value * 1.15, bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}", va="center", fontsize=7)
    ax.set_xlim(right=runtime.max() * 4)
    _save(fig, out_dir, "figure2_runtime")


# ------------------------------------------------------------------ Figure 3
def figure3_convergence(trajectories: dict, optimum: float | None,
                        out_dir: Path) -> None:
    """Best-so-far incumbent against cumulative objective evaluations."""
    by_method: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for (experiment, _window, method), (evals, values) in \
            trajectories.items():
        if experiment != "real":
            continue
        by_method.setdefault(method, []).append((evals, values))
    if not by_method:
        print("  figure3 skipped: no trajectories recorded")
        return

    max_evals = max(int(e[-1]) for runs in by_method.values()
                    for e, _ in runs)
    grid = np.unique(np.clip(
        np.round(np.logspace(0, np.log10(max(max_evals, 2)), 220)).astype(int),
        1, max_evals))

    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for method, runs in by_method.items():
        curves = []
        for evals, values in runs:
            # Right-continuous best-so-far step function, carried forward to
            # the end of the axis so that no curve terminates mid-plot.
            idx = np.searchsorted(evals, grid, side="right") - 1
            curve = np.where(idx >= 0, values[np.clip(idx, 0, None)], np.nan)
            curves.append(curve)
        stack = np.vstack(curves)
        median = np.nanmedian(stack, axis=0)
        lower = np.nanpercentile(stack, 25, axis=0)
        upper = np.nanpercentile(stack, 75, axis=0)
        color = PALETTE.get(method, "#999999")
        ax.step(grid, median, where="post", color=color, linewidth=1.4,
                label=method, zorder=3 if method == PROPOSED else 2)
        ax.fill_between(grid, lower, upper, step="post", color=color,
                        alpha=0.12, linewidth=0)
        last = max(int(e[-1]) for e, _ in runs)
        final = float(np.nanmedian([v[-1] for _, v in runs]))
        ax.plot([last], [final], marker=MARKERS.get(method, "o"),
                color=color, markersize=5, markeredgecolor="white",
                markeredgewidth=0.6, zorder=4)

    if optimum is not None and np.isfinite(optimum):
        ax.axhline(optimum, color="black", linestyle="--", linewidth=1.0,
                   zorder=1)
        ax.text(grid[-1], optimum, " exact optimum (enumeration)",
                fontsize=7, va="bottom", ha="right")

    ax.set_xscale("log")
    ax.set_xlabel("Cumulative objective-function evaluations (logarithmic)")
    ax.set_ylabel("Best-so-far in-sample Sharpe ratio")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    _save(fig, out_dir, "figure3_convergence")


# ------------------------------------------------------------------ Figure 4
def figure4_risk_return(synth: pd.DataFrame, out_dir: Path) -> None:
    methods = [PROPOSED, "Classical MVO", "Strong Greedy"]
    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    for method in methods:
        block = synth[synth["method"] == method]
        if block.empty:
            continue
        vol = pd.to_numeric(block["vol"], errors="coerce").to_numpy()
        ret = pd.to_numeric(block["ret"], errors="coerce").to_numpy()
        color = PALETTE.get(method, "#999999")
        ax.scatter(vol, ret, s=26, alpha=0.6, color=color,
                   marker=MARKERS.get(method, "o"), edgecolor="white",
                   linewidth=0.4, label=method)
        ax.errorbar(np.nanmean(vol), np.nanmean(ret),
                    xerr=np.nanstd(vol), yerr=np.nanstd(ret),
                    fmt=MARKERS.get(method, "o"), color=color, markersize=10,
                    markeredgecolor="black", markeredgewidth=0.8,
                    elinewidth=1.1, capsize=3, zorder=5)

    ax.axhline(0.0, color="#666666", linewidth=0.7, linestyle=":")
    ax.set_xlabel("Annualised portfolio volatility $\\sigma$")
    ax.set_ylabel("Annualised expected portfolio return $\\mu$")
    # Legend anchored outside the axes: this is the overlap the editor flagged.
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0.0, frameon=False,
              title="Method (large marker = centroid)",
              title_fontsize=8)
    _save(fig, out_dir, "figure4_risk_return")


# ------------------------------------------------------------------ Figure 5
def figure5_sharpe_distribution(synth: pd.DataFrame, out_dir: Path) -> None:
    from scipy.stats import gaussian_kde

    methods = [PROPOSED, "Classical MVO", "Strong Greedy"]
    series = {m: pd.to_numeric(synth[synth["method"] == m]["sharpe_is"],
                               errors="coerce").dropna().to_numpy()
              for m in methods}
    series = {m: v for m, v in series.items() if v.size > 1}
    if not series:
        print("  figure5 skipped: no synthetic Sharpe ratios")
        return

    all_values = np.concatenate(list(series.values()))
    lo, hi = all_values.min(), all_values.max()
    pad = 0.15 * (hi - lo if hi > lo else 1.0)
    grid = np.linspace(lo - pad, hi + pad, 400)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.2, 5.0),
                                   height_ratios=[2.0, 1.0], sharex=True)
    for method, values in series.items():
        color = PALETTE.get(method, "#999999")
        density = gaussian_kde(values)(grid)
        ax1.plot(grid, density, color=color, linewidth=1.4, label=method)
        ax1.fill_between(grid, density, color=color, alpha=0.12, linewidth=0)
        mean = float(np.mean(values))
        ax1.axvline(mean, color=color, linestyle="--", linewidth=1.0)
        ax1.annotate(f"{mean:.3f}", xy=(mean, ax1.get_ylim()[1]),
                     xytext=(2, -2), textcoords="offset points",
                     fontsize=7.5, color=color, rotation=90, va="top")

    ax1.set_ylabel("Kernel density")
    ax1.legend(loc="upper left", frameon=False)
    ax1.text(0.99, 0.97, "(a)", transform=ax1.transAxes, ha="right",
             va="top", fontsize=9, fontweight="bold")

    labels = list(series)
    box = ax2.boxplot([series[m] for m in labels], vert=False,
                      patch_artist=True, showmeans=True, widths=0.6,
                      meanprops=dict(marker="D", markerfacecolor="black",
                                     markeredgecolor="black", markersize=4))
    for patch, method in zip(box["boxes"], labels):
        patch.set_facecolor(PALETTE.get(method, "#999999"))
        patch.set_alpha(0.35)
        patch.set_edgecolor("#333333")
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("Sharpe ratio across synthetic scenarios")
    ax2.text(0.99, 0.92, "(b)", transform=ax2.transAxes, ha="right",
             va="top", fontsize=9, fontweight="bold")

    _save(fig, out_dir, "figure5_sharpe_distribution")


# ------------------------------------------------------------------- driver
def write_figures(results_path: Path, out_dir: Path,
                  trajectories: dict | None = None) -> None:
    _style()
    results = pd.read_csv(results_path)
    out_dir = Path(out_dir)
    synth = results[results["experiment"] == "synthetic"]
    real = results[results["experiment"] == "real"]

    figure1_architecture(out_dir)
    if not real.empty:
        figure2_runtime(real, out_dir)
        if trajectories:
            enumeration = real[real["method"] == ENUMERATION]
            optimum = (pd.to_numeric(enumeration["sharpe_is"],
                                     errors="coerce").median()
                       if not enumeration.empty else None)
            figure3_convergence(trajectories, optimum, out_dir)
    if not synth.empty:
        figure4_risk_return(synth, out_dir)
        figure5_sharpe_distribution(synth, out_dir)
