#!/usr/bin/env python3
"""
Generate Figure 1 and Figure 2 for the PEARC paper.

Figure 1 — Static θ threshold tradeoff curve
  X-axis: θ (0.05 → 0.95)
  Primary Y-axis (left):  HIGH recall, macro-F1
  Secondary Y-axis (right): cloud routing rate (cost proxy)
  The intersection of HIGH recall and cloud rate highlights the operating point.

Figure 2 — Budget-aware adaptive routing simulation (30 days)
  Two subplots:
    (a) Tier routing breakdown by day: stacked area (LOCAL, HPC, CLOUD)
        for fixed-θ vs adaptive-θ — shows adaptive cutting cloud as budget depletes
    (b) Cumulative spend: fixed vs adaptive over 30 days, horizontal budget cap line

Input files (from train_balanced_classifier.py):
  results/theta_sweep.json
  results/budget_simulation.json

Output:
  figures/figure1_theta_curve.pdf
  figures/figure1_theta_curve.png
  figures/figure2_budget_routing.pdf
  figures/figure2_budget_routing.png

Usage:
  python plot_figures.py
  python plot_figures.py --show    # open interactive window
"""

import argparse
import json
from pathlib import Path

import numpy as np

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

SWEEP_PATH = RESULTS_DIR / "theta_sweep.json"
SIM_PATH = RESULTS_DIR / "budget_simulation.json"

# PEARC double-column figure width: 3.33 in per column, 7.0 in full width
FIG_WIDTH_SINGLE = 3.33
FIG_WIDTH_DOUBLE = 7.0


# ---------------------------------------------------------------------------
# Figure 1: Static θ curve
# ---------------------------------------------------------------------------


def plot_figure1(data: dict, out_dir: Path, show: bool) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    sweep = data["sweep"]
    thetas = [r["theta"] for r in sweep]
    high_recall = [r["high_recall"] for r in sweep]
    macro_f1 = [r["macro_f1"] for r in sweep]
    cloud_rate = [r["cloud_routing_rate"] for r in sweep]

    # Find the "balanced" operating point: maximize HIGH recall - cloud_rate
    # (useful tradeoff: catch most HIGH without over-routing to cloud)
    score = np.array(high_recall) - np.array(cloud_rate)
    best_idx = int(np.argmax(score))
    best_theta = thetas[best_idx]

    fig, ax1 = plt.subplots(figsize=(FIG_WIDTH_DOUBLE * 0.65, 2.8))

    # Left axis: HIGH recall and macro-F1
    color_recall = "#d62728"  # red
    color_f1 = "#1f77b4"  # blue
    color_cloud = "#ff7f0e"  # orange

    (l1,) = ax1.plot(thetas, high_recall, color=color_recall, lw=1.8, label="HIGH recall")
    (l2,) = ax1.plot(thetas, macro_f1, color=color_f1, lw=1.8, linestyle="--", label="Macro-F1")
    ax1.set_xlabel("Classification threshold θ", fontsize=9)
    ax1.set_ylabel("Recall / Macro-F1", fontsize=9, color="black")
    ax1.tick_params(axis="y", labelcolor="black", labelsize=8)
    ax1.tick_params(axis="x", labelsize=8)
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0, 1.05)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

    # Right axis: cloud routing rate
    ax2 = ax1.twinx()
    (l3,) = ax2.plot(
        thetas, cloud_rate, color=color_cloud, lw=1.8, linestyle=":", label="Cloud routing rate"
    )
    ax2.set_ylabel("Cloud routing rate", fontsize=9, color=color_cloud)
    ax2.tick_params(axis="y", labelcolor=color_cloud, labelsize=8)
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

    # Annotate recommended operating point
    ax1.axvline(best_theta, color="gray", lw=1.0, linestyle="--", alpha=0.7)
    ax1.annotate(
        f"θ*={best_theta:.2f}",
        xy=(best_theta, high_recall[best_idx]),
        xytext=(best_theta + 0.05, high_recall[best_idx] - 0.12),
        arrowprops={"arrowstyle": "->", "color": "gray", "lw": 0.8},
        fontsize=7.5,
        color="gray",
    )

    lines = [l1, l2, l3]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="lower left", fontsize=7.5, framealpha=0.9)

    ax1.set_title("Figure 1: Threshold–recall–cost tradeoff (θ sweep)", fontsize=9, pad=4)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        path = out_dir / f"figure1_theta_curve.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {path}")

    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Budget-aware adaptive routing
# ---------------------------------------------------------------------------


def plot_figure2(sim: dict, out_dir: Path, show: bool) -> None:
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    fixed_days = sim["fixed"]
    adaptive_days = sim["adaptive"]

    # Use normalized axes: period_fraction (0–1) and spend_fraction (0–1)
    t_fix = [d["period_fraction"] for d in fixed_days]
    t_adap = [d["period_fraction"] for d in adaptive_days]
    f_cloud = [d["cloud_n"] / d["total_n"] for d in fixed_days]
    a_cloud = [d["cloud_n"] / d["total_n"] for d in adaptive_days]
    f_cum = [d["spend_fraction"] for d in fixed_days]
    a_cum = [d["spend_fraction"] for d in adaptive_days]
    fig = plt.figure(figsize=(FIG_WIDTH_DOUBLE * 0.85, 4.2))
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.45)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1])

    # ---- Top: cloud routing rate comparison ----
    ax_top.plot(t_fix, f_cloud, color="#ff7f0e", lw=1.6, label="Fixed θ")
    ax_top.plot(t_adap, a_cloud, color="#1f77b4", lw=1.6, linestyle="--", label="Adaptive θ")
    ax_top.fill_between(
        t_adap,
        a_cloud,
        f_cloud,
        where=[a < f for a, f in zip(a_cloud, f_cloud, strict=False)],
        alpha=0.15,
        color="#1f77b4",
        label="Savings region",
    )
    ax_top.set_ylabel("Cloud routing rate", fontsize=9)
    ax_top.set_xlabel("")
    ax_top.tick_params(labelsize=8)
    ax_top.set_xlim(0, 1)
    ax_top.set_ylim(0, max(max(f_cloud), max(a_cloud)) * 1.2)
    ax_top.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax_top.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax_top.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    ax_top.set_title("(a) Daily cloud routing rate: fixed vs adaptive θ", fontsize=8.5, pad=3)

    # ---- Bottom: cumulative spend (normalized) ----
    ax_bot.plot(t_fix, f_cum, color="#ff7f0e", lw=1.6, label="Fixed θ")
    ax_bot.plot(t_adap, a_cum, color="#1f77b4", lw=1.6, linestyle="--", label="Adaptive θ")
    ax_bot.axhline(1.0, color="red", lw=1.2, linestyle=":", alpha=0.8, label="Budget cap (B = 1.0)")
    ax_bot.fill_between(
        t_adap,
        a_cum,
        f_cum,
        where=[a < f for a, f in zip(a_cum, f_cum, strict=False)],
        alpha=0.15,
        color="#1f77b4",
    )
    ax_bot.set_ylabel("Cumulative spend (fraction of B)", fontsize=9)
    ax_bot.set_xlabel("Budget period (fraction)", fontsize=9)
    ax_bot.tick_params(labelsize=8)
    ax_bot.set_xlim(0, 1)
    ax_bot.set_ylim(0, 1.15)
    ax_bot.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax_bot.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax_bot.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    ax_bot.set_title("(b) Cumulative cloud spend over budget period", fontsize=8.5, pad=3)

    fig.suptitle("Figure 2: Budget-aware adaptive routing", fontsize=9, y=1.01)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        path = out_dir / f"figure2_budget_routing.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {path}")

    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_summary(data: dict) -> None:
    sweep = data["sweep"]
    thetas = [r["theta"] for r in sweep]
    high_recall = [r["high_recall"] for r in sweep]
    macro_f1 = [r["macro_f1"] for r in sweep]
    cloud_rate = [r["cloud_routing_rate"] for r in sweep]

    print("\nθ sweep summary (selected thresholds):")
    print(f"  {'θ':>5}  {'HIGH recall':>12}  {'Macro-F1':>9}  {'Cloud rate':>11}")
    print(f"  {'─'*5}  {'─'*12}  {'─'*9}  {'─'*11}")
    for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        # find closest
        idx = min(range(len(thetas)), key=lambda i: abs(thetas[i] - t))
        print(
            f"  {thetas[idx]:>5.2f}"
            f"  {high_recall[idx]:>12.1%}"
            f"  {macro_f1[idx]:>9.3f}"
            f"  {cloud_rate[idx]:>11.1%}"
        )

    # Best balanced operating point
    score = np.array(high_recall) - np.array(cloud_rate)
    best = int(np.argmax(score))
    print(
        f"\n  Recommended θ* = {thetas[best]:.2f}  "
        f"HIGH recall={high_recall[best]:.1%}  "
        f"cloud rate={cloud_rate[best]:.1%}  "
        f"macro-F1={macro_f1[best]:.3f}"
    )

    default_eval = data.get("eval_at_default_theta")
    if default_eval:
        print(f"\nAt default θ={default_eval['theta']}:")
        print(
            f"  Accuracy={default_eval['accuracy']:.3f}  "
            f"Macro-F1={default_eval['macro_f1']:.3f}  "
            f"FREE-tier retention={default_eval['free_tier_retention_pct']:.1f}%"
        )
        ci = default_eval.get("wilson_ci", {})
        for cls in ["LOW", "MEDIUM", "HIGH"]:
            if cls in ci:
                c = ci[cls]
                print(
                    f"  {cls:6s} recall 95% CI: [{c['recall_lo']:.3f}, {c['recall_hi']:.3f}]"
                    f"  ({c['k']}/{c['n']})"
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", default=str(SWEEP_PATH))
    parser.add_argument("--sim", default=str(SIM_PATH))
    parser.add_argument("--out", default=str(FIGURES_DIR))
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--fig1-only", action="store_true")
    parser.add_argument("--fig2-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    do_fig1 = not args.fig2_only
    do_fig2 = not args.fig1_only

    try:
        import matplotlib

        matplotlib.rcParams.update(
            {
                "font.family": "serif",
                "font.size": 9,
                "axes.linewidth": 0.8,
                "xtick.major.width": 0.6,
                "ytick.major.width": 0.6,
                "legend.framealpha": 0.85,
                "figure.dpi": 150,
            }
        )
    except ImportError:
        print("[ERROR] matplotlib not installed. Run: pip install matplotlib")
        return

    if do_fig1:
        sweep_path = Path(args.sweep)
        if not sweep_path.exists():
            print(f"[ERROR] Sweep file not found: {sweep_path}")
            print("  Run: python scripts/eval/train_balanced_classifier.py")
        else:
            with open(sweep_path) as f:
                sweep_data = json.load(f)
            print("\nGenerating Figure 1...")
            print_summary(sweep_data)
            plot_figure1(sweep_data, out_dir, args.show)

    if do_fig2:
        sim_path = Path(args.sim)
        if not sim_path.exists():
            print(f"[ERROR] Simulation file not found: {sim_path}")
            print("  Run: python scripts/eval/train_balanced_classifier.py")
        else:
            with open(sim_path) as f:
                sim_data = json.load(f)
            print("\nGenerating Figure 2...")
            plot_figure2(sim_data, out_dir, args.show)

    print("\nDone. Figures in:", out_dir)


if __name__ == "__main__":
    main()
