#!/usr/bin/env python3
"""
Generate Figure 1 and Figure 2 for the PEARC paper.

Figure 1 — Static θ threshold tradeoff curve
  X-axis: θ (0.0 → 1.0)
  Left Y-axis:  HIGH recall, macro-F1
  Right Y-axis: cloud routing rate (cost proxy)
  Vertical lines mark θ=0.5 (paper default) and the score-optimal θ*.

Figure 2 — Budget-aware adaptive routing simulation
  Two subplots:
    (a) θ_eff over time: flat at θ_base until spend fraction crosses it, then rises.
        Shows exactly when/how the adaptive mechanism activates.
    (b) Cumulative spend: fixed vs adaptive, with budget cap line.
        Fixed θ visibly overshoots; adaptive stays under.

Input files:
  results/theta_sweep.json
  results/budget_simulation.json

Output:
  <out_dir>/figure1_theta_curve.pdf  (.png)
  <out_dir>/figure2_budget_routing.pdf  (.png)

Usage:
  python plot_figures.py
  python plot_figures.py --show
"""

import argparse
import json
from pathlib import Path

import numpy as np

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

SWEEP_PATH = RESULTS_DIR / "theta_sweep.json"
SIM_PATH = RESULTS_DIR / "budget_simulation.json"

# ACM sigconf single-column width = 3.33 in
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

    # Score-optimal θ*: maximise HIGH recall per unit cloud cost
    score = np.array(high_recall) - np.array(cloud_rate)
    best_idx = int(np.argmax(score))
    best_theta = thetas[best_idx]

    # Single-column width, taller than before so labels don't crowd
    fig, ax1 = plt.subplots(figsize=(FIG_WIDTH_SINGLE, 2.6))

    color_recall = "#d62728"  # red
    color_f1 = "#1f77b4"  # blue
    color_cloud = "#ff7f0e"  # orange

    (l1,) = ax1.plot(thetas, high_recall, color=color_recall, lw=1.6, label="HIGH recall")
    (l2,) = ax1.plot(thetas, macro_f1, color=color_f1, lw=1.6, linestyle="--", label="Macro-F1")
    ax1.set_xlabel(r"Threshold $\theta$", fontsize=8)
    ax1.set_ylabel("Recall / Macro-F1", fontsize=8)
    ax1.tick_params(axis="both", labelsize=7)
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0, 1.05)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

    # Right axis: cloud routing rate
    ax2 = ax1.twinx()
    (l3,) = ax2.plot(
        thetas, cloud_rate, color=color_cloud, lw=1.6, linestyle=":", label="Cloud routing rate"
    )
    ax2.set_ylabel("Cloud routing rate", fontsize=8, color=color_cloud)
    ax2.tick_params(axis="y", labelcolor=color_cloud, labelsize=7)
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

    # Mark θ=0.5 (paper default) with a solid vertical line
    ax1.axvline(0.5, color="#555555", lw=0.9, linestyle="-", alpha=0.8)
    ax1.text(
        0.5 + 0.02,
        0.92,
        r"$\theta{=}0.5$" + "\n(default)",
        fontsize=6.5,
        color="#555555",
        transform=ax1.get_xaxis_transform(),
        va="top",
    )

    # Mark score-optimal θ* with a dashed line (only if meaningfully different from 0.5)
    if abs(best_theta - 0.5) > 0.05:
        ax1.axvline(best_theta, color="gray", lw=0.8, linestyle="--", alpha=0.6)
        ax1.text(
            best_theta + 0.02,
            0.72,
            r"$\theta^*{=}$" + f"{best_theta:.2f}",
            fontsize=6.5,
            color="gray",
            transform=ax1.get_xaxis_transform(),
            va="top",
        )

    lines = [l1, l2, l3]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="lower left", fontsize=6.5, framealpha=0.9, handlelength=1.8)

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
    theta_base = sim["theta_base"]

    t = [d["period_fraction"] for d in adaptive_days]
    a_theta = [d["theta_eff"] for d in adaptive_days]
    a_recall = [d["high_recall"] for d in adaptive_days]
    f_recall_val = fixed_days[0]["high_recall"]  # constant for fixed θ
    f_cum = [d["spend_fraction"] for d in fixed_days]
    a_cum = [d["spend_fraction"] for d in adaptive_days]

    # Single-column width, two stacked subplots
    fig = plt.figure(figsize=(FIG_WIDTH_SINGLE, 3.6))
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.52)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1])

    # ---- (a) θ_eff and HIGH recall over time ----
    activation_frac = sim.get("summary", {}).get("theta_activation_period_frac", None)

    color_recall = "#d62728"  # red for recall

    ax_top.plot(t, a_theta, color="#1f77b4", lw=1.6, label=r"Adaptive $\theta_\mathrm{eff}$")
    ax_top.axhline(
        theta_base,
        color="#ff7f0e",
        lw=1.2,
        linestyle="--",
        label=r"Fixed $\theta{=}$" + f"{theta_base}",
    )
    ax_top.fill_between(
        t,
        theta_base,
        a_theta,
        where=[th > theta_base for th in a_theta],
        alpha=0.18,
        color="#1f77b4",
    )

    # Right axis: HIGH recall (adaptive drops, fixed stays flat)
    ax_top_r = ax_top.twinx()
    ax_top_r.plot(
        t, a_recall, color=color_recall, lw=1.4, linestyle="-.", label="Adaptive HIGH recall"
    )
    ax_top_r.axhline(
        f_recall_val,
        color=color_recall,
        lw=1.0,
        linestyle=":",
        alpha=0.7,
        label=f"Fixed recall ({f_recall_val:.0%})",
    )
    ax_top_r.set_ylabel("HIGH recall", fontsize=7, color=color_recall)
    ax_top_r.tick_params(axis="y", labelcolor=color_recall, labelsize=6.5)
    ax_top_r.set_ylim(0, 1.05)
    ax_top_r.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

    ax_top.set_ylabel(r"$\theta_\mathrm{eff}$", fontsize=8)
    ax_top.set_xlabel("")
    ax_top.tick_params(labelsize=7)
    ax_top.set_xlim(0, 1)
    ax_top.set_ylim(0.3, 1.05)
    ax_top.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

    # Combined legend — 2 columns, bottom of panel to keep upper area clear
    lines_l, labels_l = ax_top.get_legend_handles_labels()
    lines_r, labels_r = ax_top_r.get_legend_handles_labels()
    ax_top.legend(
        lines_l + lines_r,
        labels_l + labels_r,
        fontsize=5.8,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        framealpha=0.9,
        handlelength=1.4,
        ncol=2,
        columnspacing=0.8,
    )
    ax_top.set_title(
        r"(a) $\theta_\mathrm{eff}$ and HIGH recall over budget period", fontsize=7.5, pad=3
    )

    if activation_frac is not None:
        ax_top.axvline(activation_frac, color="gray", lw=0.8, linestyle=":", alpha=0.8)
        ax_top.text(
            activation_frac + 0.03,
            0.98,
            f"activates ({activation_frac*100:.0f}%)",
            fontsize=6,
            color="gray",
            ha="left",
            va="top",
            transform=ax_top.get_xaxis_transform(),
        )

    # ---- (b) Cumulative spend: fixed overshoots, adaptive stays under ----
    ax_bot.plot(t, f_cum, color="#ff7f0e", lw=1.6, label=r"Fixed $\theta$")
    ax_bot.plot(t, a_cum, color="#1f77b4", lw=1.6, linestyle="--", label=r"Adaptive $\theta$")
    ax_bot.axhline(1.0, color="#d62728", lw=1.2, linestyle=":", alpha=0.85, label="Budget cap")
    # Blue shading: savings region (adaptive below fixed)
    ax_bot.fill_between(
        t,
        a_cum,
        f_cum,
        where=[a < f for a, f in zip(a_cum, f_cum, strict=False)],
        alpha=0.30,
        color="#1f77b4",
        label="Savings",
    )

    # Orange shading: overshoot region (fixed above cap)
    ax_bot.fill_between(
        t,
        1.0,
        f_cum,
        where=[f > 1.0 for f in f_cum],
        alpha=0.55,
        color="#ff7f0e",
        label="Overshoot",
    )

    # Overshoot annotation — place inside the visible area with arrow pointing to the peak
    overshoot_pct = (max(f_cum) - 1.0) * 100
    overshoot_t = t[f_cum.index(max(f_cum))]
    ylim_top = max(f_cum) + 0.08
    ax_bot.annotate(
        f"+{overshoot_pct:.0f}%",
        xy=(overshoot_t, max(f_cum)),
        xytext=(overshoot_t - 0.20, max(f_cum) - 0.10),
        arrowprops={"arrowstyle": "->", "color": "#ff7f0e", "lw": 0.8},
        fontsize=6.5,
        color="#ff7f0e",
        ha="center",
    )

    ax_bot.set_ylabel("Spend / Budget", fontsize=8)
    ax_bot.set_xlabel("Budget period", fontsize=8)
    ax_bot.tick_params(labelsize=7)
    ax_bot.set_xlim(0, 1)
    ax_bot.set_ylim(0, ylim_top)
    ax_bot.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax_bot.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax_bot.legend(fontsize=6.5, loc="lower right", framealpha=0.9, handlelength=1.6)
    ax_bot.set_title("(b) Cumulative cloud spend vs. budget cap", fontsize=7.5, pad=3)

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
        idx = min(range(len(thetas)), key=lambda i: abs(thetas[i] - t))
        print(
            f"  {thetas[idx]:>5.2f}"
            f"  {high_recall[idx]:>12.1%}"
            f"  {macro_f1[idx]:>9.3f}"
            f"  {cloud_rate[idx]:>11.1%}"
        )

    score = np.array(high_recall) - np.array(cloud_rate)
    best = int(np.argmax(score))
    print(
        f"\n  Score-optimal θ* = {thetas[best]:.2f}  "
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
                "font.size": 8,
                "axes.linewidth": 0.7,
                "xtick.major.width": 0.5,
                "ytick.major.width": 0.5,
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
        else:
            with open(sim_path) as f:
                sim_data = json.load(f)
            print("\nGenerating Figure 2...")
            plot_figure2(sim_data, out_dir, args.show)

    print("\nDone. Figures in:", out_dir)


if __name__ == "__main__":
    main()
