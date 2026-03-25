#!/usr/bin/env python3
"""
STREAM LaTeX Table Generator

Reads benchmark results JSON files and outputs LaTeX-formatted table rows
that can be pasted directly into the PEARC 2026 paper.

Usage:
    python scripts/eval/generate_tables.py                          # Auto-find latest results
    python scripts/eval/generate_tables.py --latency results/latency_2026-03-06.json
    python scripts/eval/generate_tables.py --routing results/routing_2026-03-07.json

Output is printed to stdout. Copy-paste into the paper's .tex file.
"""

import argparse
import json
from pathlib import Path


def find_latest_result(results_dir: Path, prefix: str) -> Path | None:
    """Find the most recent results file with the given prefix."""
    files = sorted(results_dir.glob(f"{prefix}_*.json"), reverse=True)
    return files[0] if files else None


# ---------------------------------------------------------------------------
# Table 2: Response Latency by Tier
# ---------------------------------------------------------------------------


def generate_latency_table(data: dict) -> str:
    """Generate LaTeX rows for Table 2 (Response Latency by Tier)."""

    lines = [
        "% Table 2: Response latency by tier",
        "% Columns: Tier & TTFT (s) & Total Time (s) & Throughput (tok/s) \\\\",
        "",
    ]

    # Hardware info (for paper's experimental setup section)
    hw = data.get("hardware", {})
    local_hw = hw.get("local", hw) if isinstance(hw.get("local"), dict) else hw
    lake_hw = hw.get("lakeshore", {})
    cloud_hw = hw.get("cloud", {})

    if local_hw.get("cpu"):
        lines.append(
            f"% Local hardware: {local_hw.get('cpu', 'N/A')}, "
            f"{local_hw.get('ram', 'N/A')} RAM, "
            f"{local_hw.get('gpu', 'no GPU')}"
        )
        lines.append(f"% Local OS: {local_hw.get('os', 'N/A')}")
    if lake_hw.get("gpu"):
        lines.append(
            f"% Lakeshore hardware: {lake_hw.get('gpu')}, "
            f"node {lake_hw.get('node', 'N/A')}, "
            f"{lake_hw.get('framework', 'vLLM')}"
        )
    if cloud_hw.get("note"):
        lines.append(f"% Cloud: {cloud_hw.get('model', 'N/A')} " f"({cloud_hw.get('note')})")
    lines.append("")

    tier_labels = {
        "local": "Local (Llama 3.2 3B)",
        "lakeshore": "Lakeshore (relay streaming)",
        "cloud": "Cloud (Claude Sonnet 4)",
    }

    tiers_data = data.get("tiers", {})

    for tier_key, label in tier_labels.items():
        stats = tiers_data.get(tier_key)
        if not stats or stats.get("error"):
            lines.append(f"{label:<35s} & --- & --- & --- \\\\")
            continue

        ttft = stats["ttft"]["median"]
        ttft_std = stats["ttft"]["stdev"]
        total = stats["total_time"]["median"]
        total_std = stats["total_time"]["stdev"]
        tp = stats["throughput"]["median"]

        if tp:
            lines.append(
                f"{label:<35s} & "
                f"{ttft:.2f} $\\pm$ {ttft_std:.2f} & "
                f"{total:.2f} $\\pm$ {total_std:.2f} & "
                f"{tp:.1f} \\\\"
            )
        else:
            lines.append(
                f"{label:<35s} & "
                f"{ttft:.2f} $\\pm$ {ttft_std:.2f} & "
                f"{total:.2f} $\\pm$ {total_std:.2f} & "
                f"--- \\\\"
            )

    # Add batch mode row if relay comparison data exists
    rc = data.get("relay_comparison")
    if rc and rc.get("batch", {}).get("ttft_median"):
        batch = rc["batch"]
        ttft = batch["ttft_median"]
        ttft_std = batch["ttft_stdev"]
        total = batch["total_median"]
        total_std = batch.get("total_stdev", ttft_std)

        lines.append(
            f"{'Lakeshore (batch fallback)':<35s} & "
            f"{ttft:.2f} $\\pm$ {ttft_std:.2f} & "
            f"{total:.2f} $\\pm$ {total_std:.2f} & "
            f"--- \\\\"
        )

    # Add relay speedup note
    if rc and rc.get("ttft_speedup"):
        lines.append("")
        lines.append(f"% Relay streaming TTFT speedup: {rc['ttft_speedup']:.1f}x")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 3: Routing Accuracy
# ---------------------------------------------------------------------------


def generate_routing_table(data: dict) -> str:
    """Generate LaTeX rows for Table 3 (Routing Accuracy)."""

    summary = data.get("summary", {})

    lines = [
        "% Table 3: Routing accuracy and cost savings",
        "",
    ]

    # Overall accuracy
    accuracy = summary.get("accuracy", 0) * 100
    correct = summary.get("correct", 0)
    total = summary.get("total_queries", 0)
    strategy = summary.get("strategy", "unknown")

    lines.append(f"% Judge strategy: {strategy}")
    lines.append(f"% Overall accuracy: {correct}/{total} ({accuracy:.1f}\\%)")
    lines.append("")

    # Per-class accuracy rows
    lines.append("% Per-class accuracy rows:")
    lines.append("% Columns: Class & Correct & Total & Accuracy \\\\")
    per_class = summary.get("per_class", {})
    for label in ["LOW", "MEDIUM", "HIGH"]:
        pc = per_class.get(label, {})
        c = pc.get("correct", 0)
        t = pc.get("total", 0)
        a = pc.get("accuracy", 0) * 100
        lines.append(f"{label:<8s} & {c} & {t} & {a:.1f}\\% \\\\")

    lines.append("")

    # Confusion matrix
    lines.append("% Confusion matrix (for supplementary material):")
    lines.append("% Columns: & Pred LOW & Pred MED & Pred HIGH \\\\")
    confusion = summary.get("confusion_matrix", {})
    for actual in ["LOW", "MEDIUM", "HIGH"]:
        row = confusion.get(actual, {})
        lines.append(
            f"Actual {actual:<6s} & "
            f"{row.get('LOW', 0)} & "
            f"{row.get('MEDIUM', 0)} & "
            f"{row.get('HIGH', 0)} \\\\"
        )

    lines.append("")

    # Cost analysis
    ca = summary.get("cost_analysis", {})
    lines.append("% Cost analysis:")
    lines.append(
        f"% Queries on free tiers: {ca.get('queries_on_free_tiers', 0)}/{total} "
        f"({ca.get('queries_on_free_pct', 0):.1f}\\%)"
    )
    lines.append(f"% Estimated cost (auto):      ${ca.get('estimated_cost_auto', 0):.4f}")
    lines.append(f"% Estimated cost (all cloud): ${ca.get('estimated_cost_all_cloud', 0):.4f}")
    lines.append(f"% Cost savings:               {ca.get('cost_savings_pct', 0):.1f}\\%")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 4: Compression Impact
# ---------------------------------------------------------------------------


def generate_compression_table(data: dict) -> str:
    """Generate LaTeX rows for compression impact results."""

    summary = data.get("summary", {})

    lines = [
        "% Compression impact results",
        "% Simple queries staying on local tier at each probe turn",
        "% Columns: Turn & Local Retention \\\\",
        "",
    ]

    per_turn = summary.get("per_turn", {})
    for turn_str, stats in sorted(per_turn.items(), key=lambda x: int(x[0])):
        pct = stats["local_pct"]
        stayed = stats["stayed_local"]
        total = stats["total"]
        lines.append(f"Turn {turn_str:<4s} & {stayed}/{total} ({pct:.0f}\\%) \\\\")

    lines.append("")

    forced = summary.get("forced_upgrades", 0)
    total_probes = summary.get("total_probes", 0)
    retention = summary.get("local_retention_pct", 0)
    lines.append(f"% Forced tier upgrades: {forced}/{total_probes}")
    lines.append(f"% Local retention: {retention:.0f}\\%")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from STREAM benchmark results"
    )
    parser.add_argument("--latency", type=str, help="Path to latency results JSON")
    parser.add_argument("--routing", type=str, help="Path to routing results JSON")
    parser.add_argument("--compression", type=str, help="Path to compression results JSON")
    args = parser.parse_args()

    results_dir = Path(__file__).parent / "results"

    print("=" * 60)
    print("  STREAM LaTeX Table Generator")
    print("=" * 60)

    # Latency table
    latency_file = (
        Path(args.latency) if args.latency else find_latest_result(results_dir, "latency")
    )
    if latency_file and latency_file.exists():
        with open(latency_file) as f:
            data = json.load(f)
        print(f"\n--- Latency Table (from {latency_file.name}) ---\n")
        print(generate_latency_table(data))
    else:
        print("\n  No latency results found. Run benchmark_latency.py first.")

    # Routing table
    routing_file = (
        Path(args.routing) if args.routing else find_latest_result(results_dir, "routing")
    )
    if routing_file and routing_file.exists():
        with open(routing_file) as f:
            data = json.load(f)
        print(f"\n--- Routing Table (from {routing_file.name}) ---\n")
        print(generate_routing_table(data))
    else:
        print("\n  No routing results found. Run benchmark_routing.py first.")

    # Compression table
    compression_file = (
        Path(args.compression)
        if args.compression
        else find_latest_result(results_dir, "compression")
    )
    if compression_file and compression_file.exists():
        with open(compression_file) as f:
            data = json.load(f)
        print(f"\n--- Compression Table (from {compression_file.name}) ---\n")
        print(generate_compression_table(data))
    else:
        print("\n  No compression results found. Run benchmark_compression.py first.")

    print()


if __name__ == "__main__":
    main()
