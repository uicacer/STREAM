#!/usr/bin/env python3
"""
Compare vLLM Baseline vs Optimized Throughput Results

Reads the JSON results from vllm_throughput_test.sh and generates:
1. A side-by-side comparison table
2. LaTeX table rows for the paper
3. The speedup factor (for the "3 to 25 tok/s" claim)

Usage:
    python scripts/eval/compare_throughput.py
    python scripts/eval/compare_throughput.py --baseline results/throughput_baseline_*.json
    python scripts/eval/compare_throughput.py --optimized results/throughput_optimized_*.json
"""

import argparse
import json
import sys
from pathlib import Path


def find_latest_result(results_dir: Path, label: str) -> Path | None:
    """Find the most recent results file with the given label."""
    files = sorted(results_dir.glob(f"throughput_{label}_*.json"), reverse=True)
    return files[0] if files else None


def main():
    parser = argparse.ArgumentParser(description="Compare vLLM throughput results")
    parser.add_argument("--baseline", type=str, help="Path to baseline results JSON")
    parser.add_argument("--optimized", type=str, help="Path to optimized results JSON")
    args = parser.parse_args()

    results_dir = Path(__file__).parent / "results"

    baseline_file = (
        Path(args.baseline) if args.baseline else find_latest_result(results_dir, "baseline")
    )
    optimized_file = (
        Path(args.optimized) if args.optimized else find_latest_result(results_dir, "optimized")
    )

    if not baseline_file or not baseline_file.exists():
        print("ERROR: No baseline results found.")
        print("Run: bash scripts/eval/vllm_throughput_test.sh <host> <port> baseline")
        sys.exit(1)

    if not optimized_file or not optimized_file.exists():
        print("ERROR: No optimized results found.")
        print("Run: bash scripts/eval/vllm_throughput_test.sh <host> <port> optimized")
        sys.exit(1)

    with open(baseline_file) as f:
        baseline = json.load(f)
    with open(optimized_file) as f:
        optimized = json.load(f)

    b_tp = baseline["results"]["throughput"]
    o_tp = optimized["results"]["throughput"]

    speedup = o_tp["median"] / b_tp["median"] if b_tp["median"] > 0 else 0

    print("=" * 70)
    print("  vLLM Throughput: Baseline vs Optimized")
    print("=" * 70)
    print()
    print(f"  Hardware: {baseline.get('hardware', {}).get('gpu', 'N/A')}")
    print(f"  Model:    {baseline.get('model', 'N/A')}")
    print()

    # Configuration comparison
    b_cfg = baseline.get("config", {}).get("baseline", {})
    o_cfg = optimized.get("config", {}).get("optimized", {})

    print(f"  {'Configuration':<30s}  {'Baseline':<25s}  {'Optimized':<25s}")
    print(f"  {'-'*80}")
    print(
        f"  {'Container':<30s}  {b_cfg.get('container', 'N/A'):<25s}  {o_cfg.get('container', 'N/A'):<25s}"
    )
    print(
        f"  {'Quantization':<30s}  {b_cfg.get('quantization', 'N/A'):<25s}  {o_cfg.get('quantization', 'N/A'):<25s}"
    )
    print(
        f"  {'Context length':<30s}  {str(b_cfg.get('context_length', 'N/A')):<25s}  {str(o_cfg.get('context_length', 'N/A')):<25s}"
    )
    print(
        f"  {'GPU memory utilization':<30s}  {str(b_cfg.get('gpu_memory_utilization', 'N/A')):<25s}  {str(o_cfg.get('gpu_memory_utilization', 'N/A')):<25s}"
    )
    print(
        f"  {'Prefix caching':<30s}  {str(b_cfg.get('prefix_caching', 'N/A')):<25s}  {str(o_cfg.get('prefix_caching', 'N/A')):<25s}"
    )
    print(
        f"  {'Chunked prefill':<30s}  {str(b_cfg.get('chunked_prefill', 'N/A')):<25s}  {str(o_cfg.get('chunked_prefill', 'N/A')):<25s}"
    )
    print()

    # Results comparison
    print(f"  {'Metric':<30s}  {'Baseline':<25s}  {'Optimized':<25s}")
    print(f"  {'-'*80}")
    print(
        f"  {'Throughput (median)':<30s}  {b_tp['median']:.1f} tok/s{'':<17s}  {o_tp['median']:.1f} tok/s"
    )
    print(
        f"  {'Throughput (mean)':<30s}  {b_tp['mean']:.1f} tok/s{'':<17s}  {o_tp['mean']:.1f} tok/s"
    )
    print(
        f"  {'Throughput (stdev)':<30s}  {b_tp['stdev']:.1f} tok/s{'':<17s}  {o_tp['stdev']:.1f} tok/s"
    )
    print(
        f"  {'Throughput (range)':<30s}  {b_tp['min']:.1f}-{b_tp['max']:.1f} tok/s{'':<12s}  {o_tp['min']:.1f}-{o_tp['max']:.1f} tok/s"
    )
    print()
    print(f"  {'SPEEDUP':<30s}  {speedup:.1f}x")
    print()

    # Verify the paper's claim
    print("=" * 70)
    print("  PAPER CLAIM VERIFICATION")
    print("=" * 70)
    print()
    print('  Claim: "improved throughput from 3 to 25 tok/s"')
    print()
    print(f"  Measured baseline:  {b_tp['median']:.1f} tok/s (median)")
    print(f"  Measured optimized: {o_tp['median']:.1f} tok/s (median)")
    print(f"  Measured speedup:   {speedup:.1f}x")
    print()

    if b_tp["median"] <= 5 and o_tp["median"] >= 20:
        print("  VERDICT: CLAIM VERIFIED")
        print("  Baseline is in the ~3 tok/s range, optimized is in the ~25 tok/s range.")
    elif b_tp["median"] <= 5:
        print("  VERDICT: PARTIALLY VERIFIED")
        print(
            f"  Baseline confirmed (~3 tok/s), but optimized is {o_tp['median']:.1f} tok/s (expected ~25)."
        )
    elif o_tp["median"] >= 20:
        print("  VERDICT: PARTIALLY VERIFIED")
        print(
            f"  Optimized confirmed (~25 tok/s), but baseline is {b_tp['median']:.1f} tok/s (expected ~3)."
        )
    else:
        print("  VERDICT: NEEDS REVIEW")
        print(
            f"  Results differ from claim. Baseline: {b_tp['median']:.1f}, Optimized: {o_tp['median']:.1f}"
        )

    print()

    # LaTeX output
    print("=" * 70)
    print("  LaTeX Table Row (for paper)")
    print("=" * 70)
    print()
    print("% Throughput comparison: Baseline vs Optimized vLLM on H100 NVL")
    print(f"% Model: {baseline.get('model', 'N/A')}")
    print(
        f"Plain AWQ (pre-built container)     & {b_tp['median']:.1f} $\\pm$ {b_tp['stdev']:.1f} \\\\"
    )
    print(
        f"Marlin AWQ + optimizations          & {o_tp['median']:.1f} $\\pm$ {o_tp['stdev']:.1f} \\\\"
    )
    print(f"% Speedup: {speedup:.1f}x")
    print()

    # Optimization breakdown (for paper narrative)
    print("% Optimization breakdown (for paper narrative):")
    print("% 1. Marlin AWQ kernel: plain AWQ -> Marlin = ~2x (3 -> 6-8 tok/s)")
    print("% 2. Runtime optimizations (prefix caching + chunked prefill + memory tuning):")
    print("%    6-8 -> ~25 tok/s = ~3-4x additional improvement")
    print(f"% 3. Combined: {speedup:.1f}x total improvement")
    print()


if __name__ == "__main__":
    main()
