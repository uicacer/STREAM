#!/usr/bin/env python3
"""
STREAM Routing Accuracy Benchmark (Day 2)

Sends 60 hand-labeled queries through the complexity judge and measures
how accurately it classifies LOW/MEDIUM/HIGH complexity.

Also estimates cost savings from correct routing vs. sending everything
to the cloud tier.

Usage:
    python scripts/eval/benchmark_routing.py                  # Full benchmark
    python scripts/eval/benchmark_routing.py --verbose         # Show per-query results
    python scripts/eval/benchmark_routing.py --strategy haiku  # Use a specific judge

Results are saved to scripts/eval/results/routing_<timestamp>.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STREAM_URL = os.getenv("STREAM_URL", "http://localhost:5000")
CHAT_ENDPOINT = f"{STREAM_URL}/v1/chat/completions"

# Cost per token for each tier (from litellm_config.yaml)
TIER_COSTS = {
    "local": {"input": 0.0, "output": 0.0},
    "lakeshore": {"input": 0.0, "output": 0.0},
    "cloud": {"input": 0.000003, "output": 0.000015},  # Claude Sonnet 4
}

# Average tokens per query (estimated for cost calculation)
AVG_INPUT_TOKENS = 50
AVG_OUTPUT_TOKENS = 200

# Complexity → tier mapping (how STREAM routes in AUTO mode)
COMPLEXITY_TO_TIER = {
    "LOW": "local",
    "MEDIUM": "lakeshore",
    "HIGH": "cloud",
}

# ---------------------------------------------------------------------------
# Judge Query
# ---------------------------------------------------------------------------


def judge_query_via_stream(query: str, timeout: float = 30.0) -> dict:
    """
    Send a query to STREAM and extract the complexity judgment and tier.

    We send a minimal streaming request and look for the stream_metadata
    SSE event, which contains the tier and complexity judgment.
    """
    payload = {
        "model": "auto",  # AUTO mode triggers the complexity judge
        "messages": [{"role": "user", "content": query}],
        "temperature": 0.7,
        "stream": True,
    }

    t_start = time.perf_counter()
    tier = None
    model = None
    complexity = None
    error = None

    try:
        with (
            httpx.Client(timeout=timeout) as client,
            client.stream("POST", CHAT_ENDPOINT, json=payload) as response,
        ):
            if response.status_code != 200:
                return {"error": f"HTTP {response.status_code}: {response.text}"}

            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Extract metadata (tier, model, complexity)
                if "stream_metadata" in data:
                    meta = data["stream_metadata"]
                    if "tier" in meta:
                        tier = meta["tier"]
                    if "model" in meta:
                        model = meta["model"]
                    if "complexity" in meta:
                        complexity = meta["complexity"]
                    # Once we have tier info, we can stop reading
                    if tier:
                        break

    except httpx.TimeoutException:
        error = "Request timed out"
    except httpx.ConnectError:
        error = f"Cannot connect to STREAM at {STREAM_URL}"
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    judge_time = time.perf_counter() - t_start

    return {
        "tier": tier,
        "model": model,
        "complexity": complexity,
        "judge_time": judge_time,
        "error": error,
    }


def judge_query_direct(query: str, strategy: str = "ollama-3b", timeout: float = 30.0) -> dict:
    """
    Call the STREAM judge endpoint directly (faster than full chat).

    Falls back to the full chat endpoint if /judge is not available.
    """
    # Try the direct judge endpoint first
    judge_endpoint = f"{STREAM_URL}/judge"
    payload = {
        "query": query,
        "strategy": strategy,
    }

    t_start = time.perf_counter()

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(judge_endpoint, json=payload)

            if response.status_code == 200:
                data = response.json()
                judge_time = time.perf_counter() - t_start
                return {
                    "complexity": data.get("complexity", "").upper(),
                    "method": data.get("method"),
                    "judge_time": judge_time,
                    "error": None,
                }
            elif response.status_code in (404, 405):
                # /judge endpoint not available, fall back to chat
                pass
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}

    except httpx.ConnectError:
        return {"error": f"Cannot connect to STREAM at {STREAM_URL}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    # Fallback: use the full chat endpoint in AUTO mode
    return judge_query_via_stream(query, timeout)


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------


def run_routing_benchmark(
    queries: list[dict],
    strategy: str = "ollama-3b",
    verbose: bool = False,
) -> dict:
    """Run routing accuracy benchmark on all queries."""

    print(f"\n{'='*60}")
    print("  Routing Accuracy Benchmark")
    print(f"  Strategy: {strategy}")
    print(f"  Queries:  {len(queries)}")
    print(f"{'='*60}")

    results = []
    correct = 0
    total = 0

    # Confusion matrix: confusion[actual][predicted] = count
    labels = ["LOW", "MEDIUM", "HIGH"]
    confusion = {a: {p: 0 for p in labels} for a in labels}

    for i, q in enumerate(queries):
        query_id = q["id"]
        text = q["text"]
        ground_truth = q["ground_truth"]

        print(f"  [{i+1:2d}/{len(queries)}] ", end="", flush=True)

        result = judge_query_direct(text, strategy)

        if result.get("error"):
            print(f"ERROR: {result['error']}")
            results.append(
                {
                    "id": query_id,
                    "text": text,
                    "ground_truth": ground_truth,
                    "predicted": None,
                    "correct": False,
                    "error": result["error"],
                }
            )
            total += 1
            continue

        predicted = (result.get("complexity") or "").upper()
        is_correct = predicted == ground_truth

        if is_correct:
            correct += 1
        total += 1

        # Update confusion matrix
        if predicted in labels and ground_truth in labels:
            confusion[ground_truth][predicted] += 1

        status = "OK" if is_correct else "MISS"
        judge_time = result.get("judge_time", 0)

        if verbose or not is_correct:
            print(
                f"{status}  truth={ground_truth:<6s}  pred={predicted:<6s}  "
                f"time={judge_time:.2f}s  "
                f"{'| ' + text[:50] + '...' if len(text) > 50 else '| ' + text}"
            )
        else:
            print(
                f"{status}  truth={ground_truth:<6s}  pred={predicted:<6s}  time={judge_time:.2f}s"
            )

        results.append(
            {
                "id": query_id,
                "text": text,
                "ground_truth": ground_truth,
                "predicted": predicted,
                "correct": is_correct,
                "judge_time": judge_time,
                "method": result.get("method"),
            }
        )

    # Per-class accuracy
    per_class = {}
    for label in labels:
        class_queries = [r for r in results if r["ground_truth"] == label]
        class_correct = [r for r in class_queries if r["correct"]]
        per_class[label] = {
            "total": len(class_queries),
            "correct": len(class_correct),
            "accuracy": len(class_correct) / len(class_queries) if class_queries else 0,
        }

    # Cost analysis
    cost_auto = 0.0
    cost_all_cloud = 0.0
    queries_on_free = 0

    for r in results:
        predicted = r.get("predicted", "HIGH")
        tier = COMPLEXITY_TO_TIER.get(predicted, "cloud")

        # Auto mode cost
        tier_cost = TIER_COSTS.get(tier, TIER_COSTS["cloud"])
        cost_auto += AVG_INPUT_TOKENS * tier_cost["input"] + AVG_OUTPUT_TOKENS * tier_cost["output"]

        # All-cloud cost
        cost_all_cloud += (
            AVG_INPUT_TOKENS * TIER_COSTS["cloud"]["input"]
            + AVG_OUTPUT_TOKENS * TIER_COSTS["cloud"]["output"]
        )

        if tier in ("local", "lakeshore"):
            queries_on_free += 1

    cost_savings = (cost_all_cloud - cost_auto) / cost_all_cloud * 100 if cost_all_cloud > 0 else 0

    # Judge latency stats
    judge_times = [r["judge_time"] for r in results if "judge_time" in r and r["judge_time"]]

    # Build summary
    summary = {
        "strategy": strategy,
        "total_queries": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "cost_analysis": {
            "queries_on_free_tiers": queries_on_free,
            "queries_on_free_pct": queries_on_free / total * 100 if total > 0 else 0,
            "estimated_cost_auto": round(cost_auto, 6),
            "estimated_cost_all_cloud": round(cost_all_cloud, 6),
            "cost_savings_pct": round(cost_savings, 1),
        },
        "judge_latency": {
            "mean": sum(judge_times) / len(judge_times) if judge_times else 0,
            "min": min(judge_times) if judge_times else 0,
            "max": max(judge_times) if judge_times else 0,
        },
        "raw_results": results,
    }

    # Print summary
    print(f"\n{'='*60}")
    print("  ROUTING ACCURACY RESULTS")
    print(f"{'='*60}")
    print(f"  Overall: {correct}/{total} correct ({summary['accuracy']*100:.1f}%)")
    print()
    print("  Per-class:")
    for label in labels:
        pc = per_class[label]
        print(f"    {label:<8s} {pc['correct']}/{pc['total']} ({pc['accuracy']*100:.1f}%)")

    print("\n  Confusion Matrix:")
    print(f"  {'':>14s}  {'Pred LOW':>10s}  {'Pred MED':>10s}  {'Pred HIGH':>10s}")
    for actual in labels:
        row = confusion[actual]
        print(
            f"  {'Actual '+actual:>14s}  {row['LOW']:>10d}  {row['MEDIUM']:>10d}  {row['HIGH']:>10d}"
        )

    print("\n  Cost Impact:")
    ca = summary["cost_analysis"]
    print(
        f"    Queries on free tiers: {ca['queries_on_free_tiers']}/{total} ({ca['queries_on_free_pct']:.1f}%)"
    )
    print(f"    Estimated cost (auto):      ${ca['estimated_cost_auto']:.4f}")
    print(f"    Estimated cost (all cloud): ${ca['estimated_cost_all_cloud']:.4f}")
    print(f"    Cost savings:               {ca['cost_savings_pct']:.1f}%")

    if judge_times:
        jl = summary["judge_latency"]
        print(f"\n  Judge Latency: {jl['mean']:.2f}s avg ({jl['min']:.2f}s - {jl['max']:.2f}s)")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="STREAM Routing Accuracy Benchmark")
    parser.add_argument("--verbose", action="store_true", help="Show per-query details")
    parser.add_argument(
        "--strategy",
        type=str,
        default="ollama-3b",
        choices=["ollama-3b", "gemma-vision", "haiku"],
        help="Judge strategy to test (default: ollama-3b)",
    )
    parser.add_argument("--url", type=str, help="STREAM URL (default: http://localhost:5000)")
    args = parser.parse_args()

    if args.url:
        global STREAM_URL, CHAT_ENDPOINT
        STREAM_URL = args.url
        CHAT_ENDPOINT = f"{STREAM_URL}/v1/chat/completions"

    # Load test queries
    queries_file = Path(__file__).parent / "test_queries.json"
    if not queries_file.exists():
        print(f"ERROR: Test queries file not found: {queries_file}")
        sys.exit(1)

    with open(queries_file) as f:
        data = json.load(f)
    queries = data["queries"]
    print(f"Loaded {len(queries)} test queries from {queries_file.name}")

    # Check connectivity
    print(f"Connecting to STREAM at {STREAM_URL}...")
    try:
        r = httpx.get(f"{STREAM_URL}/health", timeout=5)
        print(f"  Connected! Status: {r.status_code}")
    except Exception as e:
        print(f"  ERROR: Cannot connect to STREAM: {e}")
        print(f"  Make sure STREAM is running at {STREAM_URL}")
        sys.exit(1)

    # Run benchmark
    summary = run_routing_benchmark(queries, strategy=args.strategy, verbose=args.verbose)

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_file = results_dir / f"routing_{timestamp}.json"

    output = {
        "timestamp": datetime.now().isoformat(),
        "stream_url": STREAM_URL,
        "strategy": args.strategy,
        "test_queries_file": str(queries_file),
        "summary": summary,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  Results saved to: {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
