#!/usr/bin/env python3
"""
STREAM Latency & Throughput Benchmark (Day 1)

Measures TTFT, total response time, and throughput across all tiers.
Also compares relay streaming vs batch mode for the Lakeshore tier.

Usage:
    python scripts/eval/benchmark_latency.py                  # Full benchmark (20 runs)
    python scripts/eval/benchmark_latency.py --runs 3          # Quick test (3 runs)
    python scripts/eval/benchmark_latency.py --tiers local     # Specific tier only
    python scripts/eval/benchmark_latency.py --relay-comparison-only  # Lakeshore relay vs batch

Results are saved to scripts/eval/results/latency_<timestamp>.json
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
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

# A single representative query used for all tiers. Content is held constant
# so latency differences reflect tier performance, not prompt length effects.
BENCHMARK_QUERY = "Explain the concept of recursion in programming. Give a clear example."

# Use exact tier names ("local", "lakeshore", "cloud") to bypass the complexity
# judge and measure pure tier inference latency. Routing overhead (judge) is
# reported separately from the routing benchmark (avg ~0.68s).
TIER_MODELS = {
    "local": "local",
    "lakeshore": "lakeshore",
    "cloud": "cloud",
}

# ---------------------------------------------------------------------------
# Hardware Detection
# ---------------------------------------------------------------------------


def detect_hardware() -> dict:
    """Auto-detect local hardware for reproducibility reporting."""
    hw = {
        "os": "",
        "cpu": "",
        "ram": "",
        "gpu": None,
    }

    # OS
    system = platform.system()
    if system == "Darwin":
        ver = platform.mac_ver()[0]
        hw["os"] = f"macOS {ver} ({platform.machine()})"
    elif system == "Linux":
        try:
            with open("/etc/os-release") as f:
                for line in f:
                    if line.startswith("PRETTY_NAME"):
                        hw["os"] = line.split("=")[1].strip().strip('"')
                        break
        except Exception:
            hw["os"] = f"Linux {platform.release()}"
    else:
        hw["os"] = f"{system} {platform.release()}"

    # CPU
    try:
        if system == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                hw["cpu"] = result.stdout.strip()
            else:
                # Apple Silicon
                result = subprocess.run(
                    ["system_profiler", "SPHardwareDataType"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                for line in result.stdout.split("\n"):
                    if "Chip" in line and ":" in line:
                        hw["cpu"] = f"Apple {line.split(':')[1].strip()}"
                        break
        elif system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line.lower():
                        hw["cpu"] = line.split(":")[1].strip()
                        break
    except Exception:
        hw["cpu"] = platform.processor() or "Unknown"

    # RAM
    try:
        if system == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                ram_gb = int(result.stdout.strip()) / (1024**3)
                hw["ram"] = f"{ram_gb:.0f} GB"
        elif system == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        hw["ram"] = f"{kb / (1024**2):.0f} GB"
                        break
    except Exception:
        hw["ram"] = "Unknown"

    # GPU
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 2:
                hw["gpu"] = f"{parts[0].strip()} {int(parts[1].strip()) // 1024}GB"
    except FileNotFoundError:
        pass

    if not hw["gpu"] and system == "Darwin":
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in result.stdout.split("\n"):
                if "Chipset Model" in line:
                    hw["gpu"] = f"{line.split(':')[1].strip()} (Metal)"
                    break
        except Exception:
            pass

    return hw


# ---------------------------------------------------------------------------
# SSE Parsing Helpers
# ---------------------------------------------------------------------------


def send_streaming_request(model: str, query: str, timeout: float = 180.0) -> dict:
    """
    Send a streaming chat request to STREAM and measure timing.

    Returns:
        {
            "ttft": float,          # Time to first content token (seconds)
            "total_time": float,    # Total response time (seconds)
            "output_tokens": int,   # Number of output tokens (from cost summary)
            "throughput": float,    # Tokens per second
            "tier": str,            # Which tier actually handled it
            "model": str,           # Which model was used
            "content_length": int,  # Response length in characters
            "error": str | None,    # Error message if failed
        }
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": query}],
        "stream": True,
    }

    t_start = time.perf_counter()
    ttft = None
    content_parts = []
    output_tokens = 0
    tier = None
    actual_model = None
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

                data_str = line[6:]  # Strip "data: " prefix

                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Stream metadata (tier, model info)
                if "stream_metadata" in data:
                    meta = data["stream_metadata"]
                    if "tier" in meta:
                        tier = meta["tier"]
                    if "model" in meta:
                        actual_model = meta["model"]
                    # Cost summary at the end
                    if "cost" in meta:
                        cost_info = meta["cost"]
                        output_tokens = cost_info.get("output_tokens", 0)
                    continue

                # Content tokens
                choices = data.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        if ttft is None:
                            ttft = time.perf_counter() - t_start
                        content_parts.append(content)

    except httpx.TimeoutException:
        error = "Request timed out"
    except httpx.ConnectError:
        error = f"Cannot connect to STREAM at {STREAM_URL}"
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    t_total = time.perf_counter() - t_start
    content_text = "".join(content_parts)

    # Calculate throughput
    throughput = output_tokens / t_total if output_tokens > 0 and t_total > 0 else 0

    return {
        "ttft": ttft,
        "total_time": t_total,
        "output_tokens": output_tokens,
        "throughput": throughput,
        "tier": tier,
        "model": actual_model,
        "content_length": len(content_text),
        "error": error,
    }


def send_batch_request(model: str, query: str, timeout: float = 180.0) -> dict:
    """
    Simulate batch mode by consuming the full SSE stream before recording TTFT.

    The STREAM server always returns SSE (even with stream=False). In real batch
    mode (no relay), Globus Compute returns the complete response at once, so the
    user sees nothing until all tokens arrive. We simulate this by reading the
    entire stream and setting TTFT = total time.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": query}],
        "temperature": 0.7,
        "stream": True,
    }

    t_start = time.perf_counter()
    output_tokens = 0
    content_parts = []
    error = None
    actual_model = model

    try:
        with (
            httpx.Client(timeout=timeout) as client,
            client.stream("POST", CHAT_ENDPOINT, json=payload) as response,
        ):
            if response.status_code != 200:
                return {"error": f"HTTP {response.status_code}"}

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

                if "stream_metadata" in data:
                    meta = data["stream_metadata"]
                    if "model" in meta:
                        actual_model = meta["model"]
                    if "cost" in meta:
                        output_tokens = meta["cost"].get("output_tokens", 0)
                    continue

                choices = data.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        content_parts.append(content)

    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    t_total = time.perf_counter() - t_start

    return {
        "ttft": t_total,  # In batch mode, TTFT = total time (user waits for everything)
        "total_time": t_total,
        "output_tokens": output_tokens,
        "throughput": output_tokens / t_total if output_tokens > 0 else 0,
        "tier": model.split("-")[0] if "-" in model else "unknown",
        "model": actual_model,
        "content_length": len("".join(content_parts)),
        "error": error,
    }


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------


def run_tier_benchmark(tier: str, model: str, runs: int, warmup: bool = True) -> dict:
    """Run latency benchmark for a single tier."""
    print(f"\n{'='*60}")
    print(f"  Benchmarking: {tier} ({model})")
    print(f"  Runs: {runs} (+1 warmup)" if warmup else f"  Runs: {runs}")
    print(f"{'='*60}")

    results = []
    query = BENCHMARK_QUERY

    total_runs = runs + (1 if warmup else 0)
    for i in range(total_runs):
        is_warmup = warmup and i == 0
        label = "warmup" if is_warmup else f"run {i if not warmup else i}"

        print(f"  [{label}] ", end="", flush=True)
        result = send_streaming_request(model, query)

        if result.get("error"):
            print(f"ERROR: {result['error']}")
            if not is_warmup:
                results.append(result)
            continue

        ttft_val = result.get("ttft")
        print(
            f"TTFT={ttft_val:.2f}s  " if ttft_val is not None else "TTFT=N/A  ",
            f"Total={result['total_time']:.2f}s  "
            f"Tokens={result['output_tokens']}  "
            f"Throughput={result['throughput']:.1f} tok/s",
            sep="",
        )

        if not is_warmup:
            results.append(result)

    # Calculate statistics
    successful = [r for r in results if not r.get("error") and r.get("ttft") is not None]
    failed = len(results) - len(successful)

    if not successful:
        return {
            "tier": tier,
            "model": model,
            "runs": runs,
            "successful": 0,
            "failed": failed,
            "error": "All runs failed",
        }

    ttfts = [r["ttft"] for r in successful if r["ttft"] is not None]
    totals = [r["total_time"] for r in successful]
    throughputs = [r["throughput"] for r in successful if r["throughput"] > 0]

    stats = {
        "tier": tier,
        "model": model,
        "runs": runs,
        "successful": len(successful),
        "failed": failed,
        "ttft": {
            "median": statistics.median(ttfts) if ttfts else None,
            "mean": statistics.mean(ttfts) if ttfts else None,
            "stdev": statistics.stdev(ttfts) if len(ttfts) > 1 else 0,
            "min": min(ttfts) if ttfts else None,
            "max": max(ttfts) if ttfts else None,
            "values": ttfts,
        },
        "total_time": {
            "median": statistics.median(totals),
            "mean": statistics.mean(totals),
            "stdev": statistics.stdev(totals) if len(totals) > 1 else 0,
            "min": min(totals),
            "max": max(totals),
            "values": totals,
        },
        "throughput": {
            "median": statistics.median(throughputs) if throughputs else None,
            "mean": statistics.mean(throughputs) if throughputs else None,
            "stdev": statistics.stdev(throughputs) if len(throughputs) > 1 else 0,
            "values": throughputs,
        },
    }

    print("\n  Summary:")
    print(
        f"    TTFT:       {stats['ttft']['median']:.2f}s (median) +/- {stats['ttft']['stdev']:.2f}s"
    )
    print(
        f"    Total:      {stats['total_time']['median']:.2f}s (median) +/- {stats['total_time']['stdev']:.2f}s"
    )
    if stats["throughput"]["median"]:
        print(f"    Throughput: {stats['throughput']['median']:.1f} tok/s (median)")

    return stats


def run_relay_comparison(model: str, runs: int) -> dict:
    """Compare relay streaming vs batch mode for Lakeshore."""
    print(f"\n{'='*60}")
    print(f"  Relay vs Batch Comparison ({model})")
    print(f"  Runs: {runs} per mode")
    print(f"{'='*60}")

    # Streaming (relay)
    print("\n  --- Relay Streaming ---")
    streaming_results = []
    for i in range(runs):
        print(f"  [relay {i+1}/{runs}] ", end="", flush=True)
        result = send_streaming_request(model, BENCHMARK_QUERY)
        if result.get("error"):
            print(f"ERROR: {result['error']}")
        else:
            print(f"TTFT={result['ttft']:.2f}s  Total={result['total_time']:.2f}s")
        streaming_results.append(result)

    # Batch (no relay)
    print("\n  --- Batch Mode ---")
    batch_results = []
    for i in range(runs):
        print(f"  [batch {i+1}/{runs}] ", end="", flush=True)
        result = send_batch_request(model, BENCHMARK_QUERY)
        if result.get("error"):
            print(f"ERROR: {result['error']}")
        else:
            print(f"TTFT={result['ttft']:.2f}s (= total, all at once)")
        batch_results.append(result)

    # Statistics
    stream_ok = [r for r in streaming_results if not r.get("error")]
    batch_ok = [r for r in batch_results if not r.get("error")]

    stream_ttfts = [r["ttft"] for r in stream_ok if r["ttft"] is not None]
    batch_ttfts = [r["ttft"] for r in batch_ok if r["ttft"] is not None]

    comparison = {
        "streaming": {
            "ttft_median": statistics.median(stream_ttfts) if stream_ttfts else None,
            "ttft_stdev": statistics.stdev(stream_ttfts) if len(stream_ttfts) > 1 else 0,
            "total_median": statistics.median([r["total_time"] for r in stream_ok])
            if stream_ok
            else None,
            "successful": len(stream_ok),
        },
        "batch": {
            "ttft_median": statistics.median(batch_ttfts) if batch_ttfts else None,
            "ttft_stdev": statistics.stdev(batch_ttfts) if len(batch_ttfts) > 1 else 0,
            "total_median": statistics.median([r["total_time"] for r in batch_ok])
            if batch_ok
            else None,
            "successful": len(batch_ok),
        },
    }

    if stream_ttfts and batch_ttfts:
        speedup = statistics.median(batch_ttfts) / statistics.median(stream_ttfts)
        comparison["ttft_speedup"] = speedup
        print(f"\n  TTFT Speedup: {speedup:.1f}x faster with relay streaming")
        print(f"    Relay:  {comparison['streaming']['ttft_median']:.2f}s")
        print(f"    Batch:  {comparison['batch']['ttft_median']:.2f}s")

    return comparison


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="STREAM Latency & Throughput Benchmark")
    parser.add_argument(
        "--runs", type=int, default=20, help="Number of runs per tier (default: 20)"
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        choices=["local", "lakeshore", "cloud"],
        help="Tiers to benchmark (default: all)",
    )
    parser.add_argument(
        "--relay-comparison-only", action="store_true", help="Only run relay vs batch comparison"
    )
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup run")
    parser.add_argument("--url", type=str, help="STREAM URL (default: http://localhost:5000)")
    args = parser.parse_args()

    if args.url:
        global STREAM_URL, CHAT_ENDPOINT
        STREAM_URL = args.url
        CHAT_ENDPOINT = f"{STREAM_URL}/v1/chat/completions"

    # Detect hardware
    print("Detecting hardware...")
    hardware = detect_hardware()
    print(f"  OS:  {hardware['os']}")
    print(f"  CPU: {hardware['cpu']}")
    print(f"  RAM: {hardware['ram']}")
    print(f"  GPU: {hardware['gpu'] or 'None detected'}")

    # Check connectivity
    print(f"\nConnecting to STREAM at {STREAM_URL}...")
    try:
        r = httpx.get(f"{STREAM_URL}/health", timeout=5)
        print(f"  Connected! Status: {r.status_code}")
    except Exception as e:
        print(f"  ERROR: Cannot connect to STREAM: {e}")
        print(f"  Make sure STREAM is running at {STREAM_URL}")
        sys.exit(1)

    tiers = args.tiers or list(TIER_MODELS.keys())
    results = {
        "timestamp": datetime.now().isoformat(),
        "stream_url": STREAM_URL,
        "hardware": {
            "local": hardware,
            "lakeshore": {
                "gpu": "NVIDIA H100 NVL 94 GB",
                "node": "ghi2-002",
                "partition": "batch_gpu2",
                "model": "Qwen2.5-VL-72B-Instruct-AWQ",
                "framework": "vLLM with Marlin kernels",
            },
            "cloud": {
                "note": "Infrastructure not disclosed by providers",
                "model": "Claude Sonnet 4 via OpenRouter",
            },
        },
        "benchmark_query": BENCHMARK_QUERY,
        "runs_per_tier": args.runs,
        "tiers": {},
        "relay_comparison": None,
    }

    if not args.relay_comparison_only:
        for tier in tiers:
            model = TIER_MODELS.get(tier)
            if not model:
                print(f"\nSkipping unknown tier: {tier}")
                continue
            stats = run_tier_benchmark(tier, model, args.runs, warmup=not args.no_warmup)
            results["tiers"][tier] = stats

    # Relay vs batch comparison for Lakeshore
    if "lakeshore" in tiers or args.relay_comparison_only:
        model = TIER_MODELS["lakeshore"]
        comparison = run_relay_comparison(model, args.runs)
        results["relay_comparison"] = comparison

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_file = results_dir / f"latency_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  Results saved to: {output_file}")
    print(f"{'='*60}")

    # Print summary table
    print(f"\n{'='*60}")
    print("  SUMMARY TABLE (for paper)")
    print(f"{'='*60}")
    print(f"  {'Tier':<30} {'TTFT (s)':<15} {'Total (s)':<15} {'tok/s':<10}")
    print(f"  {'-'*70}")
    for tier, stats in results["tiers"].items():
        if stats.get("error"):
            print(f"  {tier:<30} ERROR: {stats['error']}")
            continue
        ttft = stats["ttft"]["median"]
        ttft_std = stats["ttft"]["stdev"]
        total = stats["total_time"]["median"]
        total_std = stats["total_time"]["stdev"]
        tp = stats["throughput"]["median"]
        print(
            f"  {tier:<30} "
            f"{ttft:.2f} +/- {ttft_std:.2f}  "
            f"{total:.2f} +/- {total_std:.2f}  "
            f"{tp:.1f}"
            if tp
            else "---"
        )

    if results["relay_comparison"]:
        rc = results["relay_comparison"]
        print("\n  Relay vs Batch (Lakeshore):")
        if rc["streaming"]["ttft_median"] and rc["batch"]["ttft_median"]:
            print(f"    Streaming TTFT: {rc['streaming']['ttft_median']:.2f}s")
            print(f"    Batch TTFT:     {rc['batch']['ttft_median']:.2f}s")
            print(f"    Speedup:        {rc.get('ttft_speedup', 0):.1f}x")


if __name__ == "__main__":
    main()
