#!/usr/bin/env python3
"""
Benchmark the Llama 3.2 3B local LLM judge against the balanced test set.

Calls Ollama directly (no STREAM middleware required).
Requires: ollama running locally with llama3.2:3b pulled.

Usage:
  python scripts/eval/benchmark_llama_judge.py
  python scripts/eval/benchmark_llama_judge.py --n 100   # quick sample
  python scripts/eval/benchmark_llama_judge.py --model llama3.2:1b

Output:
  scripts/eval/results/llama_judge_report.json
"""

import argparse
import json
import time
from pathlib import Path

import httpx
import numpy as np

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2:3b"
TEST_DATASET = Path("scripts/eval/balanced_test_dataset.json")
RESULTS_DIR = Path("scripts/eval/results")

JUDGE_PROMPT = """You are a query complexity classifier for an AI routing system used by students and researchers across ALL fields (science, engineering, humanities, business, healthcare, etc.).

Classification Guidelines:

LOW complexity (simple, factual - route to local):
- Greetings and thanks (hi, hello, thank you)
- Simple definitions: "What is photosynthesis?", "Define GDP", "What is Python?"
- Single factual lookups: "Who invented the telephone?", "What year did X happen?"
- Yes/no questions with obvious answers
- One-word or very short answers expected
- No reasoning or explanation needed

MEDIUM complexity (explanations, moderate analysis - route to campus GPU):
- "Explain how X works" (single concept)
- Compare 2-3 things: "Compare Python and JavaScript"
- Step-by-step instructions or tutorials
- Basic calculations or problem-solving
- Summarize a concept or article
- Write a single function or short code snippet
- Moderate technical questions with straightforward answers

HIGH complexity (deep analysis, design, research - route to cloud):
- System design or architecture (any domain: software, business, scientific)
- Multi-factor analysis or trade-off evaluation
- Research-level questions requiring domain expertise
- Design patterns, frameworks, methodologies
- Security, scalability, optimization, or performance considerations
- Multi-step reasoning across multiple concepts or domains
- Policy analysis, strategic planning, decision frameworks
- Scientific experiment design or research methodology
- Complex debugging, troubleshooting, or root cause analysis
- Creative works requiring extensive planning (essays, stories, reports)
- Anything requiring synthesis of multiple concepts or domains
- Questions with "design", "architect", "analyze trade-offs", "evaluate", "comprehensive"

Respond with ONLY ONE WORD: LOW, MEDIUM, or HIGH

User Query: {query}"""


def call_ollama(text: str, model: str, timeout: float = 30.0) -> tuple[str, float]:
    """Call Ollama, return (label, latency_ms). Returns MEDIUM on failure."""
    prompt = JUDGE_PROMPT.format(query=text)
    t0 = time.perf_counter()
    try:
        resp = httpx.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 10},
            },
            timeout=timeout,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        raw = resp.json().get("response", "").strip().upper()
        if "LOW" in raw:
            return "LOW", latency_ms
        elif "HIGH" in raw:
            return "HIGH", latency_ms
        elif "MEDIUM" in raw:
            return "MEDIUM", latency_ms
        else:
            return "MEDIUM", latency_ms  # fallback
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        print(f"  [WARN] Ollama error: {e}")
        return "MEDIUM", latency_ms


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dataset", default=str(TEST_DATASET))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--n", type=int, default=None, help="Limit to first N queries (for quick testing)"
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load test set
    data = json.loads(Path(args.test_dataset).read_text())
    queries = data["queries"]
    if args.n:
        queries = queries[: args.n]
    print(f"Loaded {len(queries)} test queries")
    print(f"Model: {args.model}")

    # Check Ollama is reachable
    try:
        httpx.get("http://localhost:11434", timeout=3)
    except Exception:
        print("[ERROR] Ollama not reachable at localhost:11434. Is it running?")
        return

    # Run benchmark
    labels_true = []
    labels_pred = []
    latencies = []

    for i, q in enumerate(queries):
        pred, lat = call_ollama(q["text"], args.model)
        labels_true.append(q["ground_truth"])
        labels_pred.append(pred)
        latencies.append(lat)

        if (i + 1) % 50 == 0 or i + 1 == len(queries):
            acc_so_far = sum(t == p for t, p in zip(labels_true, labels_pred, strict=False)) / len(
                labels_true
            )
            print(
                f"  [{i+1}/{len(queries)}]  acc={acc_so_far:.3f}  lat_p50={np.percentile(latencies, 50):.0f}ms"
            )

    # Metrics
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    classes = ["LOW", "MEDIUM", "HIGH"]
    overall_acc = sum(t == p for t, p in zip(labels_true, labels_pred, strict=False)) / len(
        labels_true
    )
    macro_f1 = f1_score(labels_true, labels_pred, average="macro", labels=classes, zero_division=0)
    cm = confusion_matrix(labels_true, labels_pred, labels=classes).tolist()
    report = classification_report(
        labels_true, labels_pred, labels=classes, output_dict=True, zero_division=0
    )

    # Free-tier retention: fraction of non-HIGH queries NOT sent to cloud
    non_high_true = [t for t, p in zip(labels_true, labels_pred, strict=False) if t != "HIGH"]
    leaked = sum(
        1 for t, p in zip(labels_true, labels_pred, strict=False) if t != "HIGH" and p == "HIGH"
    )
    retention = (1 - leaked / len(non_high_true)) * 100 if non_high_true else 0.0

    # Wilson CIs on per-class recall
    wilson = {}
    for cls in classes:
        n = sum(1 for t in labels_true if t == cls)
        k = sum(1 for t, p in zip(labels_true, labels_pred, strict=False) if t == cls and p == cls)
        lo, hi = wilson_ci(k, n)
        wilson[cls] = {"k": k, "n": n, "recall_lo": round(lo, 4), "recall_hi": round(hi, 4)}

    lat_arr = np.array(latencies)

    print(f"\n{'='*50}")
    print(f"Model: {args.model}  |  N={len(queries)}")
    print(f"  Accuracy:  {overall_acc:.3f}")
    print(f"  Macro-F1:  {macro_f1:.3f}")
    print(f"  Free-tier retention: {retention:.1f}%")
    print(f"  Latency p50={np.percentile(lat_arr,50):.0f}ms  p95={np.percentile(lat_arr,95):.0f}ms")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(f"  {'':8s} {'LOW':>6s} {'MED':>6s} {'HIGH':>6s}")
    for i, cls in enumerate(classes):
        print(f"  {cls:6s}   {cm[i][0]:6d} {cm[i][1]:6d} {cm[i][2]:6d}")
    print("\nPer-class recall + Wilson 95% CI:")
    for cls in classes:
        pc = report[cls]
        w = wilson[cls]
        print(
            f"  {cls:6s}  recall={pc['recall']:.3f}  [{w['recall_lo']:.3f}, {w['recall_hi']:.3f}]  "
            f"f1={pc['f1-score']:.3f}"
        )

    result = {
        "model": args.model,
        "n_test": len(queries),
        "accuracy": round(overall_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "free_tier_retention_pct": round(retention, 2),
        "latency_ms": {
            "p50": round(float(np.percentile(lat_arr, 50)), 1),
            "p95": round(float(np.percentile(lat_arr, 95)), 1),
            "p99": round(float(np.percentile(lat_arr, 99)), 1),
            "mean": round(float(lat_arr.mean()), 1),
        },
        "per_class": {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall": round(report[cls]["recall"], 4),
                "f1": round(report[cls]["f1-score"], 4),
                "wilson_ci": wilson[cls],
            }
            for cls in classes
        },
        "confusion_matrix": cm,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    out = RESULTS_DIR / "llama_judge_report.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nReport saved: {out}")


if __name__ == "__main__":
    main()
