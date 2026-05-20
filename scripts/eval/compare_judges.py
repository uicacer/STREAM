#!/usr/bin/env python3
"""
Head-to-head comparison of ModernBERT vs LLM judges on the same test set.

Runs both judges on identical queries so the accuracy and latency numbers
are directly comparable. This is the correct way to report "ModernBERT
matches the LLM judge while being 26× faster."

Usage:
    # On the real-world Arena test set (recommended for paper):
    python scripts/eval/compare_judges.py \
        --testset scripts/eval/realworld_testset.json

    # On a held-out domain split (pass a JSON with a "queries" list):
    python scripts/eval/compare_judges.py \
        --testset scripts/eval/realworld_testset.json \
        --llm-judge haiku

    # Dry-run with just 20 queries (to check setup before spending API credits):
    python scripts/eval/compare_judges.py \
        --testset scripts/eval/realworld_testset.json \
        --n 20

Output:
    scripts/eval/results/judge_comparison_<testset_stem>.json

LLM judge options:
    haiku    (default) claude-haiku-4-5 via Anthropic API — fast, cheap, good
    sonnet   claude-sonnet-4-6 — same model that labeled the training data
    ollama   ollama/llama3.2:3b — free, but requires Ollama running locally

Notes:
    - Both judges use the same reasoning-depth rubric (not the production
      JUDGE_PROMPT which uses a format-proxy rubric). This ensures we measure
      *rubric agreement*, not rubric divergence.
    - The ModernBERT model must already be trained:
      run train_modernbert.py first.
    - For the Anthropic judges, ANTHROPIC_API_KEY must be set.
    - Estimated cost: 400 queries × haiku ≈ $0.02
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

MODEL_DIR = Path("scripts/eval/models/modernbert")
RESULTS_DIR = Path("scripts/eval/results")
MAX_LENGTH = 128
ID2LABEL = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

# The same reasoning-depth rubric used to label the training and test data.
# We deliberately do NOT use the production JUDGE_PROMPT (which is format-based)
# because we want to measure how well each judge learns the rubric, not whether
# they use the same shortcut heuristics.
REASONING_DEPTH_RUBRIC = """You are a query complexity classifier for an LLM routing system.

**Complexity rubric** (based on reasoning depth, NOT question format):

LOW: Single retrievable fact or trivial computation. The answer can be stated
     in one sentence with no reasoning chain required.
     Example: "What is the capital of France?" / "Is Python interpreted?"

MEDIUM: Apply a standard procedure or assemble 2-4 concepts. The reasoning
        path is textbook-level and well-established.
        Example: "Explain quicksort and its time complexity." / "Compare TCP and UDP."

HIGH: Construct a novel reasoning path, formal derivation, or expert judgment.
      No standard procedure exists — the solver must build the path.
      Example: "Is P=NP? Summarize the state of evidence."
      / "Design a fault-tolerant distributed key-value store."

**Key rule**: Format is NOT the complexity signal. "What is X?" can be
LOW, MEDIUM, or HIGH depending on what reasoning is required to answer.

Respond with ONLY ONE WORD: LOW, MEDIUM, or HIGH

User Query: {query}"""

LABEL2ID = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


# ---------------------------------------------------------------------------
# ModernBERT judge
# ---------------------------------------------------------------------------


def run_modernbert(queries: list[dict], model_dir: Path) -> list[dict]:
    """Run ModernBERT on all queries. Returns list of result dicts."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not model_dir.exists():
        raise FileNotFoundError(f"Model not found: {model_dir}\n" "Run train_modernbert.py first.")

    print(f"  Loading ModernBERT from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    model.to("cpu")

    texts = [q["text"] for q in queries]
    results = []

    for i in range(0, len(texts), 64):
        batch = texts[i : i + 64]
        enc = tokenizer(
            batch,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
            padding=True,
        )
        enc = {k: v.to("cpu") for k, v in enc.items()}
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(**enc).logits
        batch_ms = (time.perf_counter() - t0) * 1000

        probs = torch.softmax(logits, dim=-1).tolist()
        preds = torch.argmax(logits, dim=-1).tolist()
        per_query_ms = batch_ms / len(batch)

        for pred, prob in zip(preds, probs, strict=True):
            results.append(
                {
                    "pred": ID2LABEL[pred],
                    "scores": {
                        "LOW": round(prob[0], 4),
                        "MEDIUM": round(prob[1], 4),
                        "HIGH": round(prob[2], 4),
                    },
                    "latency_ms": round(per_query_ms, 2),
                }
            )

    return results


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

LLM_MODELS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "ollama": "ollama/llama3.2:3b",
}


def run_llm_judge(queries: list[dict], judge: str, dry_run: bool = False) -> list[dict]:
    """Run LLM judge on all queries. Returns list of result dicts."""
    model_id = LLM_MODELS.get(judge, judge)
    is_anthropic = judge in ("haiku", "sonnet")
    is_ollama = judge == "ollama"

    if dry_run:
        print("  [DRY RUN] Returning fake LLM labels (no API calls)")
        import random

        return [
            {"pred": random.choice(["LOW", "MEDIUM", "HIGH"]), "latency_ms": 0.0} for _ in queries
        ]

    if is_anthropic:
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            env_file = Path(__file__).parent.parent.parent / ".env"
            if env_file.exists():
                for line in env_file.read_text().splitlines():
                    if line.startswith("ANTHROPIC_API_KEY="):
                        api_key = line.split("=", 1)[1].strip()
                        break
        if not api_key:
            raise OSError("ANTHROPIC_API_KEY not set")
        client = anthropic.Anthropic(api_key=api_key)
    elif is_ollama:
        import httpx
    else:
        raise ValueError(f"Unknown judge: {judge}. Choose haiku, sonnet, or ollama.")

    results = []
    n = len(queries)

    for i, q in enumerate(queries):
        prompt = REASONING_DEPTH_RUBRIC.format(query=q["text"])
        t0 = time.perf_counter()

        try:
            if is_anthropic:
                response = client.messages.create(
                    model=model_id,
                    max_tokens=10,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = response.content[0].text.strip().upper()
            elif is_ollama:
                with httpx.Client(timeout=30) as http:
                    resp = http.post(
                        "http://localhost:11434/api/generate",
                        json={"model": "llama3.2:3b", "prompt": prompt, "stream": False},
                    )
                raw = resp.json().get("response", "").strip().upper()

            latency_ms = (time.perf_counter() - t0) * 1000

            # Parse: take first word, allow "LOW.", "MEDIUM\n" etc.
            for word in raw.split():
                word = word.strip(".,;:")
                if word in ("LOW", "MEDIUM", "HIGH"):
                    pred = word
                    break
            else:
                pred = "MEDIUM"  # fallback

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            pred = "MEDIUM"
            print(f"  [WARN] Query {i+1}: LLM judge failed ({e}), using MEDIUM")

        results.append({"pred": pred, "latency_ms": round(latency_ms, 2)})

        if (i + 1) % 50 == 0 or i + 1 == n:
            print(f"  LLM judge: {i+1}/{n} queries done")

        time.sleep(0.1)  # gentle rate limiting

    return results


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------


def compute_metrics(true_labels: list[str], preds: list[str]) -> dict:
    from sklearn.metrics import classification_report, confusion_matrix

    report = classification_report(
        true_labels, preds, labels=["LOW", "MEDIUM", "HIGH"], output_dict=True
    )
    cm = confusion_matrix(true_labels, preds, labels=["LOW", "MEDIUM", "HIGH"])

    n_free = sum(1 for lb in true_labels if lb in ("LOW", "MEDIUM"))
    n_leaked = sum(
        1 for t, p in zip(true_labels, preds, strict=True) if t in ("LOW", "MEDIUM") and p == "HIGH"
    )
    retention = (1 - n_leaked / n_free) * 100 if n_free else 0.0

    return {
        "accuracy": round(report["accuracy"], 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "per_class": {
            cls: {
                "f1": round(report[cls]["f1-score"], 4),
                "precision": round(report[cls]["precision"], 4),
                "recall": round(report[cls]["recall"], 4),
            }
            for cls in ["LOW", "MEDIUM", "HIGH"]
        },
        "confusion_matrix": cm.tolist(),
        "free_tier_retention_pct": round(retention, 2),
    }


def print_comparison(label: str, metrics: dict, lat: dict):
    print(f"\n  {label}:")
    print(f"    Accuracy: {metrics['accuracy']:.3f}   Macro-F1: {metrics['macro_f1']:.3f}")
    for cls in ["LOW", "MEDIUM", "HIGH"]:
        pc = metrics["per_class"][cls]
        print(f"    {cls:6s} F1={pc['f1']:.3f}  " f"p={pc['precision']:.3f}  r={pc['recall']:.3f}")
    print(f"    Latency — p50: {lat['p50_ms']} ms  p95: {lat['p95_ms']} ms")
    print(f"    Free-tier retention: {metrics['free_tier_retention_pct']:.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--testset",
        default="scripts/eval/realworld_testset.json",
        help="Test set JSON with a 'queries' list, each entry having 'text' and 'ground_truth'",
    )
    parser.add_argument(
        "--llm-judge",
        choices=["haiku", "sonnet", "ollama"],
        default="haiku",
        help="Which LLM to use as the judge (default: haiku)",
    )
    parser.add_argument("--model", default=str(MODEL_DIR), help="Path to trained ModernBERT")
    parser.add_argument("--n", type=int, default=None, help="Limit to first N queries")
    parser.add_argument("--dry-run", action="store_true", help="Fake LLM labels (no API calls)")
    args = parser.parse_args()

    testset_path = Path(args.testset)
    model_dir = Path(args.model)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not testset_path.exists():
        print(f"Test set not found: {testset_path}")
        if "realworld" in str(testset_path):
            print("Run build_realworld_testset.py first.")
        return

    print(f"Loading test set: {testset_path}")
    with open(testset_path) as f:
        data = json.load(f)
    queries = data["queries"]
    if args.n:
        queries = queries[: args.n]
    print(f"  {len(queries)} queries")

    true_labels = [q["ground_truth"] for q in queries]

    # ---- ModernBERT ----
    print("\nRunning ModernBERT...")
    t_mb_start = time.perf_counter()
    mb_results = run_modernbert(queries, model_dir)
    t_mb_total = time.perf_counter() - t_mb_start
    mb_preds = [r["pred"] for r in mb_results]
    mb_latencies = [r["latency_ms"] for r in mb_results]
    mb_latencies_sorted = sorted(mb_latencies)
    mb_lat = {
        "p50_ms": round(float(np.percentile(mb_latencies_sorted, 50)), 2),
        "p95_ms": round(float(np.percentile(mb_latencies_sorted, 95)), 2),
        "p99_ms": round(float(np.percentile(mb_latencies_sorted, 99)), 2),
        "mean_ms": round(float(np.mean(mb_latencies)), 2),
        "total_s": round(t_mb_total, 2),
    }
    mb_metrics = compute_metrics(true_labels, mb_preds)

    # ---- LLM Judge ----
    print(f"\nRunning LLM judge ({args.llm_judge})...")
    t_llm_start = time.perf_counter()
    llm_results = run_llm_judge(queries, args.llm_judge, dry_run=args.dry_run)
    t_llm_total = time.perf_counter() - t_llm_start
    llm_preds = [r["pred"] for r in llm_results]
    llm_latencies = [r["latency_ms"] for r in llm_results]
    llm_latencies_sorted = sorted(llm_latencies)
    llm_lat = {
        "p50_ms": round(float(np.percentile(llm_latencies_sorted, 50)), 2),
        "p95_ms": round(float(np.percentile(llm_latencies_sorted, 95)), 2),
        "p99_ms": round(float(np.percentile(llm_latencies_sorted, 99)), 2),
        "mean_ms": round(float(np.mean(llm_latencies)), 2),
        "total_s": round(t_llm_total, 2),
    }
    llm_metrics = compute_metrics(true_labels, llm_preds)

    # ---- Agreement between the two judges ----
    n_agree = sum(1 for mb, llm in zip(mb_preds, llm_preds, strict=True) if mb == llm)
    agreement_pct = n_agree / len(queries) * 100

    # ---- Print results ----
    print(f"\n{'='*60}")
    print("HEAD-TO-HEAD JUDGE COMPARISON")
    print(f"Test set: {testset_path.name}  ({len(queries)} queries)")
    print(f"{'='*60}")
    print_comparison("ModernBERT (~15ms)", mb_metrics, mb_lat)
    print_comparison(f"LLM judge ({args.llm_judge})", llm_metrics, llm_lat)

    speedup = llm_lat["p50_ms"] / mb_lat["p50_ms"] if mb_lat["p50_ms"] > 0 else float("inf")
    f1_delta = mb_metrics["macro_f1"] - llm_metrics["macro_f1"]

    print(f"\n  Latency speedup (ModernBERT vs LLM): {speedup:.1f}×")
    print(f"  Macro-F1 delta (ModernBERT − LLM):  {f1_delta:+.3f}")
    print(f"  Judge agreement (same label):        {agreement_pct:.1f}%")

    # ---- Save report ----
    result = {
        "testset": str(testset_path),
        "n_queries": len(queries),
        "llm_judge": args.llm_judge,
        "llm_model": LLM_MODELS.get(args.llm_judge, args.llm_judge),
        "modernbert": {**mb_metrics, "latency_ms": mb_lat},
        "llm": {**llm_metrics, "latency_ms": llm_lat},
        "comparison": {
            "latency_speedup_x": round(speedup, 1),
            "macro_f1_delta": round(f1_delta, 4),
            "judge_agreement_pct": round(agreement_pct, 2),
        },
    }

    out_path = RESULTS_DIR / f"judge_comparison_{testset_path.stem}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nReport saved to {out_path}")

    print(f"\n{'='*60}")
    print("PAPER NUMBERS (paste into paper):")
    print(f"{'='*60}")
    print(f"  ModernBERT accuracy:  {mb_metrics['accuracy']:.1%}")
    print(f"  ModernBERT macro-F1:  {mb_metrics['macro_f1']:.3f}")
    print(f"  ModernBERT latency:   {mb_lat['p50_ms']} ms (p50)")
    print(f"  LLM judge accuracy:   {llm_metrics['accuracy']:.1%}")
    print(f"  LLM judge macro-F1:   {llm_metrics['macro_f1']:.3f}")
    print(f"  LLM judge latency:    {llm_lat['p50_ms']} ms (p50)")
    print(f"  Speedup:              {speedup:.1f}×")
    print(f"  Judge agreement:      {agreement_pct:.1f}%")


if __name__ == "__main__":
    main()
