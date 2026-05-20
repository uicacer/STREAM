#!/usr/bin/env python3
"""
Evaluate all three classifiers against GPT-4o as an independent reference annotator.

GPT-4o labels the 400 Arena OOD queries independently of Claude — no circularity.
All three classifiers (Claude Sonnet 4.6, ModernBERT, Llama 3.2 3B) are then
evaluated against those GPT-4o labels on the same 400 queries.

This gives a fully rigorous classifier comparison:
  - GPT-4o as reference: independent of all training data and labels
  - Arena queries: real human-written text, never seen during training
  - Result: F1 of each classifier vs. independent ground truth

Usage:
    uv run python3 scripts/eval/evaluate_classifiers_gpt4o.py
    uv run python3 scripts/eval/evaluate_classifiers_gpt4o.py --dry-run  # 20 queries only

Requires:
    - OPENAI_API_KEY in env or .env file
    - ANTHROPIC_API_KEY in env or .env file (for Claude baseline)
    - Ollama running with llama3.2:3b (for Llama baseline)
    - scripts/eval/models/modernbert/ (trained ModernBERT weights)
    - scripts/eval/realworld_testset.json (Arena OOD queries with Claude labels)

Output:
    scripts/eval/results/classifier_comparison_gpt4o.json
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

ARENA_PATH = Path("scripts/eval/realworld_testset.json")
RESULTS_DIR = Path("scripts/eval/results")
OUTPUT_PATH = RESULTS_DIR / "classifier_comparison_gpt4o.json"
MODERNBERT_DIR = Path("scripts/eval/models/modernbert")

CLASSES = ["LOW", "MEDIUM", "HIGH"]

RUBRIC = (
    "You are a query complexity classifier for an LLM routing system.\n\n"
    "LOW: Single retrievable fact or trivial computation. Answer statable in one "
    "sentence with no reasoning chain required.\n"
    "MEDIUM: Apply a standard procedure or assemble 2-4 concepts. Textbook-level "
    "reasoning path, well-established.\n"
    "HIGH: Construct a novel reasoning path, formal derivation, or expert judgment. "
    "No standard procedure exists — the solver must build the path.\n\n"
    "Key rule: question format is NOT the complexity signal. "
    "'What is X?' can be LOW, MEDIUM, or HIGH depending on the reasoning required.\n\n"
    "Respond with ONLY ONE WORD: LOW, MEDIUM, or HIGH\n\n"
    "Query: {query}"
)


# ---------------------------------------------------------------------------
# API key helpers
# ---------------------------------------------------------------------------


def get_api_key(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        env_file = Path(__file__).parent.parent.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith(f"{var}="):
                    val = line.split("=", 1)[1].strip()
                    break
    if not val:
        raise OSError(f"{var} not set in environment or .env file")
    return val


# ---------------------------------------------------------------------------
# GPT-4o labeling (reference annotator)
# ---------------------------------------------------------------------------


def label_with_gpt4o(queries: list[dict], dry_run: bool) -> list[str]:
    from openai import OpenAI

    api_key = get_api_key("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    preds = []
    n = len(queries)
    print(f"  Labeling {n} queries with GPT-4o (reference annotator)...")

    for i, q in enumerate(queries):
        prompt = RUBRIC.format(query=q["text"])
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=5,
                    temperature=0,
                )
                raw = resp.choices[0].message.content.strip().upper()
                label = raw if raw in CLASSES else "MEDIUM"
                preds.append(label)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  [WARN] query {i}: failed: {e} — defaulting MEDIUM")
                    preds.append("MEDIUM")
                else:
                    time.sleep(2)

        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  GPT-4o: {i+1}/{n} done")
        time.sleep(0.05)

    return preds


# ---------------------------------------------------------------------------
# Claude Sonnet 4.6 classifier (API baseline)
# ---------------------------------------------------------------------------


def classify_with_claude(queries: list[dict]) -> list[str]:
    import anthropic

    api_key = get_api_key("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)

    preds = []
    n = len(queries)
    print(f"  Classifying {n} queries with Claude Sonnet 4.6...")

    for i, q in enumerate(queries):
        prompt = RUBRIC.format(query=q["text"])
        for attempt in range(3):
            try:
                resp = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=5,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.content[0].text.strip().upper()
                label = raw if raw in CLASSES else "MEDIUM"
                preds.append(label)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  [WARN] query {i}: failed: {e} — defaulting MEDIUM")
                    preds.append("MEDIUM")
                else:
                    time.sleep(2)

        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  Claude: {i+1}/{n} done")
        time.sleep(0.05)

    return preds


# ---------------------------------------------------------------------------
# Llama 3.2 3B via Ollama
# ---------------------------------------------------------------------------


def classify_with_llama(queries: list[dict]) -> list[str]:
    import litellm

    preds = []
    n = len(queries)
    print(f"  Classifying {n} queries with Llama 3.2 3B (Ollama)...")

    for i, q in enumerate(queries):
        prompt = RUBRIC.format(query=q["text"])
        for attempt in range(3):
            try:
                resp = litellm.completion(
                    model="ollama/llama3.2:3b",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0,
                )
                raw = resp.choices[0].message.content.strip().upper()
                label = raw if raw in CLASSES else "MEDIUM"
                preds.append(label)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  [WARN] query {i}: failed: {e} — defaulting MEDIUM")
                    preds.append("MEDIUM")
                else:
                    time.sleep(1)

        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  Llama: {i+1}/{n} done")
        time.sleep(0.05)

    return preds


# ---------------------------------------------------------------------------
# ModernBERT classifier
# ---------------------------------------------------------------------------


def classify_with_modernbert(queries: list[dict]) -> list[str]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not MODERNBERT_DIR.exists():
        print(f"  [WARN] ModernBERT model not found at {MODERNBERT_DIR} — skipping")
        return None

    print(f"  Classifying {len(queries)} queries with ModernBERT ({MODERNBERT_DIR})...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODERNBERT_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODERNBERT_DIR))
    model.eval()

    id2label = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

    preds = []
    batch_size = 32
    texts = [q["text"] for q in queries]

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
        pred_ids = logits.argmax(dim=-1).tolist()
        preds.extend(id2label[p] for p in pred_ids)

    print(f"  ModernBERT: {len(preds)} done")
    return preds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics_vs_reference(preds: list[str], reference: list[str], name: str) -> dict:
    acc = accuracy_score(reference, preds)
    macro_f1 = f1_score(reference, preds, labels=CLASSES, average="macro", zero_division=0)
    per_class_f1 = f1_score(reference, preds, labels=CLASSES, average=None, zero_division=0)
    kappa = cohen_kappa_score(reference, preds)
    cm = confusion_matrix(reference, preds, labels=CLASSES).tolist()

    # Leakage rates (row-normalized)
    cm_arr = np.array(cm, dtype=float)
    row_totals = cm_arr.sum(axis=1, keepdims=True)
    row_totals[row_totals == 0] = 1
    cm_rates = (cm_arr / row_totals * 100).tolist()

    print(f"\n  [{name}] vs GPT-4o reference:")
    print(f"    Accuracy:  {acc:.3f}")
    print(f"    Macro-F1:  {macro_f1:.3f}")
    print(f"    Kappa:     {kappa:.3f}")
    print(
        f"    Per-class F1: LOW={per_class_f1[0]:.3f}  MEDIUM={per_class_f1[1]:.3f}  HIGH={per_class_f1[2]:.3f}"
    )
    print(f"    Confusion matrix (rows=true GPT-4o, cols=pred {name}):")
    print(f"      {'':8s} {'LOW':>6s} {'MED':>6s} {'HIGH':>6s}")
    for i, cls in enumerate(CLASSES):
        print(
            f"      {cls:8s} {cm[i][0]:6d} {cm[i][1]:6d} {cm[i][2]:6d}  "
            f"({cls}→wrong: {100-cm_rates[i][i]:.1f}%)"
        )

    return {
        "classifier": name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "kappa_vs_gpt4o": kappa,
        "per_class_f1": {
            "LOW": per_class_f1[0],
            "MEDIUM": per_class_f1[1],
            "HIGH": per_class_f1[2],
        },
        "confusion_matrix": cm,
        "confusion_matrix_rates": cm_rates,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Use only 20 queries")
    parser.add_argument("--skip-claude", action="store_true", help="Skip Claude API classifier")
    parser.add_argument("--skip-llama", action="store_true", help="Skip Llama 3.2 3B classifier")
    parser.add_argument("--skip-modernbert", action="store_true", help="Skip ModernBERT classifier")
    parser.add_argument(
        "--reuse-gpt4o-labels",
        default=None,
        help="Path to cached GPT-4o labels JSON to skip re-labeling",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load Arena queries
    with open(ARENA_PATH) as f:
        arena_data = json.load(f)
    queries = arena_data["queries"]

    if args.dry_run:
        queries = queries[:20]
        print(f"[DRY RUN] Using {len(queries)} queries")
    else:
        print(f"Loaded {len(queries)} Arena OOD queries")

    from collections import Counter

    claude_labels = [q["ground_truth"] for q in queries]
    print(f"Claude label distribution: {Counter(claude_labels)}")

    # Step 1: GPT-4o reference labels
    print("\n" + "=" * 60)
    print("Step 1: GPT-4o reference labeling")
    print("=" * 60)

    gpt4o_cache = Path("scripts/eval/results/gpt4o_arena_labels.json")

    if args.reuse_gpt4o_labels and Path(args.reuse_gpt4o_labels).exists():
        with open(args.reuse_gpt4o_labels) as f:
            gpt4o_labels = json.load(f)
        print(f"  Reusing cached GPT-4o labels from {args.reuse_gpt4o_labels}")
    elif gpt4o_cache.exists() and not args.dry_run:
        with open(gpt4o_cache) as f:
            gpt4o_labels = json.load(f)
        print(f"  Reusing cached GPT-4o labels from {gpt4o_cache}")
    else:
        gpt4o_labels = label_with_gpt4o(queries, args.dry_run)
        if not args.dry_run:
            with open(gpt4o_cache, "w") as f:
                json.dump(gpt4o_labels, f)
            print(f"  GPT-4o labels cached to {gpt4o_cache}")

    print(f"  GPT-4o label distribution: {Counter(gpt4o_labels)}")

    # Inter-model agreement: Claude vs GPT-4o (validity check)
    kappa_claude_gpt4o = cohen_kappa_score(claude_labels, gpt4o_labels)
    agree = sum(c == g for c, g in zip(claude_labels, gpt4o_labels, strict=False)) / len(
        claude_labels
    )
    print(f"\n  Claude vs GPT-4o agreement: {agree:.1%}  κ={kappa_claude_gpt4o:.3f}")
    print("  (κ≥0.61 = substantial, κ≥0.41 = moderate — Landis & Koch 1977)")

    # Step 2: Classify with all models
    print("\n" + "=" * 60)
    print("Step 2: Running all classifiers")
    print("=" * 60)

    results = []

    # Claude Sonnet 4.6
    if not args.skip_claude:
        claude_preds = classify_with_claude(queries)
        results.append(
            compute_metrics_vs_reference(claude_preds, gpt4o_labels, "Claude Sonnet 4.6")
        )
        results[-1]["latency_note"] = "~1s per query, $0.003/query, sends data to API"
    else:
        print("  [skipped] Claude Sonnet 4.6")

    # ModernBERT
    if not args.skip_modernbert:
        mb_preds = classify_with_modernbert(queries)
        if mb_preds is not None:
            results.append(
                compute_metrics_vs_reference(mb_preds, gpt4o_labels, "ModernBERT (distilled)")
            )
            results[-1]["latency_note"] = "~15ms per query, $0, fully local"
    else:
        print("  [skipped] ModernBERT")

    # Llama 3.2 3B
    if not args.skip_llama:
        llama_preds = classify_with_llama(queries)
        results.append(compute_metrics_vs_reference(llama_preds, gpt4o_labels, "Llama 3.2 3B"))
        results[-1]["latency_note"] = "~390ms per query, $0, fully local"
    else:
        print("  [skipped] Llama 3.2 3B")

    # Step 3: Summary table
    print("\n" + "=" * 70)
    print("CLASSIFIER COMPARISON — all evaluated against GPT-4o reference labels")
    print(f"Arena OOD queries (n={len(queries)}) — real human-written, never in training")
    print(f"Inter-annotator reliability: Claude vs GPT-4o κ={kappa_claude_gpt4o:.3f}")
    print("=" * 70)
    print(f"  {'Classifier':<28} {'Macro-F1':>10} {'Acc':>8} {'κ vs GPT-4o':>12} {'HIGH F1':>8}")
    print(f"  {'─'*28} {'─'*10} {'─'*8} {'─'*12} {'─'*8}")
    for r in results:
        print(
            f"  {r['classifier']:<28}"
            f"  {r['macro_f1']:>8.3f}"
            f"  {r['accuracy']:>6.1%}"
            f"  {r['kappa_vs_gpt4o']:>10.3f}"
            f"  {r['per_class_f1']['HIGH']:>6.3f}"
        )
    print("=" * 70)

    # Save
    output = {
        "metadata": {
            "n_queries": len(queries),
            "reference_annotator": "gpt-4o",
            "inter_annotator_kappa_claude_vs_gpt4o": kappa_claude_gpt4o,
            "inter_annotator_agreement_claude_vs_gpt4o": agree,
            "arena_label_distribution_claude": dict(Counter(claude_labels)),
            "arena_label_distribution_gpt4o": dict(Counter(gpt4o_labels)),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "classifiers": results,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
