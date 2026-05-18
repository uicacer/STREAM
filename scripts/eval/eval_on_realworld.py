#!/usr/bin/env python3
"""
Evaluate the trained ModernBERT classifier on the real-world test set.

The real-world test set is sourced from LMSYS Chatbot Arena — genuine user
queries that were never part of the training data. This is the most rigorous
evaluation because it tests generalization to a completely different distribution.

Usage:
    python scripts/eval/eval_on_realworld.py
    python scripts/eval/eval_on_realworld.py --model scripts/eval/models/modernbert
    python scripts/eval/eval_on_realworld.py --testset scripts/eval/realworld_testset.json

Output:
    scripts/eval/results/modernbert_realworld_report.json

Requirements:
    Trained model: scripts/eval/models/modernbert/
    Test set:      scripts/eval/realworld_testset.json
    Run build_realworld_testset.py first if the test set doesn't exist.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

MODEL_DIR = Path("scripts/eval/models/modernbert")
TESTSET_PATH = Path("scripts/eval/realworld_testset.json")
REPORT_PATH = Path("scripts/eval/results/modernbert_realworld_report.json")

MAX_LENGTH = 128
ID2LABEL = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
LABEL2ID = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def measure_latency(model, tokenizer, texts, n=200, device="cpu"):
    import torch

    model.eval()
    model.to(device)
    latencies = []
    for text in texts[:n]:
        enc = tokenizer(
            text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt", padding=True
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(**enc)
        latencies.append((time.perf_counter() - t0) * 1000)
    latencies.sort()
    return {
        "p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "p99_ms": round(float(np.percentile(latencies, 99)), 2),
        "n": len(latencies),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(MODEL_DIR))
    parser.add_argument("--testset", default=str(TESTSET_PATH))
    parser.add_argument("--output", default=str(REPORT_PATH))
    args = parser.parse_args()

    import torch
    from sklearn.metrics import classification_report, confusion_matrix
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = Path(args.model)
    testset_path = Path(args.testset)
    report_path = Path(args.output)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_dir.exists():
        print(f"Model not found: {model_dir}")
        print("Run train_modernbert.py first.")
        return

    if not testset_path.exists():
        print(f"Test set not found: {testset_path}")
        print("Run build_realworld_testset.py first.")
        return

    print(f"Loading model from {model_dir}...")
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    print(f"Loading real-world test set from {testset_path}...")
    with open(testset_path) as f:
        data = json.load(f)
    queries = data["queries"]
    print(f"  {len(queries)} queries")
    print(f"  Source: {data['metadata'].get('source', 'unknown')}")

    texts = [q["text"] for q in queries]
    true_labels = [q["ground_truth"] for q in queries]

    # Inference
    print("\nRunning inference on CPU...")
    model.eval()
    model.to("cpu")
    all_preds = []

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
        with torch.no_grad():
            logits = model(**enc).logits
        all_preds.extend(torch.argmax(logits, dim=-1).tolist())

    pred_names = [ID2LABEL[p] for p in all_preds]

    # Metrics
    report = classification_report(
        true_labels, pred_names, labels=["LOW", "MEDIUM", "HIGH"], output_dict=True
    )
    cm = confusion_matrix(true_labels, pred_names, labels=["LOW", "MEDIUM", "HIGH"])

    n_free = sum(1 for lb in true_labels if lb in ("LOW", "MEDIUM"))
    n_leaked = sum(
        1
        for t, p in zip(true_labels, pred_names, strict=True)
        if t in ("LOW", "MEDIUM") and p == "HIGH"
    )
    retention = (1 - n_leaked / n_free) * 100 if n_free else 0.0

    print(f"\nReal-world test set results ({len(queries)} queries):")
    print(f"  Accuracy:  {report['accuracy']:.3f}")
    print(f"  Macro-F1:  {report['macro avg']['f1-score']:.3f}")
    for cls in ["LOW", "MEDIUM", "HIGH"]:
        pc = report[cls]
        print(
            f"  {cls:6s} F1: {pc['f1-score']:.3f}  "
            f"(precision={pc['precision']:.3f}, recall={pc['recall']:.3f})"
        )

    print("\nConfusion matrix (rows=true, cols=pred):")
    print(f"{'':8s} {'LOW':>6s} {'MED':>6s} {'HIGH':>6s}")
    for i, cls in enumerate(["LOW", "MEDIUM", "HIGH"]):
        print(f"  {cls:6s} {cm[i][0]:6d} {cm[i][1]:6d} {cm[i][2]:6d}")

    print(f"\n  Free-tier retention: {retention:.1f}%")

    lat = measure_latency(model, tokenizer, texts[:200])
    print(
        f"  CPU latency — p50: {lat['p50_ms']} ms  p95: {lat['p95_ms']} ms  p99: {lat['p99_ms']} ms"
    )

    result = {
        "eval_mode": "realworld",
        "model_dir": str(model_dir),
        "testset": str(testset_path),
        "testset_source": data["metadata"].get("source"),
        "n_queries": len(queries),
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
        "latency_ms": lat,
    }

    with open(report_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nReport saved to {report_path}")

    print("\n" + "=" * 60)
    print("PAPER NUMBERS (real-world test set — LMSYS Chatbot Arena):")
    print("=" * 60)
    print(f"  Accuracy:  {report['accuracy']:.1%}")
    print(f"  Macro-F1:  {report['macro avg']['f1-score']:.3f}")
    for cls in ["LOW", "MEDIUM", "HIGH"]:
        print(f"  {cls:6s} F1: {report[cls]['f1-score']:.3f}")
    print(f"  Latency p50: {lat['p50_ms']} ms")
    print(f"  Free-tier retention: {retention:.1f}%")
    print(
        "\nNext: python scripts/eval/generate_tables.py\n"
        "      (combines all three eval reports into paper tables)"
    )


if __name__ == "__main__":
    main()
