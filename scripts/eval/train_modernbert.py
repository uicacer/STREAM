#!/usr/bin/env python3
"""
Train ModernBERT-base on the v2 complexity dataset.

Distills Claude Sonnet 4.6's routing judgment into a local 149M-parameter
classifier. After training, the model runs at ~15ms per query with no API
dependency, replacing the 390ms Llama 3.2 3B LLM judge.

Usage:
    python scripts/eval/train_modernbert.py
    python scripts/eval/train_modernbert.py --dataset scripts/eval/benchmark_dataset_v2.json
    python scripts/eval/train_modernbert.py --dry-run   # sanity check, 10 examples

Output:
    scripts/eval/models/modernbert/   — fine-tuned model + tokenizer
    scripts/eval/results/modernbert_training_report.json

Requirements:
    pip install transformers torch datasets scikit-learn accelerate
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

DATASET_PATH = Path("scripts/eval/benchmark_dataset_v2.json")
OUTPUT_DIR = Path("scripts/eval/models/modernbert")
REPORT_PATH = Path("scripts/eval/results/modernbert_training_report.json")

MODEL_NAME = "answerdotai/ModernBERT-base"
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 5
LR = 2e-5
SEED = 42

LABEL2ID = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
ID2LABEL = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}


def load_dataset(path: Path, dry_run: bool = False):
    with open(path) as f:
        data = json.load(f)
    queries = data["queries"]
    if dry_run:
        queries = queries[:30]
    return queries


def train_val_test_split(queries, seed=SEED):
    import random

    rng = random.Random(seed)
    rng.shuffle(queries)
    n = len(queries)
    train = queries[: int(0.70 * n)]
    val = queries[int(0.70 * n) : int(0.85 * n)]
    test = queries[int(0.85 * n) :]
    return train, val, test


def to_hf_dataset(queries):
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "text": [q["text"] for q in queries],
            "label": [LABEL2ID[q["ground_truth"]] for q in queries],
        }
    )


def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def measure_latency(model, tokenizer, sample_texts, n=200, device="cpu"):
    """Measure single-query CPU inference latency (p50/p95/p99)."""
    import torch

    model.eval()
    latencies = []
    for text in sample_texts[:n]:
        enc = tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
            padding=True,
        )
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
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument(
        "--dry-run", action="store_true", help="Train on 30 examples for a quick sanity check"
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        print("Run generate_benchmark_dataset_v2.py first.")
        return

    print(f"Loading dataset from {dataset_path}")
    queries = load_dataset(dataset_path, dry_run=args.dry_run)
    print(f"  Total queries: {len(queries)}")
    print(f"  Distribution: {Counter(q['ground_truth'] for q in queries)}")

    train_q, val_q, test_q = train_val_test_split(queries)
    print(f"  Split: train={len(train_q)}, val={len(val_q)}, test={len(test_q)}")

    train_ds = to_hf_dataset(train_q)
    val_ds = to_hf_dataset(val_q)
    test_ds = to_hf_dataset(test_q)

    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    for ds in [train_ds, val_ds, test_ds]:
        ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=args.epochs if not args.dry_run else 1,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=64,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="best",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        seed=SEED,
        # CPU-friendly settings (no GPU required)
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print(f"\nTraining ModernBERT-base for {training_args.num_train_epochs} epochs...")
    t_train_start = time.perf_counter()
    trainer.train()
    train_time = time.perf_counter() - t_train_start
    print(f"Training complete in {train_time/60:.1f} minutes")

    # Save model and tokenizer
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"Model saved to {OUTPUT_DIR}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    import torch
    from sklearn.metrics import classification_report, confusion_matrix

    model.eval()
    all_preds, all_labels = [], []

    test_texts = [q["text"] for q in test_q]
    test_labels = [LABEL2ID[q["ground_truth"]] for q in test_q]

    for i in range(0, len(test_texts), 64):
        batch_texts = test_texts[i : i + 64]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            logits = model(**enc).logits
        preds = torch.argmax(logits, dim=-1).tolist()
        all_preds.extend(preds)

    all_labels = test_labels
    pred_names = [ID2LABEL[p] for p in all_preds]
    label_names = [ID2LABEL[lbl] for lbl in all_labels]

    report = classification_report(
        label_names,
        pred_names,
        labels=["LOW", "MEDIUM", "HIGH"],
        output_dict=True,
    )
    cm = confusion_matrix(label_names, pred_names, labels=["LOW", "MEDIUM", "HIGH"])

    print("\nTest set results:")
    print(f"  Accuracy:  {report['accuracy']:.3f}")
    print(f"  Macro-F1:  {report['macro avg']['f1-score']:.3f}")
    for cls in ["LOW", "MEDIUM", "HIGH"]:
        print(
            f"  {cls:6s} F1: {report[cls]['f1-score']:.3f}  "
            f"(precision={report[cls]['precision']:.3f}, recall={report[cls]['recall']:.3f})"
        )

    print("\nConfusion matrix (rows=true, cols=pred):")
    print(f"{'':8s} {'LOW':>6s} {'MED':>6s} {'HIGH':>6s}")
    for i, cls in enumerate(["LOW", "MEDIUM", "HIGH"]):
        print(f"  {cls:6s} {cm[i][0]:6d} {cm[i][1]:6d} {cm[i][2]:6d}")

    # CPU latency
    print("\nMeasuring CPU inference latency (single query)...")
    lat = measure_latency(model, tokenizer, test_texts[:200])
    print(f"  p50: {lat['p50_ms']} ms   p95: {lat['p95_ms']} ms   p99: {lat['p99_ms']} ms")

    # Free-tier retention
    n_free_worthy = sum(1 for lbl in label_names if lbl in ("LOW", "MEDIUM"))
    n_leaked = sum(
        1
        for t, p in zip(label_names, pred_names, strict=False)
        if t in ("LOW", "MEDIUM") and p == "HIGH"
    )
    free_tier_retention = (1 - n_leaked / n_free_worthy) * 100 if n_free_worthy else 0
    print(
        f"\n  Free-tier retention: {free_tier_retention:.1f}%  "
        f"({n_free_worthy - n_leaked}/{n_free_worthy} free-worthy queries stay on free tiers)"
    )

    # Save report
    result = {
        "model": MODEL_NAME,
        "dataset": str(dataset_path),
        "train_size": len(train_q),
        "val_size": len(val_q),
        "test_size": len(test_q),
        "epochs": training_args.num_train_epochs,
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
        "latency_ms": lat,
        "free_tier_retention_pct": round(free_tier_retention, 2),
        "train_time_minutes": round(train_time / 60, 1),
        "output_dir": str(OUTPUT_DIR),
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nReport saved to {REPORT_PATH}")

    print("\n" + "=" * 60)
    print("PAPER NUMBERS (paste into pearc26-stream-paper.tex):")
    print("=" * 60)
    print(f"  ModernBERT accuracy:          {report['accuracy']:.1%}")
    print(f"  ModernBERT macro-F1:          {report['macro avg']['f1-score']:.3f}")
    print(f"  ModernBERT LOW F1:            {report['LOW']['f1-score']:.3f}")
    print(f"  ModernBERT MEDIUM F1:         {report['MEDIUM']['f1-score']:.3f}")
    print(f"  ModernBERT HIGH F1:           {report['HIGH']['f1-score']:.3f}")
    print(f"  ModernBERT latency p50:       {lat['p50_ms']} ms")
    print(f"  Free-tier retention:          {free_tier_retention:.1f}%")
    print(
        f"\nNext: python scripts/eval/benchmark_routing.py "
        f"--queries {dataset_path} --judge modernbert"
    )


if __name__ == "__main__":
    main()
