#!/usr/bin/env python3
"""
Train ModernBERT on the balanced HPC/research dataset and sweep classification
threshold θ to produce a precision-recall-cost curve.

Design
------
Training:
  - Single run on 3,000 balanced queries (1,000/class, seed=42 fixed split)
  - ModernBERT-base, 5 epochs, weighted-cross-entropy loss (uniform weights
    because the dataset is already balanced — no re-weighting needed)
  - Model saved to models/modernbert_balanced/

Threshold sweep:
  - After training, extract P(HIGH) soft scores on the fixed 750-query test set
  - Sweep θ ∈ [0.0, 1.0] in steps of 0.01 (101 points)
  - At each θ: query is routed to CLOUD if P(HIGH) ≥ θ, else to HPC/LOCAL
  - Compute: HIGH recall, cloud routing rate, macro-F1, cost-leak, qual-leak
  - Output: results/theta_sweep.json   ← used by plot_figures.py

Budget-aware adaptive routing (for Figure 2 simulation):
  - θ_effective(t) = max(θ_base, spend(t) / budget)
  - Simulated on the test set with Pareto-sampled daily query arrivals
  - Output: results/budget_simulation.json

Usage
-----
  python train_balanced_classifier.py
  python train_balanced_classifier.py --dry-run
  python train_balanced_classifier.py --skip-train  # reuse saved model
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

TRAIN_DATASET = Path("balanced_training_dataset.json")
TEST_DATASET = Path("balanced_test_dataset.json")
MODEL_DIR = Path("models/modernbert_balanced")
RESULTS_DIR = Path("results")

MODEL_NAME = "answerdotai/ModernBERT-base"
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 5
LR = 2e-5
SEED = 42

LABEL2ID = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
ID2LABEL = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

THETA_MIN = 0.0
THETA_MAX = 1.0
THETA_STEP = 0.01

# Budget simulation parameters
SIM_DAYS = 30
CLOUD_COST_PER_1K = 0.003  # USD per 1,000 tokens (rough Claude Haiku rate)
AVG_TOKENS = 800  # avg tokens per query


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_split(path: Path, dry_run: bool = False) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    queries = data["queries"]
    if dry_run:
        queries = queries[:60]
    return queries


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(train_q: list[dict], val_q: list[dict], args) -> object:
    import torch
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length"
        )

    def to_ds(qs):
        ds = Dataset.from_dict(
            {"text": [q["text"] for q in qs], "label": [LABEL2ID[q["ground_truth"]] for q in qs]}
        )
        return ds.map(tokenize, batched=True)

    train_ds = to_ds(train_q)
    val_ds = to_ds(val_q)
    for ds in (train_ds, val_ds):
        ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "macro_f1": float(
                f1_score(labels, preds, average="macro", labels=[0, 1, 2], zero_division=0)
            ),
        }

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        num_train_epochs=1 if args.dry_run else EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=64,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        seed=SEED,
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

    t0 = time.perf_counter()
    trainer.train()
    print(f"  Training complete ({(time.perf_counter()-t0)/60:.1f} min)")

    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print(f"  Model saved to {MODEL_DIR}")

    return trainer.model, tokenizer


# ---------------------------------------------------------------------------
# Soft-score extraction
# ---------------------------------------------------------------------------


def get_soft_scores(model, tokenizer, queries: list[dict]) -> np.ndarray:
    """Return (N, 3) softmax probabilities for [LOW, MEDIUM, HIGH]."""
    import torch

    model.eval()
    model.to("cpu")

    all_probs = []
    for i in range(0, len(queries), 64):
        batch = [q["text"] for q in queries[i : i + 64]]
        enc = tokenizer(
            batch,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------


def sweep_theta(probs: np.ndarray, labels: list[str]) -> list[dict]:
    """
    For each θ, route query to CLOUD if P(HIGH) >= θ, else cheapest tier.
    Returns list of dicts with per-θ metrics.
    """
    from sklearn.metrics import f1_score

    label_ids = np.array([LABEL2ID[lb] for lb in labels])
    p_high = probs[:, 2]  # P(HIGH) for each query

    results = []
    for theta in np.arange(THETA_MIN, THETA_MAX + THETA_STEP / 2, THETA_STEP):
        theta = round(float(theta), 3)
        preds = np.where(p_high >= theta, LABEL2ID["HIGH"], np.argmax(probs[:, :2], axis=1))

        # Per-class metrics (labels=[0,1,2] forces 3-class scoring even at extremes)
        macro_f1 = float(
            f1_score(label_ids, preds, average="macro", labels=[0, 1, 2], zero_division=0)
        )
        high_mask = label_ids == LABEL2ID["HIGH"]
        high_recall = (
            float((preds[high_mask] == LABEL2ID["HIGH"]).mean()) if high_mask.any() else 0.0
        )
        cloud_rate = float((preds == LABEL2ID["HIGH"]).mean())

        # Operational metrics
        non_high = label_ids != LABEL2ID["HIGH"]
        cost_leak = float((preds[non_high] == LABEL2ID["HIGH"]).mean()) if non_high.any() else 0.0

        med_high = label_ids >= LABEL2ID["MEDIUM"]
        under_route = ((label_ids == LABEL2ID["MEDIUM"]) & (preds == LABEL2ID["LOW"])) | (
            (label_ids == LABEL2ID["HIGH"]) & (preds < LABEL2ID["HIGH"])
        )
        qual_leak = float(under_route.sum() / max(med_high.sum(), 1))

        results.append(
            {
                "theta": theta,
                "macro_f1": round(macro_f1, 4),
                "high_recall": round(high_recall, 4),
                "cloud_routing_rate": round(cloud_rate, 4),
                "cost_leak": round(cost_leak, 4),
                "qual_leak": round(qual_leak, 4),
            }
        )

    return results


# ---------------------------------------------------------------------------
# Budget-aware adaptive routing simulation
# ---------------------------------------------------------------------------


def simulate_budget_routing(
    probs: np.ndarray,
    labels: list[str],
    theta_base: float = 0.5,
    monthly_budget: float = 10.0,  # USD
    seed: int = SEED,
) -> dict:
    """
    Simulate 30 days of queries with adaptive θ.

    Two strategies compared:
      fixed:    always use theta_base (ignores budget)
      adaptive: θ_eff = max(θ_base, spend/budget) — raises θ as budget depletes

    Queries arrive daily from the test pool (sampled with replacement).
    Cost: CLOUD_COST_PER_1K * AVG_TOKENS / 1000 per cloud-routed query.
    """
    rng = np.random.default_rng(seed)
    n_test = len(labels)

    # Daily query volume: Pareto-like (bursty)
    daily_counts = rng.integers(10, 60, size=SIM_DAYS)

    label_ids = np.array([LABEL2ID[lb] for lb in labels])
    p_high = probs[:, 2]

    cost_per_query = CLOUD_COST_PER_1K * AVG_TOKENS / 1000

    def run_strategy(adaptive: bool):
        spend = 0.0
        days = []
        for day_count in daily_counts:
            idx = rng.integers(0, n_test, size=day_count)
            day_p_high = p_high[idx]
            day_labels = label_ids[idx]

            theta_eff = max(theta_base, spend / monthly_budget) if adaptive else theta_base
            theta_eff = min(theta_eff, 0.999)  # never block all HIGH

            day_preds = np.where(
                day_p_high >= theta_eff, LABEL2ID["HIGH"], np.argmax(probs[idx, :2], axis=1)
            )

            cloud_n = int((day_preds == LABEL2ID["HIGH"]).sum())
            day_cost = cloud_n * cost_per_query
            spend += day_cost

            high_mask = day_labels == LABEL2ID["HIGH"]
            high_recall = (
                float((day_preds[high_mask] == LABEL2ID["HIGH"]).mean())
                if high_mask.any()
                else float("nan")
            )

            days.append(
                {
                    "day": int(len(days) + 1),
                    "theta_eff": round(float(theta_eff), 4),
                    "cloud_n": cloud_n,
                    "total_n": int(day_count),
                    "day_cost_usd": round(float(day_cost), 4),
                    "cumulative_spend_usd": round(float(spend), 4),
                    "high_recall": round(float(high_recall), 4)
                    if not np.isnan(high_recall)
                    else None,
                }
            )
        return days

    return {
        "theta_base": theta_base,
        "monthly_budget_usd": monthly_budget,
        "cost_per_query_usd": round(cost_per_query, 6),
        "fixed": run_strategy(adaptive=False),
        "adaptive": run_strategy(adaptive=True),
    }


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------


def measure_latency(model, tokenizer, sample_texts: list[str], n: int = 200) -> dict:
    import torch

    model.eval()
    model.to("cpu")
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


# ---------------------------------------------------------------------------
# Full evaluation at a single θ
# ---------------------------------------------------------------------------


def eval_at_theta(probs: np.ndarray, labels: list[str], theta: float) -> dict:
    from sklearn.metrics import classification_report, confusion_matrix

    p_high = probs[:, 2]
    preds = np.where(p_high >= theta, LABEL2ID["HIGH"], np.argmax(probs[:, :2], axis=1))
    pred_names = [ID2LABEL[p] for p in preds]
    label_names = labels

    report = classification_report(
        label_names,
        pred_names,
        labels=["LOW", "MEDIUM", "HIGH"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(label_names, pred_names, labels=["LOW", "MEDIUM", "HIGH"])

    non_high = [lb for lb in label_names if lb != "HIGH"]
    n_leaked = sum(
        1 for t, p in zip(label_names, pred_names, strict=False) if t != "HIGH" and p == "HIGH"
    )
    retention = (1 - n_leaked / len(non_high)) * 100 if non_high else 0.0

    return {
        "theta": theta,
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
        "n_test": len(labels),
    }


# ---------------------------------------------------------------------------
# Wilson score CI
# ---------------------------------------------------------------------------


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset", default=str(TRAIN_DATASET))
    parser.add_argument("--test-dataset", default=str(TEST_DATASET))
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument(
        "--dry-run", action="store_true", help="60 queries, 1 epoch, no real training"
    )
    parser.add_argument(
        "--skip-train", action="store_true", help="Skip training; load saved model from --model-dir"
    )
    parser.add_argument(
        "--theta-default",
        type=float,
        default=0.5,
        help="θ for the single-point eval report (default: 0.5)",
    )
    parser.add_argument(
        "--budget", type=float, default=10.0, help="Monthly budget (USD) for adaptive simulation"
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading datasets...")
    train_q = load_split(Path(args.train_dataset), args.dry_run)
    test_q = load_split(Path(args.test_dataset), args.dry_run)
    print(f"  Train: {len(train_q)} | Test: {len(test_q)}")

    # Use 10% of train as val for early-stopping
    import random

    rng = random.Random(SEED)
    rng.shuffle(train_q)
    val_n = max(1, len(train_q) // 10)
    val_q = train_q[:val_n]
    train_q = train_q[val_n:]
    print(f"  Train (after val split): {len(train_q)} | Val: {len(val_q)}")

    # ------------------------------------------------------------------
    # Train or load
    # ------------------------------------------------------------------
    if args.skip_train:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        print(f"Loading saved model from {args.model_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    else:
        print("\nTraining ModernBERT on balanced dataset...")
        model, tokenizer = train(train_q, val_q, args)

    # ------------------------------------------------------------------
    # Soft scores on test set
    # ------------------------------------------------------------------
    print("\nExtracting soft scores on test set...")
    probs = get_soft_scores(model, tokenizer, test_q)
    labels = [q["ground_truth"] for q in test_q]

    # ------------------------------------------------------------------
    # Single-point evaluation at θ_default
    # ------------------------------------------------------------------
    print(f"\nEvaluating at θ={args.theta_default}...")
    eval_report = eval_at_theta(probs, labels, args.theta_default)

    print(f"\nTest set results (θ={args.theta_default}):")
    print(f"  Accuracy:  {eval_report['accuracy']:.3f}")
    print(f"  Macro-F1:  {eval_report['macro_f1']:.3f}")
    for cls in ["LOW", "MEDIUM", "HIGH"]:
        pc = eval_report["per_class"][cls]
        k = int(round(pc["recall"] * sum(1 for lb in labels if lb == cls)))
        n = sum(1 for lb in labels if lb == cls)
        lo, hi = wilson_ci(k, n)
        print(
            f"  {cls:6s} F1={pc['f1']:.3f}  recall={pc['recall']:.3f}  "
            f"95% CI [{lo:.3f}, {hi:.3f}]"
        )
    cm = eval_report["confusion_matrix"]
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(f"  {'':8s} {'LOW':>6s} {'MED':>6s} {'HIGH':>6s}")
    for i, cls in enumerate(["LOW", "MEDIUM", "HIGH"]):
        print(f"  {cls:6s}   {cm[i][0]:6d} {cm[i][1]:6d} {cm[i][2]:6d}")
    print(f"\n  Free-tier retention: {eval_report['free_tier_retention_pct']:.1f}%")

    # Add Wilson CIs to the report
    eval_report["wilson_ci"] = {}
    for cls in ["LOW", "MEDIUM", "HIGH"]:
        pc = eval_report["per_class"][cls]
        k = int(round(pc["recall"] * sum(1 for lb in labels if lb == cls)))
        n = sum(1 for lb in labels if lb == cls)
        lo, hi = wilson_ci(k, n)
        eval_report["wilson_ci"][cls] = {
            "recall_lo": round(lo, 4),
            "recall_hi": round(hi, 4),
            "k": k,
            "n": n,
        }

    # ------------------------------------------------------------------
    # θ sweep
    # ------------------------------------------------------------------
    print("\nSweeping θ threshold...")
    theta_curve = sweep_theta(probs, labels)

    sweep_path = RESULTS_DIR / "theta_sweep.json"
    with open(sweep_path, "w") as f:
        json.dump(
            {
                "metadata": {
                    "model": MODEL_NAME,
                    "n_test": len(test_q),
                    "label_distribution": {
                        cls: sum(1 for lb in labels if lb == cls)
                        for cls in ["LOW", "MEDIUM", "HIGH"]
                    },
                    "theta_range": {
                        "min": THETA_MIN,
                        "max": THETA_MAX,
                        "step": THETA_STEP,
                        "n_points": 101,
                    },
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                "eval_at_default_theta": eval_report,
                "sweep": theta_curve,
            },
            f,
            indent=2,
        )
    print(f"  θ sweep saved: {sweep_path}")

    # ------------------------------------------------------------------
    # Budget simulation
    # ------------------------------------------------------------------
    print("\nRunning budget-aware routing simulation...")
    sim = simulate_budget_routing(
        probs,
        labels,
        theta_base=args.theta_default,
        monthly_budget=args.budget,
    )
    sim_path = RESULTS_DIR / "budget_simulation.json"
    with open(sim_path, "w") as f:
        json.dump(sim, f, indent=2)
    print(f"  Budget simulation saved: {sim_path}")

    # ------------------------------------------------------------------
    # Latency
    # ------------------------------------------------------------------
    print("\nMeasuring CPU inference latency...")
    lat = measure_latency(model, tokenizer, [q["text"] for q in test_q], n=200)
    print(f"  p50={lat['p50_ms']:.1f}ms  p95={lat['p95_ms']:.1f}ms  p99={lat['p99_ms']:.1f}ms")

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    report = {
        "metadata": {
            "model": MODEL_NAME,
            "train_n": len(train_q),
            "val_n": len(val_q),
            "test_n": len(test_q),
            "epochs": 1 if args.dry_run else EPOCHS,
            "seed": SEED,
            "dry_run": args.dry_run,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "latency_cpu": lat,
        "eval_at_default_theta": eval_report,
        "theta_sweep_path": str(sweep_path),
        "budget_simulation_path": str(sim_path),
    }

    report_path = RESULTS_DIR / "balanced_classifier_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved: {report_path}")

    print("\nNext:")
    print("  python plot_figures.py   # generate Figure 1 + Figure 2")


if __name__ == "__main__":
    main()
