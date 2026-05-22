#!/usr/bin/env python3
"""
Cost-sensitive fine-tuning of ModernBERT for routing classification.

Why ModernBERT and not Llama:
  Llama 3.2 3B is STREAM's local inference tier — fine-tuning it would corrupt the
  production model. More fundamentally, Llama is a generative decoder: it produces
  text token-by-token and does not expose soft class probabilities over {LOW, MEDIUM,
  HIGH}. There is no gradient signal that targets a specific confusion matrix cell.

  ModernBERT is a discriminative encoder with a 3-way classification head. It outputs
  P(LOW), P(MEDIUM), P(HIGH) directly. We can apply an asymmetric cost matrix at the
  loss level, penalising exactly the cells that cause financial harm.

Cost matrix (applied per sample, not as class weights):
  The only errors that cost money are non-HIGH queries routed to cloud (HIGH tier):
    LOW   → HIGH : cost 5  (wrong tier, paid API call)
    MEDIUM→ HIGH : cost 5  (wrong tier, paid API call)
  All other misroutes are between free tiers (cost 0 for routing, small for quality):
    LOW   → MEDIUM: cost 1  (mild quality over-provision)
    MEDIUM→ LOW   : cost 1  (mild quality under-provision)
    HIGH  → LOCAL/HPC: cost 2  (quality degraded, no $ cost)
  Correct predictions: cost 0.

  Implementation: compute standard cross-entropy with reduction='none', then multiply
  each sample's loss by cost_matrix[true_label, predicted_label] where predicted_label
  is the argmax — this steers the model away from the high-cost cells without requiring
  a custom loss function library.

  Alternatively: soft cost weighting uses expected cost under the predicted distribution:
    loss_i = sum_j [ P(j|x_i) * cost(y_i, j) ]  weighted cross-entropy
  This is differentiable and provides a smoother gradient signal.

Usage:
  cd scripts/eval
  python train_cost_sensitive.py
  python train_cost_sensitive.py --dry-run
  python train_cost_sensitive.py --skip-train   # reuse checkpoint, re-eval only
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

TRAIN_DATASET = Path("balanced_training_dataset.json")
TEST_DATASET = Path("balanced_test_dataset.json")
BASE_MODEL_DIR = Path("models/modernbert_balanced")  # start from fine-tuned checkpoint
OUT_MODEL_DIR = Path("models/modernbert_cost_sensitive")
RESULTS_DIR = Path("results")

MODEL_NAME = str(BASE_MODEL_DIR)  # continue from existing fine-tuned weights
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 3  # fewer epochs — we're continuing from a good checkpoint
LR = 5e-6  # lower LR for fine-tuning on top of fine-tuning
SEED = 42

LABEL2ID = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
ID2LABEL = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

# Cost matrix: COST[true_label_id][pred_label_id]
# Rows = true class, Cols = predicted class  (LOW=0, MEDIUM=1, HIGH=2)
COST_MATRIX = np.array(
    [
        [0.0, 1.0, 5.0],  # true=LOW:    correct=0, →MED=1 (free tier, mild), →HIGH=5 (paid leak)
        [
            1.0,
            0.0,
            5.0,
        ],  # true=MEDIUM: →LOW=1 (mild under-provision), correct=0, →HIGH=5 (paid leak)
        [2.0, 2.0, 0.0],  # true=HIGH:   →LOW=2 (quality loss), →MED=2 (quality loss), correct=0
    ],
    dtype=np.float32,
)


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
# Cost-sensitive loss
# ---------------------------------------------------------------------------


def make_cost_sensitive_trainer(cost_matrix_np: np.ndarray):
    """
    Returns a Trainer subclass that replaces cross-entropy with a cost-weighted loss.

    For each sample i with true label y_i, the loss is:
        L_i = sum_j [ P(j | x_i) * cost(y_i, j) ]

    This is the expected routing cost under the current model distribution —
    differentiable, smooth, and directly targets the cells that cause financial harm.
    """
    import torch
    from transformers import Trainer

    cost_tensor = torch.tensor(cost_matrix_np)

    class CostSensitiveTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits  # (B, 3)
            probs = torch.softmax(logits, dim=-1)  # (B, 3)

            # For each sample, look up cost row for its true label: (B, 3)
            cost = cost_tensor.to(logits.device)[labels]  # (B, 3)

            # Expected cost = sum_j P(j) * cost(y, j) per sample, then mean
            per_sample_cost = (probs * cost).sum(dim=-1)  # (B,)
            loss = per_sample_cost.mean()

            return (loss, outputs) if return_outputs else loss

    return CostSensitiveTrainer


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
        output_dir=str(OUT_MODEL_DIR),
        num_train_epochs=1 if args.dry_run else EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=64,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_ratio=0.06,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to="none",
    )

    cost_sensitive_trainer_cls = make_cost_sensitive_trainer(COST_MATRIX)

    trainer = cost_sensitive_trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    t0 = time.perf_counter()
    trainer.train()
    print(f"  Training complete ({(time.perf_counter()-t0)/60:.1f} min)")

    trainer.save_model(str(OUT_MODEL_DIR))
    tokenizer.save_pretrained(str(OUT_MODEL_DIR))
    print(f"  Model saved to {OUT_MODEL_DIR}")

    return trainer.model, tokenizer


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def get_soft_scores(model, tokenizer, queries: list[dict]) -> np.ndarray:
    import torch

    model.eval()
    model.to("cpu")
    all_probs = []
    for i in range(0, len(queries), 64):
        batch = [q["text"] for q in queries[i : i + 64]]
        enc = tokenizer(
            batch, truncation=True, max_length=MAX_LENGTH, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)


def eval_at_theta(probs: np.ndarray, labels: list[str], theta: float) -> dict:
    from sklearn.metrics import classification_report, confusion_matrix

    p_high = probs[:, 2]
    preds = np.where(p_high >= theta, LABEL2ID["HIGH"], np.argmax(probs[:, :2], axis=1))
    pred_names = [ID2LABEL[p] for p in preds]

    report = classification_report(
        labels,
        pred_names,
        labels=["LOW", "MEDIUM", "HIGH"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(labels, pred_names, labels=["LOW", "MEDIUM", "HIGH"])

    non_high = [lb for lb in labels if lb != "HIGH"]
    n_leaked = sum(
        1 for t, p in zip(labels, pred_names, strict=False) if t != "HIGH" and p == "HIGH"
    )
    retention = (1 - n_leaked / len(non_high)) * 100 if non_high else 0.0

    return {
        "theta": theta,
        "accuracy": round(report["accuracy"], 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "per_class": {
            cls: {
                "recall": round(report[cls]["recall"], 4),
                "precision": round(report[cls]["precision"], 4),
                "f1": round(report[cls]["f1-score"], 4),
            }
            for cls in ["LOW", "MEDIUM", "HIGH"]
        },
        "confusion_matrix": cm.tolist(),
        "free_tier_retention_pct": round(retention, 2),
        "n_leaked": n_leaked,
        "n_test": len(labels),
    }


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def sweep_theta(probs: np.ndarray, labels: list[str]) -> list[dict]:
    from sklearn.metrics import f1_score

    label_ids = np.array([LABEL2ID[lb] for lb in labels])
    p_high = probs[:, 2]
    results = []
    for theta in np.arange(0.0, 1.01, 0.01):
        theta = round(float(theta), 3)
        preds = np.where(p_high >= theta, LABEL2ID["HIGH"], np.argmax(probs[:, :2], axis=1))
        macro_f1 = float(
            f1_score(label_ids, preds, average="macro", labels=[0, 1, 2], zero_division=0)
        )
        high_mask = label_ids == LABEL2ID["HIGH"]
        high_recall = (
            float((preds[high_mask] == LABEL2ID["HIGH"]).mean()) if high_mask.any() else 0.0
        )
        cloud_rate = float((preds == LABEL2ID["HIGH"]).mean())
        non_high = label_ids != LABEL2ID["HIGH"]
        cost_leak = float((preds[non_high] == LABEL2ID["HIGH"]).mean()) if non_high.any() else 0.0
        results.append(
            {
                "theta": theta,
                "macro_f1": round(macro_f1, 4),
                "high_recall": round(high_recall, 4),
                "cloud_routing_rate": round(cloud_rate, 4),
                "cost_leak": round(cost_leak, 4),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training; load saved model from OUT_MODEL_DIR",
    )
    parser.add_argument("--theta-default", type=float, default=0.5)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    train_q = load_split(TRAIN_DATASET, args.dry_run)
    test_q = load_split(TEST_DATASET, args.dry_run)
    print(f"  Train: {len(train_q)} | Test: {len(test_q)}")

    import random

    rng = random.Random(SEED)
    rng.shuffle(train_q)
    val_n = max(1, len(train_q) // 10)
    val_q = train_q[:val_n]
    train_q = train_q[val_n:]
    print(f"  Train (after val split): {len(train_q)} | Val: {len(val_q)}")

    print("\nCost matrix (rows=true, cols=predicted):")
    print(f"  {'':8s} {'LOW':>6s} {'MED':>6s} {'HIGH':>6s}")
    for i, cls in enumerate(["LOW", "MEDIUM", "HIGH"]):
        print(f"  {cls:8s} {COST_MATRIX[i,0]:6.1f} {COST_MATRIX[i,1]:6.1f} {COST_MATRIX[i,2]:6.1f}")

    if args.skip_train:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        print(f"\nLoading saved model from {OUT_MODEL_DIR}...")
        tokenizer = AutoTokenizer.from_pretrained(str(OUT_MODEL_DIR))
        model = AutoModelForSequenceClassification.from_pretrained(str(OUT_MODEL_DIR))
    else:
        print(f"\nFine-tuning from {MODEL_NAME} with cost-sensitive loss...")
        model, tokenizer = train(train_q, val_q, args)

    print("\nExtracting soft scores on test set...")
    probs = get_soft_scores(model, tokenizer, test_q)
    labels = [q["ground_truth"] for q in test_q]

    print(f"\nEvaluating at θ={args.theta_default}...")
    result = eval_at_theta(probs, labels, args.theta_default)

    cm = result["confusion_matrix"]
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(f"  {'':8s} {'LOW':>6s} {'MED':>6s} {'HIGH':>6s}  {'Recall':>8s}")
    for i, cls in enumerate(["LOW", "MEDIUM", "HIGH"]):
        recall = result["per_class"][cls]["recall"]
        print(f"  {cls:8s} {cm[i][0]:6d} {cm[i][1]:6d} {cm[i][2]:6d}  {recall:8.1%}")
    print(f"\n  Overall accuracy: {result['accuracy']:.3f}")
    print(f"  Macro-F1:         {result['macro_f1']:.3f}")
    print(f"  Leaked (non-HIGH→HIGH): {result['n_leaked']}")
    print(f"  Free-tier retention:    {result['free_tier_retention_pct']:.1f}%")

    for cls in ["LOW", "MEDIUM", "HIGH"]:
        pc = result["per_class"][cls]
        k = int(round(pc["recall"] * sum(1 for lb in labels if lb == cls)))
        n = sum(1 for lb in labels if lb == cls)
        lo, hi = wilson_ci(k, n)
        print(f"  {cls:6s} recall 95% CI: [{lo:.3f}, {hi:.3f}]")

    # Save theta sweep (overwrites — this is now the active model)
    print("\nSweeping θ threshold...")
    theta_curve = sweep_theta(probs, labels)
    sweep_out = {
        "metadata": {
            "model": str(OUT_MODEL_DIR),
            "training": "cost_sensitive",
            "cost_matrix": COST_MATRIX.tolist(),
            "n_test": len(test_q),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "eval_at_default_theta": result,
        "sweep": theta_curve,
    }
    sweep_path = RESULTS_DIR / "theta_sweep.json"
    sweep_path.write_text(json.dumps(sweep_out, indent=2))
    print(f"  Saved: {sweep_path}")

    # Save full report
    report_path = RESULTS_DIR / "cost_sensitive_report.json"
    report_path.write_text(
        json.dumps(
            {
                "cost_matrix": COST_MATRIX.tolist(),
                "eval_at_default_theta": result,
                "theta_sweep_path": str(sweep_path),
            },
            indent=2,
        )
    )
    print(f"  Saved: {report_path}")

    print("\nNext:")
    print("  python fix_budget_simulation.py   # regenerate budget sim from new sweep")
    print("  python plot_figures.py             # regenerate figures")


if __name__ == "__main__":
    main()
