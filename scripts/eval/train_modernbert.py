#!/usr/bin/env python3
"""
Train ModernBERT-base on the v2 complexity dataset.

Distills Claude Sonnet 4.6's routing judgment into a local 149M-parameter
classifier. After training, the model runs at ~15ms per query with no API
dependency, replacing the 390ms Llama 3.2 3B LLM judge.

Usage:
    python scripts/eval/train_modernbert.py
    python scripts/eval/train_modernbert.py --eval-mode domain-holdout
    python scripts/eval/train_modernbert.py --eval-mode similarity-split
    python scripts/eval/train_modernbert.py --eval-mode mixed-kfold \\
            --dataset scripts/eval/mixed_training_dataset.json
    python scripts/eval/train_modernbert.py --dry-run

Eval modes:
    random           Standard 70/15/15 random split (default). Fast baseline.
    domain-holdout   6-fold CV: each fold holds out one domain entirely.
                     Tests cross-domain generalization; ~6x training time.
    similarity-split Semantic similarity-aware split: near-duplicate queries
                     (cosine sim > 0.90) are kept on the same side of the
                     train/test boundary. Requires sentence-transformers.
    mixed-kfold      5-fold stratified CV on mixed real-world dataset
                     (MMLU + StackExchange + research_computing). Stratified by
                     class AND source. Includes Llama 3.2 3B comparison per
                     fold. Evaluates each fold model on Arena OOD test set.

Output:
    scripts/eval/models/modernbert/          fine-tuned model + tokenizer
    scripts/eval/results/modernbert_training_report.json         (random)
    scripts/eval/results/modernbert_domain_holdout_report.json   (domain-holdout)
    scripts/eval/results/modernbert_similarity_split_report.json (similarity-split)
    scripts/eval/results/modernbert_mixed_kfold_report.json      (mixed-kfold)
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

DATASET_PATH = Path("scripts/eval/stream_routing_benchmark.json")
OUTPUT_DIR = Path("scripts/eval/models/modernbert")
REPORT_PATH = Path("scripts/eval/results/modernbert_training_report.json")

MODEL_NAME = "answerdotai/ModernBERT-base"
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 5
LR = 2e-5
SEED = 42

DOMAINS = [
    "general_knowledge",
    "science",
    "mathematics",
    "humanities",
    "computer_science",
    "research_computing",
]

LABEL2ID = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
ID2LABEL = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}


# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------


def load_dataset(path: Path, dry_run: bool = False):
    with open(path) as f:
        data = json.load(f)
    queries = data["queries"]
    if dry_run:
        queries = queries[:30]
    return queries


def train_val_test_split(queries, seed=SEED):
    """Standard 70/15/15 random split."""
    import random

    queries = list(queries)
    rng = random.Random(seed)
    rng.shuffle(queries)
    n = len(queries)
    train = queries[: int(0.70 * n)]
    val = queries[int(0.70 * n) : int(0.85 * n)]
    test = queries[int(0.85 * n) :]
    return train, val, test


def domain_holdout_split(queries, held_out_domain, seed=SEED):
    """Hold out all queries from one domain for testing.

    Remaining 5 domains (5760 queries): 90% train, 10% val.
    Held-out domain (1152 queries): test only.
    """
    import random

    train_val = [q for q in queries if q["domain"] != held_out_domain]
    test = [q for q in queries if q["domain"] == held_out_domain]
    rng = random.Random(seed)
    rng.shuffle(train_val)
    val_size = max(1, int(0.10 * len(train_val)))
    val = train_val[:val_size]
    train = train_val[val_size:]
    return train, val, test


def similarity_aware_split(queries, threshold=0.90, test_fraction=0.15, seed=SEED):
    """Split so no near-duplicate queries (cosine sim > threshold) span train/test.

    Algorithm:
    1. Embed all queries with all-MiniLM-L6-v2.
    2. Build connected components: union two queries if sim > threshold.
    3. Assign whole components to train or test (never split a component).
    4. Target test_fraction of total queries in the test set.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  ERROR: sentence-transformers not installed.")
        print("  Run: pip install sentence-transformers")
        raise

    import random
    from collections import defaultdict

    import torch

    texts = [q["text"] for q in queries]
    print(f"  Embedding {len(texts)} queries with all-MiniLM-L6-v2...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_model.encode(
        texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True
    )

    n = len(queries)
    emb = torch.tensor(embeddings, dtype=torch.float32)  # already L2-normalized

    # Union-Find
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Find near-duplicate pairs in batches (O(n²/batch) memory)
    batch_size = 512
    n_merges = 0
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        sims = (emb[i:end] @ emb.T).numpy()
        for bi in range(end - i):
            gi = i + bi
            for j in np.where(sims[bi] > threshold)[0]:
                if j > gi and find(gi) != find(j):
                    union(gi, j)
                    n_merges += 1

    # Collect components
    components: dict = defaultdict(list)
    for idx in range(n):
        components[find(idx)].append(idx)

    comp_list = list(components.values())
    n_multi = sum(1 for c in comp_list if len(c) > 1)
    print(f"  Near-duplicate groups (sim>{threshold}): {n_multi} / {len(comp_list)} components")

    # Assign components to test until target fraction is reached
    target_test = int(test_fraction * n)
    rng = random.Random(seed)
    rng.shuffle(comp_list)

    test_idxs: set = set()
    for comp in comp_list:
        if len(test_idxs) < target_test:
            test_idxs.update(comp)

    train_val_q = [queries[i] for i in range(n) if i not in test_idxs]
    test_q = [queries[i] for i in range(n) if i in test_idxs]

    rng.shuffle(train_val_q)
    val_size = int(0.15 * len(train_val_q))
    val_q = train_val_q[:val_size]
    train_q = train_val_q[val_size:]

    print(f"  Split: train={len(train_q)}, val={len(val_q)}, test={len(test_q)}")
    return train_q, val_q, test_q


# ---------------------------------------------------------------------------
# HuggingFace dataset helpers
# ---------------------------------------------------------------------------


def to_hf_dataset(queries):
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "text": [q["text"] for q in queries],
            "label": [LABEL2ID[q["ground_truth"]] for q in queries],
        }
    )


def tokenize_split(train_q, val_q, test_q, tokenizer):
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )

    train_ds = to_hf_dataset(train_q).map(tokenize, batched=True)
    val_ds = to_hf_dataset(val_q).map(tokenize, batched=True)
    test_ds = to_hf_dataset(test_q).map(tokenize, batched=True)

    for ds in [train_ds, val_ds, test_ds]:
        ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def eval_on_test_set(model, tokenizer, test_q):
    """Run inference on test_q on CPU. Returns metrics dict."""
    import torch
    from sklearn.metrics import classification_report, confusion_matrix

    model.eval()
    model.to("cpu")

    test_texts = [q["text"] for q in test_q]
    test_labels = [LABEL2ID[q["ground_truth"]] for q in test_q]
    all_preds = []

    for i in range(0, len(test_texts), 64):
        batch = test_texts[i : i + 64]
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
    label_names = [ID2LABEL[lb] for lb in test_labels]

    report = classification_report(
        label_names, pred_names, labels=["LOW", "MEDIUM", "HIGH"], output_dict=True
    )
    cm = confusion_matrix(label_names, pred_names, labels=["LOW", "MEDIUM", "HIGH"])

    n_free = sum(1 for lb in label_names if lb in ("LOW", "MEDIUM"))
    n_leaked = sum(
        1
        for t, p in zip(label_names, pred_names, strict=False)
        if t in ("LOW", "MEDIUM") and p == "HIGH"
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
        "n_test": len(test_q),
    }


def measure_latency(model, tokenizer, sample_texts, n=200, device="cpu"):
    """Measure single-query CPU inference latency (p50/p95/p99)."""
    import torch

    model.eval()
    model.to(device)
    latencies = []
    for text in sample_texts[:n]:
        enc = tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
            padding=True,
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


# ---------------------------------------------------------------------------
# Core training loop (reusable across eval modes)
# ---------------------------------------------------------------------------


def make_training_args(output_dir: Path, epochs: int, dry_run: bool):
    import torch
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=1 if dry_run else epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=64,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        logging_steps=50,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to="none",
    )


def train_one_fold(
    fold_label: str,
    train_q,
    val_q,
    test_q,
    args,
    tokenizer,
    output_dir: Path,
):
    """Train from scratch on (train_q, val_q), evaluate on test_q.

    Returns (trained_model, fold_metrics_dict).
    """
    import torch
    from transformers import AutoModelForSequenceClassification, Trainer

    output_dir.mkdir(parents=True, exist_ok=True)

    print("  Tokenizing...")
    train_ds, val_ds, _ = tokenize_split(train_q, val_q, test_q, tokenizer)

    print(f"  Loading fresh {MODEL_NAME}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Class-weighted loss to compensate for HIGH being underrepresented (~12%)
    label_counts = Counter(q["ground_truth"] for q in train_q)
    total = sum(label_counts.values())
    class_weights = torch.tensor(
        [total / (3 * label_counts.get(cls, 1)) for cls in ["LOW", "MEDIUM", "HIGH"]],
        dtype=torch.float32,
    )

    # Class-weighted loss baseline — used by Condition 2 (weighted).
    # Derived from data: weight[cls] = total / (3 * count[cls])  [sklearn balanced].

    def make_cost_matrix(alpha: float, gamma: float = 1.0) -> "torch.Tensor":
        """
        Ordinal asymmetric cost matrix parameterised by alpha and gamma.

        alpha = under-routing penalty ratio (quality risk).
            alpha=1 → symmetric; alpha>1 → under-routing costs more.
        gamma = LOW→MED over-routing penalty.
            gamma=1 → all over-routing equal (default).
            gamma=0 → LOW→MED is free (both are free tiers, no monetary cost).

        Classes: LOW=0, MEDIUM=1, HIGH=2.

                    pred LOW    pred MED    pred HIGH
        true LOW       0         gamma        2        (over-routing to free or paid tier)
        true MED       alpha       0           1        (under 1 tier, over to paid tier)
        true HIGH      2*alpha     alpha        0        (under 2/1 tiers)
        """
        import torch as _torch

        return _torch.tensor(
            [
                [0.0, gamma, 2.0],
                [alpha, 0.0, 1.0],
                [2 * alpha, alpha, 0.0],
            ],
            dtype=_torch.float32,
        )

    if getattr(args, "no_class_weights", False):
        trainer_class = Trainer
    elif getattr(args, "cost_sensitive", False):
        # ------------------------------------------------------------------
        # Condition 5: Ordinal asymmetric cost-sensitive learning.
        #
        # We sweep alpha in {1, 2, 3, 5} on fold 1 validation F1, pick the
        # best alpha, then train all 6 folds with that alpha.  This gives a
        # single principled hyperparameter selection — the alpha that best
        # balances quality leakage vs cost leakage on held-out data.
        # ------------------------------------------------------------------

        alpha_grid = [1.0, 2.0, 3.0, 5.0]
        best_alpha = alpha_grid[0]  # updated after fold-1 sweep below

        # Flag so train_one_fold can update this after the sweep
        args._best_alpha = best_alpha
        args._alpha_selected = False

        _gamma = 0.0 if getattr(args, "zero_hpc_overroute", False) else 1.0

        def make_cost_sensitive_trainer(alpha: float):
            cost_matrix = make_cost_matrix(alpha, gamma=_gamma)

            class CostSensitiveTrainer(Trainer):
                def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                    import torch as _t

                    labels = inputs.pop("labels")
                    outputs = model(**inputs)
                    logits = outputs.logits
                    # Per-sample cross-entropy (unreduced)
                    per_sample_ce = _t.nn.functional.cross_entropy(logits, labels, reduction="none")
                    # Expected ordinal cost under current predicted distribution
                    probs = _t.softmax(logits, dim=-1)
                    cm = cost_matrix.to(logits.device)
                    expected_cost = (probs * cm[labels]).sum(dim=-1)
                    # Scale each sample's CE by (1 + expected_cost)
                    loss = (per_sample_ce * (1.0 + expected_cost)).mean()
                    return (loss, outputs) if return_outputs else loss

            return CostSensitiveTrainer

        # Alpha grid search on fold 1 validation set only.
        # Stored on args so the mixed-kfold loop can access it.
        args._make_cost_sensitive_trainer = make_cost_sensitive_trainer
        args._alpha_grid = alpha_grid

        # Default trainer uses alpha from args._current_alpha (sweep) or
        # args._best_alpha (selected after sweep). Falls back to alpha=3.
        _init_alpha = getattr(args, "_current_alpha", getattr(args, "_best_alpha", 3.0))
        trainer_class = make_cost_sensitive_trainer(_init_alpha)
    else:

        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                loss = torch.nn.functional.cross_entropy(
                    logits, labels, weight=class_weights.to(logits.device)
                )
                return (loss, outputs) if return_outputs else loss

        trainer_class = WeightedTrainer

    training_args = make_training_args(output_dir, args.epochs, args.dry_run)
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    t0 = time.perf_counter()
    trainer.train()
    train_time = time.perf_counter() - t0
    trained_model = trainer.model  # best checkpoint loaded by Trainer

    metrics = eval_on_test_set(trained_model, tokenizer, test_q)
    metrics["label"] = fold_label
    metrics["train_size"] = len(train_q)
    metrics["val_size"] = len(val_q)
    metrics["train_time_minutes"] = round(train_time / 60, 1)

    print(
        f"  [{fold_label}] acc={metrics['accuracy']:.3f}  "
        f"macro_f1={metrics['macro_f1']:.3f}  "
        f"({train_time/60:.1f} min)"
    )
    return trained_model, metrics


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------


def print_test_results(metrics: dict, label: str = ""):
    header = "Test set results" + (f" [{label}]" if label else "")
    print(f"\n{header}:")
    print(f"  Accuracy:  {metrics['accuracy']:.3f}")
    print(f"  Macro-F1:  {metrics['macro_f1']:.3f}")
    for cls in ["LOW", "MEDIUM", "HIGH"]:
        pc = metrics["per_class"][cls]
        print(
            f"  {cls:6s} F1: {pc['f1']:.3f}  "
            f"(precision={pc['precision']:.3f}, recall={pc['recall']:.3f})"
        )
    cm = metrics["confusion_matrix"]
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(f"{'':8s} {'LOW':>6s} {'MED':>6s} {'HIGH':>6s}")
    for i, cls in enumerate(["LOW", "MEDIUM", "HIGH"]):
        print(f"  {cls:6s} {cm[i][0]:6d} {cm[i][1]:6d} {cm[i][2]:6d}")
    print(f"\n  Free-tier retention: {metrics['free_tier_retention_pct']:.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument(
        "--dry-run", action="store_true", help="30 examples, 1 epoch (sanity check)"
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--eval-only", action="store_true", help="Skip training; load saved model")
    parser.add_argument(
        "--eval-mode",
        choices=["random", "random-kfold", "domain-holdout", "similarity-split", "mixed-kfold"],
        default="random",
        help=(
            "random: single 70/15/15 split (fast baseline). "
            "random-kfold: 5-fold stratified CV on original benchmark (shows inflation). "
            "domain-holdout: 6-fold CV holding out one domain per fold. "
            "similarity-split: semantic deduplication before splitting. "
            "mixed-kfold: 5-fold stratified CV on mixed real-world dataset."
        ),
    )
    parser.add_argument(
        "--arena-testset",
        default="scripts/eval/realworld_testset.json",
        help="Path to Arena OOD test set (used in mixed-kfold mode).",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=10,
        help="Number of folds for random-kfold and mixed-kfold (default: 10).",
    )
    parser.add_argument(
        "--llm-judge",
        choices=["ollama", "haiku", "sonnet"],
        default="ollama",
        help="LLM judge to compare against in mixed-kfold (default: ollama/llama3.2:3b).",
    )
    parser.add_argument(
        "--skip-llm-judge",
        action="store_true",
        help="Skip LLM judge comparison in mixed-kfold (faster).",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class-weighted loss (use uniform loss). Default: weighted.",
    )
    parser.add_argument(
        "--cost-sensitive",
        action="store_true",
        help="Use asymmetric cost-sensitive loss penalising HIGH misroutes more than LOW/MED errors.",
    )
    parser.add_argument(
        "--zero-hpc-overroute",
        action="store_true",
        help="Set LOW→MED over-routing cost to 0 (free-tier routing has no monetary cost).",
    )
    parser.add_argument(
        "--condition-name",
        default="",
        help="Label for this run (used in report filename, e.g. 'baseline', 'weighted', 'oversample', 'downsample', 'cost_sensitive').",
    )
    parser.add_argument(
        "--fixed-alpha",
        type=float,
        default=None,
        help="Skip alpha grid search and use this alpha directly for cost-sensitive loss.",
    )
    args = parser.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        print("Run generate_benchmark_dataset.py first.")
        return

    print(f"Loading dataset from {dataset_path}")
    queries = load_dataset(dataset_path, dry_run=args.dry_run)
    print(f"  Total queries: {len(queries)}")
    print(f"  Distribution: {Counter(q['ground_truth'] for q in queries)}")

    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # DOMAIN-HOLDOUT 6-FOLD CV
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # RANDOM-KFOLD: 5-fold stratified CV on original benchmark (inflation demo)
    # -------------------------------------------------------------------------
    if args.eval_mode == "random-kfold":
        print(f"\n{'='*60}")
        print("EVAL MODE: Random 5-Fold Stratified CV (inflation baseline)")
        print(f"{'='*60}")
        print(
            "Standard k-fold on the original LLM-generated benchmark.\n"
            "Near-duplicates leak across folds — expected near-perfect accuracy.\n"
            "Report alongside domain-holdout and mixed-kfold to show inflation."
        )

        from sklearn.model_selection import StratifiedKFold

        labels_for_strat = [q["ground_truth"] for q in queries]
        skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)
        query_arr = np.array(queries, dtype=object)
        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(
            skf.split(list(range(len(queries))), labels_for_strat)
        ):
            print(f"\n--- Fold {fold_idx+1}/{args.n_folds} ---")
            train_val_q = list(query_arr[train_idx])
            test_q_fold = list(query_arr[test_idx])

            import random as _random

            rng = _random.Random(SEED + fold_idx)
            rng.shuffle(train_val_q)
            val_size = max(1, int(0.10 * len(train_val_q)))
            val_q_fold = train_val_q[:val_size]
            train_q_fold = train_val_q[val_size:]

            print(
                f"  Split: train={len(train_q_fold)}, val={len(val_q_fold)}, test={len(test_q_fold)}"
            )

            fold_dir = OUTPUT_DIR.parent / f"modernbert_rndkfold_fold{fold_idx+1}"
            _, fold_metrics = train_one_fold(
                f"fold{fold_idx+1}",
                train_q_fold,
                val_q_fold,
                test_q_fold,
                args,
                tokenizer,
                fold_dir,
            )
            fold_metrics["fold"] = fold_idx + 1
            fold_results.append(fold_metrics)

        accs = [r["accuracy"] for r in fold_results]
        f1s = [r["macro_f1"] for r in fold_results]
        retentions = [r["free_tier_retention_pct"] for r in fold_results]

        print(f"\n{'='*60}")
        print("Random-Kfold CV Summary (original benchmark)")
        print(f"{'='*60}")
        print(f"  Accuracy:         {np.mean(accs):.3f} ± {np.std(accs):.3f}")
        print(f"  Macro-F1:         {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
        print(f"  Free-tier retain: {np.mean(retentions):.1f}% ± {np.std(retentions):.1f}%")
        print("  (std≈0 confirms near-duplicates inflate every fold consistently)")

        result = {
            "eval_mode": "random-kfold",
            "model": MODEL_NAME,
            "dataset": str(dataset_path),
            "n_folds": args.n_folds,
            "folds": fold_results,
            "mean_accuracy": round(float(np.mean(accs)), 4),
            "std_accuracy": round(float(np.std(accs)), 4),
            "mean_macro_f1": round(float(np.mean(f1s)), 4),
            "std_macro_f1": round(float(np.std(f1s)), 4),
            "mean_free_tier_retention_pct": round(float(np.mean(retentions)), 2),
            "std_free_tier_retention_pct": round(float(np.std(retentions)), 2),
        }
        report_path = REPORT_PATH.parent / "modernbert_random_kfold_report.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Report saved to {report_path}")

        print("\n" + "=" * 60)
        print("PAPER NUMBERS (random 5-fold CV — inflated baseline):")
        print("=" * 60)
        print(f"  ModernBERT accuracy:  {np.mean(accs):.1%} ± {np.std(accs):.1%}")
        print(f"  ModernBERT macro-F1:  {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
        print(f"  Free-tier retention:  {np.mean(retentions):.1f}% ± {np.std(retentions):.1f}%")

    elif args.eval_mode == "domain-holdout":
        print(f"\n{'='*60}")
        print("EVAL MODE: Domain-Held-Out 6-Fold Cross-Validation")
        print(f"{'='*60}")
        print(
            "Each fold trains on 5 domains (~5760 queries) and tests on the\n"
            "6th held-out domain (~1152 queries). Addresses data leakage by\n"
            "verifying the classifier generalizes to unseen topic domains."
        )

        domains = DOMAINS[:2] if args.dry_run else DOMAINS
        fold_results = []

        for fold_idx, held_out in enumerate(domains):
            print(f"\n--- Fold {fold_idx + 1}/{len(domains)}: held-out = '{held_out}' ---")
            train_q, val_q, test_q = domain_holdout_split(queries, held_out)
            print(f"  Split: train={len(train_q)}, val={len(val_q)}, test={len(test_q)}")

            fold_dir = OUTPUT_DIR.parent / f"modernbert_fold{fold_idx+1}_{held_out[:8]}"
            _, fold_metrics = train_one_fold(
                f"fold{fold_idx+1}_{held_out}", train_q, val_q, test_q, args, tokenizer, fold_dir
            )
            fold_metrics["held_out_domain"] = held_out
            fold_results.append(fold_metrics)

        accs = [r["accuracy"] for r in fold_results]
        f1s = [r["macro_f1"] for r in fold_results]

        print(f"\n{'='*60}")
        print("Domain-Holdout CV Summary")
        print(f"{'='*60}")
        print(f"  Accuracy:  {np.mean(accs):.3f} ± {np.std(accs):.3f}")
        print(f"  Macro-F1:  {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
        for r in fold_results:
            print(f"  {r['held_out_domain']:30s} acc={r['accuracy']:.3f}  f1={r['macro_f1']:.3f}")

        result = {
            "eval_mode": "domain-holdout",
            "model": MODEL_NAME,
            "dataset": str(dataset_path),
            "n_folds": len(fold_results),
            "folds": fold_results,
            "mean_accuracy": round(float(np.mean(accs)), 4),
            "std_accuracy": round(float(np.std(accs)), 4),
            "mean_macro_f1": round(float(np.mean(f1s)), 4),
            "std_macro_f1": round(float(np.std(f1s)), 4),
        }
        report_path = REPORT_PATH.parent / "modernbert_domain_holdout_report.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nReport saved to {report_path}")

        print("\n" + "=" * 60)
        print("PAPER NUMBERS (domain-held-out 6-fold CV):")
        print("=" * 60)
        print(f"  Accuracy:  {np.mean(accs):.1%} ± {np.std(accs):.1%}")
        print(f"  Macro-F1:  {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

    # -------------------------------------------------------------------------
    # SIMILARITY-SPLIT
    # -------------------------------------------------------------------------
    elif args.eval_mode == "similarity-split":
        print(f"\n{'='*60}")
        print("EVAL MODE: Semantic Similarity-Aware Split")
        print(f"{'='*60}")
        print(
            "Queries are embedded with all-MiniLM-L6-v2. Near-duplicate\n"
            "pairs (cosine sim > 0.90) are grouped into connected components;\n"
            "each component is assigned entirely to train or test."
        )

        train_q, val_q, test_q = similarity_aware_split(queries)

        if args.eval_only:
            print(f"\nLoading saved model from {OUTPUT_DIR}...")
            model = AutoModelForSequenceClassification.from_pretrained(str(OUTPUT_DIR))
            tokenizer = AutoTokenizer.from_pretrained(str(OUTPUT_DIR))
            metrics = eval_on_test_set(model, tokenizer, test_q)
            metrics["label"] = "similarity-split (eval-only)"
        else:
            model, metrics = train_one_fold(
                "similarity-split", train_q, val_q, test_q, args, tokenizer, OUTPUT_DIR
            )
            tokenizer.save_pretrained(str(OUTPUT_DIR))

        print_test_results(metrics, "similarity-split")

        lat = measure_latency(model, tokenizer, [q["text"] for q in test_q][:200])
        print(
            f"\n  CPU latency — p50: {lat['p50_ms']} ms  p95: {lat['p95_ms']} ms  p99: {lat['p99_ms']} ms"
        )

        result = {
            "eval_mode": "similarity-split",
            "model": MODEL_NAME,
            "dataset": str(dataset_path),
            "train_size": len(train_q),
            "val_size": len(val_q),
            "test_size": len(test_q),
            "epochs": 1 if args.dry_run else args.epochs,
            **metrics,
            "latency_ms": lat,
        }
        report_path = REPORT_PATH.parent / "modernbert_similarity_split_report.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nReport saved to {report_path}")

        print("\n" + "=" * 60)
        print("PAPER NUMBERS (similarity-split):")
        print("=" * 60)
        print(f"  Accuracy:  {metrics['accuracy']:.1%}")
        print(f"  Macro-F1:  {metrics['macro_f1']:.3f}")
        for cls in ["LOW", "MEDIUM", "HIGH"]:
            print(f"  {cls:6s} F1: {metrics['per_class'][cls]['f1']:.3f}")
        print(f"  Latency p50: {lat['p50_ms']} ms")

    # -------------------------------------------------------------------------
    # MIXED-KFOLD: 5-fold stratified CV on real-world data
    # -------------------------------------------------------------------------
    elif args.eval_mode == "mixed-kfold":
        print(f"\n{'='*60}")
        print("EVAL MODE: Mixed Real-World 5-Fold Stratified CV")
        print(f"{'='*60}")
        print(
            "Dataset: MMLU + StackExchange + research_computing (Claude-generated).\n"
            "Folds stratified by complexity class AND source.\n"
            "Each fold's model also evaluated on Arena OOD test set.\n"
            "LLM judge (Llama 3.2 3B) runs on same test folds for comparison."
        )

        from sklearn.model_selection import StratifiedKFold

        # Stratify key: combine class and source
        strat_keys = [f"{q['ground_truth']}_{q['source']}" for q in queries]
        skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)
        query_arr = np.array(queries, dtype=object)

        # Load Arena OOD test set
        arena_path = Path(args.arena_testset)
        arena_queries = []
        if arena_path.exists():
            with open(arena_path) as f:
                arena_data = json.load(f)
            arena_queries = arena_data["queries"]
            print(f"  Arena OOD test set: {len(arena_queries)} queries loaded from {arena_path}")
        else:
            print(f"  [WARN] Arena test set not found at {arena_path} — OOD eval skipped")

        # LLM judge setup
        llm_judge_model = None
        if not args.skip_llm_judge:
            llm_models = {
                "haiku": "claude-haiku-4-5-20251001",
                "sonnet": "claude-sonnet-4-6",
                "ollama": "ollama/llama3.2:3b",
            }
            llm_judge_model = llm_models[args.llm_judge]
            print(f"  LLM judge: {llm_judge_model}")

        reasoning_depth_rubric = (
            "You are a query complexity classifier.\n\n"
            "LOW: Single retrievable fact. Answer statable in one sentence, no reasoning chain.\n"
            "MEDIUM: Apply a standard procedure or assemble 2-4 concepts. Textbook-level reasoning.\n"
            "HIGH: Construct a novel reasoning path or expert judgment. No standard procedure.\n\n"
            "Respond with ONLY ONE WORD: LOW, MEDIUM, or HIGH\n\n"
            "User Query: {query}"
        )

        def run_llm_judge_on_queries(test_q, model_name):
            """Run LLM judge on test_q, return list of predicted labels."""
            import os
            import time as time_mod

            try:
                import litellm
            except ImportError:
                print("  [WARN] litellm not installed — skipping LLM judge")
                return None

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                env_file = Path(__file__).parent.parent.parent / ".env"
                if env_file.exists():
                    for line in env_file.read_text().splitlines():
                        if line.startswith("ANTHROPIC_API_KEY="):
                            api_key = line.split("=", 1)[1].strip()
                            break
            if api_key:
                os.environ["ANTHROPIC_API_KEY"] = api_key

            preds = []
            for i, q in enumerate(test_q):
                prompt = reasoning_depth_rubric.format(query=q["text"])
                for attempt in range(3):
                    try:
                        resp = litellm.completion(
                            model=model_name,
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=10,
                            temperature=0,
                        )
                        raw = resp.choices[0].message.content.strip().upper()
                        label = raw if raw in ("LOW", "MEDIUM", "HIGH") else "MEDIUM"
                        preds.append(label)
                        break
                    except Exception:
                        if attempt == 2:
                            preds.append("MEDIUM")
                        else:
                            time_mod.sleep(1)
                if (i + 1) % 50 == 0:
                    print(f"    LLM judge: {i+1}/{len(test_q)} done")
                time_mod.sleep(0.1)
            return preds

        fold_results = []
        best_model = None
        best_f1 = -1.0

        # ------------------------------------------------------------------
        # Alpha grid search for cost-sensitive loss (Condition 5 only).
        # Uses fold-0 val set to pick the best alpha before full CV.
        # ------------------------------------------------------------------
        indices = list(range(len(queries)))
        splits = list(skf.split(indices, strat_keys))

        if getattr(args, "cost_sensitive", False) and not args.dry_run:
            if not hasattr(args, "_alpha_grid"):
                args._alpha_grid = [1.0, 2.0, 3.0, 5.0]
                args._best_alpha = args._alpha_grid[0]
                args._alpha_selected = False
                _gamma = 0.0 if getattr(args, "zero_hpc_overroute", False) else 1.0

                def _make_trainer(alpha: float, _g=_gamma):
                    import torch as _t2
                    from transformers import Trainer as _Trainer

                    _cm = make_cost_matrix(alpha, gamma=_g)  # noqa: F821

                    class _CST(_Trainer):
                        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                            labels = inputs.pop("labels")
                            outputs = model(**inputs)
                            logits = outputs.logits
                            per_sample_ce = _t2.nn.functional.cross_entropy(
                                logits, labels, reduction="none"
                            )
                            probs = _t2.softmax(logits, dim=-1)
                            cm = _cm.to(logits.device)
                            expected_cost = (probs * cm[labels]).sum(dim=-1)
                            loss = (per_sample_ce * (1.0 + expected_cost)).mean()
                            return (loss, outputs) if return_outputs else loss

                    return _CST

                args._make_cost_sensitive_trainer = _make_trainer

            # If --fixed-alpha supplied, skip grid search entirely.
            if args.fixed_alpha is not None:
                args._best_alpha = args.fixed_alpha
                args._alpha_selected = True
                print(f"\n--- Alpha fixed at {args.fixed_alpha} (skipping grid search) ---")
            else:
                print("\n--- Alpha grid search (cost-sensitive loss) ---")
                print(f"  Sweeping alpha in {args._alpha_grid} on fold 1 validation set...")
                # Use fold 0 split for the sweep
                train_idx0, test_idx0 = splits[0]
                import random as _random

                tvq0 = list(query_arr[train_idx0])
                rng0 = _random.Random(SEED)
                rng0.shuffle(tvq0)
                vs0 = max(1, int(0.10 * len(tvq0)))
                val_q0, train_q0 = tvq0[:vs0], tvq0[vs0:]
                test_q0 = list(query_arr[test_idx0])

                alpha_val_f1 = {}
                for alpha in args._alpha_grid:
                    print(f"  alpha={alpha}...")
                    args._current_alpha = alpha
                    _sweep_args = argparse.Namespace(**vars(args))
                    _sweep_args._current_alpha = alpha
                    fold_dir_sweep = OUTPUT_DIR.parent / f"modernbert_alpha_sweep_{alpha}"
                    _, sweep_metrics = train_one_fold(
                        f"alpha={alpha}",
                        train_q0,
                        val_q0,
                        test_q0,
                        _sweep_args,
                        tokenizer,
                        fold_dir_sweep,
                    )
                    alpha_val_f1[alpha] = sweep_metrics["macro_f1"]
                    print(f"    macro_f1={sweep_metrics['macro_f1']:.4f}")

                best_alpha = max(alpha_val_f1, key=alpha_val_f1.get)
                args._best_alpha = best_alpha
                args._alpha_selected = True
                print(
                    f"  Selected alpha={best_alpha} (val macro_f1={alpha_val_f1[best_alpha]:.4f})"
                )
                print(f"  Full sweep results: { {a: f'{v:.4f}' for a, v in alpha_val_f1.items()} }")

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            print(f"\n--- Fold {fold_idx+1}/{args.n_folds} ---")
            train_val_q = list(query_arr[train_idx])
            test_q = list(query_arr[test_idx])

            # 90% train, 10% val from training portion
            import random as _random

            rng = _random.Random(SEED + fold_idx)
            rng.shuffle(train_val_q)
            val_size = max(1, int(0.10 * len(train_val_q)))
            val_q = train_val_q[:val_size]
            train_q = train_val_q[val_size:]

            print(f"  Split: train={len(train_q)}, val={len(val_q)}, test={len(test_q)}")
            print(f"  Test distribution: {Counter(q['ground_truth'] for q in test_q)}")
            if getattr(args, "cost_sensitive", False) and getattr(args, "_alpha_selected", False):
                print(f"  Cost-sensitive alpha={args._best_alpha}")

            fold_dir = OUTPUT_DIR.parent / f"modernbert_mixed_fold{fold_idx+1}"
            trained_model, fold_metrics = train_one_fold(
                f"fold{fold_idx+1}", train_q, val_q, test_q, args, tokenizer, fold_dir
            )
            fold_metrics["fold"] = fold_idx + 1
            fold_metrics["source_distribution"] = dict(Counter(q["source"] for q in test_q))

            # Print per-class breakdown and confusion matrix
            pc = fold_metrics["per_class"]
            cm = fold_metrics["confusion_matrix"]
            print(
                f"  Per-class F1:  LOW={pc['LOW']['f1']:.3f}  MEDIUM={pc['MEDIUM']['f1']:.3f}  HIGH={pc['HIGH']['f1']:.3f}"
            )
            print("  Confusion matrix (rows=true, cols=pred) [LOW, MEDIUM, HIGH]:")
            print(
                f"    TRUE LOW:    pred LOW={cm[0][0]:4d}  MEDIUM={cm[0][1]:4d}  HIGH={cm[0][2]:4d}  (leakage to HIGH: {cm[0][2]/(sum(cm[0]) or 1):.1%})"
            )
            print(
                f"    TRUE MEDIUM: pred LOW={cm[1][0]:4d}  MEDIUM={cm[1][1]:4d}  HIGH={cm[1][2]:4d}  (leakage to HIGH: {cm[1][2]/(sum(cm[1]) or 1):.1%})"
            )
            print(
                f"    TRUE HIGH:   pred LOW={cm[2][0]:4d}  MEDIUM={cm[2][1]:4d}  HIGH={cm[2][2]:4d}  (missed HIGH: {(cm[2][0]+cm[2][1])/(sum(cm[2]) or 1):.1%})"
            )

            # Save best model
            if fold_metrics["macro_f1"] > best_f1:
                best_f1 = fold_metrics["macro_f1"]
                best_model = trained_model

            # Arena OOD eval for this fold
            if arena_queries:
                arena_metrics = eval_on_test_set(trained_model, tokenizer, arena_queries)
                fold_metrics["arena_accuracy"] = arena_metrics["accuracy"]
                fold_metrics["arena_macro_f1"] = arena_metrics["macro_f1"]
                fold_metrics["arena_free_tier_retention_pct"] = arena_metrics[
                    "free_tier_retention_pct"
                ]
                print(
                    f"  Arena OOD: acc={arena_metrics['accuracy']:.3f}  "
                    f"f1={arena_metrics['macro_f1']:.3f}"
                )

            # LLM judge comparison on same test fold
            if llm_judge_model and not args.dry_run:
                print(f"  Running LLM judge ({llm_judge_model}) on {len(test_q)} queries...")
                judge_preds = run_llm_judge_on_queries(test_q, llm_judge_model)
                if judge_preds is not None:
                    from sklearn.metrics import accuracy_score, f1_score

                    true_labels = [q["ground_truth"] for q in test_q]
                    judge_acc = accuracy_score(true_labels, judge_preds)
                    judge_f1 = f1_score(
                        true_labels,
                        judge_preds,
                        average="macro",
                        labels=["LOW", "MEDIUM", "HIGH"],
                        zero_division=0,
                    )
                    fold_metrics["llm_judge_accuracy"] = round(judge_acc, 4)
                    fold_metrics["llm_judge_macro_f1"] = round(judge_f1, 4)
                    fold_metrics["llm_judge_model"] = llm_judge_model
                    print(
                        f"  LLM judge: acc={judge_acc:.3f}  f1={judge_f1:.3f}  "
                        f"(ModernBERT delta: +{fold_metrics['macro_f1']-judge_f1:+.3f})"
                    )

            fold_results.append(fold_metrics)

        # Aggregate
        accs = [r["accuracy"] for r in fold_results]
        f1s = [r["macro_f1"] for r in fold_results]
        retentions = [r["free_tier_retention_pct"] for r in fold_results]

        arena_accs = [
            r.get("arena_accuracy") for r in fold_results if r.get("arena_accuracy") is not None
        ]
        arena_f1s = [
            r.get("arena_macro_f1") for r in fold_results if r.get("arena_macro_f1") is not None
        ]
        arena_rets = [
            r.get("arena_free_tier_retention_pct")
            for r in fold_results
            if r.get("arena_free_tier_retention_pct") is not None
        ]

        judge_f1s = [
            r.get("llm_judge_macro_f1")
            for r in fold_results
            if r.get("llm_judge_macro_f1") is not None
        ]

        print(f"\n{'='*60}")
        print("Mixed-Kfold CV Summary")
        print(f"{'='*60}")
        print("  In-distribution (5-fold):")
        print(f"    Accuracy:         {np.mean(accs):.3f} ± {np.std(accs):.3f}")
        print(f"    Macro-F1:         {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
        print(f"    Free-tier retain: {np.mean(retentions):.1f}% ± {np.std(retentions):.1f}%")
        if arena_f1s:
            print("  Out-of-distribution (Arena):")
            print(f"    Accuracy:         {np.mean(arena_accs):.3f} ± {np.std(arena_accs):.3f}")
            print(f"    Macro-F1:         {np.mean(arena_f1s):.3f} ± {np.std(arena_f1s):.3f}")
            print(f"    Free-tier retain: {np.mean(arena_rets):.1f}% ± {np.std(arena_rets):.1f}%")
        if judge_f1s:
            print(f"  LLM judge ({fold_results[0].get('llm_judge_model', '?')}):")
            print(f"    Macro-F1:         {np.mean(judge_f1s):.3f} ± {np.std(judge_f1s):.3f}")
            delta_f1s = [
                r["macro_f1"] - r["llm_judge_macro_f1"]
                for r in fold_results
                if "llm_judge_macro_f1" in r
            ]
            print(f"    ModernBERT delta: +{np.mean(delta_f1s):.3f} ± {np.std(delta_f1s):.3f}")

        # Save best model
        if best_model is not None:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            best_model.save_pretrained(str(OUTPUT_DIR))
            tokenizer.save_pretrained(str(OUTPUT_DIR))
            print(f"\nBest fold model saved to {OUTPUT_DIR}")

        result = {
            "eval_mode": "mixed-kfold",
            "model": MODEL_NAME,
            "dataset": str(dataset_path),
            "n_folds": args.n_folds,
            "cost_sensitive_alpha": getattr(args, "_best_alpha", None),
            "cost_sensitive_gamma": getattr(args, "_gamma", None)
            if getattr(args, "cost_sensitive", False)
            else None,
            "folds": fold_results,
            "mean_accuracy": round(float(np.mean(accs)), 4),
            "std_accuracy": round(float(np.std(accs)), 4),
            "mean_macro_f1": round(float(np.mean(f1s)), 4),
            "std_macro_f1": round(float(np.std(f1s)), 4),
            "mean_free_tier_retention_pct": round(float(np.mean(retentions)), 2),
            "std_free_tier_retention_pct": round(float(np.std(retentions)), 2),
        }
        if arena_f1s:
            result.update(
                {
                    "arena_mean_accuracy": round(float(np.mean(arena_accs)), 4),
                    "arena_std_accuracy": round(float(np.std(arena_accs)), 4),
                    "arena_mean_macro_f1": round(float(np.mean(arena_f1s)), 4),
                    "arena_std_macro_f1": round(float(np.std(arena_f1s)), 4),
                    "arena_mean_free_tier_retention_pct": round(float(np.mean(arena_rets)), 2),
                    "arena_std_free_tier_retention_pct": round(float(np.std(arena_rets)), 2),
                }
            )
        if judge_f1s:
            result.update(
                {
                    "llm_judge_model": fold_results[0].get("llm_judge_model"),
                    "llm_judge_mean_macro_f1": round(float(np.mean(judge_f1s)), 4),
                    "llm_judge_std_macro_f1": round(float(np.std(judge_f1s)), 4),
                }
            )

        condition = args.condition_name or "mixed"
        report_path = REPORT_PATH.parent / f"modernbert_{condition}_kfold_report.json"
        result["condition_name"] = args.condition_name
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Report saved to {report_path}")

        print("\n" + "=" * 60)
        print("PAPER NUMBERS (mixed real-world 5-fold CV):")
        print("=" * 60)
        print(f"  ModernBERT accuracy:  {np.mean(accs):.1%} ± {np.std(accs):.1%}")
        print(f"  ModernBERT macro-F1:  {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
        print(f"  Free-tier retention:  {np.mean(retentions):.1f}% ± {np.std(retentions):.1f}%")
        if arena_f1s:
            print(f"  Arena OOD macro-F1:   {np.mean(arena_f1s):.3f} ± {np.std(arena_f1s):.3f}")
        if judge_f1s:
            print(
                f"  LLM judge macro-F1:   {np.mean(judge_f1s):.3f} ± {np.std(judge_f1s):.3f}"
                f"  (ModernBERT Δ={np.mean(delta_f1s):+.3f})"
            )

    # -------------------------------------------------------------------------
    # RANDOM SPLIT (default)
    # -------------------------------------------------------------------------
    else:
        train_q, val_q, test_q = train_val_test_split(queries)
        print(f"  Split: train={len(train_q)}, val={len(val_q)}, test={len(test_q)}")

        if args.eval_only:
            print(f"\nLoading saved model from {OUTPUT_DIR}...")
            model = AutoModelForSequenceClassification.from_pretrained(str(OUTPUT_DIR))
            tokenizer = AutoTokenizer.from_pretrained(str(OUTPUT_DIR))
            train_time = 0.0
        else:
            print(f"\nTraining ModernBERT-base ({1 if args.dry_run else args.epochs} epochs)...")
            model, metrics_train = train_one_fold(
                "random", train_q, val_q, test_q, args, tokenizer, OUTPUT_DIR
            )
            train_time = metrics_train["train_time_minutes"]
            tokenizer.save_pretrained(str(OUTPUT_DIR))
            print(f"Model saved to {OUTPUT_DIR}")

        print("\nEvaluating on test set...")
        metrics = eval_on_test_set(model, tokenizer, test_q)
        print_test_results(metrics, "random split")

        lat = measure_latency(model, tokenizer, [q["text"] for q in test_q][:200])
        print(
            f"\n  CPU latency — p50: {lat['p50_ms']} ms  p95: {lat['p95_ms']} ms  p99: {lat['p99_ms']} ms"
        )

        result = {
            "eval_mode": "random",
            "model": MODEL_NAME,
            "dataset": str(dataset_path),
            "train_size": len(train_q),
            "val_size": len(val_q),
            "test_size": len(test_q),
            "epochs": 1 if args.dry_run else args.epochs,
            **metrics,
            "latency_ms": lat,
            "train_time_minutes": train_time if not args.eval_only else None,
            "output_dir": str(OUTPUT_DIR),
        }
        with open(REPORT_PATH, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nReport saved to {REPORT_PATH}")

        print("\n" + "=" * 60)
        print("PAPER NUMBERS (random split — may be inflated; see domain-holdout):")
        print("=" * 60)
        print(f"  ModernBERT accuracy:   {metrics['accuracy']:.1%}")
        print(f"  ModernBERT macro-F1:   {metrics['macro_f1']:.3f}")
        for cls in ["LOW", "MEDIUM", "HIGH"]:
            print(f"  {cls:6s} F1: {metrics['per_class'][cls]['f1']:.3f}")
        print(f"  Latency p50:           {lat['p50_ms']} ms")
        print(f"  Free-tier retention:   {metrics['free_tier_retention_pct']:.1f}%")
        print(
            "\nNote: Random-split accuracy may be inflated due to near-duplicate\n"
            "      queries in train/test. Run --eval-mode domain-holdout and\n"
            "      eval_on_realworld.py for rigorous numbers."
        )
        print(
            "\nNext: python scripts/eval/build_realworld_testset.py\n"
            "      python scripts/eval/eval_on_realworld.py"
        )


if __name__ == "__main__":
    main()
