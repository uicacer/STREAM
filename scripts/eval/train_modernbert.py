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
    python scripts/eval/train_modernbert.py --dry-run

Eval modes:
    random           Standard 70/15/15 random split (default). Fast baseline.
    domain-holdout   6-fold CV: each fold holds out one domain entirely.
                     Tests cross-domain generalization; ~6x training time.
    similarity-split Semantic similarity-aware split: near-duplicate queries
                     (cosine sim > 0.90) are kept on the same side of the
                     train/test boundary. Requires sentence-transformers.

Output:
    scripts/eval/models/modernbert/          fine-tuned model + tokenizer
    scripts/eval/results/modernbert_training_report.json         (random)
    scripts/eval/results/modernbert_domain_holdout_report.json   (domain-holdout)
    scripts/eval/results/modernbert_similarity_split_report.json (similarity-split)
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
        for t, p in zip(label_names, pred_names, strict=True)
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
        save_strategy="best",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
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

    training_args = make_training_args(output_dir, args.epochs, args.dry_run)
    trainer = Trainer(
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
        choices=["random", "domain-holdout", "similarity-split"],
        default="random",
        help=(
            "random: 70/15/15 split (default). "
            "domain-holdout: 6-fold CV holding out one domain per fold. "
            "similarity-split: semantic deduplication before splitting."
        ),
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
    if args.eval_mode == "domain-holdout":
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
