#!/usr/bin/env python3
"""
Upload dataset and model to HuggingFace.

Dataset: anasnassar/llm-query-complexity-benchmark
  - train.jsonl  (4,800 queries, 160/domain/class across 10 domains)
  - test.jsonl   (1,200 queries, 40/domain/class)
  - README.md    (dataset card)

Model: anasnassar/llm-query-complexity-classifier
  - models/modernbert_balanced/  (full model + tokenizer)
  - README.md                    (model card)

Usage:
  python scripts/eval/upload_to_huggingface.py            # upload both
  python scripts/eval/upload_to_huggingface.py --dataset  # dataset only
  python scripts/eval/upload_to_huggingface.py --model    # model only

Requires: huggingface-cli login  (run once)
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

DATASET_REPO = "anasnassar/llm-query-complexity-benchmark"
MODEL_REPO = "anasnassar/llm-query-complexity-classifier"

api = HfApi()


def upload_dataset():
    print(f"\nUploading dataset → {DATASET_REPO}")

    files = {
        "scripts/eval/hf_dataset_card.md": "README.md",
        "scripts/eval/balanced_train.jsonl": "train.jsonl",
        "scripts/eval/balanced_test.jsonl": "test.jsonl",
    }

    for local, remote in files.items():
        path = Path(local)
        if not path.exists():
            print(f"  [SKIP] {local} not found")
            continue
        print(f"  {local} → {remote}")
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=remote,
            repo_id=DATASET_REPO,
            repo_type="dataset",
            commit_message=f"Upload {remote} (balanced 6,000-query 10-domain dataset)",
        )

    print(f"  Done: https://huggingface.co/datasets/{DATASET_REPO}")


def upload_model():
    print(f"\nUploading model → {MODEL_REPO}")

    model_dir = Path("scripts/eval/models/modernbert_balanced")
    if not model_dir.exists():
        print(f"  [SKIP] Model not found at {model_dir}")
        print("  Run train_balanced_classifier.py first, then sync results from Lakeshore.")
        return

    card = Path("scripts/eval/hf_model_card.md")
    if card.exists():
        print("  Uploading model card...")
        api.upload_file(
            path_or_fileobj=str(card),
            path_in_repo="README.md",
            repo_id=MODEL_REPO,
            repo_type="model",
            commit_message="Update model card",
        )

    print(f"  Uploading model weights from {model_dir} (skipping checkpoints)...")
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=MODEL_REPO,
        repo_type="model",
        commit_message="Upload ModernBERT-base classifier (balanced dataset, PEARC '26)",
        ignore_patterns=["checkpoint-*/"],
    )

    print(f"  Done: https://huggingface.co/models/{MODEL_REPO}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="store_true", help="Upload dataset only")
    parser.add_argument("--model", action="store_true", help="Upload model only")
    args = parser.parse_args()

    do_all = not args.dataset and not args.model
    if args.dataset or do_all:
        upload_dataset()
    if args.model or do_all:
        upload_model()

    print("\nDone.")


if __name__ == "__main__":
    main()
