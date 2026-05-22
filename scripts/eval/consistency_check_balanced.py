"""
Consistency check for the balanced 6,000-query dataset.

Samples 5% stratified (100 per class = 300 total), re-labels with GPT-4o-mini,
computes Cohen's kappa against the Claude-generated ground truth labels.

Usage:
  python scripts/eval/consistency_check_balanced.py [--dry-run]

Output:
  scripts/eval/consistency_report_balanced.json
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

TRAIN_FILE = "scripts/eval/balanced_train.jsonl"
TEST_FILE = "scripts/eval/balanced_test.jsonl"
OUTPUT_REPORT = "scripts/eval/consistency_report_balanced.json"
CONSISTENCY_MODEL = "gpt-4o-mini"
N_PER_CLASS = 100  # 300 total = 5% of 6000
RANDOM_SEED = 42
BATCH_SIZE = 20

COMPLEXITY_RUBRIC = """You are a query complexity classifier. Classify each query by REASONING DEPTH:

LOW: Single retrievable fact or definition. No reasoning chain needed.
  Examples: "What year did WWII end?", "What is photosynthesis?"

MEDIUM: Requires applying a known procedure or assembling 2-4 concepts.
  Examples: "How does TCP/IP handle packet loss?", "Compare RNA and DNA replication."

HIGH: Requires constructing a novel reasoning path, formal derivation, or expert judgment.
  Examples: "Derive the time complexity of Dijkstra's algorithm from first principles.",
            "What are the ethical implications of CRISPR germline editing?"

Respond with ONLY a JSON array of labels, one per query, in order.
Example: ["LOW", "HIGH", "MEDIUM"]
"""


def sample_stratified(queries: list[dict], n_per_class: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_class: dict[str, list] = {}
    for q in queries:
        by_class.setdefault(q["ground_truth"], []).append(q)
    sample = []
    for cls in ["LOW", "MEDIUM", "HIGH"]:
        pool = by_class.get(cls, [])
        n = min(n_per_class, len(pool))
        sample.extend(rng.sample(pool, n))
    rng.shuffle(sample)
    return sample


def label_batch(texts: list[str], client, dry_run: bool) -> list[str]:
    if dry_run:
        return [random.choice(["LOW", "MEDIUM", "HIGH"]) for _ in texts]
    prompt = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=CONSISTENCY_MODEL,
                max_tokens=300,
                temperature=0,
                messages=[
                    {"role": "system", "content": COMPLEXITY_RUBRIC},
                    {"role": "user", "content": f"Queries:\n{prompt}"},
                ],
            )
            raw = resp.choices[0].message.content.strip()
            # Strip markdown fences
            import re

            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            labels = json.loads(raw)
            if isinstance(labels, list) and len(labels) == len(texts):
                return [lb.upper() for lb in labels]
        except Exception as e:
            print(f"    [attempt {attempt+1}] error: {e}")
    return ["MEDIUM"] * len(texts)  # fallback


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    assert len(labels_a) == len(labels_b)
    n = len(labels_a)
    classes = ["LOW", "MEDIUM", "HIGH"]
    p_o = sum(a == b for a, b in zip(labels_a, labels_b, strict=False)) / n
    count_a = Counter(labels_a)
    count_b = Counter(labels_b)
    p_e = sum((count_a[c] / n) * (count_b[c] / n) for c in classes)
    return (p_o - p_e) / (1 - p_e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Use random labels (no API calls)")
    args = parser.parse_args()

    # Load dataset
    queries = []
    for f in [TRAIN_FILE, TEST_FILE]:
        with open(f) as fh:
            queries.extend(json.loads(line) for line in fh)
    print(f"Loaded {len(queries)} queries")

    # Stratified sample
    sample = sample_stratified(queries, N_PER_CLASS, RANDOM_SEED)
    print(f"Sampled {len(sample)} queries ({N_PER_CLASS} per class)")

    # Re-label with GPT-4o-mini
    client = None
    if not args.dry_run:
        from openai import OpenAI

        client = OpenAI()

    gpt_labels = []
    n_batches = (len(sample) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(n_batches):
        batch = sample[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        texts = [q["text"] for q in batch]
        print(f"  Batch {i+1}/{n_batches} ({len(texts)} queries)...", end=" ", flush=True)
        labels = label_batch(texts, client, args.dry_run)
        gpt_labels.extend(labels)
        print("done")

    # Compute kappa
    claude_labels = [q["ground_truth"] for q in sample]
    valid = [
        (c, g)
        for c, g in zip(claude_labels, gpt_labels, strict=False)
        if g in ["LOW", "MEDIUM", "HIGH"]
    ]
    n_invalid = len(sample) - len(valid)
    claude_valid = [c for c, g in valid]
    gpt_valid = [g for c, g in valid]

    kappa = cohen_kappa(claude_valid, gpt_valid)
    agreement = sum(c == g for c, g in valid) / len(valid)

    per_class = {}
    for cls in ["LOW", "MEDIUM", "HIGH"]:
        cls_pairs = [(c, g) for c, g in valid if c == cls]
        if cls_pairs:
            per_class[cls] = {
                "n": len(cls_pairs),
                "agreement": sum(c == g for c, g in cls_pairs) / len(cls_pairs),
            }

    # Interpret kappa
    if kappa >= 0.81:
        interpretation = "almost perfect"
    elif kappa >= 0.61:
        interpretation = "substantial"
    elif kappa >= 0.41:
        interpretation = "moderate"
    else:
        interpretation = "fair"

    report = {
        "primary_model": "claude-sonnet-4-6",
        "consistency_model": CONSISTENCY_MODEL,
        "dataset": "balanced_train.jsonl + balanced_test.jsonl (6,000 queries)",
        "n_sampled": len(sample),
        "n_invalid": n_invalid,
        "agreement": round(agreement, 4),
        "cohens_kappa": round(kappa, 4),
        "interpretation": interpretation,
        "per_class": per_class,
    }

    print("\n=== Consistency Report ===")
    print(f"  n={len(valid)}, invalid={n_invalid}")
    print(f"  Agreement: {agreement:.1%}")
    print(f"  Cohen's kappa: {kappa:.4f} ({interpretation})")
    for cls, v in per_class.items():
        print(f"  {cls}: n={v['n']}, agreement={v['agreement']:.1%}")

    Path(OUTPUT_REPORT).write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
