"""
Sample a stratified subset of queries for author validation.

Outputs two files:
  - validation_sample.csv   : the queries to label (no Claude label shown)
  - validation_answers.json : the hidden Claude labels (to compare after)

Usage:
  python scripts/eval/sample_for_validation.py

After labeling:
  - Open validation_sample.csv, fill in the YOUR_LABEL column (LOW/MEDIUM/HIGH)
  - Save as validation_sample_labeled.csv
  - Run: python scripts/eval/compute_validation_kappa.py
"""

import csv
import json
import random

DATASET_PATH = "scripts/eval/benchmark_dataset_384.json"
OUTPUT_CSV = "scripts/eval/validation_sample.csv"
OUTPUT_ANSWERS = "scripts/eval/validation_answers.json"

N_PER_CELL = 14  # 14 × 18 cells = 252 queries
RANDOM_SEED = 42  # reproducible sample


def main():
    with open(DATASET_PATH) as f:
        data = json.load(f)
    queries = data["queries"]

    # Group by (domain, complexity) cell
    cells: dict[tuple, list] = {}
    for q in queries:
        key = (q["domain"], q["ground_truth"])
        cells.setdefault(key, []).append(q)

    rng = random.Random(RANDOM_SEED)
    sample = []
    for _key, items in sorted(cells.items()):
        chosen = rng.sample(items, N_PER_CELL)
        sample.extend(chosen)

    rng.shuffle(sample)  # shuffle so domain/class aren't grouped visually

    # Write CSV for annotation (Claude's label hidden)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "domain", "query", "YOUR_LABEL"])
        for q in sample:
            writer.writerow([q["id"], q["domain"], q["text"], ""])

    # Write hidden answers
    answers = {str(q["id"]): q["ground_truth"] for q in sample}
    with open(OUTPUT_ANSWERS, "w") as f:
        json.dump(answers, f, indent=2)

    print(f"Wrote {len(sample)} queries to {OUTPUT_CSV}")
    print(f"Wrote hidden labels to {OUTPUT_ANSWERS}")
    print()
    print("Instructions:")
    print("  1. Open validation_sample.csv")
    print("  2. For each query, write LOW, MEDIUM, or HIGH in the YOUR_LABEL column")
    print("     Rubric:")
    print("       LOW    — answer is a single retrievable fact or trivial computation;")
    print("                no reasoning chain required")
    print("       MEDIUM — requires applying a known procedure or assembling 2-4 concepts;")
    print("                the reasoning path is established, not invented")
    print("       HIGH   — requires constructing a novel reasoning path, formal derivation,")
    print("                or expert judgment; reasonable experts could approach it differently")
    print("  3. Save as validation_sample_labeled.csv")
    print("  4. Run: python scripts/eval/compute_validation_kappa.py")


if __name__ == "__main__":
    main()
