"""
Compute Cohen's kappa between author labels and Claude's labels.

Usage:
  python scripts/eval/compute_validation_kappa.py

Input files:
  - scripts/eval/validation_sample_labeled.csv  (your labels, YOUR_LABEL column)
  - scripts/eval/validation_answers.json         (Claude's labels, auto-generated)

Output:
  - Prints kappa, per-class agreement, confusion matrix
  - Writes scripts/eval/validation_kappa_report.json
"""

import csv
import json
from collections import Counter

LABELED_CSV = "scripts/eval/validation_sample_labeled.csv"
ANSWERS_JSON = "scripts/eval/validation_answers.json"
REPORT_JSON = "scripts/eval/validation_kappa_report.json"

VALID_LABELS = {"LOW", "MEDIUM", "HIGH"}


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Compute Cohen's kappa for two label sequences."""
    assert len(labels_a) == len(labels_b)
    n = len(labels_a)
    classes = sorted(VALID_LABELS)

    # Observed agreement
    p_o = sum(a == b for a, b in zip(labels_a, labels_b, strict=False)) / n

    # Expected agreement (product of marginals)
    count_a = Counter(labels_a)
    count_b = Counter(labels_b)
    p_e = sum((count_a[c] / n) * (count_b[c] / n) for c in classes)

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def main():
    # Load Claude's hidden labels
    with open(ANSWERS_JSON) as f:
        claude_labels: dict[str, str] = json.load(f)

    # Load your labels
    your_labels: dict[str, str] = {}
    skipped = []
    with open(LABELED_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = str(row["id"])
            label = row["YOUR_LABEL"].strip().upper()
            if label not in VALID_LABELS:
                skipped.append((qid, row["YOUR_LABEL"]))
                continue
            your_labels[qid] = label

    if skipped:
        print(f"WARNING: {len(skipped)} rows skipped (invalid label):")
        for qid, raw in skipped[:5]:
            print(f"  id={qid}: '{raw}'")

    # Align on shared IDs
    shared_ids = sorted(set(your_labels) & set(claude_labels))
    n = len(shared_ids)
    print(f"Comparing {n} labeled queries")

    you = [your_labels[i] for i in shared_ids]
    claude = [claude_labels[i] for i in shared_ids]

    # Overall kappa
    kappa = cohen_kappa(you, claude)
    raw_agreement = sum(a == b for a, b in zip(you, claude, strict=False)) / n

    print(
        f"\nOverall agreement: {raw_agreement:.1%}  ({sum(a==b for a,b in zip(you,claude, strict=False))}/{n})"
    )
    print(f"Cohen's kappa:     {kappa:.3f}")

    interpretation = (
        "almost perfect"
        if kappa >= 0.80
        else "substantial"
        if kappa >= 0.60
        else "moderate"
        if kappa >= 0.40
        else "fair"
        if kappa >= 0.20
        else "slight"
    )
    print(f"Interpretation:    {interpretation} agreement (Landis & Koch 1977)")

    # Per-class breakdown
    print("\nPer-class agreement:")
    per_class = {}
    for cls in sorted(VALID_LABELS):
        indices = [i for i, c in enumerate(claude) if c == cls]
        if not indices:
            continue
        agree = sum(you[i] == claude[i] for i in indices)
        pct = agree / len(indices)
        per_class[cls] = {"n": len(indices), "agree": agree, "pct": round(pct, 4)}
        print(f"  {cls:6s}: {pct:.1%}  ({agree}/{len(indices)})")

    # Confusion matrix (rows=you, cols=claude)
    classes = ["LOW", "MEDIUM", "HIGH"]
    print("\nConfusion matrix (rows=your label, cols=Claude label):")
    print(f"{'':8s} " + "  ".join(f"{c:6s}" for c in classes))
    confusion = {}
    for r in classes:
        row_counts = []
        for c in classes:
            count = sum(1 for y, cl in zip(you, claude, strict=False) if y == r and cl == c)
            row_counts.append(count)
            confusion[f"{r}->{c}"] = count
        print(f"  {r:6s} " + "  ".join(f"{x:6d}" for x in row_counts))

    # Save report
    report = {
        "n": n,
        "raw_agreement": round(raw_agreement, 4),
        "cohen_kappa": round(kappa, 4),
        "interpretation": interpretation,
        "per_class": per_class,
        "confusion": confusion,
        "skipped": len(skipped),
    }
    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {REPORT_JSON}")
    print()
    print("Use this sentence in the paper:")
    print(f'  "The authors independently annotated a stratified sample of {n} queries')
    print("   (14 per domain-complexity cell, blind to Claude's labels) using the")
    print(f"   reasoning-depth rubric, yielding Cohen's κ = {kappa:.3f}")
    print(f"   ({interpretation} agreement) against Claude's labels.\"")


if __name__ == "__main__":
    main()
