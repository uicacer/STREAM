#!/usr/bin/env python3
"""
Full quality audit of the balanced dataset before pushing to HuggingFace.

Checks every query across all 10 domains for:
  1. Field completeness — all required fields present and non-empty
  2. Field values — domain/source/ground_truth within allowed sets
  3. Length — text 10–400 chars
  4. Language — ≥80% ASCII (English)
  5. Alphabetic content — ≥30% alpha ratio
  6. Word count — ≥3 meaningful words
  7. Context leakage — "which of the following", "the following", "the passage", etc.
  8. Error messages — stack traces, exception strings, version numbers
  9. Fill-in-the-blank — ___ patterns
 10. Answer choices — (A) (B) (C) (D) leaked into question text
 11. Duplicates — exact text matches within and across splits
 12. Cross-contamination — same text appears in both train and test
 13. Balance check — per domain per class counts (flag if not 200/200/200)
 14. Random sample review — print 3 random queries per domain for human spot-check

Usage:
  python scripts/eval/audit_dataset.py
  python scripts/eval/audit_dataset.py --train scripts/eval/balanced_training_dataset.json \
                                        --test  scripts/eval/balanced_test_dataset.json
  python scripts/eval/audit_dataset.py --sample 5   # more spot-check samples per domain
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from random import Random

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRAIN_DEFAULT = Path("scripts/eval/balanced_training_dataset.json")
TEST_DEFAULT = Path("scripts/eval/balanced_test_dataset.json")

REQUIRED_FIELDS = {"id", "text", "domain", "source", "subject", "ground_truth"}
VALID_DOMAINS = {
    "hpc",
    "mathematics",
    "statistics_ml",
    "physics_chemistry",
    "engineering",
    "life_sciences",
    "cs_software",
    "philosophy_ethics",
    "social_sciences",
    "history_culture",
}
VALID_SOURCES = {"stackexchange", "mmlu", "mmlu_pro", "pubmedqa", "synthetic_high"}
VALID_CLASSES = {"LOW", "MEDIUM", "HIGH"}
DOMAIN_TARGET = 200  # expected per domain per class (full run)
MIN_TEXT_LEN = 10
MAX_TEXT_LEN = 400

_CONTEXT_PATTERNS = [
    r"\bwhich of the following\b",
    r"\bthe following\b",
    r"\baccording to (the )?(passage|text|article|author|table|figure)\b",
    r"\bthe passage\b",
    r"\bthe text above\b",
    r"\bthe (table|figure|graph|diagram|chart) (above|below)\b",
    r"\bthe scenario (above|below|described)\b",
    r"_{3,}",
    r"\(A\)|\(B\)|\(C\)|\(D\)",
]

_ERROR_PATTERNS = [
    r"^error:",
    r"^exception:",
    r"^traceback",
    r"^at line \d",
    r"^\w+error\b",
    r"^compile error",
    r"^runtime error",
    r"^syntax error",
    r"^error while",
    r"^error when",
    r"^error trying",
    r"NullPointerException",
    r"segmentation fault",
    r"^\d+\.\d+\.\d+\b",
    r"exception\s+\w+\.\w+\.\w+",
    r"com\.mysql\.",
    r"java\.lang\.",
    r"java\.io\.",
    r"no such file or directory",
    r"permission denied",
    r"command not found",
    r"cannot find symbol",
    r"ORA-\d{5}",
]


def _matches_any(text: str, patterns: list) -> bool:
    return any(re.search(p, text) for p in patterns)


# ---------------------------------------------------------------------------
# Per-query checks — returns list of issue strings (empty = clean)
# ---------------------------------------------------------------------------


def audit_query(q: dict, idx: int, split: str) -> list[str]:
    issues = []
    pfx = f"[{split}#{idx}]"

    # 1. Required fields
    for f in REQUIRED_FIELDS:
        if f not in q or q[f] is None or str(q[f]).strip() == "":
            issues.append(f"{pfx} missing/empty field: {f!r}")

    if issues:
        return issues  # can't continue without text

    text = str(q["text"]).strip()
    tl = text.lower()

    # 2. Valid enum values
    if q.get("domain") not in VALID_DOMAINS:
        issues.append(f"{pfx} unknown domain: {q['domain']!r}")
    if q.get("source") not in VALID_SOURCES:
        issues.append(f"{pfx} unknown source: {q['source']!r}")
    if q.get("ground_truth") not in VALID_CLASSES:
        issues.append(f"{pfx} unknown class: {q['ground_truth']!r}")

    # 3. Text length
    if not (MIN_TEXT_LEN <= len(text) <= MAX_TEXT_LEN):
        issues.append(f"{pfx} bad length {len(text)}: {text[:60]!r}")

    # 4. ASCII ratio
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
    if ascii_ratio < 0.80:
        issues.append(f"{pfx} low ASCII ratio {ascii_ratio:.2f}: {text[:60]!r}")

    # 5. Alpha ratio
    alpha_ratio = len(re.findall(r"[a-zA-Z]", text)) / max(len(text), 1)
    if alpha_ratio < 0.30:
        issues.append(f"{pfx} low alpha ratio {alpha_ratio:.2f}: {text[:60]!r}")

    # 6. Word count
    words = re.findall(r"[a-zA-Z]{2,}", text)
    if len(words) < 3:
        issues.append(f"{pfx} too few words ({len(words)}): {text[:60]!r}")

    # 7. Context leakage — apply stricter "the following" check only to MMLU sources
    # (SE titles like "Show the following limit: ..." are self-contained)
    source = q.get("source", "")
    patterns_to_check = _CONTEXT_PATTERNS
    if source == "stackexchange":
        # For SE, skip the broad "the following" pattern — formula/code IS in the title
        patterns_to_check = [p for p in _CONTEXT_PATTERNS if p not in (r"\bthe following\b",)]
    if _matches_any(tl, patterns_to_check):
        matched = [p for p in patterns_to_check if re.search(p, tl)]
        issues.append(f"{pfx} context leak pattern {matched[0]!r}: {text[:80]!r}")

    # 8. Error messages
    if _matches_any(tl, _ERROR_PATTERNS):
        matched = [p for p in _ERROR_PATTERNS if re.search(p, tl)]
        issues.append(f"{pfx} error message pattern {matched[0]!r}: {text[:80]!r}")

    return issues


# ---------------------------------------------------------------------------
# Load + audit a split
# ---------------------------------------------------------------------------


def load_split(path: Path, split_name: str) -> list[dict]:
    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "queries" in data:
        queries = data["queries"]
    elif isinstance(data, list):
        queries = data
    else:
        print(f"[ERROR] Unexpected JSON structure in {path}")
        sys.exit(1)
    print(f"Loaded {split_name}: {len(queries):,} queries")
    return queries


def audit_split(queries: list[dict], split_name: str) -> list[str]:
    all_issues = []
    for i, q in enumerate(queries):
        issues = audit_query(q, i + 1, split_name)
        all_issues.extend(issues)
    return all_issues


# ---------------------------------------------------------------------------
# Balance check
# ---------------------------------------------------------------------------


def check_balance(train: list[dict], test: list[dict], target: int) -> list[str]:
    issues = []
    for split_name, queries in [("TRAIN", train), ("TEST", test)]:
        per_domain_class: dict[str, Counter] = defaultdict(Counter)
        for q in queries:
            per_domain_class[q.get("domain", "?")][q.get("ground_truth", "?")] += 1

        expected = round(target * (0.8 if split_name == "TRAIN" else 0.2))
        for domain in VALID_DOMAINS:
            counts = per_domain_class.get(domain, Counter())
            for cls in VALID_CLASSES:
                n = counts.get(cls, 0)
                if n < expected * 0.9:  # allow 10% tolerance
                    issues.append(
                        f"[BALANCE {split_name}] {domain}/{cls}: {n} (expected ~{expected})"
                    )
    return issues


# ---------------------------------------------------------------------------
# Duplicate + contamination check
# ---------------------------------------------------------------------------


def check_duplicates(train: list[dict], test: list[dict]) -> list[str]:
    issues = []

    # Within-train duplicates
    train_texts = [q["text"] for q in train]
    train_counts = Counter(train_texts)
    for text, n in train_counts.items():
        if n > 1:
            issues.append(f"[DUP TRAIN] {n}x: {text[:70]!r}")

    # Within-test duplicates
    test_texts = [q["text"] for q in test]
    test_counts = Counter(test_texts)
    for text, n in test_counts.items():
        if n > 1:
            issues.append(f"[DUP TEST] {n}x: {text[:70]!r}")

    # Cross-contamination
    train_set = set(train_texts)
    test_set = set(test_texts)
    overlap = train_set & test_set
    for text in sorted(overlap):
        issues.append(f"[CONTAMINATION] in both splits: {text[:70]!r}")

    return issues


# ---------------------------------------------------------------------------
# Spot-check sample
# ---------------------------------------------------------------------------


def print_sample(queries: list[dict], n_per_domain: int, seed: int = 42) -> None:
    rng = Random(seed)
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for q in queries:
        by_domain[q.get("domain", "?")].append(q)

    print(f"\n{'='*70}")
    print(f"SPOT-CHECK SAMPLE ({n_per_domain}/domain, train split)")
    print(f"{'='*70}")
    for domain in sorted(VALID_DOMAINS):
        pool = by_domain.get(domain, [])
        sample = rng.sample(pool, min(n_per_domain, len(pool)))
        print(f"\n--- {domain.upper()} ({len(pool)} total) ---")
        for q in sample:
            label = q.get("ground_truth", "?")
            src = q.get("source", "?")
            print(f"  [{label:6s}][{src:12s}] {q['text'][:100]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(TRAIN_DEFAULT))
    parser.add_argument("--test", default=str(TEST_DEFAULT))
    parser.add_argument(
        "--target", type=int, default=DOMAIN_TARGET, help="Expected queries per domain per class"
    )
    parser.add_argument(
        "--sample", type=int, default=3, help="Spot-check samples per domain to print"
    )
    args = parser.parse_args()

    train = load_split(Path(args.train), "TRAIN")
    test = load_split(Path(args.test), "TEST")

    print(f"\n{'='*70}")
    print("QUALITY AUDIT")
    print(f"{'='*70}\n")

    # --- Per-query checks ---
    print("Running per-query checks...")
    train_issues = audit_split(train, "TRAIN")
    test_issues = audit_split(test, "TEST")
    query_issues = train_issues + test_issues

    # --- Duplicate + contamination ---
    print("Checking for duplicates and train/test contamination...")
    dup_issues = check_duplicates(train, test)

    # --- Balance ---
    print("Checking class/domain balance...")
    bal_issues = check_balance(train, test, args.target)

    # --- Summary ---
    all_issues = query_issues + dup_issues + bal_issues
    total = len(all_issues)

    print(f"\n{'='*70}")
    print(f"RESULTS: {total} issue(s) found")
    print(f"{'='*70}")
    if query_issues:
        print(f"\n[PER-QUERY] {len(query_issues)} issues:")
        for iss in query_issues[:50]:
            print(f"  {iss}")
        if len(query_issues) > 50:
            print(f"  ... ({len(query_issues) - 50} more)")
    if dup_issues:
        print(f"\n[DUPLICATES/CONTAMINATION] {len(dup_issues)} issues:")
        for iss in dup_issues[:20]:
            print(f"  {iss}")
    if bal_issues:
        print(f"\n[BALANCE] {len(bal_issues)} issues:")
        for iss in bal_issues:
            print(f"  {iss}")

    if total == 0:
        print("\n✓ All checks passed — dataset is ready for HuggingFace upload.")
    else:
        print(f"\n⚠  Fix the {total} issue(s) above before uploading.")

    # --- Dataset statistics ---
    print(f"\n{'='*70}")
    print("DATASET STATISTICS")
    print(f"{'='*70}")
    for split_name, queries in [("TRAIN", train), ("TEST", test)]:
        class_dist = Counter(q.get("ground_truth") for q in queries)
        domain_dist = Counter(q.get("domain") for q in queries)
        source_dist = Counter(q.get("source") for q in queries)
        print(f"\n{split_name} ({len(queries):,} queries):")
        print(f"  Classes: {dict(sorted(class_dist.items()))}")
        print(f"  Domains: {dict(sorted(domain_dist.items()))}")
        print(f"  Sources: {dict(sorted(source_dist.items()))}")

    # Per-domain per-class table
    print("\nPer-domain per-class counts (TRAIN):")
    by_dc: dict[str, Counter] = defaultdict(Counter)
    for q in train:
        by_dc[q.get("domain", "?")][q.get("ground_truth", "?")] += 1
    print(f"  {'Domain':22s}  {'LOW':>5}  {'MED':>5}  {'HIGH':>5}  {'Total':>6}")
    print(f"  {'-'*22}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*6}")
    for domain in sorted(VALID_DOMAINS):
        c = by_dc.get(domain, Counter())
        low, med, hi = c.get("LOW", 0), c.get("MEDIUM", 0), c.get("HIGH", 0)
        print(f"  {domain:22s}  {low:5}  {med:5}  {hi:5}  {low+med+hi:6}")

    # --- Spot-check ---
    print_sample(train, args.sample)

    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
