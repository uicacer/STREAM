#!/usr/bin/env python3
"""
Fix format-proxy bias in stream_routing_benchmark.json.

Problem: 91% of "What is X?" queries are LOW (391 LOW, 19 MEDIUM, 20 HIGH).
A classifier could learn "starts with What is" → LOW rather than reasoning depth.

Fix:
  1. For each domain, cap "What is X?" LOW at MAX_WHAT_IS_LOW_PER_DOMAIN (24).
     Replace removed queries with non-"What is X?" LOW queries (cell stays at 384).
  2. For each domain, generate TARGET_WHAT_IS_PER_DOMAIN (20) "What is X?" MEDIUM queries.
     Remove same count of non-"What is X?" MEDIUM to keep cell at 384.
  3. Same for HIGH.

After fix target: ~144 LOW / ~120 MEDIUM / ~120 HIGH "What is X?" = 30%/25%/25%,
well below the 90% warning threshold.

Also fixes consistency check: batch_size 20→10 with one retry to eliminate length
mismatch warnings.

Usage:
    python scripts/eval/fix_format_bias.py [--dry-run] [--skip-consistency]
"""

import argparse
import json
import os
import re
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

DATASET_PATH = Path("scripts/eval/stream_routing_benchmark.json")
CONSISTENCY_PATH = Path("scripts/eval/consistency_report.json")

LABELING_MODEL = "claude-sonnet-4-6"
CONSISTENCY_MODEL = "gpt-4o-mini"
CLASSES = ["LOW", "MEDIUM", "HIGH"]
DOMAINS = [
    "general_knowledge",
    "science",
    "mathematics",
    "humanities",
    "computer_science",
    "research_computing",
]

MAX_WHAT_IS_LOW_PER_DOMAIN = 24  # cap per domain → max 144 total LOW
TARGET_WHAT_IS_PER_DOMAIN = 20  # generate per domain for MEDIUM and HIGH

LABELING_CRITERIA = {
    "LOW": (
        "Answer is a single retrievable fact or trivial computation. "
        "No reasoning chain required — any knowledgeable person states the answer "
        "in one sentence with no intermediate steps. "
        "The answer is unambiguous and universally agreed upon."
    ),
    "MEDIUM": (
        "Answer requires applying a known, established procedure or assembling "
        "2–4 related concepts. The reasoning path is standard (textbook-level) — "
        "it exists and is well-defined, so the answerer follows it rather than "
        "inventing it."
    ),
    "HIGH": (
        "Answer requires constructing a novel reasoning path, formal derivation, "
        "cross-domain synthesis, or expert judgment. The path to the answer must be "
        "built, not retrieved or assembled from textbook steps. Reasonable experts "
        "could approach this differently."
    ),
}


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


def is_what_is(text: str) -> bool:
    return bool(re.match(r"^what is ", text.lower()))


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def call_anthropic(prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic(timeout=120.0)
    msg = client.messages.create(
        model=LABELING_MODEL,
        max_tokens=16000,
        thinking={"type": "enabled", "budget_tokens": 10000},
        messages=[{"role": "user", "content": prompt}],
    )
    for block in msg.content:
        if block.type == "text":
            return block.text.strip()
    return msg.content[-1].text.strip()


def call_openai(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=CONSISTENCY_MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


def safe_json_parse(text: str) -> list | None:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Generation prompts
# ---------------------------------------------------------------------------


def build_replacement_low_prompt(domain: str, n: int, existing_texts: list[str]) -> str:
    """Generate non-'What is X?' LOW queries to replace removed ones."""
    existing_block = "\n".join(f"  - {t}" for t in existing_texts[-30:])
    criteria = LABELING_CRITERIA["LOW"]
    return f"""You are generating replacement LOW-complexity queries for a benchmark dataset.

Domain: {domain}
Complexity: LOW
Definition: {criteria}

CRITICAL CONSTRAINT: Do NOT generate any query that starts with "What is".
Use OTHER formats: True/false, Is X?, How many, Who, When, Which, Name, Define,
fill-in-the-blank, yes/no factual questions, etc.

Every query must have a SINGLE, UNIVERSALLY AGREED answer statable in one sentence.
That is what makes it LOW — not its format.

Queries already in this cell (avoid duplicating):
{existing_block}

Generate exactly {n} diverse LOW queries for domain="{domain}".
None may start with "What is".
Return ONLY a JSON array of {n} query strings, no other text.
Format: ["Query 1.", "Query 2.", ...]"""


def build_what_is_medium_prompt(domain: str, n: int, existing_texts: list[str]) -> str:
    """Generate 'What is X?' MEDIUM queries."""
    existing_block = "\n".join(f"  - {t}" for t in existing_texts[-20:])
    criteria = LABELING_CRITERIA["MEDIUM"]
    return f"""You are generating MEDIUM-complexity "What is X?" queries for a benchmark dataset.

Domain: {domain}
Complexity: MEDIUM
Definition: {criteria}

CRITICAL: Every query MUST start with "What is" — but the answer must require
multi-step reasoning, not a one-sentence fact. The "What is X?" format can be
MEDIUM when X requires explaining a mechanism, tradeoff, or procedure.

Good examples of MEDIUM "What is X?" queries:
  - "What is the most efficient strategy for handling cache invalidation in a distributed system?"
    (requires reasoning through tradeoffs, not a single fact)
  - "What is the significance of the p-value in hypothesis testing and why is 0.05 commonly used?"
    (requires explaining statistical reasoning and convention)
  - "What is the role of backpropagation in training neural networks?"
    (requires explaining a multi-step algorithmic process)

BAD examples (these are LOW, not MEDIUM):
  - "What is the capital of France?" (single fact)
  - "What is 2+2?" (trivial)
  - "What is DNA?" (single-sentence definition)

Queries already in this cell (avoid duplicating):
{existing_block}

Generate exactly {n} "What is X?" MEDIUM queries for domain="{domain}".
Every query MUST start with "What is".
Return ONLY a JSON array of {n} query strings, no other text.
Format: ["What is ...", "What is ...", ...]"""


def build_what_is_high_prompt(domain: str, n: int, existing_texts: list[str]) -> str:
    """Generate 'What is X?' HIGH queries."""
    existing_block = "\n".join(f"  - {t}" for t in existing_texts[-20:])
    criteria = LABELING_CRITERIA["HIGH"]
    return f"""You are generating HIGH-complexity "What is X?" queries for a benchmark dataset.

Domain: {domain}
Complexity: HIGH
Definition: {criteria}

CRITICAL: Every query MUST start with "What is" — but the answer must require
constructing a novel reasoning path, expert judgment, or engaging with a contested
question. The "What is X?" format can be HIGH when the answer is genuinely open,
contested, or requires deep synthesis.

Good examples of HIGH "What is X?" queries:
  - "What is the most defensible interpretation of quantum mechanics given current experimental evidence?"
    (requires expert-level synthesis, genuinely contested)
  - "What is the right approach to aligning advanced AI systems with human values?"
    (open research question, no settled answer)
  - "What is the fundamental reason P≠NP is so hard to prove?"
    (requires deep mathematical insight, no simple answer)
  - "What is the best theoretical framework for understanding consciousness?"
    (contested across disciplines, requires constructing a position)

BAD examples (these are LOW or MEDIUM, not HIGH):
  - "What is quantum entanglement?" (MEDIUM — textbook explanation)
  - "What is the speed of light?" (LOW — single fact)

Queries already in this cell (avoid duplicating):
{existing_block}

Generate exactly {n} "What is X?" HIGH queries for domain="{domain}".
Every query MUST start with "What is".
Return ONLY a JSON array of {n} query strings, no other text.
Format: ["What is ...", "What is ...", ...]"""


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------


def generate_batch(prompt: str, n: int, dry_run: bool = False) -> list[str]:
    if dry_run:
        return [f"[DRY RUN] query {i+1}" for i in range(n)]

    for attempt in range(3):
        try:
            response = call_anthropic(prompt)
            batch = safe_json_parse(response)
            if isinstance(batch, list) and len(batch) >= 1:
                return [t.strip() for t in batch if isinstance(t, str) and len(t) >= 10]
            print(f"    [WARN] Parse failed on attempt {attempt+1}, retrying...")
            time.sleep(2)
        except Exception as e:
            print(f"    [ERROR] {e}, retrying in 5s...")
            time.sleep(5)
    return []


# ---------------------------------------------------------------------------
# Consistency check (fixed: batch_size=10, one retry)
# ---------------------------------------------------------------------------


def build_labeling_prompt(queries: list[str]) -> str:
    criteria_block = "\n".join(f"  {k}: {v}" for k, v in LABELING_CRITERIA.items())
    queries_block = "\n".join(f"{i+1}. {q}" for i, q in enumerate(queries))
    return f"""Classify each query by REASONING DEPTH required to answer it.
This is NOT about question format — "Is X?" and "True or false: X" can be LOW, MEDIUM, or HIGH.

Complexity levels:
{criteria_block}

THE KEY TEST: Does answering require (a) retrieving a single fact [LOW],
(b) following an established multi-step procedure [MEDIUM], or
(c) constructing a novel reasoning path or expert judgment [HIGH]?

Return ONLY a JSON array of labels, one per query in order.
Example for 3 queries: ["LOW", "HIGH", "MEDIUM"]

Queries:
{queries_block}"""


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    n = len(labels_a)
    classes = sorted(set(labels_a) | set(labels_b))
    po = sum(a == b for a, b in zip(labels_a, labels_b, strict=False)) / n
    count_a = Counter(labels_a)
    count_b = Counter(labels_b)
    pe = sum((count_a[c] / n) * (count_b[c] / n) for c in classes)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def run_consistency_check(queries: list[dict], dry_run: bool = False) -> dict:
    import random

    sample_n = max(50, int(len(queries) * 0.05))
    sample = random.sample(queries, sample_n)
    texts = [q["text"] for q in sample]
    original_labels = [q["ground_truth"] for q in sample]

    print(f"\nConsistency check: re-labeling {sample_n} queries with {CONSISTENCY_MODEL}...")

    if dry_run:
        check_labels = [
            lbl if random.random() > 0.1 else random.choice(CLASSES) for lbl in original_labels
        ]
    else:
        check_labels = []
        batch_size = 10  # reduced from 20 to eliminate length mismatch
        _norm = {"low": "LOW", "medium": "MEDIUM", "med": "MEDIUM", "high": "HIGH"}
        n_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_idx = i // batch_size
            batch_labels = None

            for attempt in range(2):  # one retry before padding
                try:
                    response = call_openai(build_labeling_prompt(batch_texts))
                    parsed = safe_json_parse(response)
                    if isinstance(parsed, list) and len(parsed) == len(batch_texts):
                        batch_labels = parsed
                        break
                    if attempt == 0:
                        print(
                            f"  [RETRY] Batch {batch_idx+1}/{n_batches}: length mismatch, retrying..."
                        )
                        time.sleep(1)
                except Exception as e:
                    print(f"  [ERROR] Batch {batch_idx+1}/{n_batches}: {e}")
                    if attempt == 0:
                        time.sleep(2)

            if batch_labels is None:
                print(
                    f"  [WARN] Batch {batch_idx+1}/{n_batches}: failed after retry, padding with UNKNOWN"
                )
                batch_labels = ["UNKNOWN"] * len(batch_texts)

            batch_labels = [
                _norm.get(str(lbl).strip().lower(), str(lbl).strip().upper())
                for lbl in batch_labels
            ]
            check_labels.extend(batch_labels)
            valid_count = sum(lbl in CLASSES for lbl in batch_labels)
            print(f"  Batch {batch_idx+1}/{n_batches}: {valid_count}/{len(batch_texts)} valid")
            time.sleep(0.3)

    valid = [(o, c) for o, c in zip(original_labels, check_labels, strict=False) if c in CLASSES]
    orig_valid, check_valid = zip(*valid, strict=False)
    kappa = cohen_kappa(list(orig_valid), list(check_valid))
    agreement = sum(a == b for a, b in valid) / len(valid)

    per_class = {}
    for cls in CLASSES:
        cls_pairs = [(o, c) for o, c in valid if o == cls]
        if cls_pairs:
            per_class[cls] = {
                "n": len(cls_pairs),
                "agreement": round(sum(o == c for o, c in cls_pairs) / len(cls_pairs), 4),
            }

    interpretation = (
        "near-perfect"
        if kappa >= 0.80
        else "substantial"
        if kappa >= 0.60
        else "moderate"
        if kappa >= 0.40
        else "fair/poor — review rubric"
    )

    report = {
        "primary_model": LABELING_MODEL,
        "consistency_model": CONSISTENCY_MODEL,
        "rubric_version": "v2 (reasoning-depth)",
        "n_checked": len(valid),
        "n_invalid": len(check_labels) - len(valid),
        "agreement": round(agreement, 4),
        "cohens_kappa": round(kappa, 4),
        "interpretation": interpretation,
        "per_class": per_class,
        "timestamp": utcnow(),
    }
    print(f"  Agreement: {agreement:.1%}, Cohen's κ = {kappa:.3f} ({interpretation})")
    return report


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_format_decoupling(queries: list[dict]) -> bool:
    print("\nFormat-decoupling validation:")
    yes_no = [
        q
        for q in queries
        if re.match(
            r"^(is |are |does |do |was |were |can |could |true or false)", q["text"].lower()
        )
    ]
    what_is = [q for q in queries if is_what_is(q["text"])]

    all_ok = True
    for label, subset in [("yes/no or T/F", yes_no), ("What is X?", what_is)]:
        dist = Counter(q["ground_truth"] for q in subset)
        total = len(subset)
        if total == 0:
            continue
        print(
            f"  {label} ({total} queries): LOW={dist['LOW']} MEDIUM={dist['MEDIUM']} HIGH={dist['HIGH']}"
        )
        for cls in CLASSES:
            pct = dist[cls] / total * 100
            if pct > 90:
                print(
                    f"  ⚠️  WARNING: {pct:.0f}% of '{label}' queries are {cls} — format-proxy bias detected!"
                )
                all_ok = False
            elif pct == 0:
                print(f"  ⚠️  WARNING: zero '{label}' queries in {cls} — format not decoupled!")
                all_ok = False
    if all_ok:
        print("  ✓ No format-proxy bias detected")
    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Fix format-proxy bias in stream_routing_benchmark.json"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-consistency", action="store_true")
    args = parser.parse_args()

    if not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        raise OSError("ANTHROPIC_API_KEY not set")

    with open(DATASET_PATH) as f:
        data = json.load(f)
    queries: list[dict] = data["queries"]

    print(f"Loaded {len(queries)} queries from {DATASET_PATH.name}")
    validate_format_decoupling(queries)

    # Build cell index
    cell: dict[tuple, list[dict]] = {}
    for d in DOMAINS:
        for c in CLASSES:
            cell[(d, c)] = []
    for q in queries:
        cell[(q["domain"], q["ground_truth"])].append(q)

    new_queries: list[dict] = []

    for domain in DOMAINS:
        print(f"\n--- {domain} ---")

        # ── FIX 1: cap "What is X?" in LOW ──────────────────────────────────
        low_all = cell[(domain, "LOW")]
        what_is_low = [q for q in low_all if is_what_is(q["text"])]
        other_low = [q for q in low_all if not is_what_is(q["text"])]

        n_to_replace = max(0, len(what_is_low) - MAX_WHAT_IS_LOW_PER_DOMAIN)
        kept_what_is_low = what_is_low[:MAX_WHAT_IS_LOW_PER_DOMAIN]

        if n_to_replace > 0:
            print(
                f"  LOW: removing {n_to_replace} 'What is X?' queries, generating replacements..."
            )
            existing_texts = [q["text"] for q in low_all]
            prompt = build_replacement_low_prompt(domain, n_to_replace, existing_texts)
            replacements = generate_batch(prompt, n_to_replace, dry_run=args.dry_run)
            existing_set = {t.lower().strip() for t in existing_texts}
            replacement_records = []
            for text in replacements:
                if text.lower().strip() not in existing_set:
                    existing_set.add(text.lower().strip())
                    replacement_records.append(
                        {
                            "text": text,
                            "domain": domain,
                            "ground_truth": "LOW",
                            "generated_by": LABELING_MODEL,
                            "generated_at": utcnow(),
                            "fix": "replacement_for_what_is_bias",
                        }
                    )
            if len(replacement_records) < n_to_replace:
                print(f"    [WARN] Only got {len(replacement_records)}/{n_to_replace} replacements")
            new_low = kept_what_is_low + other_low + replacement_records
        else:
            print(
                f"  LOW: {len(what_is_low)} 'What is X?' already ≤ {MAX_WHAT_IS_LOW_PER_DOMAIN}, no change"
            )
            new_low = low_all

        # Trim/pad to exactly 384
        new_low = new_low[:384]
        new_queries.extend(new_low)

        # ── FIX 2: add "What is X?" MEDIUM ──────────────────────────────────
        med_all = cell[(domain, "MEDIUM")]
        what_is_med = [q for q in med_all if is_what_is(q["text"])]
        other_med = [q for q in med_all if not is_what_is(q["text"])]
        n_med_needed = max(0, TARGET_WHAT_IS_PER_DOMAIN - len(what_is_med))

        if n_med_needed > 0:
            print(f"  MEDIUM: generating {n_med_needed} 'What is X?' queries...")
            existing_texts = [q["text"] for q in med_all]
            prompt = build_what_is_medium_prompt(domain, n_med_needed, existing_texts)
            new_what_is_med = generate_batch(prompt, n_med_needed, dry_run=args.dry_run)
            existing_set = {t.lower().strip() for t in existing_texts}
            new_med_records = []
            for text in new_what_is_med:
                if text.lower().strip() not in existing_set and is_what_is(text):
                    existing_set.add(text.lower().strip())
                    new_med_records.append(
                        {
                            "text": text,
                            "domain": domain,
                            "ground_truth": "MEDIUM",
                            "generated_by": LABELING_MODEL,
                            "generated_at": utcnow(),
                            "fix": "added_what_is_medium",
                        }
                    )
            # Remove same count of non-"What is X?" MEDIUM to keep cell at 384
            n_remove = len(new_med_records)
            other_med_kept = (
                other_med[:-n_remove] if n_remove > 0 and n_remove < len(other_med) else other_med
            )
            new_med = what_is_med + new_med_records + other_med_kept
        else:
            print(
                f"  MEDIUM: already have {len(what_is_med)} 'What is X?' ≥ {TARGET_WHAT_IS_PER_DOMAIN}"
            )
            new_med = med_all

        new_med = new_med[:384]
        new_queries.extend(new_med)

        # ── FIX 3: add "What is X?" HIGH ────────────────────────────────────
        high_all = cell[(domain, "HIGH")]
        what_is_high = [q for q in high_all if is_what_is(q["text"])]
        other_high = [q for q in high_all if not is_what_is(q["text"])]
        n_high_needed = max(0, TARGET_WHAT_IS_PER_DOMAIN - len(what_is_high))

        if n_high_needed > 0:
            print(f"  HIGH: generating {n_high_needed} 'What is X?' queries...")
            existing_texts = [q["text"] for q in high_all]
            prompt = build_what_is_high_prompt(domain, n_high_needed, existing_texts)
            new_what_is_high = generate_batch(prompt, n_high_needed, dry_run=args.dry_run)
            existing_set = {t.lower().strip() for t in existing_texts}
            new_high_records = []
            for text in new_what_is_high:
                if text.lower().strip() not in existing_set and is_what_is(text):
                    existing_set.add(text.lower().strip())
                    new_high_records.append(
                        {
                            "text": text,
                            "domain": domain,
                            "ground_truth": "HIGH",
                            "generated_by": LABELING_MODEL,
                            "generated_at": utcnow(),
                            "fix": "added_what_is_high",
                        }
                    )
            n_remove = len(new_high_records)
            other_high_kept = (
                other_high[:-n_remove]
                if n_remove > 0 and n_remove < len(other_high)
                else other_high
            )
            new_high = what_is_high + new_high_records + other_high_kept
        else:
            print(
                f"  HIGH: already have {len(what_is_high)} 'What is X?' ≥ {TARGET_WHAT_IS_PER_DOMAIN}"
            )
            new_high = high_all

        new_high = new_high[:384]
        new_queries.extend(new_high)

    # Re-assign IDs
    for i, q in enumerate(new_queries, start=1):
        q["id"] = i

    # Verify cell counts
    print("\nVerifying cell counts after fix:")
    cell_counts = Counter((q["domain"], q["ground_truth"]) for q in new_queries)
    all_ok = True
    for d in DOMAINS:
        for c in CLASSES:
            n = cell_counts[(d, c)]
            flag = "✓" if n == 384 else f"← {n} !"
            print(f"  {d:30s} {c:6s} {n} {flag}")
            if n != 384:
                all_ok = False
    if not all_ok:
        print("\n[ERROR] Some cells are not 384 — review the fix script logic.")

    # Format validation
    validate_format_decoupling(new_queries)

    # Save
    data["queries"] = new_queries
    data["_stats"]["total"] = len(new_queries)
    data["_stats"]["generated_at"] = utcnow()
    data["_fix_applied"] = {
        "timestamp": utcnow(),
        "description": "Format-proxy bias fix: capped What is X? LOW per domain, added What is X? MEDIUM and HIGH",
        "max_what_is_low_per_domain": MAX_WHAT_IS_LOW_PER_DOMAIN,
        "target_what_is_per_domain_med_high": TARGET_WHAT_IS_PER_DOMAIN,
    }

    with open(DATASET_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n✓ Saved updated dataset to {DATASET_PATH} ({len(new_queries)} queries)")

    # Consistency check with fixed batch size
    if not args.skip_consistency and (os.environ.get("OPENAI_API_KEY") or args.dry_run):
        report = run_consistency_check(new_queries, dry_run=args.dry_run)
        with open(CONSISTENCY_PATH, "w") as f:
            json.dump(report, f, indent=2)
        print(f"✓ Consistency report saved to {CONSISTENCY_PATH}")
    else:
        print("\n[SKIPPED] Consistency check (--skip-consistency or no OPENAI_API_KEY)")

    print("\nDone. Next step: python scripts/eval/train_modernbert.py")


if __name__ == "__main__":
    main()
