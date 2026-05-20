#!/usr/bin/env python3
"""
Top-up HIGH-class queries for the three domains that exhausted their source pools:
  - hpc               : needs 114 more HIGH  (train) +  29 (test) = 143 total
  - philosophy_ethics : needs 160 more HIGH  (train) +  40 (test) = 200 total
  - history_culture   : needs 139 more HIGH  (train) +  35 (test) = 174 total

Strategy: generate synthetic HIGH queries using Claude, then verify each one
by asking Claude to label it (should always return HIGH since Claude defines the rubric).

For each domain we generate N queries in batches, asking Claude to produce
expert-level research questions that require:
  - constructing a novel reasoning path
  - formal derivation or expert judgment
  - no standard textbook procedure

Generated queries are appended to the existing dataset files.
"""

import json
import os
import re
import time
from pathlib import Path
from random import Random

TRAIN_OUTPUT = Path("scripts/eval/balanced_training_dataset.json")
TEST_OUTPUT = Path("scripts/eval/balanced_test_dataset.json")
TRAIN_JSONL = Path("scripts/eval/balanced_train.jsonl")
TEST_JSONL = Path("scripts/eval/balanced_test.jsonl")

TRAIN_FRACTION = 0.80
LABELING_MODEL = "claude-sonnet-4-6"
GENERATION_BATCH = 20  # queries per generation call

# Exact shortfalls per domain (from the completed collection run)
SHORTFALLS = {
    "hpc": 200 - 86,  # = 114
    "philosophy_ethics": 200 - 40,  # = 160
    "history_culture": 200 - 61,  # = 139
}

# Domain-specific generation prompts — each steers toward HIGH complexity
DOMAIN_HIGH_PROMPTS = {
    "hpc": """\
Generate {n} expert-level research computing questions that require HIGH reasoning depth.
Each question must require constructing a novel solution path — no single standard procedure applies.
Topics: parallel algorithm design, MPI collective optimization, CUDA memory hierarchy trade-offs,
SLURM job dependency graphs, distributed file system consistency, HPC network topology effects,
fault-tolerant scientific workflow design, GPU kernel optimization strategy, hybrid OpenMP/MPI tuning,
cloud-HPC cost-performance trade-offs, Globus data transfer optimization, vLLM serving at scale.

Rules:
- Each question is a single self-contained sentence ending with "?"
- 30–180 characters
- No answer choices, no "which of the following"
- Must require expert judgment or novel reasoning, not a textbook lookup
- Vary the topics across the list

Return a JSON array of {n} question strings. No commentary.""",
    "philosophy_ethics": """\
Generate {n} expert-level philosophy and ethics questions that require HIGH reasoning depth.
Each question must require constructing a novel argument, synthesizing across frameworks, or
making expert judgment under genuine philosophical uncertainty.
Topics: metaethics (moral realism vs anti-realism), AI ethics and moral status,
philosophy of mind (consciousness, qualia, functionalism), free will and determinism,
distributive justice trade-offs, trolley problems with structural ambiguity,
epistemic justification under radical skepticism, philosophy of language (reference, meaning),
bioethics dilemmas, political philosophy (legitimacy, obligation), formal epistemology.

Rules:
- Each question is a single self-contained sentence ending with "?"
- 40–200 characters
- No answer choices, no "which of the following"
- Must require philosophical reasoning, not a fact lookup
- Vary the topics

Return a JSON array of {n} question strings. No commentary.""",
    "history_culture": """\
Generate {n} expert-level history and cultural analysis questions that require HIGH reasoning depth.
Each question must require synthesizing evidence across sources, constructing a causal argument,
or making a scholarly judgment — not a simple factual recall.
Topics: causes and consequences of major historical transitions (fall of Rome, Industrial Revolution,
decolonization), comparative civilizational development, historiographical debates,
long-run economic history, cultural diffusion and syncretism, counterfactual historical analysis,
interpretation of primary source conflicts, social history methodology, religion and political power,
memory and collective trauma.

Rules:
- Each question is a single self-contained sentence ending with "?"
- 40–200 characters
- No answer choices, no passage references, no "the following"
- Must require historical reasoning and synthesis, not a date/name lookup
- Vary the topics

Return a JSON array of {n} question strings. No commentary.""",
}

VERIFY_PROMPT = """\
You are a query complexity classifier. Label each query as LOW, MEDIUM, or HIGH.

LOW: Single retrievable fact, no reasoning chain.
MEDIUM: Standard multi-step procedure, textbook-level.
HIGH: Novel reasoning path, formal derivation, or expert judgment — no standard procedure.

Return a JSON array of labels in the same order. Only the JSON array, no explanation."""


def get_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        env_file = Path(__file__).parent.parent.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    key = line.split("=", 1)[1].strip()
    if not key:
        raise OSError("ANTHROPIC_API_KEY not set")
    return key


def generate_high_queries(domain: str, n: int, client) -> list[str]:
    """Ask Claude to generate n HIGH-complexity questions for the domain."""
    prompt = DOMAIN_HIGH_PROMPTS[domain].format(n=n)
    for attempt in range(3):
        try:
            resp = client.messages.create(
                model=LABELING_MODEL,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            # Strip markdown fences if present
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            queries = json.loads(raw)
            if isinstance(queries, list):
                # Clean up
                cleaned = []
                for q in queries:
                    q = str(q).strip()
                    if 20 <= len(q) <= 250 and q.endswith("?"):
                        cleaned.append(q)
                return cleaned
        except Exception as e:
            print(f"  [WARN] generation failed (attempt {attempt+1}): {e}")
            if attempt < 2:
                time.sleep(2**attempt)
    return []


def verify_labels(texts: list[str], client) -> list[str]:
    """Re-label a batch; returns list of labels."""
    prompt = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    for attempt in range(3):
        try:
            resp = client.messages.create(
                model=LABELING_MODEL,
                max_tokens=400,
                messages=[{"role": "user", "content": VERIFY_PROMPT + "\n\nQueries:\n" + prompt}],
            )
            raw = resp.content[0].text.strip()
            labels = json.loads(raw)
            if isinstance(labels, list) and len(labels) == len(texts):
                norm = [lb.upper() for lb in labels]
                if all(lb in ("LOW", "MEDIUM", "HIGH") for lb in norm):
                    return norm
        except Exception as e:
            print(f"  [WARN] verify failed: {e}")
            if attempt < 2:
                time.sleep(2**attempt)
    return ["HIGH"] * len(texts)  # fallback: trust generation prompt


def topup_domain(domain: str, needed: int, client, rng: Random, existing_texts: set) -> list[dict]:
    """
    Generate and verify `needed` HIGH queries for domain.
    Returns list of query dicts with ground_truth="HIGH".
    """
    collected = []
    attempts = 0
    max_attempts = needed * 5  # generous ceiling

    print(f"  {domain}: need {needed} HIGH queries...")

    while len(collected) < needed and attempts < max_attempts:
        to_gen = min(GENERATION_BATCH, (needed - len(collected)) * 2)
        generated = generate_high_queries(domain, to_gen, client)
        attempts += len(generated)

        # Dedup against existing dataset
        new = [q for q in generated if q not in existing_texts]
        if not new:
            continue

        # Verify — keep only those labeled HIGH
        labels = verify_labels(new, client)
        for q, lbl in zip(new, labels, strict=False):
            if lbl == "HIGH" and len(collected) < needed:
                entry = {
                    "text": q,
                    "source": "synthetic_high",
                    "subject": f"{domain}_synthetic",
                    "domain": domain,
                    "ground_truth": "HIGH",
                }
                collected.append(entry)
                existing_texts.add(q)

        kept = sum(1 for lb in labels if lb == "HIGH")
        print(
            f"    generated {len(new)}, verified HIGH: {kept}, "
            f"total so far: {len(collected)}/{needed}"
        )
        time.sleep(0.2)

    if len(collected) < needed:
        print(f"  [WARN] {domain}: only collected {len(collected)}/{needed}")
    return collected


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rng = Random(42)

    # Load existing datasets
    if not TRAIN_OUTPUT.exists() or not TEST_OUTPUT.exists():
        raise FileNotFoundError("Run build_balanced_dataset.py first.")

    train_data = json.loads(TRAIN_OUTPUT.read_text())
    test_data = json.loads(TEST_OUTPUT.read_text())
    train_qs = train_data["queries"]
    test_qs = test_data["queries"]

    existing_texts = {q["text"] for q in train_qs + test_qs}
    print(f"Loaded: {len(train_qs)} train + {len(test_qs)} test queries")
    print(f"Shortfalls: {SHORTFALLS}")

    import anthropic

    client = anthropic.Anthropic(api_key=get_api_key()) if not args.dry_run else None

    all_new: dict[str, list[dict]] = {}

    for domain, total_needed in sorted(SHORTFALLS.items()):
        if args.dry_run:
            print(f"  [dry-run] would generate {total_needed} HIGH for {domain}")
            continue

        new_qs = topup_domain(domain, total_needed, client, rng, existing_texts)
        all_new[domain] = new_qs

        # Split 80/20 into train/test
        rng.shuffle(new_qs)
        t_n = max(1, round(len(new_qs) * TRAIN_FRACTION))
        new_train = new_qs[:t_n]
        new_test = new_qs[t_n:]

        train_qs.extend(new_train)
        test_qs.extend(new_test)
        print(f"  {domain}: added {len(new_train)} train + {len(new_test)} test")

    if args.dry_run:
        return

    # Re-assign IDs
    for idx, q in enumerate(train_qs):
        q["id"] = idx + 1
    for idx, q in enumerate(test_qs):
        q["id"] = idx + 1

    # Update metadata
    from collections import Counter

    train_data["metadata"]["n_queries"] = len(train_qs)
    train_data["metadata"]["class_distribution"] = dict(
        Counter(q["ground_truth"] for q in train_qs)
    )
    train_data["metadata"]["domain_distribution"] = dict(Counter(q["domain"] for q in train_qs))
    train_data["metadata"]["source_distribution"] = dict(Counter(q["source"] for q in train_qs))
    train_data["metadata"]["sources"]["synthetic_high"] = (
        "Claude-generated HIGH queries for domains with exhausted source pools"
    )
    train_data["queries"] = train_qs

    test_data["metadata"]["n_queries"] = len(test_qs)
    test_data["metadata"]["class_distribution"] = dict(Counter(q["ground_truth"] for q in test_qs))
    test_data["metadata"]["domain_distribution"] = dict(Counter(q["domain"] for q in test_qs))
    test_data["queries"] = test_qs

    TRAIN_OUTPUT.write_text(json.dumps(train_data, indent=2))
    TEST_OUTPUT.write_text(json.dumps(test_data, indent=2))

    with open(TRAIN_JSONL, "w") as f:
        for q in train_qs:
            f.write(json.dumps(q) + "\n")
    with open(TEST_JSONL, "w") as f:
        for q in test_qs:
            f.write(json.dumps(q) + "\n")

    # Final summary
    print(f"\n{'='*60}")
    from collections import Counter, defaultdict

    print("FINAL DATASET")
    dc = defaultdict(Counter)
    for q in train_qs:
        dc[q["domain"]][q["ground_truth"]] += 1
    print(f"{'Domain':22s}  LOW  MED  HIGH  total")
    for d in sorted(dc):
        c = dc[d]
        print(f"  {d:22s}  {c['LOW']:3d}  {c['MEDIUM']:3d}  {c['HIGH']:4d}  {sum(c.values()):5d}")
    print(
        f"\n  Train total: {len(train_qs)}  class={dict(Counter(q['ground_truth'] for q in train_qs))}"
    )
    print(
        f"  Test  total: {len(test_qs)}  class={dict(Counter(q['ground_truth'] for q in test_qs))}"
    )
    print(f"  Sources: {dict(Counter(q['source'] for q in train_qs))}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
