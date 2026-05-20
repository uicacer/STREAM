#!/usr/bin/env python3
"""
Build a mixed real-world training dataset for ModernBERT.

Sources:
  1. MMLU (cais/mmlu) — academic exam questions, 57 subjects, ungated
  2. StackExchange (sentence-transformers/stackexchange-duplicates) — 304K real SE titles
  3. research_computing — from existing benchmark (only Claude-generated domain
     kept: HPC, Globus, vLLM, SLURM — no real-world source exists for this)

Arena (lmsys/chatbot_arena_conversations) is kept as a fixed held-out OOD
test set and is NEVER included in this training dataset.

All queries are labeled by Claude using the same reasoning-depth rubric as
the original benchmark.

Usage:
    python scripts/eval/build_mixed_dataset.py
    python scripts/eval/build_mixed_dataset.py --dry-run   # 30 queries, no API

Output:
    scripts/eval/mixed_training_dataset.json

Estimated cost: ~$0.70 (5,760 queries × ~130 tokens × $0.003/1K input)
Estimated time: ~15 minutes
"""

import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

BENCHMARK_PATH = Path("scripts/eval/stream_routing_benchmark.json")
OUTPUT_PATH = Path("scripts/eval/mixed_training_dataset.json")

# 1,536 = next multiple of 384 (Wald-formula minimum for ±5pp CI) that
# ensures total HIGH queries ≥ 384 given the empirical HIGH rates across
# sources: MMLU ~5%, StackExchange general ~10%, StackExchange HPC ~10%.
# Combined rate: 0.25 → need n_per_source ≥ 384/0.25 = 1,531 → 1,536.
N_MMLU = 1536
N_STACKEXCHANGE = 1536
LABEL_BATCH_SIZE = 10
LABELING_MODEL = "claude-sonnet-4-6"

STACKEXCHANGE_DATASET = "sentence-transformers/stackexchange-duplicates"
STACKEXCHANGE_CONFIG = "title-title-pair"

COMPLEXITY_RUBRIC = """You are a query complexity classifier for an LLM routing system.

**Complexity rubric** (based on reasoning depth, NOT question format):

LOW: Single retrievable fact or trivial computation. The answer can be stated
     in one sentence with no reasoning chain required.
     Example: "What is the capital of France?" / "Is Python interpreted?"

MEDIUM: Apply a standard procedure or assemble 2-4 concepts. The reasoning
        path is textbook-level and well-established.
        Example: "Explain quicksort and its time complexity." / "Compare TCP and UDP."

HIGH: Construct a novel reasoning path, formal derivation, or expert judgment.
      No standard procedure exists — the solver must build the path.
      Example: "Is P=NP? Summarize the state of evidence."
      / "Design a fault-tolerant distributed key-value store."

**Key rule**: Format is NOT the complexity signal. "What is X?" can be
LOW, MEDIUM, or HIGH depending on what reasoning is required to answer.

Label each query with exactly one of: LOW, MEDIUM, or HIGH
Return a JSON array of labels in the same order as the input queries.
Do NOT include any explanation. Only the JSON array.

Example response for 3 queries: ["LOW", "HIGH", "MEDIUM"]"""


# ---------------------------------------------------------------------------
# Source 1: MMLU
# ---------------------------------------------------------------------------


def load_mmlu(n: int, dry_run: bool) -> list[dict]:
    """Sample n questions from MMLU across all 57 subjects."""
    import random

    from datasets import load_dataset

    print("Loading MMLU (cais/mmlu)...")
    # Use the 'all' config which contains all subjects
    ds = load_dataset("cais/mmlu", "all", split="test")
    print(f"  Total MMLU questions: {len(ds):,}")

    # Convert multiple-choice to free-form: just use the question stem
    # Skip questions that are just "A", "B", "C", "D" references
    questions = []
    for row in ds:
        q = row["question"].strip()
        # Filter: must be a real question, not just a reference
        if (
            len(q) >= 20
            and len(q) <= 300
            and "?" in q
            or any(
                q.lower().startswith(w)
                for w in [
                    "what",
                    "which",
                    "how",
                    "why",
                    "when",
                    "where",
                    "who",
                    "explain",
                    "describe",
                    "define",
                    "calculate",
                    "find",
                    "prove",
                    "show",
                    "is ",
                    "are ",
                    "does ",
                    "do ",
                    "can ",
                ]
            )
        ):
            questions.append(
                {
                    "text": q,
                    "subject": row.get("subject", "unknown"),
                    "source": "mmlu",
                }
            )

    print(f"  After filtering: {len(questions):,}")

    if dry_run:
        n = min(n, 20)

    rng = random.Random(42)
    rng.shuffle(questions)
    sampled = questions[:n]
    print(f"  Sampled {len(sampled)} MMLU questions")
    return sampled


# ---------------------------------------------------------------------------
# Source 2: StackExchange
# ---------------------------------------------------------------------------


def load_stackexchange(n: int, dry_run: bool, exclude_texts: set = None) -> list[dict]:
    """Sample n question titles from sentence-transformers/stackexchange-duplicates.

    304K clean SE question titles (title-title-pair config), covering diverse
    topics: programming, math, science, language, law, etc. No loading scripts.
    Both title1 and title2 from each row are used as independent questions.

    exclude_texts: set of already-used titles to exclude (for cross-source dedup).
    """
    import random

    from datasets import load_dataset

    exclude_texts = exclude_texts or set()

    print(f"  Loading {STACKEXCHANGE_DATASET} ({STACKEXCHANGE_CONFIG})...")
    ds = load_dataset(STACKEXCHANGE_DATASET, STACKEXCHANGE_CONFIG, split="train")
    print(f"  Raw dataset size: {len(ds):,} pairs ({len(ds)*2:,} titles)")

    seen = set()
    questions = []
    for row in ds:
        for field in ("title1", "title2"):
            text = row.get(field, "").strip()
            if (
                20 <= len(text) <= 300
                and re.search(r"[a-zA-Z]", text)
                and text not in exclude_texts
                and text not in seen
            ):
                seen.add(text)
                questions.append(
                    {
                        "text": text,
                        "subject": "stackexchange",
                        "source": "stackexchange",
                    }
                )

    print(f"  After filtering: {len(questions):,}")

    if dry_run:
        n = min(n, 20)

    rng = random.Random(42)
    rng.shuffle(questions)
    sampled = questions[:n]
    print(f"  Sampled {len(sampled)} StackExchange questions")
    return sampled


# ---------------------------------------------------------------------------
# Source 3: research_computing — keyword-filtered StackExchange titles
# ---------------------------------------------------------------------------

# Keywords that identify questions from researchers and students working with
# computational tools: HPC infrastructure, scientific Python, numerical methods,
# research workflows, and scientific domains.  All are unambiguous in context.
RESEARCH_COMPUTING_KEYWORDS = [
    # HPC infrastructure
    "slurm",
    "sbatch",
    "mpirun",
    "mpiexec",
    "openmpi",
    "apptainer",
    "singularity",
    "module load",
    "lmod",
    "compute cluster",
    "hpc cluster",
    "login node",
    "compute node",
    "batch job",
    "job scheduler",
    "qsub",
    "infiniband",
    # Scientific Python stack
    "numpy",
    "scipy",
    "matplotlib",
    "pandas dataframe",
    "jupyter notebook",
    "pytorch",
    "tensorflow",
    "scikit-learn",
    "sklearn",
    "cuda",
    "gpu memory",
    "gpu training",
    # Numerical / scientific computing
    "numerical method",
    "differential equation",
    "finite element",
    "monte carlo",
    "linear algebra",
    "matrix multiplication",
    "parallel processing",
    "multiprocessing",
    "multithreading",
    "distributed computing",
    "map reduce",
    # Research computing workflows
    "ssh tunnel",
    "ssh key",
    "remote server",
    "bash script",
    "environment variable",
    "virtual environment",
    "conda",
    "docker container",
    "kubernetes",
    "dask",
    "spark",
    "data pipeline",
    "workflow automation",
    # Scientific domains
    "molecular dynamics",
    "protein structure",
    "genome",
    "bioinformatics",
    "climate model",
    "fluid dynamics",
    "finite difference",
    "computational chemistry",
    "quantum computing circuit",
]


def load_research_computing_stackexchange(n: int, dry_run: bool) -> list[dict]:
    """Sample n research-computing queries from the StackExchange duplicates dataset.

    Uses keyword filtering to select titles relevant to researchers and students
    working with HPC infrastructure, scientific Python, numerical methods, and
    research workflows.  All queries are real human-written text — no Claude
    generation involved, eliminating circular label bias.

    Source: sentence-transformers/stackexchange-duplicates (title-title-pair)
    Same dataset already used for Source 2; filtered to research-computing topics.
    """
    import random

    from datasets import load_dataset

    print("  Loading StackExchange (research-computing filter)...")
    ds = load_dataset(STACKEXCHANGE_DATASET, STACKEXCHANGE_CONFIG, split="train")

    matches = []
    seen = set()
    for row in ds:
        for field in ("title1", "title2"):
            text = row.get(field, "").strip()
            tl = text.lower()
            if (
                20 <= len(text) <= 300
                and re.search(r"[a-zA-Z]", text)
                and text not in seen
                and any(kw in tl for kw in RESEARCH_COMPUTING_KEYWORDS)
            ):
                matches.append(
                    {
                        "text": text,
                        "subject": "research_computing",
                        "source": "stackexchange_hpc",
                    }
                )
                seen.add(text)

    print(f"  Research-computing matches: {len(matches):,}")

    if dry_run:
        n = min(n, 20)

    if len(matches) < n:
        print(f"  [WARN] Only {len(matches)} matches found, need {n}. Using all.")
        n = len(matches)

    rng = random.Random(42)
    rng.shuffle(matches)
    sampled = matches[:n]
    print(f"  Sampled {len(sampled)} research-computing queries from StackExchange")
    return sampled


# ---------------------------------------------------------------------------
# Labeling with Claude
# ---------------------------------------------------------------------------


def get_api_key() -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        env_file = Path(__file__).parent.parent.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break
    if not api_key:
        raise OSError("ANTHROPIC_API_KEY not set")
    return api_key


def label_with_claude(queries: list[dict], dry_run: bool) -> list[dict]:
    """Add 'ground_truth' field to queries that don't already have one."""
    to_label = [q for q in queries if "ground_truth" not in q]
    already_labeled = [q for q in queries if "ground_truth" in q]

    if not to_label:
        print(f"  All {len(queries)} queries already labeled")
        return queries

    print(f"  Labeling {len(to_label)} queries with {LABELING_MODEL}...")
    print(f"  ({len(already_labeled)} already have ground_truth from benchmark)")

    if dry_run:
        import random

        for q in to_label:
            q["ground_truth"] = random.choice(["LOW", "MEDIUM", "HIGH"])
        print("  [DRY RUN] Assigned fake labels")
        return already_labeled + to_label

    import anthropic

    client = anthropic.Anthropic(api_key=get_api_key())

    texts = [q["text"] for q in to_label]
    n_batches = (len(texts) + LABEL_BATCH_SIZE - 1) // LABEL_BATCH_SIZE
    all_labels: list[str] = []

    for batch_idx in range(n_batches):
        start = batch_idx * LABEL_BATCH_SIZE
        batch_texts = texts[start : start + LABEL_BATCH_SIZE]
        prompt = "\n".join(f"{i+1}. {t}" for i, t in enumerate(batch_texts))

        for attempt in range(3):
            try:
                response = client.messages.create(
                    model=LABELING_MODEL,
                    max_tokens=128,
                    messages=[
                        {"role": "user", "content": COMPLEXITY_RUBRIC + "\n\nQueries:\n" + prompt}
                    ],
                )
                raw = response.content[0].text.strip()
                labels = json.loads(raw)
                if isinstance(labels, list) and len(labels) == len(batch_texts):
                    valid = [lb.upper() for lb in labels if lb.upper() in ("LOW", "MEDIUM", "HIGH")]
                    if len(valid) == len(batch_texts):
                        all_labels.extend(valid)
                        break
                print(f"  [WARN] Batch {batch_idx+1}/{n_batches}: unexpected response, retrying...")
                if attempt == 2:
                    all_labels.extend(["MEDIUM"] * len(batch_texts))
            except Exception as e:
                if attempt == 2:
                    print(f"  [WARN] Batch {batch_idx+1}/{n_batches}: failed: {e}")
                    all_labels.extend(["MEDIUM"] * len(batch_texts))
                    break
                time.sleep(2)

        if (batch_idx + 1) % 10 == 0 or batch_idx + 1 == n_batches:
            print(f"  Batch {batch_idx+1}/{n_batches}: {len(all_labels)} labeled so far")
        time.sleep(0.3)

    for q, label in zip(to_label, all_labels, strict=False):
        q["ground_truth"] = label

    return already_labeled + to_label


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="20 queries per source, fake labels")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument(
        "--skip-stackexchange",
        action="store_true",
        help="Skip StackExchange (use if dataset is unavailable)",
    )
    parser.add_argument(
        "--reuse-mmlu",
        action="store_true",
        help="Reuse already-labeled MMLU queries from base dataset file",
    )
    parser.add_argument(
        "--reuse-stackexchange",
        action="store_true",
        help="Reuse already-labeled StackExchange queries from base dataset file",
    )
    parser.add_argument(
        "--reuse-research-computing",
        action="store_true",
        help="Reuse already-labeled research-computing queries from base dataset file",
    )
    parser.add_argument(
        "--reuse-from",
        default=str(OUTPUT_PATH),
        help="Source file to reuse labeled queries from (default: mixed_training_dataset.json)",
    )
    parser.add_argument(
        "--balance-mode",
        choices=["none", "oversample", "downsample"],
        default="none",
        help="none: keep raw imbalanced data. oversample: repeat minority to majority count. downsample: cut majority to minority count.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    reuse_path = Path(args.reuse_from)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_mmlu = 20 if args.dry_run else N_MMLU
    n_se = 20 if args.dry_run else N_STACKEXCHANGE

    all_queries = []

    # Source 1: MMLU
    print("\n--- Source 1: MMLU ---")
    if args.reuse_mmlu and reuse_path.exists():
        with open(reuse_path) as f:
            existing = json.load(f)
        mmlu_queries = [q for q in existing["queries"] if q.get("source") == "mmlu"][:n_mmlu]
        print(f"  Reusing {len(mmlu_queries)} already-labeled MMLU queries from {reuse_path}")
    else:
        mmlu_queries = load_mmlu(n_mmlu, args.dry_run)
        mmlu_queries = label_with_claude(mmlu_queries, args.dry_run)
    all_queries.extend(mmlu_queries)
    print(f"  Distribution: {Counter(q['ground_truth'] for q in mmlu_queries)}")

    # Source 3: research_computing — StackExchange keyword-filtered (real human text)
    # Built BEFORE source 2 so we can exclude HPC titles from the general SE sample.
    print("\n--- Source 3: research_computing (StackExchange keyword-filtered) ---")
    n_rc = 20 if args.dry_run else N_MMLU  # 1,536 — same as MMLU and SE for symmetric sources
    if args.reuse_research_computing and reuse_path.exists():
        with open(reuse_path) as f:
            existing = json.load(f)
        rc_queries = [q for q in existing["queries"] if q.get("source") == "stackexchange_hpc"]
        print(
            f"  Reusing {len(rc_queries)} already-labeled research-computing queries from {reuse_path}"
        )
    else:
        rc_queries = load_research_computing_stackexchange(n_rc, args.dry_run)
        rc_queries = label_with_claude(rc_queries, args.dry_run)
    all_queries.extend(rc_queries)
    print(f"  Distribution: {Counter(q['ground_truth'] for q in rc_queries)}")

    # Source 2: StackExchange general — exclude any titles already used in source 3
    rc_texts = {q["text"] for q in rc_queries}
    if not args.skip_stackexchange:
        print("\n--- Source 2: StackExchange ---")
        if args.reuse_stackexchange and reuse_path.exists():
            with open(reuse_path) as f:
                existing = json.load(f)
            se_queries = [
                q
                for q in existing["queries"]
                if q.get("source") == "stackexchange" and q["text"] not in rc_texts
            ][:n_se]
            print(
                f"  Reusing {len(se_queries)} already-labeled StackExchange queries from {reuse_path}"
            )
        else:
            se_queries = load_stackexchange(n_se, args.dry_run, exclude_texts=rc_texts)
            se_queries = label_with_claude(se_queries, args.dry_run)
        all_queries.extend(se_queries)
        print(f"  Distribution: {Counter(q['ground_truth'] for q in se_queries)}")
    else:
        print("\n--- Source 2: StackExchange (skipped) ---")

    import random as _random

    rng_bal = _random.Random(42)

    by_class: dict = {"LOW": [], "MEDIUM": [], "HIGH": []}
    for q in all_queries:
        by_class[q["ground_truth"]].append(q)

    print(f"\n  Class counts before balancing: { {k: len(v) for k, v in by_class.items()} }")

    if args.balance_mode == "oversample":
        target = max(len(v) for v in by_class.values())
        print(f"  Oversampling minority classes to {target} per class")
        balanced = []
        for _cls, qs in by_class.items():
            if len(qs) < target:
                oversampled = qs * (target // len(qs) + 1)
                rng_bal.shuffle(oversampled)
                balanced.extend(oversampled[:target])
            else:
                balanced.extend(qs)
        rng_bal.shuffle(balanced)
        all_queries = balanced

    elif args.balance_mode == "downsample":
        target = min(len(v) for v in by_class.values())
        print(f"  Downsampling majority classes to {target} per class")
        balanced = []
        for _cls, qs in by_class.items():
            rng_bal.shuffle(qs)
            balanced.extend(qs[:target])
        rng_bal.shuffle(balanced)
        all_queries = balanced

    else:
        print("  No balancing applied (imbalanced baseline)")

    # Assign IDs
    for i, q in enumerate(all_queries):
        q["id"] = i + 1

    dist = Counter(q["ground_truth"] for q in all_queries)
    source_dist = Counter(q["source"] for q in all_queries)

    print(f"\n{'='*50}")
    print("Mixed dataset summary:")
    print(f"  Total queries: {len(all_queries)}")
    print(f"  By source:     {dict(source_dist)}")
    print(f"  By class:      {dict(dist)}")
    print(f"{'='*50}")

    dataset = {
        "metadata": {
            "description": "Mixed real-world training dataset for ModernBERT complexity classifier",
            "sources": {
                "mmlu": "cais/mmlu — academic exam questions (57 subjects)",
                "stackexchange": "sentence-transformers/stackexchange-duplicates (title-title-pair) — general SE question titles",
                "stackexchange_hpc": "sentence-transformers/stackexchange-duplicates (title-title-pair) — keyword-filtered research computing titles (HPC, scientific Python, numerical methods)",
            },
            "note": "All queries are real human-written text. No Claude-generated queries. Arena (lmsys/chatbot_arena_conversations) is kept as fixed OOD test — not included here.",
            "labeling_model": LABELING_MODEL,
            "n_queries": len(all_queries),
            "label_distribution": dict(dist),
            "source_distribution": dict(source_dist),
            "dry_run": args.dry_run,
            "balance_mode": args.balance_mode,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "queries": all_queries,
    }

    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\nSaved to {output_path}")
    print("\nNext: python scripts/eval/train_modernbert.py --eval-mode mixed-kfold \\")
    print(f"            --dataset {output_path}")


if __name__ == "__main__":
    main()
