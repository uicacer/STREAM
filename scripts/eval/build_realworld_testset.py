#!/usr/bin/env python3
"""
Build a real-world test set from LMSYS Chatbot Arena.

Downloads ~1M real user prompts from the Chatbot Arena dataset on HuggingFace,
samples a domain-stratified subset, then labels each prompt with Claude
(same rubric used to build the training set).

The resulting test set is fully independent of the training data: it contains
real user queries (not LLM-generated ones), which tests whether the classifier
generalizes beyond its own training distribution.

Usage:
    python scripts/eval/build_realworld_testset.py
    python scripts/eval/build_realworld_testset.py --n 200 --dry-run

Output:
    scripts/eval/realworld_testset.json

Requirements:
    pip install datasets anthropic
    ANTHROPIC_API_KEY must be set.

Estimated API cost:
    400 queries × ~100 tokens/query × $0.003/1K = ~$0.12 total
"""

import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

OUTPUT_PATH = Path("scripts/eval/realworld_testset.json")

# Label distribution in real prompts skews toward simple queries;
# we sample from harder-to-find categories to avoid trivial test sets.
TARGET_N = 400
MAX_TEXT_LEN = 400  # characters; longer prompts are usually multi-turn or pastes
MIN_TEXT_LEN = 15

LABELING_MODEL = "claude-sonnet-4-6"
LABEL_BATCH_SIZE = 10

COMPLEXITY_RUBRIC = """
You are a query complexity labeler for an LLM routing system.

**Complexity rubric** (based on reasoning depth, NOT question format):

LOW: Single retrievable fact or trivial computation. The answer can be stated
     in one sentence with no reasoning chain. Example: "What is the capital of France?"

MEDIUM: Apply a standard procedure or assemble 2-4 concepts. The reasoning path
        is textbook-level and well-known. Example: "Explain quicksort and its time complexity."

HIGH: Construct a novel reasoning path, formal derivation, or expert judgment.
      No standard procedure exists — the solver must build the path.
      Example: "Is P=NP? Summarize the state of evidence."

**Key rule**: Format is NOT the signal. "What is X?" can be LOW, MEDIUM, or HIGH.
              Judge by the *depth of reasoning required*, not the question format.

Label each query with exactly one of: LOW, MEDIUM, or HIGH
Return a JSON array of labels in the same order as the input queries.
Do NOT include any explanation. Only the JSON array.

Example response for 3 queries: ["LOW", "HIGH", "MEDIUM"]
"""


def load_arena_data(n_sample: int, dry_run: bool, seed: int = 42):
    """Load and filter LMSYS Chatbot Arena prompts."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: datasets package not installed. Run: pip install datasets")
        raise

    import random

    print("Loading LMSYS Chatbot Arena dataset (lmsys/chatbot_arena_conversations)...")
    print("  (This downloads ~500 MB on first run; cached afterwards)")
    ds = load_dataset("lmsys/chatbot_arena_conversations", split="train")
    print(f"  Raw dataset size: {len(ds):,} conversations")

    # Extract first user turn from each conversation
    prompts = []
    for row in ds:
        try:
            turns = row["conversation_a"]
            first_human = next((t["content"] for t in turns if t["role"] == "user"), None)
            if first_human is None:
                continue
            text = first_human.strip()
            if (
                MIN_TEXT_LEN <= len(text) <= MAX_TEXT_LEN
                and re.search(r"[a-zA-Z]", text)
                and len(re.findall(r"[a-zA-Z]", text)) > len(text) * 0.3
            ):
                prompts.append(text)
        except (KeyError, IndexError, TypeError):
            continue

    print(f"  After filtering (len {MIN_TEXT_LEN}–{MAX_TEXT_LEN}, English): {len(prompts):,}")

    if dry_run:
        n_sample = min(n_sample, 30)

    rng = random.Random(seed)
    rng.shuffle(prompts)
    sampled = prompts[:n_sample]
    print(f"  Sampled {len(sampled)} prompts for labeling")
    return sampled


def label_with_claude(texts: list[str], dry_run: bool) -> list[str]:
    """Label queries using Claude with the same rubric as the training set."""
    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic package not installed. Run: pip install anthropic")
        raise

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        raise OSError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)

    if dry_run:
        print("  [DRY RUN] Returning fake labels (no API calls)")
        import random

        return [random.choice(["LOW", "MEDIUM", "HIGH"]) for _ in texts]

    all_labels: list[str] = []
    n_batches = (len(texts) + LABEL_BATCH_SIZE - 1) // LABEL_BATCH_SIZE

    for batch_idx in range(n_batches):
        start = batch_idx * LABEL_BATCH_SIZE
        batch = texts[start : start + LABEL_BATCH_SIZE]

        prompt = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(batch))

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
                if isinstance(labels, list) and len(labels) == len(batch):
                    valid = [lb.upper() for lb in labels if lb.upper() in ("LOW", "MEDIUM", "HIGH")]
                    if len(valid) == len(batch):
                        all_labels.extend(valid)
                        break
                print(f"  [WARN] Batch {batch_idx+1}/{n_batches}: unexpected response, retrying...")
            except Exception as e:
                if attempt == 2:
                    print(f"  [WARN] Batch {batch_idx+1}/{n_batches}: failed after 3 attempts: {e}")
                    # Pad with MEDIUM as fallback (neutral label)
                    all_labels.extend(["MEDIUM"] * len(batch))
                    break
                time.sleep(2)

        print(
            f"  Batch {batch_idx+1}/{n_batches}: labeled {len(batch)} queries"
            f"  (total so far: {len(all_labels)})"
        )
        time.sleep(0.3)  # gentle rate limiting

    return all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=TARGET_N, help="Number of queries to sample")
    parser.add_argument("--dry-run", action="store_true", help="Skip API calls, use fake labels")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Download and filter Arena data
    texts = load_arena_data(args.n, args.dry_run)

    # Label with Claude
    print(f"\nLabeling {len(texts)} queries with {LABELING_MODEL}...")
    labels = label_with_claude(texts, args.dry_run)

    # Build dataset
    queries = [
        {
            "id": i + 1,
            "text": text,
            "ground_truth": label,
            "source": "lmsys/chatbot_arena_conversations",
        }
        for i, (text, label) in enumerate(zip(texts, labels, strict=True))
    ]

    dist = Counter(q["ground_truth"] for q in queries)
    print(f"\nLabel distribution: {dict(dist)}")

    dataset = {
        "metadata": {
            "description": "Real-world test set sampled from LMSYS Chatbot Arena",
            "source": "lmsys/chatbot_arena_conversations (HuggingFace)",
            "labeling_model": LABELING_MODEL,
            "labeling_method": "LLM-supervised (same rubric as training set)",
            "n_queries": len(queries),
            "label_distribution": dict(dist),
            "dry_run": args.dry_run,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "queries": queries,
    }

    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Saved {len(queries)} labeled queries to {output_path}")
    print("\nNext: python scripts/eval/eval_on_realworld.py")


if __name__ == "__main__":
    main()
