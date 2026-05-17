#!/usr/bin/env python3
"""
Generate benchmark dataset v2 for STREAM routing evaluation.

KEY CHANGE FROM V1: Uses a reasoning-depth rubric instead of a format-based
rubric. Complexity is defined by the depth of reasoning required to answer,
NOT by question format. "True or false: X" and "Is X?" and "What is X?"
appear in all three complexity classes — format is never the signal.

Target: 384 queries per (domain × class) cell → 6,912 total (from scratch)

Usage:
    python generate_benchmark_dataset_v2.py [--dry-run] [--cells N] [--check-only]

Output:
    scripts/eval/benchmark_dataset_v2.json   — full dataset
    scripts/eval/consistency_report_v2.json  — cross-model agreement report
"""

import argparse
import json
import os
import random
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_FILE = Path(__file__).parent / "benchmark_dataset_v2.json"
CONSISTENCY_FILE = Path(__file__).parent / "consistency_report_v2.json"

DOMAINS = [
    "general_knowledge",
    "science",
    "mathematics",
    "humanities",
    "computer_science",
    "research_computing",
]
CLASSES = ["LOW", "MEDIUM", "HIGH"]
TARGET_PER_CELL = 384

LABELING_MODEL = "claude-sonnet-4-6"
CONSISTENCY_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Reasoning-depth rubric
# ---------------------------------------------------------------------------
# This rubric defines complexity as an intrinsic property of the query,
# independent of which models are deployed. Swapping the local/HPC/cloud
# models does not change what these labels mean.

LABELING_CRITERIA = {
    "LOW": (
        "Answer is a single retrievable fact or trivial computation. "
        "No reasoning chain required — any knowledgeable person states the answer "
        "in one sentence with no intermediate steps. "
        "The answer is unambiguous and universally agreed upon. "
        "Examples: capitals, dates, acronyms, simple arithmetic, yes/no questions "
        "whose answer is a single settled fact."
    ),
    "MEDIUM": (
        "Answer requires applying a known, established procedure or assembling "
        "2–4 related concepts. The reasoning path is standard (textbook-level) — "
        "it exists and is well-defined, so the answerer follows it rather than "
        "inventing it. Multi-step explanations, implementations of known algorithms, "
        "comparisons of well-understood concepts, how-to procedures."
    ),
    "HIGH": (
        "Answer requires constructing a novel reasoning path, formal derivation, "
        "cross-domain synthesis, or expert judgment. The path to the answer must be "
        "built, not retrieved or assembled from textbook steps. Reasonable experts "
        "could approach this differently. Includes formal proofs, open research "
        "questions, system design with contested tradeoffs, and deep analysis where "
        "no single correct answer exists."
    ),
}

# ---------------------------------------------------------------------------
# Few-shot examples — carefully chosen to illustrate reasoning-depth rubric.
# CRITICAL: Each example set includes multiple question formats across the
# complexity levels to demonstrate that FORMAT IS NOT THE SIGNAL.
# ---------------------------------------------------------------------------

EXAMPLES = {
    # ── computer_science ────────────────────────────────────────────────────
    ("computer_science", "LOW"): [
        "What does CPU stand for?",
        "Is Python an interpreted language?",  # yes/no, single fact
        "True or false: a linked list provides O(1) random access.",  # T/F, single fact
        "What is the time complexity of binary search?",
        "How many bits are in a byte?",
        "What does HTTP stand for?",
        "Is SQL a programming language or a query language?",  # yes/no, factual
    ],
    ("computer_science", "MEDIUM"): [
        "Explain how quicksort works and analyze its average and worst-case time complexity.",
        "Implement a binary search tree in Python with insert, search, and delete.",
        "Compare TCP and UDP — when would you choose each?",
        "How does garbage collection work in the JVM?",
        "Is a hash table always faster than a binary search tree? Explain when each is preferable.",  # yes/no but requires reasoning
        "Walk me through how a compiler translates source code to machine code.",
    ],
    ("computer_science", "HIGH"): [
        "Prove the Ω(n log n) lower bound for comparison-based sorting.",
        "Design a fault-tolerant distributed database providing strong consistency and horizontal scalability.",
        "Is P equal to NP? Present the current state of evidence and why this is hard.",  # yes/no but open expert question
        "Critically analyze the CAP theorem — is it a fundamental limit or an oversimplification?",
        "What are the open problems in practical Byzantine fault tolerance for blockchain systems?",
    ],
    # ── general_knowledge ───────────────────────────────────────────────────
    ("general_knowledge", "LOW"): [
        "What is the capital of France?",
        "True or false: the Great Wall of China is visible from space.",  # T/F, single fact
        "Who wrote Romeo and Juliet?",
        "Is Australia both a country and a continent?",  # yes/no, single fact
        "How many continents are there?",
        "What year did the Berlin Wall fall?",
        "Is gold a metal?",
    ],
    ("general_knowledge", "MEDIUM"): [
        "Summarize the main causes of World War I and explain how they led to the conflict.",
        "Compare the parliamentary and presidential systems of government.",
        "How does the United Nations Security Council work, and what are its limitations?",
        "Is democracy the most stable form of government? Explain the arguments on both sides.",  # yes/no but requires analysis
        "Explain how the global financial crisis of 2008 unfolded and what triggered it.",
    ],
    ("general_knowledge", "HIGH"): [
        "Analyze the long-term geopolitical consequences of World War II on the current international order.",
        "Is Western liberal democracy a universal end-state for political development, or a culturally specific form? Evaluate the evidence.",  # yes/no but deep contested question
        "Compare the philosophical foundations of realism, liberalism, and constructivism in international relations.",
        "What explains the democratic backsliding observed in multiple countries since 2010? Synthesize competing explanations.",
    ],
    # ── humanities ──────────────────────────────────────────────────────────
    ("humanities", "LOW"): [
        "Who wrote Hamlet?",
        "True or false: the Iliad was written by Homer.",
        "Is the Mona Lisa painted in oil or watercolor?",  # yes/no, single fact
        "What language did Dante write the Divine Comedy in?",
        "Is existentialism a branch of philosophy?",  # yes/no, definitional
        "What century did the Renaissance begin?",
    ],
    ("humanities", "MEDIUM"): [
        "Analyze the major themes in Shakespeare's Hamlet.",
        "Explain the social contract as developed by Hobbes, Locke, and Rousseau.",
        "Compare the Romantic and Enlightenment movements in European literature.",
        "Is Nietzsche's concept of the Übermensch compatible with democratic values? Explain both positions.",  # yes/no but requires textual analysis
        "Describe how postcolonial theory critiques traditional literary canon formation.",
    ],
    ("humanities", "HIGH"): [
        "Critically analyze Rawls' theory of justice and how Nozick's entitlement theory challenges the difference principle.",
        "Is moral relativism self-defeating? Construct and evaluate the strongest case for and against it.",  # yes/no but requires original philosophical reasoning
        "Compare Nietzsche's critique of morality with Kant's categorical imperative and assess whether a synthesis is possible.",
        "What are the unresolved tensions in feminist theory between equality-based and difference-based frameworks?",
    ],
    # ── mathematics ─────────────────────────────────────────────────────────
    ("mathematics", "LOW"): [
        "What is 2 + 2?",
        "True or false: a right angle is 90 degrees.",
        "Is zero an even number?",  # yes/no, single fact
        "What is the formula for the area of a circle?",
        "How many sides does a hexagon have?",
        "What is 15% of 200?",
        "Is π rational or irrational?",  # yes/no, single fact
    ],
    ("mathematics", "MEDIUM"): [
        "Solve the quadratic equation x² − 5x + 6 = 0 and explain the method.",
        "Explain how to find the derivative of f(x) = x³ + 2x² − 5x + 1.",
        "Walk through the proof that the sum of angles in a triangle is 180°.",
        "Is the geometric series ∑(1/2)^n convergent? Show why or why not.",  # yes/no but requires standard proof
        "Compare the mean, median, and mode — when is each the better measure of central tendency?",
    ],
    ("mathematics", "HIGH"): [
        "Prove that the set of real numbers is uncountable using Cantor's diagonal argument.",
        "Derive the Euler-Lagrange equations from Hamilton's principle of least action.",
        "Is the Riemann Hypothesis likely true? Present the heuristic and numerical evidence and explain why a proof remains elusive.",  # yes/no but open expert question
        "Prove that there are infinitely many primes using at least two distinct proof strategies.",
        "What are the deepest open problems in algebraic topology and why have they resisted resolution?",
    ],
    # ── research_computing ──────────────────────────────────────────────────
    ("research_computing", "LOW"): [
        "What does HPC stand for?",
        "True or false: SLURM is a job scheduler.",
        "Is MPI used for parallel computing?",  # yes/no, single fact
        "What does GPU stand for?",
        "Is Globus Compute a job scheduler?",  # yes/no, factual — it is NOT; it's a federated function execution service
        "What is a supercomputer?",
        "What file system is commonly used on HPC clusters?",
    ],
    ("research_computing", "MEDIUM"): [
        "Explain how vLLM achieves high throughput for LLM serving using PagedAttention.",
        "How does Globus Compute authenticate users and dispatch tasks to HPC endpoints?",
        "Compare running an LLM inference workload via SLURM batch job vs a persistent vLLM server.",
        "Walk through the steps of submitting a multi-node MPI job on a SLURM cluster.",
        "Is AWQ quantization always better than GPTQ for inference throughput? Explain the tradeoffs.",  # yes/no but requires technical reasoning
    ],
    ("research_computing", "HIGH"): [
        "Design a multi-tier LLM inference system for a research university. Analyze latency, cost, and quality tradeoffs.",
        "Is WebSocket relay a better approach than Globus Streams for interactive HPC token streaming? Build the comparison rigorously.",  # yes/no but requires deep analysis
        "Critically evaluate the architectural tradeoffs between federated function execution (Globus Compute) and traditional HPC job schedulers for interactive AI workloads.",
        "What are the fundamental limits of streaming LLM tokens from HPC clusters through institutional firewalls, and what approaches address them?",
    ],
    # ── science ─────────────────────────────────────────────────────────────
    ("science", "LOW"): [
        "What is the speed of light?",
        "True or false: sound travels faster than light.",
        "Is water a compound or an element?",  # yes/no, single fact
        "What planet is closest to the Sun?",
        "How many bones are in the adult human body?",
        "Is the moon a natural satellite or a planet?",  # yes/no, single fact
        "What is DNA?",
    ],
    ("science", "MEDIUM"): [
        "Explain how photosynthesis converts light energy into chemical energy.",
        "Describe the process of DNA replication and why it is important for cell division.",
        "How does CRISPR-Cas9 work as a genome editing tool?",
        "Is antibiotic resistance an evolutionary phenomenon? Explain the mechanism.",  # yes/no but requires mechanism explanation
        "Compare nuclear fission and fusion — why is fusion harder to achieve practically?",
    ],
    ("science", "HIGH"): [
        "Derive the Michaelis-Menten equation for enzyme kinetics from first principles.",
        "Is string theory a scientific theory in the Popperian sense? Evaluate the arguments.",  # yes/no but requires deep analysis
        "Analyze the molecular mechanisms underlying antibiotic resistance and how horizontal gene transfer drives resistance evolution.",
        "What are the unresolved questions in the interpretation of quantum mechanics, and why has no interpretation been definitively confirmed?",
        "Derive the equations governing population dynamics (logistic and exponential) and analyze their stability.",
    ],
}


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def build_single_query_prompt(domain: str, complexity: str, existing_texts: list[str]) -> str:
    """Targeted prompt for when only 1 slot remains — maximizes uniqueness."""
    criteria = LABELING_CRITERIA[complexity]
    existing_block = "\n".join(f"  - {t}" for t in existing_texts)
    return f"""You are generating ONE unique query for a benchmark dataset.

Domain: {domain}
Complexity: {complexity}
Definition: {criteria}

The following {len(existing_texts)} queries already exist and MUST NOT be duplicated or paraphrased:
{existing_block}

Generate exactly ONE new query that:
1. Is clearly {complexity} complexity by reasoning depth (not format)
2. Is genuinely different in topic and phrasing from every query listed above
3. Is a natural user question or instruction

Return ONLY the query string, no quotes, no explanation."""


def build_generation_prompt(
    domain: str,
    complexity: str,
    existing_texts: list[str],
    batch_size: int,
    show_all_existing: bool = False,
) -> str:
    examples = EXAMPLES.get((domain, complexity), [])
    if show_all_existing:
        existing_sample = existing_texts
    else:
        existing_sample = random.sample(existing_texts, min(5, len(existing_texts)))

    criteria_block = "\n".join(f"  {k}: {v}" for k, v in LABELING_CRITERIA.items())
    example_block = "\n".join(f"  - {ex}" for ex in examples)
    existing_block = (
        "\n".join(f"  - {t}" for t in existing_sample) if existing_sample else "  (none yet)"
    )

    # Format diversity requirements vary by class but apply to ALL classes.
    # The key constraint: the SAME format (yes/no, true/false, what-is) must
    # appear in LOW, MEDIUM, and HIGH — complexity is never determined by format.
    format_guidance = {
        "LOW": (
            "Format variety: include factual, yes/no, true/false, fill-in-the-blank, "
            "definition, conversion, and name/identify questions. "
            "CRITICAL: every query must have a SINGLE, UNIVERSALLY AGREED answer "
            "statable in one sentence — that is what makes it LOW, not its format."
        ),
        "MEDIUM": (
            "Format variety: include explain, implement, compare, how-to, step-by-step, "
            "and yes/no questions whose answer REQUIRES multi-step reasoning to justify. "
            "Example of MEDIUM yes/no: 'Is a hash table always faster than a BST?' — "
            "the answer requires explaining conditions, not just saying yes or no. "
            "CRITICAL: the reasoning path must be established (textbook-level), not invented."
        ),
        "HIGH": (
            "Format variety: include derive, prove, design, analyze, critique, synthesize, "
            "and yes/no or true/false questions that are genuinely OPEN or CONTESTED. "
            "Example of HIGH yes/no: 'Is P=NP?' or 'Is string theory falsifiable?' — "
            "the answer requires constructing an expert position, not retrieving a fact. "
            "CRITICAL: the reasoning path must be BUILT, not retrieved — that is what makes it HIGH."
        ),
    }[complexity]

    return f"""You are generating a benchmark dataset for evaluating AI routing systems.

TASK: Generate exactly {batch_size} diverse user queries for domain="{domain}" at complexity="{complexity}".

COMPLEXITY RUBRIC (read carefully — this is reasoning-depth based, NOT format-based):
{criteria_block}

THE MOST IMPORTANT RULE:
Question FORMAT does not determine complexity class. The SAME format appears in all three classes:
  - "True or false: X" can be LOW (single fact), MEDIUM (requires reasoning), or HIGH (open/contested)
  - "Is X?" can be LOW (yes/no, single fact), MEDIUM (yes, but explain why with multi-step reasoning), or HIGH (yes/no, but genuinely contested or requires expert judgment)
  - "What is X?" can be LOW (one-sentence definition), MEDIUM (requires multi-step explanation), or HIGH (no settled answer)
You are generating "{complexity}" queries — make sure the REASONING DEPTH matches, regardless of the surface format.

{format_guidance}

Domain guidance for "{domain}":
  - general_knowledge: geography, history, culture, everyday facts, world affairs
  - science: biology, chemistry, physics, earth science, astronomy
  - mathematics: arithmetic, algebra, calculus, statistics, proofs, number theory
  - humanities: literature, philosophy, ethics, linguistics, art history, political theory
  - computer_science: programming, algorithms, systems, networks, databases, AI/ML
  - research_computing: HPC, scientific workflows, LLM deployment, Globus, vLLM, SLURM, MPI

Few-shot examples of "{domain}/{complexity}" queries (these set the calibration bar):
{example_block}

Existing queries to AVOID duplicating:
{existing_block}

OUTPUT REQUIREMENTS:
1. Return ONLY a JSON array of {batch_size} query strings — no other text, no markdown
2. Every query must be a natural user question or instruction
3. Queries must be diverse — different subtopics, phrasing styles, question formats
4. Each query must clearly match complexity="{complexity}" by reasoning depth, not by format
5. Do NOT generate only one format type — vary across the format types listed above

Format: ["Query 1 here.", "Query 2 here.", ...]"""


def build_labeling_prompt(queries: list[str]) -> str:
    criteria_block = "\n".join(f"  {k}: {v}" for k, v in LABELING_CRITERIA.items())
    queries_block = "\n".join(f"{i+1}. {q}" for i, q in enumerate(queries))

    return f"""Classify each query by REASONING DEPTH required to answer it.
This is NOT about question format — "Is X?" and "True or false: X" can be LOW, MEDIUM, or HIGH.

Complexity levels:
{criteria_block}

THE KEY TEST: Does answering require (a) retrieving a single fact [LOW], (b) following an established multi-step procedure [MEDIUM], or (c) constructing a novel reasoning path or expert judgment [HIGH]?

Return ONLY a JSON array of labels, one per query in order.
Example for 3 queries: ["LOW", "HIGH", "MEDIUM"]

Queries:
{queries_block}"""


# ---------------------------------------------------------------------------
# API helpers (unchanged from v1)
# ---------------------------------------------------------------------------


def call_anthropic(prompt: str, model: str = LABELING_MODEL, max_tokens: int = 4096) -> str:
    """
    Call Claude Sonnet with extended thinking enabled.

    Extended thinking (budget_tokens=10000) lets the model reason carefully
    about the reasoning-depth rubric before committing to each query and label.
    This is the key step that makes knowledge distillation meaningful: we are
    capturing frontier-quality judgment, not fast pattern-matching.

    The response contains two blocks: a thinking block (internal reasoning,
    discarded) and a text block (the JSON array we use).
    """
    import anthropic

    client = anthropic.Anthropic(timeout=120.0)  # fail fast if API hangs
    msg = client.messages.create(
        model=model,
        max_tokens=16000,  # must be > budget_tokens
        thinking={
            "type": "enabled",
            "budget_tokens": 10000,  # reasoning tokens per batch
        },
        messages=[{"role": "user", "content": prompt}],
    )
    # Extract the text block (skip the thinking block)
    for block in msg.content:
        if block.type == "text":
            return block.text.strip()
    return msg.content[-1].text.strip()


def call_openai(prompt: str, model: str = CONSISTENCY_MODEL, max_tokens: int = 2048) -> str:
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


def safe_json_parse(text: str) -> list | None:
    import re

    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            for v in result.values():
                if isinstance(v, list):
                    return v
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
# Generation
# ---------------------------------------------------------------------------


def generate_queries_for_cell(
    domain: str,
    complexity: str,
    existing_texts: list[str],
    n_needed: int,
    dry_run: bool = False,
    batch_size: int = 30,
) -> list[dict]:
    generated = []
    existing_set = {t.lower().strip() for t in existing_texts}
    consecutive_empty = 0

    print(f"  Generating {n_needed} queries for {domain}/{complexity}...")

    while len(generated) < n_needed:
        remaining = n_needed - len(generated)
        this_batch = min(batch_size, remaining)
        # When close to the target, show ALL existing queries so Claude can avoid them
        show_all = remaining <= batch_size * 2
        single_mode = remaining <= 5  # one-at-a-time for the last few slots
        if dry_run:
            for _ in range(this_batch):
                generated.append(
                    {
                        "text": f"[DRY RUN] {domain} {complexity} query #{len(generated)+1}",
                        "domain": domain,
                        "ground_truth": complexity,
                        "generated_by": LABELING_MODEL,
                        "generated_at": utcnow(),
                    }
                )
            break

        all_existing = existing_texts + [q["text"] for q in generated]
        if single_mode:
            prompt = build_single_query_prompt(domain, complexity, all_existing)
        else:
            prompt = build_generation_prompt(
                domain, complexity, all_existing, this_batch, show_all_existing=show_all
            )
        try:
            response = call_anthropic(prompt)

            if single_mode:
                # Response is a plain string, not a JSON array
                text = response.strip().strip('"').strip("'")
                batch = [text] if text and len(text) >= 10 else []
            else:
                batch = safe_json_parse(response)

            if not single_mode and not isinstance(batch, list):
                print("    [WARN] Could not parse response, retrying...")
                time.sleep(2)
                continue

            new_this_batch = 0
            for text in batch:
                if not isinstance(text, str) or len(text) < 10:
                    continue
                if text.lower().strip() in existing_set:
                    continue
                existing_set.add(text.lower().strip())
                generated.append(
                    {
                        "text": text.strip(),
                        "domain": domain,
                        "ground_truth": complexity,
                        "generated_by": LABELING_MODEL,
                        "generated_at": utcnow(),
                    }
                )
                new_this_batch += 1
                if len(generated) >= n_needed:
                    break

            print(f"    Progress: {len(generated)}/{n_needed}")
            time.sleep(0.5)

            if new_this_batch == 0:
                consecutive_empty += 1
                if consecutive_empty >= 20:
                    print(
                        f"    [WARN] Stopping at {len(generated)}/{n_needed} after 20 consecutive empty batches — truly exhausted"
                    )
                    break
            else:
                consecutive_empty = 0

        except Exception as e:
            print(f"    [ERROR] {e}, retrying in 5s...")
            time.sleep(5)

    return generated[:n_needed]


# ---------------------------------------------------------------------------
# Format-decoupling validation
# ---------------------------------------------------------------------------


def validate_format_decoupling(queries: list[dict]) -> None:
    """
    Warn if any format type is concentrated in a single complexity class.
    This detects if the format-proxy bias crept back in.
    """
    import re

    print("\nFormat-decoupling validation:")

    # Detect yes/no / true-false questions
    yes_no = [
        q
        for q in queries
        if re.match(
            r"^(is |are |does |do |was |were |can |could |true or false)", q["text"].lower()
        )
    ]
    what_is = [q for q in queries if re.match(r"^what is ", q["text"].lower())]

    for label, subset in [("yes/no or T/F", yes_no), ("What is X?", what_is)]:
        dist = Counter(q["ground_truth"] for q in subset)
        total = len(subset)
        if total == 0:
            continue
        print(
            f"  {label} ({total} queries): LOW={dist['LOW']} MEDIUM={dist['MEDIUM']} HIGH={dist['HIGH']}"
        )
        for cls in ["LOW", "MEDIUM", "HIGH"]:
            pct = dist[cls] / total * 100
            if pct > 90:
                print(
                    f"  ⚠️  WARNING: {pct:.0f}% of '{label}' queries are {cls} — format-proxy bias detected!"
                )
            elif pct == 0:
                print(f"  ⚠️  WARNING: zero '{label}' queries in {cls} — format not decoupled!")


# ---------------------------------------------------------------------------
# Consistency check (unchanged from v1)
# ---------------------------------------------------------------------------


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    assert len(labels_a) == len(labels_b)
    n = len(labels_a)
    classes = sorted(set(labels_a) | set(labels_b))
    po = sum(a == b for a, b in zip(labels_a, labels_b, strict=False)) / n
    count_a = Counter(labels_a)
    count_b = Counter(labels_b)
    pe = sum((count_a[c] / n) * (count_b[c] / n) for c in classes)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def run_consistency_check(
    all_queries: list[dict], sample_frac: float = 0.05, dry_run: bool = False
) -> dict:
    sample_n = max(50, int(len(all_queries) * sample_frac))
    sample = random.sample(all_queries, sample_n)
    texts = [q["text"] for q in sample]
    original_labels = [q["ground_truth"] for q in sample]

    print(f"\nConsistency check: re-labeling {sample_n} queries with {CONSISTENCY_MODEL}...")

    if dry_run:
        check_labels = [
            lbl if random.random() > 0.1 else random.choice(CLASSES) for lbl in original_labels
        ]
    else:
        check_labels = []
        batch_size = 20
        n_batches = (len(texts) + batch_size - 1) // batch_size
        _norm = {"low": "LOW", "medium": "MEDIUM", "med": "MEDIUM", "high": "HIGH"}
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_idx = i // batch_size
            try:
                response = call_openai(build_labeling_prompt(batch_texts))
                batch_labels = safe_json_parse(response)
                if not isinstance(batch_labels, list):
                    print(
                        f"  [WARN] Batch {batch_idx+1}/{n_batches}: parse failed, marking UNKNOWN"
                    )
                    check_labels.extend(["UNKNOWN"] * len(batch_texts))
                    continue
                batch_labels = [
                    _norm.get(str(lbl).strip().lower(), str(lbl).strip().upper())
                    for lbl in batch_labels
                ]
                if len(batch_labels) != len(batch_texts):
                    print(f"  [WARN] Batch {batch_idx+1}/{n_batches}: length mismatch, padding")
                    if len(batch_labels) < len(batch_texts):
                        batch_labels.extend(["UNKNOWN"] * (len(batch_texts) - len(batch_labels)))
                    else:
                        batch_labels = batch_labels[: len(batch_texts)]
                check_labels.extend(batch_labels)
                print(
                    f"  Batch {batch_idx+1}/{n_batches}: {sum(lbl in CLASSES for lbl in batch_labels)}/{len(batch_texts)} valid"
                )
            except Exception as e:
                print(f"  [ERROR] Batch {batch_idx+1}/{n_batches}: {e}")
                check_labels.extend(["UNKNOWN"] * len(batch_texts))
            time.sleep(0.3)

    valid = [(o, c) for o, c in zip(original_labels, check_labels, strict=False) if c in CLASSES]
    if not valid:
        return {"error": "No valid labels returned"}

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
# Save helper
# ---------------------------------------------------------------------------


def _save_output(queries: list[dict], target: int) -> None:
    for i, q in enumerate(queries, start=1):
        q.setdefault("id", i)

    output = {
        "_description": (
            f"STREAM routing benchmark dataset v2 (reasoning-depth rubric). "
            f"{len(queries)} queries across {len(DOMAINS)} domains × 3 complexity levels. "
            f"Target: {target} per cell for ±5pp 95% CIs."
        ),
        "_rubric_version": "v2",
        "_labeling_criteria": LABELING_CRITERIA,
        "_rubric_note": (
            "Complexity is defined by reasoning depth, NOT question format. "
            "'Is X?' and 'True or false: X' appear in all three classes. "
            "LOW = single retrievable fact. MEDIUM = established procedure. "
            "HIGH = construct novel reasoning path or expert judgment."
        ),
        "_domains": DOMAINS,
        "_labeling_model": LABELING_MODEL,
        "_labeling_method": (
            f"All queries generated from scratch by {LABELING_MODEL} using the v2 "
            "reasoning-depth rubric with explicit format-decoupling instructions. "
            "Author validation (252-query stratified sample, blind) and cross-model "
            "consistency check (GPT-4o-mini) provide label quality bounds."
        ),
        "_stats": {
            "total": len(queries),
            "domains": len(DOMAINS),
            "classes": 3,
            "target_per_cell": target,
            "generated_at": utcnow(),
        },
        "queries": queries,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate STREAM benchmark dataset v2")
    parser.add_argument("--dry-run", action="store_true", help="No API calls — fake data only")
    parser.add_argument(
        "--cells", type=int, default=None, help="Limit to first N cells (for testing)"
    )
    parser.add_argument(
        "--check-only", action="store_true", help="Only run consistency check on existing output"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run format-decoupling validation on existing output",
    )
    parser.add_argument("--target", type=int, default=TARGET_PER_CELL)
    args = parser.parse_args()

    if (
        not args.dry_run
        and not args.check_only
        and not args.validate_only
        and not os.environ.get("ANTHROPIC_API_KEY")
    ):
        raise OSError("ANTHROPIC_API_KEY not set")

    # Load existing partial output to resume, if any
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            existing_data = json.load(f)
        all_queries = existing_data["queries"]
        all_queries = [q for q in all_queries if "[DRY RUN]" not in q.get("text", "")]
        print(f"Resuming from {OUTPUT_FILE.name}: {len(all_queries)} queries already generated")
    else:
        all_queries = []
        print("Starting fresh — generating all queries from scratch")

    if args.validate_only:
        validate_format_decoupling(all_queries)
        return

    if args.check_only:
        if not all_queries:
            print("No queries found.")
            return
        report = run_consistency_check(all_queries, dry_run=args.dry_run)
        with open(CONSISTENCY_FILE, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Consistency report saved to {CONSISTENCY_FILE}")
        return

    # Build cell lookup for resume support
    cell_texts: dict[tuple, list[str]] = defaultdict(list)
    for q in all_queries:
        cell_texts[(q["domain"], q["ground_truth"])].append(q["text"])

    all_cells = [(d, c) for d in DOMAINS for c in CLASSES]
    if args.cells:
        all_cells = all_cells[: args.cells]

    for domain, complexity in all_cells:
        existing_in_cell = cell_texts[(domain, complexity)]
        n_have = len(existing_in_cell)
        n_needed = max(0, args.target - n_have)

        if n_needed == 0:
            print(f"  {domain}/{complexity}: already have {n_have} ≥ {args.target}, skipping")
            continue

        generated = generate_queries_for_cell(
            domain,
            complexity,
            existing_in_cell,
            n_needed,
            dry_run=args.dry_run,
        )
        all_queries.extend(generated)
        cell_texts[(domain, complexity)].extend(q["text"] for q in generated)

        _save_output(all_queries, args.target)
        print(f"    [SAVED] checkpoint written ({len(all_queries)} total)")

    print(f"\nGeneration complete: {len(all_queries)} queries total")

    # Distribution check
    dist = Counter((q["domain"], q["ground_truth"]) for q in all_queries)
    print("\nFinal distribution:")
    for key in sorted(dist.keys()):
        flag = " ✓" if dist[key] >= args.target else f" ← only {dist[key]}"
        print(f"  {key[0]:30s} {key[1]:8s} {dist[key]}{flag}")

    # Format-decoupling validation
    validate_format_decoupling(all_queries)

    # Consistency check
    if os.environ.get("OPENAI_API_KEY") or args.dry_run:
        report = run_consistency_check(all_queries, dry_run=args.dry_run)
        with open(CONSISTENCY_FILE, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Consistency report → {CONSISTENCY_FILE}")
    else:
        print("\n[SKIPPED] Consistency check — set OPENAI_API_KEY to enable")

    print("\nDone.")
    print(f"  Dataset:     {OUTPUT_FILE}")
    print(f"  Consistency: {CONSISTENCY_FILE}")
    print("\nNext steps:")
    print("  1. python scripts/eval/sample_for_validation.py  (uses v2 dataset automatically)")
    print("  2. Label validation_sample.csv (252 queries, ~2 hours)")
    print("  3. python scripts/eval/compute_validation_kappa.py")
    print("  4. python scripts/eval/train_modernbert.py")


if __name__ == "__main__":
    main()
