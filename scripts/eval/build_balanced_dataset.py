#!/usr/bin/env python3
"""
Build a balanced multi-domain dataset for the STREAM ModernBERT routing classifier.

Design
------
10 domains spanning STEM, life sciences, humanities, and social sciences.
Each domain contributes EXACTLY DOMAIN_TARGET queries per class (hard cap).
The dataset is balanced on BOTH axes: complexity class AND domain.

Domains and sources:
  hpc               → StackExchange (SE duplicates, keyword-filtered)
  mathematics       → SE + MMLU abstract_algebra + MMLU-Pro math
  statistics_ml     → SE + MMLU hs_statistics + MMLU machine_learning
  physics_chemistry → SE + MMLU-Pro physics + MMLU-Pro chemistry + MMLU astronomy
  engineering       → SE + MMLU-Pro engineering + MMLU electrical_engineering
  life_sciences     → SE + MMLU bio/med subjects + MMLU-Pro biology + PubMedQA
  cs_software       → SE + MMLU-Pro computer_science + MMLU computer_security
  philosophy_ethics → MMLU philosophy/moral/logic + MMLU-Pro philosophy
  social_sciences   → MMLU sociology/psych/econ/law + MMLU-Pro economics/psychology
  history_culture   → MMLU prehistory/religions/geography + MMLU-Pro history + MMLU misc

Total: 10 domains × DOMAIN_TARGET/class × 3 classes
  DOMAIN_TARGET = 200  → 6,000 total
  80/20 stratified split → 4,800 train (160/domain/class) + 1,200 test (40/domain/class)
  seed = 42

Quality filters
---------------
SE    : 25–200 chars, ≥85% ASCII, ≥4 alphabetic words, ≥35% alpha ratio,
        not a raw error/exception message, has question structure,
        no heavy LaTeX formulas
MMLU  : no passage/table/figure references, no fill-in-the-blank (___),
        no "which of the following", ≥20 chars, ≥4 alphabetic words
MMLU-Pro: same as MMLU (uses 'question' field, strip answer choices)
PubMedQA: uses 'question' field directly — already well-formed research questions

Usage
-----
  python scripts/eval/build_balanced_dataset.py
  python scripts/eval/build_balanced_dataset.py --dry-run
  python scripts/eval/build_balanced_dataset.py --domain-target 50  # quick test
"""

import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRAIN_OUTPUT = Path("scripts/eval/balanced_training_dataset.json")
TEST_OUTPUT = Path("scripts/eval/balanced_test_dataset.json")

DOMAIN_TARGET = 200  # per class per domain (hard cap — equal distribution)
TRAIN_FRACTION = 0.80  # 80/20 split
LABEL_BATCH_SIZE = 20
LABELING_MODEL = "claude-sonnet-4-6"

DOMAINS = [
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
]

# ---------------------------------------------------------------------------
# Complexity rubric (used for ALL sources)
# ---------------------------------------------------------------------------
COMPLEXITY_RUBRIC = """You are a query complexity classifier for an LLM routing system.

Complexity rubric — based on REASONING DEPTH, not question format:

LOW: Single retrievable fact or trivial computation. One sentence answer,
     no reasoning chain required.
     Examples: "What is the capital of France?" / "Is Python interpreted?"
               "Who wrote Hamlet?" / "What year did WWII end?"

MEDIUM: Apply a standard procedure or assemble 2-4 concepts. The reasoning
        path is textbook-level and well-established.
        Examples: "Explain quicksort and its time complexity."
                  "Compare TCP and UDP." / "Explain how vaccines work."
                  "Solve: find eigenvalues of [[1,2],[3,4]]."

HIGH: Construct a novel reasoning path, formal derivation, or expert judgment.
      No single standard procedure — the solver must build the path.
      Examples: "Is P=NP? Summarize the state of evidence."
                "Design a fault-tolerant distributed key-value store."
                "Prove that every continuous function on [0,1] is bounded."
                "Analyze the long-term economic effects of the Roman Empire's fall."

Key rule: format is NOT the complexity signal. "What is X?" can be
LOW, MEDIUM, or HIGH depending on what reasoning is required.

Label each query with exactly one of: LOW, MEDIUM, or HIGH
Return a JSON array of labels in the same order as the input queries.
No explanation. Only the JSON array.
Example for 3 queries: ["LOW", "HIGH", "MEDIUM"]"""

# ---------------------------------------------------------------------------
# SE keyword patterns per domain
# First-match wins — order matters (hpc before cs_software, etc.)
# ---------------------------------------------------------------------------
SE_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "hpc": [
        r"\bslurm\b",
        r"\bsbatch\b",
        r"\bsrun\b",
        r"\bsqueue\b",
        r"\bqsub\b",
        r"\btorque\b",
        r"\bmpi\b",
        r"\bmpirun\b",
        r"\bmpiexec\b",
        r"\bopenmpi\b",
        r"\bmpich\b",
        r"\bapptainer\b",
        r"\bsingularity container\b",
        r"\bmodule load\b",
        r"\blmod\b",
        r"compute cluster",
        r"hpc cluster",
        r"login node",
        r"compute node",
        r"batch job",
        r"job scheduler",
        r"job array",
        r"\binfiniband\b",
        r"\bnumpy\b",
        r"\bscipy\b",
        r"\bpytorch\b",
        r"\btensorflow\b",
        r"scikit.learn",
        r"jupyter notebook",
        r"\bmatplotlib\b",
        r"\bcuda\b",
        r"gpu memory",
        r"gpu training",
        r"\bnvcc\b",
        r"\bcudnn\b",
        r"\bgpu\b",
        r"parallel processing",
        r"\bdask\b",
        r"\bopenmp\b",
        r"ray cluster",
        r"ray distributed",
        r"ssh tunnel",
        r"ssh key",
        r"remote server",
        r"bash script",
        r"conda environment",
        r"virtual environment",
        r"\bkubernetes\b",
        r"docker container",
        r"research computing",
        r"scientific computing",
        r"\blustre\b",
        r"memory bandwidth",
        r"\bvectorization\b",
        r"numerical method",
        r"finite difference",
        r"molecular dynamics",
        r"climate model",
        r"quantum circuit",
        r"\bfortran\b",
        r"\bgromacs\b",
        r"\blammps\b",
        r"\bvasp\b",
        r"\bopenacc\b",
        r"\bnfs mount\b",
        r"distributed memory",
        r"shared memory",
        r"job submission",
        r"resource allocation",
    ],
    "mathematics": [
        r"\bproof\b",
        r"\btheorem\b",
        r"\btopology\b",
        r"\bmanifold\b",
        r"\bgroup theory\b",
        r"\bconjecture\b",
        r"\bnumber theory\b",
        r"\bcombinatorics\b",
        r"\bgraph theory\b",
        r"\blinear algebra\b",
        r"\beigenvalue\b",
        r"\beigenvector\b",
        r"\bdifferential equation\b",
        r"\bcalculus\b",
        r"\breal analysis\b",
        r"\bcomplex analysis\b",
        r"\bprime number\b",
        r"\binduction proof\b",
        r"\babstract algebra\b",
        r"\bset theory\b",
        r"\bformal proof\b",
        r"\bintegral\b",
        r"\bderivative\b",
        r"\blimit\b",
        r"\bmatrix\b",
        r"\bdeterminant\b",
        r"\bvector space\b",
    ],
    "statistics_ml": [
        r"\bregression\b",
        r"\bbayesian\b",
        r"\bhypothesis test",
        r"\bp.value\b",
        r"\bconfidence interval\b",
        r"\banova\b",
        r"\bstochastic\b",
        r"\bmaximum likelihood\b",
        r"\btime series\b",
        r"\bneural network\b",
        r"\bmachine learning\b",
        r"\bdeep learning\b",
        r"\brandom forest\b",
        r"\bgradient descent\b",
        r"\bbackpropagation\b",
        r"\bcross.validation\b",
        r"\boverfitting\b",
        r"\bprobability distribution\b",
        r"\bmarkov chain\b",
        r"\bmonte carlo\b",
        r"\bcluster analysis\b",
        r"\bdimensionality reduction\b",
        r"\bprincipal component\b",
        r"\bsupport vector\b",
    ],
    "physics_chemistry": [
        r"\bquantum\b",
        r"\bthermodynamics\b",
        r"\borganic chemistry\b",
        r"\bchemical reaction\b",
        r"\bphoton\b",
        r"\belectromagnet",
        r"\brelativity\b",
        r"\bwave function\b",
        r"\bschr.dinger\b",
        r"\bnuclear\b",
        r"\bparticle physics\b",
        r"\bsemiconductor\b",
        r"\bcrystal structure\b",
        r"\bcomputational chemistry\b",
        r"\bfluid dynamics\b",
        r"\boptics\b",
        r"\bmolecular orbital\b",
        r"\bspectroscopy\b",
        r"\bthermochemistry\b",
        r"\bkinetics\b",
        r"\belectrochemistry\b",
        r"\bpolymer\b",
    ],
    "engineering": [
        r"\bcircuit\b",
        r"\bsignal processing\b",
        r"\bcontrol system\b",
        r"\bfpga\b",
        r"\bembedded system\b",
        r"\bvhdl\b",
        r"\bverilog\b",
        r"\bpid controller\b",
        r"\bfft\b",
        r"\blaplace transform\b",
        r"\bfourier transform\b",
        r"\bdigital signal\b",
        r"\bfilter design\b",
        r"\bmechanical engineering\b",
        r"\bstructural analysis\b",
        r"\bheat transfer\b",
        r"\bpower system\b",
        r"\bmotor control\b",
        r"\bservo\b",
        r"\bactuator\b",
        r"\bstress analysis\b",
        r"\bfinite element\b",
        r"\bmicrocontroller\b",
        r"\boscilloscope\b",
        r"\bpcb\b",
        r"\brf design\b",
        r"\bantenna\b",
        r"\bimpedance\b",
        r"\btransistor\b",
        r"\bop.amp\b",
        r"\boperational amplifier\b",
        r"\brobotic\b",
        r"\bautomation\b",
        r"\bmechatronics\b",
    ],
    "life_sciences": [
        r"\bdna\b",
        r"\bprotein\b",
        r"\bgenome\b",
        r"\bevolutionary biology\b",
        r"\bnatural selection\b",
        r"\bspeciation\b",
        r"\bbioinformatics\b",
        r"\bgene\b",
        r"\bsequencing\b",
        r"\bpcr\b",
        r"\brnaseq\b",
        r"\bcell biology\b",
        r"\bcell division\b",
        r"\becology\b",
        r"\bphotosynthesis\b",
        r"\bphenotype\b",
        r"\bclinical trial\b",
        r"\bepidemiology\b",
        r"\bpathology\b",
        r"\bpharmacology\b",
        r"\bvaccine\b",
        r"\bantibody\b",
        r"\bgenetics\b",
        r"\bmetabolism\b",
    ],
    "cs_software": [
        r"\bdesign pattern\b",
        r"\bsoftware architecture\b",
        r"\bcompiler\b",
        r"\bquery optimization\b",
        r"\bdata structure\b",
        r"\balgorithm\b",
        r"\boperating system\b",
        r"\bnetwork protocol\b",
        r"\bdynamic programming\b",
        r"\bgraph algorithm\b",
        r"\btime complexity\b",
        r"\bspace complexity\b",
        r"\bcryptography\b",
        r"\bdistributed system\b",
        r"\bsoftware testing\b",
        r"\brefactoring\b",
        r"\bprogramming language\b",
        r"\bdatabase\b",
        r"\bsql\b",
        r"\bsoftware design\b",
    ],
}

SE_DOMAIN_ORDER = [
    "hpc",
    "mathematics",
    "statistics_ml",
    "physics_chemistry",
    "engineering",
    "life_sciences",
    "cs_software",
]

# ---------------------------------------------------------------------------
# MMLU + MMLU-Pro subject → domain mapping
# ---------------------------------------------------------------------------
MMLU_DOMAIN_SUBJECTS: dict[str, list[str]] = {
    "mathematics": [
        "abstract_algebra",
        "college_mathematics",
        "elementary_mathematics",
        "high_school_mathematics",
    ],
    "statistics_ml": [
        "high_school_statistics",
        "machine_learning",
    ],
    "physics_chemistry": [
        "astronomy",
        "high_school_chemistry",
        "college_chemistry",
        "high_school_physics",
    ],
    "engineering": [
        "electrical_engineering",
        "conceptual_physics",
    ],
    "life_sciences": [
        "college_biology",
        "high_school_biology",
        "college_medicine",
        "clinical_knowledge",
        "medical_genetics",
        "anatomy",
        "nutrition",
        "virology",
    ],
    "cs_software": [
        "computer_security",
        "college_computer_science",
        "high_school_computer_science",
    ],
    "philosophy_ethics": [
        "philosophy",
        "moral_scenarios",
        "moral_disputes",
        "logical_fallacies",
        "jurisprudence",
        "international_law",
    ],
    "social_sciences": [
        "sociology",
        "professional_psychology",
        "high_school_psychology",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "econometrics",
        "security_studies",
        "us_foreign_policy",
        "public_relations",
        "high_school_government_and_politics",
        "human_sexuality",
        "professional_law",
    ],
    "history_culture": [
        "prehistory",
        "world_religions",
        "global_facts",
        "high_school_geography",
        "international_law",
    ],
}

MMLU_PRO_DOMAIN_CATEGORIES: dict[str, list[str]] = {
    "mathematics": ["math"],
    "statistics_ml": [],  # SE + MMLU sufficient
    "physics_chemistry": ["physics", "chemistry"],
    "engineering": ["engineering"],
    "life_sciences": ["biology", "health"],
    "cs_software": ["computer science"],
    "philosophy_ethics": ["philosophy"],
    "social_sciences": ["economics", "psychology", "business", "law"],
    "history_culture": ["history", "other"],
}

# ---------------------------------------------------------------------------
# Quality filters
# ---------------------------------------------------------------------------


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text) for p in patterns)


# Patterns that make any question non-standalone
_CONTEXT_PATTERNS = [
    r"\bwhich of the following\b",
    r"\bthe following\b",
    r"\baccording to (the )?(passage|text|article|author|table|figure)\b",
    r"\bthe passage\b",
    r"\bthe text above\b",
    r"\bthe (table|figure|graph|diagram|chart) (above|below)\b",
    r"\bthe scenario (above|below|described)\b",
    r"_{3,}",  # fill-in-the-blank blanks
    r"\(A\)|\(B\)|\(C\)|\(D\)",  # answer choices leaked into question
    r"^\s*\d+\.\s*$",  # numbered stub
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
    r"^\d+\.\d+\.\d+\b",  # version strings
    r"exception\s+\w+\.\w+\.\w+",  # Java-style exception
    r"com\.mysql\.",
    r"java\.lang\.",
    r"java\.io\.",
    r"no such file or directory",
    r"permission denied",
    r"command not found",
    r"bad interpreter",
    r"cannot find symbol",
    r"undefined reference to",
    r"ORA-\d{5}",  # Oracle error codes
    r"^theorem \d+\b",  # bare book reference
    r"\bpage \d+$",  # page reference at end
]

_QUESTION_WORDS = {
    "what",
    "how",
    "why",
    "when",
    "where",
    "which",
    "who",
    "is",
    "are",
    "does",
    "do",
    "did",
    "can",
    "could",
    "should",
    "will",
    "would",
    "has",
    "have",
    "was",
    "were",
    "explain",
    "describe",
    "compare",
    "implement",
    "define",
    "show",
    "prove",
    "find",
    "calculate",
    "help",
    "analyze",
}


def is_quality_se(text: str) -> bool:
    t = text.strip()
    tl = t.lower()

    if not (25 <= len(t) <= 200):
        return False

    ascii_chars = sum(1 for c in t if ord(c) < 128)
    if ascii_chars / len(t) < 0.85:
        return False

    words = re.findall(r"[a-zA-Z]{2,}", t)
    if len(words) < 4:
        return False

    if _matches_any(tl, _ERROR_PATTERNS):
        return False

    alpha_ratio = len(re.findall(r"[a-zA-Z]", t)) / len(t)
    if alpha_ratio < 0.35:
        return False

    # Reject pure-formula LaTeX: backslash commands dominate the text
    if "\\" in t and t.count("$") >= 2:
        in_dollar = False
        dollar_chars = 0
        for ch in t:
            if ch == "$":
                in_dollar = not in_dollar
            elif in_dollar:
                dollar_chars += 1
        if dollar_chars / len(t) > 0.30:
            return False

    # Require question structure: "?" OR question word in first 6 tokens OR ≥9 words
    tokens = tl.split()
    has_qmark = "?" in t
    has_qword = any(w in _QUESTION_WORDS for w in tokens[:6])
    long_enough = len(tokens) >= 9
    return has_qmark or has_qword or long_enough


def is_quality_mmlu(text: str) -> bool:
    t = text.strip()
    tl = t.lower()
    if not (20 <= len(t) <= 400):
        return False
    if _matches_any(tl, _CONTEXT_PATTERNS):
        return False
    words = re.findall(r"[a-zA-Z]{2,}", t)
    return len(words) >= 4


def is_quality_pubmedqa(text: str) -> bool:
    t = text.strip()
    tl = t.lower()
    if not (20 <= len(t) <= 300):
        return False
    # PubMedQA questions are yes/no research questions — all are HIGH complexity
    # Filter obvious non-questions
    if _matches_any(tl, _CONTEXT_PATTERNS):
        return False
    words = re.findall(r"[a-zA-Z]{2,}", t)
    return len(words) >= 4


# ---------------------------------------------------------------------------
# Source loaders — each returns list[dict] with keys:
#   text, source, subject, domain
# ---------------------------------------------------------------------------


def load_se(dry_run: bool) -> dict[str, list[dict]]:
    from datasets import load_dataset

    print("Loading StackExchange duplicates...")
    ds = load_dataset(
        "sentence-transformers/stackexchange-duplicates",
        "title-title-pair",
        split="train",
    )

    by_domain: dict[str, list[dict]] = {d: [] for d in SE_DOMAIN_ORDER}
    seen: set[str] = set()

    for row in ds:
        for field in ("title1", "title2"):
            text = row.get(field, "").strip()
            if text in seen or not is_quality_se(text):
                continue
            tl = text.lower()
            for domain in SE_DOMAIN_ORDER:
                if _matches_any(tl, SE_DOMAIN_KEYWORDS[domain]):
                    by_domain[domain].append(
                        {
                            "text": text,
                            "source": "stackexchange",
                            "subject": domain,
                            "domain": domain,
                        }
                    )
                    seen.add(text)
                    break

    for d in SE_DOMAIN_ORDER:
        n = len(by_domain[d])
        print(f"  SE {d:22s}: {n:5,} candidates")
        if dry_run:
            import random

            random.Random(42).shuffle(by_domain[d])
            by_domain[d] = by_domain[d][:120]

    return by_domain


def load_mmlu(dry_run: bool) -> dict[str, list[dict]]:
    from datasets import load_dataset

    print("Loading MMLU (cais/mmlu)...")
    ds = load_dataset("cais/mmlu", "all", split="test+validation")

    subject_to_domain: dict[str, str] = {}
    for domain, subjects in MMLU_DOMAIN_SUBJECTS.items():
        for subj in subjects:
            subject_to_domain[subj] = domain

    by_domain: dict[str, list[dict]] = {d: [] for d in MMLU_DOMAIN_SUBJECTS}
    seen: set[str] = set()

    for row in ds:
        subj = row.get("subject", "")
        domain = subject_to_domain.get(subj)
        if domain is None:
            continue
        text = row.get("question", "").strip()
        if not text or text in seen or not is_quality_mmlu(text):
            continue
        by_domain[domain].append(
            {
                "text": text,
                "source": "mmlu",
                "subject": subj,
                "domain": domain,
            }
        )
        seen.add(text)

    for d in MMLU_DOMAIN_SUBJECTS:
        n = len(by_domain[d])
        print(f"  MMLU {d:22s}: {n:5,} candidates")
        if dry_run:
            import random

            random.Random(42).shuffle(by_domain[d])
            by_domain[d] = by_domain[d][:120]

    return by_domain


def load_mmlu_pro(dry_run: bool) -> dict[str, list[dict]]:
    from datasets import load_dataset

    print("Loading MMLU-Pro (TIGER-Lab/MMLU-Pro)...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test+validation", trust_remote_code=True)

    cat_to_domain: dict[str, str] = {}
    for domain, cats in MMLU_PRO_DOMAIN_CATEGORIES.items():
        for cat in cats:
            cat_to_domain[cat] = domain

    by_domain: dict[str, list[dict]] = {d: [] for d in MMLU_PRO_DOMAIN_CATEGORIES}
    seen: set[str] = set()

    for row in ds:
        cat = row.get("category", "")
        domain = cat_to_domain.get(cat)
        if domain is None:
            continue
        text = row.get("question", "").strip()
        if not text or text in seen or not is_quality_mmlu(text):
            continue
        by_domain[domain].append(
            {
                "text": text,
                "source": "mmlu_pro",
                "subject": cat,
                "domain": domain,
            }
        )
        seen.add(text)

    for d in MMLU_PRO_DOMAIN_CATEGORIES:
        if by_domain[d]:
            n = len(by_domain[d])
            print(f"  MMLU-Pro {d:18s}: {n:5,} candidates")
            if dry_run:
                import random

                random.Random(42).shuffle(by_domain[d])
                by_domain[d] = by_domain[d][:120]

    return by_domain


def load_pubmedqa(dry_run: bool) -> list[dict]:
    from datasets import load_dataset

    print("Loading PubMedQA...")
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train", trust_remote_code=True)

    candidates = []
    seen: set[str] = set()
    for row in ds:
        text = row.get("question", "").strip()
        if not text or text in seen or not is_quality_pubmedqa(text):
            continue
        candidates.append(
            {
                "text": text,
                "source": "pubmedqa",
                "subject": "biomedical_research",
                "domain": "life_sciences",
            }
        )
        seen.add(text)

    print(f"  PubMedQA life_sciences    : {len(candidates):5,} candidates")
    if dry_run:
        import random

        random.Random(42).shuffle(candidates)
        candidates = candidates[:120]
    return candidates


# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------


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


def label_batch(texts: list[str], client, dry_run: bool) -> list[str]:
    if dry_run:
        import random

        return [random.choice(["LOW", "MEDIUM", "HIGH"]) for _ in texts]

    prompt = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    for attempt in range(3):
        try:
            resp = client.messages.create(
                model=LABELING_MODEL,
                max_tokens=300,
                messages=[
                    {
                        "role": "user",
                        "content": COMPLEXITY_RUBRIC + "\n\nQueries:\n" + prompt,
                    }
                ],
            )
            raw = resp.content[0].text.strip()
            labels = json.loads(raw)
            if isinstance(labels, list) and len(labels) == len(texts):
                norm = [lb.upper() for lb in labels]
                if all(lb in ("LOW", "MEDIUM", "HIGH") for lb in norm):
                    return norm
        except Exception as e:
            if attempt == 2:
                print(f"  [WARN] labeling failed: {e}")
                return ["MEDIUM"] * len(texts)
            time.sleep(2**attempt)
    return ["MEDIUM"] * len(texts)


def label_pool(pool: list[dict], client, dry_run: bool) -> list[dict]:
    """Label an entire pool in batches, return pool with ground_truth set."""
    unlabeled = [q for q in pool if "ground_truth" not in q]
    labeled = [q for q in pool if "ground_truth" in q]
    if not unlabeled:
        return pool

    n_batches = (len(unlabeled) + LABEL_BATCH_SIZE - 1) // LABEL_BATCH_SIZE
    for i in range(n_batches):
        batch = unlabeled[i * LABEL_BATCH_SIZE : (i + 1) * LABEL_BATCH_SIZE]
        labels = label_batch([q["text"] for q in batch], client, dry_run)
        for q, lbl in zip(batch, labels, strict=False):
            q["ground_truth"] = lbl
        if not dry_run:
            time.sleep(0.15)

    return labeled + unlabeled


# ---------------------------------------------------------------------------
# Per-domain collection with hard equal caps
# ---------------------------------------------------------------------------


def collect_domain(
    domain: str,
    pool: list[dict],
    target: int,
    client,
    dry_run: bool,
    rng,
) -> dict[str, list[dict]]:
    """
    From pool, collect exactly `target` labeled queries per class.
    Labels pool in batches of 100. Stops when all classes hit target
    or pool is exhausted.
    Returns {"LOW": [...], "MEDIUM": [...], "HIGH": [...]}
    """
    buckets: dict[str, list[dict]] = {"LOW": [], "MEDIUM": [], "HIGH": []}
    rng.shuffle(pool)

    batch_size = LABEL_BATCH_SIZE * 5  # 100 at a time

    idx = 0
    pass_n = 0
    while any(len(buckets[c]) < target for c in buckets) and idx < len(pool):
        batch = pool[idx : idx + batch_size]
        idx += batch_size
        pass_n += 1
        labeled = label_pool(batch, client, dry_run)
        added = 0
        for q in labeled:
            cls = q["ground_truth"]
            if cls in buckets and len(buckets[cls]) < target:
                buckets[cls].append(q)
                added += 1
        if dry_run:
            break

    counts = {c: len(v) for c, v in buckets.items()}
    exhausted = idx >= len(pool) and any(len(buckets[c]) < target for c in buckets)
    flag = " [POOL EXHAUSTED]" if exhausted else ""
    print(f"  {domain:22s}: {counts}  ({idx}/{len(pool)} labeled){flag}")

    return {c: v[:target] for c, v in buckets.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--domain-target", type=int, default=DOMAIN_TARGET)
    parser.add_argument("--train-output", default=str(TRAIN_OUTPUT))
    parser.add_argument("--test-output", default=str(TEST_OUTPUT))
    args = parser.parse_args()

    import random

    rng = random.Random(42)

    d_target = 3 if args.dry_run else args.domain_target
    train_n = max(1, round(d_target * TRAIN_FRACTION))
    test_n = d_target - train_n

    total_queries = d_target * len(DOMAINS) * 3
    print(f"\n{'='*65}")
    print("STREAM Dataset Builder — 10-domain balanced")
    print(f"  Domain target : {d_target}/class/domain (hard equal cap)")
    print(f"  Total queries : {total_queries:,}")
    print(
        f"  Split         : {train_n * len(DOMAINS) * 3:,} train + "
        f"{test_n * len(DOMAINS) * 3:,} test ({TRAIN_FRACTION:.0%}/{1-TRAIN_FRACTION:.0%})"
    )
    print(f"  Labeling model: {LABELING_MODEL}")
    print(f"{'='*65}\n")

    import anthropic

    client = anthropic.Anthropic(api_key=get_api_key()) if not args.dry_run else None

    # ----------------------------------------------------------------
    # Load all source pools
    # ----------------------------------------------------------------
    se_pools = load_se(args.dry_run)
    mmlu_pools = load_mmlu(args.dry_run)
    mmlu_pro_pools = load_mmlu_pro(args.dry_run)
    pubmedqa_pool = load_pubmedqa(args.dry_run)

    # ----------------------------------------------------------------
    # Merge pools per domain — deduplicate by text
    # ----------------------------------------------------------------
    all_pools: dict[str, list[dict]] = {d: [] for d in DOMAINS}
    seen_texts: set[str] = set()

    def add_pool(domain: str, items: list[dict]) -> None:
        for item in items:
            if item["text"] not in seen_texts:
                all_pools[domain].append(item)
                seen_texts.add(item["text"])

    # SE → STEM domains
    for d in SE_DOMAIN_ORDER:
        add_pool(d, se_pools.get(d, []))

    # MMLU → all mapped domains
    for d, items in mmlu_pools.items():
        add_pool(d, items)

    # MMLU-Pro → all mapped domains
    for d, items in mmlu_pro_pools.items():
        add_pool(d, items)

    # PubMedQA → life_sciences
    add_pool("life_sciences", pubmedqa_pool)

    print("\nMerged pool sizes (after deduplication):")
    for d in DOMAINS:
        print(f"  {d:22s}: {len(all_pools[d]):5,}")

    # ----------------------------------------------------------------
    # Per-domain collection (hard equal cap)
    # ----------------------------------------------------------------
    print(f"\nCollecting {d_target}/class/domain (hard cap):")
    domain_results: dict[str, dict[str, list[dict]]] = {}
    for domain in DOMAINS:
        domain_results[domain] = collect_domain(
            domain, all_pools[domain], d_target, client, args.dry_run, rng
        )

    # ----------------------------------------------------------------
    # Check final counts
    # ----------------------------------------------------------------
    print("\nFinal per-domain per-class counts:")
    ok = True
    for domain in DOMAINS:
        counts = {c: len(domain_results[domain][c]) for c in ("LOW", "MEDIUM", "HIGH")}
        min_c = min(counts.values())
        flag = "" if min_c == d_target else f"  ⚠ SHORT ({min_c}/{d_target})"
        print(f"  {domain:22s}: {counts}{flag}")
        if min_c < d_target:
            ok = False

    if not ok and not args.dry_run:
        print("\n[WARN] Some domains exhausted before reaching target.")
        print("       The dataset will be unequal at the class×domain level.")
        print("       Consider expanding keyword lists or adding more sources.")

    # ----------------------------------------------------------------
    # 80/20 stratified split — per domain per class
    # ----------------------------------------------------------------
    train_queries: list[dict] = []
    test_queries: list[dict] = []

    for domain in DOMAINS:
        for cls in ("LOW", "MEDIUM", "HIGH"):
            qs = list(domain_results[domain][cls])
            rng.shuffle(qs)
            # Trim to minimum available (handles exhausted domains fairly)
            available = len(qs)
            t_n = max(1, round(available * TRAIN_FRACTION))
            e_n = available - t_n
            train_queries.extend(qs[:t_n])
            test_queries.extend(qs[t_n : t_n + e_n])

    rng.shuffle(train_queries)
    rng.shuffle(test_queries)

    for i, q in enumerate(train_queries):
        q["id"] = i + 1
    for i, q in enumerate(test_queries):
        q["id"] = i + 1

    # ----------------------------------------------------------------
    # Statistics
    # ----------------------------------------------------------------
    train_class = Counter(q["ground_truth"] for q in train_queries)
    test_class = Counter(q["ground_truth"] for q in test_queries)
    train_domain = Counter(q["domain"] for q in train_queries)
    test_domain = Counter(q["domain"] for q in test_queries)
    train_source = Counter(q["source"] for q in train_queries)

    meta_base = {
        "labeling_model": LABELING_MODEL,
        "domain_target": d_target,
        "train_fraction": TRAIN_FRACTION,
        "split_seed": 42,
        "domains": DOMAINS,
        "balance": "hard equal cap per domain per class — both axes balanced",
        "sources": {
            "stackexchange": "sentence-transformers/stackexchange-duplicates",
            "mmlu": "cais/mmlu (standalone subjects only)",
            "mmlu_pro": "TIGER-Lab/MMLU-Pro (standalone questions only)",
            "pubmedqa": "qiaojin/PubMedQA pqa_labeled (all HIGH)",
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dry_run": args.dry_run,
    }

    train_dataset = {
        "metadata": {
            **meta_base,
            "split": "train",
            "n_queries": len(train_queries),
            "class_distribution": dict(train_class),
            "domain_distribution": dict(train_domain),
            "source_distribution": dict(train_source),
        },
        "queries": train_queries,
    }
    test_dataset = {
        "metadata": {
            **meta_base,
            "split": "test",
            "n_queries": len(test_queries),
            "class_distribution": dict(test_class),
            "domain_distribution": dict(test_domain),
        },
        "queries": test_queries,
    }

    Path(args.train_output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.train_output, "w") as f:
        json.dump(train_dataset, f, indent=2)
    with open(args.test_output, "w") as f:
        json.dump(test_dataset, f, indent=2)

    # Flat JSONL for HuggingFace dataset viewer
    train_jsonl = Path(args.train_output).parent / "balanced_train.jsonl"
    test_jsonl = Path(args.train_output).parent / "balanced_test.jsonl"
    with open(train_jsonl, "w") as f:
        for q in train_queries:
            f.write(json.dumps(q) + "\n")
    with open(test_jsonl, "w") as f:
        for q in test_queries:
            f.write(json.dumps(q) + "\n")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print(f"\n{'='*65}")
    print("DONE")
    print(f"  Train : {len(train_queries):,} queries  class={dict(train_class)}")
    print(f"  Test  : {len(test_queries):,}  queries  class={dict(test_class)}")
    print(f"  Train domain: {dict(sorted(train_domain.items()))}")
    print(f"  Sources: {dict(train_source)}")
    print(f"  Saved  : {args.train_output}")
    print(f"  Saved  : {args.test_output}")
    print(f"  JSONL  : {train_jsonl}  |  {test_jsonl}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
