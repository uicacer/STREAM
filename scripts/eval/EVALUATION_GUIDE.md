# STREAM Evaluation Guide

This guide covers everything you need to understand and run the STREAM evaluation
pipeline for the PEARC 2026 paper. It also explains the knowledge distillation
approach used to build the ModernBERT routing classifier.

---

## Overview: Three Evaluation Areas

| Area | Script | What It Measures |
|------|--------|-----------------|
| **Routing Accuracy** | `benchmark_routing.py` | Does the complexity judge classify LOW/MEDIUM/HIGH correctly? |
| **Streaming Latency** | `benchmark_latency.py` | TTFT, throughput, relay vs batch |
| **Compression Impact** | `benchmark_compression.py` | Does compression keep simple queries on cheap tiers? |

---

## Part 1: The Complexity Classifier — What and Why

### The Problem

STREAM needs to decide: should this query go to a free local model, the free HPC tier,
or a paid cloud model? To do that, it classifies each query as LOW, MEDIUM, or HIGH
complexity. This is the **complexity judge**.

### Two Judge Approaches

| Approach | Latency | Cost | Accuracy | Dependency |
|----------|---------|------|----------|------------|
| **Llama 3.2 3B** (LLM judge) | ~390ms | Free | Good | Ollama running |
| **ModernBERT** (distilled classifier) | ~15ms | Free | Better | Model file on disk |

### Why LLM-Supervised Fine-Tuning?

The LLM judge (Llama 3.2 3B) works but is slow — 390ms per classification adds
noticeable lag to every single query. We can do better:

1. Use Claude Sonnet 4.6 as a **labeling model** — with extended thinking enabled so it applies the reasoning-depth rubric carefully before committing to a label.
2. Fine-tune ModernBERT-base (149M params) on those LLM-generated labels.
3. The small model learns the frontier model's routing judgment without needing an LLM at runtime.

This is **LLM-supervised fine-tuning**: Claude generates the labels; ModernBERT learns from them. The result runs at ~15ms with no API dependency.

Note: this is distinct from classical knowledge distillation (Hinton 2015), which transfers soft probability distributions between models of the same family. Here the "teacher" is a frontier LLM and the "student" is an encoder classifier — they share no architecture. The transfer is through hard labels, not soft logits.

### The Reasoning-Depth Rubric

The most important design decision: **complexity is about reasoning depth, not format.**

The same format ("Is X true?", "What is X?", "yes/no") can appear at all three levels:

| Complexity | Definition | Example |
|------------|------------|---------|
| **LOW** | Single retrievable fact — one lookup or direct recall | "What is the capital of France?" / "Is Python interpreted?" |
| **MEDIUM** | Apply established procedure — combine 2-4 known steps | "Write a Python function to sort a list by two keys" / "Is merge sort O(n log n)?" |
| **HIGH** | Construct novel reasoning path — no standard procedure exists | "Is the P vs NP problem likely solvable in the next decade?" / "Does quantum entanglement violate causality?" |

**Why this rubric?** Early versions used format as a proxy (yes/no → LOW, "derive X" → HIGH).
That caused the classifier to learn format patterns rather than reasoning complexity.
A query like "Is the Riemann hypothesis true?" is HIGH — but it looks like a yes/no question.
With the reasoning-depth rubric, the labeling model classifies based on what it takes to
*answer*, not how the question is *phrased*.

---

## Part 2: Dataset Generation (v2)

### Why Regenerate from Scratch?

The v1 dataset (`benchmark_dataset_384.json`) was generated with the old format-proxy rubric.
All yes/no questions were labeled LOW, all "derive X" questions were HIGH. The classifier
learned to pattern-match question format rather than reasoning depth.

The v2 dataset uses the reasoning-depth rubric and extended thinking to prevent this.

### How Dataset v2 is Generated

Script: `scripts/eval/generate_benchmark_dataset.py`

The script calls Claude Sonnet 4.6 with **extended thinking enabled**:

```python
msg = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=16000,
    thinking={"type": "enabled", "budget_tokens": 10000},
    messages=[{"role": "user", "content": prompt}],
)
```

Extended thinking means Claude reasons carefully before producing the output — it
considers edge cases, checks for format-proxy bias, and applies the rubric consistently.
The thinking block is discarded; only the text output (the labeled query) is kept.

### Dataset Size and Structure

| Total | Per domain | Per domain-complexity cell | Domains |
|-------|-----------|---------------------------|---------|
| 6,912 | 1,152 | 384 | 6 |

Domains: `general_knowledge`, `science`, `mathematics`, `coding`, `humanities`,
`professional` (medicine/law/finance).

Each query has:
```json
{
  "id": "sci_low_0042",
  "text": "What is the atomic number of carbon?",
  "ground_truth": "LOW",
  "domain": "science"
}
```

### Format Decoupling Validation

The script runs `validate_format_decoupling()` at the end. It warns if yes/no questions
or "What is X?" questions are >90% concentrated in a single class — that would indicate
the rubric is still leaking format as a signal.

### How to Run

```bash
# Generate full dataset (takes ~2-3 hours, costs ~$5-10 in API credits)
python scripts/eval/generate_benchmark_dataset.py

# Check a partially-generated or existing dataset for format bias
python scripts/eval/generate_benchmark_dataset.py --validate-only

# Use a specific dataset path
python scripts/eval/generate_benchmark_dataset.py \
    --dataset scripts/eval/stream_routing_benchmark.json
```

Progress is printed every 30 queries per cell. The script appends to the existing file,
so if it's interrupted, re-running picks up from where it left off (checks which
domain-complexity cells are already complete).

### Output

`scripts/eval/stream_routing_benchmark.json` — the full 6,912-query dataset.

---

## Part 3: Training and Evaluating ModernBERT

Script: `scripts/eval/train_modernbert.py`

### The Data Leakage Problem

A naive random 70/15/15 split on LLM-generated data is biased. Because Claude
generated 384 queries per domain-complexity cell, near-duplicate phrasings often
end up on both sides of the split. The classifier memorizes phrasing patterns
rather than generalizing, producing inflated accuracy (~99%).

We address this with three evaluation strategies of increasing rigor:

| Strategy | What it tests | Expected accuracy |
|----------|--------------|------------------|
| Random split | Baseline; may be inflated | ~99% (inflated) |
| Domain-held-out CV | Cross-domain generalization | More conservative |
| Similarity-aware split | Deduplication before splitting | Conservative |
| Real-world (Arena) | Out-of-distribution generalization | Most rigorous |

Report the **domain-held-out** and **real-world** numbers in the paper. The
random split is included for comparison only with a caveat.

### Training Hyperparameters

| Parameter | Value | Why |
|-----------|-------|-----|
| Epochs | 5 | Typical for fine-tuning on domain-specific data |
| Batch size | 32 | Fits on CPU/MPS/GPU comfortably |
| Learning rate | 2e-5 | Standard for BERT-family fine-tuning |
| Max length | 128 tokens | Covers ~95% of queries; longer truncated |
| Metric for best model | macro-F1 | Balanced across all three classes |

### Eval Mode 1: Random Split (baseline)

```bash
python scripts/eval/train_modernbert.py
# or explicitly:
python scripts/eval/train_modernbert.py --eval-mode random
```

Standard 70/15/15 split. Fast to run but inflated due to near-duplicate
LLM-generated queries in both train and test.

Output: `results/modernbert_training_report.json`

### Eval Mode 2: Domain-Held-Out Cross-Validation

```bash
python scripts/eval/train_modernbert.py --eval-mode domain-holdout
```

6-fold CV: each fold trains on 5 domains (~5,760 queries) and tests on the
held-out 6th domain (~1,152 queries). Because the test domain's queries were
never seen during training, this measures true cross-domain generalization.
Takes ~6× longer than a single run (~2-3 hours on CPU).

Output: `results/modernbert_domain_holdout_report.json`
Reports mean ± std across 6 folds.

### Eval Mode 3: Similarity-Aware Split

```bash
# Requires sentence-transformers: pip install sentence-transformers
python scripts/eval/train_modernbert.py --eval-mode similarity-split
```

Embeds all queries with `all-MiniLM-L6-v2`, groups near-duplicates (cosine
similarity > 0.90) into connected components, then assigns whole components
to either train or test — never split. This prevents near-identical
query phrasings from appearing on both sides.

Output: `results/modernbert_similarity_split_report.json`

### Common Options

```bash
# Quick sanity check (30 examples, 1 epoch)
python scripts/eval/train_modernbert.py --dry-run

# Load saved model, skip retraining
python scripts/eval/train_modernbert.py --eval-only

# Custom dataset
python scripts/eval/train_modernbert.py \
    --dataset scripts/eval/stream_routing_benchmark.json
```

**Prerequisites:**
```bash
pip install transformers torch datasets scikit-learn accelerate
# For similarity-split only:
pip install sentence-transformers
```

### Output Files

- `scripts/eval/models/modernbert/` — fine-tuned model + tokenizer (used by STREAM at runtime)
- `scripts/eval/results/modernbert_training_report.json` — random split
- `scripts/eval/results/modernbert_domain_holdout_report.json` — 6-fold CV
- `scripts/eval/results/modernbert_similarity_split_report.json` — similarity split

---

## Part 4: Real-World Test Set (LMSYS Chatbot Arena)

Scripts: `scripts/eval/build_realworld_testset.py` and `scripts/eval/eval_on_realworld.py`

### Why a Real-World Test Set?

The training data was generated by Claude — all queries were written by an LLM.
Even with domain-held-out CV, the test set still comes from the same LLM-generated
distribution. The real-world test set uses genuine user prompts from the LMSYS
Chatbot Arena (1M+ conversations), labeled by Claude with the same rubric.

This is the **most rigorous** evaluation: it tests whether the classifier
generalizes to real user behavior, not just to different LLM-generated phrasings.

### Step 1: Build the test set

```bash
# Build 400-query real-world test set (~$0.12 in API credits)
python scripts/eval/build_realworld_testset.py

# Dry run (no API calls, fake labels)
python scripts/eval/build_realworld_testset.py --dry-run

# Custom size
python scripts/eval/build_realworld_testset.py --n 200
```

Output: `scripts/eval/realworld_testset.json`

### Step 2: Evaluate the trained model

```bash
python scripts/eval/eval_on_realworld.py
```

Output: `scripts/eval/results/modernbert_realworld_report.json`

---

## Part 5: Routing Accuracy Benchmark

Script: `scripts/eval/benchmark_routing.py`

### What It Measures

Sends every query in the dataset through the complexity judge and checks whether
the predicted class matches `ground_truth`. Also computes cost savings vs. routing
everything to the cloud.

### How to Run

```bash
# Benchmark with the default judge (modernbert, after training)
python scripts/eval/benchmark_routing.py \
    --judge modernbert \
    --queries scripts/eval/stream_routing_benchmark.json

# Compare with the LLM judge (requires Ollama running)
python scripts/eval/benchmark_routing.py \
    --judge ollama-3b \
    --queries scripts/eval/stream_routing_benchmark.json

# Use the paid Haiku judge (most accurate LLM option)
python scripts/eval/benchmark_routing.py \
    --judge haiku \
    --queries scripts/eval/stream_routing_benchmark.json

# Show per-query details (see which queries were misclassified)
python scripts/eval/benchmark_routing.py \
    --judge modernbert \
    --queries scripts/eval/stream_routing_benchmark.json \
    --verbose
```

**Note:** The benchmark calls the STREAM middleware at `http://localhost:5000`.
Make sure STREAM is running before starting:
```bash
curl http://localhost:5000/health
```

### What the Output Means

```
======================================
  ROUTING ACCURACY RESULTS
======================================
  Overall: 6382/6912 correct (92.3%)

  Per-class:
    LOW      2312/2304 (97.1%)
    MEDIUM   2187/2304 (94.9%)
    HIGH     1883/2304 (81.7%)

  Confusion Matrix:
                  Pred LOW  Pred MED  Pred HIGH
  Actual LOW          2312        45         47
  Actual MED            98      2187        119
  Actual HIGH           12       118       1883

  Cost Impact:
    Queries on free tiers: 4499/6912 (65.1%)
    Estimated cost (auto):      $0.0182
    Estimated cost (all cloud): $0.0518
    Cost savings:               64.9%

  Judge Latency: 0.016s avg (0.012s - 0.024s)
```

**Key numbers for the paper:**
- Overall accuracy and macro-F1 go into Table 3
- Confusion matrix goes directly into Table 3's cells
- "Queries on free tiers" and cost savings go into the cost analysis section
- Judge latency goes into the latency comparison row (15ms vs 390ms)

---

## Part 6: How ModernBERT Integrates into STREAM

Once the model is trained and saved to `scripts/eval/models/modernbert/`, it works
automatically when you set `DEFAULT_JUDGE_STRATEGY = "modernbert"` in config.py
(already set).

### The Call Chain

```
User query
    → complexity_judge.py: judge_complexity(query, strategy="modernbert")
        → _get_classifier()          # lazy-loads pipeline on first call (~1s)
        → clf(query)                 # ~15ms inference
        → returns JudgmentResult(
              complexity="medium",
              method="classifier",
              strategy_used="modernbert",
              scores={"low": 0.12, "medium": 0.76, "high": 0.12}  # shown in UI
          )
    → query_router.py selects tier based on complexity
```

### Fallback Behavior

If the ModernBERT model file is not found (e.g., not trained yet), STREAM
automatically falls back to `ollama-3b`. You will see this in the logs:

```
WARNING  ModernBERT classifier failed: ModernBERT model not found at ...
         Falling back to LLM judge.
```

### Confidence Scores in the UI

The `scores` field (`{"low": 0.12, "medium": 0.76, "high": 0.12}`) is passed to
the frontend and shown as a bar chart in the routing metadata panel. This gives
users visibility into how confident the judge was — a near-tie (0.35/0.33/0.32)
is a much weaker signal than a confident (0.05/0.10/0.85).

---

## Part 7: Streaming & Latency Benchmarks

Script: `scripts/eval/benchmark_latency.py`

### How to Run

```bash
# Run all tiers (20 runs per tier)
python scripts/eval/benchmark_latency.py

# Quick test (3 runs per tier)
python scripts/eval/benchmark_latency.py --runs 3

# Only specific tiers
python scripts/eval/benchmark_latency.py --tiers local lakeshore

# Only the relay vs batch comparison
python scripts/eval/benchmark_latency.py --relay-comparison-only
```

### Key Metrics

| Metric | Definition | Why It Matters |
|--------|------------|----------------|
| **TTFT** | Time from sending query to receiving first token | User-perceived responsiveness — makes streaming "feel fast" |
| **Total Time** | Query to final token | End-to-end wall time |
| **Throughput** | Output tokens / second | How fast the model produces text |
| **Relay vs Batch** | TTFT with streaming vs without | Proves dual-channel architecture; the relay paper's main claim |

Results are reported as **median** (not mean) because latency distributions are
right-skewed — a single slow request should not dominate the reported value.

---

## Part 8: Compression Impact Benchmark

Script: `scripts/eval/benchmark_compression.py`

Tests whether tier-aware conversation compression keeps simple queries on the
free local tier even in long conversations.

```bash
# Full benchmark (5 simulated conversations)
python scripts/eval/benchmark_compression.py

# Single conversation with verbose output
python scripts/eval/benchmark_compression.py --conversations 1 --verbose
```

---

## Part 9: Updating the Paper

After running all benchmarks, update these placeholders in
`docs/pearc26-stream-paper.tex`:

| Placeholder | Source |
|-------------|--------|
| `TODO: modernbert accuracy` | `train_modernbert.py` output |
| `TODO: modernbert macro-f1` | `train_modernbert.py` output |
| `TODO: modernbert latency` | `train_modernbert.py` output |
| `TODO: free-tier retention` | `train_modernbert.py` output |
| Confusion matrix cells | `benchmark_routing.py` output |
| LLM judge accuracy | `benchmark_routing.py` with `--judge ollama-3b` |
| `TODO: HuggingFace URL` | After uploading model to HuggingFace |

---

## Full Pipeline Sequence

```
1. Generate dataset (runs in background, ~2-3h):
   python scripts/eval/generate_benchmark_dataset.py

2. Train ModernBERT — random split baseline (~15-30 min):
   python scripts/eval/train_modernbert.py --eval-mode random

3. Rigorous evaluation — domain-held-out 6-fold CV (~3h on CPU):
   python scripts/eval/train_modernbert.py --eval-mode domain-holdout

4. Rigorous evaluation — similarity-aware split (~30 min):
   pip install sentence-transformers
   python scripts/eval/train_modernbert.py --eval-mode similarity-split

5. Real-world test set (LMSYS Chatbot Arena):
   python scripts/eval/build_realworld_testset.py   # ~$0.12 API cost
   python scripts/eval/eval_on_realworld.py

6. Run routing benchmark for both judges:
   python scripts/eval/benchmark_routing.py --judge modernbert \
       --queries scripts/eval/stream_routing_benchmark.json
   python scripts/eval/benchmark_routing.py --judge ollama-3b \
       --queries scripts/eval/stream_routing_benchmark.json
   → Copy confusion matrix + cost numbers into paper

7. Run latency benchmarks:
   python scripts/eval/benchmark_latency.py
   → Copy TTFT / throughput numbers into paper

8. Generate all LaTeX tables:
   python scripts/eval/generate_tables.py
   → Reads all four eval reports + routing + latency → prints LaTeX rows
```

---

## File Structure

```
scripts/eval/
├── EVALUATION_GUIDE.md                          ← You are here
│
├── generate_benchmark_dataset.py                ← Generate 6,912 labeled queries
├── stream_routing_benchmark.json                ← Output: full labeled dataset (gitignored)
│
├── train_modernbert.py                          ← Train ModernBERT + three eval modes
│     --eval-mode random                         ←   70/15/15 random split (baseline)
│     --eval-mode domain-holdout                 ←   6-fold CV (rigorous)
│     --eval-mode similarity-split               ←   semantic dedup split (rigorous)
├── build_realworld_testset.py                   ← Download + label LMSYS Arena queries
├── eval_on_realworld.py                         ← Evaluate on real-world test set
│
├── models/modernbert/                           ← Trained model weights (gitignored)
│
├── benchmark_routing.py                         ← Routing accuracy for any judge
├── benchmark_latency.py                         ← TTFT / throughput across tiers
├── benchmark_compression.py                     ← Compression impact on tier selection
│
├── compute_validation_kappa.py                  ← Inter-annotator κ
│
├── generate_tables.py                           ← All results → LaTeX table rows
│
└── results/                                     ← All benchmark outputs (JSON)
    ├── modernbert_training_report.json          ← random split
    ├── modernbert_domain_holdout_report.json    ← 6-fold CV
    ├── modernbert_similarity_split_report.json  ← similarity-aware split
    ├── modernbert_realworld_report.json         ← LMSYS Arena
    ├── routing_6912_<timestamp>.json
    └── latency_<timestamp>.json
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModernBERT model not found` | Run `train_modernbert.py` first |
| `Connection refused` to STREAM | Run `curl http://localhost:5000/health`; start STREAM if down |
| Dataset generation stops mid-way | Just re-run; it resumes from the last complete cell |
| `transformers` version error | `pip install "transformers>=4.48.0"` (ModernBERT requires ≥4.48) |
| Slow training on CPU | Normal — ~30 min; use `--dry-run` to verify setup first |
| `sentence-transformers` not found | `pip install sentence-transformers` (similarity-split only) |
| `ANTHROPIC_API_KEY not set` | Set env var for `build_realworld_testset.py` |
| `lmsys/chatbot_arena_conversations` gated | Accept terms on HuggingFace, then `huggingface-cli login` |
| Lakeshore timeouts in latency benchmark | Check Globus Compute auth; the script prompts if expired |
