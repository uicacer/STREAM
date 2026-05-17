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

## Part 3: Training ModernBERT

Script: `scripts/eval/train_modernbert.py`

### What It Does

1. Loads `stream_routing_benchmark.json`
2. Splits 70% train / 15% val / 15% test (stratified by domain+complexity, seed=42)
3. Fine-tunes `answerdotai/ModernBERT-base` (149M params, 2T token pretrain, Dec 2024)
4. Evaluates on test set: accuracy, macro-F1, per-class F1
5. Measures single-query CPU latency (p50/p95/p99 over 200 queries)
6. Saves model to `scripts/eval/models/modernbert/`
7. Prints exact numbers to paste into the paper

### Training Hyperparameters

| Parameter | Value | Why |
|-----------|-------|-----|
| Epochs | 5 | Typical for fine-tuning on domain-specific data |
| Batch size | 32 | Fits on CPU/MPS/GPU comfortably |
| Learning rate | 2e-5 | Standard for BERT-family fine-tuning |
| Max length | 128 tokens | Covers ~95% of queries; longer truncated |
| Metric for best model | macro-F1 | Balanced across all three classes (not biased toward majority) |

### How to Run

```bash
# Full training (requires dataset v2 to exist)
python scripts/eval/train_modernbert.py

# Quick sanity check (30 examples, 1 epoch)
python scripts/eval/train_modernbert.py --dry-run

# Custom dataset path
python scripts/eval/train_modernbert.py \
    --dataset scripts/eval/stream_routing_benchmark.json
```

**Prerequisites:**
```bash
pip install transformers torch datasets scikit-learn accelerate
```

Training takes ~15-30 minutes on CPU (Apple M-series), or ~5 minutes with a GPU.

### Output

- `scripts/eval/models/modernbert/` — fine-tuned model + tokenizer (used by STREAM at runtime)
- `scripts/eval/results/modernbert_training_report.json` — all metrics

### Reading the Terminal Output

At the end, the script prints a "PAPER NUMBERS" block:

```
============================================================
PAPER NUMBERS (paste into pearc26-stream-paper.tex):
============================================================
  ModernBERT accuracy:          92.3%
  ModernBERT macro-F1:          0.921
  ModernBERT LOW F1:            0.948
  ModernBERT MEDIUM F1:         0.892
  ModernBERT HIGH F1:           0.923
  ModernBERT latency p50:       14.2 ms
  Free-tier retention:          97.1%
```

Copy these directly into the paper's TODO placeholders.

---

## Part 4: Routing Accuracy Benchmark

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

## Part 5: How ModernBERT Integrates into STREAM

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

## Part 6: Streaming & Latency Benchmarks

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

## Part 7: Compression Impact Benchmark

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

## Part 8: Updating the Paper

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
1. Generate dataset v2 (runs in background, ~2-3h):
   python scripts/eval/generate_benchmark_dataset.py

2. Train ModernBERT (~15-30 min):
   python scripts/eval/train_modernbert.py
   → Copy "PAPER NUMBERS" into paper

3. Run routing benchmark for both judges:
   python scripts/eval/benchmark_routing.py --judge modernbert \
       --queries scripts/eval/stream_routing_benchmark.json
   python scripts/eval/benchmark_routing.py --judge ollama-3b \
       --queries scripts/eval/stream_routing_benchmark.json
   → Copy confusion matrix + cost numbers into paper

4. Run latency benchmarks:
   python scripts/eval/benchmark_latency.py
   → Copy TTFT / throughput numbers into paper

5. (Optional) Upload to HuggingFace:
   python scripts/eval/upload_to_huggingface.py
   → Replace HuggingFace TODO in paper with real URL
```

---

## File Structure

```
scripts/eval/
├── EVALUATION_GUIDE.md                 ← You are here
│
├── generate_benchmark_dataset.py    ← Generate 6,912 labeled queries (Claude + extended thinking)
├── stream_routing_benchmark.json           ← Output: full labeled dataset
│
├── train_modernbert.py                 ← Fine-tune ModernBERT-base on v2 dataset
├── models/modernbert/                  ← Output: fine-tuned model (used by STREAM at runtime)
│
├── benchmark_routing.py                ← Measure routing accuracy for any judge strategy
├── benchmark_latency.py                ← Measure TTFT / throughput across tiers
├── benchmark_compression.py            ← Measure compression impact on tier selection
│
├── sample_for_validation.py            ← Draw a 252-query stratified sample for manual review
├── compute_validation_kappa.py         ← Compute inter-annotator κ from manual review
│
├── generate_tables.py                  ← Convert results JSON → LaTeX table rows
│
└── results/                            ← All benchmark outputs (JSON, timestamped)
    ├── modernbert_training_report.json
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
| Lakeshore timeouts in latency benchmark | Check Globus Compute auth; the script prompts if expired |
