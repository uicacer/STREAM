# STREAM Evaluation Guide

## Overview

This guide explains how to run the STREAM evaluation benchmarks for the PEARC 2026 paper.
We measure three things that align with what similar papers (FIRST, RouteLLM, FrugalGPT) evaluate:

1. **Streaming & Latency** (Day 1) — Your main contribution. How fast do tokens arrive?
2. **Routing Accuracy** (Day 2) — Does the complexity judge classify queries correctly?
3. **Compression Impact** (Day 2) — Does tier-aware compression keep simple queries on cheap tiers?

All results are saved as JSON files in `scripts/eval/results/` and can be directly used
to fill in the paper's evaluation tables.

---

## Prerequisites

Before running benchmarks, make sure:

- [ ] STREAM middleware is running (desktop or server mode)
- [ ] Ollama is running with `llama3.2:3b` and `gemma3:4b` loaded
- [ ] Lakeshore HPC is accessible (for Lakeshore benchmarks)
- [ ] WebSocket relay is running (for streaming vs batch comparison)
- [ ] Cloud API keys are configured (for cloud benchmarks)

**Quick check — is STREAM running?**
```bash
curl http://localhost:5000/health
```

---

## Day 1: Streaming & Latency Benchmarks

### What We're Measuring and Why

| Metric | What It Means | Why It Matters |
|--------|--------------|----------------|
| **TTFT** (Time to First Token) | Seconds from sending the query to receiving the first token | User-perceived responsiveness. This is what makes streaming feel "fast" |
| **Total Time** | Seconds from query to final token | End-to-end performance |
| **Throughput** (tok/s) | Output tokens generated per second | How fast the model produces text |
| **Relay vs Batch** | TTFT with relay streaming vs without | Proves the dual-channel architecture reduces perceived latency |

### How It Works

The benchmark script sends the **same query** to each tier multiple times and measures:

1. **TTFT**: We open an SSE connection and record the timestamp when the first content
   token arrives (not metadata — the actual first word of the response).

2. **Total Time**: From request sent to final `[DONE]` event.

3. **Throughput**: We count the total output tokens (from the cost summary SSE event)
   and divide by total time.

4. **Relay vs Batch**: For Lakeshore, we run the same query twice — once with the relay
   enabled (streaming) and once with the relay disabled (batch mode where the user waits
   for the complete response). The TTFT difference is the key number.

### Why the Same Query?

We use the same query for all runs ("Explain the concept of recursion in programming")
to control for variability. Different queries produce different-length responses, which
would confuse the latency measurements. By fixing the query, differences in TTFT and
throughput reflect the infrastructure, not the content.

### How to Run

```bash
# Run all Day 1 benchmarks (20 runs per tier)
python scripts/eval/benchmark_latency.py

# Run fewer iterations for a quick test
python scripts/eval/benchmark_latency.py --runs 3

# Run only specific tiers
python scripts/eval/benchmark_latency.py --tiers local lakeshore

# Run relay vs batch comparison only
python scripts/eval/benchmark_latency.py --relay-comparison-only
```

### Reading the Results

Results are saved to `scripts/eval/results/latency_YYYY-MM-DD_HHMMSS.json` and a
summary is printed to the terminal:

```
=== LATENCY BENCHMARK RESULTS ===

Local (Llama 3.2 3B):
  TTFT:       0.12s (median)  ± 0.03s (std)
  Total:      4.82s (median)  ± 0.31s (std)
  Throughput: 41.5 tok/s (median)

Lakeshore - Relay Streaming:
  TTFT:       2.34s (median)  ± 0.45s (std)
  Total:      8.12s (median)  ± 0.67s (std)
  Throughput: 24.6 tok/s (median)

Lakeshore - Batch Fallback:
  TTFT:       8.12s (median)  ← User waits this long before seeing ANY text
  Total:      8.15s (median)
  Throughput: N/A (all at once)

Cloud (Claude Sonnet):
  TTFT:       0.89s (median)  ± 0.12s (std)
  Total:      3.45s (median)  ± 0.28s (std)
  Throughput: 57.8 tok/s (median)
```

The key story: **Relay streaming reduces Lakeshore TTFT from ~8s (batch) to ~2s,
making HPC inference feel interactive.**

### Statistical Notes

- We report **median** (not mean) because latency distributions are skewed — a single
  slow request shouldn't dominate the result. This is standard practice (FIRST uses median too).
- We report **standard deviation** to show consistency.
- We run **20 iterations** per measurement for statistical significance.
- The first run is a **warm-up** (discarded) to avoid cold-start effects.

---

## Day 2: Routing Accuracy & Compression

### Benchmark 1: Routing Accuracy

#### What We're Measuring

The complexity judge (Llama 3.2 3B running locally) classifies each query as LOW, MEDIUM,
or HIGH. We test whether it classifies correctly by running 60 hand-labeled queries
through it and checking accuracy.

#### Why This Matters

If the judge misclassifies a simple query as HIGH, the user pays for cloud when local
would suffice. If it misclassifies a complex query as LOW, the response quality suffers.
RouteLLM evaluates this exact tradeoff.

#### The Test Set

The file `scripts/eval/test_queries.json` contains 60 queries:
- 20 LOW (simple facts, definitions, greetings)
- 20 MEDIUM (explanations, code writing, analysis)
- 20 HIGH (multi-step reasoning, research-level, comparisons)

Each query has a `ground_truth` label assigned by the paper authors.

#### How to Run

```bash
# Run routing accuracy benchmark
python scripts/eval/benchmark_routing.py

# Show per-query results (see which ones were misclassified)
python scripts/eval/benchmark_routing.py --verbose
```

#### Reading the Results

```
=== ROUTING ACCURACY ===

Overall: 47/60 correct (78.3%)

Per-class:
  LOW:    18/20 (90.0%)
  MEDIUM: 13/20 (65.0%)  ← Hardest to classify (borderline queries)
  HIGH:   16/20 (80.0%)

Confusion Matrix:
              Predicted LOW  Predicted MED  Predicted HIGH
Actual LOW         18             2              0
Actual MED          3            13              4
Actual HIGH         0             4             16

Cost Impact:
  Queries routed to free tiers: 31/60 (51.7%)
  Estimated cost (auto mode):   $0.042
  Estimated cost (all cloud):   $0.180
  Cost savings:                 76.7%
```

### Benchmark 2: Compression Impact

#### What We're Measuring

In a long conversation (30+ turns), does tier-aware compression keep simple queries
on the local tier? Without compression, the accumulated conversation history exceeds
the local tier's 32K context limit, forcing an upgrade to a more expensive tier.

#### How It Works

The script simulates a 30-turn conversation:

1. Sends 30 alternating user/assistant messages to build up context
2. At turns 10, 15, 20, 25, and 30, sends a simple query ("What is 2+2?")
3. Records which tier handles the simple query and the token count

It runs this twice:
- **With compression enabled** (default) — expects local tier to handle simple queries
- **With compression disabled** — expects tier upgrade after ~turn 20

#### How to Run

```bash
# Run compression benchmark (5 simulated conversations)
python scripts/eval/benchmark_compression.py

# Single conversation with verbose output
python scripts/eval/benchmark_compression.py --conversations 1 --verbose
```

#### Reading the Results

```
=== COMPRESSION IMPACT ===

Simple queries staying on local tier:
  Turn 10:  With: 100%  Without: 100%
  Turn 15:  With: 100%  Without: 100%
  Turn 20:  With: 100%  Without:  40%  ← Compression starts mattering
  Turn 25:  With:  80%  Without:   0%
  Turn 30:  With:  80%  Without:   0%

Forced tier upgrades (total across 5 conversations):
  With compression:    2
  Without compression: 8

Avg cloud cost per conversation:
  With compression:    $0.003
  Without compression: $0.021
```

---

## Filling in the Paper Tables

After running all benchmarks, the script `scripts/eval/generate_tables.py` reads the
results JSON files and outputs LaTeX-formatted table rows you can paste directly into
the paper:

```bash
python scripts/eval/generate_tables.py
```

Output:
```latex
% Table 2: Response latency by tier
Local (Llama 3.2 3B)          & 0.12 $\pm$ 0.03 & 4.82 $\pm$ 0.31 & 41.5 \\
Lakeshore (relay streaming)   & 2.34 $\pm$ 0.45 & 8.12 $\pm$ 0.67 & 24.6 \\
Lakeshore (batch fallback)    & 8.12 $\pm$ 0.54 & 8.15 $\pm$ 0.55 & --- \\
Cloud (Claude Sonnet)         & 0.89 $\pm$ 0.12 & 3.45 $\pm$ 0.28 & 57.8 \\
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Connection refused" | Make sure STREAM is running: `curl http://localhost:5000/health` |
| Lakeshore timeouts | Check Globus Compute auth: the benchmark will prompt you if needed |
| Relay not working | Verify relay URL in `.env` and that the relay server is running |
| Cloud errors | Check API keys in `.env` (OPENROUTER_API_KEY or ANTHROPIC_API_KEY) |
| Inconsistent results | Increase `--runs` to 30+ for more stable medians |

---

## File Structure

```
scripts/eval/
├── EVALUATION_GUIDE.md          ← You are here
├── benchmark_latency.py         ← Day 1: TTFT, throughput, relay vs batch
├── benchmark_routing.py         ← Day 2: Routing accuracy + cost savings
├── benchmark_compression.py     ← Day 2: Compression impact on long conversations
├── test_queries.json            ← 60 hand-labeled queries (LOW/MEDIUM/HIGH)
├── generate_tables.py           ← Converts results to LaTeX table rows
└── results/                     ← Output directory (created automatically)
    ├── latency_2026-03-06_143022.json
    ├── routing_2026-03-07_091544.json
    └── compression_2026-03-07_102311.json
```
