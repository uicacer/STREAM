---
license: apache-2.0
base_model: answerdotai/ModernBERT-base
language:
  - en
tags:
  - text-classification
  - llm-routing
  - query-complexity
  - knowledge-distillation
  - research-computing
  - hpc
pipeline_tag: text-classification
---

# LLM Query Complexity Classifier

Fine-tuned [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) (149M parameters) for three-class query complexity classification: **LOW**, **MEDIUM**, or **HIGH**.

Built for the [STREAM](https://github.com/uicacer/STREAM) project (Smart Tiered Routing Engine for AI Models) to route queries automatically to the most cost-effective inference tier — local CPU, HPC GPU, or cloud API — at ~32 ms per query (CPU p50) with no API dependency.

Covers **10 domains** representing the full breadth of a research university population: hpc, mathematics, statistics_ml, physics_chemistry, engineering, life_sciences, cs_software, philosophy_ethics, social_sciences, and history_culture.

## What It Does

Given a user query, the model predicts how much reasoning depth is required to answer it:

| Label | Definition | Example |
|-------|------------|---------|
| `LOW` | Single retrievable fact. Answer statable in one sentence, no reasoning chain. | "What is the capital of France?" |
| `MEDIUM` | Apply an established procedure or assemble 2–4 concepts. Textbook-level reasoning. | "Explain quicksort and analyze its time complexity." |
| `HIGH` | Construct a novel reasoning path or expert judgment. No standard procedure. | "Is P equal to NP? Present the current state of evidence." |

**Key design principle**: complexity is defined by *reasoning depth*, not question format. "What is X?" can be LOW, MEDIUM, or HIGH depending on what reasoning is required to answer.

## Usage

```python
from transformers import pipeline

clf = pipeline(
    "text-classification",
    model="anasnassar/llm-query-complexity-classifier",
    device=-1,      # CPU
    top_k=None,     # return all class scores
)

result = clf("Explain the difference between TCP and UDP")
# [{'label': 'MEDIUM', 'score': 0.82}, {'label': 'LOW', 'score': 0.11}, {'label': 'HIGH', 'score': 0.07}]

complexity = max(result[0], key=lambda x: x["score"])["label"]
# 'MEDIUM'
```

## Training

**Knowledge distillation approach**: Claude Sonnet 4.6 labeled 6,000 queries using a reasoning-depth rubric. ModernBERT-base was fine-tuned on those labels. The result runs at ~32 ms per query (CPU p50) with no API dependency — a 5× latency reduction vs. the LLM judge baseline.

**Training dataset**: [anasnassar/llm-query-complexity-benchmark](https://huggingface.co/datasets/anasnassar/llm-query-complexity-benchmark) — 6,000 doubly balanced queries across 10 domains × 3 complexity classes (200/domain/class hard cap; 4,800 train / 1,200 test, 80/20 stratified split, seed=42).

**Sources**: Derived from [sentence-transformers/stackexchange-duplicates](https://huggingface.co/datasets/sentence-transformers/stackexchange-duplicates) (Apache 2.0), [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) (MIT), [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) (MIT), and [qiaojin/PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) (MIT).

**Hyperparameters**:

| Parameter | Value |
|-----------|-------|
| Base model | answerdotai/ModernBERT-base |
| Epochs | 5 |
| Batch size | 32 |
| Learning rate | 2e-5 |
| Max sequence length | 128 tokens |
| Optimizer | AdamW, weight_decay=0.01 |
| Warmup | 10% of steps |
| Best model metric | macro-F1 |

## Evaluation

Evaluated on a fixed 750-query held-out test set (250/class), stratified split, seed=42.

| Metric | Value |
|--------|-------|
| Accuracy | 64.2% |
| Macro-F1 | 0.640 |
| FREE-tier retention | 85.4% |
| Latency p50 (CPU) | 32 ms |

**Per-class recall (Wilson 95% CI):**

| Class | Recall | 95% CI |
|-------|--------|--------|
| LOW   | 70.8%  | [66.1%, 75.0%] |
| MEDIUM | 49.3% | [44.4%, 54.1%] |
| HIGH  | 72.5%  | [67.9%, 76.7%] |

## Judge Comparison

| Judge | Latency p50 | Accuracy | Macro-F1 | API dependency |
|-------|-------------|----------|----------|----------------|
| ModernBERT (this model) | 32 ms | 64.2% | 0.640 | None |
| Llama 3.2 3B (LLM judge) | 164 ms | 49.0% | 0.436 | Ollama |

## Threshold-Tunable Routing

Rather than a fixed argmax decision, STREAM exposes a tunable threshold θ ∈ [0,1]. A query is routed to cloud when `P(HIGH) ≥ θ`; otherwise to HPC or local. As θ increases, cloud spend drops but HIGH recall decreases — a continuous precision-recall-cost tradeoff.

**Budget-aware adaptive routing** automatically raises θ as cloud spend approaches the monthly budget cap:

```
θ_eff(t) = max(θ_base, S(t)/B)
```

where S(t) is cumulative spend and B is the monthly budget.

## Integration in STREAM

```python
from stream.middleware.core.complexity_judge import judge_complexity

result = judge_complexity("Explain quantum entanglement", strategy="modernbert")
# JudgmentResult(complexity='medium', method='classifier', strategy_used='modernbert',
#                scores={'low': 0.08, 'medium': 0.79, 'high': 0.13})
```

## Citation

```bibtex
@inproceedings{nassar2026stream,
  title     = {{STREAM}: Multi-Tier {LLM} Inference Middleware with Dual-Channel {HPC} Token Streaming},
  author    = {Nassar, Anas and Mohr, Steve and Apanasevich, Leonard and Sharma, Himanshu},
  booktitle = {Practice and Experience in Advanced Research Computing (PEARC '26)},
  year      = {2026},
  doi       = {10.1145/3785462.3815847}
}

@misc{nassar2026benchmark,
  author    = {Nassar, Anas},
  title     = {{LLM} Query Complexity Benchmark},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/anasnassar/llm-query-complexity-benchmark}
}

% Original source datasets
@article{hendrycks2021mmlu,
  author  = {Dan Hendrycks and others},
  title   = {Measuring Massive Multitask Language Understanding},
  journal = {ICLR},
  year    = {2021},
  url     = {https://huggingface.co/datasets/cais/mmlu}
}

@article{wang2024mmlupro,
  author  = {Yubo Wang and others},
  title   = {{MMLU-Pro}: A More Robust and Challenging Multi-Task Language Understanding Benchmark},
  journal = {arXiv:2406.01574},
  year    = {2024},
  url     = {https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro}
}

@inproceedings{jin2019pubmedqa,
  author    = {Qiao Jin and others},
  title     = {{PubMedQA}: A Biomedical Research Question Answering Dataset},
  booktitle = {EMNLP},
  year      = {2019},
  url       = {https://huggingface.co/datasets/qiaojin/PubMedQA}
}

@misc{stackexchange_dataset,
  author = {Reimers, Nils and Gurevych, Iryna},
  title  = {{StackExchange} Duplicate Questions Dataset},
  year   = {2019},
  url    = {https://huggingface.co/datasets/sentence-transformers/stackexchange-duplicates}
}
```

## License

Apache 2.0
