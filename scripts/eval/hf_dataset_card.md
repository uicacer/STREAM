---
license: apache-2.0
task_categories:
  - text-classification
language:
  - en
tags:
  - llm-routing
  - query-complexity
  - knowledge-distillation
  - research-computing
  - hpc
  - modernbert
  - multi-domain
size_categories:
  - 1K<n<10K
source_datasets:
  - sentence-transformers/stackexchange-duplicates
  - cais/mmlu
  - TIGER-Lab/MMLU-Pro
  - qiaojin/PubMedQA
  - synthetic (Claude Sonnet 4.6)
configs:
  - config_name: default
    data_files:
      - split: train
        path: train.jsonl
      - split: test
        path: test.jsonl
---

# LLM Query Complexity Benchmark

A **multi-domain, perfectly balanced** dataset of **6,000 labeled queries** (4,800 train / 1,200 test) for training and evaluating LLM query complexity classifiers that route queries to the most cost-effective inference tier.

Built for the [STREAM](https://github.com/uicacer/STREAM) project (Smart Tiered Routing Engine for AI Models), which routes queries automatically between local CPU models, institutional HPC GPU clusters, and cloud API tiers.

## Dataset Summary

| Split | Queries | LOW | MEDIUM | HIGH |
|-------|---------|-----|--------|------|
| Train | 4,800 | 1,600 | 1,600 | 1,600 |
| Test  | 1,200 | 400 | 400 | 400 |
| **Total** | **6,000** | **2,000** | **2,000** | **2,000** |

The dataset is **doubly balanced**: exactly 200 queries per (domain × class) cell across 10 domains, and exactly equal class counts in both splits.

## Domain Coverage

Designed to represent the full breadth of a research university student and researcher population — not just HPC practitioners.

| Domain | Primary Sources | Description |
|--------|----------------|-------------|
| `hpc` | Stack Exchange | HPC, SLURM, MPI, CUDA, Globus, parallel computing |
| `mathematics` | Stack Exchange, MMLU-Pro | Pure and applied math, proofs, algorithms |
| `statistics_ml` | Stack Exchange, MMLU | Statistics, ML, data science, probability |
| `physics_chemistry` | Stack Exchange, MMLU, MMLU-Pro | Physics, chemistry, materials science |
| `engineering` | Stack Exchange, MMLU, MMLU-Pro | EE, software engineering, systems |
| `life_sciences` | Stack Exchange, MMLU, MMLU-Pro, PubMedQA | Biology, medicine, biochemistry |
| `cs_software` | Stack Exchange, MMLU-Pro | Programming, systems, databases, compilers |
| `philosophy_ethics` | MMLU, MMLU-Pro | Philosophy, ethics, logic, jurisprudence |
| `social_sciences` | MMLU, MMLU-Pro | Psychology, sociology, economics, law |
| `history_culture` | MMLU, MMLU-Pro | History, world religions, geography |

## Source Datasets

All queries are derived from the following original datasets. Each is used under its respective license:

| Dataset | HuggingFace ID | License | Used for |
|---------|---------------|---------|---------|
| Stack Exchange Duplicates | [`sentence-transformers/stackexchange-duplicates`](https://huggingface.co/datasets/sentence-transformers/stackexchange-duplicates) | Apache 2.0 | STEM and HPC domains (hpc, mathematics, statistics_ml, physics_chemistry, engineering, life_sciences, cs_software) |
| MMLU | [`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu) | MIT | humanities and supplement for STEM domains (philosophy_ethics, social_sciences, history_culture, plus electrical_engineering, college_physics, etc.) |
| MMLU-Pro | [`TIGER-Lab/MMLU-Pro`](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | MIT | Higher-difficulty supplement across all domains; especially important for MEDIUM/HIGH class balance in engineering, life_sciences, cs_software |
| PubMedQA | [`qiaojin/PubMedQA`](https://huggingface.co/datasets/qiaojin/PubMedQA) | MIT | HIGH-complexity supplement for life_sciences (biomedical research questions requiring expert reasoning) |
| Synthetic (Claude Sonnet 4.6) | — | Apache 2.0 | HIGH-complexity top-up for hpc (114), philosophy_ethics (160), and history_culture (139) — domains where existing sources were exhausted at the HIGH class. Generated and verified by Claude Sonnet 4.6 using domain-specific expert prompts. |

**How source data was transformed:**
- Stack Exchange: question titles only (not body or answers); filtered for length (25–200 chars), language (≥85% ASCII), and structure (≥4 words, ≥35% alpha); error-message and formula-only titles excluded
- MMLU: question stem only — answer choices (A/B/C/D) were **not included**, and questions referencing "the following" without their options were excluded to avoid format confound
- MMLU-Pro: question stem only; same exclusion criteria as MMLU
- PubMedQA: question text only (not context passages or answers); all treated as HIGH complexity by design (biomedical research reasoning)
- Synthetic: for three domains (hpc, philosophy_ethics, history_culture) where existing sources were exhausted before reaching 200 HIGH queries, Claude Sonnet 4.6 was used to generate expert-level HIGH questions using domain-specific prompts emphasizing novel reasoning paths and expert judgment. Each generated question was independently verified by Claude before inclusion (only questions labeled HIGH on re-evaluation were kept).

## Complexity Classes

Complexity is defined by **reasoning depth**, not question format or length. "What is X?" can be LOW, MEDIUM, or HIGH depending on the reasoning required.

| Class | Definition | Example |
|-------|------------|---------|
| `LOW` | Single retrievable fact or trivial computation. Answer statable in one sentence, no reasoning chain required. | "What is the capital of France?" |
| `MEDIUM` | Apply an established procedure or assemble 2–4 concepts. The reasoning path is standard (textbook-level). | "Explain how quicksort works and analyze its time complexity." |
| `HIGH` | Construct a novel reasoning path, formal derivation, or expert judgment. No standard procedure — the path must be built. | "Is P equal to NP? Present the current state of evidence." |

## Dataset Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Unique query identifier within the split |
| `text` | string | The user query |
| `domain` | string | One of the 10 domains above |
| `source` | string | Original dataset: `stackexchange`, `mmlu`, `mmlu_pro`, `pubmedqa`, or `synthetic_high` |
| `subject` | string | MMLU subject name, SE tag, `pubmedqa`, or `{domain}_synthetic` |
| `ground_truth` | string | Complexity class: `LOW`, `MEDIUM`, or `HIGH` |

## Labeling Method

All queries were labeled by **Claude Sonnet 4.6** (via the Anthropic API) using a reasoning-depth rubric that explicitly decouples complexity from surface format. Queries are sent in batches of 100; the labeler is stopped once exactly 200 per (domain × class) cell are collected.

This is **knowledge distillation via label transfer**: Claude pays the API cost once at dataset-construction time; a smaller classifier (ModernBERT-base, 149M parameters) learns to replicate its routing decisions at inference time with no API call, negligible latency, and full data privacy.

## Labeling Rubric (abridged)

The following reasoning-depth scale was used:

- **LOW**: The answer is a single fact, definition, or trivial computation directly retrievable from memory. No multi-step reasoning. Examples: capitals, dates, unit conversions, simple syntax lookups.
- **MEDIUM**: The answer requires applying a known procedure, combining 2–4 standard concepts, or working through a textbook-level derivation. The path is established; the task is to execute it correctly.
- **HIGH**: The answer requires constructing a novel reasoning path, synthesizing across disciplines, producing a formal proof, or making expert judgment under ambiguity. No standard procedure applies.

## Loading the Dataset

```python
from datasets import load_dataset

ds = load_dataset("anasnassar/llm-query-complexity-benchmark")
print(ds)
# DatasetDict({
#     train: Dataset({features: ['id', 'text', 'domain', 'source', 'subject', 'ground_truth'], num_rows: 4800})
#     test:  Dataset({features: ['id', 'text', 'domain', 'source', 'subject', 'ground_truth'], num_rows: 1200})
# })

# Filter by domain
hpc_train = ds["train"].filter(lambda x: x["domain"] == "hpc")

# Class distribution
from collections import Counter
print(Counter(ds["train"]["ground_truth"]))
# Counter({'LOW': 1600, 'MEDIUM': 1600, 'HIGH': 1600})
```

## Training the Classifier

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

ds = load_dataset("anasnassar/llm-query-complexity-benchmark")

label2id = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

def tokenize(batch):
    enc = tokenizer(batch["text"], truncation=True, max_length=128, padding="max_length")
    enc["label"] = [label2id[lb] for lb in batch["ground_truth"]]
    return enc

tokenized = ds.map(tokenize, batched=True)
model = AutoModelForSequenceClassification.from_pretrained(
    "answerdotai/ModernBERT-base", num_labels=3,
    id2label={0: "LOW", 1: "MEDIUM", 2: "HIGH"},
    label2id=label2id,
)

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="modernbert-complexity",
        num_train_epochs=5,
        per_device_train_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        metric_for_best_model="eval_loss",
    ),
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
)
trainer.train()
```

## Associated Model

The fine-tuned ModernBERT-base classifier trained on this dataset:
[anasnassar/llm-query-complexity-classifier](https://huggingface.co/anasnassar/llm-query-complexity-classifier)

## Reproducing the Dataset

```bash
git clone https://github.com/uicacer/STREAM
cd STREAM
pip install anthropic datasets

# Requires ANTHROPIC_API_KEY (~$3-5 in API cost for 6,000 labels)
python scripts/eval/build_balanced_dataset.py
```

## Citation

If you use this dataset, please cite the STREAM paper and acknowledge the original source datasets:

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
```

**Original source datasets:**

```bibtex
@inproceedings{hendrycks2021mmlu,
  title   = {Measuring Massive Multitask Language Understanding},
  author  = {Hendrycks, Dan and others},
  booktitle = {ICLR},
  year    = {2021},
  url     = {https://huggingface.co/datasets/cais/mmlu}
}

@article{wang2024mmlupro,
  title   = {{MMLU-Pro}: A More Robust and Challenging Multi-Task Language Understanding Benchmark},
  author  = {Wang, Yubo and others},
  journal = {arXiv:2406.01574},
  year    = {2024},
  url     = {https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro}
}

@inproceedings{jin2019pubmedqa,
  title   = {{PubMedQA}: A Biomedical Research Question Answering Dataset},
  author  = {Jin, Qiao and others},
  booktitle = {EMNLP},
  year    = {2019},
  url     = {https://huggingface.co/datasets/qiaojin/PubMedQA}
}

@misc{stackexchange2024duplicates,
  title   = {Stack Exchange Duplicate Questions Dataset},
  author  = {{sentence-transformers}},
  year    = {2024},
  url     = {https://huggingface.co/datasets/sentence-transformers/stackexchange-duplicates}
}
```

## License

Apache 2.0. Original source datasets are used under their respective licenses (Apache 2.0 and MIT). Labels were generated by Claude Sonnet 4.6 (Anthropic) and are released under Apache 2.0.
