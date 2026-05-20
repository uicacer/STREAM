#!/usr/bin/env python3
"""
Compare all 6 class imbalance conditions side-by-side.

Reads:
    scripts/eval/results/modernbert_baseline_kfold_report.json
    scripts/eval/results/modernbert_weighted_kfold_report.json
    scripts/eval/results/modernbert_oversample_kfold_report.json
    scripts/eval/results/modernbert_downsample_kfold_report.json
    scripts/eval/results/modernbert_cost_sensitive_kfold_report.json
    scripts/eval/results/modernbert_cost_sensitive_gamma0_kfold_report.json

Dataset: 4,608 queries (1,536 × MMLU + SE general + SE HPC-filtered), 10-fold CV.

Usage:
    python scripts/eval/compare_imbalance_conditions.py
"""

import json
from pathlib import Path

import numpy as np

RESULTS_DIR = Path("scripts/eval/results")
LAKESHORE_DIR = Path("scripts/eval/results/lakeshore")

CONDITIONS = [
    ("baseline", "Baseline (imbalanced, uniform loss)", 4608),
    ("weighted", "Class weights (imbalanced, weighted loss)", 4608),
    ("oversample", "Oversample to majority count (~6,084 total)", 6084),
    ("downsample", "Downsample to minority count (~1,158 total)", 1158),
    ("cost_sensitive", "Cost-sensitive loss (α=1, γ=1)", 4608),
    ("cost_sensitive_gamma0", "Cost-sensitive loss (α=1, γ=0, LOW→MED free)", 4608),
]

# Tier names for readable output
TIERS = ["LOCAL", "HPC", "CLOUD"]  # maps to pred [LOW, MEDIUM, HIGH]
CLASSES = ["LOW", "MEDIUM", "HIGH"]


def load_report(condition: str) -> dict | None:
    # Prefer lakeshore (authoritative A100 10-fold results) over local
    for d in [LAKESHORE_DIR, RESULTS_DIR]:
        path = d / f"modernbert_{condition}_kfold_report.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


def fmt(val, std=None, pct=False):
    if val is None:
        return "pending"
    s = f"{val:.1%}" if pct else f"{val:.3f}"
    if std is not None:
        s += f" ± {std:.1%}" if pct else f" ± {std:.3f}"
    return s


def fold_cm_rates(folds):
    """For each fold return the 3×3 rate matrix (each row sums to 1)."""
    rates = []
    for fold in folds:
        cm = fold["confusion_matrix"]  # rows=true, cols=pred; order [LOW, MED, HIGH]
        row_rates = []
        for i in range(3):
            total = max(sum(cm[i]), 1)
            row_rates.append([cm[i][j] / total for j in range(3)])
        rates.append(row_rates)
    return rates  # shape: (n_folds, 3, 3)


def mean_std_cell(rates, true_cls, pred_cls):
    vals = [r[true_cls][pred_cls] for r in rates]
    return np.mean(vals), np.std(vals)


def main():
    print("\n" + "=" * 100)
    print("Class Imbalance Study — ModernBERT Query Complexity Classifier")
    print("=" * 100)

    rows = []
    for cond, label, n_train in CONDITIONS:
        r = load_report(cond)
        rows.append((label, n_train, r))

    # -------------------------------------------------------------------------
    # Table 1: Overall in-distribution metrics
    # -------------------------------------------------------------------------
    print(f"\n{'─'*100}")
    print("TABLE 1: In-distribution (10-fold CV on 4,608-query mixed dataset)")
    print(f"{'─'*100}")
    print(
        f"  {'Condition':<43} {'N':>6} {'Accuracy':>14} {'Macro-F1':>14} {'HIGH F1':>12} {'Retention':>12}"
    )
    print(f"  {'─'*43} {'─'*6} {'─'*14} {'─'*14} {'─'*12} {'─'*12}")
    for label, _n_train, r in rows:
        if r is None:
            print(f"  {label:<43} {n_train:>6}   (pending)")
            continue
        high_f1s = [f["per_class"]["HIGH"]["f1"] for f in r["folds"]]
        print(
            f"  {label:<43} {n_train:>6}"
            f"  {fmt(r['mean_accuracy'], r['std_accuracy'], pct=True):>14}"
            f"  {fmt(r['mean_macro_f1'], r['std_macro_f1']):>14}"
            f"  {fmt(np.mean(high_f1s), np.std(high_f1s)):>12}"
            f"  {fmt(r['mean_free_tier_retention_pct']/100, r['std_free_tier_retention_pct']/100, pct=True):>12}"
        )

    # -------------------------------------------------------------------------
    # Table 2: Arena OOD
    # -------------------------------------------------------------------------
    print(f"\n{'─'*100}")
    print("TABLE 2: Out-of-distribution (Arena OOD — fixed real-user test set)")
    print(f"{'─'*100}")
    print(f"  {'Condition':<43} {'Arena Acc':>14} {'Arena F1':>14} {'Arena Retention':>16}")
    print(f"  {'─'*43} {'─'*14} {'─'*14} {'─'*16}")
    for label, _n_train, r in rows:
        if r is None or r.get("arena_mean_accuracy") is None:
            print(f"  {label:<43}   (pending)")
            continue
        print(
            f"  {label:<43}"
            f"  {fmt(r['arena_mean_accuracy'], r['arena_std_accuracy'], pct=True):>14}"
            f"  {fmt(r['arena_mean_macro_f1'], r['arena_std_macro_f1']):>14}"
            f"  {fmt(r['arena_mean_free_tier_retention_pct']/100, r['arena_std_free_tier_retention_pct']/100, pct=True):>16}"
        )

    # -------------------------------------------------------------------------
    # Table 3: Full tier-leakage matrix — ALL 6 off-diagonal directions
    # -------------------------------------------------------------------------
    print(f"\n{'─'*100}")
    print("TABLE 3: Full tier-leakage matrix (mean ± std across folds)")
    print("  Rows = true class, Cols = predicted class")
    print("  Off-diagonal = routing error rate for that class")
    print(f"{'─'*100}")

    header = f"  {'Condition':<43}  {'TRUE LOW':^28}  {'TRUE MEDIUM':^28}  {'TRUE HIGH':^28}"
    subhdr = f"  {'':43}  {'→MED':>8} {'→HIGH':>8} {'→LOW':>8} {'→HIGH':>8} {'→LOW':>8} {'→MED':>8}"
    print(header)
    print(subhdr)
    print(f"  {'─'*43}  {'─'*8} {'─'*8}  {'─'*8} {'─'*8}  {'─'*8} {'─'*8}")

    for label, _n_train, r in rows:
        if r is None:
            print(f"  {label:<43}   (pending)")
            continue
        rates = fold_cm_rates(r["folds"])
        # LOW row: →MED, →HIGH
        l_m = mean_std_cell(rates, 0, 1)
        l_h = mean_std_cell(rates, 0, 2)
        # MEDIUM row: →LOW, →HIGH
        m_l = mean_std_cell(rates, 1, 0)
        m_h = mean_std_cell(rates, 1, 2)
        # HIGH row: →LOW, →MED
        h_l = mean_std_cell(rates, 2, 0)
        h_m = mean_std_cell(rates, 2, 1)
        print(
            f"  {label:<43}"
            f"  {l_m[0]:>5.1%}±{l_m[1]:.1%}  {l_h[0]:>5.1%}±{l_h[1]:.1%}"
            f"  {m_l[0]:>5.1%}±{m_l[1]:.1%}  {m_h[0]:>5.1%}±{m_h[1]:.1%}"
            f"  {h_l[0]:>5.1%}±{h_l[1]:.1%}  {h_m[0]:>5.1%}±{h_m[1]:.1%}"
        )

    # -------------------------------------------------------------------------
    # Table 4: Aggregate operational metrics
    # -------------------------------------------------------------------------
    print(f"\n{'─'*100}")
    print("TABLE 4: Aggregate operational metrics")
    print(
        "  Monetary cost leak  = (LOW→HIGH + MED→HIGH) / all non-HIGH        [unnecessary cloud spend]"
    )
    print(
        "  HPC resource waste  = LOW→MED / all LOW                           [trivial queries on shared GPU]"
    )
    print(
        "  Quality leakage     = (MED→LOW + HIGH→MED + HIGH→LOW) / (all MED + all HIGH)  [under-served queries]"
    )
    print(
        "  Severe qual leakage = HIGH→LOW / all HIGH                         [hard query to local 1B model]"
    )
    print(
        "  HIGH recall         = HIGH→HIGH / all HIGH                        [hard queries correctly caught]"
    )
    print(f"{'─'*100}")
    print(
        f"  {'Condition':<43} {'Cost leak':>12} {'HPC waste':>11} {'Qual leak':>11} {'Severe qual':>12} {'HIGH recall':>12}"
    )
    print(f"  {'─'*43} {'─'*12} {'─'*11} {'─'*11} {'─'*12} {'─'*12}")

    for label, _n_train, r in rows:
        if r is None:
            print(f"  {label:<43}   (pending)")
            continue
        cost_leaks, hpc_wastes, qual_leaks, severe_leaks, high_recalls = [], [], [], [], []
        for fold in r["folds"]:
            cm = fold["confusion_matrix"]
            # Monetary cost: non-HIGH routed to HIGH (cloud)
            non_high_total = max(sum(cm[0]) + sum(cm[1]), 1)
            cost_leak = (cm[0][2] + cm[1][2]) / non_high_total
            # HPC resource waste: LOW routed to MEDIUM (shared GPU)
            low_total = max(sum(cm[0]), 1)
            hpc_waste = cm[0][1] / low_total
            # Quality leakage: any under-routing (MED→LOW, HIGH→MED, HIGH→LOW)
            med_total = max(sum(cm[1]), 1)
            high_total = max(sum(cm[2]), 1)
            qual_leak = (cm[1][0] + cm[2][1] + cm[2][0]) / max(med_total + high_total, 1)
            # Severe: HIGH routed to LOCAL
            severe = cm[2][0] / high_total
            recall = cm[2][2] / high_total
            cost_leaks.append(cost_leak)
            hpc_wastes.append(hpc_waste)
            qual_leaks.append(qual_leak)
            severe_leaks.append(severe)
            high_recalls.append(recall)
        print(
            f"  {label:<43}"
            f"  {np.mean(cost_leaks):>5.1%}±{np.std(cost_leaks):.1%}"
            f"  {np.mean(hpc_wastes):>5.1%}±{np.std(hpc_wastes):.1%}"
            f"  {np.mean(qual_leaks):>5.1%}±{np.std(qual_leaks):.1%}"
            f"  {np.mean(severe_leaks):>5.1%}±{np.std(severe_leaks):.1%}"
            f"  {np.mean(high_recalls):>5.1%}±{np.std(high_recalls):.1%}"
        )

    # -------------------------------------------------------------------------
    # Table 5: LLM judge comparison
    # LLM judge results come from the baseline condition only (judge runs on
    # the same fixed test folds regardless of which ModernBERT variant trained,
    # so running it once is sufficient for all conditions).
    # -------------------------------------------------------------------------
    baseline_r = load_report("baseline")
    judge_f1 = baseline_r.get("llm_judge_mean_macro_f1") if baseline_r else None
    judge_f1_std = baseline_r.get("llm_judge_std_macro_f1") if baseline_r else None

    print(f"\n{'─'*100}")
    print("TABLE 5: LLM judge comparison (Llama 3.2 3B via Ollama — run once on baseline folds)")
    print(f"{'─'*100}")
    print(
        f"  {'Condition':<43} {'ModernBERT F1':>14} {'LLM judge F1':>14} {'Δ F1':>8} {'Speedup':>8}"
    )
    print(f"  {'─'*43} {'─'*14} {'─'*14} {'─'*8} {'─'*8}")
    for label, _n_train, r in rows:
        if r is None:
            print(f"  {label:<43}   (pending)")
            continue
        if judge_f1 is None:
            print(
                f"  {label:<43}  {fmt(r['mean_macro_f1'], r['std_macro_f1']):>14}   (judge pending)"
            )
            continue
        delta = r["mean_macro_f1"] - judge_f1
        print(
            f"  {label:<43}"
            f"  {fmt(r['mean_macro_f1'], r['std_macro_f1']):>14}"
            f"  {fmt(judge_f1, judge_f1_std):>14}"
            f"  {delta:>+8.3f}"
            f"  {'~26×':>8}"
        )

    print("\n" + "=" * 100)
    print("Interpretation guide:")
    print("  Monetary cost leak  → lower is better (less unnecessary cloud API spend)")
    print("  HPC resource waste  → lower is better (less shared GPU wasted on trivial queries)")
    print(
        "  Quality leakage     → lower is better (MED→LOW + HIGH→MED + HIGH→LOW all under-serve users)"
    )
    print("  Severe qual leakage → lower is better (HIGH→LOCAL is the worst failure mode)")
    print("  HIGH recall         → higher is better (more hard queries correctly caught)")
    print("  The best condition minimizes ALL leakage directions simultaneously.")
    print("  Tradeoff: reducing quality leakage typically increases cost/HPC waste — report both.")
    print("=" * 100)


if __name__ == "__main__":
    main()
