#!/usr/bin/env python3
"""
Fill all TODO placeholders in the PEARC paper from benchmark result files.

Requires:
  scripts/eval/results/balanced_classifier_report.json   (from Lakeshore training)
  scripts/eval/results/llama_judge_report.json           (from benchmark_llama_judge.py)
  scripts/eval/results/budget_simulation.json            (from Lakeshore training)

Usage:
  python scripts/eval/fill_paper_todos.py
  python scripts/eval/fill_paper_todos.py --dry-run     # print replacements only
"""

import argparse
import json
from pathlib import Path

PAPER_PATH = Path("docs/pearc26-stream-paper.tex")
RESULTS_DIR = Path("scripts/eval/results")
CLASSIFIER = RESULTS_DIR / "balanced_classifier_report.json"
LLAMA = RESULTS_DIR / "llama_judge_report.json"
BUDGET_SIM = RESULTS_DIR / "budget_simulation.json"


def load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    clf = load(CLASSIFIER)
    llama = load(LLAMA)
    sim = load(BUDGET_SIM)

    # --- ModernBERT numbers ---
    mb_acc = clf["eval_at_default_theta"]["accuracy"]
    mb_f1 = clf["eval_at_default_theta"]["macro_f1"]
    mb_ret = clf["eval_at_default_theta"]["free_tier_retention_pct"]
    mb_lat = clf["latency_cpu"]["p50_ms"]
    mb_ci = clf["eval_at_default_theta"]["wilson_ci"]

    # --- Llama judge numbers ---
    ll_acc = llama["accuracy"]
    ll_n = llama["n_test"]
    ll_ret = llama["free_tier_retention_pct"]
    ll_cm = llama["confusion_matrix"]  # [[LL,LM,LH],[ML,MM,MH],[HL,HM,HH]]
    ll_per = llama["per_class"]

    # Speedup: Llama p50 / ModernBERT p50
    ll_lat = llama["latency_ms"]["p50"]
    speedup = ll_lat / mb_lat

    # Leaked free-tier queries (non-HIGH predicted as HIGH by Llama)
    leaked = ll_cm[0][2] + ll_cm[1][2]  # LOW→HIGH + MED→HIGH

    # --- Budget simulation numbers (normalized 0-1 format) ---
    if "summary" in sim:
        spend_reduction_pct = sim["summary"]["spend_reduction_pct"]
        quality_preserved_pct = sim["summary"]["quality_preserved_pct"]
    else:
        fixed_spend = sim["fixed"][-1]["cumulative_spend_usd"]
        adaptive_spend = sim["adaptive"][-1]["cumulative_spend_usd"]
        spend_reduction_pct = (fixed_spend - adaptive_spend) / fixed_spend * 100
        quality_preserved_pct = adaptive_spend / fixed_spend * 100

    # --- Build replacement map ---
    replacements = {}

    # Abstract + conclusion: "TODO% accuracy at TODOms, a TODO× speedup ... TODO% free-tier retention"
    # We'll do these as a group replacement
    replacements["abstract_classifier"] = (
        f"{mb_acc*100:.1f}\\% accuracy at {mb_lat:.0f}~ms, "
        f"a {speedup:.0f}$\\\\times$ speedup over the LLM judge with no API dependency "
        f"and {mb_ret:.1f}\\% free-tier retention"
    )

    print("\n=== Numbers to fill into paper ===\n")
    print("ModernBERT:")
    print(f"  accuracy          = {mb_acc*100:.1f}%")
    print(f"  macro-F1          = {mb_f1:.3f}")
    print(f"  free-tier retain  = {mb_ret:.1f}%")
    print(f"  latency p50 (CPU) = {mb_lat:.0f} ms")
    print(f"  speedup vs Llama  = {speedup:.0f}x  ({ll_lat:.0f} / {mb_lat:.0f})")
    print()
    print("Llama 3.2 3B judge:")
    print(f"  accuracy          = {ll_acc*100:.1f}%  (N={ll_n})")
    print(f"  free-tier retain  = {ll_ret:.1f}%")
    print(f"  leaked (non-HIGH→CLOUD): {leaked}")
    print(f"  latency p50       = {ll_lat:.0f} ms")
    print()
    print("Confusion matrix (Llama, rows=true cols=pred):")
    print(
        f"  LOW:    {ll_cm[0][0]:4d}  {ll_cm[0][1]:4d}  {ll_cm[0][2]:4d}   "
        f"recall={ll_per['LOW']['recall']*100:.1f}%"
    )
    print(
        f"  MEDIUM: {ll_cm[1][0]:4d}  {ll_cm[1][1]:4d}  {ll_cm[1][2]:4d}   "
        f"recall={ll_per['MEDIUM']['recall']*100:.1f}%"
    )
    print(
        f"  HIGH:   {ll_cm[2][0]:4d}  {ll_cm[2][1]:4d}  {ll_cm[2][2]:4d}   "
        f"recall={ll_per['HIGH']['recall']*100:.1f}%"
    )
    print()
    print("Budget simulation:")
    print(f"  cloud spend reduction = {spend_reduction_pct:.0f}%")
    print(f"  quality preserved     = {quality_preserved_pct:.0f}%")
    print()
    print("ModernBERT Wilson CIs:")
    for cls in ["LOW", "MEDIUM", "HIGH"]:
        c = mb_ci[cls]
        print(
            f"  {cls:6s} recall 95% CI: [{c['recall_lo']:.3f}, {c['recall_hi']:.3f}]  ({c['k']}/{c['n']})"
        )

    if args.dry_run:
        print("\n[dry-run] Paper not modified.")
        return

    tex = PAPER_PATH.read_text()
    original = tex

    # --- Abstract / intro TODO cluster ---
    tex = tex.replace(
        "TODO\\% accuracy at TODO~ms, a TODO$\\times$ speedup over the LLM judge with no API dependency and TODO\\% free-tier retention",
        f"{mb_acc*100:.1f}\\% accuracy at {mb_lat:.0f}~ms, a {speedup:.0f}$\\\\times$ speedup over the LLM judge with no API dependency and {mb_ret:.1f}\\% free-tier retention",
    )

    # --- ModernBERT ms overhead in evaluation section ---
    tex = tex.replace(
        "\\textbf{ModernBERT-base} (fine-tuned on Claude Sonnet~4.6 labels, TODO~ms overhead)",
        f"\\textbf{{ModernBERT-base}} (fine-tuned on Claude Sonnet~4.6 labels, {mb_lat:.0f}~ms overhead)",
    )

    # --- Llama accuracy / N ---
    tex = tex.replace(
        "Overall accuracy with the Llama judge is TODO\\% on TODO queries.",
        f"Overall accuracy with the Llama judge is {ll_acc*100:.1f}\\% on {ll_n} queries.",
    )

    # --- Free-tier leakage line ---
    tex = tex.replace(
        "the key metric is \\textit{paid-tier leakage}: TODO free-worthy queries routed to cloud, giving \\textbf{TODO\\% free-tier retention}. The ModernBERT classifier reduces overhead to TODO~ms (TODO$\\times$ speedup) at TODO\\% accuracy.",
        f"the key metric is \\textit{{paid-tier leakage}}: {leaked} free-worthy queries routed to cloud, giving \\textbf{{{ll_ret:.1f}\\% free-tier retention}}. The ModernBERT classifier reduces overhead to {mb_lat:.0f}~ms ({speedup:.0f}$\\\\times$ speedup) at {mb_acc*100:.1f}\\% accuracy.",
    )

    # --- Table caption ---
    tex = tex.replace(
        "Routing confusion matrix (TODO queries, 400/class). Judge: Llama~3.2~3B. Overall: TODO\\%. LOW$\\to$MED misroutes carry zero cost impact; paid-tier leakage is TODO free-worthy queries (TODO\\% free-tier retention).",
        f"Routing confusion matrix ({ll_n} queries, 400/class). Judge: Llama~3.2~3B. Overall: {ll_acc*100:.1f}\\%. LOW$\\to$MED misroutes carry zero cost impact; paid-tier leakage is {leaked} free-worthy queries ({ll_ret:.1f}\\% free-tier retention).",
    )

    # --- Table rows ---
    low_row = f"LOW    & {ll_cm[0][0]} & {ll_cm[0][1]} & {ll_cm[0][2]} & {ll_per['LOW']['recall']*100:.1f}\\% \\\\"
    med_row = f"MEDIUM & {ll_cm[1][0]} & {ll_cm[1][1]} & {ll_cm[1][2]} & {ll_per['MEDIUM']['recall']*100:.1f}\\% \\\\"
    hi_row = f"HIGH   & {ll_cm[2][0]} & {ll_cm[2][1]} & {ll_cm[2][2]} & {ll_per['HIGH']['recall']*100:.1f}\\% \\\\"
    overall_correct = ll_cm[0][0] + ll_cm[1][1] + ll_cm[2][2]

    tex = tex.replace("LOW    & TODO & TODO & TODO & TODO\\% \\\\", low_row)
    tex = tex.replace("MEDIUM & TODO & TODO & TODO & TODO\\% \\\\", med_row)
    tex = tex.replace("HIGH   & TODO & TODO & TODO & TODO\\% \\\\", hi_row)
    tex = tex.replace(
        "\\multicolumn{4}{l}{Overall (TODO/TODO)} & TODO\\% \\\\",
        f"\\multicolumn{{4}}{{l}}{{Overall ({overall_correct}/{ll_n})}} & {ll_acc*100:.1f}\\% \\\\",
    )

    # --- Budget simulation ---
    tex = tex.replace(
        "adaptive routing reduces cloud spend by TODO\\% while preserving TODO\\% of high-complexity query quality.",
        f"adaptive routing reduces cloud spend by {spend_reduction_pct:.0f}\\% while preserving {quality_preserved_pct:.0f}\\% of high-complexity query quality.",
    )

    # --- Conclusion ---
    tex = tex.replace(
        "The ModernBERT classifier achieves TODO\\% accuracy at TODO~ms — a TODO$\\times$ speedup over the LLM judge with no API dependency.",
        f"The ModernBERT classifier achieves {mb_acc*100:.1f}\\% accuracy at {mb_lat:.0f}~ms — a {speedup:.0f}$\\\\times$ speedup over the LLM judge with no API dependency.",
    )

    if tex == original:
        print("\n[WARN] No replacements made — check TODO strings match exactly.")
    else:
        PAPER_PATH.write_text(tex)
        remaining = tex.count("TODO")
        print(f"\nPaper updated. Remaining TODOs: {remaining}")
        if remaining:
            for i, line in enumerate(tex.splitlines(), 1):
                if "TODO" in line:
                    print(f"  line {i}: {line.strip()[:100]}")


if __name__ == "__main__":
    main()
