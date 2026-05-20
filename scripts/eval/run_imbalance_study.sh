#!/bin/bash
# Run all 6 conditions for the ModernBERT class imbalance + loss study.
# Each condition: train 10-fold CV on 4,608-query dataset → report results.
#
# Dataset: 1,536 queries × 3 sources (MMLU, SE general, SE HPC-filtered)
# = 4,608 total. Per-source size derived from Wald formula: guarantees
# total HIGH queries ≥ 384 (±5pp 95% CI on minority class).
#
# LLM judge (Llama 3.2 3B) runs ONCE on the baseline condition only.
# All other conditions skip the judge — it runs on the same fixed test
# folds regardless of which ModernBERT variant trained.
#
# Total estimated time: ~17 hours (6 conditions × 10 folds × ~17 min/fold)
#
# Run overnight:
#   mkdir -p logs
#   nohup bash scripts/eval/run_imbalance_study.sh > logs/imbalance_study.log 2>&1 &
#   echo "Running as PID $!"

PYTHON="${PYTHON:-uv run python3}"
BASE_DATASET="scripts/eval/mixed_training_dataset.json"
OVERSAMPLE_DATASET="scripts/eval/mixed_oversample_dataset.json"
DOWNSAMPLE_DATASET="scripts/eval/mixed_downsample_dataset.json"

mkdir -p scripts/eval/results
mkdir -p logs

echo "============================================================"
echo "Class Imbalance Study — $(date)"
echo "6 conditions × 10-fold CV × Arena OOD (1,200 queries)"
echo "Dataset: 4,608 queries (1,536 × MMLU + SE general + SE HPC)"
echo "LLM judge runs once (baseline only) — reused for comparison"
echo "============================================================"

echo ""
echo "--- Building dataset variants (reusing labels, no API cost) ---"

echo "Building oversample variant..."
$PYTHON scripts/eval/build_mixed_dataset.py \
    --reuse-mmlu --reuse-stackexchange --reuse-research-computing \
    --reuse-from "$BASE_DATASET" \
    --balance-mode oversample \
    --output "$OVERSAMPLE_DATASET"

echo "Building downsample variant..."
$PYTHON scripts/eval/build_mixed_dataset.py \
    --reuse-mmlu --reuse-stackexchange --reuse-research-computing \
    --reuse-from "$BASE_DATASET" \
    --balance-mode downsample \
    --output "$DOWNSAMPLE_DATASET"

echo ""
echo "============================================================"
echo "--- Condition 1/6: Baseline (imbalanced, uniform loss) ---"
echo "--- Dataset: $BASE_DATASET ---"
echo "--- LLM judge: YES (runs once, reused by all other conditions) ---"
echo "============================================================"
$PYTHON scripts/eval/train_modernbert.py \
    --eval-mode mixed-kfold \
    --dataset "$BASE_DATASET" \
    --n-folds 10 \
    --condition-name baseline || echo "[WARN] Condition 1 failed"

echo ""
echo "============================================================"
echo "--- Condition 2/6: Class weights (imbalanced, weighted loss) ---"
echo "--- Dataset: $BASE_DATASET ---"
echo "--- LLM judge: SKIPPED ---"
echo "============================================================"
$PYTHON scripts/eval/train_modernbert.py \
    --eval-mode mixed-kfold \
    --dataset "$BASE_DATASET" \
    --n-folds 10 \
    --skip-llm-judge \
    --condition-name weighted || echo "[WARN] Condition 2 failed"

echo ""
echo "============================================================"
echo "--- Condition 3/6: Oversample (balanced, uniform loss) ---"
echo "--- Dataset: $OVERSAMPLE_DATASET ---"
echo "--- LLM judge: SKIPPED ---"
echo "============================================================"
$PYTHON scripts/eval/train_modernbert.py \
    --eval-mode mixed-kfold \
    --dataset "$OVERSAMPLE_DATASET" \
    --n-folds 10 \
    --no-class-weights \
    --skip-llm-judge \
    --condition-name oversample || echo "[WARN] Condition 3 failed"

echo ""
echo "============================================================"
echo "--- Condition 4/6: Downsample (balanced, uniform loss) ---"
echo "--- Dataset: $DOWNSAMPLE_DATASET ---"
echo "--- LLM judge: SKIPPED ---"
echo "============================================================"
$PYTHON scripts/eval/train_modernbert.py \
    --eval-mode mixed-kfold \
    --dataset "$DOWNSAMPLE_DATASET" \
    --n-folds 10 \
    --no-class-weights \
    --skip-llm-judge \
    --condition-name downsample || echo "[WARN] Condition 4 failed"

echo ""
echo "============================================================"
echo "--- Condition 5/6: Cost-sensitive loss (alpha=1, gamma=1) ---"
echo "--- Dataset: $BASE_DATASET ---"
echo "--- LLM judge: SKIPPED ---"
echo "============================================================"
$PYTHON scripts/eval/train_modernbert.py \
    --eval-mode mixed-kfold \
    --dataset "$BASE_DATASET" \
    --n-folds 10 \
    --cost-sensitive \
    --fixed-alpha 1.0 \
    --skip-llm-judge \
    --condition-name cost_sensitive || echo "[WARN] Condition 5 failed"

echo ""
echo "============================================================"
echo "--- Condition 6/6: Cost-sensitive loss (alpha=1, gamma=0) ---"
echo "--- LOW->MED over-routing cost = 0 (both free tiers) ---"
echo "--- Dataset: $BASE_DATASET ---"
echo "--- LLM judge: SKIPPED ---"
echo "============================================================"
$PYTHON scripts/eval/train_modernbert.py \
    --eval-mode mixed-kfold \
    --dataset "$BASE_DATASET" \
    --n-folds 10 \
    --cost-sensitive \
    --fixed-alpha 1.0 \
    --zero-hpc-overroute \
    --skip-llm-judge \
    --condition-name cost_sensitive_gamma0 || echo "[WARN] Condition 6 failed"

echo ""
echo "============================================================"
echo "All conditions complete — $(date)"
echo "============================================================"
echo ""
echo "Results summary:"
$PYTHON scripts/eval/compare_imbalance_conditions.py

echo ""
echo "Individual reports:"
ls -lh scripts/eval/results/modernbert_*_kfold_report.json 2>/dev/null || echo "  (check scripts/eval/results/)"
