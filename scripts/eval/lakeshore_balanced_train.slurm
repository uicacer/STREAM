#!/bin/bash
#SBATCH --job-name=stream-balanced
#SBATCH --partition=batch_gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/balanced_train_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nassar@uic.edu

REPO_DIR="$HOME/STREAM/classifier"
VENV_DIR="$HOME/.venvs/stream-eval"

cd "$REPO_DIR"
source "$VENV_DIR/bin/activate"

python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

mkdir -p logs results figures

echo "============================================================"
echo "STREAM Balanced Classifier — $(date)"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "============================================================"

echo ""
echo "--- Training ModernBERT on balanced dataset + θ sweep ---"
python3 train_balanced_classifier.py \
    --train-dataset balanced_training_dataset.json \
    --test-dataset  balanced_test_dataset.json \
    --theta-default 0.5 \
    --budget 10.0 || { echo "[ERROR] Training failed"; exit 1; }

echo ""
echo "--- Generating figures ---"
python3 plot_figures.py || echo "[WARN] plot_figures failed (matplotlib may be missing)"

echo ""
echo "=== Done: $(date) ==="
echo ""
echo "Results:"
ls -lh results/balanced_classifier_report.json \
        results/theta_sweep.json \
        results/budget_simulation.json 2>/dev/null
ls -lh models/modernbert_balanced/ 2>/dev/null
