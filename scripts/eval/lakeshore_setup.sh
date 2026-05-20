#!/bin/bash
# One-time setup on Lakeshore: install Python deps into a venv for training.
#
# Run this interactively (NOT via SLURM) on a login or compute node:
#   bash scripts/eval/lakeshore_setup.sh
#
# Requirements:
#   - Repo already synced to Lakeshore (e.g., via rsync — see below)
#   - Python 3.11+ available (module load if needed)
#
# After setup, submit training with:
#   sbatch scripts/eval/lakeshore_imbalance_study.slurm

set -e

VENV_DIR="$HOME/.venvs/stream-eval"

echo "=== STREAM Eval Setup on Lakeshore ==="
echo "Creating venv at $VENV_DIR"

# Load Python module if needed (adjust version to what Lakeshore has)
# module load python/3.11  # uncomment if required

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip wheel

# Install PyTorch with CUDA 12.4 (matches Lakeshore drivers)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install HuggingFace + ML stack
pip install \
    transformers>=4.47.0 \
    accelerate>=1.0.0 \
    datasets>=3.0.0 \
    scikit-learn>=1.5.0 \
    numpy>=1.26.0 \
    huggingface-hub>=0.27.0

# Install repo in editable mode (for imports)
pip install -e ".[dev]" --no-deps 2>/dev/null || true

echo ""
echo "=== Setup complete ==="
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Submit job with: sbatch scripts/eval/lakeshore_imbalance_study.slurm"
