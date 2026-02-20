#!/bin/bash
#SBATCH --job-name=stream-qwq-32b
#SBATCH --partition=batch_gpu
#SBATCH --nodelist=ga-002
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.log

# =============================================================================
# QwQ 32B AWQ — Reasoning model (o1-style thinking)
# =============================================================================
# Hardware: A100 3g.40gb MIG slice (39.5 GiB usable VRAM)
# Weights:  ~18 GiB (32B params, AWQ 4-bit quantization)
# Context:  16K tokens (--max-model-len 16384)
#
# Same memory profile as the other 32B AWQ models.
# See vllm-qwen-32b.sh for detailed memory budget and flag explanations.
# =============================================================================

MODEL="Qwen/QwQ-32B-AWQ"
PORT=8003

echo "=========================================="
echo "STREAM vLLM: ${MODEL}"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES | Port: $PORT"
echo "Started: $(date)"
echo "=========================================="

module load apptainer

# Container stored in ACER project space (10 TiB quota) instead of home (100 GiB quota)
CONTAINER="/projects/acer_hpc_admin/nassar/containers/vllm-0.15.1"
export CUDA_VISIBLE_DEVICES=0
# Redirect HuggingFace cache to project space (home dir too small for LLM weights)
export HF_HOME=/projects/acer_hpc_admin/nassar/huggingface
NODE_IP=$(hostname -I | awk '{print $1}')
echo "Service: http://${NODE_IP}:${PORT}"

apptainer exec --nv ${CONTAINER} \
    vllm serve ${MODEL} \
    --host 0.0.0.0 \
    --port ${PORT} \
    --tensor-parallel-size 1 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.9 \
    --dtype auto \
    --quantization awq \
    --enforce-eager

echo "Service stopped: $(date)"
