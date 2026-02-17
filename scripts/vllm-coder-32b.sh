#!/bin/bash
#SBATCH --job-name=stream-coder-32b
#SBATCH --partition=batch_gpu
#SBATCH --nodelist=ga-002
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.log

# =============================================================================
# Qwen 2.5 Coder 32B Instruct AWQ — Coding specialist
# =============================================================================
# Hardware: A100 3g.40gb MIG slice (39.5 GiB usable VRAM)
# Weights:  ~18 GiB (32B params, AWQ 4-bit quantization)
# Context:  16K tokens (--max-model-len 16384)
#
# Same memory profile as Qwen 2.5 32B (same architecture, same param count).
# See vllm-qwen-32b.sh for detailed memory budget and flag explanations.
# =============================================================================

MODEL="Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"
PORT=8001

echo "=========================================="
echo "STREAM vLLM: ${MODEL}"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES | Port: $PORT"
echo "Started: $(date)"
echo "=========================================="

module load apptainer

CONTAINER="/home/nassar/STREAM/containers/vllm-openai_v0.13.0.sif"
export CUDA_VISIBLE_DEVICES=0

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
