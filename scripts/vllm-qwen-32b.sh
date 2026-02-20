#!/bin/bash
#SBATCH --job-name=stream-qwen-32b
#SBATCH --partition=batch_gpu
#SBATCH --nodelist=ga-002
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.log

# =============================================================================
# Qwen 2.5 32B Instruct AWQ — General purpose model
# =============================================================================
# Hardware: A100 3g.40gb MIG slice (39.5 GiB usable VRAM)
# Weights:  ~18 GiB (32B params, AWQ 4-bit quantization)
# Context:  16K tokens (--max-model-len 16384)
#
# Memory budget (with --enforce-eager, no CUDA graphs):
#   Pre-allocated (0.9):  35.55 GiB
#   Model weights:        ~18 GiB
#   KV cache:             ~17.5 GiB (for 16K context)
#   Free for sampler:     ~4 GiB (sampler needs ~300 MiB — plenty)
#
# --enforce-eager: Required on 40GB MIG slices. vLLM v0.13.0's CUDA graph
#   capture uses ~4 GiB that then causes the sampler to OOM. Skipping CUDA
#   graphs costs ~10-15% inference speed but ensures reliable startup.
#
# IF THIS FAILS:
#   1. Reduce context: --max-model-len 8192
#   2. Lower utilization: --gpu-memory-utilization 0.85
# =============================================================================

MODEL="Qwen/Qwen2.5-32B-Instruct-AWQ"
PORT=8000

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
