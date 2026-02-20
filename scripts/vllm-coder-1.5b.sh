#!/bin/bash
#SBATCH --job-name=stream-coder-1.5b
#SBATCH --partition=batch_gpu
#SBATCH --nodelist=ga-002
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.log

# =============================================================================
# Qwen 2.5 Coder 1.5B Instruct — Coding specialist (fast demo)
# =============================================================================
# Hardware: A100 3g.40gb MIG slice (39.5 GiB usable VRAM)
# Weights:  ~3 GiB (1.5B params, FP16 — small enough without quantization)
# Context:  32K tokens (plenty of room on 40GB)
#
# Same memory profile as Qwen 2.5 1.5B. See vllm-qwen-1.5b.sh for details.
# =============================================================================

MODEL="Qwen/Qwen2.5-Coder-1.5B-Instruct"
PORT=8001

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
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9

echo "Service stopped: $(date)"
