#!/bin/bash
#SBATCH --job-name=stream-qwen-1.5b
#SBATCH --partition=batch_gpu
#SBATCH --nodelist=ga-002
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.log

# =============================================================================
# Qwen 2.5 1.5B Instruct — Fast demo model
# =============================================================================
# Hardware: A100 3g.40gb MIG slice (39.5 GiB usable VRAM)
# Weights:  ~3 GiB (1.5B params, FP16 — small enough without quantization)
# Context:  32K tokens (plenty of room on 40GB)
#
# This model is much faster than the 32B AWQ models (~100+ tok/s vs ~15 tok/s).
# No --enforce-eager or --quantization needed — the model is small enough
# that CUDA graphs, KV cache, and sampler all fit comfortably.
# =============================================================================

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
PORT=8000

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
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9

echo "Service stopped: $(date)"
