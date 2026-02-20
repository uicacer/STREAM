#!/bin/bash
#SBATCH --job-name=stream-qwen-14b
#SBATCH --partition=batch_gpu
#SBATCH --nodelist=ga-002
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.log

# =============================================================================
# Qwen 2.5 14B Instruct AWQ — General purpose model
# =============================================================================
# Hardware: A100 3g.40gb MIG slice (39.5 GiB usable VRAM)
# Weights:  ~8 GiB (14B params, AWQ 4-bit quantization)
# Context:  16K tokens (--max-model-len 16384)
#
# Memory budget:
#   Total usable:  39.5 GiB
#   Model weights:  ~8 GiB
#   KV cache:      ~24 GiB (vLLM pre-allocates for 16K context)
#   Sampler:       ~300 MiB (logit sorting for top-k/top-p sampling)
#   Free headroom: ~7 GiB
#
# --enforce-eager: Disables CUDA graph capture. Required because vLLM v0.13.0
#   has an OOM bug where CUDA graphs consume ~4 GiB, leaving no room for the
#   sampler's logit sorting (~300 MiB) on 40GB MIG slices. The performance
#   impact is ~10-15% slower token generation, which is negligible on A100.
#
# --max-model-len 16384: 16K context. We use 16K instead of 32K to leave
#   enough VRAM headroom for the sampler and PyTorch internals. 16K is
#   sufficient for most conversations and code generation tasks.
# =============================================================================

MODEL="Qwen/Qwen2.5-14B-Instruct-AWQ"
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
