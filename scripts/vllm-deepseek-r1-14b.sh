#!/bin/bash
#SBATCH --job-name=stream-deepseek-r1-14b
#SBATCH --partition=batch_gpu
#SBATCH --nodelist=ga-002
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.log

# =============================================================================
# DeepSeek R1 Distill Qwen 14B AWQ — Deep reasoning (R1 chain-of-thought)
# =============================================================================
# Hardware: A100 3g.40gb MIG slice (39.5 GiB usable VRAM)
# Weights:  ~8 GiB (14B params, AWQ 4-bit quantization)
# Context:  16K tokens (--max-model-len 16384)
#
# Community AWQ quantization by casperhansen. The official
# DeepSeek-R1-Distill-Qwen-14B is BF16 (~28 GiB) which would leave
# too little room for KV cache. AWQ 4-bit cuts weights to ~8 GiB.
#
# See vllm-qwen-14b.sh for detailed memory budget and flag explanations.
# =============================================================================

MODEL="casperhansen/DeepSeek-R1-Distill-Qwen-14B-AWQ"
PORT=8002

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
