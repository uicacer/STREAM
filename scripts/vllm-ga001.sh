#!/bin/bash
#SBATCH --job-name=stream-vllm-ga-001
#SBATCH --partition=batch_gpu
#SBATCH --nodelist=ga-001              # Run on specific node!
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.log

# Configuration
#MODEL=${MODEL:-meta-llama/Llama-3.2-3B-Instruct}
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
#MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}  # ← NO AUTHENTICATION NEEDED!
# Alternative: mistralai/Mistral-7B-Instruct-v0.3
# If using Hagging Face Ollama:
# MODEL=${MODEL:-meta-llama/Llama-3.2-3B-Instruct}
# ← ADD YOUR HUGGING FACE TOKEN HERE
# export HF_TOKEN="hf_YOUR_TOKEN_HERE"


PORT=${PORT:-8000}

echo "=========================================="
echo "STREAM vLLM Service"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Started: $(date)"
echo "=========================================="

# Load Apptainer
module load apptainer

# Container
# Container stored in ACER project space (10 TiB quota) instead of home (100 GiB quota)
CONTAINER="/projects/acer_hpc_admin/nassar/containers/vllm-0.15.1"

# Set GPU (for MIG workaround if needed)
export CUDA_VISIBLE_DEVICES=0
# Redirect HuggingFace cache to project space (home dir too small for LLM weights)
export HF_HOME=/projects/acer_hpc_admin/nassar/huggingface
# Get node IP
NODE_IP=$(hostname -I | awk '{print $1}')
echo "Service will be available at: http://${NODE_IP}:${PORT}"
echo "Or via hostname: http://$(hostname):${PORT}"
echo "=========================================="

# Launch vLLM
# Increased max-model-len from 2048 to 8192 for longer conversations
# Using 85% GPU memory to leave headroom
apptainer exec --nv ${CONTAINER} \
    vllm serve ${MODEL} \
    --host 0.0.0.0 \
    --port ${PORT} \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85

echo "=========================================="
echo "Service stopped: $(date)"
echo "=========================================="


# Note:

# Why 4 CPUs for vLLM?
# Model Loading: Parallel data loading and preprocessing
# Tokenization: CPU-bound operation, benefits from parallelism
# Request Batching: Scheduling and queue management
# API Server: FastAPI/uvicorn workers for concurrent requests
# Result Processing: Post-processing and serialization

# Typical Configuration:
# Small Models (<7B): 2-4 CPUs sufficient
# Medium Models (7B-13B): 4-8 CPUs recommended
# Large Models (>13B): 8-16 CPUs for optimal throughput
