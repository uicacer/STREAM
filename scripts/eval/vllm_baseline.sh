#!/bin/bash
#SBATCH --job-name=bench-baseline
#SBATCH --partition=batch_gpu2
#SBATCH --nodelist=ghi2-002
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=logs/bench-baseline-%j.log

# =============================================================================
# vLLM BASELINE Benchmark — Plain AWQ (No Optimizations)
# =============================================================================
# This script measures throughput with the UNOPTIMIZED configuration:
#   - Plain AWQ kernel (--quantization awq), NOT Marlin
#   - No prefix caching
#   - No chunked prefill
#   - Default GPU memory utilization (0.85)
#   - Default context length (32K)
#
# Measured throughput: ~20.1 tok/s (median over 5 runs)
#
# This serves as the "before" measurement for the paper's claim:
#   "achieving 28.5 vs. 20.1 tok/s with the pre-built container (1.4x improvement)"
#
# IMPORTANT: This uses the official vLLM container (vllm-0.15.1) with
# plain AWQ because the Marlin kernels in that container require CUDA 12.9+,
# which driver 550 does not support. This is the actual baseline — what you
# get if you use the pre-built container without building from source.
#
# Usage:
#   sbatch scripts/eval/vllm_baseline.sh
#   # Wait for vLLM to start (~60s), then run the benchmark:
#   bash scripts/eval/vllm_throughput_test.sh <node_ip> 8000 baseline
# =============================================================================

MODEL="Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
PORT=8000

echo "=========================================="
echo "BASELINE BENCHMARK (Plain AWQ, No Optimizations)"
echo "Model:   ${MODEL}"
echo "Job ID:  $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "GPU:     $CUDA_VISIBLE_DEVICES | Port: $PORT"
echo "Started: $(date)"
echo "=========================================="

module load apptainer

# Use the OFFICIAL vLLM container (CUDA 12.9, Marlin broken on driver 550)
# This forces plain AWQ fallback, which is the actual baseline scenario.
CONTAINER="/projects/acer_hpc_admin/nassar/containers/vllm-0.15.1"
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/projects/acer_hpc_admin/nassar/huggingface

NODE_IP=$(hostname -I | awk '{print $1}')
echo "Service: http://${NODE_IP}:${PORT}"
echo ""
echo "BASELINE CONFIG:"
echo "  Container:  vllm-0.15.1 (CUDA 12.9, Marlin incompatible with driver 550)"
echo "  Quantization: awq (plain, ~20 tok/s)"
echo "  Context:    32768 (default)"
echo "  GPU util:   0.85 (default safe)"
echo "  Prefix caching: OFF"
echo "  Chunked prefill: OFF"
echo "  torch.compile: OFF (--enforce-eager)"
echo ""
echo "Waiting for vLLM to start..."
echo "Once ready, run: bash scripts/eval/vllm_throughput_test.sh ${NODE_IP} ${PORT} baseline"
echo "=========================================="

apptainer exec --nv ${CONTAINER} \
    vllm serve ${MODEL} \
    --host 0.0.0.0 \
    --port ${PORT} \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 256 \
    --dtype float16 \
    --quantization awq \
    --enforce-eager \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill

echo "Baseline server stopped: $(date)"
