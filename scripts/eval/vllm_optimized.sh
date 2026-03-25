#!/bin/bash
#SBATCH --job-name=bench-optimized
#SBATCH --partition=batch_gpu2
#SBATCH --nodelist=ghi2-002
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=logs/bench-optimized-%j.log

# =============================================================================
# vLLM OPTIMIZED Benchmark — Marlin AWQ + All Optimizations
# =============================================================================
# This script measures throughput with the FULLY OPTIMIZED configuration:
#   - Marlin AWQ kernel (--quantization awq_marlin) — requires custom container
#   - Prefix caching enabled
#   - Chunked prefill enabled
#   - GPU memory utilization 0.90 (increased from 0.85)
#   - Context length 64K (increased from 32K)
#
# Expected throughput: ~25 tok/s
#
# This serves as the "after" measurement for the paper's claim:
#   "improved throughput from 3 to 25 tok/s"
#
# IMPORTANT: This uses the CUSTOM-BUILT container (vllm-cu124) compiled
# from source with CUDA 12.4 to match the driver. The Marlin kernels work
# because the PTX code matches the driver's CUDA version.
#
# Usage:
#   sbatch scripts/eval/vllm_optimized.sh
#   # Wait for vLLM to start (~60s), then run the benchmark:
#   bash scripts/eval/vllm_throughput_test.sh <node_ip> 8000 optimized
# =============================================================================

MODEL="Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
PORT=8000

echo "=========================================="
echo "OPTIMIZED BENCHMARK (Marlin AWQ + All Optimizations)"
echo "Model:   ${MODEL}"
echo "Job ID:  $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "GPU:     $CUDA_VISIBLE_DEVICES | Port: $PORT"
echo "Started: $(date)"
echo "=========================================="

module load apptainer

# Use the CUSTOM container built from source with CUDA 12.4
# Marlin AWQ kernels compiled with matching PTX version for driver 550.
CONTAINER="/projects/acer_hpc_admin/nassar/containers/vllm-cu124"
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/projects/acer_hpc_admin/nassar/huggingface

NODE_IP=$(hostname -I | awk '{print $1}')
echo "Service: http://${NODE_IP}:${PORT}"
echo ""
echo "OPTIMIZED CONFIG:"
echo "  Container:  vllm-cu124 (CUDA 12.4, custom build, Marlin compatible)"
echo "  Quantization: awq_marlin (~25 tok/s with Marlin kernels)"
echo "  Context:    65536 (64K)"
echo "  GPU util:   0.90 (increased for more KV cache)"
echo "  Prefix caching: ON"
echo "  Chunked prefill: ON"
echo "  torch.compile: OFF (--enforce-eager, driver limitation)"
echo ""
echo "Waiting for vLLM to start..."
echo "Once ready, run: bash scripts/eval/vllm_throughput_test.sh ${NODE_IP} ${PORT} optimized"
echo "=========================================="

apptainer exec --nv ${CONTAINER} \
    vllm serve ${MODEL} \
    --host 0.0.0.0 \
    --port ${PORT} \
    --tensor-parallel-size 1 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 256 \
    --dtype auto \
    --quantization awq_marlin \
    --enforce-eager \
    --enable-prefix-caching \
    --enable-chunked-prefill

echo "Optimized server stopped: $(date)"
