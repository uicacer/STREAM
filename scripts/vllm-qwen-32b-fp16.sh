#!/bin/bash
#SBATCH --job-name=stream-qwen-32b-fp16
#SBATCH --partition=batch_gpu2
#SBATCH --nodelist=ghi2-002
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.log

# =============================================================================
# Qwen 2.5 32B Instruct FP16 — Full precision on H100
# =============================================================================
# Hardware: H100 NVL full GPU (95.8 GiB VRAM) on ghi2-002
# Weights:  ~64 GiB (32B params, FP16 — no quantization)
# Context:  32K tokens (--max-model-len 32768)
#
# WHY FP16 INSTEAD OF AWQ?
# -------------------------
# The H100's CUDA driver (535 / CUDA 12.2) doesn't support the Marlin
# quantization kernels (compiled for CUDA 12.4+). Without Marlin, the
# plain AWQ kernel only gets ~3 tok/s — 10x slower than expected.
#
# FP16 avoids quantization kernels entirely. It uses standard cuBLAS GEMM
# (matrix multiplication) which is universally supported. The H100 has
# enough VRAM (96 GiB) to fit 32B FP16 (~64 GiB) with room for KV cache.
#
# Performance: ~40-60 tok/s (FP16 on H100 tensor cores)
# Quality:     Higher than 72B AWQ — no quantization loss
#
# Memory budget:
#   Total GPU:             93.2 GiB (usable by CUDA)
#   Pre-allocated (0.90):  ~83.9 GiB
#   Model weights:         ~64 GiB (32B × 2 bytes FP16)
#   KV cache:              ~14 GiB (for concurrent sequences)
#   Reserved:              ~9.3 GiB (PyTorch, CUDA context, overhead)
#
#   With 32K context, KV cache supports ~1.5-2 concurrent requests.
#   For higher concurrency, reduce --max-model-len to 16384.
#
# IF THIS FAILS (OOM):
#   1. Reduce context: --max-model-len 16384
#   2. Lower utilization: --gpu-memory-utilization 0.85
#   3. Add --enforce-eager (saves ~5 GiB from CUDA graph capture)
# =============================================================================

MODEL="Qwen/Qwen2.5-32B-Instruct"
PORT=8000

echo "=========================================="
echo "STREAM vLLM: ${MODEL}"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES | Port: $PORT"
echo "Started: $(date)"
echo "=========================================="

module load apptainer

# =============================================================================
# Apptainer container
# =============================================================================
# vLLM runs inside an Apptainer (formerly Singularity) container, which
# packages the vLLM Python environment, PyTorch, and CUDA libraries into a
# portable, reproducible runtime. This avoids version conflicts with the
# host system's Python and ensures identical behavior across compute nodes.
#
# We use a "sandbox" format (unpacked directory) rather than a compressed
# SIF file. Both work identically with `apptainer exec`, but sandboxes are
# faster to build (no SquashFS compression step) at the cost of more disk
# usage. The container is stored in the ACER project space (/projects/)
# rather than the home directory to conserve the 100 GiB home quota.
# =============================================================================
CONTAINER="/projects/acer_hpc_admin/nassar/containers/vllm-0.15.1"
export CUDA_VISIBLE_DEVICES=0

# =============================================================================
# HuggingFace cache location
# =============================================================================
# By default, HuggingFace stores downloaded model weights in ~/.cache/huggingface/
# inside the user's home directory. On Lakeshore, the home directory quota is
# only 100 GiB — too small for large LLM weights (this model alone is ~64 GiB).
#
# We redirect the cache to the ACER project space (/projects/acer_hpc_admin),
# which has a 10 TiB quota shared across the team. This is persistent storage
# (unlike /scratch which is purged periodically), and is accessible from all
# compute nodes via the GPFS parallel filesystem.
#
# HF_HOME controls where HuggingFace Hub stores everything: model weights,
# tokenizer files, config files, and download metadata. vLLM reads this
# environment variable to locate cached models.
# =============================================================================
export HF_HOME=/projects/acer_hpc_admin/nassar/huggingface

NODE_IP=$(hostname -I | awk '{print $1}')
echo "Service: http://${NODE_IP}:${PORT}"

# =============================================================================
# vLLM serve — Parameter Reference
# =============================================================================
#
# --tensor-parallel-size 1
#   Single GPU. The 32B FP16 model fits on one H100 (64 GiB < 96 GiB).
#
# --max-model-len 32768
#   Maximum tokens (input + output) per request. 32K is the model's native
#   context length. KV cache memory scales with this:
#     For Qwen 32B (64 layers, 8 KV heads, 128 head dim, FP16 KV):
#     64 x 2 x 8 x 128 x 32768 x 2 bytes = ~8.6 GiB per sequence
#   If OOM: reduce to 16384 (halves KV cache).
#
# --gpu-memory-utilization 0.90
#   Use 90% of GPU memory. Model is ~64 GiB, leaving ~20 GiB for KV cache
#   and overhead. Higher than the 72B AWQ (0.85) because FP16 has simpler
#   memory management — no quantization buffers or scale tensors.
#   If OOM: lower to 0.85.
#
# --dtype auto
#   vLLM reads the model config and uses FP16. No quantization involved.
#   Standard cuBLAS FP16 GEMM — fast and universally supported.
#
# No --quantization flag needed (model is not quantized).
#
# =============================================================================

# =============================================================================
# FIRST RUN: Download the model weights (~64 GiB) before submitting this job.
# vLLM downloads from HuggingFace on first use, which can exceed the job's
# time limit. Pre-download in an interactive session:
#
#   srun --partition=batch --account=ts_acer_chi --time=02:00:00 --pty bash
#   module load apptainer
#   apptainer exec /home/nassar/STREAM/containers/vllm-0.15.1 \
#       huggingface-cli download Qwen/Qwen2.5-32B-Instruct
#
# After downloading once, subsequent runs start immediately from the cache.
# =============================================================================

# =============================================================================
# NOTE: Requires CUDA driver 550+ (CUDA 12.4+)
# =============================================================================
# vLLM 0.15.1's V1 engine uses torch.compile + Triton kernels that need
# CUDA 12.4+ PTX support. The current H100 driver (535 / CUDA 12.2) causes
# these kernels to fall back to slow paths (~1.7 tok/s instead of ~40-60).
#
# This script is ready to go once Steve updates the CUDA driver to 550+.
# Expected performance after driver update: ~40-60 tok/s (BF16 on H100).
# =============================================================================

apptainer exec --nv ${CONTAINER} \
    vllm serve ${MODEL} \
    --host 0.0.0.0 \
    --port ${PORT} \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 256 \
    --dtype auto

echo "Service stopped: $(date)"
