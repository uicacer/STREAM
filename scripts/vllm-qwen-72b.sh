#!/bin/bash
#SBATCH --job-name=stream-qwen-72b
#SBATCH --partition=batch_gpu2
#SBATCH --nodelist=ghi2-002
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.log

# =============================================================================
# Qwen 2.5 72B Instruct AWQ — Flagship open-source model
# =============================================================================
# Hardware: H100 NVL full GPU (95.8 GiB VRAM) on ghi2-002
# Weights:  ~36 GiB (72B params, AWQ 4-bit quantization)
# Context:  32K tokens (--max-model-len 32768)
#
# Memory budget (actual measurements from H100 NVL):
#   Total GPU:             93.2 GiB (usable by CUDA)
#   Pre-allocated (0.85):  ~79.2 GiB
#   Model weights:         38.8 GiB (actual — slightly more than theoretical ~36)
#   CUDA graphs:            6.3 GiB (captured at startup for fast inference)
#   KV cache:             ~34 GiB (pre-allocated for concurrent sequences)
#   Reserved:             ~14 GiB (PyTorch overhead, sampler warmup, CUDA context)
#
# --max-num-seqs 256: Maximum concurrent sequences per batch.
#   Default is 1024, but the sampler warmup allocates vocab_size x max_num_seqs
#   tensors during startup. For Qwen 72B (152K vocab), 1024 sequences needs
#   ~1.74 GiB for the sort operation — too much at 0.90 utilization.
#   256 is still generous for a campus service (256 simultaneous requests
#   on one GPU). The real throughput bottleneck is inference speed, not concurrency.
#
# IF THIS STILL FAILS:
#   1. Reduce context: --max-model-len 16384
#   2. Lower utilization: --gpu-memory-utilization 0.80
#   3. Add --enforce-eager (saves ~6 GiB from CUDA graph capture)
# =============================================================================

MODEL="Qwen/Qwen2.5-72B-Instruct-AWQ"
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
# only 100 GiB — too small for large LLM weights (the 72B AWQ alone is ~39 GiB).
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
#   How many GPUs to split the model across. 1 = single GPU (no tensor
#   parallelism). Only increase if the model doesn't fit on one GPU.
#   Qwen 72B AWQ is ~36 GiB — fits easily on one H100 96GB.
#
# --max-model-len 32768
#   Maximum number of tokens (input + output) per request.
#   This directly controls how much VRAM is allocated for the KV cache:
#
#     KV cache memory = num_layers x 2 x num_kv_heads x head_dim x max_len x dtype_size
#
#   For Qwen 72B (80 layers, 8 KV heads, 128 head dim, FP16 KV):
#     80 x 2 x 8 x 128 x 32768 x 2 bytes = ~10.7 GiB per sequence
#     vLLM pre-allocates for multiple concurrent sequences within the budget.
#
#   32768 is a good balance: handles long conversations and documents
#   while leaving headroom. The model supports up to 128K natively,
#   but that would need ~4x more KV cache memory.
#   If OOM: reduce to 16384 (halves KV cache memory).
#
# --gpu-memory-utilization 0.85
#   Fraction of total GPU memory vLLM is allowed to use (0.0 to 1.0).
#   vLLM pre-allocates this much at startup for weights + KV cache.
#
#     Usable = 93.2 GiB x 0.85 = ~79.2 GiB
#     Model weights:  ~38.8 GiB (AWQ 4-bit, actual measured)
#     Remaining:      ~40 GiB --> KV cache + CUDA graphs
#
#   The reserved 15% (~14 GiB) is for PyTorch overhead, CUDA context,
#   sampler warmup tensors, and the sampling/decoding step.
#   0.90 caused OOM during sampler warmup — 0.85 leaves enough headroom.
#   If OOM: lower to 0.80 (frees ~5 GiB more for overhead).
#
# --max-num-seqs 256
#   Maximum number of sequences vLLM will batch together.
#   Also controls the sampler warmup size at startup (default 1024 caused OOM).
#   256 concurrent sequences is more than enough for a campus AI service —
#   on a single GPU the inference speed is the real throughput bottleneck.
#
# --dtype auto
#   Lets vLLM pick the data type from the model config.
#   For AWQ models, weights are stored as INT4 but computation runs
#   in FP16 — vLLM handles this automatically.
#
# --quantization awq_marlin
#   Tells vLLM to use AWQ with Marlin kernels (optimized for NVIDIA GPUs).
#   AWQ compresses weights from FP16 (2 bytes) to INT4 (0.5 bytes) — 4x
#   smaller — while preserving quality by keeping "salient" weights at
#   higher precision. This is why 72B params fit in ~36 GiB instead of ~144 GiB.
#
#   IMPORTANT: Use "awq_marlin" NOT "awq". The plain "awq" kernel is ~10x
#   slower (~3 tok/s vs ~30+ tok/s on H100). Marlin kernels are specifically
#   optimized for NVIDIA tensor cores and deliver dramatically better throughput.
#
# =============================================================================

# =============================================================================
# FIRST RUN: Download the model weights (~36 GiB) before submitting this job.
# vLLM downloads from HuggingFace on first use, which can exceed the job's
# time limit. Pre-download in an interactive session:
#
#   srun --partition=batch_gpu2 --gres=gpu:1 --time=01:00:00 --pty bash
#   apptainer exec --nv /home/nassar/STREAM/containers/vllm-openai_v0.13.0.sif \
#       huggingface-cli download Qwen/Qwen2.5-72B-Instruct-AWQ
#
# After downloading once, subsequent runs start immediately from the cache.
# =============================================================================

# =============================================================================
# NOTE: Requires CUDA driver 550+ (CUDA 12.4+)
# =============================================================================
# vLLM 0.15.1's V1 engine uses torch.compile + Triton kernels that need
# CUDA 12.4+ PTX support. The current H100 driver (535 / CUDA 12.2) causes
# these kernels to fall back to slow paths.
#
# Additionally, --quantization awq_marlin uses Marlin kernels which also
# require CUDA 12.4+ PTX. Without the driver update, Marlin fails entirely.
#
# This script is ready to go once Steve updates the CUDA driver to 550+.
# Expected performance after driver update: ~30-40 tok/s (AWQ Marlin on H100).
# =============================================================================

apptainer exec --nv ${CONTAINER} \
    vllm serve ${MODEL} \
    --host 0.0.0.0 \
    --port ${PORT} \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 256 \
    --dtype auto \
    --quantization awq_marlin

echo "Service stopped: $(date)"
