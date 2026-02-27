#!/bin/bash
#SBATCH --job-name=stream-qwen-vl-72b
#SBATCH --partition=batch_gpu2
#SBATCH --nodelist=ghi2-002
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.log

# =============================================================================
# Qwen 2.5 VL 72B Instruct AWQ — Vision-Language Flagship
# =============================================================================
# Hardware: H100 NVL full GPU (95.8 GiB VRAM) on ghi2-002
# Weights:  ~36 GiB (72B params, AWQ 4-bit quantization)
# Context:  64K tokens (--max-model-len 65536)
#
# This is a VISION-LANGUAGE model: it handles both text-only queries AND
# image+text queries. It replaces the text-only Qwen 2.5 72B as the single
# Lakeshore model — no need for a separate text model.
#
# vLLM multimodal flags:
#   --limit-mm-per-prompt '{"image": 4}'
#     Limits the number of images per request. Each image consumes extra
#     VRAM for the visual encoder + cross-attention KV cache. 4 images is
#     generous for a chat interface (most requests have 0-1 images).
#     Increase if users need to send more images per message.
#     NOTE: vLLM 0.15.1 requires JSON format, not the older key=value format.
#
# Memory budget (estimated for H100 NVL with --enforce-eager, 0.90 util):
#   Total GPU:             93.2 GiB (usable by CUDA)
#   Pre-allocated (0.90):  ~83.9 GiB
#   Model weights:         ~38 GiB (AWQ 4-bit, LLM + ViT encoder)
#   CUDA graphs:            0 GiB (disabled by --enforce-eager)
#   KV cache:             ~45 GiB (pre-allocated for concurrent sequences)
#   Reserved:              ~9 GiB (PyTorch overhead, sampler warmup, CUDA context)
#
#   At 64K max-model-len, one full-length sequence uses ~20.5 GiB KV cache.
#   The 45 GiB KV budget can hold ~2 concurrent 64K sequences, or many
#   shorter conversations. Most chat requests use <8K tokens.
#
#   NOTE: 0.90 was previously untested with --enforce-eager. If OOM occurs
#   at startup, revert to 0.85 (which had a proven 14 GiB reserve).
#
# --max-num-seqs 256: Same reasoning as text-only 72B — reduces sampler
#   warmup memory for the large 152K vocab.
#
# IF OOM OCCURS AT STARTUP:
#   1. Lower utilization: --gpu-memory-utilization 0.85 (proven safe)
#   2. Reduce context: --max-model-len 32768 (halves KV per sequence)
#   3. Reduce --limit-mm-per-prompt image=1
#
# NOTE: --enforce-eager is already enabled (see below for why).
# =============================================================================

MODEL="Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
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
# -----------------------------------------------------------------------------
# CONTAINER CHOICE:
#   vllm-cu124:  Custom-built with CUDA 12.4 → Marlin AWQ works (~30-40 tok/s)
#   vllm-0.15.1: Official Docker image with CUDA 12.9 → Marlin fails,
#                must use plain AWQ with --quantization awq --dtype float16
#                (~3 tok/s, functional but slow)
#
# To build the cu124 container: see scripts/vllm-cu124.def
# -----------------------------------------------------------------------------
CONTAINER="/projects/acer_hpc_admin/nassar/containers/vllm-cu124"
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
#   Qwen VL 72B AWQ is ~38 GiB — fits easily on one H100 96GB.
#
# --max-model-len 65536
#   Maximum number of tokens (input + output) per request.
#   This directly controls how much VRAM is allocated for the KV cache:
#
#     KV cache memory = num_layers x 2 x num_kv_heads x head_dim x max_len x dtype_size
#
#   For Qwen VL 72B (80 layers, 8 KV heads, 128 head dim, FP16 KV):
#     80 x 2 x 8 x 128 x 65536 x 2 bytes = ~20.5 GiB per sequence
#     vLLM pre-allocates for multiple concurrent sequences within the budget.
#
#   65536 (64K) enables long documents and extended conversations. The model
#   supports up to 128K natively, but 128K would leave room for only ~1
#   concurrent sequence. 64K is a good balance: handles most use cases while
#   supporting 2+ concurrent requests.
#   If OOM: reduce to 32768 (halves KV cache memory per sequence).
#
# --gpu-memory-utilization 0.90
#   Fraction of total GPU memory vLLM is allowed to use (0.0 to 1.0).
#   vLLM pre-allocates this much at startup for weights + KV cache.
#
#     Usable = 93.2 GiB x 0.90 = ~83.9 GiB
#     Model weights:  ~38 GiB (AWQ 4-bit, LLM + ViT encoder)
#     Remaining:      ~45 GiB --> KV cache (no CUDA graphs with --enforce-eager)
#
#   The reserved 10% (~9.3 GiB) is for PyTorch overhead, CUDA context,
#   sampler warmup tensors, and the sampling/decoding step.
#
#   NOTE: 0.90 previously caused OOM with CUDA graphs enabled (6.3 GiB overhead).
#   With --enforce-eager (no CUDA graphs), the freed 6 GiB compensates for the
#   smaller reserve. If OOM still occurs, revert to 0.85.
#   If OOM: lower to 0.85 (adds ~4.7 GiB more reserve headroom).
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
#   higher precision. This is why 72B params fit in ~38 GiB instead of ~144 GiB.
#
#   IMPORTANT: Use "awq_marlin" NOT "awq". The plain "awq" kernel is ~10x
#   slower (~3 tok/s vs ~30+ tok/s on H100). Marlin kernels are specifically
#   optimized for NVIDIA tensor cores and deliver dramatically better throughput.
#
#   CUDA COMPATIBILITY NOTE:
#   Marlin kernels contain PTX code (GPU intermediate instructions). The GPU
#   driver must support the PTX version used during compilation:
#     - vllm-cu124 container: CUDA 12.4 PTX → works with driver 550 (CUDA 12.4)
#     - vllm-0.15.1 container: CUDA 12.9 PTX → needs driver 570+ (CUDA 12.9)
#   If you see cudaErrorUnsupportedPtxVersion, you're using the wrong container.
#   Build the cu124 container with: scripts/build-vllm-cu124.sh
#
# --enforce-eager
#   Disables torch.compile and CUDA graph capture. In our vllm-cu124
#   container (dev build 0.8.5.post2), torch.compile crashes with:
#     RuntimeError: CUDA error: an illegal memory access was encountered
#   during Triton kernel autotuning. This is a known compatibility issue
#   between the Marlin AWQ kernels and torch.compile in this build.
#
#   Performance impact: The Marlin AWQ kernel (the 10x speedup over
#   plain AWQ) still works with --enforce-eager. torch.compile only
#   adds ~20-30% on top, so real-world throughput is ~6-8 tok/s
#   instead of ~30-40 tok/s (the theoretical Marlin + torch.compile max).
#   This is still ~2x faster than plain AWQ (~3 tok/s).
#
#   If a future driver or container update fixes torch.compile, remove
#   this flag to unlock the full ~30-40 tok/s throughput.
#
# --enable-prefix-caching
#   Caches KV computations for repeated prompt prefixes across requests.
#   When multiple users send queries with the same system prompt, vLLM
#   reuses the cached KV values instead of recomputing them. Zero VRAM
#   overhead — the cache uses the existing KV cache pool. Free speedup
#   for workloads with shared prefixes.
#
# --enable-chunked-prefill
#   Processes long prompts in smaller chunks, interleaving prefill
#   (compute-bound, processing the input) with decode (memory-bound,
#   generating tokens). This improves GPU utilization by ~10-15% for
#   mixed workloads where some requests are in prefill and others are
#   in decode phase simultaneously.
#
# NOTE: --limit-mm-per-prompt is intentionally NOT set. The real bottleneck
#   is the Globus payload limit (6 MB for images in the 8 MB total payload),
#   which naturally caps practical use at ~12-20 images depending on size.
#   The frontend compresses images (max 1024px, JPEG 85%) to ~100-500 KB each.
#   No need for an arbitrary count limit — the size limit is more flexible.
#
# =============================================================================

# =============================================================================
# FIRST RUN: Download the model weights (~36 GiB) before submitting this job.
# vLLM downloads from HuggingFace on first use, which can exceed the job's
# time limit. Pre-download in an interactive session:
#
#   srun --partition=batch_gpu2 --gres=gpu:1 --time=01:00:00 --pty bash
#   module load apptainer
#   export HF_HOME=/projects/acer_hpc_admin/nassar/huggingface
#   apptainer exec --nv /projects/acer_hpc_admin/nassar/containers/vllm-cu124 \
#       huggingface-cli download Qwen/Qwen2.5-VL-72B-Instruct-AWQ
#
# After downloading once, subsequent runs start immediately from the cache.
# =============================================================================

# =============================================================================
# CUDA driver and container compatibility
# =============================================================================
# Driver: 550.163.01 (supports CUDA ≤ 12.4), installed 2026-02-25.
#
# The official vLLM v0.15.1 Docker image ships with CUDA 12.9 inside the
# container. The Marlin AWQ kernels in that image use CUDA 12.9 PTX, which
# driver 550 cannot JIT-compile → cudaErrorUnsupportedPtxVersion.
#
# Solution: we built a custom container (vllm-cu124) from source using the
# nvidia/cuda:12.4.1-devel base image. All CUDA kernels — including Marlin —
# are compiled with CUDA 12.4 PTX, compatible with driver 550.
#
# See scripts/vllm-cu124.def for full details on why and how.
#
# Measured performance with vllm-cu124 + --enforce-eager + optimizations: ~25 tok/s
#   (prefix caching + chunked prefill + 0.90 GPU util + 64K context)
#   Baseline enforce-eager without optimizations: ~6-8 tok/s
# Fallback with vllm-0.15.1: ~3 tok/s (plain AWQ, no Marlin)
# =============================================================================

# =============================================================================
# Triton workaround experiments (uncomment to test torch.compile)
# =============================================================================
# The torch.compile crash occurs during Triton kernel autotuning. These env
# vars may bypass the crash, enabling full ~30-40 tok/s throughput.
# To test: uncomment ONE set of env vars below, remove --enforce-eager from
# the serve command, and submit the job. If vLLM starts and serves without
# crashing, the workaround works. If it crashes, re-add --enforce-eager.
#
# export TRITON_DISABLE_AUTOTUNE=1          # Skip autotuning, use defaults
# export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}  # Fresh cache per job
# export VLLM_USE_TRITON_FLASH_ATTN=0       # Disable Triton flash-attn
# =============================================================================

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

echo "Service stopped: $(date)"
