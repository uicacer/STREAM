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
# Context:  32K tokens (--max-model-len 32768)
#
# This is a VISION-LANGUAGE model: it handles both text-only queries AND
# image+text queries. It replaces the text-only Qwen 2.5 72B as the single
# Lakeshore model — no need for a separate text model.
#
# vLLM multimodal flags:
#   --limit-mm-per-prompt image=4
#     Limits the number of images per request. Each image consumes extra
#     VRAM for the visual encoder + cross-attention KV cache. 4 images is
#     generous for a chat interface (most requests have 0-1 images).
#     Increase if users need to send more images per message.
#
# Memory budget (estimated for H100 NVL):
#   Total GPU:             93.2 GiB (usable by CUDA)
#   Pre-allocated (0.85):  ~79.2 GiB
#   Model weights:         ~38 GiB (AWQ 4-bit, LLM + ViT encoder)
#   CUDA graphs:            ~6 GiB (captured at startup)
#   KV cache:             ~34 GiB (pre-allocated for concurrent sequences)
#   Reserved:             ~14 GiB (PyTorch overhead, sampler warmup, CUDA context)
#
# --max-num-seqs 256: Same reasoning as text-only 72B — reduces sampler
#   warmup memory for the large 152K vocab.
#
# IF THIS STILL FAILS:
#   1. Reduce context: --max-model-len 16384
#   2. Lower utilization: --gpu-memory-utilization 0.80
#   3. Add --enforce-eager (saves ~6 GiB from CUDA graph capture)
#   4. Reduce --limit-mm-per-prompt image=1
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
# See scripts/vllm-qwen-72b.sh for full parameter documentation.
# Key difference from text-only 72B: --limit-mm-per-prompt image=4
#
# --quantization awq_marlin
#   IMPORTANT: Requires CUDA driver 550+ (CUDA 12.4+). Without the driver
#   update, Marlin fails. Fall back to --quantization awq if needed (10x slower).
#
# =============================================================================

# =============================================================================
# FIRST RUN: Download the model weights (~36 GiB) before submitting this job.
# vLLM downloads from HuggingFace on first use, which can exceed the job's
# time limit. Pre-download in an interactive session:
#
#   srun --partition=batch_gpu2 --gres=gpu:1 --time=01:00:00 --pty bash
#   apptainer exec --nv /projects/acer_hpc_admin/nassar/containers/vllm-0.15.1 \
#       huggingface-cli download Qwen/Qwen2.5-VL-72B-Instruct-AWQ
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
    --quantization awq_marlin \
    --limit-mm-per-prompt image=4

echo "Service stopped: $(date)"
