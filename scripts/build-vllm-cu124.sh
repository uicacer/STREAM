#!/bin/bash
# =============================================================================
# Build vLLM Container with CUDA 12.4 for Marlin AWQ Compatibility
# =============================================================================
#
# WHAT THIS SCRIPT DOES:
#   Builds a new Apptainer container that has vLLM v0.15.1 compiled with
#   CUDA 12.4. This fixes the Marlin AWQ kernel incompatibility with our
#   GPU driver (550.163.01, CUDA 12.4), enabling fast inference (~30-40
#   tok/s instead of ~3 tok/s with the plain AWQ fallback).
#
#   See scripts/vllm-cu124.def for a detailed explanation of why this is
#   needed and how the container works.
#
# HOW TO RUN THIS SCRIPT:
#   There are two ways to run the build:
#
#   OPTION 1: Submit as a SLURM batch job (RECOMMENDED)
#   -------------------------------------------------
#   The build runs as a background job. It won't die if your SSH disconnects.
#   You can even close your laptop and come back later.
#
#     sbatch scripts/build-vllm-cu124.sh
#
#   Monitor progress:
#     tail -f logs/build-vllm-cu124-<jobid>.log
#
#   Check if it's done:
#     squeue -u $USER
#
#   OPTION 2: Run interactively in a screen session
#   ------------------------------------------------
#   If you prefer watching the output in real-time, you MUST use screen
#   or tmux first. The build takes ~1.5-2 hours — without screen, it will
#   be killed if your SSH connection drops.
#
#     ssh lakeshore
#     screen -S vllm-build              # Start a named screen session
#     bash scripts/build-vllm-cu124.sh  # Run the build
#     # Press Ctrl+A then D to detach (safe to disconnect SSH now)
#     # Later: screen -r vllm-build     # Reattach to see progress
#
# PREREQUISITES:
#   - Apptainer must be available (module load apptainer)
#   - --fakeroot must be enabled for your user account (ask ACER admins
#     if "apptainer build --fakeroot" gives a permission error)
#   - The definition file scripts/vllm-cu124.def must exist
#   - Enough disk space in /projects/ (~30 GiB for the sandbox)
#
# AFTER BUILDING:
#   1. Test the container (see verification commands at the end of output)
#   2. Update scripts/vllm-qwen-vl-72b.sh:
#      - Change CONTAINER to point to the new container
#      - Change --quantization awq to --quantization awq_marlin
#      - Remove --dtype float16
#   3. Submit the model: sbatch scripts/vllm-qwen-vl-72b.sh
#   4. STREAM users get ~30-40 tok/s automatically!
#
# =============================================================================

# =============================================================================
# SLURM settings (only used when submitted with sbatch)
# =============================================================================
# These lines starting with #SBATCH are directives for the SLURM job
# scheduler. They're ignored when running the script directly with bash.
#
# We use the "batch" partition (CPU-only, no GPU needed) because building
# a container is a compilation task — nvcc (the CUDA compiler) runs on
# the CPU, not the GPU. The GPU is only needed at runtime.
#
# MEMORY: 64 GiB is required because the FlashAttention 3 (FA3) Hopper
# kernels are exceptionally memory-hungry during compilation — each nvcc
# process uses 8-16 GiB of RAM (vs 2-4 GiB for typical kernels). With
# MAX_JOBS=4 in the .def file, four FA3 kernels compiling simultaneously
# can peak at 32-64 GiB. At 32 GiB, the Linux OOM killer silently kills
# nvcc processes (just prints "Killed" with no error). 64 GiB provides
# sufficient headroom.
# =============================================================================
#SBATCH --job-name=build-vllm-cu124
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/build-vllm-cu124-%j.log

# Load Apptainer (the container runtime)
module load apptainer

# =============================================================================
# Configuration
# =============================================================================
CONTAINER_DIR="/projects/acer_hpc_admin/nassar/containers"
CONTAINER_NAME="vllm-cu124"
# NOTE: We use an absolute path here because SLURM copies the script to a
# temporary spool directory before executing it. $(dirname "$0") would point
# to the spool directory, not the original scripts directory.
DEF_FILE="/home/nassar/STREAM/scripts/vllm-cu124.def"

# Check that the definition file exists
if [ ! -f "${DEF_FILE}" ]; then
    echo "ERROR: Definition file not found: ${DEF_FILE}"
    echo "       Make sure scripts/vllm-cu124.def exists."
    exit 1
fi

# Check that the output directory exists
if [ ! -d "${CONTAINER_DIR}" ]; then
    echo "ERROR: Container directory not found: ${CONTAINER_DIR}"
    echo "       Check that /projects/acer_hpc_admin is mounted."
    exit 1
fi

# Warn if the container already exists
if [ -d "${CONTAINER_DIR}/${CONTAINER_NAME}" ]; then
    echo "WARNING: Container already exists at ${CONTAINER_DIR}/${CONTAINER_NAME}"
    echo "         The build will overwrite it."
    echo "         Press Ctrl+C within 10 seconds to cancel..."
    sleep 10
fi

# =============================================================================
# Build the container
# =============================================================================
echo "============================================"
echo "Building vLLM container with CUDA 12.4"
echo "============================================"
echo "Definition file: ${DEF_FILE}"
echo "Output:          ${CONTAINER_DIR}/${CONTAINER_NAME}"
echo "Started:         $(date)"
echo "============================================"
echo ""
echo "This will take ~1.5-2 hours. Major steps:"
echo "  1. Pull nvidia/cuda:12.4.1-devel-ubuntu22.04 base image (~5 min)"
echo "  2. Install Python 3.12 and system packages (~5 min)"
echo "  3. Install PyTorch for CUDA 12.4 (~5 min)"
echo "  4. Clone and compile vLLM from source (~60-90 min)"
echo "  5. Cleanup (~1 min)"
echo ""

# --fakeroot: emulate root privileges for apt-get install (see .def file)
# --sandbox:  create unpacked directory (faster than compressed .sif)
apptainer build --fakeroot --sandbox \
    "${CONTAINER_DIR}/${CONTAINER_NAME}" \
    "${DEF_FILE}"

BUILD_EXIT_CODE=$?

echo ""
echo "============================================"
if [ $BUILD_EXIT_CODE -eq 0 ]; then
    echo "BUILD SUCCESSFUL"
    echo "============================================"
    echo "Finished: $(date)"
    echo "Container: ${CONTAINER_DIR}/${CONTAINER_NAME}"
    echo ""
    echo "NEXT STEPS — Verify the container:"
    echo ""
    echo "  # 1. Check vLLM version:"
    echo "  apptainer exec --nv ${CONTAINER_DIR}/${CONTAINER_NAME} \\"
    echo "      python3 -c \"import vllm; print('vLLM:', vllm.__version__)\""
    echo ""
    echo "  # 2. Check CUDA version (should say 12.4):"
    echo "  apptainer exec --nv ${CONTAINER_DIR}/${CONTAINER_NAME} \\"
    echo "      python3 -c \"import torch; print('CUDA:', torch.version.cuda)\""
    echo ""
    echo "  # 3. Test Marlin AWQ with a quick model load:"
    echo "  # (Run this on a compute node with GPU access)"
    echo "  srun --partition=batch_gpu2 --gres=gpu:1 --time=00:30:00 --pty bash"
    echo "  module load apptainer"
    echo "  export HF_HOME=/projects/acer_hpc_admin/nassar/huggingface"
    echo "  export CUDA_VISIBLE_DEVICES=0"
    echo "  apptainer exec --nv ${CONTAINER_DIR}/${CONTAINER_NAME} \\"
    echo "      vllm serve Qwen/Qwen2.5-VL-72B-Instruct-AWQ \\"
    echo "      --host 0.0.0.0 --port 8000 \\"
    echo "      --quantization awq_marlin --dtype auto \\"
    echo "      --max-model-len 4096 --max-num-seqs 256"
    echo ""
    echo "  # 4. If Marlin test passes, update the production script:"
    echo "  #    Edit scripts/vllm-qwen-vl-72b.sh:"
    echo "  #      CONTAINER=\"${CONTAINER_DIR}/${CONTAINER_NAME}\""
    echo "  #      --quantization awq_marlin"
    echo "  #      (remove --dtype float16)"
else
    echo "BUILD FAILED (exit code: ${BUILD_EXIT_CODE})"
    echo "============================================"
    echo "Finished: $(date)"
    echo ""
    echo "COMMON ISSUES:"
    echo ""
    echo "  'FATAL: could not use fakeroot':"
    echo "    --fakeroot is not enabled for your account."
    echo "    Ask ACER admins: 'Can you enable Apptainer fakeroot for my user?'"
    echo ""
    echo "  'FATAL: ... permission denied':"
    echo "    Check write permissions on ${CONTAINER_DIR}"
    echo ""
    echo "  Build fails during 'pip install --no-build-isolation .':"
    echo "    - Memory issue: try reducing MAX_JOBS in vllm-cu124.def"
    echo "    - Network issue: check that the build node can reach github.com"
    echo "      and pypi.org (some compute nodes have restricted internet)"
    echo ""
    echo "  ALTERNATIVE: Build without --fakeroot"
    echo "    If --fakeroot is not available, you can build the Docker image"
    echo "    on another machine and convert:"
    echo "      # On a machine with Docker:"
    echo "      git clone --branch v0.15.1 https://github.com/vllm-project/vllm.git"
    echo "      cd vllm"
    echo "      docker build --build-arg CUDA_VERSION=12.4.1 -t vllm-cu124 ."
    echo "      docker save vllm-cu124 -o vllm-cu124.tar"
    echo "      # Copy to Lakeshore, then:"
    echo "      apptainer build --sandbox vllm-cu124 docker-archive://vllm-cu124.tar"
fi
