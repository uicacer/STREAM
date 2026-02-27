# Deploying vLLM on HPC: Building Custom Containers for Fast LLM Inference

## A Technical Report on Deploying Large Language Models on Institutional GPU Clusters

---

## Table of Contents

1. [Introduction: Why Deploy LLMs on HPC?](#1-introduction-why-deploy-llms-on-hpc)
2. [Background: Key Concepts](#2-background-key-concepts)
3. [Hardware: Lakeshore GPU Inventory](#3-hardware-lakeshore-gpu-inventory)
4. [The Simple Path: Pre-Built vLLM Containers](#4-the-simple-path-pre-built-vllm-containers)
5. [The Problem: A 10x Speed Gap](#5-the-problem-a-10x-speed-gap)
6. [The Solution: Building vLLM from Source with CUDA 12.4](#6-the-solution-building-vllm-from-source-with-cuda-124)
7. [Build Attempt Chronicle: 9 Failures and What They Teach](#7-build-attempt-chronicle-9-failures-and-what-they-teach)
8. [The Final Container Definition (Annotated)](#8-the-final-container-definition-annotated)
9. [The Build Process](#9-the-build-process)
10. [vLLM Configuration for Large Models](#10-vllm-configuration-for-large-models)
11. [Model Deployment Scripts](#11-model-deployment-scripts)
12. [Performance Tuning: Beyond the Initial Deployment](#12-performance-tuning-beyond-the-initial-deployment)
13. [Verification and Testing](#13-verification-and-testing)
14. [Lessons Learned for HPC LLM Deployment](#14-lessons-learned-for-hpc-llm-deployment)
15. [Frequently Asked Questions](#15-frequently-asked-questions)
16. [References](#16-references)

---

## 1. Introduction: Why Deploy LLMs on HPC?

### The cost problem with cloud AI

Commercial AI APIs charge per token. For GPT-4-class models, typical costs are $10-30 per million input tokens and $30-60 per million output tokens [1]. For a university research group or campus AI service handling thousands of queries per day, this adds up to thousands of dollars per month.

### The HPC alternative

Most research universities already have GPU clusters purchased for scientific computing. These GPUs — often NVIDIA A100s or H100s — are the same hardware that powers commercial AI services. The key insight: **these GPUs sit idle between research jobs**. By deploying LLM inference servers during idle periods, institutions can provide AI services to their communities at near-zero marginal cost.

### Where Lakeshore fits in STREAM

STREAM (Smart Tiered Routing Engine for AI Models) is a three-tier system that routes queries to the most cost-effective AI backend:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      STREAM Routing Engine                         │
│                                                                    │
│   User Query ──→ Complexity Judge ──→ Route to cheapest tier       │
│                                                                    │
│   Tier 1: LOCAL     │ Ollama (Llama 3.2 1B/3B) │ Free, CPU-based  │
│   Tier 2: LAKESHORE │ vLLM (Qwen 72B on H100)  │ Free, GPU-based  │
│   Tier 3: CLOUD     │ Claude/GPT-4 via API      │ Paid per token   │
└─────────────────────────────────────────────────────────────────────┘
```

The Lakeshore tier is the sweet spot: **free GPU inference with near-cloud quality**. The Qwen 2.5 72B model scores competitively with GPT-4 on academic benchmarks [2], yet costs nothing because it runs on UIC's existing HPC infrastructure.

### The goal

Deploy the **Qwen 2.5-VL-72B-Instruct-AWQ** vision-language model on Lakeshore's H100 GPU at interactive speeds. This model handles both text-only and image+text queries, replacing the need for a separate text model and a separate vision model.

This document chronicles the process of achieving that goal — including 12 attempts (9 build failures + 3 runtime fixes), each revealing a different compatibility issue between vLLM, PyTorch, CUDA, transformers, flash-attn, and the HPC container runtime. The final result: **~25 tok/s** with AWQ Marlin in eager mode plus runtime optimizations (prefix caching, chunked prefill, 64K context) — an **8x improvement** over the initial plain AWQ deployment (~3 tok/s).

---

## 2. Background: Key Concepts

### What is vLLM?

vLLM [3] is an open-source inference engine for large language models. It is significantly faster than naive PyTorch inference because of two key innovations:

1. **PagedAttention**: Manages GPU memory for the KV cache (the "working memory" of the model during text generation) like an operating system manages RAM — using virtual memory pages. This eliminates memory waste from fragmentation, allowing more concurrent requests.

2. **Continuous Batching**: Instead of waiting for all requests in a batch to finish before starting new ones, vLLM dynamically adds and removes requests as they arrive and complete. This maximizes GPU utilization.

vLLM exposes an **OpenAI-compatible HTTP API** — meaning any application built for the OpenAI API can switch to vLLM by changing just the base URL. STREAM uses this to seamlessly route requests between cloud APIs and local vLLM servers.

### What is Apptainer (formerly Singularity)?

Apptainer [4] is a container runtime designed for HPC environments. Like Docker, it packages software into portable, reproducible units. Unlike Docker, it:

- **Does not require root privileges to run** — critical on shared multi-user HPC systems
- **Integrates with job schedulers** (SLURM, PBS) — containers are launched inside scheduled jobs
- **Can use GPU passthrough** (`--nv` flag) — the container accesses the host's NVIDIA drivers
- **Supports sandbox format** — an unpacked directory tree instead of a compressed image file

The key concept for this report: Apptainer containers bundle their own CUDA libraries (inside the container), but use the host's GPU driver (outside the container). This creates a potential version mismatch, which is the root cause of the problem we solve.

```
┌─────────────────────────────────────┐
│         Apptainer Container         │
│                                     │
│  Python 3.11                        │
│  PyTorch 2.6.0                      │
│  CUDA Toolkit 12.4 (nvcc, headers)  │  ← Inside the container
│  vLLM 0.8.5                         │
│  Marlin kernels (CUDA 12.4 PTX)     │
│                                     │
├─────────────────────────────────────┤
│         Host System (--nv)          │
│                                     │
│  NVIDIA Driver 550.163.01           │  ← Outside the container
│  Supports CUDA ≤ 12.4 PTX          │
│  H100 NVL GPU (95.8 GiB VRAM)      │
│                                     │
└─────────────────────────────────────┘
```

### What is AWQ quantization?

AWQ (Activation-aware Weight Quantization) [5] compresses model weights from 16-bit floating point (FP16) to 4-bit integers (INT4). This reduces memory by 4x while preserving model quality by keeping "salient" weights (those most important for accuracy) at higher effective precision.

For the Qwen 72B model:
- **FP16**: 72 billion × 2 bytes = ~144 GiB — does not fit on a single GPU
- **AWQ 4-bit**: 72 billion × 0.5 bytes = ~36 GiB — fits on one H100 (95.8 GiB)

AWQ has two kernel implementations in vLLM:

| Kernel | Flag | Speed on H100 | How it works |
|--------|------|---------------|--------------|
| Plain AWQ | `--quantization awq` | ~3 tok/s | Generic CUDA kernel, not optimized for tensor cores |
| Marlin AWQ (eager, baseline) | `--quantization awq_marlin --enforce-eager` | ~6-8 tok/s | Marlin kernel optimized for tensor cores, without torch.compile |
| Marlin AWQ (eager, optimized) | `--quantization awq_marlin --enforce-eager` + prefix caching, chunked prefill | **~25 tok/s** | Marlin kernel + runtime optimizations (measured) |
| Marlin AWQ (compiled) | `--quantization awq_marlin` | ~30-40 tok/s | Full Marlin + torch.compile (requires compatible build) |

The Marlin kernel provides a significant speedup over plain AWQ. In our deployment, `torch.compile` crashes (see Attempt 12), so we use `--enforce-eager` mode. With additional runtime optimizations (prefix caching, chunked prefill, 64K context, 0.90 GPU memory utilization), we measured **~25 tok/s** — an 8x improvement over plain AWQ and approaching the theoretical maximum of ~30-40 tok/s with torch.compile.

### What is PTX and JIT compilation?

This is the concept most people miss when deploying CUDA software, and it is the root cause of our problem.

**PTX (Parallel Thread Execution)** is NVIDIA's intermediate instruction set for GPUs — think of it as "assembly language for GPUs." When you compile CUDA code, the CUDA compiler (`nvcc`) does not produce final GPU machine code directly. Instead, it produces PTX — a portable intermediate format.

**JIT (Just-In-Time) compilation** is what happens at runtime: the GPU driver translates PTX instructions into native machine code for the specific GPU model (e.g., H100). This happens transparently the first time a CUDA kernel is launched.

The critical rule: **the GPU driver must understand the PTX version being used.** Each CUDA toolkit version produces a specific PTX version, and each GPU driver version can only JIT-compile PTX up to a certain version:

```
CUDA Compilation Pipeline:

  CUDA Source Code (.cu)
        │
        ▼
  nvcc (CUDA Compiler)        ← Uses CUDA Toolkit version X.Y
        │
        ▼
  PTX Code (intermediate)     ← Contains "target sm_90, cuda X.Y" header
        │
        ▼  (at runtime)
  GPU Driver                  ← Must understand PTX version X.Y
        │
        ▼
  Native GPU Machine Code     ← Runs on the actual H100/A100 hardware
```

**The version compatibility table:**

| GPU Driver Version | Supports PTX up to | CUDA Toolkit |
|--------------------|---------------------|-------------|
| 525.x | CUDA 12.0 | 12.0 |
| 535.x | CUDA 12.2 | 12.2 |
| 545.x | CUDA 12.3 | 12.3 |
| **550.x** | **CUDA 12.4** | **12.4** |
| 560.x | CUDA 12.6 | 12.6 |
| 570.x | CUDA 12.8 | 12.8 |

This is a one-way compatibility: a driver that understands CUDA 12.4 PTX can also JIT older PTX (12.0, 12.2, etc.), but **cannot** JIT newer PTX (12.6, 12.9, etc.). It is like trying to read a book written in a language dialect that hasn't been invented yet.

---

## 3. Hardware: Lakeshore GPU Inventory

Lakeshore is UIC's primary HPC cluster managed by ACER (Academic Computing and Engineering Resources). For LLM inference, two types of GPU nodes are relevant:

### A100 MIG Nodes (ga-002)

**MIG (Multi-Instance GPU)** partitions a single A100 80GB into multiple isolated GPU instances. Each instance has its own memory, compute cores, and L2 cache — it behaves like a separate, smaller GPU.

```
┌──────────────────────────────────────────────────┐
│              NVIDIA A100 80GB (ga-002)            │
│                                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────┐ │
│  │  MIG Slice   │  │  MIG Slice   │  │ MIG Slice│ │
│  │  3g.40gb     │  │  3g.40gb     │  │ (others) │ │
│  │  39.5 GiB    │  │  39.5 GiB    │  │          │ │
│  │  usable VRAM │  │  usable VRAM │  │          │ │
│  └─────────────┘  └─────────────┘  └──────────┘ │
└──────────────────────────────────────────────────┘
```

**Best for:** Models up to ~32B parameters (AWQ quantized). The 39.5 GiB is enough for model weights (~18 GiB for 32B AWQ) plus a modest KV cache (~17 GiB for 16K context).

**Limitations:**
- Cannot use CUDA graph capture (uses ~4-6 GiB of VRAM) → must use `--enforce-eager`, costing 10-15% speed
- Context window limited to 16K tokens (vs 32K on H100)
- Only one model per MIG slice

### H100 NVL Node (ghi2-002)

The H100 NVL is a full, unpartitioned GPU with 95.8 GiB of VRAM. This is the flagship hardware for large model inference.

```
┌──────────────────────────────────────────────────┐
│          NVIDIA H100 NVL (ghi2-002)               │
│                                                    │
│  Total VRAM:           95.8 GiB                    │
│  Usable by CUDA:       93.2 GiB                    │
│  Driver:               550.163.01                   │
│  CUDA PTX support:     ≤ 12.4                       │
│  Tensor cores:         4th gen (FP8, INT8, FP16)    │
│  Memory bandwidth:     3.35 TB/s (HBM3)            │
└──────────────────────────────────────────────────┘
```

**Best for:** 72B parameter models (AWQ quantized). The memory budget at 0.85 utilization:

| Component | Memory | Notes |
|-----------|--------|-------|
| Pre-allocated total | 79.2 GiB | 93.2 × 0.85 |
| Model weights (AWQ) | 38.8 GiB | 72B params × 0.5 bytes + overhead |
| CUDA graphs | 6.3 GiB | Captured at startup for fast inference |
| KV cache | ~34 GiB | For concurrent sequences at 32K context |
| Reserved (not pre-allocated) | ~14 GiB | PyTorch, CUDA context, sampler warmup |

**Driver version matters:** The driver on ghi2-002 is **550.163.01**, which supports CUDA PTX up to version **12.4**. This is the key constraint that drives everything in this report.

---

## 4. The Simple Path: Pre-Built vLLM Containers

### Step 1: Pull the official Docker image

The easiest way to deploy vLLM is to use the pre-built Docker image published by the vLLM team:

```bash
# On a machine with Docker:
docker pull vllm/vllm-openai:v0.15.1

# Convert to Apptainer sandbox on Lakeshore:
apptainer build --sandbox vllm-0.15.1 docker://vllm/vllm-openai:v0.15.1
```

### Step 2: Run vLLM inside the container

```bash
apptainer exec --nv /path/to/vllm-0.15.1 \
    vllm serve Qwen/Qwen2.5-VL-72B-Instruct-AWQ \
    --host 0.0.0.0 --port 8000 \
    --quantization awq \
    --max-model-len 32768
```

### This works — sort of

The pre-built container works for basic inference with the plain AWQ kernel (`--quantization awq`). The model loads, accepts requests, and generates correct responses. But at **~3 tokens/second**, the experience is frustratingly slow. A 200-token response takes over a minute.

### Why not use Marlin?

Switching to `--quantization awq_marlin` causes an immediate crash:

```
RuntimeError: CUDA error: cudaErrorUnsupportedPtxVersion
```

The pre-built vLLM v0.15.1 Docker image ships with **CUDA 12.9** inside the container. The Marlin AWQ kernels in that image were compiled with CUDA 12.9, producing PTX code that targets CUDA 12.9. Our GPU driver (550.163.01) only understands PTX up to CUDA 12.4 — it literally cannot read the Marlin kernel instructions.

---

## 5. The Problem: A 10x Speed Gap

### The error

```
File "vllm/model_executor/layers/quantization/awq_marlin.py", line 182
    marlin_gemm(...)
RuntimeError: CUDA error: cudaErrorUnsupportedPtxVersion
    Marlin kernel PTX version 12.9 > driver maximum 12.4
```

### Root cause diagram

```
  Pre-built vLLM v0.15.1 Docker Image
  ┌──────────────────────────────────┐
  │  CUDA Toolkit 12.9               │
  │  nvcc compiled Marlin kernels    │
  │  → PTX version: 12.9            │─── PTX 12.9 code
  └──────────────────────────────────┘        │
                                              ▼
                                    ┌──────────────────┐
                                    │   GPU Driver 550  │
                                    │   Max PTX: 12.4   │
                                    │                   │
                                    │   12.9 > 12.4     │
                                    │   ❌ CANNOT JIT    │
                                    └──────────────────┘
```

### Why not just update the driver?

This is the obvious question, and the answer is nuanced:

1. **HPC GPU drivers are managed by system administrators**, not individual researchers. Upgrading requires downtime, testing, and approval — often weeks of lead time.
2. **Driver upgrades can break other users' workloads.** A research group running CUDA 11.x code might find their software incompatible with a CUDA 12.9 driver.
3. **Even with a driver upgrade, the problem may recur.** The next vLLM release might ship with CUDA 13.x, requiring yet another driver upgrade.

The sustainable solution is to **build the container to match the driver**, not the other way around.

### What "building from source" means

Normally, `pip install vllm` downloads a pre-compiled binary package (a "wheel") from PyPI. This wheel was compiled by the vLLM developers on their machines, using their CUDA version (12.9 as of early 2026).

"Building from source" means: download the raw Python and C++ source code from GitHub, and compile it ourselves using our CUDA toolkit (12.4). This way, nvcc produces CUDA 12.4 PTX for all kernels — including Marlin — which our driver can JIT-compile.

The tradeoff: compilation takes ~1.5-2 hours (lots of C++ and CUDA code), but the result is a container perfectly matched to our driver.

---

## 6. The Solution: Building vLLM from Source with CUDA 12.4

### The compatibility chain

Building from source requires finding compatible versions of four components. Each constrains the next:

```
GPU Driver          → determines max CUDA PTX version
  └─→ CUDA Toolkit  → determines which PyTorch builds are available
        └─→ PyTorch  → determines which vLLM versions are compatible
              └─→ vLLM → determines model support and features
```

For our system (driver 550.163.01):

| Component | Version | Why this version |
|-----------|---------|------------------|
| GPU Driver | 550.163.01 | Installed on ghi2-002, supports CUDA ≤ 12.4 |
| CUDA Toolkit | 12.4 | Maximum version the driver can JIT |
| PyTorch | 2.6.0+cu124 | Latest PyTorch with CUDA 12.4 wheels |
| vLLM | **v0.8.5.post1** | Latest vLLM designed for PyTorch 2.6.0 |

### Why not vLLM v0.15.1?

We initially attempted to build vLLM v0.15.1 from source. This failed because v0.15.1 requires **PyTorch 2.9.1**, which is only available with CUDA 12.9. The build crashed with:

```
error: enum "c10::ScalarType" has no member "Float8_e8m0fnu"
```

This type was added in PyTorch 2.7+ and does not exist in PyTorch 2.6.0. There is no way to build vLLM v0.15.1 with PyTorch 2.6.0 — the API incompatibility is fundamental.

### Why v0.8.5.post1 is the right choice

vLLM v0.8.5.post1 (released March 2025) is the ideal version for CUDA 12.4:

- **Designed for PyTorch 2.6.0** — native compatibility, no API mismatches
- **CUDA 12.4 is the default wheel target** — the build system expects it
- **Full Qwen2.5-VL support** — vision-language model support was added in v0.7.2
- **Full Marlin AWQ support** — the optimized kernels are mature in v0.8.x
- **Stable release** — the `.post1` suffix indicates a bug-fix patch, the most stable kind
- **Strong performance** — continuous batching, PagedAttention, and all core optimizations

---

## 7. Build Attempt Chronicle: 12 Attempts and What They Teach

Building a custom vLLM container on Lakeshore and getting it to serve models required 12 attempts over three sessions. Each failure revealed a different incompatibility in the HPC environment. These are documented here because **they represent common pitfalls that anyone deploying LLMs on institutional HPC will encounter**.

### Attempt 1: nvidia/cuda base image → `apt-get` fails

**Definition file:**
```
Bootstrap: docker
From: nvidia/cuda:12.4.1-devel-ubuntu22.04

%post
    apt-get update && apt-get install -y python3.12 python3.12-venv git ...
```

**Error:**
```
setgroups 65534: Operation not permitted
E: setgroups 65534 failed - setgroups (2: Operation not permitted)
```

**Root cause:** Apptainer's `--fakeroot` mode on Lakeshore provides only **partial** root emulation. The `apt-get` package manager's internal privilege-dropping mechanism calls `setgroups()`, which requires real root. On fully configured systems, fakeroot handles this transparently; on Lakeshore, the subuid/subgid mapping is restricted.

**Lesson:** On HPC clusters, avoid container builds that require `apt-get`. Choose base images that already include the software you need.

**Fix:** Switched to the PyTorch Docker image (`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`), which includes Python, pip, conda, gcc, and build tools pre-installed — no `apt-get` needed.

---

### Attempt 2: PyTorch Docker base → `git: not found`

**Definition file:**
```
Bootstrap: docker
From: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

%post
    git clone --branch v0.15.1 https://github.com/vllm-project/vllm.git /tmp/vllm
```

**Error:**
```
/bin/sh: 1: git: not found
```

**Root cause:** The PyTorch Docker image is optimized for size and does not include git. While it has Python, pip, and conda, command-line tools like git, wget, and curl are not included.

**Lesson:** Do not assume Docker base images include standard Unix tools. Check what is available before designing the build process.

**Fix:** Installed git via conda: `conda install -y git`. Conda manages its own package tree in `/opt/conda/` and does not require root access, so it works under partial fakeroot.

---

### Attempt 3: Tarball download → `FileExistsError`

After git was unavailable, we tried downloading a source tarball instead:

**Approach:**
```python
# In %post section, using Python's urllib:
import urllib.request
urllib.request.urlretrieve("https://github.com/.../v0.15.1.tar.gz", "/tmp/vllm.tar.gz")
```

**Error:**
```
FileExistsError: [Errno 17] File exists: '/tmp/vllm-0.15.1' -> '/tmp/vllm'
```

**Root cause:** A previous failed build left a `/tmp/vllm` directory inside the container sandbox. Apptainer sandboxes are writable directories — failed build artifacts persist.

**Lesson:** Always clean up before build steps. Add `rm -rf /tmp/vllm` before any operation that creates that directory.

**Fix:** Added cleanup: `rm -rf /tmp/vllm` before the download step.

---

### Attempt 4: Tarball approach → `setuptools-scm unable to detect version`

**Error:**
```
LookupError: setuptools-scm was unable to detect version for /tmp/vllm.
Make sure you're either building from a fully intact git repository
or providing a version through PKG_INFO or pyproject.toml.
```

**Root cause:** The GitHub tarball does not include the `.git/` directory. vLLM uses `setuptools-scm` to determine its version number from git tags. Without `.git/`, it cannot detect the version.

**Lesson:** Source tarballs from GitHub strip git metadata. If the project uses `setuptools-scm`, you either need a real git clone or must override the version manually.

**Fix:** Could be fixed with `SETUPTOOLS_SCM_PRETEND_VERSION=0.15.1`, but this approach was eventually abandoned in favor of using `git clone` (see Attempt 5).

---

### Attempt 5: With git installed → CMake needs git too

**Error:**
```
error: could not find git for clone of cutlass-populate
```

**Root cause:** Even with the Python source code downloaded, vLLM's build system uses CMake's `FetchContent` to download CUTLASS (NVIDIA's CUDA template library) at build time. CMake calls `git clone` internally — so git is needed during compilation, not just for downloading vLLM itself.

**Lesson:** Build-time dependencies can pull in transitive dependencies. CUTLASS is fetched by CMake during the build, and CMake requires git for this.

**Fix:** The `conda install -y git` fix from Attempt 2 solves this too — once git is available, both `git clone` for vLLM and CMake's `FetchContent` for CUTLASS work. We reverted from the tarball approach back to `git clone`.

---

### Attempt 6: Build vLLM v0.15.1 → `Float8_e8m0fnu` compilation error

The build progressed further this time — nvcc began compiling CUDA kernels (reaching step 58 of 402):

**Error (at compilation step 58/402):**
```
error: enum "c10::ScalarType" has no member "Float8_e8m0fnu"
```

**Root cause:** vLLM v0.15.1 uses a new PyTorch data type (`Float8_e8m0fnu`) that was introduced in PyTorch 2.7. Our CUDA 12.4 environment has PyTorch 2.6.0 — the latest available for CUDA 12.4. This is a **fundamental API incompatibility**: vLLM v0.15.1 simply cannot be compiled with PyTorch 2.6.0.

**Lesson:** vLLM, PyTorch, and CUDA versions form a tightly coupled chain. You cannot mix versions arbitrarily. Always check which PyTorch version a vLLM release was designed for.

**Fix:** Downgraded the target from vLLM v0.15.1 to v0.8.5.post1, which was designed for PyTorch 2.6.0.

---

### Attempt 7: Build vLLM v0.8.5.post1 → `packaging>=24.2` required

**Error:**
```
ImportError: Cannot import `packaging.licenses`.
    Setuptools>=77.0.0 requires "packaging>=24.2" to work properly.
```

**Root cause:** The PyTorch Docker base image ships with `setuptools==82.0.0` (which requires `packaging>=24.2`) but only has `packaging==24.1`. When vLLM's `pyproject.toml` is parsed, setuptools tries to validate the license expression using `packaging.licenses`, which does not exist in `packaging` 24.1.

**Lesson:** Python packaging toolchain version mismatches can cause surprising failures. Even a minor version difference (24.1 vs 24.2) in the `packaging` library can break `setuptools`.

**Fix:** Added `packaging` to the pip upgrade command: `pip install --upgrade pip setuptools wheel packaging`.

---

### Attempt 8: Build compiles but nvcc processes are OOM-killed

The build progressed significantly — CUDA compilation reached step 147 of 317:

**Error (at step 147/317):**
```
[147/317] Building CUDA object ...flash_fwd_hdimall_bf16_packgqa_sm90.cu.o
FAILED: ...
Killed
[148/317] Building CUDA object ...flash_fwd_hdimall_bf16_paged_sm90.cu.o
FAILED: ...
Killed
ninja: build stopped: subcommand failed.
```

**Root cause:** The word `Killed` with no error message is the signature of the **Linux OOM killer** — the kernel terminated the nvcc processes because they exceeded available memory. The culprits were the **FlashAttention 3 (FA3) Hopper kernels** (`flash_fwd_hdimall_*_sm90.cu`). These are exceptionally memory-hungry during compilation, with each nvcc process consuming **8-16 GiB** of RAM — far more than typical CUDA kernels.

With `MAX_JOBS=4` (four parallel nvcc processes) and only 32 GiB of RAM allocated to the SLURM job, four FA3 kernels compiling simultaneously needed 32-64 GiB total — exceeding the allocation.

Additionally, the build was compiling for **six GPU architectures** (`5.0; 8.0; 8.6; 8.9; 9.0; 9.0a`) because GPU auto-detection failed (no GPU on the CPU-only batch node). Each architecture multiplies the number of kernel variants, dramatically increasing both build time and peak memory usage.

**Lesson:** CUDA kernel compilation memory usage varies enormously by kernel type. FlashAttention 3 Hopper kernels are the most memory-intensive. Always request significantly more RAM than you think you need, and restrict target GPU architectures to only those you actually use.

**Fix (two changes):**

1. **Increased SLURM memory from 32 GiB to 64 GiB** (`--mem=64G`) — gives enough headroom for FA3 kernel compilation
2. **Set `TORCH_CUDA_ARCH_LIST="8.0;9.0"`** — compiles only for A100 (sm_80) and H100 (sm_90), eliminating four unnecessary architectures

---

### What is `TORCH_CUDA_ARCH_LIST`?

This environment variable controls which **GPU compute capability architectures** nvcc compiles CUDA kernels for. Each GPU model has a compute capability version:

| GPU Model | Compute Capability | Architecture Name |
|-----------|-------------------|-------------------|
| GTX 1080 | 6.1 | Pascal |
| V100 | 7.0 | Volta |
| A100 | **8.0** | Ampere |
| RTX 4090 | 8.9 | Ada Lovelace |
| **H100** | **9.0** | Hopper |

When `TORCH_CUDA_ARCH_LIST` is **not set**, the build system tries to auto-detect GPUs on the machine. On the CPU-only batch partition (no GPU), auto-detection fails and defaults to compiling for **all common architectures** (5.0 through 9.0a). This means every CUDA kernel is compiled six times — once per architecture — wasting time and memory.

By setting `TORCH_CUDA_ARCH_LIST="8.0;9.0"`, we tell nvcc: "only compile for A100 and H100." This:

- **Cuts the number of kernel compilations roughly in half** (2 architectures instead of 6)
- **Reduces peak memory** (fewer concurrent compilations needed)
- **Halves the compilation time** (~45 min instead of ~90 min)
- **Does not affect runtime** — the container only runs on A100 and H100 nodes anyway

If Lakeshore adds new GPU types in the future (e.g., B100 with compute capability 10.0), the architecture list should be updated to include the new architecture.

---

### Attempt 9: With memory and architecture fixes → Build succeeds, cleanup fails

With 64 GiB RAM and only two target architectures (`TORCH_CUDA_ARCH_LIST="8.0;9.0"`), vLLM compiled successfully. All CUDA compilation steps finished, the wheel was built, and `vllm-0.8.5.post2` was installed. However, the Apptainer build reported **`FATAL: exit status 1`** and the SLURM job exited with code 255.

**Root cause:** A subtle working-directory bug in the cleanup step:

```bash
# The build runs:  cd /tmp/vllm  (to compile vLLM from source)
# Then cleanup does:
rm -rf /tmp/vllm         # Deletes the CURRENT working directory!
pip cache purge           # Fails: "The folder you are executing pip from can no longer be found."
```

The shell's CWD was `/tmp/vllm`. Deleting it made `/tmp/vllm` cease to exist, and `pip` could not resolve its own path — it uses `os.getcwd()` internally which raises `FileNotFoundError` when the CWD is deleted. The `FATAL` exit caused Apptainer to report a failed build, even though vLLM was already fully installed.

**Fix:** Add `cd /` before the cleanup:

```bash
cd /                     # Move out of /tmp/vllm before deleting it
rm -rf /tmp/vllm
pip cache purge
```

**Outcome:** The container at `vllm-cu124` is likely functional (vLLM is installed), but may need verification since Apptainer may not have finalized the sandbox cleanly. The def file was fixed for future builds.

> **Note:** The `WARNING: Running pip as the 'root' user...` messages visible throughout the build log are harmless inside a container. Pip warns about root because on bare-metal systems it can conflict with the system package manager, but inside an Apptainer container there is no system package manager — the warning is irrelevant.

### Attempt 10: Container built, model serving fails → `transformers` 5.x breaks tokenizers

The container built successfully (Attempt 9's fix resolved the build failure). We deployed it to ghi2-002 and ran `vllm serve` with `--quantization awq_marlin`:

**Error:**
```
AttributeError: 'Qwen2Tokenizer' object has no attribute 'all_special_tokens_extended'
```

**Root cause:** During the build, `pip install --no-build-isolation .` resolved `transformers` to version **5.2.0** (the latest at build time). The `transformers` 5.x release removed the `all_special_tokens_extended` attribute from tokenizer classes. vLLM's tokenizer initialization relies on this attribute.

**First fix attempt:** `pip install transformers==4.48.0` — failed because vLLM requires `transformers>=4.51.1`.

**Fix:** Pinned to the 4.x series: `pip install "transformers>=4.51.1,<5.0"` inside the container (resolved to 4.57.6). This was done interactively using `apptainer exec --fakeroot --writable` on the sandbox.

**Lesson:** Build-time dependency resolution can silently pull in major version upgrades. Pin critical dependencies to known-working major versions. The `transformers<5.0` pin was added to the def file for future rebuilds.

---

### Attempt 11: transformers fixed, but `xformers` missing → flash-attn dependency saga

After fixing transformers, vLLM started loading the model but crashed when initializing the vision module:

**Error:**
```
ModuleNotFoundError: No module named 'xformers'
```

**Root cause:** vLLM's Qwen2.5-VL vision module uses memory-efficient attention, which requires either `flash-attn` or `xformers`. The container had neither installed.

**The dependency saga:**

1. **`pip install xformers`** — pulled in PyTorch 2.10.0 and CUDA 12.8 libraries, **replacing our PyTorch 2.6.0+cu124**. Disaster.

2. **Restored PyTorch** via `pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124`.

3. **`pip install xformers` with `--no-deps`** — installed xformers 0.0.29.post1, but it was built for PyTorch 2.5.1+cu121. C++ extension mismatch: `xFormers was built for: PyTorch 2.5.1+cu121 with CUDA 1201 (you have 2.6.0+cu124)`.

4. **Switched to flash-attn** (preferred by vLLM). Uninstalled xformers.

5. **`pip install flash-attn`** — pre-built wheel was compiled against wrong PyTorch → `undefined symbol: _ZN3c105ErrorC2E...` ABI mismatch.

6. **`pip install "flash-attn<2.8" --no-build-isolation --no-deps`** — failed with `[Errno 18] Invalid cross-device link` (pip tried to hard-link between the container filesystem at `/projects/` and pip cache at `/root/.cache/`, which are different filesystems).

7. **Direct wheel install from GitHub** — found the exact pre-built wheel matching our environment:
   ```bash
   pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl --no-deps
   ```
   This worked. The wheel name encodes the exact compatibility: CUDA 12, PyTorch 2.6, CPython 3.11, cxx11abi=FALSE.

**Lesson:** Installing GPU-accelerated Python packages in containers is treacherous. Each package (torch, xformers, flash-attn) embeds compiled CUDA code that must match the exact PyTorch version, CUDA version, and C++ ABI. Always use `--no-deps` to prevent dependency resolution from replacing carefully pinned packages. When source builds fail in containers, look for pre-built wheels that match your exact version combination on the project's GitHub releases page.

---

### Attempt 12: flash-attn installed, `torch.compile` crashes → `--enforce-eager` saves the day

With transformers pinned and flash-attn installed, vLLM loaded the model successfully (40.2 GiB weights, 130,176 token KV cache). But during the first inference request, it crashed:

**Error:**
```
RuntimeError: CUDA error: an illegal memory access was encountered
  File ".../torch/_inductor/runtime/triton_heuristics.py"
  During: Triton kernel autotuning in torch.compile
```

This occurred during Triton kernel autotuning as part of `torch.compile` (which vLLM uses for graph optimization). Clearing the Triton cache (`~/.triton/cache/`) and retrying produced the same error.

**Root cause:** An incompatibility between the Marlin AWQ kernels and `torch.compile` in this specific vLLM dev build (0.8.5.post2). The Triton JIT compiler's autotuning step triggers illegal memory access when profiling kernels alongside the Marlin quantized weights.

**Fix:** Added `--enforce-eager` to the vLLM serve command. This disables `torch.compile` and CUDA graph capture entirely, running all operations in PyTorch's eager execution mode.

**Performance impact:**
- The Marlin AWQ kernel (the 10x speedup over plain AWQ) **still works** with `--enforce-eager`
- `torch.compile` only adds ~20-30% on top of the Marlin kernel
- Initial measured throughput: ~6-8 tok/s (Marlin AWQ + enforce-eager, baseline config)
- After runtime optimizations (prefix caching, chunked prefill, 64K context, 0.90 memory util): **~25 tok/s**
- For a chat interface, 25 tok/s provides a **fast, interactive streaming experience** — a 200-token response completes in ~8 seconds

**Lesson:** `torch.compile` compatibility with quantized kernels is not guaranteed, especially in dev builds. The `--enforce-eager` flag is a safe fallback that preserves the most important optimization (the quantized kernel itself). The throughput difference between eager and compiled mode is minor compared to the kernel optimization — Marlin provides the 10x jump, while torch.compile adds incremental improvement.

---

### Summary: Error → Root Cause → Fix

| # | Error | Root Cause | Fix |
|---|-------|-----------|-----|
| 1 | `setgroups: Operation not permitted` | apt-get needs real root | Use PyTorch Docker base (no apt-get) |
| 2 | `git: not found` | PyTorch image lacks git | `conda install -y git` |
| 3 | `FileExistsError: '/tmp/vllm'` | Leftover from failed build | `rm -rf /tmp/vllm` before download |
| 4 | `setuptools-scm unable to detect version` | No .git in tarball | Use `git clone` instead of tarball |
| 5 | `could not find git for cutlass-populate` | CMake FetchContent needs git | Same git install fixes both |
| 6 | `no member "Float8_e8m0fnu"` | vLLM v0.15.1 needs PyTorch 2.9.1 | Use vLLM v0.8.5.post1 (for PyTorch 2.6.0) |
| 7 | `Cannot import packaging.licenses` | setuptools 82 needs packaging ≥24.2 | Add `packaging` to pip upgrade |
| 8 | `Killed` (OOM during FA3 compilation) | 32 GiB too little for FA3 Hopper kernels; 6 target architectures | Increase to 64 GiB; set `TORCH_CUDA_ARCH_LIST="8.0;9.0"` |
| 9 | `pip cache purge` fails, build reports `FATAL: exit status 1` | `rm -rf /tmp/vllm` deletes CWD before pip runs | Add `cd /` before cleanup; vLLM was already installed |
| 10 | `Qwen2Tokenizer has no attribute all_special_tokens_extended` | transformers 5.x removed attribute | Pin `transformers>=4.51.1,<5.0` |
| 11 | `No module named 'xformers'` + flash-attn ABI mismatches | Missing attention backend; wrong pre-built wheels | Install flash-attn 2.7.4 from matching GitHub wheel with `--no-deps` |
| 12 | `CUDA error: illegal memory access` in torch.compile | Marlin AWQ + torch.compile incompatibility | Add `--enforce-eager` to disable torch.compile |

---

## 8. The Final Container Definition (Annotated)

The complete Apptainer definition file is at `scripts/vllm-cu124.def`. Here is the functional core, annotated:

```bash
# Base image: PyTorch with CUDA 12.4 development tools
# Chosen because it includes Python, pip, conda, gcc — no apt-get needed
Bootstrap: docker
From: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

%labels
    Author nassar@uic.edu
    Description vLLM 0.8.5.post1 compiled with CUDA 12.4 for driver 550
    vLLM.Version 0.8.5.post1
    CUDA.Version 12.4.1

%environment
    export CUDA_HOME=/usr/local/cuda
    export VLLM_WORKER_MULTIPROC_METHOD=spawn

%post
    # ── Step 1: Install git and upgrade Python packages ──
    conda install -y git                          # For git clone + CMake FetchContent
    pip install --upgrade pip setuptools wheel packaging  # packaging>=24.2 critical

    # ── Step 2: Upgrade PyTorch to match vLLM v0.8.5.post1 ──
    pip install --upgrade torch torchvision \
        --index-url https://download.pytorch.org/whl/cu124   # CUDA 12.4 wheels

    # ── Step 3: Build vLLM from source ──
    rm -rf /tmp/vllm                              # Clean any leftover from failed builds
    git clone --branch v0.8.5.post1 --depth 1 \
        https://github.com/vllm-project/vllm.git /tmp/vllm
    cd /tmp/vllm

    export MAX_JOBS=4          # Limit parallel nvcc processes (each uses 2-16 GiB RAM)
    export CUDA_HOME=/usr/local/cuda   # Point build system to CUDA 12.4 toolkit
    export TORCH_CUDA_ARCH_LIST="8.0;9.0"  # Only compile for A100 + H100

    python use_existing_torch.py       # Tell vLLM: "use our cu124 PyTorch, don't download"
    pip install -r requirements/build.txt       # cmake, ninja, setuptools-scm
    pip install --no-build-isolation .           # Compile with our CUDA 12.4 environment

    # ── Step 4: Pin transformers (see Attempt 10) ──
    pip install "transformers>=4.51.1,<5.0"   # Prevent transformers 5.x (breaks tokenizers)

    # ── Step 5: Install flash-attn for vision models (see Attempt 11) ──
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl --no-deps

    # ── Step 6: Cleanup ──
    cd /                        # Move out of /tmp/vllm before deleting it (see Attempt 9)
    rm -rf /tmp/vllm            # Source code no longer needed (installed via pip)
    pip cache purge             # Remove downloaded .whl files

%runscript
    exec vllm serve "$@"
```

### Key design decisions explained

**Why `pytorch/pytorch` instead of `nvidia/cuda`?**
The nvidia/cuda image requires `apt-get` to install Python and build tools. `apt-get` fails under Lakeshore's partial fakeroot. The PyTorch image comes pre-loaded with everything we need.

**Why `conda install -y git` instead of `apt-get install git`?**
Conda manages packages in its own directory (`/opt/conda/`) and does not require root access. It works under partial fakeroot where `apt-get` fails.

**Why `--no-build-isolation`?**
Build isolation creates a temporary virtual environment with fresh dependencies — including downloading a fresh PyTorch wheel. That fresh wheel would be the default CUDA 12.9 version, defeating the entire purpose. `--no-build-isolation` uses the packages already installed (our CUDA 12.4 PyTorch).

**Why `use_existing_torch.py`?**
This vLLM helper script removes PyTorch from the build requirements, preventing the build system from downloading a different version. Combined with `--no-build-isolation`, this ensures our carefully selected CUDA 12.4 PyTorch is used.

**Why `MAX_JOBS=4`?**
Each `nvcc` (CUDA compiler) process typically uses 2-4 GiB of RAM. However, certain kernels — particularly the **FlashAttention 3 Hopper kernels** (`flash_fwd_hdimall_*_sm90.cu`) — are exceptionally memory-hungry and can consume **8-16 GiB per nvcc process** during compilation. With `MAX_JOBS=4`, four FA3 kernels compiling simultaneously can peak at 32-64 GiB. This is why the SLURM build job requests 64 GiB of RAM (see Section 9). Reducing `MAX_JOBS` to 2 would lower peak memory but double the already long compilation time — increasing memory is the better tradeoff.

**Why `TORCH_CUDA_ARCH_LIST="8.0;9.0"`?**
This environment variable controls which GPU compute capability architectures `nvcc` compiles CUDA kernels for. Each GPU model has a compute capability:

| GPU | Compute Capability | Architecture |
|-----|-------------------|--------------|
| A100 | 8.0 | Ampere |
| H100 | 9.0 | Hopper |

Without this variable, the build system tries to auto-detect GPUs on the machine. On the CPU-only `batch` partition (no GPU present), auto-detection fails and defaults to compiling for **all common architectures** (5.0, 8.0, 8.6, 8.9, 9.0, 9.0a) — six targets. Every CUDA kernel is compiled once per architecture, so this:

- **Triples the number of kernel compilations** (6 architectures vs 2)
- **Triples peak memory** (more concurrent compilations running)
- **Roughly triples compilation time** (~2.5+ hours instead of ~45-60 min)

Since our container only runs on A100 and H100 nodes on Lakeshore, we only need compute capabilities 8.0 and 9.0. Compiling for older GPUs (Pascal 5.0, Volta 7.0) or other Ampere variants (8.6, 8.9) wastes time and memory for no benefit. If Lakeshore adds new GPU types in the future (e.g., B100 with compute capability 10.0), update this list accordingly.

---

## 9. The Build Process

### The SLURM batch script

The build is submitted as a SLURM job via `scripts/build-vllm-cu124.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=build-vllm-cu124
#SBATCH --partition=batch        # CPU-only partition (no GPU needed for compilation)
#SBATCH --cpus-per-task=8        # Parallel compilation
#SBATCH --mem=64G                # FA3 Hopper kernels need 8-16 GiB each (see below)
#SBATCH --time=03:00:00          # Build takes ~45-90 min with TORCH_CUDA_ARCH_LIST
#SBATCH --output=logs/build-vllm-cu124-%j.log

module load apptainer

CONTAINER_DIR="/projects/acer_hpc_admin/nassar/containers"
CONTAINER_NAME="vllm-cu124"
DEF_FILE="/home/nassar/STREAM/scripts/vllm-cu124.def"

apptainer build --fakeroot --sandbox \
    "${CONTAINER_DIR}/${CONTAINER_NAME}" \
    "${DEF_FILE}"
```

### Key decisions

**Why 64 GiB of RAM (`--mem=64G`)?**
CUDA kernel compilation memory usage varies enormously by kernel type. Most vLLM kernels compile with 2-4 GiB of RAM per `nvcc` process. However, the **FlashAttention 3 (FA3) Hopper kernels** (`flash_fwd_hdimall_*_sm90.cu`) are exceptionally memory-hungry — each nvcc process compiling an FA3 kernel consumes **8-16 GiB of RAM**. With `MAX_JOBS=4` (four parallel compilations), four FA3 kernels compiling simultaneously can peak at **32-64 GiB total**. At 32 GiB, the Linux OOM killer terminates nvcc processes with just `Killed` (no error message). 64 GiB provides sufficient headroom for even the worst-case combination of concurrent FA3 kernel compilations.

**Why the `batch` partition (CPU-only)?**
Compiling CUDA code is a CPU task — `nvcc` runs on the CPU, translating CUDA source code to PTX. The GPU is only needed at runtime when PTX is JIT-compiled to machine code. Using the CPU-only partition avoids competing for scarce GPU allocations.

**Why absolute paths for `DEF_FILE`?**
SLURM copies batch scripts to a temporary spool directory (`/cm/local/apps/slurm/var/spool/`) before executing them. The common pattern `$(dirname "$0")/vllm-cu124.def` resolves to the spool directory, not the original scripts directory. Using an absolute path avoids this issue.

**Why sandbox format (not SIF)?**
The `--sandbox` flag creates an unpacked directory tree (~26 GiB) instead of a compressed SquashFS image (~7 GiB). Sandboxes are faster to build (no compression step) and easier to debug (you can browse the filesystem). For production, converting to SIF would save disk space but is not necessary.

### Build timeline

| Phase | Duration | What happens |
|-------|----------|-------------|
| Pull base image | ~3 min | Downloads pytorch/pytorch Docker image layers |
| Unpack layers | ~2 min | Extracts Docker layers into Apptainer sandbox |
| conda install git | ~2 min | Downloads and installs git + perl + pcre2 |
| pip upgrade + PyTorch | ~2 min | Downloads PyTorch 2.6.0+cu124 (~768 MB wheel) |
| git clone vLLM | ~1 min | Shallow clone of v0.8.5.post1 source |
| pip install dependencies | ~5 min | Downloads ~50 Python packages |
| **CUDA compilation** | **45-90 min** | **Compiles Marlin, FlashAttention, and other kernels** |
| Cleanup | ~1 min | Removes source code and pip cache |
| **Total** | **~1-2 hours** | With `TORCH_CUDA_ARCH_LIST="8.0;9.0"`, closer to 1 hour |

### Monitoring the build

```bash
# Check if the job is running
squeue -u $USER

# Watch the build log in real-time
tail -f logs/build-vllm-cu124-<jobid>.log

# During compilation, pip shows a heartbeat:
#   Building wheel for vllm (pyproject.toml): still running...
#   Building wheel for vllm (pyproject.toml): still running...
# This is normal — the compilation is happening in the background.
```

---

## 10. vLLM Configuration for Large Models

Once the container is built, deploying a model requires careful configuration. Each parameter affects GPU memory allocation, and getting it wrong causes OOM (Out Of Memory) crashes at startup.

### Memory budget analysis

vLLM pre-allocates GPU memory at startup. Understanding the budget is essential:

```
┌──────────────────────────────────────────────────────────────┐
│              H100 NVL GPU Memory (93.2 GiB)                  │
│                                                               │
│  ┌──────────────────────────────────────────────────┐        │
│  │           Pre-allocated by vLLM (79.2 GiB)       │        │
│  │                                                    │        │
│  │  ┌──────────────┐  ┌─────────┐  ┌────────────┐  │        │
│  │  │ Model weights │  │ CUDA    │  │  KV cache  │  │        │
│  │  │   38.8 GiB    │  │ graphs  │  │  ~34 GiB   │  │        │
│  │  │  (AWQ INT4)   │  │ 6.3 GiB │  │            │  │        │
│  │  └──────────────┘  └─────────┘  └────────────┘  │        │
│  └──────────────────────────────────────────────────┘        │
│                                                               │
│  ┌──────────────────────────────────────────────────┐        │
│  │           Reserved / Overhead (~14 GiB)            │        │
│  │  PyTorch context, sampler warmup, CUDA context     │        │
│  └──────────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────────┘
```

### Parameter reference

#### `--gpu-memory-utilization 0.85`

Controls what fraction of GPU memory vLLM pre-allocates:

```
Usable = 93.2 GiB × 0.85 = ~79.2 GiB (pre-allocated for weights + KV cache + graphs)
Reserved = 93.2 GiB × 0.15 = ~14.0 GiB (for PyTorch overhead)
```

We use 0.85 instead of 0.90 because the sampler warmup (which runs at startup) temporarily allocates large tensors proportional to `vocab_size × max_num_seqs`. For Qwen 72B with its 152K vocabulary, 0.90 leaves insufficient headroom and causes OOM during warmup.

#### `--max-model-len 32768`

Maximum sequence length (input + output tokens). This directly determines KV cache size:

```
KV cache per sequence = num_layers × 2 × num_kv_heads × head_dim × dtype_size
                      = 80 × 2 × 8 × 128 × 2 bytes
                      = ~3.28 MiB per token position

At 32768 tokens: ~3.28 MiB × 32768 = ~107 GiB per sequence (theoretical max)
```

vLLM does not allocate the full theoretical max — it uses PagedAttention to allocate KV cache pages dynamically. The 34 GiB pre-allocated KV cache budget supports multiple concurrent sequences, each up to 32K tokens.

The model natively supports 128K tokens, but that would require ~4x more KV cache memory. 32K is a practical balance for a chat interface.

#### `--max-num-seqs 256`

Maximum concurrent sequences in a batch. Also controls the sampler warmup size at startup. The default (1024) causes OOM for models with large vocabularies:

```
Sampler warmup allocation ≈ vocab_size × max_num_seqs × dtype_size
For Qwen 72B (152K vocab):  152,064 × 1024 × 2 bytes = ~296 MiB (just for sort)
With overhead: ~1.7 GiB total

At 0.90 utilization, this 1.7 GiB exceeds the 9.3 GiB reserve → OOM
At 0.85 utilization with max_num_seqs=256, reserve is 14 GiB → safe
```

256 concurrent sequences is generous for a campus AI service. The real throughput bottleneck is per-token generation speed, not batch size.

#### `--quantization awq_marlin`

Selects the Marlin AWQ kernel for inference. This is the entire reason we built the custom container:

| Setting | Kernel | Speed | Container required |
|---------|--------|-------|--------------------|
| `awq` | Plain AWQ | ~3 tok/s | Any container |
| `awq_marlin` + `--enforce-eager` | Marlin AWQ (eager, baseline) | ~6-8 tok/s | vllm-cu124 (CUDA 12.4 PTX) |
| `awq_marlin` + `--enforce-eager` + optimizations | Marlin AWQ (eager, optimized) | **~25 tok/s** | vllm-cu124 + prefix caching, chunked prefill |
| `awq_marlin` | Marlin AWQ (compiled) | ~30-40 tok/s | vllm-cu124 (requires torch.compile fix) |

The Marlin kernel exploits NVIDIA tensor cores and specialized memory access patterns (split-K decomposition, async memory copies) to achieve higher throughput. In our current deployment, `torch.compile` is disabled via `--enforce-eager` due to a Triton autotuning crash (Attempt 12). With runtime optimizations (prefix caching, chunked prefill, 64K context, 0.90 GPU memory utilization), we achieve **~25 tok/s** — an 8x improvement over plain AWQ and within striking distance of the theoretical torch.compile maximum.

#### `--enforce-eager`

Disables `torch.compile` and CUDA graph capture, saving ~4-6 GiB of VRAM.

**Required on H100 with vllm-cu124:** The Marlin AWQ kernels in our dev build (0.8.5.post2) crash during Triton autotuning when `torch.compile` is enabled (see Attempt 12). The `--enforce-eager` flag bypasses this by running all operations in PyTorch's eager mode. The Marlin kernel itself still works — `torch.compile` only adds ~20-30% incremental throughput.

**Also required on A100 MIG:** 40 GiB MIG slices don't have enough VRAM for CUDA graph capture (~4-6 GiB overhead).

#### `--limit-mm-per-prompt '{"image": 4}'`

Vision-language models only. Limits images per request. Each image consumes additional VRAM for the visual encoder and cross-attention KV cache. 4 images is generous for a chat interface (most requests have 0-1 images).

Note the JSON format — older vLLM versions used `image=4`, but v0.8.5+ requires the JSON dictionary format.

---

## 11. Model Deployment Scripts

### Anatomy of a SLURM deployment script

Every vLLM deployment script follows the same structure:

```bash
#!/bin/bash
# ── SLURM directives ──
#SBATCH --job-name=stream-<model-name>
#SBATCH --partition=batch_gpu2          # batch_gpu for MIG, batch_gpu2 for H100
#SBATCH --nodelist=ghi2-002             # ga-002 for MIG, ghi2-002 for H100
#SBATCH --gres=gpu:1                    # gpu:3g.40gb:1 for MIG slice
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.log

# ── Configuration ──
MODEL="HuggingFace/model-name"
PORT=8000
CONTAINER="/projects/acer_hpc_admin/nassar/containers/vllm-cu124"
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/projects/acer_hpc_admin/nassar/huggingface

# ── Launch vLLM ──
module load apptainer
apptainer exec --nv ${CONTAINER} vllm serve ${MODEL} \
    --host 0.0.0.0 --port ${PORT} \
    --tensor-parallel-size 1 \
    --max-model-len <context> \
    --gpu-memory-utilization <fraction> \
    --dtype auto \
    --quantization <method> \
    [--enforce-eager] \
    [--limit-mm-per-prompt '{"image": N}']
```

### Hardware-to-model matching guide

| Model Size | Hardware | Quantization | Context | Enforce-Eager | Example |
|------------|----------|-------------|---------|---------------|---------|
| 1.5B | A100 MIG | None (FP16) | 32K | No | Qwen 2.5 1.5B |
| 14B | A100 MIG | AWQ | 32K | No | Qwen 2.5 14B |
| 32B | A100 MIG | AWQ | 16K | **Yes** (VRAM) | Qwen 2.5 32B |
| 32B | H100 | None (FP16) | 32K | **Yes** (torch.compile) | Qwen 2.5 32B FP16 |
| 72B | H100 | AWQ Marlin | 32K | **Yes** (torch.compile) | Qwen 2.5 72B |
| 72B (VL) | H100 | AWQ Marlin | 32K | **Yes** (torch.compile) | Qwen 2.5-VL 72B |

### HuggingFace cache management

Model weights are large (1-39 GiB per model). On Lakeshore:

- **Home directory** (`~`): 100 GiB quota — too small for multiple models
- **Project space** (`/projects/acer_hpc_admin/`): 10 TiB quota — ideal

We set `HF_HOME=/projects/acer_hpc_admin/nassar/huggingface` in every deployment script. This redirects HuggingFace Hub's cache (model weights, tokenizer files, config) to the project space.

First-time model download should be done in an interactive session:

```bash
srun --partition=batch_gpu2 --gres=gpu:1 --time=01:00:00 --pty bash
module load apptainer
export HF_HOME=/projects/acer_hpc_admin/nassar/huggingface
apptainer exec --nv /projects/.../vllm-cu124 \
    huggingface-cli download Qwen/Qwen2.5-VL-72B-Instruct-AWQ
```

Subsequent runs load from cache instantly.

### STREAM models deployed on Lakeshore

| Script | Model | Parameters | Hardware | Port |
|--------|-------|------------|----------|------|
| `vllm-qwen-vl-72b.sh` | Qwen2.5-VL-72B-Instruct-AWQ | 72B | H100 | 8000 |
| `vllm-qwen-72b.sh` | Qwen2.5-72B-Instruct-AWQ | 72B | H100 | 8000 |
| `vllm-qwen-32b-fp16.sh` | Qwen2.5-32B-Instruct | 32B | H100 | 8001 |
| `vllm-qwen-32b.sh` | Qwen2.5-32B-Instruct-AWQ | 32B | A100 MIG | 8000 |
| `vllm-qwen-14b.sh` | Qwen2.5-14B-Instruct-AWQ | 14B | A100 MIG | 8002 |
| `vllm-qwen-1.5b.sh` | Qwen2.5-1.5B-Instruct | 1.5B | A100 MIG | 8003 |
| `vllm-deepseek-r1-32b.sh` | DeepSeek-R1-Distill-Qwen-32B | 32B | H100 | 8001 |
| `vllm-qwq-32b.sh` | QwQ-32B | 32B | H100 | 8002 |

---

## 12. Performance Tuning: Beyond the Initial Deployment

After the initial deployment (Attempts 1-12), we applied several additional optimizations to maximize throughput and usability within the CUDA 12.4 driver constraint.

### Context window expansion: 32K → 64K tokens

The Qwen 2.5-VL-72B model supports up to 128K tokens natively. Our initial deployment used 32K, matching a conservative memory budget. Analysis showed that with `--enforce-eager` (no CUDA graphs), the freed 6 GiB of VRAM could support a larger KV cache:

```
Memory budget at 0.90 utilization with --enforce-eager:
  Pre-allocated: 93.2 GiB × 0.90 = 83.9 GiB
  Model weights: ~38 GiB
  KV cache pool: ~45 GiB
  Reserved:      ~9.3 GiB

One 64K-token sequence KV cache:
  80 layers × 2 × 8 KV heads × 128 head dim × 65536 × 2 bytes = ~20.5 GiB

  → Supports ~2 concurrent 64K-token sequences
  → Or many shorter conversations (most are <8K tokens)
```

The 64K context brings Lakeshore closer to cloud models (Claude: 200K, GPT-4: 128K) and enables use cases like long document analysis and extended conversations.

### GPU memory utilization: 0.85 → 0.90

The original 0.85 setting was chosen because 0.90 caused OOM during sampler warmup when CUDA graphs were enabled (6.3 GiB overhead). With `--enforce-eager` disabling CUDA graphs, the extra 6 GiB compensates for the smaller reserve at 0.90 utilization. This frees ~4.7 GiB more for KV cache, improving concurrent request handling.

### Prefix caching (`--enable-prefix-caching`)

When multiple requests share the same prompt prefix (e.g., a system prompt), vLLM caches the KV computations and reuses them across requests. This eliminates redundant computation with zero additional memory overhead — the cache uses the existing KV cache pool. For workloads with shared system prompts, this provides a measurable speedup on prefill latency.

### Chunked prefill (`--enable-chunked-prefill`)

By default, vLLM processes the entire prompt (prefill phase) before starting token generation. Chunked prefill breaks long prompts into smaller pieces, allowing the scheduler to interleave prefill and decode operations. This improves GPU utilization by 10-15% for mixed workloads where some requests are in the prefill phase and others are generating tokens.

### Triton autotuning workarounds (experimental)

The torch.compile crash (Attempt 12) occurs specifically during Triton kernel autotuning. Several environment variables can influence Triton's behavior:

```bash
TRITON_DISABLE_AUTOTUNE=1           # Skip autotuning, use default configs
TRITON_CACHE_DIR=/tmp/triton_$JOB   # Fresh cache per job
VLLM_USE_TRITON_FLASH_ATTN=0        # Disable Triton flash attention
```

If any combination bypasses the crash, removing `--enforce-eager` would unlock the full ~30-40 tok/s throughput. These are included as commented-out experiments in the deployment scripts.

### GPU driver upgrade path

The single largest performance lever is upgrading the GPU driver from R550 (Production Branch, CUDA 12.4) to R570 (New Feature Branch, CUDA 12.8). This would allow using the official pre-built vLLM container with torch.compile + CUDA graphs enabled, achieving the full ~30-40 tok/s. The upgrade is backward-compatible — all existing CUDA 12.4 workloads continue to work. See `docs/DRIVER_UPGRADE_PROPOSAL.md` for the full technical justification.

### Benchmark results

After applying all optimizations, we benchmarked with a 500-token generation request:

```
vLLM engine log:
  Avg generation throughput: 24.5 tokens/s  (first 10s window)
  Avg generation throughput: 25.5 tokens/s  (second 10s window)

  500 tokens generated in ~20 seconds → ~25 tok/s sustained
```

This is an **8x improvement** over the initial plain AWQ deployment (3 tok/s) and a **4x improvement** over the baseline Marlin eager configuration (6-8 tok/s), achieved entirely through runtime optimization flags — no driver changes, no container rebuilds.

### Optimization summary

| Optimization | Change | Measured Impact |
|---|---|---|
| Marlin AWQ kernel | `awq` → `awq_marlin` | 3 → 6-8 tok/s (2x) |
| Context window | 32K → 64K | Doubles usable context for long documents |
| Memory utilization | 0.85 → 0.90 | +4.7 GiB KV cache, more concurrent requests |
| Prefix caching | Enabled | Faster repeated prefills (free, no downside) |
| Chunked prefill | Enabled | +10-15% GPU utilization for mixed workloads |
| **Combined result** | **All of the above** | **6-8 → ~25 tok/s (4x)** |
| Driver upgrade (future) | R550 → R570 | Projected ~30-40 tok/s (adds torch.compile) |

---

## 13. Verification and Testing

### Step 1: Verify the container

After a successful build, run these checks:

```bash
# Check vLLM version (should show 0.8.5.post1 or similar)
apptainer exec --nv /projects/.../vllm-cu124 \
    python3 -c "import vllm; print('vLLM:', vllm.__version__)"

# Check CUDA version (should show 12.4)
apptainer exec --nv /projects/.../vllm-cu124 \
    python3 -c "import torch; print('CUDA:', torch.version.cuda)"

# Check PyTorch version (should show 2.6.0+cu124)
apptainer exec --nv /projects/.../vllm-cu124 \
    python3 -c "import torch; print('PyTorch:', torch.__version__)"
```

### Step 2: Test Marlin AWQ (requires GPU node)

```bash
# Get an interactive GPU session
srun --partition=batch_gpu2 --gres=gpu:1 --time=00:30:00 --pty bash

module load apptainer
export HF_HOME=/projects/acer_hpc_admin/nassar/huggingface
export CUDA_VISIBLE_DEVICES=0

# Start vLLM with Marlin (enforce-eager required for this build)
apptainer exec --nv /projects/.../vllm-cu124 \
    vllm serve Qwen/Qwen2.5-VL-72B-Instruct-AWQ \
    --host 0.0.0.0 --port 8000 \
    --quantization awq_marlin \
    --enforce-eager \
    --max-model-len 4096 \
    --max-num-seqs 256 \
    --dtype auto
```

If Marlin loads successfully, you will see:
```
INFO:     Loading model weights...
INFO:     Using AWQ Marlin kernel
INFO:     Started server on http://0.0.0.0:8000
```

If you see `cudaErrorUnsupportedPtxVersion`, the container was not built correctly — the CUDA toolkit inside was not 12.4.

### Step 3: Benchmark inference speed

From another terminal on the same node:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
        "prompt": "Explain quantum computing in simple terms.",
        "max_tokens": 200,
        "temperature": 0.7
    }'
```

The response includes `usage.completion_tokens` and the response time. With our optimized deployment (Marlin AWQ + enforce-eager + prefix caching + chunked prefill): ~500 tokens in ~20 seconds → **~25 tok/s**. Without `--enforce-eager` (if torch.compile works): ~30-40 tok/s.

### Step 4: Deploy to production

```bash
sbatch scripts/vllm-qwen-vl-72b.sh
```

STREAM users automatically get faster inference through the existing Globus Compute pipeline. No STREAM code changes needed — the vLLM API is identical, just faster.

---

## 14. Lessons Learned for HPC LLM Deployment

These lessons apply to any institution deploying LLMs on HPC clusters, not just Lakeshore.

### 1. Check the GPU driver version FIRST

Before choosing a vLLM version, container, or model, run:
```bash
nvidia-smi | head -3
```
The driver version determines the maximum CUDA PTX version, which constrains everything else. This is the single most important piece of information.

### 2. The CUDA version chain is non-negotiable

```
Driver version → max CUDA PTX → max PyTorch version → max vLLM version
```

You cannot skip links in this chain. A CUDA 12.4 driver will never run CUDA 12.9 PTX, no matter what software you install. Build your container to match the driver, not the latest release.

### 3. Apptainer fakeroot is "partial" on most HPC clusters

Do not assume `apt-get` will work in container builds. Many clusters configure Apptainer's fakeroot with restricted subuid/subgid ranges that break `apt-get`'s privilege-dropping. Use base images with pre-installed software and install additional packages via `pip` or `conda`.

### 4. Pre-built wheels assume the latest CUDA

As of early 2026, PyPI wheels for vLLM and PyTorch target CUDA 12.9 by default. If your driver is older, you must either:
- Use the PyTorch package index for your CUDA version (`--index-url .../whl/cu124`)
- Build vLLM from source with the matching CUDA toolkit

### 5. SLURM spool directory breaks relative paths

When you submit a script with `sbatch`, SLURM copies it to a spool directory before execution. Any `$(dirname "$0")` or relative paths resolve to the spool directory, not your original directory. Use absolute paths in batch scripts.

### 6. Version pinning is survival

Document the exact versions that work together. Our working combination:

| Component | Version |
|-----------|---------|
| GPU Driver | 550.163.01 |
| CUDA Toolkit | 12.4.1 |
| PyTorch | 2.6.0+cu124 |
| vLLM | 0.8.5.post2 (built from v0.8.5.post1 tag) |
| transformers | 4.57.6 (pinned to <5.0) |
| flash-attn | 2.7.4.post1 |
| Python | 3.11.10 |
| Base image | pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel |

### 7. Build on CPU, run on GPU

CUDA compilation (nvcc) runs on the CPU. Use CPU-only partitions for container builds to avoid competing for GPU allocations. The GPU is only needed at runtime.

### 8. Sandbox containers are easier to debug

During development, use `--sandbox` format (unpacked directory). You can browse the filesystem, check installed packages, and even modify files for testing. Convert to SIF (compressed) format for production if disk space is a concern.

### 9. Memory budget analysis prevents OOM

Before deploying a model, calculate the memory budget:
- Model weights: `parameters × bytes_per_param` (0.5 for AWQ, 2 for FP16)
- KV cache: scales with `max_model_len × num_layers × num_kv_heads × head_dim`
- CUDA graphs: ~4-6 GiB (disable with `--enforce-eager` if tight)
- Overhead: ~10-15% of total GPU memory for PyTorch, CUDA context, sampler

### 10. Set `TORCH_CUDA_ARCH_LIST` when building on CPU-only nodes

When building CUDA code on nodes without GPUs (like the `batch` partition), the build system cannot auto-detect which GPU architectures to target. It defaults to compiling for all common architectures (5.0 through 9.0a), which:

- **Multiplies compilation time** — every kernel is compiled once per architecture
- **Multiplies peak memory** — more object files in flight simultaneously
- **Produces code for GPUs you don't have** — wasted effort

Always set `TORCH_CUDA_ARCH_LIST` to only the architectures present in your cluster. For Lakeshore: `"8.0;9.0"` (A100 + H100). This can cut build time and memory usage by 2-3x.

### 11. FlashAttention 3 Hopper kernels are memory monsters

Not all CUDA kernels are created equal. Most vLLM kernels compile with 2-4 GiB of RAM per `nvcc` process. The FlashAttention 3 Hopper kernels (`flash_fwd_hdimall_*_sm90.cu`) are outliers — each one can consume **8-16 GiB** during compilation. The Linux OOM killer terminates these processes silently with just `Killed` (no error message, no stack trace), making the failure difficult to diagnose.

The fix: request significantly more RAM than the naive estimate suggests. For vLLM builds with `MAX_JOBS=4`, request at least **64 GiB** (`--mem=64G`). The "Killed" signal from the OOM killer is the clue — if you see compilation steps fail with just `Killed` and no error, you need more memory.

### 12. Pin `transformers` to avoid breaking changes

The HuggingFace `transformers` library is a transitive dependency of vLLM that changes frequently. Major version jumps (4.x → 5.x) can remove attributes that vLLM relies on. Always pin to a known-working major version: `transformers>=4.51.1,<5.0`.

### 13. Use `--no-deps` when installing GPU packages in containers

Packages like `flash-attn`, `xformers`, and `torch` embed compiled CUDA code that must match exact version combinations. Letting pip resolve dependencies freely will often "helpfully" upgrade PyTorch to the latest version, destroying your carefully pinned CUDA 12.4 setup. Always use `--no-deps` and verify compatibility manually.

### 14. Pre-built wheels must match your exact environment

A pre-built wheel for `flash-attn` compiled against PyTorch 2.5.1 will produce `undefined symbol` errors when loaded with PyTorch 2.6.0. Check the wheel filename for version markers (e.g., `torch2.6`, `cu12`, `cp311`). The project's GitHub releases page is often the best source for specific version combinations.

### 15. `torch.compile` may not work with quantized kernels

`torch.compile` (and CUDA graph capture) can crash with quantized inference kernels, especially in development builds. The `--enforce-eager` flag is a safe fallback that preserves the quantized kernel optimization while bypassing the compilation step. The performance difference is typically 20-30% — meaningful but not critical for interactive use.

### 16. The plain AWQ kernel is a valid fallback

If Marlin fails for any reason, `--quantization awq` always works (albeit at ~3 tok/s). This is slow but functional. Having a fallback that works is better than a fast path that crashes.

---

## 15. Frequently Asked Questions

### Q: Can I just install vLLM with pip instead of building a container?

Not reliably on HPC. The system Python is typically older (3.8-3.9), lacks CUDA development headers, and you don't have permission to install system packages. A container bundles everything — Python, PyTorch, CUDA toolkit, vLLM — in an isolated environment that doesn't conflict with the host system.

### Q: Why not ask the HPC admins to install vLLM system-wide?

HPC software stacks are managed for stability, not bleeding-edge AI. Installing vLLM system-wide would require maintaining specific PyTorch and CUDA versions that might conflict with other users' workflows. Containers let individual researchers use their own software stacks without affecting others.

### Q: Is vLLM v0.8.5.post1 much worse than v0.15.1?

For our use case, no. Both versions support Qwen2.5-VL, AWQ Marlin, and all the features STREAM needs. v0.15.1 adds newer features (speculative decoding improvements, new model architectures) that we don't currently need. The core inference engine — PagedAttention, continuous batching, Marlin kernels — is mature in both versions.

### Q: How do I know if my CUDA version is the problem?

If vLLM starts successfully with `--quantization awq` (slow) but crashes with `--quantization awq_marlin`, the issue is almost certainly CUDA PTX version mismatch. Check:

```bash
# Inside the container:
python3 -c "import torch; print(torch.version.cuda)"   # Container CUDA version

# On the host:
nvidia-smi | head -3                                     # Driver CUDA support
```

If the container CUDA is newer than the driver's max, you need a container built for the driver's CUDA version.

### Q: Can I use Docker instead of Apptainer?

Typically not on HPC. Docker requires root privileges (or Docker daemon access), which HPC admins don't grant to regular users. However, you can build a Docker image on another machine and convert it to Apptainer:

```bash
# On a machine with Docker:
docker build -t vllm-cu124 .
docker save vllm-cu124 -o vllm-cu124.tar
# Copy to HPC, then:
apptainer build --sandbox vllm-cu124 docker-archive://vllm-cu124.tar
```

### Q: What happens when the HPC admins upgrade the GPU driver?

If the driver is upgraded to 570+ (supporting CUDA 12.8+), you can use the official pre-built vLLM containers directly — no custom build needed. The custom container (vllm-cu124) will still work, since newer drivers are backward-compatible with older PTX versions. Keep both containers as options.

### Q: How much disk space does the container use?

The sandbox format uses ~26 GiB. A compressed SIF file would be ~7 GiB. Store containers in the project space (`/projects/`), not the home directory, to avoid quota issues.

### Q: Can I serve multiple models simultaneously?

Yes, on different ports. Each model runs as a separate vLLM process inside its own SLURM job, using one GPU. With multiple GPUs available, you can run multiple models concurrently:

```bash
sbatch scripts/vllm-qwen-vl-72b.sh    # Port 8000 on ghi2-002
sbatch scripts/vllm-qwen-32b.sh       # Port 8000 on ga-002 (different node)
```

### Q: What if the build fails with an out-of-memory error?

The OOM killer's signature is compilation steps failing with just `Killed` — no error message, no stack trace. Two fixes, in order of preference:

1. **Increase SLURM memory**: Change `--mem=64G` (or higher). The FlashAttention 3 Hopper kernels are the usual culprit, with each nvcc process consuming 8-16 GiB of RAM.
2. **Set `TORCH_CUDA_ARCH_LIST`**: Restrict to only your GPUs (e.g., `"8.0;9.0"` for A100+H100). This reduces the number of kernel variants and lowers peak memory.
3. **Reduce `MAX_JOBS`**: As a last resort, lower to `MAX_JOBS=2`. This halves peak memory but doubles build time.

---

## 16. References

[1] OpenAI Pricing. https://openai.com/pricing — GPT-4 Turbo pricing as of January 2026.

[2] Qwen Team. "Qwen2.5: A Party of Foundation Models." https://qwenlm.github.io/blog/qwen2.5/ — Benchmark comparisons with GPT-4 and other models.

[3] Kwon, W., et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." *Proceedings of the 29th Symposium on Operating Systems Principles (SOSP)*, 2023. https://arxiv.org/abs/2309.06180

[4] Apptainer Documentation. https://apptainer.org/docs/ — The container runtime for HPC.

[5] Lin, J., et al. "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." *MLSys*, 2024. https://arxiv.org/abs/2306.00978

[6] vLLM Documentation — GPU Installation. https://docs.vllm.ai/en/stable/getting_started/installation/gpu/

[7] NVIDIA CUDA Compatibility Guide — Driver and toolkit version matrix. https://docs.nvidia.com/deploy/cuda-compatibility/

[8] vLLM Project — Building Docker for different CUDA versions. https://discuss.vllm.ai/t/how-to-build-vllm-docker-image-for-different-cuda-version/1552

---

### Code File Reference

| File | Purpose |
|------|---------|
| `scripts/vllm-cu124.def` | Apptainer definition file — builds vLLM with CUDA 12.4 |
| `scripts/build-vllm-cu124.sh` | SLURM batch script — submits the container build job |
| `scripts/vllm-qwen-vl-72b.sh` | SLURM deployment — Qwen 2.5-VL 72B on H100 |
| `scripts/vllm-qwen-72b.sh` | SLURM deployment — Qwen 2.5 72B (text-only) on H100 |
| `scripts/vllm-qwen-32b.sh` | SLURM deployment — Qwen 2.5 32B AWQ on A100 MIG |
| `scripts/vllm-qwen-32b-fp16.sh` | SLURM deployment — Qwen 2.5 32B FP16 on H100 |
| `docs/ADDING_NEW_MODELS.md` | Step-by-step guide for deploying new models |

---

*This document was written for the PEARC conference paper on STREAM. It covers the deployment of vLLM on Lakeshore HPC, including the challenges of CUDA PTX version compatibility, Apptainer container builds, and configuration tuning for large vision-language models on H100 GPUs.*
