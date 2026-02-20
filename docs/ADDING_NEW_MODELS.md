# Adding a New Model to STREAM

This guide walks through the complete end-to-end process of adding a new LLM to STREAM, from choosing a model on HuggingFace to seeing it in the frontend dropdown. Every step includes the reasoning behind each decision so you can adapt the process to any model.

---

## Overview

STREAM's architecture has three layers that a new model must be registered in:

1. **Lakeshore HPC (GPU inference)** -- Models run as vLLM servers inside Apptainer containers on SLURM-managed GPU nodes. Each model is a separate process on its own port.
2. **Middleware (routing and config)** -- A Python backend that knows which models exist, where they run (host:port), and their context limits. It routes user requests to the correct vLLM instance via Globus Compute.
3. **Frontend (user interface)** -- A React app that renders model selection dropdowns. The model must be added to TypeScript types and UI configuration objects so users can select it.

The data flow for a Lakeshore request:

```
Browser  -->  Middleware  -->  Globus Compute  -->  vLLM on Lakeshore GPU  -->  response back
```

Adding a model means: (1) getting it running on a GPU, then (2) telling every layer about it.

---

## Step 1: Choose Your Model

### Where to find models

Browse [HuggingFace Hub](https://huggingface.co/models) filtered by:
- **Task**: Text Generation
- **Library**: Transformers or vLLM
- **Sort by**: Trending or Most Downloads

Look for models with `-Instruct` or `-Chat` in the name -- these are fine-tuned for conversation. Base models (without the suffix) are for completion, not chat.

### Estimating VRAM requirements

The main question is: **will it fit on the GPU?** Model weights dominate memory usage.

| Precision | Formula | Example (32B params) |
|-----------|---------|---------------------|
| FP16 / BF16 (full precision) | params x 2 bytes | 32B x 2 = **64 GiB** |
| AWQ 4-bit quantization | params x 0.5 bytes | 32B x 0.5 = **16 GiB** |

These are weight-only estimates. Total VRAM usage includes KV cache (scales with context length), CUDA graphs (~4-6 GiB if enabled), and PyTorch overhead (~2-4 GiB). A safe rule of thumb: **weights should be no more than 60-70% of usable VRAM** to leave room for everything else.

### Matching model to hardware

Lakeshore has two classes of GPU:

| Hardware | Node(s) | Partition | Usable VRAM | Max weight size | GRES flag |
|----------|---------|-----------|-------------|-----------------|-----------|
| A100 MIG 3g.40gb | ga-001, ga-002 | `batch_gpu` | ~39.5 GiB | ~18 GiB (with KV cache + overhead) | `gpu:3g.40gb:1` |
| H100 NVL (full) | ghi2-002 | `batch_gpu2` | ~93.2 GiB | ~64 GiB (with KV cache + overhead) | `gpu:1` |

**Decision tree:**

```
Model params <= 7B
  --> FP16 fits on A100 MIG (7B x 2 = 14 GiB). No quantization needed.

Model params 13B-32B
  --> AWQ 4-bit on A100 MIG (32B x 0.5 = 16 GiB). Needs --enforce-eager.
  --> OR FP16 on H100 (32B x 2 = 64 GiB). No quantization needed.

Model params 70B-72B
  --> AWQ 4-bit on H100 (72B x 0.5 = 36 GiB). Fits with 32K context.
  --> FP16 will NOT fit on a single H100 (72B x 2 = 144 GiB > 93 GiB).

Model params > 100B
  --> Needs multi-GPU tensor parallelism (not currently set up on Lakeshore).
```

### Quantization considerations

**AWQ 4-bit** reduces model size 4x by compressing weights from FP16 (2 bytes) to INT4 (0.5 bytes) while preserving quality for "salient" weights. However:

- **Use `awq_marlin`, not `awq`**. The plain AWQ kernel is ~10x slower (~3 tok/s vs ~30+ tok/s). Marlin kernels are optimized for NVIDIA tensor cores and deliver dramatically better throughput.
- Marlin requires CUDA 12.4+ drivers. The H100 on ghi2-002 has CUDA 12.2 (driver 535), which causes PTX compilation errors with Marlin. If you hit this, either wait for a driver update or **use FP16 instead** (no quantization kernels needed, just standard cuBLAS).
- AWQ models are published separately on HuggingFace (e.g., `Qwen/Qwen2.5-32B-Instruct-AWQ`). Not every model has an official AWQ variant -- community quantizations exist (e.g., `casperhansen/DeepSeek-R1-Distill-Qwen-32B-AWQ`).

**FP16** avoids all kernel compatibility issues. It uses standard cuBLAS GEMM (matrix multiply) which works everywhere. The tradeoff is 4x more VRAM for weights. Use FP16 when: the GPU has enough VRAM, or when AWQ kernels cause problems.

---

## Step 2: Download the Model on Lakeshore

### Storage location

Use the ACER project space for all model weights:

```
/projects/acer_hpc_admin/nassar/huggingface/
```

**Why this directory?**
- **10 TiB quota** (shared across team) -- enough for many large models
- **Persistent** -- not purged, unlike `/scratch`
- **Accessible from all compute nodes** via GPFS parallel filesystem
- The home directory (`~`) has only a ~100 GiB quota -- a single 72B AWQ model is 39 GiB

**Never download to:**
- `~/` or `~/.cache/huggingface/` -- will blow the home quota
- `/scratch/` -- periodically purged by sysadmins, your model will disappear

### Method 1: Download inside the container on a compute node (preferred)

This uses the exact Python environment that vLLM will use, avoiding version mismatches.

```bash
# Get an interactive session on a compute node
srun --partition=batch --account=ts_acer_chi --time=02:00:00 --pty bash

# Load the container runtime
module load apptainer

# Point HuggingFace cache to project space
export HF_HOME=/projects/acer_hpc_admin/nassar/huggingface

# Download using the CLI tool inside the container
apptainer exec /projects/acer_hpc_admin/nassar/containers/vllm-0.15.1 \
    huggingface-cli download ORGANIZATION/MODEL-NAME
```

Replace `ORGANIZATION/MODEL-NAME` with the actual HuggingFace ID (e.g., `Qwen/Qwen2.5-32B-Instruct-AWQ`).

### Method 2: Download from the login node using Python

If you cannot get a compute node, you can download directly on the login node. This does not require a GPU -- it is just a file download.

```bash
# Set up Python paths for the login node
export PYTHONUSERBASE=/projects/acer_hpc_admin/nassar/python
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.9/site-packages
export HF_HOME=/projects/acer_hpc_admin/nassar/huggingface

# Download using the HuggingFace Hub Python library
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('ORGANIZATION/MODEL-NAME')
"
```

### Tips for large downloads

- **Use `screen`** so the download survives SSH disconnects:
  ```bash
  screen -S download_modelname
  # ... run the download command ...
  # Ctrl+A, D to detach. screen -r download_modelname to reattach.
  ```
- Downloads can take 30-60 minutes for large models (36-64 GiB over network).
- HuggingFace downloads are resumable. If interrupted, re-run the same command.

### Verify the download

```bash
ls /projects/acer_hpc_admin/nassar/huggingface/hub/
```

You should see a directory like `models--Qwen--Qwen2.5-32B-Instruct-AWQ/` containing `blobs/`, `refs/`, and `snapshots/` subdirectories.

---

## Step 3: Create the SBATCH Script

Each model needs a SLURM batch script that starts the vLLM server. Copy an existing script from `scripts/` and modify it.

### Which template to copy

| Your model | Copy this template | Why |
|------------|-------------------|-----|
| Small model (<=7B) on A100 MIG | `scripts/vllm-qwen-1.5b.sh` | Minimal flags, no quantization, no enforce-eager |
| 32B AWQ on A100 MIG | `scripts/vllm-qwen-32b.sh` | AWQ quantization, enforce-eager, 16K context |
| 32B FP16 on H100 | `scripts/vllm-qwen-32b-fp16.sh` | No quantization, full precision, 32K context |
| 70B+ AWQ on H100 | `scripts/vllm-qwen-72b.sh` | AWQ Marlin kernels, max-num-seqs tuning |

### Example: A100 MIG script for a 32B AWQ model

```bash
#!/bin/bash
#SBATCH --job-name=stream-your-model
#SBATCH --partition=batch_gpu
#SBATCH --nodelist=ga-002
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.log

MODEL="ORGANIZATION/MODEL-NAME-AWQ"
PORT=8005    # <-- Pick an unused port (see port assignments below)

echo "=========================================="
echo "STREAM vLLM: ${MODEL}"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES | Port: $PORT"
echo "Started: $(date)"
echo "=========================================="

module load apptainer

CONTAINER="/projects/acer_hpc_admin/nassar/containers/vllm-0.15.1"
export CUDA_VISIBLE_DEVICES=0
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
```

### Key parameters explained

#### SBATCH headers

| Parameter | A100 MIG | H100 NVL | What it does |
|-----------|----------|----------|--------------|
| `--partition` | `batch_gpu` | `batch_gpu2` | SLURM partition (determines which nodes are eligible) |
| `--nodelist` | `ga-001` or `ga-002` | `ghi2-002` | Specific node to run on |
| `--gres` | `gpu:3g.40gb:1` | `gpu:1` | GPU resource. MIG slices use the `3g.40gb` type; H100 uses a full GPU |

#### Container

All scripts use the same vLLM 0.15.1 container stored as a sandbox (unpacked directory):

```
CONTAINER="/projects/acer_hpc_admin/nassar/containers/vllm-0.15.1"
```

This is a "sandbox" format (unpacked directory tree) rather than a compressed SIF file. Both formats work identically with `apptainer exec`, but sandboxes are faster to build and avoid the slow SquashFS compression step. The tradeoff is more disk usage (~26 GiB vs ~7 GiB), which is fine given the 10 TiB project space quota.

#### Environment variables (always set these)

```bash
export CUDA_VISIBLE_DEVICES=0    # Use the first (and only) allocated GPU
export HF_HOME=/projects/acer_hpc_admin/nassar/huggingface  # Model weight cache
```

Without `HF_HOME`, vLLM looks in `~/.cache/huggingface/` and will not find your downloaded model.

#### vLLM serve flags

| Flag | What it does | How to choose a value |
|------|--------------|----------------------|
| `--tensor-parallel-size 1` | Number of GPUs to split model across | Always 1 (single GPU per model on Lakeshore) |
| `--max-model-len N` | Maximum tokens (input + output) per request | Directly controls KV cache VRAM. 32768 for small models or H100. 16384 for 32B AWQ on A100 MIG (KV cache is tight). 8192 if still OOM. |
| `--gpu-memory-utilization F` | Fraction of GPU memory vLLM pre-allocates (0.0-1.0) | 0.90 for most cases. 0.85 if OOM during sampler warmup. 0.75 if you want CUDA graphs on a tight MIG slice. |
| `--quantization TYPE` | Quantization kernel to use | `awq_marlin` for AWQ on vLLM >=0.15 with CUDA 12.4+. `awq` for AWQ on older vLLM/drivers. Omit entirely for FP16 models. |
| `--enforce-eager` | Disable CUDA graph capture | Needed on 40GB MIG slices -- CUDA graph capture uses ~4-6 GiB that causes OOM. Costs ~10-15% inference speed. Not needed on H100 (plenty of VRAM). |
| `--dtype auto` | Let vLLM pick data type from model config | Always use `auto`. |
| `--max-num-seqs N` | Maximum concurrent sequences in a batch | Default 1024 is fine for most models. Reduce to 256 for very large vocab models (e.g., Qwen 72B with 152K vocab) to avoid OOM during sampler warmup. |

#### Port assignments

Each model needs a unique port on its node. Current assignments:

| Port | Model | Node |
|------|-------|------|
| 8000 | Qwen 2.5 general purpose | ga-002 (1.5B) or ghi2-002 (32B FP16 / 72B AWQ) |
| 8001 | Qwen 2.5 Coder | ga-002 |
| 8002 | DeepSeek R1 (reasoning) | ga-002 |
| 8003 | QwQ (reasoning) | ga-002 |
| 8004 | Qwen 2.5 32B AWQ | ga-002 |

Pick the next available port (e.g., 8005) or reuse a port on a different node.

---

## Step 4: Update Backend Configuration

Three files need changes. All three must agree on the model's identifier string (e.g., `lakeshore-your-model`).

### 4a. Middleware Config

**File:** `stream/middleware/config.py`

#### Add to `LAKESHORE_MODELS`

This dict tells the middleware where each model's vLLM instance is running. The proxy uses this to route requests to the correct host:port on Lakeshore.

```python
LAKESHORE_MODELS = {
    # ... existing models ...

    # --- Your new model ---
    "lakeshore-your-model": {
        "hf_name": "ORGANIZATION/MODEL-NAME",  # Must match MODEL= in SBATCH script
        "port": 8005,                            # Must match PORT= in SBATCH script
        "description": "Your model description",
    },
}
```

If the model runs on a **different node** than the default (ga-002), add a `"host"` field:

```python
    "lakeshore-your-model": {
        "hf_name": "ORGANIZATION/MODEL-NAME",
        "host": "ghi2-002",       # Override default host from VLLM_SERVER_URL
        "port": 8000,
        "description": "Your model on H100",
    },
```

The `hf_name` field **must exactly match** the model name passed to `vllm serve` in the SBATCH script, because vLLM's OpenAI-compatible API uses this as the model identifier in chat completion requests.

#### Add to `MODEL_CONTEXT_LIMITS`

This tells the middleware how many tokens the model can handle, so it can truncate long conversations before sending them to the GPU.

```python
MODEL_CONTEXT_LIMITS = {
    # ... existing models ...

    "lakeshore-your-model": {"total": 16384, "reserve_output": 2048},
}
```

- `total`: Must match `--max-model-len` in the SBATCH script
- `reserve_output`: Tokens reserved for the model's response. 2048 (~1 page) is a good default. Use 1024 for shorter-context models (8K total).

### 4b. LiteLLM Gateway

**File:** `stream/gateway/litellm_config.yaml`

LiteLLM is the gateway that normalizes different model APIs into a single OpenAI-compatible interface. Even though Lakeshore models go through the proxy (not directly through LiteLLM in the current architecture), they are registered here for cost tracking and timeout configuration.

#### Add model entry under `model_list`

```yaml
model_list:
  # ... existing models ...

  # Your new model
  - model_name: lakeshore-your-model
    litellm_params:
      model: openai/ORGANIZATION/MODEL-NAME    # "openai/" prefix + HuggingFace ID
      api_base: http://lakeshore-proxy:8001/v1  # Always the proxy URL
      api_key: dummy                             # vLLM doesn't need auth
    model_info:
      input_cost_per_token: 0.0                  # Free (university GPU)
      output_cost_per_token: 0.0
```

The `model` field uses the `openai/` prefix because vLLM exposes an OpenAI-compatible API, and LiteLLM needs this prefix to know which client library to use.

#### Add timeout under `router_settings`

```yaml
router_settings:
  model_timeout:
    # ... existing models ...
    lakeshore-your-model: 120    # Seconds. Use 180 for very large models.
```

### 4c. Query Router

**File:** `stream/middleware/core/query_router.py`

The `get_model_for_tier()` function maps the user's tier + model selection to an actual model name. **You usually do not need to change this file.** It works generically:

```python
def get_model_for_tier(tier, cloud_provider=None, local_model=None, lakeshore_model=None):
    if tier == "lakeshore" and lakeshore_model:
        return lakeshore_model  # Returns whatever the frontend sends
    return DEFAULT_MODELS.get(tier, DEFAULT_MODELS["local"])
```

The router just passes through the model name selected in the frontend. As long as that name matches a key in `LAKESHORE_MODELS`, the proxy will find it.

**When you DO need to change this:** If your model should be the new default for the Lakeshore tier, update `DEFAULT_MODELS` in `config.py`:

```python
DEFAULT_MODELS = {
    "local": "local-llama",
    "lakeshore": "lakeshore-your-model",  # Change default here
    "cloud": DEFAULT_CLOUD_PROVIDER,
}
```

---

## Step 5: Update Frontend

Two files need changes so the model appears in the UI dropdown.

### 5a. TypeScript Types

**File:** `frontends/react/src/types/settings.ts`

Add your model ID to the `LakeshoreModel` union type. This gives compile-time safety -- TypeScript will catch typos.

```typescript
export type LakeshoreModel =
  | 'lakeshore-qwen-1.5b'
  | 'lakeshore-qwen-32b-fp16'
  | 'lakeshore-qwen-72b'
  | 'lakeshore-qwen-32b'
  | 'lakeshore-coder-1.5b'
  | 'lakeshore-deepseek-r1'
  | 'lakeshore-qwq'
  | 'lakeshore-your-model'      // <-- Add here
```

### 5b. Settings Panel

**File:** `frontends/react/src/components/sidebar/SettingsPanel.tsx`

Add your model to the `LAKESHORE_MODEL_CONFIG` object. This controls the label and description shown in the model selection dropdown.

```typescript
const LAKESHORE_MODEL_CONFIG: Record<LakeshoreModel, { label: string; description: string }> = {
  // ... existing models ...

  'lakeshore-your-model': {
    label: 'Your Model Name',           // Human-readable name
    description: 'Brief description',    // Shown as subtitle in dropdown
  },
}
```

The key (`'lakeshore-your-model'`) must match:
- The key in `LAKESHORE_MODELS` (config.py)
- The `model_name` in `litellm_config.yaml`
- The union member in `settings.ts`

All four locations must use the **exact same string**.

---

## Step 6: Test

### 1. Submit the SBATCH job

```bash
sbatch scripts/vllm-your-model.sh
```

### 2. Check SLURM status

```bash
squeue -u nassar    # See if the job is running
```

### 3. Watch the logs

```bash
tail -f logs/stream-your-model-*.log
```

Look for the line: `INFO: Uvicorn running on http://0.0.0.0:PORT` -- that means vLLM started successfully.

Common startup messages to watch for:
- `Loading model weights...` -- downloading or loading from cache
- `CUDA out of memory` -- model is too big (see Troubleshooting)
- `torch.cuda.OutOfMemoryError` during sampler warmup -- reduce `--max-num-seqs` or `--gpu-memory-utilization`

### 4. Test vLLM directly from Lakeshore

SSH into the node and curl the health endpoint:

```bash
curl http://NODE_IP:PORT/v1/models
```

This should return JSON listing your model's HuggingFace ID. If it does, vLLM is serving correctly.

### 5. Test through STREAM

1. Start the STREAM application (middleware + frontend)
2. Select the Lakeshore tier in the sidebar
3. Open the Lakeshore Models dropdown and select your new model
4. Send a test message
5. Check that the response comes back with the correct model metadata

---

## Quick Reference: Storage Layout

```
/projects/acer_hpc_admin/nassar/
├── containers/
│   └── vllm-0.15.1/                          # Sandbox (~26 GiB), used by all scripts
├── huggingface/
│   └── hub/
│       ├── models--Qwen--Qwen2.5-1.5B-Instruct/
│       ├── models--Qwen--Qwen2.5-32B-Instruct/         # ~64 GiB (FP16)
│       ├── models--Qwen--Qwen2.5-32B-Instruct-AWQ/     # ~18 GiB
│       ├── models--Qwen--Qwen2.5-72B-Instruct-AWQ/     # ~39 GiB
│       └── models--ORGANIZATION--MODEL-NAME/            # Your new model
└── python/                                    # pip packages for login node use
```

---

## Quick Reference: Checking Quotas

```bash
# Per-user home directory quota (~100 GiB)
mmlsquota -u nassar mmfs1

# Project space quota (10 TiB shared)
mmlsquota -j acer_hpc_admin mmfs1
```

If `mmlsquota` shows you are near the limit, clean up old model downloads you no longer need from the HuggingFace cache.

---

## Quick Reference: Checklist

When adding a new model, make sure you have updated **all** of these:

- [ ] Model weights downloaded to `/projects/acer_hpc_admin/nassar/huggingface/`
- [ ] SBATCH script created in `scripts/vllm-your-model.sh`
- [ ] `LAKESHORE_MODELS` dict in `stream/middleware/config.py`
- [ ] `MODEL_CONTEXT_LIMITS` dict in `stream/middleware/config.py`
- [ ] `model_list` in `stream/gateway/litellm_config.yaml`
- [ ] `model_timeout` in `stream/gateway/litellm_config.yaml`
- [ ] `LakeshoreModel` type in `frontends/react/src/types/settings.ts`
- [ ] `LAKESHORE_MODEL_CONFIG` in `frontends/react/src/components/sidebar/SettingsPanel.tsx`
- [ ] Model string is identical across all four locations

---

## Troubleshooting

### OOM on startup

**Symptom:** `torch.cuda.OutOfMemoryError` in the logs during model loading or CUDA graph capture.

**Fixes (try in order):**
1. Add `--enforce-eager` -- skips CUDA graph capture, saves ~4-6 GiB
2. Reduce `--max-model-len` (e.g., 16384 to 8192) -- halves KV cache memory
3. Lower `--gpu-memory-utilization` (e.g., 0.90 to 0.85) -- leaves more room for overhead
4. Reduce `--max-num-seqs` (e.g., 1024 to 256) -- reduces sampler warmup memory

### Slow generation (~3 tok/s when you expect 30+)

**Cause:** Using the plain `awq` kernel instead of `awq_marlin`. The plain kernel does not use tensor cores and is roughly 10x slower.

**Fix:** Change `--quantization awq` to `--quantization awq_marlin` in the SBATCH script. This requires vLLM >= 0.15 (the sandbox container) and CUDA 12.4+ drivers. If Marlin crashes with a CUDA PTX error, the driver is too old -- use FP16 instead (no quantization flag).

### Disk quota exceeded

**Symptom:** Download fails with `Disk quota exceeded` or `OSError: [Errno 122]`.

**Cause:** Downloading to the home directory instead of project space.

**Fix:** Make sure `HF_HOME=/projects/acer_hpc_admin/nassar/huggingface` is exported before downloading and in the SBATCH script.

### Model not found by vLLM

**Symptom:** vLLM tries to download the model at startup instead of using the cached copy, or errors with `Model not found`.

**Cause:** `HF_HOME` is not set in the SBATCH script, so vLLM looks in `~/.cache/huggingface/` instead of the project space where you downloaded it.

**Fix:** Add `export HF_HOME=/projects/acer_hpc_admin/nassar/huggingface` to the SBATCH script before the `apptainer exec` command.

### Model appears in UI but requests fail

**Checklist:**
1. Is the SLURM job running? (`squeue -u nassar`)
2. Does the `hf_name` in `config.py` exactly match the `MODEL=` in the SBATCH script?
3. Does the `port` in `config.py` match the `PORT=` in the SBATCH script?
4. If the model runs on a non-default node, does `config.py` have a `"host"` field?
5. Is Globus Compute authenticated? (Check the Lakeshore auth panel in the sidebar)

### vLLM starts but model name mismatch

**Symptom:** vLLM is running, `curl /v1/models` works, but STREAM cannot inference.

**Cause:** The model name in the `curl` response (which is the HuggingFace ID) does not match the `hf_name` in `config.py`. vLLM uses the HuggingFace model ID as the model name in its OpenAI-compatible API. If you pass a different name in the chat completion request, vLLM returns a 404.

**Fix:** Make `hf_name` match exactly what vLLM reports in `/v1/models`.
