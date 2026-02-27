# GPU Driver Upgrade Proposal: R550 → R570 on ghi2-002

## Summary

Upgrading the NVIDIA driver on ghi2-002 from **R550** (Production Branch, CUDA 12.4) to **R570** (New Feature Branch, CUDA 12.8) would unlock **full torch.compile optimization** for STREAM's AI inference service — pushing throughput from ~25 tok/s to ~30-40 tok/s and eliminating the need for custom container builds.

This upgrade is backward-compatible and low-risk. No existing user workloads would break.

---

## The Problem

STREAM serves a 72-billion-parameter AI model (Qwen 2.5-VL-72B) on ghi2-002's H100 GPU. The vLLM inference engine uses two key optimizations:

1. **Marlin AWQ kernels** — specialized GPU programs for 4-bit quantized inference (~10x speedup)
2. **torch.compile** — PyTorch's graph optimizer that fuses operations (~3-5x additional speedup)

With driver R550, we can use Marlin (optimization 1) but **not** torch.compile (optimization 2). This is because:

- Our custom container (vllm-cu124) was built from source with CUDA 12.4 to match driver 550
- The torch.compile + Triton JIT combination in this build crashes during autotuning
- We must use `--enforce-eager` to disable torch.compile

Through runtime optimizations (prefix caching, chunked prefill, memory tuning), we've pushed throughput to **~25 tok/s** — but torch.compile would add the final 20-50% to reach ~30-40 tok/s.

### Performance comparison

| Configuration | Throughput | Status |
|---|---|---|
| Driver R550 + plain AWQ | ~3 tok/s | Previous deployment |
| Driver R550 + Marlin AWQ + enforce-eager (baseline) | ~6-8 tok/s | Initial Marlin deployment |
| Driver R550 + Marlin AWQ + enforce-eager + optimizations | **~25 tok/s** | **Current deployment** |
| Driver R570 + official vLLM + Marlin AWQ + torch.compile | **~30-40 tok/s** | **Projected with upgrade** |

We've closed most of the gap through runtime optimizations, but the driver upgrade would add the final ~20-50% and, critically, eliminate the need for custom container builds.

---

## Why R570?

### NVIDIA driver branches

NVIDIA maintains two parallel driver release tracks:

| Branch | Type | Latest Version | CUDA PTX Support |
|---|---|---|---|
| **R550** | Production Branch (Long-Lived) | 550.163.01 | CUDA 12.4 |
| **R570** | New Feature Branch | 570.x | CUDA 12.8 |

The **Production Branch** receives security patches and bug fixes but no new CUDA features. The **New Feature Branch** adds support for newer CUDA toolkit versions.

### Why R570 is safe

1. **Backward-compatible**: All CUDA 12.4 (and older) workloads continue to work on R570. Newer drivers can JIT-compile all older PTX versions. No existing user code would break.

2. **Production-ready**: R570 is used in production at AWS, GCP, Azure, and major AI companies. It's not bleeding-edge.

3. **Tested with H100**: R570 is certified for H100 NVL GPUs.

4. **Single-node scope**: The upgrade only needs to happen on ghi2-002 (the H100 node). Other nodes can remain on R550 if desired, providing an isolated test.

---

## What Changes on Our End

If the driver is upgraded, we would:

1. **Switch to the official pre-built vLLM container** (no more custom builds):
   ```bash
   apptainer build --sandbox vllm-latest docker://vllm/vllm-openai:v0.15.1
   ```

2. **Remove `--enforce-eager`** from deployment scripts:
   ```bash
   # Before (R550):
   vllm serve ... --quantization awq_marlin --enforce-eager

   # After (R570):
   vllm serve ... --quantization awq_marlin
   ```

3. **No other changes needed**: The vLLM API is identical. STREAM's middleware, routing, and frontend work without modification.

---

## Impact on Users

### STREAM users (campus AI service)
- **Faster responses**: A 200-token response drops from ~8 seconds to ~5 seconds
- **Better streaming experience**: Tokens appear faster in the chat interface
- **Same model quality**: The Qwen 2.5-VL-72B model is unchanged — only inference speed improves

### Other Lakeshore users
- **No impact**: CUDA backward compatibility ensures all existing workloads (CUDA 11.x, 12.0-12.4) continue to run without modification
- **New capability**: Users with newer CUDA code (12.5-12.8) would also benefit from being able to run their workloads

---

## Request

Could we upgrade the NVIDIA driver on **ghi2-002 only** from R550 (550.163.01) to R570 (latest available for H100 NVL)?

This would involve:
1. Download the R570 driver from https://www.nvidia.com/Download/index.aspx
   - Product Type: Data Center / Tesla
   - Product: H100
   - Operating System: (matching ghi2-002's OS)
   - **CUDA Toolkit**: select the New Feature Branch (R570)
2. Schedule a maintenance window for ghi2-002
3. Install the new driver (standard `nvidia-installer` process)
4. Verify with `nvidia-smi` that the driver is 570.x and CUDA shows 12.8

If testing on ghi2-002 goes well, the same upgrade could be applied to other GPU nodes at a later date.

---

## Timeline

- **Immediate**: We operate at ~25 tok/s with runtime optimizations (current deployment)
- **After upgrade**: We switch to the official vLLM container and expect ~30-40 tok/s
- **PEARC paper**: The performance improvement would strengthen our conference submission

---

*Prepared by Anas Nassar (nassar@uic.edu) — STREAM Project, UIC*
