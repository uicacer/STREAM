# Multimodal Support for STREAM

## Adding Image Understanding Across All Three Tiers

---

## Table of Contents

1. [The Problem: Why Multimodal Matters for Campus LLM Systems](#1-the-problem)
2. [Background: How Multimodal LLMs Process Images](#2-background)
3. [Architecture: How Images Flow Through STREAM](#3-architecture)
4. [The OpenAI Vision Message Format](#4-message-format)
5. [Model Selection: Why Gemma 3 4B and Qwen2.5-VL-72B](#5-model-selection)
6. [Modality-Aware Routing Strategy](#6-routing-strategy)
7. [The Vision Judge: Image-Aware Complexity Classification](#7-vision-judge)
8. [Globus Compute Constraint: The 10 MB Payload Limit](#8-globus-constraint)
9. [Frontend Image Handling: Compress, Preview, Encode](#9-frontend)
10. [Backend Changes: Message Model and Token Estimation](#10-backend)
11. [Testing Strategy](#11-testing)
12. [PEARC Paper Integration (Contribution C4)](#12-pearc)
13. [Code File Reference](#13-code-reference)
14. [Frequently Asked Questions](#14-faq)

---

## 1. The Problem: Why Multimodal Matters for Campus LLM Systems {#1-the-problem}

### The gap in campus LLM deployments

Campus LLM systems — including STREAM, Dartmouth Chat, Purdue GenAI Studio, and Tufts
LLM-Hub — currently handle text-only queries. But students and researchers increasingly
need AI to understand images:

- **STEM students** upload homework diagrams, circuit schematics, or lab results
- **Medical researchers** share microscopy images, X-rays, or histology slides
- **Data scientists** paste charts, plots, or dashboards for interpretation
- **Humanities students** analyze artworks, historical documents, or maps
- **Anyone** screenshots an error message or UI bug

Without multimodal support, users must manually describe images in text — losing
information and adding friction. This is especially limiting for visual content
like charts, diagrams, and medical imaging where textual description is inadequate.

### What "multimodal" means in practice

A multimodal LLM can accept both text and images as input. The user sends a message
like:

```
User: "What does this chart show?" + [attached image of a bar chart]
AI:   "This bar chart shows quarterly revenue for 2025, with Q3 having the
       highest revenue at $4.2M..."
```

The LLM processes both the text query and the image pixels together, using a vision
encoder (typically a Vision Transformer, or ViT) to convert the image into tokens
that the language model can reason about alongside the text tokens.

### Why no campus system has done this yet

1. **Infrastructure complexity** — Campus HPC systems use batch schedulers (SLURM)
   and remote execution frameworks (Globus Compute) that have payload size limits.
   Sending images through these systems requires careful size management.

2. **Model availability** — Vision-language models are larger and more resource-intensive
   than text-only models. Running them locally or on shared GPU resources requires
   careful resource planning.

3. **Routing complexity** — When a system supports both text and multimodal queries,
   the routing decision becomes richer: the router must consider input modality alongside
   query complexity, tier availability, and context window limits.

STREAM is the first campus LLM system to address all three challenges.

---

## 2. Background: How Multimodal LLMs Process Images {#2-background}

### The vision encoder

Multimodal LLMs have two components:

1. **Vision Encoder** (typically a Vision Transformer / ViT) — converts an image into
   a sequence of "image tokens" (dense vector representations). A single image typically
   produces 256-1024 image tokens, depending on the model and image resolution.

2. **Language Model** (the standard LLM) — processes both text tokens and image tokens
   together. The image tokens are projected into the same embedding space as text tokens,
   so the language model treats them as just another part of the input sequence.

```
Input image (224x224 pixels)
    ↓
Vision Encoder (ViT)
    ↓
Image tokens (e.g., 576 tokens for a 224x224 image at 14x14 patch size)
    ↓
Projection layer (maps image tokens to text token embedding space)
    ↓
Concatenated with text tokens: [image_token_1, ..., image_token_576, "What", "is", "this", "?"]
    ↓
Language Model processes everything together
    ↓
Output: "This is a photo of a golden retriever..."
```

### Token cost of images

Images consume tokens from the model's context window. The exact count depends on
the model:

| Model | Image token cost | Source |
|-------|-----------------|--------|
| OpenAI GPT-4 Turbo | ~765 tokens (low detail) to ~1105 (high detail) | OpenAI docs |
| Claude Sonnet 4 | ~1600 tokens per 1024x1024 image | Anthropic docs |
| Qwen2.5-VL-72B | ~256-1280 tokens (varies by resolution) | Qwen docs |
| Gemma 3 4B | ~256 tokens per image | Google docs |

For STREAM's context window validation, we use a conservative estimate of **765 tokens
per image** (OpenAI's low-detail baseline). This provides a reasonable approximation
across models without over- or under-counting.

---

## 3. Architecture: How Images Flow Through STREAM {#3-architecture}

### The end-to-end flow

```
User drags/pastes image into chat input
    ↓
Frontend: Canvas compress (max 1024px, JPEG 85%)
    ↓
Frontend: Base64 encode → build OpenAI vision content array
    ↓
POST /v1/chat/completions with multimodal message
    ↓
Backend: Pydantic validates (content: str | list[dict])
    ↓
Backend: Extract text from content for complexity judge
    ↓
Backend: Judge complexity (text-only or vision judge)
    ↓
Backend: Modality-aware routing → select tier + vision-capable model
    ↓
┌────────────────────────────────────────────────────┐
│  LOCAL:      Gemma 3 4B via Ollama (direct HTTP)   │
│  LAKESHORE:  Qwen2.5-VL-72B via Globus Compute    │
│  CLOUD:      Claude / GPT-4 via LiteLLM           │
└────────────────────────────────────────────────────┘
    ↓
Streaming response (SSE) → Frontend displays answer
```

### What changes vs text-only

The existing STREAM pipeline passes `messages` as a list of dicts from the frontend
all the way to each tier's inference engine. The pipeline itself does not transform
message content — it passes it through. This means the core change is small:

1. **Frontend** — build content arrays instead of plain strings when images are present
2. **Backend Pydantic model** — accept `str | list[dict]` instead of just `str`
3. **Backend helpers** — extract text from multimodal content for the judge and token estimator
4. **Routing** — check modality to select vision-capable models
5. **Display** — render images in chat message bubbles

The LiteLLM client, Globus Compute remote functions, vLLM server, and cloud APIs all
already accept the OpenAI vision format. No changes needed in those layers.

---

## 4. The OpenAI Vision Message Format {#4-message-format}

### Standard format (what all providers accept)

The OpenAI Chat Completions API defines a standard format for multimodal messages.
Instead of `content` being a string, it becomes an array of content blocks:

```python
# Text-only message (existing format):
{"role": "user", "content": "What is Python?"}

# Multimodal message (new format):
{"role": "user", "content": [
    {"type": "text", "text": "What is in this image?"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}}
]}
```

### Why this format works everywhere

This format is supported by:
- **OpenAI** (GPT-4 Turbo, GPT-4o) — they defined it
- **Anthropic** (Claude) — LiteLLM translates it to Anthropic's format automatically
- **vLLM** — accepts it natively for vision-language models
- **Ollama** — accepts it for vision-capable models (Gemma 3, LLaVA, etc.)

STREAM's pipeline passes `messages` as-is through every layer, so once the Pydantic
model accepts this format, it flows through to the inference engine unchanged.

### Base64 vs URL

There are two ways to pass images:

1. **Base64 data URL** — `"data:image/jpeg;base64,/9j/4AAQ..."` — the image bytes
   are embedded directly in the JSON payload. This is what STREAM uses because:
   - No external server needed to host images
   - Works offline (desktop mode)
   - No CORS issues
   - Image is self-contained in the message

2. **External URL** — `"https://example.com/image.jpg"` — the model fetches the image.
   STREAM does NOT use this because:
   - HPC compute nodes (Lakeshore) are behind a firewall and may not have outbound access
   - Requires hosting infrastructure
   - Images could disappear (broken links)

---

## 5. Model Selection: Why Gemma 3 4B and Qwen2.5-VL-72B {#5-model-selection}

### Local tier: Gemma 3 4B

STREAM's local tier now has two models:

| Model | Type | Size | RAM | Context | Use case |
|-------|------|------|-----|---------|----------|
| **Llama 3.2 3B** | Text-only | ~2 GB | ~4 GB | 32K | General text queries |
| **Gemma 3 4B** | Vision + Text | ~3.3 GB | ~6 GB | 32K | Image queries + text |

**Why Gemma 3 4B?**

1. **Fits in memory** — Docker Ollama is configured with 8 GB RAM. Gemma 3 4B uses
   ~6 GB, leaving headroom for the OS and other processes.

2. **Good vision quality for its size** — Based on Google's Gemini 2.0 architecture,
   Gemma 3 achieves strong scores on vision benchmarks (48.8 MMMU, 75.8 DocVQA for
   the 4B variant).

3. **Open source and free** — Released under Google's Gemma Terms of Use, which
   permits academic and commercial use with attribution.

4. **Fast inference** — Small enough for responsive local inference on consumer
   hardware (Apple Silicon, consumer GPUs, or CPU-only).

**Why not larger vision models locally?**

- **Llama 3.2 Vision 11B** (~7.8 GB) barely fits in 8 GB and would be too slow
- **LLaVA 7B** is older and lower quality than Gemma 3
- **Moondream 2** (2B) fits easily but vision quality is noticeably worse

**Removed models:**

- `llama3.2:1b` (local-llama-tiny) — too small for quality responses. Was used as
  a judge option, but the 3B model is fast enough for judging.
- `llama3.1:8b` (local-llama-quality) — replaced by Gemma 3 4B, which adds vision
  capability at similar quality and size.

### Lakeshore tier: Qwen2.5-VL-72B-Instruct-AWQ

This model was already configured in STREAM's `config.py` but not yet utilized:

```python
"lakeshore-qwen-vl-72b": {
    "hf_name": "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
    "host": "ghi2-002",
    "port": 8000,
    "description": "Vision + Text (72B AWQ, multimodal flagship)",
}
```

**Why Qwen2.5-VL-72B?**

1. **State-of-the-art vision quality** — 72B parameters with vision-language
   training produces excellent results on complex images (medical, scientific,
   technical diagrams).

2. **AWQ quantization** — 4-bit quantization reduces memory from ~140 GB to ~40 GB,
   fitting on a single H100 NVL 96 GB or an A100 80 GB GPU.

3. **Already on Lakeshore** — The model is deployed on `ghi2-002` as part of
   STREAM's existing vLLM infrastructure.

4. **vLLM native support** — vLLM handles Qwen2.5-VL natively with the OpenAI
   vision format, so no special handling is needed.

### Cloud tier: Claude Sonnet 4, GPT-4o, and GPT-4o Mini

All cloud models support vision natively:

- **Claude Sonnet 4** — Anthropic's flagship model, excellent at image analysis
- **GPT-4o** — OpenAI's strong general-purpose model with vision
- **GPT-4o Mini** — OpenAI's fast and affordable model with vision

LiteLLM automatically translates the OpenAI vision format for Anthropic's API,
so no changes are needed in STREAM's cloud path.

### Licensing

Both new local models are open source and free:

| Model | License | Commercial use | Attribution required |
|-------|---------|---------------|---------------------|
| **Llama 3.2 3B** | Meta Community License | Yes | "Built with Llama" |
| **Gemma 3 4B** | Google Gemma Terms of Use | Yes | Pass license terms to recipients |

---

## 6. Modality-Aware Routing Strategy {#6-routing-strategy}

### The routing decision before multimodal

STREAM's router previously made a simple decision:

```
route(query_text, complexity, tier_health) → tier
```

The complexity judge classifies the query as LOW, MEDIUM, or HIGH, and the router
maps this to a tier:

| Complexity | Preferred tier | Fallback chain |
|-----------|---------------|----------------|
| LOW | Local | Local → Lakeshore → Cloud |
| MEDIUM | Lakeshore | Lakeshore → Cloud → Local |
| HIGH | Cloud | Cloud → Lakeshore → Local |

### The richer routing decision with multimodal

With image support, the routing decision becomes:

```
route(query_text, complexity, modality, tier_health, model_capabilities) → (tier, model)
```

The key addition is **modality awareness**: the router must ensure the selected model
can actually handle the input. A text-only model cannot process images, so the router
must either:

1. Select a vision-capable model within the same tier
2. Fall back to a different tier that has a vision-capable model

### Routing rules for image queries

STREAM follows a core design principle: **when the user explicitly selects something,
respect it — don't silently override.** This principle applies to multimodal routing
in three distinct scenarios:

**Scenario 1: AUTO mode (user delegated the decision to STREAM)**

The router picks both the tier and model. It defaults to MEDIUM complexity for image
queries (most are "describe" or "explain" level). The keyword/LLM judge still runs
on the text portion and may override this if strong complexity signals are present.
The router automatically selects a vision-capable model (`local-vision`,
`lakeshore-qwen-vl-72b`, or the cloud provider).

**Scenario 2: Tier-only selection (e.g., user selected "Local" without specifying a model)**

STREAM picks the model within that tier. It automatically selects the vision-capable
model for that tier (e.g., `local-vision` instead of `local-llama`). This is not a
silent override because the user only chose the tier, not the model — STREAM is
making the model decision, which is its job.

**Scenario 3: Explicit model selection (e.g., user specifically chose `local-llama`)**

The user chose a specific text-only model. STREAM does NOT silently switch to a
different model. Instead, it returns a clear error:

> "Llama 3.2 3B is text-only and cannot process images. Switch to Gemma Vision 4B
> in settings, or set tier to Auto."

This is consistent with how STREAM handles explicit tier selection — if a user
picks a tier that is unavailable, the system raises an error rather than silently
falling back.

**Summary table:**

| User selection | Image present? | Behavior |
|---------------|---------------|----------|
| AUTO | No | Normal complexity routing |
| AUTO | Yes | Select vision-capable model, default MEDIUM |
| LOCAL (no model) | Yes | Auto-select `local-vision` |
| LOCAL + `local-llama` | Yes | **Error**: model cannot handle images |
| LOCAL + `local-vision` | Yes | Works normally |
| LAKESHORE (no model) | Yes | Auto-select `lakeshore-qwen-vl-72b` |
| LAKESHORE + text model | Yes | **Error**: model cannot handle images |
| CLOUD | Yes | Works normally (all cloud models support vision) |

### The model capabilities check

```python
VISION_CAPABLE_MODELS = {
    "local-vision",              # Gemma 3 4B (Ollama)
    "lakeshore-qwen-vl-72b",    # Qwen2.5-VL-72B (vLLM)
    "cloud-claude",              # Claude Sonnet 4
    "cloud-gpt",                 # GPT-4o
    "cloud-gpt-cheap",           # GPT-4o Mini
}
```

The routing logic checks: if the message has images and the selected model is not in
`VISION_CAPABLE_MODELS`, switch to the vision-capable model for that tier.

---

## 7. The Vision Judge: Image-Aware Complexity Classification {#7-vision-judge}

### The problem with text-only judges

STREAM's complexity judge currently analyzes only the text of the query:

```python
judge_complexity("What is this?")  →  MEDIUM (ambiguous text)
```

But the same text with different images should route differently:

| Text | Image | Should be | Text-only judge says |
|------|-------|-----------|---------------------|
| "What is this?" | Photo of a dog | LOW | MEDIUM |
| "What is this?" | Circuit diagram | HIGH | MEDIUM |
| "What is this?" | Brain MRI | HIGH | MEDIUM |
| "Analyze this" | Simple bar chart | MEDIUM | HIGH |

### The solution: an optional vision judge

STREAM adds a new judge strategy called `gemma-vision` that uses the local Gemma 3 4B
model. Unlike the text-only judges (ollama-3b, haiku), this judge can "see" images
and factor them into the complexity assessment.

```python
JUDGE_STRATEGIES = {
    "ollama-3b": {
        "model": "local-llama",
        "name": "Ollama 3b",
        "description": "Balanced accuracy, free",
        "timeout": 60,
    },
    "gemma-vision": {
        "model": "local-vision",
        "name": "Gemma Vision 4B",
        "description": "Sees images for complexity, slower, free",
        "timeout": 30,
        "multimodal": True,
    },
    "haiku": {
        "model": "cloud-haiku",
        "name": "Claude Haiku",
        "description": "Fastest & most accurate, ~$1 per 5,000 judgments",
        "timeout": 15,
    },
}
```

### Latency tradeoff

| Judge | Text-only latency | With image latency |
|-------|------------------|--------------------|
| `ollama-3b` (default) | ~1-3s | ~1-3s (ignores images) |
| `gemma-vision` | ~1-2s | **~3-8s** (processes image) |
| `haiku` | ~1-2s | ~2-4s (supports vision) |

The vision judge adds 3-8 seconds per image query. This is why it is **optional and
not the default**. The default judge (`ollama-3b`) uses text-only analysis, which is
fast and good enough for most cases.

### Default behavior for image queries (when using text-only judge)

When the default text-only judge encounters an image query, the complexity is
determined by the text portion only. If no strong complexity signal is found in the
text (e.g., "what is this?" has no keywords), the system defaults to **MEDIUM**
complexity, routing to Lakeshore or Cloud — both of which have strong vision models.

This is a reasonable default: very few image queries are truly LOW complexity (a local
1.5B-4B model would give poor answers for most image understanding tasks), and most
benefit from the larger vision models on Lakeshore or Cloud.

---

## 8. Globus Compute Constraint: The 10 MB Payload Limit {#8-globus-constraint}

### The constraint

Globus Compute enforces a **10 MB limit on task submissions**:

> "The current data limit is set to 10MB on task submissions, which applies to both
> individual functions as well as batch submissions."
> — [Globus Compute Limits documentation](https://globus-compute.readthedocs.io/en/stable/limits.html)

When the limit is exceeded, the SDK raises:
```
GlobusAPIError: TASK_PAYLOAD_TOO_LARGE
```

### Why this matters for images

A base64-encoded image is ~33% larger than the raw file (base64 encoding converts
3 bytes into 4 ASCII characters). Image sizes after base64 encoding:

| Image type | Raw size | Base64 size | Fits in 10 MB? |
|-----------|----------|-------------|----------------|
| Phone screenshot (1080p JPEG) | ~300 KB | ~400 KB | Yes |
| High-res photo (4K JPEG) | ~2-4 MB | ~3-5 MB | Tight |
| PNG screenshot (uncompressed) | ~1-3 MB | ~1.5-4 MB | Maybe |
| Multiple images in conversation | varies | cumulative | Risky |

The 10 MB includes the serialized function code (~2 KB), model name, temperature,
and the entire messages array (including conversation history). So the usable space
for image data is roughly **8-9 MB**.

### STREAM's solution: four-layer protection

STREAM uses a layered approach to prevent payload limit violations, while keeping
the user experience simple (no upload restrictions):

**Layer 1 — Frontend compression (always runs):**

All images are automatically compressed before encoding:
- Resize to max 1024 pixels on the longest side
- Convert to JPEG with 85% quality
- This typically produces images of 100-400 KB (well within limits)

**Layer 2 — Frontend Lakeshore warning (at send time):**

When the user hits send with images exceeding **6 MB**, the frontend reacts
based on the selected tier:

- **Explicit Lakeshore**: The message is **blocked** with a warning and action
  buttons to switch to Local or Cloud.
- **Auto mode**: The message is **sent** with an info banner: "Images total X MB
  (over 6 MB) — Lakeshore will be skipped for this message. Routing to Local
  or Cloud instead." This is informational, not blocking.
- **Local or Cloud**: No check needed — these tiers have no payload limit.

Action buttons on the blocking warning let the user switch tier with one click.

This limit applies **only to the Lakeshore tier** because it is the only tier
that routes through Globus Compute. Local and Cloud tiers have no practical
image size limit (images are sent via direct HTTP).

The 6 MB threshold is derived from the Globus Compute budget:

| Layer | Budget |
|-------|--------|
| Globus Compute hard limit | 10 MB |
| STREAM safety limit (`GLOBUS_MAX_PAYLOAD_BYTES`) | 8 MB |
| Reserve for text history + serialization overhead | ~2 MB |
| **Available for images in current message** | **6 MB** |

**Layer 3 — Backend payload validation + old image stripping:**

Before submitting to Globus Compute, the backend:
1. Strips images from all messages **except** the latest user message
   (`strip_old_images` in `multimodal.py`). This prevents long conversations
   with multiple image-bearing messages from exceeding the payload limit.
2. Estimates the serialized payload size. If it exceeds 8 MB, the request is
   rejected with a `payload_too_large` error.

**Layer 4 — Backend runtime fallback for payload errors:**

If Lakeshore rejects a request as `payload_too_large` during streaming, the
runtime fallback in `streaming.py` catches it and automatically routes to the
next available tier (typically Cloud). Unlike tier failures (connection errors,
timeouts), a payload error does **not** mark Lakeshore as unavailable — the
tier is healthy, only this specific request was too large. The user sees a
fallback notification ("Lakeshore unavailable — using Cloud instead") and the
response continues seamlessly.

### Why not use Globus Transfer for images?

We considered using Globus Transfer to upload images to Lakeshore's shared filesystem,
then referencing them by path in the Globus Compute function. We rejected this because:

1. **Latency** — Globus Transfer is designed for bulk data movement, not real-time chat.
   A transfer task takes 5-30+ seconds to complete.
2. **Infrastructure** — Requires STREAM's server to be registered as a Globus Collection.
3. **Cleanup** — Would need to delete images from Lakeshore after inference.
4. **OAuth** — Globus Transfer requires separate consent from the user.

Frontend compression is simpler, faster, and handles 99% of real-world images.

---

## 9. Frontend Image Handling: Compress, Preview, Encode {#9-frontend}

### Image input methods

STREAM supports four ways to add images:

1. **Click the upload button** (ImagePlus icon) — Opens a file picker (accepts JPEG, PNG, GIF, WebP)
2. **Click the camera button** (Camera icon) — Takes a photo (see "Camera capture" below)
3. **Drag and drop** — Drop an image file onto the chat input area
4. **Paste from clipboard** — Cmd+V (Mac) / Ctrl+V (Windows) to paste a screenshot

### Camera capture — cross-platform strategy

The camera button adapts to the user's device and environment using **feature detection**
(not user-agent sniffing, which breaks with iPads, foldables, and Capacitor apps).

#### Three-tier detection

```
Camera button clicked
    ↓
Is this a touch device? (navigator.maxTouchPoints > 0)
    ├── YES → Tier 1: Open NATIVE CAMERA app via <input capture="environment">
    │         (rear camera — flash, zoom, HDR, tap-to-focus, etc.)
    │
    └── NO → Is getUserMedia available? (navigator.mediaDevices?.getUserMedia)
              ├── YES → Tier 2: Open WEBCAM MODAL with live video preview
              │         (desktop browsers: Chrome, Firefox, Safari, Edge)
              │
              └── NO → Tier 3: Open FILE PICKER as fallback
                        (PyWebView without media support, old browsers)
```

#### Why `maxTouchPoints` instead of user-agent regex

User-agent strings are unreliable for device detection:
- iPads report desktop Safari user agents since iPadOS 13
- Galaxy Fold switches between mobile/desktop UAs depending on screen state
- Capacitor/Tauri apps may have custom user agents
- New device categories appear every year

`navigator.maxTouchPoints` is a **hardware capability check** that directly answers
"does this device have a touchscreen?" — regardless of what the user agent says.
It returns 0 on desktops, 5+ on modern smartphones, and 1-10 on touch laptops/tablets.

#### Platform-specific camera configuration

| Platform | Renderer | Camera Support | Configuration Needed |
|---|---|---|---|
| **Server mode (any browser)** | Chrome/Firefox/Safari/Edge | Full getUserMedia | None (works on localhost) |
| **Desktop macOS** | WKWebView | Needs entitlement | `NSCameraUsageDescription` in Info.plist (`stream.spec`) |
| **Desktop Windows** | WebView2 | Shows system prompt | None (WebView2 handles permissions natively) |
| **Desktop Linux** | WebKitGTK or Qt | Needs env var | `QTWEBENGINE_CHROMIUM_FLAGS` in `main.py` |
| **Mobile browser** | Chrome/Safari | Native camera | None (uses `<input capture>`) |
| **Future Capacitor app** | Native WebView | Native camera | `CAMERA` permission in app manifest |

#### getUserMedia error handling

The CameraModal handles every WebRTC error type with specific, actionable messages:

| Error | Cause | User Message |
|---|---|---|
| `NotAllowedError` | Permission denied | "Allow camera in browser/system settings" |
| `NotFoundError` | No camera hardware | "Connect a camera or use Upload" |
| `NotReadableError` | Camera busy | "Close Zoom/Teams and try again" |
| `OverconstrainedError` | facingMode unsupported | Auto-retry without facingMode |
| API unavailable | PyWebView limitation | "Try STREAM in your browser" |

#### Camera stream cleanup

The camera video stream **must** be stopped when the modal closes. Without cleanup:
- The camera LED stays on (privacy concern)
- Other apps can't access the camera
- Battery drain on laptops

The CameraModal stops all tracks in three places:
1. React `useEffect` cleanup (component unmounts)
2. `handleClose` function (user clicks Cancel/Close)
3. Cancelled flag check (avoids state updates after unmount)

### Compression pipeline

All images pass through the same compression pipeline before being attached:

```
Raw image file/blob
    ↓
Create an HTMLImageElement (load the image into memory)
    ↓
Calculate new dimensions:
  - If longest side > 1024px → scale down proportionally
  - If longest side ≤ 1024px → keep original size
    ↓
Draw onto a Canvas element at the new size
    ↓
Export as JPEG with quality 0.85 (85%)
    ↓
Result: base64 data URL ("data:image/jpeg;base64,...")
    ↓
Add to message (no size-based upload blocking)
```

There is **no frontend upload limit**. Users can attach as many images as they want.
Size enforcement happens only at send time, and only for the Lakeshore tier (see
Section 8). Local and Cloud tiers accept any number of compressed images.

### Why 1024px and JPEG 85%?

- **1024px** — Vision models process images at 224-768px internally (the vision
  encoder resizes). Sending a 4K image wastes bandwidth without improving quality.
  1024px provides good detail for charts, diagrams, and photos.

- **JPEG 85%** — Good quality with significant compression. At 85%, JPEG artifacts
  are barely visible but file size is 5-10x smaller than PNG. Even charts and
  screenshots with text remain readable at this quality.

- **Combined effect** — A 4K JPEG photo (~4 MB) becomes a 1024px JPEG (~200 KB).
  A full-screen PNG screenshot (~3 MB) becomes a 1024px JPEG (~150 KB). Both fit
  easily within the 10 MB Globus Compute limit.

### Message construction

When the user sends a message with images, the frontend constructs the OpenAI
vision format:

```typescript
// Text-only (existing behavior):
{ role: "user", content: "What is Python?" }

// With images (new behavior):
{ role: "user", content: [
    { type: "text", text: "What is in this image?" },
    { type: "image_url", image_url: { url: "data:image/jpeg;base64,/9j/4AAQ..." } }
]}
```

If the message has no images, `content` remains a plain string for backwards
compatibility. The backend accepts both formats.

---

## 10. Backend Changes: Message Model and Token Estimation {#10-backend}

### Pydantic Message model

The core change is in `stream/middleware/routes/chat.py`:

```python
# BEFORE: text-only
class Message(BaseModel):
    role: str
    content: str

# AFTER: text or multimodal
class Message(BaseModel):
    role: str
    content: str | list[dict]
```

This is the single change that "unlocks" the entire pipeline. Once Pydantic accepts
the content array, it flows through to every tier unchanged.

### Text extraction helper

Several components need the text portion of a message (the complexity judge, logging,
the keyword fallback). A helper extracts text from both formats:

```python
def extract_text_content(content: str | list[dict]) -> str:
    """
    Extract the text portion from a message's content.
    Works for both text-only (str) and multimodal (list[dict]) formats.
    """
    if isinstance(content, str):
        return content
    return " ".join(
        block.get("text", "") for block in content if block.get("type") == "text"
    )
```

### Modality detection helper

The router needs to know if a message contains images:

```python
def has_images(messages: list[dict]) -> bool:
    """Check if any message in the conversation contains images."""
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            if any(block.get("type") == "image_url" for block in content):
                return True
    return False
```

### Token estimation for multimodal

The existing token estimator uses `len(str(content)) // 4` which would count base64
data as tokens (a 200 KB image would be estimated as ~50,000 tokens — wildly wrong).

The fix:
- For string content: use existing character-based estimation
- For list content: count only text blocks, add 765 tokens per image

```python
def estimate_tokens(messages: list[dict]) -> int:
    total_chars = 0
    image_count = 0

    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if block.get("type") == "text":
                    total_chars += len(block.get("text", ""))
                elif block.get("type") == "image_url":
                    image_count += 1

    text_tokens = total_chars // 4
    image_tokens = image_count * 765  # Conservative estimate per image
    return text_tokens + image_tokens
```

---

## 11. Testing Strategy {#11-testing}

### Unit tests (`tests/test_multimodal.py`)

| Test | What it verifies |
|------|-----------------|
| `test_extract_text_from_string` | `extract_text_content("hello")` returns `"hello"` |
| `test_extract_text_from_content_array` | Extracts text blocks from multimodal content |
| `test_has_images_true` | Detects images in content array |
| `test_has_images_false` | Returns False for text-only messages |
| `test_message_model_accepts_string` | Pydantic accepts `content: "hello"` |
| `test_message_model_accepts_list` | Pydantic accepts `content: [{type: "text", ...}]` |
| `test_estimate_tokens_text_only` | Token estimation for text messages (unchanged) |
| `test_estimate_tokens_with_images` | Token estimation includes 765 tokens per image |
| `test_estimate_tokens_ignores_base64` | Base64 data is NOT counted as text tokens |

### Integration tests (`tests/test_multimodal_routing.py`)

| Test | What it verifies |
|------|-----------------|
| `test_image_routes_to_vision_model_local` | Image query on LOCAL tier uses `local-vision` |
| `test_image_routes_to_vision_model_lakeshore` | Image query on Lakeshore uses VL model |
| `test_text_only_model_skipped_for_images` | Text-only models are not used for image queries |
| `test_payload_size_validation` | Large payloads are rejected with helpful error |

### Manual end-to-end testing

1. **Local tier** — Send image with "What is this?" → verify Gemma 3 4B responds
2. **Lakeshore tier** — Send image → verify it goes through Globus Compute to VL model
3. **Cloud tier** — Send image → verify Claude/GPT-4 responds with image understanding
4. **Compression** — Send a 4K image → verify it's compressed before sending
5. **Persistence** — Send image, refresh page → verify image appears in history (IndexedDB)
6. **Error handling** — Send oversized image → verify helpful error message

---

## 12. PEARC Paper Integration (Contribution C4) {#12-pearc}

This work directly implements **Contribution C4: Modality-Aware Routing** from the
PEARC 2026 paper outline. The key claims:

1. **STREAM's router considers input modality** alongside query complexity, automatically
   directing multimodal queries to vision-capable tiers/models.

2. **No campus LLM system handles modality-aware routing.** The routing decision becomes
   richer: complexity + modality + context window + tier health.

3. **Three-tier vision coverage**: Gemma 3 4B locally (free, fast), Qwen2.5-VL-72B on
   HPC (free, powerful), and Claude/GPT-4 in the cloud (paid, most capable).

### Updated routing formulation

The routing function signature evolves from:

```
route(query, history, tier_health) → tier
```

to:

```
route(query, history, modality, tier_health, model_capabilities) → (tier, model)
```

This is a novel joint optimization that no existing campus LLM system implements.

### Models per tier (updated table for paper)

| Tier | Model | Parameters | Context | Multimodal |
|------|-------|------------|---------|------------|
| LOCAL (text) | Llama 3.2 3B (Ollama) | 3B | 32K | No |
| LOCAL (vision) | Gemma 3 4B (Ollama) | 4B | 32K | **Yes** |
| Lakeshore (text) | Qwen 2.5 32B AWQ (vLLM) | 32B | 32K | No |
| Lakeshore (vision) | Qwen 2.5 VL 72B AWQ (vLLM) | 72B | 32K | **Yes** |
| Cloud | Claude Sonnet 4 | Unknown | 200K | **Yes** |
| Cloud | GPT-4o | Unknown | 128K | **Yes** |
| Cloud | GPT-4o Mini | Unknown | 128K | **Yes** |

---

## 13. Code File Reference {#13-code-reference}

### Backend files modified

| File | Changes |
|------|---------|
| `stream/middleware/routes/chat.py` | `Message.content` type, `extract_text_content()`, `has_images()` |
| `stream/middleware/config.py` | `OLLAMA_MODELS`, `JUDGE_STRATEGIES`, `VISION_CAPABLE_MODELS` |
| `stream/middleware/core/query_router.py` | Modality-aware routing logic |
| `stream/middleware/core/complexity_judge.py` | Vision judge support, image query defaults |
| `stream/middleware/utils/token_estimator.py` | Multimodal token estimation |
| `stream/middleware/utils/multimodal.py` | `extract_text_content()`, `has_images()`, `strip_old_images()` |
| `stream/middleware/core/globus_compute_client.py` | Payload size validation, old image stripping |
| `stream/desktop/main.py` | `QTWEBENGINE_CHROMIUM_FLAGS` for Qt camera support |

### Build files modified

| File | Changes |
|------|---------|
| `stream.spec` | `NSCameraUsageDescription` in Info.plist for macOS camera permission |

### Frontend files modified

| File | Changes |
|------|---------|
| `frontends/react/src/types/message.ts` | `images` field on Message |
| `frontends/react/src/types/settings.ts` | Updated `JudgeStrategy` and `LocalModel` types |
| `frontends/react/src/components/input/ImageUpload.tsx` | Upload, compress, camera capture, preview. Cross-platform camera detection (`isTouchDevice()`, `hasGetUserMedia()`, `getCameraStrategy()`), `CameraModal` with WebRTC, `OverconstrainedError` retry |
| `frontends/react/src/components/input/ChatInput.tsx` | Image upload integration |
| `frontends/react/src/components/chat/ChatContainer.tsx` | Content array construction |
| `frontends/react/src/components/chat/Message.tsx` | Image rendering in messages |
| `frontends/react/src/api/stream.ts` | Preserve content arrays in API call |
| `frontends/react/src/stores/chatStore.ts` | Image storage in messages |
| `frontends/react/src/components/sidebar/SettingsPanel.tsx` | Updated model/judge options |

### Test files created

| File | Purpose |
|------|---------|
| `tests/test_multimodal.py` | Unit tests for message model, helpers, token estimation |
| `tests/test_multimodal_routing.py` | Integration tests for modality-aware routing |
| `tests/test_camera_cross_platform.py` | 37 tests: macOS Info.plist, Qt env var, feature detection, error handling, stream cleanup, mobile readiness |

---

## 14. Frequently Asked Questions {#14-faq}

### Q: Can I send multiple images in one message?

Yes. The OpenAI vision format supports multiple `image_url` blocks in the content
array. Each image is compressed independently and adds ~765 tokens to the context
window estimate. There is no hard upload limit — you can attach as many images as
you want. However, if you are using the **Lakeshore tier**, the total image data
per message is limited to **6 MB** due to Globus Compute's 10 MB task payload limit
([reference](https://globus-compute.readthedocs.io/en/stable/limits.html)). If your
images exceed 6 MB, the UI will suggest switching to Local or Cloud tier instead.
Local and Cloud tiers have no practical image size limit.

### Q: What happens if I send an image to a text-only model?

It depends on how you selected the model:

- **AUTO mode or tier-only selection** — STREAM automatically picks a vision-capable
  model for you. You'll see the selected model in the response metadata.
- **Explicit model selection** (e.g., you specifically chose `local-llama`) — STREAM
  returns an error asking you to switch to a vision-capable model or set the tier to
  Auto. This is by design: STREAM respects your explicit choices and never silently
  overrides them.

### Q: Does the conversation history retain images?

Yes. Images are stored as base64 data URLs in IndexedDB (the browser's local
database). They persist across page reloads and session changes.

For the **Lakeshore tier**, images from older messages are automatically stripped
before sending to Globus Compute (`strip_old_images` in `multimodal.py`). Only
the latest user message's images are included in the payload. The model's previous
text responses about those images remain in the history for context, so follow-up
questions about earlier images still work. This keeps the payload within Globus
Compute's 10 MB limit even for long conversations.

Local and Cloud tiers send the full conversation history including all images.

### Q: How much does image processing cost on Cloud tier?

Image tokens are priced the same as text tokens by cloud providers. A single image
at low detail (~765 tokens) costs roughly:
- Claude Sonnet 4: ~$0.002 per image
- GPT-4 Turbo: ~$0.008 per image

These are input token costs only; the output (response) costs are the same as text.

### Q: Can the vision judge be the default?

It could be, but we recommend against it. The vision judge adds 3-8 seconds of
latency per image query, which is noticeable in a chat interface. The text-only
judge with keyword fallback (defaulting to MEDIUM for image queries) is fast and
adequate for most routing decisions.

### Q: What image formats are supported?

JPEG, PNG, GIF, and WebP. All formats are converted to JPEG during compression
(for smaller file size). The frontend validates the file type before processing.

### Q: Can I paste a screenshot?

Yes. Pressing Cmd+V (Mac) or Ctrl+V (Windows) when the chat input is focused will
capture any image on the clipboard. This is especially useful for pasting screenshots,
error messages, or content copied from other applications.
