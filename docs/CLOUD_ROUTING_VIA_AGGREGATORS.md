# Cloud Routing via API Aggregators (OpenRouter & Alternatives)

## Purpose of This Report

This report is a technical implementation guide for adding API aggregator support
to STREAM's cloud tier. It documents the current architecture, evaluates all major
aggregator services, recommends which to use, and provides a concrete implementation
plan that a future session can pick up and execute.

---

## 1. The Problem with Direct API Keys

STREAM's cloud tier currently requires users to obtain and configure API keys from
each individual provider:

```
Current UX (painful for new users):
  1. Go to platform.openai.com → sign up → add billing → copy API key
  2. Go to console.anthropic.com → sign up → add billing → copy API key
  3. (Optionally) Go to Google AI Studio → sign up → copy API key
  4. Paste each key into STREAM settings or environment variables
  5. Each provider bills separately
```

This is a barrier for UIC students and researchers who just want to try STREAM.
They need 2-3 accounts, 2-3 credit cards on file, and 2-3 dashboards to monitor
usage. Most will give up after step 1.

**The solution**: An API aggregator gives one API key that routes to all providers.
The user signs up once, adds one payment method, and gets access to every model.

---

## 2. Current STREAM Cloud Architecture

### How a cloud query flows today

```
Frontend (React)                     Backend (Python)
─────────────────                    ────────────────
User picks "Cloud" tier
User picks "cloud-claude"
        │
        ▼
POST /chat/completions ─────────►  chat.py receives request
  {                                   │
    model: "cloud",                   ▼
    cloud_provider: "cloud-claude"   query_router.py
  }                                   │ get_model_for_tier("cloud", "cloud-claude")
                                      │ returns "cloud-claude"
                                      ▼
                                    litellm_direct.py
                                      │ _resolve_model("cloud-claude")
                                      │ looks up litellm_config.yaml
                                      │ "cloud-claude" → "claude-sonnet-4-20250514"
                                      ▼
                                    litellm.acompletion(
                                      model="claude-sonnet-4-20250514",
                                      api_key=os.environ["ANTHROPIC_API_KEY"],
                                      messages=[...],
                                      stream=True,
                                    )
                                      │
                                      ▼
                                    Anthropic API (api.anthropic.com)
```

### Key files and their roles

| File | Role |
|------|------|
| `frontends/react/src/stores/settingsStore.ts` | Stores `cloudProvider: CloudProvider` type, persisted to localStorage |
| `frontends/react/src/components/sidebar/SettingsPanel.tsx` | UI for selecting cloud provider (radio buttons) |
| `stream/middleware/config.py` | Defines `CLOUD_PROVIDERS` dict and `DEFAULT_MODELS` |
| `stream/gateway/litellm_config.yaml` | Maps friendly names to actual model IDs + API keys |
| `stream/middleware/core/litellm_direct.py` | `_resolve_model()` translates names, `forward_direct()` streams responses |
| `stream/middleware/core/query_router.py` | `get_model_for_tier()` returns the model name for the selected provider |
| `stream/middleware/core/tier_health.py` | Health checks per provider, caches results as `cloud:{provider}` |
| `stream/middleware/routes/chat.py` | Receives `cloud_provider` in `ChatCompletionRequest` |
| `stream/middleware/routes/health.py` | `/health/cloud-providers` endpoint returns available providers |

### Key variables and types

```typescript
// Frontend (settingsStore.ts)
type CloudProvider = 'cloud-claude' | 'cloud-gpt' | 'cloud-gpt-cheap'
```

```python
# Backend (config.py)
CLOUD_PROVIDERS = {
    "cloud-claude":    {"name": "Claude Sonnet 4",  "provider": "Anthropic", "env_key": "ANTHROPIC_API_KEY"},
    "cloud-gpt":       {"name": "GPT-4 Turbo",      "provider": "OpenAI",    "env_key": "OPENAI_API_KEY"},
    "cloud-gpt-cheap": {"name": "GPT-3.5 Turbo",    "provider": "OpenAI",    "env_key": "OPENAI_API_KEY"},
}
```

```yaml
# litellm_config.yaml
- model_name: cloud-claude
  litellm_params:
    model: claude-sonnet-4-20250514
    api_key: os.environ/ANTHROPIC_API_KEY
```

---

## 3. What is an API Aggregator?

An API aggregator is a service that provides a single API endpoint to access models
from multiple AI providers. Instead of calling OpenAI, Anthropic, and Google directly,
you call the aggregator, and it routes to the right provider.

```
WITHOUT aggregator:
  STREAM → OpenAI API    (needs OPENAI_API_KEY)
  STREAM → Anthropic API (needs ANTHROPIC_API_KEY)
  STREAM → Google API    (needs GOOGLE_API_KEY)
  = 3 accounts, 3 keys, 3 billing dashboards

WITH aggregator:
  STREAM → OpenRouter API (needs OPENROUTER_API_KEY)
         └→ Routes to OpenAI, Anthropic, Google, Meta, Mistral, etc.
  = 1 account, 1 key, 1 billing dashboard
```

The aggregator handles:
- Authentication with each upstream provider (using THEIR API keys, not yours)
- Billing consolidation (you pay the aggregator, they pay providers)
- Model discovery (one catalog of all available models)
- API translation (you always use OpenAI-compatible format)

---

## 4. All Major Aggregator Services Compared

### 4.1 OpenRouter

- **What**: Pure API aggregator. 500+ models from 60+ providers.
- **API**: OpenAI-compatible. Change `base_url` to `https://openrouter.ai/api/v1`.
- **LiteLLM prefix**: `openrouter/` (native support)
- **Pricing**: No markup on provider prices. 5.5% platform fee on credit purchases.
  BYOK (bring your own key) mode charges 5% usage fee.
- **Free tier**: 18-31 free models (rate-limited: ~20 req/min, ~200 req/day).
  No credit card required.
- **Strengths**: Largest model catalog, single API key for everything, free models
  for testing, transparent pricing (shows per-model costs).
- **Weaknesses**: Adds a network hop (~50-100ms latency). Depends on third-party
  uptime. 5.5% fee on purchases.

### 4.2 Together AI

- **What**: Inference platform focused on open-source models + fine-tuning.
- **API**: OpenAI-compatible.
- **LiteLLM prefix**: `together_ai/` (native support)
- **Pricing**: Pay-per-token. $25 free credits on signup.
- **Free tier**: $25 credits (no ongoing free tier).
- **Models**: 200+ open-source models (Llama, Qwen, Mistral, DeepSeek, etc.).
  No proprietary models (no GPT, no Claude).
- **Strengths**: Fine-tuning support, strong open-source focus, good for research.
- **Weaknesses**: No proprietary models. Once $25 credits run out, must pay.

### 4.3 Groq

- **What**: Ultra-fast inference using custom LPU hardware. 300-1,200 tokens/sec.
- **API**: OpenAI-compatible.
- **LiteLLM prefix**: `groq/` (native support)
- **Pricing**: Pay-per-token. Example: Llama 3.3 70B at $0.59/M input, $0.79/M output.
- **Free tier**: Yes. 1,000 req/day, 6,000 tokens/min. No credit card needed.
- **Models**: ~15 models (curated for speed: Llama, Mixtral, Gemma, Whisper).
  No proprietary models.
- **Strengths**: 10x faster than GPU inference. Free tier is very usable.
  Sub-300ms time-to-first-token.
- **Weaknesses**: Very limited model selection. No GPT, no Claude. Speed varies
  under load.

### 4.4 Fireworks AI

- **What**: Fast inference platform with serverless and dedicated options.
- **API**: OpenAI-compatible.
- **LiteLLM prefix**: `fireworks_ai/` (native support)
- **Pricing**: Pay-per-token. 50% discount on cached inputs.
- **Free tier**: Developer tier is free — 6,000 RPM and 2.5B tokens/day.
  $1 free credits on signup.
- **Models**: Dozens of open-source models. No proprietary models.
- **Strengths**: Absurdly generous free tier (2.5B tokens/day). Fine-tuning
  at no extra serving cost.
- **Weaknesses**: No proprietary models. Smaller catalog than OpenRouter.

### 4.5 DeepInfra

- **What**: Low-cost inference for open-source models.
- **API**: OpenAI-compatible.
- **LiteLLM prefix**: `deepinfra/` (native support)
- **Pricing**: Pay-per-token. Among the cheapest available.
  Example: Llama 3.2 1B at $0.01/M tokens.
- **Free tier**: Limited trial (can try models without API key).
- **Models**: Dozens of open-source models.
- **Strengths**: Cheapest per-token pricing available. Simple API.
- **Weaknesses**: No proprietary models. No free ongoing tier.

### 4.6 Cerebras

- **What**: Ultra-fast inference using custom Wafer-Scale Engine chips.
- **API**: OpenAI-compatible.
- **LiteLLM prefix**: `cerebras/` (native support)
- **Pricing**: Llama 3.1 8B at $0.10/M input. 405B at $6/M input.
- **Free tier**: Yes (with generous limits).
- **Models**: ~10 models (Llama family primarily).
- **Strengths**: Extreme speed (up to 3,000 tok/s). Free tier.
  Competes with Groq on speed.
- **Weaknesses**: Very limited model selection. Newer service.

### 4.7 Amazon Bedrock

- **What**: AWS's managed foundation model service.
- **API**: Not natively OpenAI-compatible, but AWS provides a compatibility gateway.
- **LiteLLM prefix**: `bedrock/` (native support)
- **Pricing**: Per-token, provisioned throughput, or batch (50% off).
- **Free tier**: AWS Free Tier includes limited usage. Academic credit programs.
- **Models**: 30+ models (Claude, Llama, Mistral, Cohere, Amazon Titan, OpenAI).
- **Strengths**: Enterprise-grade. HIPAA/SOC2 compliant. AWS academic programs.
- **Weaknesses**: Requires AWS account. Complex IAM setup. 15-40% overhead
  above advertised prices.

### 4.8 Azure AI Foundry

- **What**: Microsoft's model marketplace. 1,900+ models.
- **API**: OpenAI-compatible for Azure OpenAI models.
- **LiteLLM prefix**: `azure/` or `azure_ai/` (native support)
- **Pricing**: Per-token or Provisioned Throughput Units (PTUs).
- **Free tier**: Azure for Students provides $100 credits.
- **Models**: 1,900+ (largest catalog). GPT, Claude, Llama, Mistral, DeepSeek.
- **Strengths**: Largest model catalog. Azure for Students/Education programs.
  Microsoft ecosystem integration.
- **Weaknesses**: Requires Azure account. Complex deployment setup. Enterprise
  overhead in pricing.

### 4.9 Google Vertex AI Model Garden

- **What**: Google Cloud's model marketplace.
- **API**: OpenAI-compatible endpoints available.
- **LiteLLM prefix**: `vertex_ai/` (native support)
- **Pricing**: Per-token for generative AI. Compute-based for custom models.
- **Free tier**: $300 credits for new accounts. Some Gemini usage free via AI Studio.
- **Models**: Hundreds (Gemini, Claude, Llama, Qwen, open models).
- **Strengths**: Native Gemini access. Strong MLOps tooling. Academic programs.
- **Weaknesses**: Requires Google Cloud account. Complex setup.

### 4.10 Replicate

- **What**: Platform to run open-source ML models via API. Strong in image/video/audio.
- **API**: Own format (not OpenAI-compatible), but LiteLLM translates.
- **LiteLLM prefix**: `replicate/` (native support)
- **Pricing**: Time-based (per second of compute) or output-based.
- **Free tier**: Limited free runs on select models.
- **Models**: Thousands (community-contributed). Best for multimodal.
- **Strengths**: Broadest ecosystem including image, video, audio, 3D models.
- **Weaknesses**: Not ideal for text-only LLM inference. Non-standard API.

---

## 5. Comparison Matrix

| Provider | Proprietary Models | Open-Source Models | Free Tier | LiteLLM Native | Best For |
|----------|-------------------|-------------------|-----------|---------------|----------|
| **OpenRouter** | GPT, Claude, Gemini | Llama, Mistral, etc. | 18-31 free models | Yes | Maximum variety, single key |
| **Together AI** | None | 200+ | $25 credits | Yes | Open-source + fine-tuning |
| **Groq** | None | ~15 | 1K req/day | Yes | Fastest inference |
| **Fireworks AI** | None | Dozens | 2.5B tok/day | Yes | Most generous free tier |
| **DeepInfra** | None | Dozens | Trial only | Yes | Cheapest per-token |
| **Cerebras** | None | ~10 | Yes | Yes | Ultra-fast inference |
| **Amazon Bedrock** | Claude, GPT, Titan | Llama, Mistral | AWS Free Tier | Yes | Enterprise / AWS shops |
| **Azure AI** | GPT, Claude | 1,900+ catalog | $100 student | Yes | Microsoft ecosystem |
| **Vertex AI** | Gemini, Claude | Llama, Qwen | $300 new account | Yes | Google ecosystem |
| **Replicate** | None | Thousands | Limited | Yes | Multimodal (image/video) |

Key observation: **Only OpenRouter and the cloud hyperscalers (AWS/Azure/GCP) offer
both proprietary AND open-source models through one API.** The others are open-source only.

---

## 6. Recommendation for STREAM

### Primary: OpenRouter

**Why OpenRouter is the right choice for STREAM:**

1. **Single key for all models.** One API key gives access to GPT-4, Claude, Gemini,
   Llama, Mistral, and 500+ more. This is the biggest UX improvement possible.

2. **LiteLLM native support.** Prefix any model with `openrouter/` and it works.
   Minimal code changes needed.

3. **Free models for development and demos.** Students can use STREAM's cloud tier
   without spending any money. Free models include Llama, Mistral, Gemma, and others.

4. **Transparent per-model pricing.** OpenRouter shows the exact cost per model.
   Users can make informed choices (fast+cheap vs slow+smart).

5. **No commitment.** Pay-as-you-go. No subscriptions. No minimum spend.

6. **Preserves direct key support.** Users who already have OpenAI or Anthropic keys
   can continue using them directly. OpenRouter is an additional option, not a replacement.

### Secondary (future): Groq or Cerebras as "Speed Tier"

Once OpenRouter is integrated, consider adding Groq or Cerebras as a dedicated
speed option. Their custom hardware delivers 10x faster inference than GPU-based
providers. This would give STREAM four tiers:

```
Local      → Ollama (private, free, slower)
Lakeshore  → Campus GPU (free for UIC, medium speed)
Cloud      → OpenRouter (paid, many models, normal speed)
Cloud Fast → Groq/Cerebras (paid, limited models, extreme speed)
```

But this is a future enhancement. Start with OpenRouter.

---

## 7. Implementation Plan

### 7.1 Backend Changes

#### File: `stream/middleware/config.py`

Add OpenRouter to `CLOUD_PROVIDERS` alongside existing providers:

```python
CLOUD_PROVIDERS = {
    # --- OpenRouter (aggregator — one key for all models) ---
    "cloud-openrouter-claude": {
        "name": "Claude Sonnet 4",
        "provider": "OpenRouter",
        "description": "Best for complex reasoning and coding",
        "env_key": "OPENROUTER_API_KEY",
        "category": "smart",
    },
    "cloud-openrouter-gpt": {
        "name": "GPT-4o",
        "provider": "OpenRouter",
        "description": "Strong general-purpose model",
        "env_key": "OPENROUTER_API_KEY",
        "category": "balanced",
    },
    "cloud-openrouter-cheap": {
        "name": "GPT-4o Mini",
        "provider": "OpenRouter",
        "description": "Fast and affordable",
        "env_key": "OPENROUTER_API_KEY",
        "category": "fast",
    },
    "cloud-openrouter-llama": {
        "name": "Llama 3.1 70B",
        "provider": "OpenRouter",
        "description": "Strong open-source model (may be free)",
        "env_key": "OPENROUTER_API_KEY",
        "category": "open-source",
    },
    # --- Direct provider keys (advanced users) ---
    "cloud-claude": {
        "name": "Claude Sonnet 4",
        "provider": "Anthropic (Direct)",
        "description": "Direct Anthropic API key",
        "env_key": "ANTHROPIC_API_KEY",
        "category": "direct",
    },
    "cloud-gpt": {
        "name": "GPT-4 Turbo",
        "provider": "OpenAI (Direct)",
        "description": "Direct OpenAI API key",
        "env_key": "OPENAI_API_KEY",
        "category": "direct",
    },
}
```

#### File: `stream/gateway/litellm_config.yaml`

Add OpenRouter model entries:

```yaml
# --- OpenRouter models (one key for all) ---
- model_name: cloud-openrouter-claude
  litellm_params:
    model: openrouter/anthropic/claude-sonnet-4
    api_key: os.environ/OPENROUTER_API_KEY

- model_name: cloud-openrouter-gpt
  litellm_params:
    model: openrouter/openai/gpt-4o
    api_key: os.environ/OPENROUTER_API_KEY

- model_name: cloud-openrouter-cheap
  litellm_params:
    model: openrouter/openai/gpt-4o-mini
    api_key: os.environ/OPENROUTER_API_KEY

- model_name: cloud-openrouter-llama
  litellm_params:
    model: openrouter/meta-llama/llama-3.1-70b-instruct
    api_key: os.environ/OPENROUTER_API_KEY

# --- Direct provider models (existing, unchanged) ---
- model_name: cloud-claude
  litellm_params:
    model: claude-sonnet-4-20250514
    api_key: os.environ/ANTHROPIC_API_KEY
```

LiteLLM handles the `openrouter/` prefix natively — it knows to route to
`https://openrouter.ai/api/v1` and format the request correctly. No custom
code needed for the routing itself.

#### File: `stream/middleware/core/litellm_direct.py`

No changes needed. `_resolve_model()` already reads from `litellm_config.yaml`
and passes kwargs to `litellm.acompletion()`. The `openrouter/` prefix is handled
internally by LiteLLM.

#### File: `stream/middleware/core/tier_health.py`

No structural changes needed. The existing health check logic already:
- Reads `env_key` from `CLOUD_PROVIDERS` to check if the key is set
- Makes a 1-token test call to verify the key works
- Caches results per provider (`cloud:cloud-openrouter-claude`)

The only consideration: OpenRouter uses a single key for all models, so if the
key is invalid, ALL OpenRouter models fail simultaneously. The health check
should detect this correctly since each model still gets its own test call.

#### File: `stream/middleware/routes/health.py`

The existing `/health/cloud-providers` endpoint already returns `CLOUD_PROVIDERS`.
Adding new entries to that dict automatically exposes them to the frontend.

### 7.2 Frontend Changes

#### File: `frontends/react/src/stores/settingsStore.ts`

Expand the `CloudProvider` type:

```typescript
type CloudProvider =
  // OpenRouter (aggregator)
  | 'cloud-openrouter-claude'
  | 'cloud-openrouter-gpt'
  | 'cloud-openrouter-cheap'
  | 'cloud-openrouter-llama'
  // Direct provider keys
  | 'cloud-claude'
  | 'cloud-gpt'
  | 'cloud-gpt-cheap'
```

Update the default to an OpenRouter model:

```typescript
cloudProvider: 'cloud-openrouter-claude' as CloudProvider,
```

Bump the store version to trigger a migration for existing users.

#### File: `frontends/react/src/components/sidebar/SettingsPanel.tsx`

Redesign the cloud provider section to group by routing method:

```
Cloud Model
┌─────────────────────────────────────────────┐
│ Via OpenRouter (one API key for all models)  │
│                                              │
│  ○ Claude Sonnet 4    — Smart, reasoning     │
│  ○ GPT-4o             — General purpose      │
│  ○ GPT-4o Mini        — Fast, affordable     │
│  ○ Llama 3.1 70B      — Open-source, may be  │
│                          free                 │
│                                              │
│ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
│                                              │
│ Direct API Keys (advanced)                   │
│                                              │
│  ○ Claude Sonnet 4    — Needs ANTHROPIC key  │
│  ○ GPT-4 Turbo        — Needs OPENAI key     │
└─────────────────────────────────────────────┘
```

The `CLOUD_PROVIDER_CONFIG` record in `SettingsPanel.tsx` (currently hardcoded)
should be replaced by fetching from the `/health/cloud-providers` endpoint,
which already returns the full `CLOUD_PROVIDERS` dict from config.py.

### 7.3 API Key Management

#### Desktop mode

The user sets `OPENROUTER_API_KEY` in their `~/.stream/` config or through a
future settings UI. In the current implementation, API keys are environment
variables. The desktop app could offer a settings panel where the user pastes
their OpenRouter key, and the app writes it to `~/.stream/.env` or sets it
in the process environment.

```
Settings → API Keys
┌─────────────────────────────────────────────┐
│                                              │
│ OpenRouter API Key (recommended):            │
│ ┌──────────────────────────────────────────┐ │
│ │ sk-or-v1-abc123...                       │ │
│ └──────────────────────────────────────────┘ │
│ Get a key at openrouter.ai/keys              │
│                                              │
│ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
│                                              │
│ Or use direct provider keys (advanced):      │
│ ┌──────────────────────────────────────────┐ │
│ │ Anthropic: sk-ant-...                    │ │
│ │ OpenAI:    sk-...                        │ │
│ └──────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

#### Server mode (Docker)

The `.env` file gets a new variable:

```bash
# .env
OPENROUTER_API_KEY=sk-or-v1-...

# Optional: direct keys for users who prefer them
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

### 7.4 What Does NOT Change

These components work without modification:

- **`stream/middleware/routes/chat.py`** — Already passes `cloud_provider` through
  the entire chain. New provider names flow through the same path.
- **`stream/middleware/core/query_router.py`** — `get_model_for_tier()` already
  returns whatever `cloud_provider` string the frontend sends.
- **`stream/middleware/core/streaming.py`** — SSE pipeline is model-agnostic. It
  processes whatever chunks `forward_direct()` yields.
- **React chat components** — `ChatContainer.tsx`, `Message.tsx`, etc. are completely
  unaware of which model generated the response.
- **Health polling** — `healthStore.ts` already passes `cloudProvider` to the
  backend and displays whatever status comes back.

---

## 8. OpenRouter-Specific Details

### Authentication

```
API Key format: sk-or-v1-<64 hex characters>
Base URL:       https://openrouter.ai/api/v1
Auth header:    Authorization: Bearer sk-or-v1-...
```

### Model naming convention

OpenRouter uses `provider/model-name` format:

```
openrouter/openai/gpt-4o
openrouter/anthropic/claude-sonnet-4
openrouter/meta-llama/llama-3.1-70b-instruct
openrouter/google/gemini-2.0-flash
openrouter/mistralai/mistral-large
```

In LiteLLM, prefix with `openrouter/`:
```python
litellm.completion(model="openrouter/anthropic/claude-sonnet-4", ...)
```

### Free models

OpenRouter offers 18-31 free models (count varies). These are rate-limited
(~20 req/min, ~200 req/day) but require no payment. Examples:

- `meta-llama/llama-3.1-8b-instruct:free`
- `mistralai/mistral-7b-instruct:free`
- `google/gemma-2-9b-it:free`

These are perfect for UIC students who want to try STREAM without any cost.
The `:free` suffix in the model name selects the free tier.

### Pricing transparency

OpenRouter's `/api/v1/models` endpoint returns pricing per model:

```json
{
  "id": "anthropic/claude-sonnet-4",
  "pricing": {
    "prompt": "0.000003",
    "completion": "0.000015"
  },
  "context_length": 200000
}
```

This could be displayed in the STREAM settings UI so users know what each
model costs before selecting it.

### BYOK (Bring Your Own Key) mode

OpenRouter also supports BYOK — users who already have an Anthropic or OpenAI
key can route through OpenRouter using their own key. OpenRouter charges a 5%
fee instead of the 5.5% credit purchase fee. This is the worst of both worlds
for STREAM (extra hop + fee + need own key) and should NOT be used. If a user
has their own key, they should use direct mode instead.

---

## 9. Future Enhancement: Dynamic Model Discovery

Once the basic OpenRouter integration works, a future enhancement could fetch
the full model catalog dynamically:

```python
# Backend: new endpoint
@router.get("/health/openrouter-models")
async def get_openrouter_models():
    """Fetch available models from OpenRouter's catalog."""
    async with httpx.AsyncClient() as client:
        resp = await client.get("https://openrouter.ai/api/v1/models")
        models = resp.json()["data"]

    # Filter and categorize for the UI
    return {
        "models": [
            {
                "id": m["id"],
                "name": m["name"],
                "provider": m["id"].split("/")[0],
                "price_per_1k_input": float(m["pricing"]["prompt"]) * 1000,
                "price_per_1k_output": float(m["pricing"]["completion"]) * 1000,
                "context_length": m["context_length"],
                "is_free": ":free" in m["id"],
            }
            for m in models
            if m.get("pricing", {}).get("prompt") is not None
        ]
    }
```

The frontend could then show a searchable model browser with real-time pricing.
But this is a later enhancement — the curated list in Section 7 is sufficient
to start.

---

## 10. Migration Path

### Phase 1: Add OpenRouter alongside existing providers

- Add OpenRouter models to `config.py` and `litellm_config.yaml`
- Expand `CloudProvider` type in frontend
- Group providers in settings UI (OpenRouter vs Direct)
- Keep all existing direct-key providers working
- Default new users to OpenRouter, existing users keep their selection

### Phase 2: API key settings UI

- Add a settings panel where users can enter API keys directly in the app
- Store keys in `~/.stream/.env` (desktop) or pass through frontend (server)
- Show which keys are configured and which models are available

### Phase 3: Dynamic model catalog

- Fetch models from OpenRouter API
- Show searchable/filterable model browser in settings
- Display pricing, context length, and provider info
- Let users favorite/pin frequently used models

---

## 11. Summary

| Aspect | Current | With OpenRouter |
|--------|---------|----------------|
| Keys needed | 2-3 (one per provider) | 1 (OpenRouter) |
| Models available | 3 (Claude, GPT-4, GPT-3.5) | 500+ (all providers) |
| Free options | None | 18-31 free models |
| User signup effort | 3 accounts + 3 billing setups | 1 account |
| Code changes | — | ~50 lines config + ~100 lines UI |
| Backend logic changes | — | None (LiteLLM handles routing) |
| Direct keys still work? | Yes | Yes (kept as "advanced" option) |
