# Cloud Routing via API Aggregators

## What This Document Covers

This document is a complete educational guide to how STREAM's cloud tier works.
It explains, step by step, how a user's model selection travels from the React
frontend all the way to a cloud AI provider (like OpenAI, Anthropic, or Google)
and how the response streams back — including reasoning/thinking content and
model verification.

If you're new to this codebase, read this document front to back. By the end,
you'll understand every hop in the journey and why each piece exists.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [What Is an API Aggregator?](#2-what-is-an-api-aggregator)
3. [The Complete Journey of a Cloud Chat Message](#3-the-complete-journey-of-a-cloud-chat-message)
4. [Step 1: The User Selects a Model (Frontend)](#4-step-1-the-user-selects-a-model-frontend)
5. [Step 2: The User Sends a Message (Frontend → Backend)](#5-step-2-the-user-sends-a-message-frontend--backend)
6. [Step 3: The Backend Receives the Request](#6-step-3-the-backend-receives-the-request)
7. [Step 4: Model Resolution — The Key Translation Step](#7-step-4-model-resolution--the-key-translation-step)
8. [Step 5: API Key Injection](#8-step-5-api-key-injection)
9. [Step 6: The LiteLLM Call](#9-step-6-the-litellm-call)
10. [Step 7: The Response Streams Back](#10-step-7-the-response-streams-back)
11. [Step 8: The Frontend Renders the Stream](#11-step-8-the-frontend-renders-the-stream)
12. [Desktop Mode vs. Server Mode](#12-desktop-mode-vs-server-mode)
13. [The Dynamic Model Catalog](#13-the-dynamic-model-catalog)
14. [Tier Fallback: What Happens When a Tier Fails](#14-tier-fallback-what-happens-when-a-tier-fails)
15. [Reasoning/Thinking Content](#15-reasoningthinking-content)
16. [Model Verification](#16-model-verification)
17. [Error Handling](#17-error-handling)
18. [API Key Management and Security](#18-api-key-management-and-security)
19. [All Major Aggregator Services Compared](#19-all-major-aggregator-services-compared)
20. [Why OpenRouter?](#20-why-openrouter)
21. [File Reference](#21-file-reference)

---

## 1. The Big Picture

STREAM has three tiers for running AI models:

```
LOCAL       →  Ollama on your machine (private, free, slower)
LAKESHORE   →  Campus GPU cluster at UIC (free for students, medium speed)
CLOUD       →  Commercial AI providers (paid, many models, fastest/smartest)
```

The cloud tier is where this document focuses. It connects STREAM to hundreds
of AI models from companies like OpenAI (GPT), Anthropic (Claude), Google
(Gemini), Meta (Llama), Mistral, and many more.

The core challenge: how do you give users access to 500+ cloud models without
requiring them to sign up for dozens of different AI provider accounts?

The answer: an **API aggregator** called **OpenRouter**.

---

## 2. What Is an API Aggregator?

An API aggregator is a service that sits between your application and multiple
AI providers. Instead of calling each provider directly, you call the aggregator
once, and it routes your request to the right provider.

```
WITHOUT an aggregator (the old way):

  STREAM  →  OpenAI API      (needs OPENAI_API_KEY)
  STREAM  →  Anthropic API   (needs ANTHROPIC_API_KEY)
  STREAM  →  Google API      (needs GOOGLE_API_KEY)

  = 3 accounts, 3 API keys, 3 billing dashboards, 3 credit cards

WITH an aggregator (what we do now):

  STREAM  →  OpenRouter API  (needs 1 OPENROUTER_API_KEY)
              ├→  Routes to OpenAI
              ├→  Routes to Anthropic
              ├→  Routes to Google
              ├→  Routes to Meta, Mistral, DeepSeek, Cohere, ...
              └→  500+ models total

  = 1 account, 1 API key, 1 billing dashboard
```

OpenRouter handles:
- **Authentication** with each upstream provider (using THEIR API keys, not yours)
- **Billing consolidation** (you pay OpenRouter, they pay the providers)
- **Model discovery** (one catalog of all available models with pricing)
- **API translation** (you always use OpenAI-compatible format, regardless of provider)

The key insight: **LiteLLM** (the Python library STREAM uses for all LLM calls)
has native OpenRouter support. Prefix any model with `openrouter/` and LiteLLM
knows to route it through `https://openrouter.ai/api/v1` automatically.

---

## 3. The Complete Journey of a Cloud Chat Message

Before diving into details, here's the 30-second overview of what happens when
a user sends a message using a cloud model:

```
┌─────────────────────────── FRONTEND (React) ───────────────────────────┐
│                                                                        │
│  1. User selects "Claude Sonnet 4" in settings                        │
│     → cloudProvider = "cloud-or-claude"                                │
│                                                                        │
│  2. User types "Explain recursion" and hits Enter                     │
│     → POST /v1/chat/completions                                       │
│     → Body: { cloud_provider: "cloud-or-claude",                      │
│               openrouter_api_key: "sk-or-v1-abc123...",               │
│               messages: [...] }                                        │
│                                                                        │
└────────────────────────────────┬───────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────── BACKEND (Python) ───────────────────────────┐
│                                                                        │
│  3. chat.py receives the request, extracts API key                    │
│     → user_api_keys = {"OPENROUTER_API_KEY": "sk-or-v1-abc123..."}    │
│                                                                        │
│  4. query_router.py returns the model name                            │
│     → get_model_for_tier("cloud", "cloud-or-claude") = "cloud-or-claude" │
│                                                                        │
│  5. streaming.py orchestrates the stream                              │
│     → Passes model + user_api_keys to litellm_client.py               │
│                                                                        │
│  6. litellm_direct.py resolves the model name                         │
│     → _resolve_model("cloud-or-claude")                               │
│     → Looks up litellm_config.yaml                                    │
│     → Returns {model: "openrouter/anthropic/claude-sonnet-4"}         │
│                                                                        │
│  7. litellm_direct.py injects the user's API key                      │
│     → kwargs["api_key"] = "sk-or-v1-abc123..."                        │
│                                                                        │
│  8. litellm.acompletion() is called                                   │
│     → LiteLLM sees "openrouter/" prefix                               │
│     → Routes to https://openrouter.ai/api/v1                          │
│     → OpenRouter routes to Anthropic's Claude API                     │
│                                                                        │
│  9. Response streams back as SSE events                                │
│     → streaming.py extracts thinking content, verified model, costs   │
│     → Forwards tokens to the frontend                                 │
│                                                                        │
└────────────────────────────────┬───────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────── FRONTEND (React) ───────────────────────────┐
│                                                                        │
│  10. stream.ts parses SSE events                                      │
│      → onMetadata() → shows tier, model, cost                         │
│      → onThinking() → shows reasoning in collapsible block            │
│      → onToken() → streams response text word by word                 │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

Now let's go through each step in detail.

---

## 4. Step 1: The User Selects a Model (Frontend)

### Where it happens

`frontends/react/src/components/sidebar/SettingsPanel.tsx`

### Three types of cloud models

The settings panel shows cloud models in three groups:

**1. Curated models** — Pre-selected "best of" models that we've tested and
configured. These have friendly IDs like `cloud-or-claude`, `cloud-or-gpt4o`.
They're defined both in the frontend (`OPENROUTER_MODELS` array in
SettingsPanel.tsx) and in the backend (`CLOUD_PROVIDERS` dict in config.py +
entries in litellm_config.yaml).

**2. Favorite models** — Models the user has "starred" from the catalog browser.
These are dynamic models (from OpenRouter's API) that the user wants quick access
to. They're stored as an array of OpenRouter model IDs in localStorage, like
`["openai/gpt-4o", "anthropic/claude-sonnet-4"]`.

**3. Dynamic catalog models** — The full catalog of 300+ models fetched from
OpenRouter's API. The user browses these in a searchable modal with filters
(by provider, free, multimodal, etc.).

### How model selection works

When the user clicks a model, the `handleCloudProviderChange()` function fires:

```
Curated model clicked:
  handleCloudProviderChange("cloud-or-claude")

Favorite model clicked:
  handleCloudProviderChange("cloud-or-dynamic-anthropic/claude-sonnet-4")

Catalog model clicked:
  handleCloudProviderChange("cloud-or-dynamic-openai/gpt-5")
```

Notice the naming convention:
- `cloud-or-*` → Curated models (known to the backend, in litellm_config.yaml)
- `cloud-or-dynamic-*` → Dynamic models (NOT in litellm_config.yaml, discovered at runtime)
- `cloud-claude`, `cloud-gpt` → Direct provider models (use provider API key directly)

This naming convention is critical. The backend uses these prefixes to decide
how to resolve the model name and which API key to use.

### Where the selection is stored

`frontends/react/src/stores/settingsStore.ts`

The Zustand store persists `cloudProvider` (a `string` type) to localStorage.
It also stores API keys and favorite models:

```typescript
// Simplified — key fields in the settings store
{
  cloudProvider: "cloud-or-claude",         // Currently selected model
  openrouterApiKey: "sk-or-v1-abc123...",   // User's OpenRouter key
  anthropicApiKey: "",                       // Optional direct Anthropic key
  openaiApiKey: "",                          // Optional direct OpenAI key
  favoriteModels: ["openai/gpt-4o", ...],   // Starred model IDs
}
```

The `CloudProvider` type is just `string` (not a union type) because we need
to support 300+ dynamic model IDs that aren't known at compile time.

---

## 5. Step 2: The User Sends a Message (Frontend → Backend)

### Where it happens

`frontends/react/src/api/stream.ts` → `streamChat()` function

### What gets sent

When the user hits Enter, `ChatContainer.tsx` calls `streamChat()`, which makes
a POST request to the backend:

```
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "cloud",                          ← The tier (always "cloud" for cloud)
  "cloud_provider": "cloud-or-claude",       ← The specific model the user picked
  "openrouter_api_key": "sk-or-v1-abc123",   ← Only sent for cloud-or-* models
  "messages": [
    {"role": "user", "content": "Explain recursion"}
  ],
  "temperature": 0.7,
  "stream": true
}
```

### Smart key selection

The frontend only sends the API key that's needed for the selected model.
This is a security optimization — don't send credentials you don't need:

```typescript
// In stream.ts — only the relevant key is included in the request body

if (cloudProvider starts with "cloud-or")    → send openrouter_api_key
if (cloudProvider === "cloud-claude")        → send anthropic_api_key
if (cloudProvider === "cloud-gpt" or "cloud-gpt-cheap") → send openai_api_key
```

For example, if the user selected a direct Anthropic model (`cloud-claude`),
only `anthropic_api_key` is sent. The OpenRouter key stays in the browser.

### The response format

The backend responds with a **Server-Sent Events (SSE)** stream. This is a
long-lived HTTP connection where the server sends data incrementally:

```
data: {"stream_metadata": {"tier": "cloud", "model": "cloud-or-claude"}}

data: {"thinking": "Let me think about how to explain recursion..."}

data: {"choices": [{"delta": {"content": "Recursion "}}]}
data: {"choices": [{"delta": {"content": "is "}}]}
data: {"choices": [{"delta": {"content": "when a function calls itself."}}]}

data: {"stream_metadata": {"cost": {"total": 0.003}, "duration": 1.2}}

data: [DONE]
```

Each `data:` line is a separate event. The frontend parses them one by one
and updates the UI in real time — that's what creates the "typing" effect.

---

## 6. Step 3: The Backend Receives the Request

### Where it happens

`stream/middleware/routes/chat.py` → `chat_completions()` endpoint

### Request parsing

The request body is validated against a Pydantic model called
`ChatCompletionRequest`. The key fields for cloud routing:

```python
class ChatCompletionRequest(BaseModel):
    model: str                              # "cloud" (the tier)
    cloud_provider: str | None = None       # "cloud-or-claude" (the specific model)
    openrouter_api_key: str | None = None   # User's OpenRouter key
    anthropic_api_key: str | None = None    # User's Anthropic key (direct mode)
    openai_api_key: str | None = None       # User's OpenAI key (direct mode)
    messages: list[dict]                    # The conversation history
    temperature: float = 0.7
    stream: bool = True
```

### API key extraction

The endpoint builds a `user_api_keys` dictionary that maps environment variable
names to key values. This standardized format lets the rest of the pipeline
work the same way regardless of which provider's key was sent:

```python
user_api_keys: dict[str, str] = {}

if request_body.openrouter_api_key:
    user_api_keys["OPENROUTER_API_KEY"] = request_body.openrouter_api_key

if request_body.anthropic_api_key:
    user_api_keys["ANTHROPIC_API_KEY"] = request_body.anthropic_api_key

if request_body.openai_api_key:
    user_api_keys["OPENAI_API_KEY"] = request_body.openai_api_key
```

Why map to environment variable names like `OPENROUTER_API_KEY`? Because
`CLOUD_PROVIDERS` in config.py uses `env_key` to specify which environment
variable each model needs. By using the same naming convention, the key
injection code in litellm_direct.py can look up the right key with one
dictionary lookup.

### Tier and model determination

The endpoint calls two routing functions:

```python
tier = get_tier_for_query(...)           # Returns "cloud" (based on user selection)
model = get_model_for_tier(              # Returns the cloud_provider as-is
    tier="cloud",
    cloud_provider="cloud-or-claude"
)
# model = "cloud-or-claude"
```

`get_model_for_tier()` in `query_router.py` is simple for cloud: if the user
specified a `cloud_provider`, it returns it directly. This is because the real
model resolution happens later in `litellm_direct.py`.

### Handing off to the streaming pipeline

Finally, `chat.py` creates the streaming response, passing through the model
name, messages, and user API keys:

```python
return StreamingResponse(
    create_streaming_response(
        model="cloud-or-claude",
        messages=[...],
        temperature=0.7,
        user_api_keys={"OPENROUTER_API_KEY": "sk-or-v1-abc123..."},
        cloud_provider="cloud-or-claude",
        ...
    ),
    media_type="text/event-stream"
)
```

---

## 7. Step 4: Model Resolution — The Key Translation Step

### Where it happens

`stream/middleware/core/litellm_direct.py` → `_resolve_model()` function

### The problem this solves

The frontend sends a friendly name like `cloud-or-claude`. But LiteLLM
(the library that actually calls the AI provider) needs a specific model
identifier like `openrouter/anthropic/claude-sonnet-4`. Something needs
to translate between the two.

### How it works for curated models

When the backend starts, `litellm_direct.py` loads `litellm_config.yaml`
and builds an in-memory lookup table called `_MODEL_MAP`:

```yaml
# litellm_config.yaml — what's on disk
- model_name: cloud-or-claude
  litellm_params:
    model: openrouter/anthropic/claude-sonnet-4
    api_key: os.environ/OPENROUTER_API_KEY
```

```python
# _MODEL_MAP — what's in memory after loading
{
    "cloud-or-claude": {
        "model": "openrouter/anthropic/claude-sonnet-4",
        "api_key": "os.environ/OPENROUTER_API_KEY"
    },
    "cloud-or-gpt4o": {
        "model": "openrouter/openai/gpt-4o",
        "api_key": "os.environ/OPENROUTER_API_KEY"
    },
    ...
}
```

When `_resolve_model("cloud-or-claude")` is called:
1. It finds `"cloud-or-claude"` in `_MODEL_MAP`
2. Returns `{"model": "openrouter/anthropic/claude-sonnet-4"}`
3. This becomes the `model` argument to `litellm.acompletion()`

### How it works for dynamic models (NOT in the config)

This is where things get interesting. When a user picks a model from the
catalog browser (say, "openai/gpt-5"), the frontend sets:

```
cloudProvider = "cloud-or-dynamic-openai/gpt-5"
```

This model ID is NOT in `litellm_config.yaml`. It was discovered at runtime
from OpenRouter's catalog API. So `_resolve_model()` won't find it in
`_MODEL_MAP`.

Here's the fallback logic:

```python
def _resolve_model(friendly_name: str) -> dict:
    if friendly_name not in _MODEL_MAP:
        # Not a curated model — is it a dynamic catalog model?
        if friendly_name.startswith("cloud-or-dynamic-"):
            # Strip the prefix to get the OpenRouter model ID
            openrouter_model_id = friendly_name.removeprefix("cloud-or-dynamic-")
            # → "openai/gpt-5"

            # Prepend "openrouter/" so LiteLLM routes to OpenRouter
            return {"model": f"openrouter/{openrouter_model_id}"}
            # → {"model": "openrouter/openai/gpt-5"}

        # Unknown model — raise an error
        raise ValueError(f"Unknown model: {friendly_name}")
```

This is how STREAM supports 300+ models without adding 300+ entries to the
YAML config file. The `cloud-or-dynamic-` prefix is a convention that tells
the system: "I don't know this model from my config — construct the LiteLLM
call dynamically."

### The full prefix convention

```
cloud-or-claude           →  Curated OpenRouter model (in litellm_config.yaml)
cloud-or-gpt4o            →  Curated OpenRouter model (in litellm_config.yaml)
cloud-or-dynamic-X        →  Dynamic catalog model (NOT in litellm_config.yaml)
cloud-claude              →  Direct Anthropic model (in litellm_config.yaml)
cloud-gpt                 →  Direct OpenAI model (in litellm_config.yaml)
```

---

## 8. Step 5: API Key Injection

### Where it happens

`stream/middleware/core/litellm_direct.py` → `forward_direct()` function,
in the "USER API KEY INJECTION" section

### The problem this solves

The user's API key was sent in the HTTP request body and extracted into
`user_api_keys` by chat.py. Now we need to get it into the actual LiteLLM
call. LiteLLM accepts an `api_key` keyword argument that overrides any
environment variable.

### How it works for curated models

```python
if user_api_keys and model.startswith("cloud"):
    # Step 1: Look up the model in CLOUD_PROVIDERS to find which env var it needs
    provider_info = CLOUD_PROVIDERS.get(model)  # e.g., CLOUD_PROVIDERS["cloud-or-claude"]
    # → {"name": "Claude Sonnet 4", "env_key": "OPENROUTER_API_KEY", ...}

    if provider_info:
        env_key_name = provider_info.get("env_key")  # → "OPENROUTER_API_KEY"

        # Step 2: Check if the user provided a key for this env var
        user_key = user_api_keys.get(env_key_name)  # → "sk-or-v1-abc123..."

        if user_key:
            # Step 3: Inject it into the LiteLLM call
            kwargs["api_key"] = user_key
```

The lookup chain:
1. Model name → `CLOUD_PROVIDERS` dict → `env_key` field → env var name
2. Env var name → `user_api_keys` dict → actual key value
3. Key value → `kwargs["api_key"]` → passed to `litellm.acompletion()`

### How it works for dynamic models

Dynamic models (from the catalog) aren't in `CLOUD_PROVIDERS`, so the lookup
above returns `None`. There's a fallback:

```python
elif model.startswith("cloud-or-dynamic-"):
    # Dynamic models always use the OpenRouter key
    user_key = user_api_keys.get("OPENROUTER_API_KEY")
    if user_key:
        kwargs["api_key"] = user_key
```

This works because every model in the OpenRouter catalog is accessed through
OpenRouter's API, so they all need the OpenRouter key.

### What if no user key is provided?

If `user_api_keys` is empty (e.g., the user didn't enter a key in settings),
the `kwargs["api_key"]` is never set. In that case, LiteLLM falls back to
reading the key from the environment variable (e.g., `OPENROUTER_API_KEY`
set in the server's `.env` file). This is how server-managed deployments work
— the admin sets the key once, and all users share it.

---

## 9. Step 6: The LiteLLM Call

### Where it happens

`stream/middleware/core/litellm_direct.py` → `forward_direct()` function

### What happens

After model resolution and key injection, the actual AI call is made:

```python
response = await litellm.acompletion(
    model="openrouter/anthropic/claude-sonnet-4",
    messages=[{"role": "user", "content": "Explain recursion"}],
    temperature=0.7,
    stream=True,
    api_key="sk-or-v1-abc123...",           # User's key (or env var)
    # max_tokens=8192,                       # Only set for curated models
    # reasoning_effort="low",                # Only for reasoning models with direct providers
)
```

### How LiteLLM routes to the right provider

LiteLLM uses the model name prefix to determine where to send the request:

```
"openrouter/anthropic/claude-sonnet-4"  →  https://openrouter.ai/api/v1
"claude-sonnet-4-20250514"              →  https://api.anthropic.com
"gpt-4o"                                →  https://api.openai.com/v1
"ollama/llama3.2"                       →  http://localhost:11434
```

For OpenRouter models, LiteLLM:
1. Strips the `openrouter/` prefix to get `anthropic/claude-sonnet-4`
2. Sends it to OpenRouter's API with the user's key
3. OpenRouter recognizes `anthropic/claude-sonnet-4` and routes to Anthropic
4. Anthropic generates the response, which flows back through OpenRouter

### max_tokens: curated vs. dynamic

For **curated models** (like `cloud-or-claude`), we set `max_tokens` from
`MODEL_CONTEXT_LIMITS` in config.py. This prevents the model from reserving
too much output capacity, which would cause OpenRouter to pre-deduct more
credits than needed.

For **dynamic models** (like `cloud-or-dynamic-openai/gpt-5`), we intentionally
do NOT set `max_tokens`. The model uses its full output capacity. This is
because we don't know the right limit for every possible model in the catalog.

```python
if model.startswith("cloud-or-") and not model.startswith("cloud-or-dynamic-"):
    limits = MODEL_CONTEXT_LIMITS.get(model)
    if limits:
        kwargs["max_tokens"] = limits["reserve_output"]
```

---

## 10. Step 7: The Response Streams Back

### Where it happens

`stream/middleware/core/streaming.py` → `create_streaming_response()` function

### How SSE streaming works

The response comes back as a stream of Server-Sent Events (SSE). Each event
is a line starting with `data: ` followed by JSON:

```
data: {"choices": [{"delta": {"content": "Recursion "}}]}
data: {"choices": [{"delta": {"content": "is "}}]}
data: {"choices": [{"delta": {"content": "when "}}]}
...
data: [DONE]
```

`litellm_direct.py` yields these lines one by one. `streaming.py` consumes
them, adds metadata (cost, duration, tier info), and forwards them to the
frontend. Think of `streaming.py` as a pipeline processor — it sits between
the raw LiteLLM output and the browser.

### What streaming.py does with each line

For every SSE line, streaming.py:

1. **Checks for [DONE]** — Holds it back, sends cost metadata first, then [DONE]
2. **Intercepts internal events** — `stream_verified_model` (consumed, not forwarded)
3. **Extracts thinking content** — Forwards `{"thinking": ...}` events as-is
4. **Extracts reasoning from chunks** — In server mode, reasoning content is
   embedded in `delta.reasoning_content`; streaming.py extracts it and emits
   a separate `{"thinking": ...}` event
5. **Forwards content chunks** — Regular `{"choices": ...}` chunks pass through
6. **Tracks token usage** — Parses `usage` objects for cost calculation
7. **Sends metadata** — Initial (tier, model), fallback (if tier changed),
   final (cost, duration)

### The SSE event types the frontend receives

```
1. METADATA (initial):     {"stream_metadata": {"tier": "cloud", "model": "cloud-or-claude"}}
2. THINKING (reasoning):   {"thinking": "Let me think about this..."}
3. CONTENT (tokens):       {"choices": [{"delta": {"content": "word "}}]}
4. METADATA (fallback):    {"stream_metadata": {"fallback": true, "current_tier": "cloud"}}
5. METADATA (final):       {"stream_metadata": {"cost": {"total": 0.003}, "duration": 1.2}}
6. DONE:                   [DONE]
```

---

## 11. Step 8: The Frontend Renders the Stream

### Where it happens

`frontends/react/src/api/stream.ts` → SSE parsing loop
`frontends/react/src/components/chat/ChatContainer.tsx` → state management

### How stream.ts parses events

The `streamChat()` function reads the SSE stream using the browser's
ReadableStream API and parses each line:

```typescript
const parsed = JSON.parse(data)

// 1. Metadata events (tier info, cost, fallback notifications)
if (parsed.stream_metadata) {
    callbacks.onMetadata(parsed.stream_metadata)
}

// 2. Thinking content (reasoning models only)
if (parsed.thinking) {
    callbacks.onThinking(parsed.thinking)
}

// 3. Regular content tokens
const content = parsed.choices?.[0]?.delta?.content
if (content) {
    callbacks.onToken(content)
}
```

### How ChatContainer.tsx uses the callbacks

```
onToken("Recursion ")    → appendToken() → currentResponse += "Recursion "
onToken("is ")           → appendToken() → currentResponse += "is "
onThinking("Let me...")  → appendThinking() → currentThinking += "Let me..."
onMetadata({tier, cost}) → setMetadata() → updates UI header
onComplete()             → finishStreaming() → saves message to history
```

The `currentResponse` builds up word by word, creating the typing effect.
`currentThinking` builds up separately and is displayed in a collapsible
"Thought process" block above the response.

---

## 12. Desktop Mode vs. Server Mode

STREAM can run in two modes, and the cloud routing works slightly differently
in each:

### Desktop mode

```
Frontend → chat.py → streaming.py → litellm_client.py → litellm_direct.py → litellm.acompletion()
                                                              ↑
                                                     Direct library call
                                                     (same Python process)
```

In desktop mode, there's no separate LiteLLM server. The Python library is
called directly via `litellm.acompletion()`. This is like cooking at home —
you use the cookbook (litellm library) directly.

`litellm_client.py` checks `STREAM_MODE == "desktop"` and delegates to
`litellm_direct.py`'s `forward_direct()` function, which:
- Resolves model names via `_resolve_model()`
- Injects user API keys
- Calls `litellm.acompletion()` directly
- Yields SSE lines from the streamed response

### Server mode (Docker)

```
Frontend → chat.py → streaming.py → litellm_client.py → HTTP POST → LiteLLM Server (:4000) → Provider API
                                          ↑
                                 HTTP client (httpx)
                                 to a separate server
```

In server mode, LiteLLM runs as a separate HTTP server on port 4000. The
backend sends HTTP requests to it, like ordering at a restaurant — you send
your order (HTTP request) to the kitchen (LiteLLM server).

`litellm_client.py` sends an HTTP POST to `http://litellm:4000/chat/completions`
with the model name and messages. The LiteLLM server handles model resolution
using its own `litellm_config.yaml` and responds with an SSE stream.

### Why does this matter?

The output format is identical in both modes — SSE lines with the same JSON
structure. `streaming.py` doesn't know or care which mode is active. But there
are differences in how features like reasoning and model verification work:

- **Desktop mode**: `litellm_direct.py` extracts reasoning content directly
  from the LiteLLM response chunks and emits `{"thinking": ...}` events.
  It also emits `{"stream_verified_model": ...}` events.

- **Server mode**: The LiteLLM server returns raw chunks. `streaming.py`
  extracts `delta.reasoning_content` from the chunks and creates the
  `{"thinking": ...}` events itself. Verified model info comes from the
  `model` field in each chunk.

---

## 13. The Dynamic Model Catalog

### The problem

We want users to browse and select from 300+ models. But we can't put 300+
entries in `litellm_config.yaml` — that would be a maintenance nightmare,
and new models appear on OpenRouter weekly.

### The solution: runtime discovery

```
OpenRouter API                    STREAM Backend                  STREAM Frontend
───────────────                  ───────────────                 ────────────────

GET /api/v1/models          →   GET /v1/models/catalog     →   fetchModelCatalog()
Returns 300+ models              Proxies + caches 1 hour        Caches 5 minutes
with pricing info                Categorizes (recommended,       Displays in
                                 free, by provider)              searchable modal
```

### Backend: the catalog proxy

`stream/middleware/routes/models.py` has a `GET /v1/models/catalog` endpoint.

Why proxy through the backend instead of calling OpenRouter directly from the
browser?

1. **CORS**: Browsers block cross-origin requests. The frontend at
   `localhost:5173` can't call `openrouter.ai` directly.
2. **Caching**: One user's fetch benefits everyone for 1 hour.
3. **Privacy**: The user's API key isn't sent to a third-party domain from the
   browser's network inspector.

### Frontend: the catalog browser

`SettingsPanel.tsx` shows the catalog in a modal with:
- **Search**: Filter by model name or provider
- **Category filters**: Recommended, Free, Multimodal, Provider-specific
- **Pricing info**: Input/output cost per million tokens
- **Context window**: Maximum token capacity
- **Favorites**: Star icon to pin models for quick access

When a user selects a catalog model:

```
Catalog model ID:   "anthropic/claude-sonnet-4"
Stored as:          "cloud-or-dynamic-anthropic/claude-sonnet-4"
                     ↑
                     The "cloud-or-dynamic-" prefix signals to the backend
                     that this model is NOT in litellm_config.yaml
```

### API key validation

`models.py` also has a `POST /v1/health/validate-key` endpoint. When the user
enters their API key in the settings panel, it's validated with a minimal
1-token test call to the cheapest available model. The frontend shows a
spinner, then a checkmark (valid) or X (invalid) next to the key input.

---

## 14. Tier Fallback: What Happens When a Tier Fails

### Where it happens

`stream/middleware/core/streaming.py` → retry loop in `create_streaming_response()`

### The fallback chain

If a tier fails (e.g., Ollama is not running, or Lakeshore is down), STREAM
automatically tries the next tier:

```
LOCAL  →  fails  →  LAKESHORE  →  fails  →  CLOUD
```

The user sees a notification: "Lakeshore unavailable, falling back to Cloud"

### What triggers fallback

- **Connection errors**: Service unreachable (Ollama not running, Lakeshore down)
- **Timeout**: No response within the tier's timeout threshold
- **Model errors**: Requested model not available on the tier

### What does NOT trigger fallback

- **Billing errors** (402): The user needs to add credits — falling back won't help
- **Auth errors** (401/403): Invalid API key — falling back won't help
- **Rate limit errors** (429): Temporary, but user should wait — no fallback

These errors are returned directly to the frontend with specific error types
so the UI can show appropriate messages (see Error Handling section).

### Fallback and cloud_provider

When falling back TO the cloud tier, the user's selected `cloud_provider` is
used. So if the user selected "Claude Sonnet 4" as their cloud model but
started the query on Local, and Local fails, the fallback will use Claude
Sonnet 4 (not a random default).

---

## 15. Reasoning/Thinking Content

### What it is

Some advanced AI models have a "thinking" phase where they reason through the
problem before producing output. Claude Sonnet/Opus 4, OpenAI o1/o3, and
DeepSeek R1 are examples of reasoning models.

The thinking content looks like:
```
"Let me consider how to explain recursion...
A recursive function calls itself, so I should start with the base case...
I'll use a factorial example since it's the clearest..."
```

This is displayed in a collapsible "Thought process" block in the chat UI,
similar to how Claude.ai and ChatGPT show reasoning.

### How thinking content flows through the system

The flow differs slightly between desktop and server mode:

**Desktop mode:**

```
litellm.acompletion() returns chunks with delta.reasoning_content
         │
         ▼
litellm_direct.py extracts reasoning_content from the delta
         │  Uses delta.pop("reasoning_content") — .pop() removes it
         │  from the chunk to prevent double extraction downstream
         │
         ├→ Yields: data: {"thinking": "Let me think..."}
         └→ Yields: data: {"choices": [...]}  (chunk WITHOUT reasoning_content)
                     │
                     ▼
streaming.py sees {"thinking": ...} → forwards to frontend as-is
streaming.py sees {"choices": [...]} → no reasoning_content to extract
         │                             (already removed by .pop())
         ▼
Frontend receives ONE copy of thinking content ✓
```

The `.pop()` is critical. Before we used `.pop()`, we used `.get()`, which left
`reasoning_content` in the chunk. Then streaming.py would extract it AGAIN,
causing every word to appear twice ("Considering Considering need need for for...").

**Server mode:**

```
LiteLLM Server returns chunks with delta.reasoning_content
         │
         ▼
litellm_client.py forwards raw SSE lines (no extraction)
         │
         ▼
streaming.py extracts delta.reasoning_content from each chunk
         │
         ├→ Yields: data: {"thinking": "Let me think..."}
         └→ Yields: data: {"choices": [...]}  (original chunk, untouched)
         │
         ▼
Frontend receives ONE copy of thinking content ✓
```

### Reasoning effort parameter

For models accessed via **direct provider API keys** (not through OpenRouter),
we send `reasoning_effort="low"` to LiteLLM. This tells the provider to enable
reasoning output.

We do NOT send this for OpenRouter models because LiteLLM's internal validation
rejects the `reasoning_effort` parameter for the `openrouter/` prefix (it raises
`UnsupportedParamsError`). OpenRouter handles reasoning natively — if the
underlying model supports it, reasoning content flows through automatically.

### Which models support reasoning

Defined in `config.py`:

```python
REASONING_MODEL_PATTERNS = [
    "claude-sonnet-4", "claude-opus", "claude-4",
    "o1", "o3", "o4",
    "deepseek-r1", "deepseek/deepseek-r1",
]
```

The `is_reasoning_model()` function checks if a model name contains any of
these patterns (case-insensitive).

---

## 16. Model Verification

### The problem

When a user selects "Claude Opus 4" via OpenRouter, how do we know that
Claude Opus 4 is actually responding? OpenRouter might route to a different
model, or the model might not exist. We need to verify.

### How it works

The first chunk of every streaming response contains a `model` field that
tells us what model the provider actually used:

```json
{
  "model": "anthropic/claude-opus-4-20250514",
  "choices": [{"delta": {"content": ""}}]
}
```

In **desktop mode**, `litellm_direct.py` extracts this from the first chunk
and emits it as a special SSE event:

```python
if first_chunk:
    response_model = chunk_dict.get("model", "unknown")
    yield f"data: {json.dumps({'stream_verified_model': response_model})}"
    first_chunk = False
```

`streaming.py` intercepts this event (it does NOT forward it to the frontend).
Instead, it stores the verified model internally for logging and diagnostics.

In **server mode**, there's no separate `stream_verified_model` event. Instead,
`streaming.py` reads the `model` field directly from the regular content chunks.

The frontend shows the verified model name next to the response, so the user
can confirm which model actually answered.

---

## 17. Error Handling

### Error types and what the user sees

| Error | HTTP Status | Cause | User sees | Fallback? |
|-------|------------|-------|-----------|-----------|
| Auth error | 401/403 | Invalid API key or expired subscription | "API key invalid" with link to fix | No |
| Billing error | 402 | No credits / spending limit reached | "Increase your credit limit on OpenRouter" | No |
| Rate limit | 429 | Too many requests too fast | "Rate limited, try again in a moment" | No |
| Context too long | 400 | Conversation exceeds model's context window | Dialog showing token count vs. limit | No |
| Connection error | 503 | Provider unreachable | Automatic fallback to next tier | Yes |
| Model not found | 404 | Model ID doesn't exist | "Model not available" | Yes |

### Error classification in the backend

`litellm_direct.py` catches LiteLLM exceptions and translates them to HTTP
errors with structured details:

```python
except litellm.AuthenticationError:
    raise HTTPException(status_code=401, detail={
        "error_type": "auth_subscription",
        "message": "Your API key is invalid or expired",
        "provider": "OpenRouter",
    })

except litellm.RateLimitError:
    raise HTTPException(status_code=429, detail={
        "error_type": "rate_limit",
        "message": "Rate limited. Wait a moment and try again.",
    })
```

### Billing errors: a special case

OpenRouter billing errors (402) are tricky because they can look like context
length errors. The error message might say "not enough credits to process
65536 tokens" — which mentions "tokens" and could be misclassified.

The backend explicitly checks for billing keywords before context keywords:

```python
is_billing = "credits" in msg or "afford" in msg or "payment required" in msg
is_context = not is_billing and ("context" in msg or "too long" in msg)
```

This ensures billing errors show the correct message ("add credits") instead
of the wrong one ("conversation too long").

---

## 18. API Key Management and Security

### Where keys live

```
Frontend (browser):
  localStorage["stream-settings"] = {
    openrouterApiKey: "sk-or-v1-...",
    anthropicApiKey: "sk-ant-...",
    openaiApiKey: "sk-..."
  }

Backend (server):
  .env file or environment variables:
    OPENROUTER_API_KEY=sk-or-v1-...
    ANTHROPIC_API_KEY=sk-ant-...
    OPENAI_API_KEY=sk-...
```

### Key priority

User-provided keys (from the frontend) override server environment variables.
If the user enters a key in the settings panel, it's used. If they don't,
the backend falls back to whatever is in the environment.

### Security design

1. **Keys never stored server-side**: User keys live in their browser's
   `localStorage` and are sent per-request in the POST body. If the server
   is compromised, no user keys are exposed.

2. **Keys sent only when needed**: The frontend only includes the key for
   the selected provider. If you're using OpenRouter, your Anthropic key
   isn't sent.

3. **Keys not logged**: The backend's debug logging explicitly excludes
   `api_key` from the logged kwargs:
   ```python
   safe_kwargs = {k: v for k, v in kwargs.items() if k != "api_key"}
   ```

4. **Validation without spending**: Key validation uses a 1-token test call
   to the cheapest model, costing effectively $0.

---

## 19. All Major Aggregator Services Compared

We evaluated 10 services before choosing OpenRouter:

| Provider | Proprietary Models | Open-Source Models | Free Tier | Best For |
|----------|-------------------|-------------------|-----------|----------|
| **OpenRouter** | GPT, Claude, Gemini | Llama, Mistral, etc. | 18-31 free models | Maximum variety, single key |
| **Together AI** | None | 200+ | $25 credits | Open-source + fine-tuning |
| **Groq** | None | ~15 | 1K req/day | Fastest inference |
| **Fireworks AI** | None | Dozens | 2.5B tok/day | Most generous free tier |
| **DeepInfra** | None | Dozens | Trial only | Cheapest per-token |
| **Cerebras** | None | ~10 | Yes | Ultra-fast inference |
| **Amazon Bedrock** | Claude, GPT, Titan | Llama, Mistral | AWS Free Tier | Enterprise / AWS shops |
| **Azure AI** | GPT, Claude | 1,900+ catalog | $100 student | Microsoft ecosystem |
| **Vertex AI** | Gemini, Claude | Llama, Qwen | $300 new account | Google ecosystem |
| **Replicate** | None | Thousands | Limited | Multimodal (image/video) |

Key observation: **Only OpenRouter and the cloud hyperscalers (AWS/Azure/GCP)
offer both proprietary AND open-source models through one API.** The others
are open-source only.

---

## 20. Why OpenRouter?

1. **Single key for all models.** One API key gives access to GPT, Claude,
   Gemini, Llama, Mistral, and 300+ more. This is the biggest UX improvement.

2. **LiteLLM native support.** Prefix any model with `openrouter/` and it
   works. Minimal code changes.

3. **Free models.** Students can use STREAM's cloud tier without spending
   money. Free models include Llama, Mistral, Gemma, and others.

4. **Transparent pricing.** OpenRouter's API returns per-model pricing. Users
   see costs before selecting a model.

5. **No commitment.** Pay-as-you-go. No subscriptions. No minimum spend.

6. **Preserves direct keys.** Users who already have OpenAI or Anthropic keys
   can still use them directly. OpenRouter is an addition, not a replacement.

### OpenRouter-specific details

```
API Key format:   sk-or-v1-<64 hex characters>
Base URL:         https://openrouter.ai/api/v1
Auth header:      Authorization: Bearer sk-or-v1-...
Model naming:     provider/model-name  (e.g., anthropic/claude-sonnet-4)
Pricing:          5.5% platform fee on credit purchases
Free models:      Suffixed with ":free" (e.g., meta-llama/llama-3.1-8b-instruct:free)
```

---

## 21. File Reference

Every file involved in cloud routing, and what it does:

### Frontend

| File | Role |
|------|------|
| `frontends/react/src/components/sidebar/SettingsPanel.tsx` | UI for model selection, API key input, catalog browser, favorites |
| `frontends/react/src/stores/settingsStore.ts` | Zustand store: persists `cloudProvider`, API keys, favorites to localStorage |
| `frontends/react/src/api/stream.ts` | Makes POST request with cloud_provider + API key, parses SSE events |
| `frontends/react/src/api/models.ts` | Fetches model catalog, validates API keys, caches results |
| `frontends/react/src/types/settings.ts` | `CloudProvider` type (string), known provider constants |
| `frontends/react/src/components/chat/ChatContainer.tsx` | Manages streaming state: currentResponse, currentThinking |
| `frontends/react/src/components/chat/ThinkingBlock.tsx` | Collapsible UI for reasoning/thinking content |
| `frontends/react/src/components/chat/Message.tsx` | Renders messages, shows verified model, thinking block |
| `frontends/react/src/components/icons/ProviderLogos.tsx` | Model logos/icons, provider registry, model-to-provider mapping |

### Backend

| File | Role |
|------|------|
| `stream/middleware/routes/chat.py` | HTTP endpoint, extracts API keys, creates streaming response |
| `stream/middleware/core/streaming.py` | Orchestrates SSE stream, tier fallback, cost tracking, thinking extraction |
| `stream/middleware/core/litellm_direct.py` | Desktop mode: model resolution, key injection, direct LiteLLM calls |
| `stream/middleware/core/litellm_client.py` | Server mode: HTTP forwarding to LiteLLM server; delegates to litellm_direct.py in desktop mode |
| `stream/middleware/core/query_router.py` | `get_model_for_tier()`: returns model name for the selected cloud provider |
| `stream/middleware/config.py` | `CLOUD_PROVIDERS` dict, `is_reasoning_model()`, context limits, constants |
| `stream/gateway/litellm_config.yaml` | Maps friendly names → LiteLLM model params (curated models only) |
| `stream/middleware/routes/models.py` | Catalog proxy endpoint, API key validation endpoint |
| `stream/middleware/core/tier_health.py` | Health checks per provider, caches results |
| `stream/middleware/routes/health.py` | `/health/cloud-providers` endpoint |

### Tests

| File | Role |
|------|------|
| `tests/test_openrouter.py` | 42 tests: config, API key threading, dynamic models, catalog, context |
| `tests/test_reasoning.py` | 34 tests: reasoning detection, thinking extraction, desktop/server mode |
