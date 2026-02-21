# STREAM Cost Calculation Architecture

## Table of Contents

1. [Overview](#1-overview)
2. [Cost Flow Architecture](#2-cost-flow-architecture)
3. [File Details](#3-file-details)
4. [Model Pricing](#4-model-pricing)
5. [Cloud Cost Sources](#5-cloud-cost-sources)
6. [Token Tracking](#6-token-tracking)
7. [Context Window Limits](#7-context-window-limits)
8. [Key Design Decisions](#8-key-design-decisions)

---

## 1. Overview

STREAM uses a **single source of truth** architecture for cost calculation. The middleware calculates costs in real-time during streaming and sends them to the frontend. The frontend displays costs without any independent calculation.

**Key Principle**: If the middleware doesn't return a cost, something is wrong and should be fixed. There is no fallback estimation in the frontend.

---

## 2. Cost Flow Architecture

```
+-----------------------------------------------------------------------------+
|                           COST CALCULATION FLOW                             |
+-----------------------------------------------------------------------------+
|                                                                             |
|   1. DEFINITION                                                             |
|   +------------------------------------------+                              |
|   |  stream/gateway/litellm_config.yaml      | <-- Pricing defined here     |
|   |  - input_cost_per_token: 0.000003        |                              |
|   |  - output_cost_per_token: 0.000015       |                              |
|   +--------------------+---------------------+                              |
|                        |                                                    |
|                        v                                                    |
|   2. READING                                                                |
|   +------------------------------------------+                              |
|   |  stream/middleware/utils/cost_reader.py  | <-- Reads & caches pricing   |
|   |  - load_model_pricing()                  |                              |
|   |  - get_model_cost(model)                 |                              |
|   +--------------------+---------------------+                              |
|                        |                                                    |
|                        v                                                    |
|   3. CALCULATION                                                            |
|   +------------------------------------------+                              |
|   | stream/middleware/utils/cost_calculator.py| <-- Calculates total cost   |
|   |  - calculate_query_cost(model, in, out)  |                              |
|   |  - Returns: input*rate + output*rate     |                              |
|   +--------------------+---------------------+                              |
|                        |                                                    |
|                        v                                                    |
|   4. STREAMING                                                              |
|   +------------------------------------------+                              |
|   |  stream/middleware/core/streaming.py     | <-- Sends cost in SSE        |
|   |  Line 249: cost = calculate_query_cost() |                              |
|   |  Line 261-265: sends cost in response    |                              |
|   +--------------------+---------------------+                              |
|                        |                                                    |
|                        v                                                    |
|   5. SDK                                                                    |
|   +------------------------------------------+                              |
|   |  stream/sdk/python/chat_handler.py       | <-- Extracts cost from stream|
|   |  - Parses stream_metadata                |                              |
|   |  - Returns cost to frontend              |                              |
|   +--------------------+---------------------+                              |
|                        |                                                    |
|                        v                                                    |
|   6. DISPLAY                                                                |
|   +------------------------------------------+                              |
|   |  frontends/streamlit/streamlit_app.py    | <-- Displays cost (no calc)  |
|   |  - Gets cost from stream_meta["cost"]    |                              |
|   |  - Accumulates in session_stats          |                              |
|   +------------------------------------------+                              |
|                                                                             |
+-----------------------------------------------------------------------------+
```

---

## 3. File Details

### 3.1 litellm_config.yaml - Pricing Definitions

**File**: [`stream/gateway/litellm_config.yaml`](../stream/gateway/litellm_config.yaml)

This is the **single source of truth** for all model pricing. Each model has `input_cost_per_token` and `output_cost_per_token` defined in `model_info`:

```yaml
model_list:
  - model_name: cloud-claude
    litellm_params:
      model: claude-sonnet-4-20250514
      api_key: os.environ/ANTHROPIC_API_KEY
    model_info:
      input_cost_per_token: 0.000003    # $3 per 1M input tokens
      output_cost_per_token: 0.000015   # $15 per 1M output tokens
```

### 3.2 cost_reader.py - Pricing Loader

**File**: [`stream/middleware/utils/cost_reader.py`](../stream/middleware/utils/cost_reader.py)

Reads pricing from `litellm_config.yaml` at startup and caches it in memory:

```python
def load_model_pricing() -> dict:
    """Load model pricing from LiteLLM config."""
    global _MODEL_PRICING
    if _MODEL_PRICING is not None:
        return _MODEL_PRICING  # Return cached pricing

    # Read and parse litellm_config.yaml
    config_path = Path(__file__).parent.parent.parent / "gateway" / "litellm_config.yaml"
    with open(config_path) as f:
        litellm_config = yaml.safe_load(f)

    # Extract pricing from model_list
    pricing = {}
    for model_def in litellm_config.get("model_list", []):
        model_name = model_def.get("model_name")
        model_info = model_def.get("model_info", {})
        pricing[model_name] = {
            "input": model_info.get("input_cost_per_token", 0.0),
            "output": model_info.get("output_cost_per_token", 0.0),
        }
    _MODEL_PRICING = pricing
    return pricing
```

### 3.3 cost_calculator.py - Cost Calculation

**File**: [`stream/middleware/utils/cost_calculator.py`](../stream/middleware/utils/cost_calculator.py)

Calculates cost using pricing from `cost_reader`:

```python
def calculate_query_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost for a query."""
    costs = get_model_cost(model)  # From cost_reader

    input_cost = input_tokens * costs["input"]
    output_cost = output_tokens * costs["output"]
    total_cost = input_cost + output_cost

    return total_cost
```

### 3.4 streaming.py - Cost in SSE Response

**File**: [`stream/middleware/core/streaming.py`](../stream/middleware/core/streaming.py) (Lines 248-269)

Calculates and sends cost at the end of streaming:

```python
# Calculate cost only once (using centralized cost_reader!)
cost = calculate_query_cost(current_model, input_tokens, output_tokens)

# Send final cost summary
cost_event = {
    "stream_metadata": {
        "tier": current_tier,
        "model": current_model,
        "cost": {
            "total": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }
}
yield f"data: {json.dumps(cost_event)}\n\n"
```

### 3.5 chat_handler.py - SDK Cost Extraction

**File**: [`stream/sdk/python/chat_handler.py`](../stream/sdk/python/chat_handler.py)

The SDK extracts cost from the SSE stream metadata and passes it to the frontend:

```python
# In _stream_response method
if "stream_metadata" in data:
    meta = data["stream_metadata"]
    if "cost" in meta:
        self._last_stream_metadata["cost"] = meta["cost"]["total"]
        self._last_stream_metadata["input_tokens"] = meta["cost"]["input_tokens"]
        self._last_stream_metadata["output_tokens"] = meta["cost"]["output_tokens"]
```

### 3.6 streamlit_app.py - Cost Display

**File**: [`frontends/streamlit/streamlit_app.py`](../frontends/streamlit/streamlit_app.py)

The frontend displays cost from metadata without any calculation:

```python
# Get cost from middleware (single source of truth)
cost = stream_meta.get("cost", 0.0)

# Update session stats
st.session_state.session_stats["total_cost"] += cost
```

---

## 4. Model Pricing

| Model | Input Cost (per token) | Output Cost (per token) | Tier |
|-------|------------------------|-------------------------|------|
| `local-llama` | $0.0 | $0.0 | Local |
| `local-vision` | $0.0 | $0.0 | Local |
| `lakeshore-qwen` | $0.0000005 | $0.0000005 | Lakeshore |
| `cloud-claude` | $0.000003 | $0.000015 | Cloud |
| `cloud-gpt` | $0.00001 | $0.00003 | Cloud |
| `cloud-gpt-cheap` | $0.0000005 | $0.0000015 | Cloud |

### Cost Per 1 Million Tokens

| Model | Input (per 1M) | Output (per 1M) | Tier |
|-------|----------------|-----------------|------|
| `local-llama-*` | $0.00 | $0.00 | Local |
| `lakeshore-qwen` | $0.0005 | $0.0005 | Lakeshore |
| `cloud-claude` | $3.00 | $15.00 | Cloud |
| `cloud-gpt` | $10.00 | $30.00 | Cloud |
| `cloud-gpt-cheap` | $0.50 | $1.50 | Cloud |

---

## 5. Cloud Cost Sources

The cloud model pricing is based on official provider pricing pages:

### Anthropic (Claude)

| Model | Input | Output | Source |
|-------|-------|--------|--------|
| Claude Sonnet 4 | $3/1M tokens | $15/1M tokens | [anthropic.com/pricing](https://www.anthropic.com/pricing) |

*Note: Pricing as of January 2025. Check the official page for current rates.*

### OpenAI (GPT)

| Model | Input | Output | Source |
|-------|-------|--------|--------|
| GPT-4 Turbo | $10/1M tokens | $30/1M tokens | [openai.com/pricing](https://openai.com/pricing) |
| GPT-3.5 Turbo | $0.50/1M tokens | $1.50/1M tokens | [openai.com/pricing](https://openai.com/pricing) |

*Note: Pricing as of January 2025. Check the official page for current rates.*

---

## 6. Token Tracking

Token counts are obtained from two methods:

### Method 1: From LLM Response (Primary)

**File**: [`stream/middleware/core/streaming.py`](../stream/middleware/core/streaming.py) (Lines 194-206)

LiteLLM includes `usage` in the stream:

```python
if "usage" in data and data["usage"]:
    usage = data["usage"]
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
```

### Method 2: Estimation Fallback

**File**: [`stream/middleware/core/streaming.py`](../stream/middleware/core/streaming.py) (Lines 230-246)

If no usage data is provided by the LLM, estimate from text:

```python
if output_tokens == 0 and output_text:
    output_tokens = estimate_tokens_from_text(full_output)

if input_tokens == 0:
    input_tokens = estimate_tokens(messages)
```

The estimation uses `stream/middleware/utils/token_estimator.py` which applies the rule: **~4 characters per token**.

---

## 7. Context Window Limits

**File**: [`stream/middleware/config.py`](../stream/middleware/config.py) (Lines 116-126)

| Model | Total Tokens | Reserved for Output | Available for Input |
|-------|-------------|---------------------|---------------------|
| `local-llama-*` | 8,192 | 1,024 | ~7,168 |
| `lakeshore-qwen` | 8,192 | 500 | ~7,692 |
| `cloud-claude` | 200,000 | 4,000 | ~196,000 |
| `cloud-gpt` | 128,000 | 4,000 | ~124,000 |
| `cloud-gpt-cheap` | 16,385 | 1,000 | ~15,385 |

### Token to Word Conversion (Approximate)

- 1 token ~ 0.75 words (or ~4 characters)
- 1,000 tokens ~ 750 words
- 8,192 tokens ~ 6,100 words total context

---

## 8. Key Design Decisions

### 8.1 Single Source of Truth

All pricing is defined in `litellm_config.yaml`. This ensures:
- Consistency across all components
- Easy updates (change one file)
- No drift between middleware and frontend

### 8.2 No Frontend Cost Calculation

The frontend **only displays** costs received from middleware. It never calculates costs independently. This ensures:
- Accuracy (middleware has actual token counts)
- Simplicity (no duplicate logic)
- Debuggability (if cost is wrong, check middleware)

### 8.3 Cached at Startup

Pricing is loaded once when the middleware starts and cached in memory:
- Performance (no file I/O per request)
- Consistency (pricing doesn't change mid-session)
- Requires restart to update pricing

### 8.4 Real-time in SSE

Cost is calculated and sent at the end of each streaming response:
- Users see cost immediately after response completes
- Cost is tied to actual token usage, not estimates

### 8.5 LiteLLM for Analytics

LiteLLM also stores costs in PostgreSQL for historical tracking:

**File**: [`stream/middleware/routes/costs.py`](../stream/middleware/routes/costs.py)

```python
@router.get("/costs/summary")
async def get_cost_summary(days: int = 7):
    """Get usage and cost summary from LiteLLM_SpendLogs table."""
    cur.execute("""
        SELECT model, COUNT(*) as requests, SUM(spend) as total_cost
        FROM "LiteLLM_SpendLogs"
        WHERE "startTime" >= %s
        GROUP BY model
    """, (start_date,))
```

This provides:
- Historical cost tracking
- Per-model breakdowns
- Time-based analytics

---

## Summary

```
litellm_config.yaml  -->  cost_reader.py  -->  cost_calculator.py
     (defines)              (reads)              (calculates)
                                                      |
                                                      v
                                              streaming.py
                                              (sends in SSE)
                                                      |
                                                      v
                                              chat_handler.py
                                              (extracts)
                                                      |
                                                      v
                                              streamlit_app.py
                                              (displays)
```

**Remember**: The middleware is the authority for cost. If you see incorrect costs, check the middleware logs and `litellm_config.yaml`.

---

*This document is part of the STREAM project. Last updated: February 2025.*
