# Tier-Aware Rolling Context Summarization

## Solving the Context-Routing Mismatch in Multi-Tier LLM Systems

---

## Table of Contents

1. [The Problem: Context-Routing Mismatch](#1-the-problem-context-routing-mismatch)
2. [Why Existing Systems Don't Solve This](#2-why-existing-systems-dont-solve-this)
3. [Our Solution: Rolling Summarization with Differential Compression](#3-our-solution-rolling-summarization-with-differential-compression)
4. [How Rolling Summarization Works](#4-how-rolling-summarization-works)
5. [Tier-Aware Differential Compression](#5-tier-aware-differential-compression)
6. [Architecture and Data Flow](#6-architecture-and-data-flow)
7. [Summarization Strategy](#7-summarization-strategy)
8. [Multimodal Considerations](#8-multimodal-considerations)
9. [Feature Toggle and A/B Evaluation](#9-feature-toggle-and-ab-evaluation)
10. [Comparison with Prior Work](#10-comparison-with-prior-work)
11. [Limitations and Degradation](#11-limitations-and-degradation)
12. [Implementation Reference](#12-implementation-reference)
13. [Code File Reference](#13-code-file-reference)
14. [References](#14-references)

---

## 1. The Problem: Context-Routing Mismatch

### STREAM's routing promise

STREAM routes user queries to the cheapest capable tier based on query complexity:

- **LOCAL** (Ollama, 1-3B params): Simple questions — free, instant, private
- **Lakeshore** (HPC, 72B params): Complex questions — free, GPU-accelerated
- **Cloud** (Claude, GPT-4): Expert-level questions — paid, highest quality

At turn 1 of a conversation, a simple question like *"what is a for loop?"* routes to LOCAL. It costs nothing, responds instantly, and the answer is perfectly adequate. This is STREAM working as intended.

### The mismatch

At turn 25 of the same conversation, the user asks another simple question: *"remind me, what is a for loop?"* The query itself is trivially simple — LOCAL could answer it perfectly. But the conversation history now contains 24 previous exchanges totaling 8,000+ tokens. The full message array (system prompt + history + new query) exceeds LOCAL's context window.

STREAM's current behavior: **reject the request with an HTTP 400 error** if the context exceeds the model's limit. The user must either start a new conversation or the query gets routed to a more expensive tier.

This is the **context-routing mismatch**: the routing decision is correct (simple query → cheap tier), but the context window constraint prevents the cheap tier from handling it. The query complexity hasn't changed — only the accumulated history has grown.

### Why this matters

| Turn | Query | Complexity | Ideal Tier | Actual Tier (without summarization) |
|------|-------|-----------|------------|--------------------------------------|
| 1 | "what is a for loop?" | LOW | LOCAL (free) | LOCAL (free) |
| 10 | "what does `break` do?" | LOW | LOCAL (free) | LOCAL (free) |
| 25 | "remind me, what is a for loop?" | LOW | LOCAL (free) | **Error: context too long** |
| 25 | (same query, force-upgraded) | LOW | LOCAL (free) | **Cloud ($0.003)** |

Every forced upgrade from LOCAL to Cloud costs money and adds latency — for a question a 1-billion-parameter model could answer perfectly.

### The scale of the problem

In a typical multi-turn session:
- A student asking homework questions might have 20-30 exchanges
- A researcher iterating on code might have 40-50 exchanges
- Each exchange adds ~200-500 tokens (question + answer)
- By turn 30, raw history is ~9,000-15,000 tokens
- LOCAL's context window: 32,768 tokens (usable input: ~30,720 after output reserve)
- Lakeshore's context window: 65,536 tokens (usable input: ~61,440)
- Cloud's context window: 128,000-200,000 tokens

LOCAL can hold about 60-150 exchanges before exceeding its limit. Lakeshore can hold about 120-300. But some conversations with long code blocks, document uploads, or web search results consume context much faster.

---

## 2. Why Existing Systems Don't Solve This

### LLM routing systems ignore context length

The entire LLM routing literature treats routing as a **per-query decision** and ignores the accumulated context:

| System | Routes by | Considers context length? | Compresses context? |
|--------|-----------|--------------------------|---------------------|
| **RouteLLM** (Berkeley, 2024) | Query quality preference | No | No |
| **FrugalGPT** (Stanford, 2023) | Cost/quality cascade | No | No |
| **AutoMix** (NeurIPS 2024) | Self-verification confidence | No | No |
| **Hybrid LLM** (ICLR 2024) | Query difficulty | No | No |
| **STREAM** (this work) | Query complexity + modality | **Yes** | **Yes (differential)** |

These systems all assume the model can handle whatever context is provided. None of them address the scenario where a simple query fails on a small model due to long history.

### Context compression systems ignore routing

The context compression literature produces a single compressed output for a single target model:

| System | Compression method | Per-tier compression? | Routing-aware? |
|--------|-------------------|----------------------|----------------|
| **LLMLingua** (EMNLP 2023) | Token-level pruning | No | No |
| **LangChain SummaryBuffer** | Rolling summary | No | No |
| **Mem0** | Structured memory extraction | No | No |
| **MemGPT/Letta** | OS-inspired virtual memory | No | No |
| **ACON** (2025) | Failure-driven compression | No | No |
| **STREAM** (this work) | Rolling summarization | **Yes (3 levels)** | **Yes** |

These systems compress to a single representation. They don't produce different compression levels for different downstream models. A summary designed for a 200K-context model is unnecessarily lossy for that model, and a summary designed for a 4K-context model won't fit at all in the larger model's expectations.

### The gap STREAM fills

STREAM is the first system to **jointly optimize routing and context compression**:

```
route(query, history, tier_health, context_size) → (tier, compression_level)
```

The routing decision and the compression decision are coupled: the chosen tier determines how much compression is needed, and the available compression determines which tiers remain viable.

---

## 3. Our Solution: Rolling Summarization with Differential Compression

### Core idea

When a conversation's history exceeds a tier's context window threshold, STREAM compresses older messages into a summary while keeping recent messages raw. The compression level is determined by the target tier's context window:

- **LOCAL (32K context)**: Aggressive compression — up to 2K summary + last 3 exchanges
- **Lakeshore (64K context)**: Moderate compression — up to 4K summary + last 6 exchanges
- **Cloud (128-200K context)**: No compression — raw history fits

### The key insight

This means the **same conversation** can produce **three different message arrays** for three different tiers:

```
Turn 30 conversation (raw: ~10,000 tokens)

For LOCAL tier:
  [system] + [4K summary of turns 1-24] + [turns 25-30 raw]
  Total: ~5,500 tokens ✓ (fits in 32K)

For Lakeshore tier:
  [system] + [8K summary of turns 1-22] + [turns 23-30 raw]
  Total: ~10,500 tokens ✓ (fits in 64K)

For Cloud tier:
  [system] + [all 30 turns raw]
  Total: ~10,000 tokens ✓ (fits in 200K)
```

A simple query at turn 30 can still route to LOCAL because the compressed history fits within LOCAL's context window. The routing engine doesn't need to force-upgrade to an expensive tier.

---

## 4. How Rolling Summarization Works

### Message array structure

A typical STREAM conversation has this message array structure:

```json
[
  {"role": "system", "content": "You are a helpful assistant..."},     // System prompt
  {"role": "system", "content": "Web search results: ..."},            // Web search (optional)
  {"role": "user", "content": "What is Python?"},                      // Turn 1
  {"role": "assistant", "content": "Python is a programming..."},      // Turn 1 response
  {"role": "user", "content": "Show me a for loop"},                   // Turn 2
  {"role": "assistant", "content": "Here's a for loop..."},            // Turn 2 response
  ...
  {"role": "user", "content": "What is a list comprehension?"},        // Turn N (latest)
]
```

### The rolling window

Summarization splits this array into three segments:

```
┌──────────────┬─────────────────────────────┬───────────────────────┐
│   System     │       Old messages          │    Recent messages    │
│   messages   │    (to be summarized)       │    (kept raw)         │
│              │                             │                       │
│  role=system │  All non-system messages    │  Last N pairs of      │
│  (preserved) │  before the recent window   │  user + assistant     │
│              │                             │  messages             │
└──────────────┴─────────────────────────────┴───────────────────────┘
       ↓                    ↓                           ↓
    Keep as-is        Compress into              Keep as-is
                      summary message
```

### After summarization

```
┌──────────────┬─────────────────────────────┬───────────────────────┐
│   System     │   Summary message           │    Recent messages    │
│   messages   │   (single system msg)       │    (kept raw)         │
│              │                             │                       │
│  role=system │  "Previous conversation     │  Last N pairs of      │
│  (preserved) │   summary: The user asked   │  user + assistant     │
│              │   about Python basics..."   │  messages             │
└──────────────┴─────────────────────────────┴───────────────────────┘
```

### Example

**Before** (10 exchanges, ~3,500 tokens):
```
[system prompt]
[user: "What is Python?"]
[assistant: "Python is a high-level programming language..."]
[user: "What are variables?"]
[assistant: "Variables are named containers that store values..."]
[user: "Show me a for loop"]
[assistant: "A for loop iterates over a sequence..."]
[user: "What about while loops?"]
[assistant: "A while loop repeats as long as a condition..."]
[user: "How do functions work?"]
[assistant: "Functions are reusable blocks of code..."]
[user: "What is a list?"]
[assistant: "A list is an ordered, mutable collection..."]
[user: "How do I sort a list?"]                              ← Turn 7 (latest query)
```

**After summarization** (keep last 3 pairs, ~1,800 tokens):
```
[system prompt]
[system: "Previous conversation summary: The user is learning Python
  basics. Topics covered: Python as a high-level language, variables
  as named containers for values, for loops for iterating over
  sequences, and while loops for conditional repetition."]
[user: "How do functions work?"]                              ← Recent (raw)
[assistant: "Functions are reusable blocks of code..."]       ← Recent (raw)
[user: "What is a list?"]                                     ← Recent (raw)
[assistant: "A list is an ordered, mutable collection..."]    ← Recent (raw)
[user: "How do I sort a list?"]                               ← Latest query (raw)
```

The summary captures the essential context (learning Python, topics covered) while the recent messages preserve full detail for continuity.

---

## 5. Tier-Aware Differential Compression

### Compression parameters per tier

| Parameter | LOCAL | Lakeshore | Cloud |
|-----------|-------|-----------|-------|
| Context window (total) | 32,768 | 65,536 | 128,000-200,000 |
| Usable input (after output reserve) | ~30,720 | ~61,440 | ~124,000-196,000 |
| Summarization trigger | > 80% of usable input | > 80% of usable input | > 80% (rarely triggers) |
| Trigger threshold (tokens) | ~24,576 | ~49,152 | ~99,200-156,800 |
| Max summary length | 2,048 tokens | 4,096 tokens | 8,192 tokens |
| Recent pairs kept raw | 3 (6 messages) | 6 (12 messages) | 8 (16 messages) |
| Summarization enabled | Yes | Yes | No (default) |

### Why different compression levels?

**LOCAL (aggressive compression)**:
- Small model (1-3B params) — benefits from shorter, cleaner context
- Limited context window — must compress to fit
- Simple queries don't need full history — just key facts
- Summary capped at 2K tokens (actual summaries are usually 300-1,000 tokens)

**Lakeshore (moderate compression)**:
- Large model (72B params) — can use more context effectively
- Moderate context window (64K) — some compression needed for very long conversations
- Handles medium-complexity queries that may need more historical context
- Keeps 6 recent pairs (vs LOCAL's 3) because 72B benefits from more raw context
- Summary capped at 4K tokens

**Cloud (minimal/no compression)**:
- Largest models — most capable of using full context
- Massive context windows (128K-200K) — rarely exceeded
- Handles complex queries where full context matters most
- Only compress in extreme cases (very long sessions)

### The differential advantage

Consider a turn-50 conversation where the raw history is 20,000 tokens. A simple follow-up question routes to LOCAL:

**Without tier-aware compression**:
- 20,000 tokens > would need to fit in LOCAL's 30K input
- Currently works, but at turn 80 it would fail
- System has no mechanism to handle growth

**With tier-aware compression**:
- LOCAL gets: system prompt (~200) + ~1K summary + last 3 pairs (~1,500) = ~2,700 tokens
- Fits easily in 30K — LOCAL can handle simple queries indefinitely
- If the query is complex and routes to Lakeshore: ~2K summary + last 6 pairs = ~5,200 tokens
- If it routes to Cloud: full 20,000 tokens unchanged

---

## 6. Architecture and Data Flow

### Where summarization fits in the request pipeline

Summarization is split across two files for live UX feedback:

1. **`chat.py` Step 5b** — **Check only**: Determines if summarization is needed (fast token count check). Sets a `needs_summarization` flag but does NOT run the actual summarization.
2. **`streaming.py` Step 0** — **Execute inside the SSE generator**: Runs the actual summarization after the SSE stream has opened, so the frontend receives a "Compressing history..." status event immediately.

```
STEP 1: Parse request
STEP 2: Extract user query, detect images
STEP 2b: Web search (optional)
STEP 3: Analyze complexity → LOW / MEDIUM / HIGH
STEP 4: Route to tier → local / lakeshore / cloud
STEP 5: Prepare messages (convert to dicts)

  ┌─────────────────────────────────────────────────┐
  │  STEP 5b: Check if summarization needed (NEW)   │
  │                                                 │
  │  IF summarization enabled:                      │
  │    1. Estimate token count                      │
  │    2. Check if > 80% of tier's max input        │
  │    3. Set needs_summarization = True/False       │
  │    (DO NOT run summarization here — defer to     │
  │     streaming generator for live UX feedback)    │
  └─────────────────────────────────────────────────┘

STEP 6: Validate context window (skip if summarization pending)
STEP 7: Build SSE headers
STEP 8: Stream response to user → create_streaming_response()

  ┌─────────────────────────────────────────────────┐
  │  STEP 0 (inside streaming generator):           │
  │                                                 │
  │  IF needs_summarization:                        │
  │    1. Yield SSE: {"status": "summarizing_context"}│
  │       → Frontend shows "Compressing history..."  │
  │    2. Run apply_rolling_summarization()          │
  │    3. Yield SSE: {"status": "summarization_complete"}│
  │       → Frontend clears the banner               │
  │  THEN: continue with normal metadata + tokens    │
  └─────────────────────────────────────────────────┘
```

### Why this two-step architecture?

1. **Live UX feedback**: If summarization ran in `chat.py` (before the SSE stream opens), the user would see a frozen UI for 1-3 seconds with no explanation. By deferring to the streaming generator, the SSE connection is already open and we can push status events in real-time — the user sees "Compressing conversation history..." immediately. The frontend enforces a **5-second minimum display time** for this banner so the user can read it even when summarization is fast (e.g., short histories that compress in ~1-2 seconds).
2. **After tier selection (Step 4)**: We know which tier will handle the request, so we can apply the correct compression level.
3. **After message dict conversion (Step 5)**: Messages are already in the dict format the summarizer needs.
4. **Context validation safety**: Step 6 is skipped when `needs_summarization=True` because the generator will compress the messages before inference. If compression still leaves messages too large, the LLM itself will reject the request (caught by streaming error handling).

### Lazy evaluation

Summarization only runs when needed:
- Short conversations (< threshold): messages pass through unchanged, zero overhead
- Most conversations never trigger summarization — typical sessions are 5-15 turns
- Only long sessions (30+ turns, or sessions with large document uploads) trigger compression

---

## 7. Summarization Strategy

### Which model performs summarization?

**Local Ollama (Llama 3.2:3b)** — the same model that powers STREAM's LOCAL tier.

This choice is deliberate:

1. **Free**: No API cost for a background operation. Summarization should never add to the user's bill.
2. **Private**: Conversation history never leaves the user's machine (in desktop mode) or the deployment server. No data sent to external APIs for compression.
3. **Always available**: Ollama is a required component in both deployment modes. If LOCAL tier is healthy, the summarizer is available.
4. **Fast**: A 3B model summarizing ~2,000 tokens of conversation takes ~1-2 seconds on CPU.
5. **Philosophically consistent**: STREAM applies its own cost-aware routing principle to itself — summarization is a simple task, so it uses the cheapest tier.

### The summarization prompt

```
You are summarizing the earlier portion of an ongoing conversation.
The conversation is NOT over — it will continue after this summary.
Your summary will be injected as context so the AI can seamlessly
continue the discussion.

Write the summary as prior context, preserving:
- Key facts and decisions made so far
- Important context (names, preferences, technical details)
- The overall topic and current direction of the conversation
- Any code, commands, or technical specifics discussed
- Where things left off (what was the last topic or question?)

Be concise but complete. Start with 'Previous conversation summary: '
and write in past tense for completed topics and present tense for
ongoing threads.

Conversation to summarize:
---
User: [message 1]
Assistant: [response 1]
User: [message 2]
Assistant: [response 2]
...
---
```

### What gets preserved vs. summarized

| Content | Action | Reason |
|---------|--------|--------|
| System prompt(s) | **Preserved** (never summarized) | Core instructions for the LLM |
| Web search system message | **Preserved** | Contains recent search results relevant to the current query |
| Last N user+assistant pairs | **Preserved** (kept raw) | Recent context needed for conversational continuity |
| Older user+assistant pairs | **Summarized** | Compressed into a summary system message |
| Image data in old messages | **Dropped** (text description kept) | Base64 images consume too many tokens; `strip_old_images()` already handles this |

### Fallback on failure

If the summarization call to Ollama fails (Ollama is down, timeout, etc.):

1. **First fallback**: Naive truncation — take the first 200 characters of each old message and concatenate them as a rough summary
2. **Second fallback**: Simple message dropping — keep only system messages + last N pairs, drop everything else
3. **Never crash**: Summarization failure should never block the user's request

---

## 8. Multimodal Considerations

### Images in conversation history

STREAM supports image uploads (camera, file, clipboard) and document uploads (PDF with embedded images). As conversations grow, old messages may contain base64-encoded images.

### How summarization handles images

1. **Old messages with images**: The `extract_text_content()` utility from `multimodal.py` extracts only the text portions. Base64 image data is excluded from the summarization input. The text content (including any descriptions the user or assistant provided about the images) is included in the summary.

2. **Recent messages with images**: Kept raw, including full image data. The last N pairs are never modified.

3. **Existing infrastructure**: STREAM already has `strip_old_images()` in `multimodal.py` which removes images from old messages for the Globus Compute payload limit (6 MB). The summarization step works in concert with this — images are already stripped before summarization would encounter them.

4. **Summary of image discussions**: If the user uploaded an image and the assistant described it, the assistant's description is included in the summarization input. The summary might say: *"The user uploaded a circuit diagram. The assistant identified it as a series RLC circuit and explained the resonance frequency formula."*

---

## 9. Feature Toggle and A/B Evaluation

### Master toggle

```bash
# Enable (default)
ROLLING_SUMMARIZATION_ENABLED=true

# Disable (for comparison)
ROLLING_SUMMARIZATION_ENABLED=false
```

This environment variable can be set in the **`.env` file** — a single location that works for both deployment modes. Desktop mode loads `.env` at startup via `apply_desktop_defaults()`, and Docker Compose loads it via `env_file: - .env` in each service definition. You only need to change it in one place.

Alternatively, you can override it per-session via shell environment (`ROLLING_SUMMARIZATION_ENABLED=false python -m stream.desktop.main --dev`).

### A/B evaluation methodology for the paper

To demonstrate the value of tier-aware summarization, run the same set of benchmark conversations twice:

**Experiment A** — Summarization OFF:
1. Set `ROLLING_SUMMARIZATION_ENABLED=false`
2. Run N multi-turn conversations (30+ turns each) with a mix of LOW/MEDIUM/HIGH complexity queries
3. Record for each query: tier selected, cost, response time, whether context limit was exceeded

**Experiment B** — Summarization ON:
1. Set `ROLLING_SUMMARIZATION_ENABLED=true`
2. Run the same N conversations with the same queries
3. Record the same metrics

**Metrics to compare**:

| Metric | Without Summarization | With Summarization |
|--------|----------------------|-------------------|
| % of simple queries routed to LOCAL after turn 20 | Expected: declining | Expected: stable |
| % of queries that hit context limit error | Expected: increasing | Expected: near zero |
| Total cost per conversation | Expected: higher (forced upgrades) | Expected: lower |
| Number of forced tier upgrades | Expected: increasing with turns | Expected: minimal |
| Average response latency | Expected: higher (cloud calls) | Expected: lower (local calls) |

**Expected result**: *"Without tier-aware compression, X% of queries force-upgrade to Cloud after turn 20. With it, LOCAL handles Y% of simple queries even at turn 50."*

This directly validates the paper's C3 claim about the context-routing mismatch problem.

---

## 10. Comparison with Prior Work

### Conversation memory approaches

| System | Method | Target | Tier-Aware? | Routing-Aware? |
|--------|--------|--------|-------------|----------------|
| **LangChain ConversationSummaryBufferMemory** | Keep last N tokens raw, summarize rest | Single model | No | No |
| **Mem0** | Extract discrete facts as structured memories, store in vector DB | Single model | No | No |
| **MemGPT/Letta** | OS-inspired virtual memory with agent-controlled archival/recall | Single model | No | No |
| **LLMLingua** (EMNLP 2023) | Token-level pruning based on perplexity | Single model | No | No |
| **ACON** (2025) | Failure-driven compression for agent tasks | Single model | No | No |
| **ReSum** (2025) | Rolling summarization for web search agents | Single model | No | No |
| **STREAM** (this work) | Rolling summarization with differential compression | **Multiple tiers** | **Yes** | **Yes** |

### What's novel in STREAM's approach

1. **Differential compression**: The same conversation history produces different compressed representations depending on the target tier. This has not been published anywhere.

2. **Joint routing + compression**: The routing decision (which tier?) and the compression decision (how much?) are coupled. The router knows that LOCAL can handle a simple query IF the context is compressed, so it doesn't force-upgrade to Cloud.

3. **The "context-routing mismatch" problem**: This specific problem — where routing correctness depends on context length, not just query complexity — has not been named or addressed in the routing literature. RouteLLM, FrugalGPT, AutoMix, and Hybrid LLM all treat routing as a per-query decision independent of accumulated context.

### What's NOT novel (and that's fine)

- Rolling summarization itself is well-established (LangChain, Mem0, MemGPT all do it)
- Using an LLM to summarize conversation turns is standard practice
- The specific summarization technique (keep recent, compress old) is a known pattern

The contribution is not the compression technique — it's the application of compression to solve a problem that the routing literature has ignored.

---

## 11. Limitations and Degradation

### Information loss

Summarization is inherently lossy. Older turns lose:
- Exact wording and phrasing
- Minor details that weren't captured in the summary
- Nuanced context that a full transcript would preserve
- Code snippets that were paraphrased rather than quoted

### Summarization quality

The local model (Llama 3.2:3b) produces adequate but not perfect summaries:
- May miss subtle connections between topics
- May over-compress or under-compress certain discussions
- Handles English well; other languages may see quality degradation
- Technical content (code, math) is harder to summarize accurately

### Degradation over many rounds

If a conversation runs for 100+ turns with repeated summarization:
- Early context becomes increasingly abstracted ("The user discussed Python" rather than specific details)
- Each summarization cycle compounds information loss
- After ~50-100 summarization cycles, the summary may become generic
- This is a fundamental limitation of all rolling summarization approaches (see: Microsoft Research "LLMs Get Lost In Multi-Turn Conversation", May 2025)

### Not truly infinite

Rolling summarization extends conversations significantly (from ~60 turns to hundreds) but does not enable truly infinite conversations with perfect recall. For long-running sessions that need precise recall of early context, structured memory extraction (Mem0-style) or retrieval-augmented memory (MemGPT-style) would be more appropriate. These are potential future enhancements.

### Latency overhead

- Summarization adds ~1-2 seconds to the first request that triggers it (local model inference)
- Subsequent requests in the same session may trigger summarization again if the conversation continues to grow
- The overhead is only incurred when summarization is actually needed (lazy evaluation)
- For most conversations (< 15 turns), there is zero overhead

---

## 12. Implementation Reference

### File structure

```
stream/middleware/
├── config.py                          # SUMMARIZATION_CONFIG, toggle
├── routes/
│   └── chat.py                        # Step 5b: check if summarization needed
├── core/
│   └── streaming.py                   # Step 0: run summarization with live UX
└── utils/
    ├── summarization.py               # Core summarization logic
    ├── context_window.py              # Existing: token limits, validation
    ├── token_estimator.py             # Existing: estimate_tokens()
    └── multimodal.py                  # Existing: extract_text_content()

frontends/react/src/
├── types/message.ts                   # StreamMetadata: status, context_compressed
├── api/stream.ts                      # Passes status events to callbacks
└── components/chat/
    ├── ChatContainer.tsx              # Tracks isCompressingContext state
    └── TypingIndicator.tsx            # Shows "Compressing history..." banner
```

### Core functions in `summarization.py`

#### `should_summarize(messages, model, tier) -> bool`

Determines whether the current message array needs summarization for the given tier.

- Uses `estimate_tokens()` from `token_estimator.py` to count tokens
- Looks up the tier's threshold from `SUMMARIZATION_CONFIG`
- Returns `False` if the master toggle is off or the tier has summarization disabled

#### `get_compression_target(tier, model) -> dict`

Returns the compression parameters for the given tier:

```python
# LOCAL example:
{
    "max_summary_tokens": 2048,    # Maximum length of the summary (safety cap)
    "keep_recent_pairs": 3,        # Number of recent user+assistant pairs to keep raw
    "threshold_ratio": 0.8,        # Trigger when tokens exceed 80% of max input
    "enabled": True,               # Whether summarization is active for this tier
}
```

#### `summarize_messages(messages_to_summarize, correlation_id) -> str`

Takes a list of old message dicts and returns a text summary.

- Extracts text content from each message (handles multimodal messages)
- Builds a summarization prompt
- Calls local Ollama asynchronously
- Returns summary string
- On failure: returns a naive concatenation of first 200 chars per message

#### `apply_rolling_summarization(messages, model, tier, correlation_id) -> list`

Main entry point. Returns the message array with old messages replaced by a summary.

Steps:
1. Check `should_summarize()` — return original if not needed
2. Split messages into `[system] + [old] + [recent]`
3. Summarize old messages
4. Return `[system] + [summary_system_msg] + [recent]`

### Integration point in `chat.py` (Step 5b — check only)

```python
# Check if summarization is needed, but defer execution to the streaming
# generator for live UX feedback ("Compressing history..." banner).
needs_summarization = False
if ROLLING_SUMMARIZATION_ENABLED:
    from stream.middleware.utils.summarization import should_summarize
    needs_summarization = should_summarize(messages, model, tier)

# Pass the flag to the streaming generator
create_streaming_response(..., needs_summarization=needs_summarization)
```

### Execution point in `streaming.py` (Step 0 — inside SSE generator)

```python
if needs_summarization:
    # Send status event → frontend shows "Compressing history..."
    yield f"data: {json.dumps({'stream_metadata': {'status': 'summarizing_context'}})}\n\n"

    # Run summarization (~1-3 seconds on CPU)
    messages = await apply_rolling_summarization(messages, model, tier, correlation_id)

    # Notify frontend summarization is complete
    yield f"data: {json.dumps({'stream_metadata': {'status': 'summarization_complete'}})}\n\n"
```

### Configuration in `config.py`

```python
# Master toggle (env var for A/B evaluation)
ROLLING_SUMMARIZATION_ENABLED = os.getenv(
    "ROLLING_SUMMARIZATION_ENABLED", "true"
).lower() == "true"

# Per-tier settings — all tiers use 80% threshold for consistency
SUMMARIZATION_CONFIG = {
    "local": {
        "enabled": True,
        "threshold_ratio": 0.8,      # Trigger at 80% of context capacity
        "max_summary_tokens": 2048,  # Safety cap (~1.5 pages max)
        "keep_recent_pairs": 3,      # Keep last 6 messages raw
    },
    "lakeshore": {
        "enabled": True,
        "threshold_ratio": 0.8,
        "max_summary_tokens": 4096,  # Safety cap (~3 pages max)
        "keep_recent_pairs": 6,      # Keep last 12 messages raw
    },
    "cloud": {
        "enabled": False,            # Cloud rarely needs compression
        "threshold_ratio": 0.8,
        "max_summary_tokens": 8192,  # Safety cap (~6 pages max)
        "keep_recent_pairs": 8,      # Keep last 16 messages raw
    },
}

# Summarizer model (local Ollama — free, private)
SUMMARIZATION_MODEL = os.getenv("SUMMARIZATION_MODEL", "ollama/llama3.2:3b")
```

---

## 13. Code File Reference

| File | Purpose |
|------|---------|
| `stream/middleware/utils/summarization.py` | Core summarization logic (new) |
| `stream/middleware/routes/chat.py` | Main chat endpoint — Step 5b checks if summarization needed |
| `stream/middleware/core/streaming.py` | SSE generator — Step 0 runs summarization with live UX feedback |
| `stream/middleware/config.py` | Configuration: toggle, per-tier settings, summarizer model |
| `stream/middleware/utils/context_window.py` | Existing context window validation (unchanged) |
| `stream/middleware/utils/token_estimator.py` | Existing token counting (reused) |
| `stream/middleware/utils/multimodal.py` | Existing text extraction from multimodal messages (reused) |
| `stream/middleware/core/query_router.py` | Tier routing — runs before summarization (unchanged) |
| `stream/middleware/core/complexity_judge.py` | Complexity analysis — sees only latest query (unchanged) |
| `frontends/react/src/types/message.ts` | StreamMetadata type — added `status` and `context_compressed` fields |
| `frontends/react/src/components/chat/ChatContainer.tsx` | Tracks `isCompressingContext` state from SSE status events |
| `frontends/react/src/components/chat/TypingIndicator.tsx` | Shows "Compressing history..." banner during summarization |
| `tests/test_summarization.py` | Unit tests for summarization functions (new, 23 tests) |

---

## 14. References

### Context compression

1. LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models. Jiang et al., EMNLP 2023. https://arxiv.org/abs/2310.05736
2. ACON: Optimizing Context Compression for Long-Horizon LLM Agents. arXiv:2510.00615, 2025.
3. ReSum: Unlocking Long-Horizon Search Intelligence via Context Summarization. arXiv:2509.13313, 2025.
4. Recursively Summarizing Enables Long-Term Dialogue Memory in Large Language Models. arXiv:2308.15022, 2023. Published in Neurocomputing, 2025.

### LLM memory systems

5. Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory. arXiv:2504.19413, 2025.
6. MemGPT: Towards LLMs as Operating Systems. Packer et al., arXiv:2310.08560, 2023.
7. ProMem: Beyond Static Summarization — Proactive Memory with Self-Questioning. arXiv:2601.04463, 2026.
8. A-MEM: Agentic Memory for LLM Agents. arXiv:2502.12110, NeurIPS 2025.

### LLM routing

9. RouteLLM: Learning to Route LLMs with Preference Data. Ong et al., 2024. https://arxiv.org/abs/2406.18665
10. FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance. Chen et al., 2023.
11. AutoMix: Automatically Mixing Language Models. NeurIPS 2024.
12. Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing. ICLR 2024.

### Multi-turn conversation degradation

13. LLMs Get Lost In Multi-Turn Conversation. Microsoft Research, arXiv:2505.06120, 2025.

### Campus LLM systems

14. FIRST: Federated Inference Resource Scheduling Toolkit. Argonne National Lab, arXiv:2510.13724, SC'25 Workshops.
15. Providing On-Prem GenAI Inference Services to a Campus Community. Purdue, PEARC 2025.
16. Dartmouth Chat — Deploying an Open-Source LLM Stack at Scale. Dartmouth, PEARC 2025.

---

*Prepared by Anas Nassar (nassar@uic.edu) — STREAM Project, University of Illinois Chicago*
