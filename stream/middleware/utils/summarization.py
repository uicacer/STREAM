"""
Tier-aware rolling context summarization.

THE PROBLEM (Context-Routing Mismatch):
=======================================
STREAM routes queries to the cheapest capable tier based on complexity.
A simple question like "what is a for loop?" routes to LOCAL (free).
But by turn 30 of a conversation, the accumulated history might be
15,000+ tokens — too much for LOCAL's 32K context window to hold
comfortably. Without summarization, the query either:
  1. Gets rejected with a "context too long" error, or
  2. Gets force-upgraded to Cloud (expensive) even though LOCAL
     could answer it perfectly if the history were shorter.

This is the "context-routing mismatch": the query is simple, but the
accumulated history forces it onto an expensive tier.

THE SOLUTION (Rolling Summarization):
=====================================
When conversation history exceeds a tier's threshold, we compress
older messages into a concise summary while keeping recent messages
raw. The compression level varies by tier:

  LOCAL (32K context):     Aggressive — up to 2K summary + last 3 exchanges
  Lakeshore (64K context): Moderate  — up to 4K summary + last 6 exchanges
  Cloud (128-200K context): No compression needed (huge context window)

This means the SAME conversation can produce DIFFERENT compressed
message arrays for different tiers. A simple follow-up question at
turn 30 can still route to LOCAL because the compressed history fits.

HOW IT WORKS IN BOTH DEPLOYMENT MODES:
=======================================
The summarizer calls Ollama directly via HTTP (not through LiteLLM).
This works identically in both modes because:

  Server mode:  Ollama runs in Docker → OLLAMA_BASE_URL = "http://ollama:11434"
  Desktop mode: Ollama runs natively  → OLLAMA_BASE_URL = "http://localhost:11434"

Both modes already set OLLAMA_BASE_URL correctly in config.py, so
the summarizer works without any mode-specific code.

WHERE IT FITS IN THE PIPELINE:
==============================
Summarization runs between Step 5 (Prepare Messages) and Step 6
(Validate Context Window) in chat.py. This means:
  - We already know which tier was selected (Step 4)
  - Messages are already in dict format (Step 5)
  - Summarization reduces token count BEFORE validation (Step 6)
  - The LLM receives compressed messages transparently (Step 8)

See docs/ROLLING_SUMMARIZATION.md for the full design document.
"""

import logging

import httpx

from stream.middleware.config import (
    OLLAMA_BASE_URL,
    ROLLING_SUMMARIZATION_ENABLED,
    SUMMARIZATION_CONFIG,
    SUMMARIZATION_MODEL,
)
from stream.middleware.utils.context_window import get_max_input_tokens
from stream.middleware.utils.multimodal import extract_text_content
from stream.middleware.utils.token_estimator import estimate_tokens

logger = logging.getLogger(__name__)


# =============================================================================
# HELPER: Determine tier from model name
# =============================================================================


def _get_tier_from_model(model: str) -> str:
    """Derive the tier name from a model identifier.

    STREAM uses a naming convention where model names start with their tier:
      "local-llama"           → "local"
      "local-vision"          → "local"
      "lakeshore-qwen-vl-72b" → "lakeshore"
      "cloud-claude"          → "cloud"
      "cloud-or-gemini-pro"   → "cloud"

    This function extracts the tier prefix so we can look up the
    correct compression settings in SUMMARIZATION_CONFIG.
    """
    if model.startswith("local"):
        return "local"
    if model.startswith("lakeshore"):
        return "lakeshore"
    # Everything else is cloud (direct or OpenRouter)
    return "cloud"


# =============================================================================
# STEP 1: Should we summarize?
# =============================================================================


def get_compression_target(tier: str) -> dict:
    """Get the compression parameters for a given tier.

    Each tier has different settings because of their different
    context windows and model capabilities:

      LOCAL:     threshold_ratio=0.8, max_summary_tokens=2048,
                 keep_recent_pairs=3, enabled=True
      Lakeshore: threshold_ratio=0.8, max_summary_tokens=4096,
                 keep_recent_pairs=6, enabled=True
      Cloud:     threshold_ratio=0.8, max_summary_tokens=8192,
                 keep_recent_pairs=8, enabled=False

    What each setting means:

    - threshold_ratio (0.8 = 80%): Trigger summarization when the
      conversation fills 80% of the model's max input capacity.
      This gives conversations plenty of room to grow before compression
      kicks in, while still leaving 20% headroom after compression.
      All tiers use the same 80% threshold for simplicity.

    - max_summary_tokens: The MAXIMUM length Ollama can generate for
      the summary. This is a safety cap — actual summaries are usually
      much shorter because the LLM naturally produces concise output:

        20 turns → ~300-500 token summary (1 paragraph)
        50 turns → ~500-1,000 token summary (2-3 paragraphs)
        100 turns → ~1,000-2,000 token summary (half a page)

      The caps are proportional to model capability:
        LOCAL:     2,048 tokens max (~1.5 pages) — punchy, suited to 3B model
        Lakeshore: 4,096 tokens max (~3 pages)   — more detail for 72B model
        Cloud:     8,192 tokens max (~6 pages)    — most detail (rarely used)

    - keep_recent_pairs: Number of recent user+assistant exchanges
      to keep raw (not summarized). These preserve conversational
      continuity — the model sees the last few exchanges verbatim.
      Lakeshore keeps more (6) than LOCAL (3) because it has double
      the context window and its 72B model benefits from more raw context.

    - enabled: Whether summarization is active for this tier.
      Cloud is disabled by default because its context window is
      huge (128-200K tokens) — rarely exceeded in normal use.

    Returns:
        Dict with the compression settings for the requested tier.
        Falls back to cloud settings for unknown tiers (most permissive).
    """
    return SUMMARIZATION_CONFIG.get(tier, SUMMARIZATION_CONFIG["cloud"])


def should_summarize(messages: list[dict], model: str, tier: str) -> bool:
    """Determine whether the message array needs summarization.

    This is a quick check that runs on every request. It must be fast
    because it runs BEFORE the actual inference call. The check is:

      1. Is the master toggle on? (ROLLING_SUMMARIZATION_ENABLED)
      2. Is summarization enabled for this specific tier?
      3. Does the estimated token count exceed the tier's threshold?

    Only if ALL three conditions are true do we proceed with the
    (slower) summarization step.

    Example with LOCAL tier (32K context, threshold_ratio=0.8):
        max_input = 30,720 tokens (32K total - 2K output reserve)
        threshold = 30,720 × 0.8 = 24,576 tokens

        - 20 messages, ~3,000 tokens → 3,000 < 24,576 → No (fast path)
        - 80 messages, ~25,000 tokens → 25,000 > 24,576 → Yes (summarize)

    Example with Lakeshore tier (64K context, threshold_ratio=0.8):
        max_input = 61,440 tokens (64K total - 4K output reserve)
        threshold = 61,440 × 0.8 = 49,152 tokens

        - Most conversations never reach this — Lakeshore rarely summarizes.

    Args:
        messages: Full message array (system + history + current query).
        model: The target model identifier (e.g. "local-llama").
        tier: The target tier name ("local", "lakeshore", "cloud").

    Returns:
        True if summarization should be applied, False otherwise.
    """
    # Check 1: Master toggle (env var ROLLING_SUMMARIZATION_ENABLED)
    # This allows A/B testing — run the same conversations with
    # summarization on vs. off to measure the cost impact for the paper.
    if not ROLLING_SUMMARIZATION_ENABLED:
        return False

    # Check 2: Per-tier toggle
    # Cloud tier has summarization disabled by default because its
    # context window (128-200K) is rarely exceeded in normal use.
    config = get_compression_target(tier)
    if not config.get("enabled", False):
        return False

    # Check 3: Token count vs. threshold
    # We estimate the current token count and compare it against 80%
    # of the model's max input capacity. We compress at 80% (not 100%)
    # to leave headroom for new messages and avoid hitting the hard limit.
    estimated = estimate_tokens(messages)
    max_input = get_max_input_tokens(model)
    threshold = int(max_input * config["threshold_ratio"])

    return estimated > threshold


# =============================================================================
# STEP 2: Split messages into segments
# =============================================================================


def _split_messages(
    messages: list[dict], keep_recent_pairs: int
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split the message array into three segments for summarization.

    The message array typically looks like this:

        [system_prompt, web_search_system, u1, a1, u2, a2, ..., uN, aN, uN+1]

    We split it into:

        SYSTEM MESSAGES:  All role="system" messages (preserved as-is)
                          These include the system prompt, web search context,
                          and any other injected system messages.

        OLD MESSAGES:     Non-system messages BEFORE the recent window.
                          These are the ones we'll compress into a summary.

        RECENT MESSAGES:  The last N user+assistant pairs (kept raw).
                          These preserve conversational continuity so the
                          model knows what was just discussed.

    Visual example (keep_recent_pairs=3, as used for LOCAL tier):

        Input:  [sys, u1, a1, u2, a2, u3, a3, u4, a4, u5, a5, u6]
                 ↑sys  ←───── old ─────→  ←──── recent (3 pairs) ───→

        Output: system = [sys]
                old    = [u1, a1, u2, a2]     ← to be summarized
                recent = [u3, a3, u4, a4, u5, a5, u6]  ← kept raw

    Visual example (keep_recent_pairs=6, as used for Lakeshore tier):

        Input:  [sys, u1, a1, u2, a2, u3, a3, u4, a4, u5, a5, u6, a6, u7, a7, u8]
                 ↑sys  ←─ old ──→  ←────────── recent (6 pairs) ──────────────────→

        Output: system = [sys]
                old    = [u1, a1, u2, a2]     ← to be summarized
                recent = [u3, a3, u4, a4, u5, a5, u6, a6, u7, a7, u8]  ← kept raw

    Note: The last message might be a lone user message (no assistant
    response yet) — that's fine, it's included in the recent segment.

    Args:
        messages: Full message array.
        keep_recent_pairs: Number of recent user+assistant pairs to keep.
            LOCAL=3 (keeps 6 messages), Lakeshore=6 (keeps 12 messages).

    Returns:
        Tuple of (system_messages, old_messages, recent_messages).
        If there aren't enough messages to split, old_messages will be empty.
    """
    system_msgs = []
    non_system_msgs = []

    # Separate system messages from conversation messages.
    # System messages are NEVER summarized — they contain the system prompt,
    # web search results, and other injected context that must be preserved.
    for msg in messages:
        if msg.get("role") == "system":
            system_msgs.append(msg)
        else:
            non_system_msgs.append(msg)

    # Calculate how many non-system messages to keep raw.
    # Each "pair" is one user message + one assistant response = 2 messages.
    # Example: keep_recent_pairs=3 → keep_count=6 (3 user + 3 assistant msgs)
    keep_count = keep_recent_pairs * 2

    if len(non_system_msgs) <= keep_count:
        # Not enough messages to warrant splitting.
        # Everything is "recent" — nothing to summarize.
        return system_msgs, [], non_system_msgs

    # Split: everything before the recent window is "old"
    old = non_system_msgs[:-keep_count]
    recent = non_system_msgs[-keep_count:]

    return system_msgs, old, recent


# =============================================================================
# STEP 3: Format messages for the summarizer
# =============================================================================


def _format_messages_for_summary(messages: list[dict]) -> str:
    """Convert a list of message dicts into a readable transcript.

    The summarizer LLM needs a plain-text transcript to work with.
    This function converts each message into "Role: text" format:

        User: What is Python?
        Assistant: Python is a high-level programming language...
        User: Show me a for loop
        Assistant: Here's a for loop example...

    MULTIMODAL HANDLING:
    If a message contains images (OpenAI vision format), we extract
    only the text portions using extract_text_content(). The base64
    image data is dropped — it's too large and the summarizer can't
    process images anyway. However, the assistant's TEXT descriptions
    of those images ARE preserved, so the summary can reference what
    was discussed about the images.

    Args:
        messages: List of message dicts to format.

    Returns:
        A newline-separated transcript string.
    """
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown").capitalize()
        # extract_text_content() handles both string content and
        # multimodal list content (extracts text, drops images).
        # This function already exists in multimodal.py and is used
        # throughout STREAM for the same purpose.
        text = extract_text_content(msg.get("content", ""))
        if text.strip():
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


# =============================================================================
# STEP 4: Call Ollama to generate the summary
# =============================================================================


def _naive_fallback_summary(messages: list[dict]) -> str:
    """Create a rough summary by truncating each message.

    This is the "emergency" fallback when Ollama is unavailable
    (crashed, not started yet, network issue, etc.).

    Instead of a proper LLM-generated summary, we take the first
    200 characters of each message and concatenate them. It's crude
    but functional — the model gets SOME context about older turns
    rather than losing them entirely.

    This ensures summarization failure never blocks the user's request.
    A degraded summary is always better than an error.

    Args:
        messages: The older messages that would have been summarized.

    Returns:
        A crude truncated summary string.
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown").capitalize()
        text = extract_text_content(msg.get("content", ""))
        if text.strip():
            # Take first 200 chars of each message — just enough to
            # remind the model what each turn was about
            truncated = text[:200] + ("..." if len(text) > 200 else "")
            parts.append(f"{role}: {truncated}")
    return "Previous conversation (truncated): " + " | ".join(parts)


async def summarize_messages(
    messages_to_summarize: list[dict],
    max_summary_tokens: int,
    correlation_id: str,
) -> str:
    """Call local Ollama to summarize a list of older messages.

    WHY OLLAMA DIRECTLY (not LiteLLM)?
    We call Ollama's REST API directly via httpx instead of going through
    LiteLLM. This is simpler and works identically in both modes:

      Server mode:  POST http://ollama:11434/api/chat
      Desktop mode: POST http://localhost:11434/api/chat

    Both use OLLAMA_BASE_URL from config.py, which is already set
    correctly per deployment mode. No mode-specific code needed.

    WHY LOCAL OLLAMA (not Cloud)?
    The summarizer uses the local Ollama model (Llama 3.2:3b) because:
      1. FREE: Summarization should never add to the user's bill
      2. PRIVATE: Conversation history never leaves the machine
      3. ALWAYS AVAILABLE: Ollama is a required STREAM component
      4. FAST: A 3B model summarizes ~2K tokens in ~1-3 seconds on CPU
      5. CONSISTENT: Uses STREAM's own cost-aware philosophy — summarization
         is a simple task, so it uses the cheapest tier

    ACTUAL SUMMARY SIZES:
    The max_summary_tokens is a safety cap — the LLM naturally produces
    concise output when asked to summarize:
      - 20 turns of conversation → ~300-500 token summary (1 paragraph)
      - 50 turns → ~500-1,000 tokens (2-3 paragraphs)
      - 100 turns → ~1,000-2,000 tokens (half a page)
    The cap just prevents edge cases where the model rambles.

    FALLBACK CHAIN:
    If Ollama is unavailable (crash, timeout, etc.):
      1. First fallback: Naive truncation (first 200 chars per message)
      2. The caller (apply_rolling_summarization) returns the truncated
         version, which is better than no context at all
      3. The request proceeds normally — summarization failure is non-fatal

    Args:
        messages_to_summarize: The older message dicts to compress.
        max_summary_tokens: Safety cap on summary length (in tokens).
            This is passed to Ollama's "num_predict" parameter.
            LOCAL gets 2,048, Lakeshore gets 4,096.
            Actual summaries are usually 300-1,000 tokens.
        correlation_id: Request ID for logging.

    Returns:
        A summary string. On failure, falls back to naive truncation.
    """
    # Format the messages into a readable transcript for the summarizer.
    # This converts each message to "User: text" / "Assistant: text" format
    # and strips out any base64 image data (keeping text descriptions).
    transcript = _format_messages_for_summary(messages_to_summarize)

    if not transcript.strip():
        return ""

    # The summarization prompt tells the local LLM what to preserve.
    # Key items: facts, decisions, technical details, code snippets.
    # The prompt emphasizes that this is a *continuation* summary — the
    # conversation isn't over, it's ongoing. The LLM receiving this summary
    # needs to pick up where things left off, so the summary must be framed
    # as prior context for an active conversation, not a post-hoc recap.
    prompt = (
        "You are summarizing the earlier portion of an ongoing conversation. "
        "The conversation is NOT over — it will continue after this summary. "
        "Your summary will be injected as context so the AI can seamlessly "
        "continue the discussion.\n\n"
        "Write the summary as prior context, preserving:\n"
        "- Key facts and decisions made so far\n"
        "- Important context (names, preferences, technical details)\n"
        "- The overall topic and current direction of the conversation\n"
        "- Any code, commands, or technical specifics discussed\n"
        "- Where things left off (what was the last topic or question?)\n\n"
        "Be concise but complete. Start with 'Previous conversation summary: ' "
        "and write in past tense for completed topics and present tense for "
        "ongoing threads.\n\n"
        f"Conversation to summarize:\n---\n{transcript}\n---"
    )

    # Extract the Ollama model name.
    # SUMMARIZATION_MODEL is stored as "ollama/llama3.2:3b" (LiteLLM format).
    # Ollama's REST API expects just "llama3.2:3b" (without the "ollama/" prefix).
    ollama_model = SUMMARIZATION_MODEL
    if ollama_model.startswith("ollama/"):
        ollama_model = ollama_model[7:]  # Strip "ollama/" prefix

    try:
        # Call Ollama's chat API directly via HTTP.
        # We use stream=False because we need the complete summary
        # before we can proceed — unlike user-facing responses,
        # there's no benefit to streaming the summary token by token.
        #
        # Timeout: 30 seconds is generous for a local 3B model
        # summarizing ~2-5K tokens of text. Typical: 1-3 seconds.
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": ollama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    # num_predict limits how many tokens Ollama generates.
                    # This is a safety cap — actual summaries are usually
                    # 300-1,000 tokens. The cap prevents edge cases where
                    # the model rambles instead of summarizing concisely.
                    "options": {
                        "num_predict": max_summary_tokens,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()
            summary = data.get("message", {}).get("content", "")

            if summary.strip():
                logger.info(
                    f"[{correlation_id}] Summarized {len(messages_to_summarize)} "
                    f"messages ({len(transcript)} chars → {len(summary)} chars)"
                )
                return summary.strip()

    except Exception as e:
        # Non-fatal: if Ollama is down, we fall back to naive truncation.
        # The user's request still proceeds — just with a cruder summary.
        logger.warning(
            f"[{correlation_id}] Summarization via Ollama failed, " f"using naive fallback: {e}"
        )

    # Fallback: naive truncation (first 200 chars per message)
    return _naive_fallback_summary(messages_to_summarize)


# =============================================================================
# MAIN ENTRY POINT: Apply rolling summarization
# =============================================================================


async def apply_rolling_summarization(
    messages: list[dict],
    model: str,
    tier: str,
    correlation_id: str,
) -> list[dict]:
    """Apply tier-aware rolling summarization to the message array.

    This is the main function called from chat.py. It orchestrates
    the full summarization pipeline:

      1. CHECK:  Should we summarize? (token count vs. 80% of tier capacity)
      2. SPLIT:  Separate messages into [system] + [old] + [recent]
      3. SUMMARIZE: Call Ollama to compress [old] into a summary
      4. REASSEMBLE: Return [system] + [summary] + [recent]

    BEFORE (30 messages, ~10,000 tokens):
    ┌────────────────────────────────────────────────────────────────┐
    │ [sys_prompt] [web_search] [u1,a1] [u2,a2] ... [u14,a14] [u15]│
    └────────────────────────────────────────────────────────────────┘

    AFTER (targeting LOCAL tier, keep_recent_pairs=3):
    ┌────────────────────────────────────────────────────────────────┐
    │ [sys_prompt] [web_search] [SUMMARY of u1-u12] [u13,a13]      │
    │ [u14,a14] [u15]                                               │
    └────────────────────────────────────────────────────────────────┘
    (~5,500 tokens — fits in LOCAL's 32K context window)

    AFTER (targeting Lakeshore tier, keep_recent_pairs=6):
    ┌────────────────────────────────────────────────────────────────┐
    │ [sys_prompt] [web_search] [SUMMARY of u1-u6] [u7,a7] [u8,a8] │
    │ [u9,a9] [u10,a10] [u11,a11] [u12,a12] [u13,a13] [u14,a14]   │
    │ [u15]                                                          │
    └────────────────────────────────────────────────────────────────┘
    (~10,500 tokens — fits in Lakeshore's 64K context window)

    WORKS IN BOTH MODES:
    - Server mode:  Ollama at http://ollama:11434 (Docker container)
    - Desktop mode: Ollama at http://localhost:11434 (native install)
    Both use OLLAMA_BASE_URL from config.py — no mode-specific logic.

    IMPORTANT DESIGN DECISIONS:
    1. Lazy evaluation: Only runs when token count exceeds 80% of
       max input. Most conversations (< 15 turns) never trigger
       summarization — zero overhead for short conversations.
    2. Stateless: Summary is computed per-request, not stored in DB.
       No database changes needed. Simple, predictable behavior.
    3. Non-fatal: If summarization fails (Ollama down, timeout),
       falls back to naive truncation. The request still proceeds.
    4. System messages preserved: System prompt, web search results,
       and other system messages are NEVER summarized.

    Args:
        messages: Full message array (system + history + current query).
        model: Target model identifier (e.g. "local-llama").
            Used to look up the model's max input tokens.
        tier: Target tier name ("local", "lakeshore", "cloud").
            Used to look up compression settings (threshold,
            summary length, recent pairs to keep).
        correlation_id: Request ID for logging and debugging.

    Returns:
        The (possibly compressed) message array. If summarization
        was not needed or failed, returns the original messages unchanged.
    """
    # ── Quick check: is summarization needed? ────────────────────────
    # This is the fast path — most requests return here immediately
    # because the conversation is short enough for the target tier.
    if not should_summarize(messages, model, tier):
        return messages

    # ── Get compression settings for this tier ───────────────────────
    # LOCAL:     keep 3 recent pairs, summary capped at 2K tokens
    # Lakeshore: keep 6 recent pairs, summary capped at 4K tokens
    config = get_compression_target(tier)
    keep_recent_pairs = config["keep_recent_pairs"]
    max_summary_tokens = config["max_summary_tokens"]

    # ── Split messages into segments ─────────────────────────────────
    # system_msgs: role="system" messages (preserved as-is)
    # old_msgs:    older conversation turns (will be summarized)
    # recent_msgs: last N pairs + current query (kept raw)
    system_msgs, old_msgs, recent_msgs = _split_messages(messages, keep_recent_pairs)

    if not old_msgs:
        # Edge case: all messages are either system or recent.
        # Nothing to summarize — return unchanged.
        return messages

    # Record original token count so we can log the compression ratio
    original_tokens = estimate_tokens(messages)

    # ── Call Ollama to summarize the old messages ────────────────────
    # This is the "expensive" step (~1-3 seconds on CPU for a 3B model).
    # On failure, summarize_messages() returns a naive truncation
    # (first 200 chars per message) so we never block the request.
    summary = await summarize_messages(old_msgs, max_summary_tokens, correlation_id)

    if not summary:
        # Empty summary (e.g., all old messages were empty/images-only)
        return messages

    # ── Build the summary system message ─────────────────────────────
    # The summary is injected as a system message so the LLM treats it
    # as background context, not as something a user or assistant said.
    # The prefix "Previous conversation summary:" tells the LLM this
    # is compressed history, not a fresh instruction.
    summary_message = {
        "role": "system",
        "content": f"Previous conversation summary: {summary}",
    }

    # ── Reassemble the message array ─────────────────────────────────
    # Order: [system messages] + [summary] + [recent messages]
    #
    # This preserves the expected message structure:
    #   1. System instructions come first (system prompt, web search)
    #   2. Compressed history provides background context
    #   3. Recent raw messages give the model immediate conversational context
    #   4. The latest user query is at the end (part of recent_msgs)
    compressed = system_msgs + [summary_message] + recent_msgs

    # ── Log the compression result ───────────────────────────────────
    # This log line is important for the paper's evaluation section.
    # It shows: original tokens, compressed tokens, reduction percentage,
    # which tier, how many messages were summarized, how many kept raw.
    compressed_tokens = estimate_tokens(compressed)
    ratio = (1 - compressed_tokens / original_tokens) * 100 if original_tokens > 0 else 0

    logger.info(
        f"[{correlation_id}] Rolling summarization applied: "
        f"{original_tokens} → {compressed_tokens} tokens "
        f"({ratio:.0f}% reduction, tier={tier}, "
        f"summarized={len(old_msgs)} msgs, kept={len(recent_msgs)} recent)"
    )

    return compressed
