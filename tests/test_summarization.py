"""
Tests for tier-aware rolling context summarization.

This test module validates the summarization pipeline:
  1. should_summarize() — threshold checks per tier
  2. _split_messages() — message segmentation into system/old/recent
  3. _format_messages_for_summary() — transcript formatting
  4. _naive_fallback_summary() — emergency fallback
  5. apply_rolling_summarization() — full pipeline integration

These tests do NOT require running services (Ollama, etc.).
They test pure logic functions and mock the Ollama HTTP call.

Run with:
    pytest tests/test_summarization.py -v
"""

from unittest.mock import patch

import pytest

from stream.middleware.utils.summarization import (
    _format_messages_for_summary,
    _naive_fallback_summary,
    _split_messages,
    apply_rolling_summarization,
    get_compression_target,
    should_summarize,
)

# =============================================================================
# FIXTURES: Reusable test data
# =============================================================================
# Fixtures let us define test data once and reuse it across many tests.
# pytest automatically passes these to any test function that has a
# parameter with the same name as the fixture.


@pytest.fixture
def system_message():
    """A system prompt message (should never be summarized)."""
    return {"role": "system", "content": "You are a helpful assistant."}


@pytest.fixture
def web_search_message():
    """A web search system message (should also be preserved)."""
    return {
        "role": "system",
        "content": "Web search results: Python is a programming language...",
    }


@pytest.fixture
def short_conversation(system_message):
    """A short conversation (5 turns) — should NOT trigger summarization.

    This represents a typical short chat session. The total token count
    is well below any tier's threshold.
    """
    return [
        system_message,
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
        {"role": "user", "content": "Show me a for loop"},
        {"role": "assistant", "content": "Here is a for loop: for i in range(10): print(i)"},
        {"role": "user", "content": "Thanks!"},
    ]


@pytest.fixture
def long_conversation(system_message):
    """A long conversation (40 turns) with enough tokens to trigger summarization.

    Each message is ~200 chars (~50 tokens). 80 messages × 50 tokens = ~4,000 tokens.
    That's well below LOCAL's threshold (24,576), so we'll need to make messages
    longer to actually test the threshold. But this is useful for testing
    the split logic with many messages.
    """
    messages = [system_message]
    for i in range(40):
        messages.append(
            {
                "role": "user",
                "content": f"Question {i}: Can you explain topic number {i} in detail? " * 5,
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": f"Answer {i}: Here is a detailed explanation of topic {i}. " * 10,
            }
        )
    messages.append({"role": "user", "content": "Final question"})
    return messages


@pytest.fixture
def multimodal_conversation(system_message):
    """A conversation containing image messages (multimodal).

    Tests that images in old messages are handled correctly —
    base64 data should be stripped, text descriptions preserved.
    """
    return [
        system_message,
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/fake"}},
            ],
        },
        {"role": "assistant", "content": "The image shows a circuit diagram."},
        {"role": "user", "content": "Can you explain the circuit?"},
        {"role": "assistant", "content": "This is a series RLC circuit."},
        {"role": "user", "content": "What is the resonance frequency?"},
    ]


# =============================================================================
# TESTS: get_compression_target()
# =============================================================================


class TestGetCompressionTarget:
    """Test that each tier returns the correct compression settings."""

    def test_local_tier(self):
        """LOCAL tier should have aggressive compression settings."""
        config = get_compression_target("local")
        assert config["enabled"] is True
        assert config["threshold_ratio"] == 0.8
        assert config["max_summary_tokens"] == 2048
        assert config["keep_recent_pairs"] == 3

    def test_lakeshore_tier(self):
        """Lakeshore tier should have moderate compression settings.

        Lakeshore keeps more recent pairs (6 vs 3) because it has
        double the context window and a more capable model (72B).
        """
        config = get_compression_target("lakeshore")
        assert config["enabled"] is True
        assert config["threshold_ratio"] == 0.8
        assert config["max_summary_tokens"] == 4096
        assert config["keep_recent_pairs"] == 6

    def test_cloud_tier_disabled(self):
        """Cloud tier should have summarization disabled by default.

        Cloud models have huge context windows (128-200K) that are
        rarely exceeded in normal conversations.
        """
        config = get_compression_target("cloud")
        assert config["enabled"] is False

    def test_unknown_tier_falls_back_to_cloud(self):
        """Unknown tiers should fall back to cloud settings (safest)."""
        config = get_compression_target("unknown_tier")
        assert config["enabled"] is False


# =============================================================================
# TESTS: should_summarize()
# =============================================================================


class TestShouldSummarize:
    """Test the summarization trigger logic."""

    def test_short_conversation_no_summarization(self, short_conversation):
        """Short conversations should never trigger summarization.

        A 5-turn conversation is only ~500 tokens — far below
        LOCAL's threshold of ~24,576 tokens.
        """
        assert should_summarize(short_conversation, "local-llama", "local") is False

    def test_cloud_tier_never_summarizes(self, long_conversation):
        """Cloud tier should never summarize (disabled by default)."""
        assert should_summarize(long_conversation, "cloud-claude", "cloud") is False

    @patch("stream.middleware.utils.summarization.ROLLING_SUMMARIZATION_ENABLED", False)
    def test_master_toggle_off(self, long_conversation):
        """When the master toggle is off, no summarization happens.

        This is used for A/B evaluation — run the same conversations
        with the toggle on vs. off to measure the impact.
        """
        assert should_summarize(long_conversation, "local-llama", "local") is False

    @patch("stream.middleware.utils.summarization.estimate_tokens", return_value=25000)
    def test_above_threshold_triggers(self, mock_tokens, short_conversation):
        """When token count exceeds 80% of max input, summarize.

        LOCAL's max input is ~30,720. 80% of that is ~24,576.
        25,000 > 24,576 → should trigger summarization.
        """
        assert should_summarize(short_conversation, "local-llama", "local") is True

    @patch("stream.middleware.utils.summarization.estimate_tokens", return_value=20000)
    def test_below_threshold_no_trigger(self, mock_tokens, short_conversation):
        """When token count is below 80% of max input, don't summarize.

        20,000 < 24,576 → no summarization needed.
        """
        assert should_summarize(short_conversation, "local-llama", "local") is False


# =============================================================================
# TESTS: _split_messages()
# =============================================================================


class TestSplitMessages:
    """Test message segmentation into system/old/recent."""

    def test_basic_split_keep_3_pairs(self, system_message):
        """With keep_recent_pairs=3 (LOCAL), keep the last 6 non-system messages.

        Given 11 non-system messages (5 pairs + 1 lone user msg):
          system = [sys]
          old    = first 5 non-system msgs
          recent = last 6 non-system msgs
        """
        messages = [
            system_message,
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": "a3"},
            {"role": "user", "content": "q4"},
            {"role": "assistant", "content": "a4"},
            {"role": "user", "content": "q5"},
            {"role": "assistant", "content": "a5"},
            {"role": "user", "content": "q6"},
        ]

        system, old, recent = _split_messages(messages, keep_recent_pairs=3)

        # System messages preserved
        assert len(system) == 1
        assert system[0]["role"] == "system"

        # Old messages: first 5 non-system (11 total - 6 recent = 5 old)
        # q1, a1, q2, a2, q3
        assert len(old) == 5
        assert old[0]["content"] == "q1"
        assert old[-1]["content"] == "q3"

        # Recent messages: last 6 non-system
        # a3, q4, a4, q5, a5, q6
        assert len(recent) == 6
        assert recent[0]["content"] == "a3"
        assert recent[-1]["content"] == "q6"

    def test_basic_split_keep_6_pairs(self, system_message):
        """With keep_recent_pairs=6 (Lakeshore), keep the last 12 non-system messages.

        With 14 non-system messages, only the first 2 are "old".
        """
        messages = [system_message]
        for i in range(7):
            messages.append({"role": "user", "content": f"q{i+1}"})
            messages.append({"role": "assistant", "content": f"a{i+1}"})

        system, old, recent = _split_messages(messages, keep_recent_pairs=6)

        assert len(system) == 1
        assert len(old) == 2  # 14 - 12 = 2 old messages
        assert len(recent) == 12

    def test_too_few_messages_nothing_to_summarize(self, system_message):
        """When there are fewer messages than keep_count, nothing to summarize.

        With 4 non-system messages and keep_recent_pairs=3 (keep 6),
        all messages are "recent" — old list should be empty.
        """
        messages = [
            system_message,
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]

        system, old, recent = _split_messages(messages, keep_recent_pairs=3)

        assert len(system) == 1
        assert len(old) == 0  # Nothing to summarize
        assert len(recent) == 4  # All non-system messages are "recent"

    def test_multiple_system_messages_preserved(self, system_message, web_search_message):
        """ALL system messages should be preserved (system prompt + web search).

        System messages are never summarized because they contain
        the system prompt, web search results, and other injected context.
        """
        messages = [
            system_message,
            web_search_message,
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": "a3"},
            {"role": "user", "content": "q4"},
            {"role": "assistant", "content": "a4"},
            {"role": "user", "content": "q5"},
        ]

        system, old, recent = _split_messages(messages, keep_recent_pairs=3)

        # Both system messages preserved
        assert len(system) == 2
        assert system[0]["content"] == "You are a helpful assistant."
        assert "Web search" in system[1]["content"]


# =============================================================================
# TESTS: _format_messages_for_summary()
# =============================================================================


class TestFormatMessagesForSummary:
    """Test transcript formatting for the summarizer."""

    def test_text_messages(self):
        """Text messages should be formatted as 'Role: text' lines."""
        messages = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
        ]

        transcript = _format_messages_for_summary(messages)

        assert "User: What is Python?" in transcript
        assert "Assistant: Python is a programming language." in transcript

    def test_multimodal_messages_strip_images(self):
        """Multimodal messages should have images stripped, text preserved.

        The base64 image data is too large for the summarizer and it
        can't process images anyway. But the text description is kept.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this circuit"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/fake"}},
                ],
            },
            {"role": "assistant", "content": "This is a series RLC circuit."},
        ]

        transcript = _format_messages_for_summary(messages)

        assert "User: Describe this circuit" in transcript
        assert "Assistant: This is a series RLC circuit." in transcript
        # Base64 data should NOT appear in the transcript
        assert "base64" not in transcript

    def test_empty_messages_skipped(self):
        """Messages with empty content should be skipped."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "Goodbye"},
        ]

        transcript = _format_messages_for_summary(messages)

        assert "User: Hello" in transcript
        assert "User: Goodbye" in transcript
        # Empty assistant message should not appear
        assert transcript.count("Assistant:") == 0


# =============================================================================
# TESTS: _naive_fallback_summary()
# =============================================================================


class TestNaiveFallbackSummary:
    """Test the emergency fallback when Ollama is unavailable."""

    def test_truncates_long_messages(self):
        """Long messages should be truncated to 200 characters."""
        long_text = "A" * 500
        messages = [{"role": "user", "content": long_text}]

        summary = _naive_fallback_summary(messages)

        # Should contain truncated text with "..."
        assert "..." in summary
        assert "Previous conversation (truncated):" in summary

    def test_short_messages_no_truncation(self):
        """Short messages should not be truncated."""
        messages = [{"role": "user", "content": "Hello"}]

        summary = _naive_fallback_summary(messages)

        assert "User: Hello" in summary
        assert "..." not in summary


# =============================================================================
# TESTS: apply_rolling_summarization() — Full pipeline
# =============================================================================


class TestApplyRollingSummarization:
    """Test the full summarization pipeline (with mocked Ollama)."""

    @pytest.mark.asyncio
    async def test_short_conversation_unchanged(self, short_conversation):
        """Short conversations should pass through completely unchanged.

        No summarization, no modification — the messages array is
        returned as-is. This is the fast path for most requests.
        """
        result = await apply_rolling_summarization(
            messages=short_conversation,
            model="local-llama",
            tier="local",
            correlation_id="test-123",
        )

        # Should return the exact same list (unchanged)
        assert result == short_conversation

    @pytest.mark.asyncio
    async def test_cloud_tier_unchanged(self, long_conversation):
        """Cloud tier should never summarize (disabled by default)."""
        result = await apply_rolling_summarization(
            messages=long_conversation,
            model="cloud-claude",
            tier="cloud",
            correlation_id="test-123",
        )

        assert result == long_conversation

    @pytest.mark.asyncio
    @patch("stream.middleware.utils.summarization.estimate_tokens", return_value=25000)
    @patch("stream.middleware.utils.summarization.summarize_messages")
    async def test_summarization_produces_summary_message(
        self, mock_summarize, mock_tokens, system_message
    ):
        """When triggered, should produce a summary system message.

        The old messages get replaced by a single system message
        starting with "Previous conversation summary:".
        """
        # Build a conversation with enough messages to have "old" ones
        messages = [system_message]
        for i in range(10):
            messages.append({"role": "user", "content": f"q{i}"})
            messages.append({"role": "assistant", "content": f"a{i}"})
        messages.append({"role": "user", "content": "latest question"})

        # Mock the Ollama call to return a known summary
        mock_summarize.return_value = "User learned about Python basics."

        result = await apply_rolling_summarization(
            messages=messages,
            model="local-llama",
            tier="local",
            correlation_id="test-456",
        )

        # Result should contain:
        # 1. Original system message (preserved)
        # 2. Summary system message (new)
        # 3. Recent messages (last 3 pairs = 6 msgs + possible lone user msg)
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are a helpful assistant."

        # Find the summary message
        summary_msgs = [
            m
            for m in result
            if m["role"] == "system" and "Previous conversation summary:" in m.get("content", "")
        ]
        assert len(summary_msgs) == 1
        assert "Python basics" in summary_msgs[0]["content"]

        # Recent messages should be preserved raw
        assert result[-1]["content"] == "latest question"

    @pytest.mark.asyncio
    @patch("stream.middleware.utils.summarization.estimate_tokens", return_value=25000)
    @patch("stream.middleware.utils.summarization.summarize_messages")
    async def test_system_messages_never_summarized(
        self, mock_summarize, mock_tokens, system_message, web_search_message
    ):
        """System messages (prompt + web search) should always be preserved.

        They contain critical context that must not be lost:
        - System prompt: instructions for the LLM
        - Web search: current search results for the user's query
        """
        messages = [system_message, web_search_message]
        for i in range(10):
            messages.append({"role": "user", "content": f"q{i}"})
            messages.append({"role": "assistant", "content": f"a{i}"})

        mock_summarize.return_value = "Summary of conversation."

        result = await apply_rolling_summarization(
            messages=messages,
            model="local-llama",
            tier="local",
            correlation_id="test-789",
        )

        # Both original system messages should be at the start
        system_msgs = [m for m in result if m["role"] == "system"]
        system_contents = [m["content"] for m in system_msgs]

        assert "You are a helpful assistant." in system_contents
        assert any("Web search" in c for c in system_contents)

    @pytest.mark.asyncio
    @patch("stream.middleware.utils.summarization.ROLLING_SUMMARIZATION_ENABLED", False)
    async def test_disabled_toggle_returns_unchanged(self, long_conversation):
        """When the master toggle is off, messages pass through unchanged."""
        result = await apply_rolling_summarization(
            messages=long_conversation,
            model="local-llama",
            tier="local",
            correlation_id="test-disabled",
        )

        assert result == long_conversation
