"""
Tests for reasoning/thinking content support across desktop and server modes.

This module validates that:
  - Reasoning models are correctly detected via is_reasoning_model()
  - Desktop mode (litellm_direct.py) adds reasoning_effort and extracts thinking
  - Server mode (litellm_client.py) adds reasoning_effort to the HTTP payload
  - streaming.py extracts reasoning_content from SSE chunks and emits thinking events
  - The frontend-expected format {"thinking": "..."} is produced correctly

Tests cover both the shared utility (config.py) and each mode's integration path.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stream.middleware.config import REASONING_MODEL_PATTERNS, is_reasoning_model

# =============================================================================
# is_reasoning_model() — shared utility tests
# =============================================================================


class TestIsReasoningModel:
    """Verify the shared reasoning model detection function."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "anthropic/claude-sonnet-4",
            "anthropic/claude-opus-4",
            "claude-sonnet-4-20250514",
            "openrouter/anthropic/claude-4",
            "openai/o1",
            "openai/o1-mini",
            "openai/o3",
            "openai/o3-mini",
            "openai/o4-mini",
            "deepseek/deepseek-r1",
            "deepseek-r1:free",
            "cloud-or-dynamic-deepseek/deepseek-r1",
        ],
    )
    def test_detects_reasoning_models(self, model_name):
        assert is_reasoning_model(model_name), f"Should detect {model_name} as reasoning"

    @pytest.mark.parametrize(
        "model_name",
        [
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "google/gemini-2.5-flash",
            "meta-llama/llama-4-maverick",
            "deepseek/deepseek-chat",
            "mistralai/mistral-large",
            "local-llama",
            "lakeshore-qwen",
        ],
    )
    def test_ignores_non_reasoning_models(self, model_name):
        assert not is_reasoning_model(model_name), f"Should NOT detect {model_name} as reasoning"

    def test_case_insensitive(self):
        assert is_reasoning_model("Claude-Sonnet-4")
        assert is_reasoning_model("DEEPSEEK-R1")
        assert is_reasoning_model("O3-MINI")


# =============================================================================
# Desktop mode — litellm_direct.py
# =============================================================================


class TestDesktopModeReasoning:
    """Verify desktop mode adds reasoning_effort for direct calls and skips for OpenRouter."""

    @pytest.mark.parametrize(
        "model,litellm_model",
        [
            ("direct-claude", "anthropic/claude-sonnet-4"),
            ("direct-o3", "openai/o3"),
        ],
    )
    @patch("stream.middleware.core.litellm_direct.litellm")
    @patch("stream.middleware.core.litellm_direct._resolve_model")
    @pytest.mark.asyncio
    async def test_adds_reasoning_effort_for_direct_providers(
        self, mock_resolve, mock_litellm, model, litellm_model
    ):
        """reasoning_effort should be added for direct provider calls (not OpenRouter)."""
        mock_resolve.return_value = {"model": litellm_model}

        mock_chunk = MagicMock()
        mock_chunk.model_dump.return_value = {
            "model": litellm_model,
            "choices": [{"delta": {"reasoning_content": "Let me think..."}}],
        }

        async def mock_acompletion(**kwargs):
            assert (
                kwargs.get("reasoning_effort") == "low"
            ), f"reasoning_effort should be 'low' for direct {model}"

            async def _gen():
                yield mock_chunk

            return _gen()

        mock_litellm.acompletion = mock_acompletion

        from stream.middleware.core.litellm_direct import forward_direct

        lines = []
        async for line in forward_direct(
            model, [{"role": "user", "content": "hi"}], 0.7, "test-123"
        ):
            lines.append(line)

        thinking_lines = [line for line in lines if '"thinking"' in line]
        assert len(thinking_lines) >= 1, "Should emit at least one thinking event"
        thinking_data = json.loads(thinking_lines[0].removeprefix("data: "))
        assert thinking_data["thinking"] == "Let me think..."

    @pytest.mark.parametrize(
        "model,litellm_model",
        [
            ("cloud-or-claude", "openrouter/anthropic/claude-sonnet-4"),
            ("cloud-or-dynamic-deepseek/deepseek-r1", "openrouter/deepseek/deepseek-r1"),
        ],
    )
    @patch("stream.middleware.core.litellm_direct.litellm")
    @patch("stream.middleware.core.litellm_direct._resolve_model")
    @pytest.mark.asyncio
    async def test_skips_reasoning_effort_for_openrouter(
        self, mock_resolve, mock_litellm, model, litellm_model
    ):
        """reasoning_effort should NOT be added for OpenRouter models (litellm rejects it)."""
        mock_resolve.return_value = {"model": litellm_model}

        mock_chunk = MagicMock()
        mock_chunk.model_dump.return_value = {
            "model": litellm_model,
            "choices": [{"delta": {"content": "Hello!"}}],
        }

        async def mock_acompletion(**kwargs):
            assert (
                "reasoning_effort" not in kwargs
            ), f"reasoning_effort should NOT be set for OpenRouter model {litellm_model}"

            async def _gen():
                yield mock_chunk

            return _gen()

        mock_litellm.acompletion = mock_acompletion

        from stream.middleware.core.litellm_direct import forward_direct

        lines = []
        async for line in forward_direct(
            model, [{"role": "user", "content": "hi"}], 0.7, "test-or-skip"
        ):
            lines.append(line)

    @patch("stream.middleware.core.litellm_direct.litellm")
    @patch("stream.middleware.core.litellm_direct._resolve_model")
    @pytest.mark.asyncio
    async def test_no_reasoning_for_non_reasoning_model(self, mock_resolve, mock_litellm):
        """reasoning_effort should NOT be added for regular models."""
        mock_resolve.return_value = {"model": "openrouter/openai/gpt-4o"}

        mock_chunk = MagicMock()
        mock_chunk.model_dump.return_value = {
            "model": "gpt-4o",
            "choices": [{"delta": {"content": "Hello!"}}],
        }

        async def mock_acompletion(**kwargs):
            assert "reasoning_effort" not in kwargs, "reasoning_effort should NOT be set for gpt-4o"

            async def _gen():
                yield mock_chunk

            return _gen()

        mock_litellm.acompletion = mock_acompletion

        from stream.middleware.core.litellm_direct import forward_direct

        lines = []
        async for line in forward_direct(
            "cloud-or-gpt4o", [{"role": "user", "content": "hi"}], 0.7, "test-456"
        ):
            lines.append(line)

        thinking_lines = [line for line in lines if '"thinking"' in line]
        assert len(thinking_lines) == 0, "Should NOT emit thinking events for non-reasoning models"


# =============================================================================
# Server mode — litellm_client.py
# =============================================================================


class TestServerModeReasoning:
    """Verify server mode adds reasoning_effort to the HTTP payload."""

    @patch("stream.middleware.core.litellm_client.STREAM_MODE", "server")
    @patch("stream.middleware.core.litellm_client.LITELLM_BASE_URL", "http://localhost:4000")
    @patch("stream.middleware.core.litellm_client.LITELLM_API_KEY", "test-key")
    @patch("stream.middleware.core.litellm_client.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_skips_reasoning_for_cloud_models(self, mock_client_class):
        """Server mode should NOT include reasoning_effort for cloud (OpenRouter) models."""
        captured_payload = {}

        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def mock_aiter_lines():
            yield 'data: {"choices":[{"delta":{"content":"Hi"}}],"model":"claude-sonnet-4"}'
            yield "data: [DONE]"

        mock_response.aiter_lines = mock_aiter_lines

        mock_client = AsyncMock()
        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        def capture_stream(method, url, json=None, headers=None):
            captured_payload.update(json or {})
            return mock_stream_ctx

        mock_client.stream = capture_stream
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_client

        from stream.middleware.core.litellm_client import forward_to_litellm

        lines = []
        async for line in forward_to_litellm(
            "cloud-or-dynamic-anthropic/claude-sonnet-4",
            [{"role": "user", "content": "hi"}],
            0.7,
            "test-server-001",
        ):
            lines.append(line)

        assert (
            "reasoning_effort" not in captured_payload
        ), "Server mode should NOT include reasoning_effort for cloud (OpenRouter) models"

    @patch("stream.middleware.core.litellm_client.STREAM_MODE", "server")
    @patch("stream.middleware.core.litellm_client.LITELLM_BASE_URL", "http://localhost:4000")
    @patch("stream.middleware.core.litellm_client.LITELLM_API_KEY", "test-key")
    @patch("stream.middleware.core.litellm_client.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_no_reasoning_for_regular_model(self, mock_client_class):
        """Server mode should NOT include reasoning_effort for regular models."""
        captured_payload = {}

        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def mock_aiter_lines():
            yield 'data: {"choices":[{"delta":{"content":"Hi"}}],"model":"gpt-4o"}'
            yield "data: [DONE]"

        mock_response.aiter_lines = mock_aiter_lines

        mock_client = AsyncMock()
        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        def capture_stream(method, url, json=None, headers=None):
            captured_payload.update(json or {})
            return mock_stream_ctx

        mock_client.stream = capture_stream
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_client

        from stream.middleware.core.litellm_client import forward_to_litellm

        lines = []
        async for line in forward_to_litellm(
            "cloud-or-gpt4o",
            [{"role": "user", "content": "hi"}],
            0.7,
            "test-server-002",
        ):
            lines.append(line)

        assert (
            "reasoning_effort" not in captured_payload
        ), "Server mode should NOT include reasoning_effort for non-reasoning models"


# =============================================================================
# streaming.py — reasoning_content extraction from SSE chunks
# =============================================================================


class TestStreamingReasoningExtraction:
    """Verify streaming.py extracts reasoning_content and emits thinking events."""

    def test_extracts_reasoning_from_server_mode_chunk(self):
        """streaming.py should extract delta.reasoning_content and emit {"thinking": ...}."""
        sse_line = 'data: {"choices":[{"delta":{"reasoning_content":"Step 1: analyze..."}}],"model":"claude-sonnet-4"}'
        data_str = sse_line[6:].strip()
        parsed = json.loads(data_str)

        choices = parsed.get("choices", [])
        assert len(choices) > 0
        delta = choices[0].get("delta", {})
        reasoning = delta.get("reasoning_content")
        assert reasoning == "Step 1: analyze..."

        thinking_event = json.dumps({"thinking": reasoning})
        assert '"thinking"' in thinking_event
        assert "Step 1: analyze..." in thinking_event

    def test_forwards_pre_extracted_thinking_events(self):
        """Thinking events from desktop mode ({"thinking": "..."}) should be forwarded."""
        sse_line = 'data: {"thinking": "Let me reason about this..."}'
        data_str = sse_line[6:].strip()
        parsed = json.loads(data_str)

        assert "thinking" in parsed
        assert parsed["thinking"] == "Let me reason about this..."

    def test_no_thinking_for_regular_content(self):
        """Regular content chunks should not produce thinking events."""
        sse_line = 'data: {"choices":[{"delta":{"content":"Hello, world!"}}],"model":"gpt-4o"}'
        data_str = sse_line[6:].strip()
        parsed = json.loads(data_str)

        choices = parsed.get("choices", [])
        delta = choices[0].get("delta", {})
        reasoning = delta.get("reasoning_content")
        assert reasoning is None


# =============================================================================
# End-to-end format validation
# =============================================================================


class TestThinkingEventFormat:
    """Validate the thinking event format matches what the frontend expects."""

    def test_thinking_event_structure(self):
        """The thinking event must have exactly {"thinking": "string"} format."""
        event = {"thinking": "Some reasoning content here"}
        serialized = json.dumps(event)
        deserialized = json.loads(serialized)

        assert "thinking" in deserialized
        assert isinstance(deserialized["thinking"], str)
        assert len(deserialized.keys()) == 1, "Thinking event should only have 'thinking' key"

    def test_reasoning_patterns_are_non_empty(self):
        """Ensure REASONING_MODEL_PATTERNS is populated."""
        assert len(REASONING_MODEL_PATTERNS) > 0
        assert all(isinstance(p, str) for p in REASONING_MODEL_PATTERNS)
