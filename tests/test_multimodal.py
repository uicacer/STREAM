"""
Tests for multimodal (image) support in STREAM.

This test module validates the core multimodal functionality:
  1. Multimodal utility functions (extract_text_content, has_images, count_images)
  2. Token estimation with images
  3. Message model validation (str and list content)
  4. Payload size estimation for Globus Compute

These tests do NOT require any running services (Ollama, vLLM, etc.).
They test pure logic functions that can run anywhere.

Run with:
    pytest tests/test_multimodal.py -v
"""

import pytest

# =============================================================================
# FIXTURES: Reusable test data
# =============================================================================
# Fixtures let us define test data once and reuse it across many tests.
# pytest automatically passes these to any test function that has a
# parameter with the same name as the fixture.


@pytest.fixture
def text_only_message():
    """A simple text-only message (the traditional format)."""
    return {"role": "user", "content": "What is Python?"}


@pytest.fixture
def multimodal_message():
    """A message with text and one image (OpenAI vision format)."""
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this image?"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQfake"},
            },
        ],
    }


@pytest.fixture
def multi_image_message():
    """A message with text and two images."""
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "Compare these two images"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,/9j/image1fake"},
            },
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,iVBORimage2fake"},
            },
        ],
    }


@pytest.fixture
def image_only_message():
    """A message with only an image and no text."""
    return {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQfake"},
            },
        ],
    }


@pytest.fixture
def conversation_with_images():
    """A multi-turn conversation where one message has an image."""
    return [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help you?"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this photo?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,/9j/fake"},
                },
            ],
        },
    ]


@pytest.fixture
def conversation_text_only():
    """A multi-turn conversation with no images."""
    return [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "What is machine learning?"},
    ]


# =============================================================================
# TESTS: extract_text_content()
# =============================================================================


class TestExtractTextContent:
    """Tests for the extract_text_content utility function."""

    def test_string_content_returns_as_is(self):
        """String content should be returned unchanged."""
        from stream.middleware.utils.multimodal import extract_text_content

        result = extract_text_content("What is Python?")
        assert result == "What is Python?"

    def test_empty_string_returns_empty(self):
        """Empty string should return empty string."""
        from stream.middleware.utils.multimodal import extract_text_content

        result = extract_text_content("")
        assert result == ""

    def test_multimodal_extracts_text_blocks(self, multimodal_message):
        """Should extract text from multimodal content blocks."""
        from stream.middleware.utils.multimodal import extract_text_content

        result = extract_text_content(multimodal_message["content"])
        assert result == "What is in this image?"

    def test_image_only_returns_empty(self, image_only_message):
        """Content with only images should return empty string."""
        from stream.middleware.utils.multimodal import extract_text_content

        result = extract_text_content(image_only_message["content"])
        assert result == ""

    def test_multiple_text_blocks_joined(self):
        """Multiple text blocks should be joined with spaces."""
        from stream.middleware.utils.multimodal import extract_text_content

        content = [
            {"type": "text", "text": "First part"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,fake"}},
            {"type": "text", "text": "Second part"},
        ]
        result = extract_text_content(content)
        assert result == "First part Second part"

    def test_empty_list_returns_empty(self):
        """Empty content list should return empty string."""
        from stream.middleware.utils.multimodal import extract_text_content

        result = extract_text_content([])
        assert result == ""


# =============================================================================
# TESTS: has_images()
# =============================================================================


class TestHasImages:
    """Tests for the has_images utility function."""

    def test_text_only_messages_return_false(self, conversation_text_only):
        """Text-only conversations should return False."""
        from stream.middleware.utils.multimodal import has_images

        assert has_images(conversation_text_only) is False

    def test_single_text_message_returns_false(self, text_only_message):
        """A single text message should return False."""
        from stream.middleware.utils.multimodal import has_images

        assert has_images([text_only_message]) is False

    def test_message_with_image_returns_true(self, multimodal_message):
        """A message containing an image should return True."""
        from stream.middleware.utils.multimodal import has_images

        assert has_images([multimodal_message]) is True

    def test_conversation_with_image_returns_true(self, conversation_with_images):
        """Should detect images in multi-turn conversations."""
        from stream.middleware.utils.multimodal import has_images

        assert has_images(conversation_with_images) is True

    def test_empty_messages_returns_false(self):
        """Empty message list should return False."""
        from stream.middleware.utils.multimodal import has_images

        assert has_images([]) is False

    def test_image_only_message_returns_true(self, image_only_message):
        """A message with only an image (no text) should return True."""
        from stream.middleware.utils.multimodal import has_images

        assert has_images([image_only_message]) is True


# =============================================================================
# TESTS: count_images()
# =============================================================================


class TestCountImages:
    """Tests for the count_images utility function."""

    def test_no_images_returns_zero(self, conversation_text_only):
        """Text-only conversations should have zero images."""
        from stream.middleware.utils.multimodal import count_images

        assert count_images(conversation_text_only) == 0

    def test_one_image_returns_one(self, multimodal_message):
        """A message with one image should count as 1."""
        from stream.middleware.utils.multimodal import count_images

        assert count_images([multimodal_message]) == 1

    def test_two_images_returns_two(self, multi_image_message):
        """A message with two images should count as 2."""
        from stream.middleware.utils.multimodal import count_images

        assert count_images([multi_image_message]) == 2

    def test_images_across_messages(self, multimodal_message, multi_image_message):
        """Should count images across multiple messages."""
        from stream.middleware.utils.multimodal import count_images

        # 1 image + 2 images = 3 total
        assert count_images([multimodal_message, multi_image_message]) == 3

    def test_empty_messages_returns_zero(self):
        """Empty message list should have zero images."""
        from stream.middleware.utils.multimodal import count_images

        assert count_images([]) == 0


# =============================================================================
# TESTS: Token Estimation
# =============================================================================


class TestTokenEstimation:
    """Tests for multimodal-aware token estimation."""

    def test_text_only_estimation(self):
        """Text-only messages should estimate at ~4 chars per token."""
        from stream.middleware.utils.token_estimator import estimate_tokens

        messages = [{"role": "user", "content": "Hello world"}]  # 11 chars
        tokens = estimate_tokens(messages)
        assert tokens == 11 // 4  # 2 tokens

    def test_image_adds_fixed_tokens(self):
        """Each image should add 765 tokens to the estimate."""
        from stream.middleware.utils.token_estimator import estimate_tokens

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},  # 13 chars = 3 tokens
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,/9j/verylongbase64string"},
                    },
                ],
            }
        ]
        tokens = estimate_tokens(messages)
        text_tokens = 13 // 4  # 3
        image_tokens = 765  # Fixed per image
        assert tokens == text_tokens + image_tokens

    def test_multiple_images_add_correctly(self):
        """Multiple images should each contribute 765 tokens."""
        from stream.middleware.utils.token_estimator import estimate_tokens

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare"},  # 7 chars = 1 token
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,fake1"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,fake2"},
                    },
                ],
            }
        ]
        tokens = estimate_tokens(messages)
        text_tokens = 7 // 4  # 1
        image_tokens = 2 * 765  # 1530
        assert tokens == text_tokens + image_tokens

    def test_base64_not_counted_as_text(self):
        """
        The base64 image data should NOT be counted as text characters.

        This was the bug before multimodal support: len(str(content)) would
        convert the entire list (including base64 data) to a string and count
        all characters. A 500 KB base64 string would produce ~125,000 "tokens",
        causing every image query to be rejected as "context too long."
        """
        from stream.middleware.utils.token_estimator import estimate_tokens

        # Create a message with a large fake base64 string (500 KB)
        large_base64 = "A" * 500_000
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{large_base64}"},
                    },
                ],
            }
        ]
        tokens = estimate_tokens(messages)
        # Should be: text tokens + 765 (for 1 image)
        # NOT: 500,000+ (from counting base64 chars)
        assert tokens < 1000  # Well under 1000, not 125,000+

    def test_mixed_conversation_estimation(self):
        """Mixed text and multimodal messages should estimate correctly."""
        from stream.middleware.utils.token_estimator import estimate_tokens

        messages = [
            {"role": "system", "content": "You are helpful."},  # 15 chars = 3 tokens
            {"role": "user", "content": "Hello"},  # 5 chars = 1 token
            {"role": "assistant", "content": "Hi there!"},  # 9 chars = 2 tokens
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},  # 13 chars = 3 tokens
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,fake"},
                    },  # 765 tokens
                ],
            },
        ]
        tokens = estimate_tokens(messages)
        expected_text = (15 + 5 + 9 + 13) // 4  # 42 // 4 = 10
        expected_images = 765
        assert tokens == expected_text + expected_images


# =============================================================================
# TESTS: Message Pydantic Model
# =============================================================================


class TestMessageModel:
    """Tests for the updated Message Pydantic model."""

    def test_string_content_accepted(self):
        """Traditional string content should still work."""
        from stream.middleware.routes.chat import Message

        msg = Message(role="user", content="Hello!")
        assert msg.content == "Hello!"
        assert msg.role == "user"

    def test_list_content_accepted(self):
        """Multimodal list content should be accepted."""
        from stream.middleware.routes.chat import Message

        content = [
            {"type": "text", "text": "What is this?"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,fake"},
            },
        ]
        msg = Message(role="user", content=content)
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2

    def test_model_dump_preserves_format(self):
        """model_dump() should preserve the content format (str or list)."""
        from stream.middleware.routes.chat import Message

        # String content
        msg_str = Message(role="user", content="Hello")
        dump_str = msg_str.model_dump()
        assert isinstance(dump_str["content"], str)

        # List content
        content_list = [{"type": "text", "text": "Hi"}]
        msg_list = Message(role="user", content=content_list)
        dump_list = msg_list.model_dump()
        assert isinstance(dump_list["content"], list)

    def test_empty_string_content_accepted(self):
        """Empty string content should be accepted (edge case)."""
        from stream.middleware.routes.chat import Message

        msg = Message(role="user", content="")
        assert msg.content == ""


# =============================================================================
# TESTS: Vision Model Configuration
# =============================================================================


class TestVisionConfig:
    """Tests for the vision model configuration in config.py."""

    def test_vision_capable_models_set_exists(self):
        """VISION_CAPABLE_MODELS should be defined and non-empty."""
        from stream.middleware.config import VISION_CAPABLE_MODELS

        assert isinstance(VISION_CAPABLE_MODELS, set)
        assert len(VISION_CAPABLE_MODELS) > 0

    def test_local_vision_in_ollama_models(self):
        """local-vision should be a valid Ollama model."""
        from stream.middleware.config import OLLAMA_MODELS

        assert "local-vision" in OLLAMA_MODELS
        assert OLLAMA_MODELS["local-vision"] == "gemma3:4b"

    def test_local_vision_is_vision_capable(self):
        """local-vision should be in the vision-capable set."""
        from stream.middleware.config import VISION_CAPABLE_MODELS

        assert "local-vision" in VISION_CAPABLE_MODELS

    def test_lakeshore_vl_is_vision_capable(self):
        """lakeshore-qwen-vl-72b should be in the vision-capable set."""
        from stream.middleware.config import VISION_CAPABLE_MODELS

        assert "lakeshore-qwen-vl-72b" in VISION_CAPABLE_MODELS

    def test_cloud_models_are_vision_capable(self):
        """All cloud models should be vision-capable."""
        from stream.middleware.config import VISION_CAPABLE_MODELS

        assert "cloud-claude" in VISION_CAPABLE_MODELS
        assert "cloud-gpt" in VISION_CAPABLE_MODELS

    def test_text_only_models_not_vision_capable(self):
        """Text-only models should NOT be in the vision-capable set."""
        from stream.middleware.config import VISION_CAPABLE_MODELS

        assert "local-llama" not in VISION_CAPABLE_MODELS
        assert "lakeshore-qwen-vl-72b" in VISION_CAPABLE_MODELS

    def test_default_vision_models_exist(self):
        """DEFAULT_VISION_MODELS should have entries for all tiers."""
        from stream.middleware.config import DEFAULT_VISION_MODELS

        assert "local" in DEFAULT_VISION_MODELS
        assert "lakeshore" in DEFAULT_VISION_MODELS
        assert "cloud" in DEFAULT_VISION_MODELS

    def test_default_vision_models_are_vision_capable(self):
        """Default vision models should actually be vision-capable."""
        from stream.middleware.config import DEFAULT_VISION_MODELS, VISION_CAPABLE_MODELS

        for tier, model in DEFAULT_VISION_MODELS.items():
            assert model in VISION_CAPABLE_MODELS, (
                f"Default vision model for {tier} ({model}) " f"is not in VISION_CAPABLE_MODELS"
            )

    def test_removed_models_not_in_ollama(self):
        """llama3.2:1b and llama3.1:8b should be removed from OLLAMA_MODELS."""
        from stream.middleware.config import OLLAMA_MODELS

        assert "local-llama-tiny" not in OLLAMA_MODELS
        assert "local-llama-quality" not in OLLAMA_MODELS

    def test_gemma_vision_judge_strategy_exists(self):
        """gemma-vision should be a valid judge strategy."""
        from stream.middleware.config import JUDGE_STRATEGIES

        assert "gemma-vision" in JUDGE_STRATEGIES
        assert JUDGE_STRATEGIES["gemma-vision"]["model"] == "local-vision"
        assert JUDGE_STRATEGIES["gemma-vision"].get("vision") is True

    def test_ollama_1b_judge_removed(self):
        """ollama-1b judge strategy should be removed."""
        from stream.middleware.config import JUDGE_STRATEGIES

        assert "ollama-1b" not in JUDGE_STRATEGIES

    def test_context_limits_for_new_models(self):
        """New models should have context limits defined."""
        from stream.middleware.config import MODEL_CONTEXT_LIMITS

        assert "local-vision" in MODEL_CONTEXT_LIMITS
        assert MODEL_CONTEXT_LIMITS["local-vision"]["total"] > 0

    def test_globus_max_payload_constant(self):
        """GLOBUS_MAX_PAYLOAD_BYTES should be defined and reasonable."""
        from stream.middleware.config import GLOBUS_MAX_PAYLOAD_BYTES

        assert GLOBUS_MAX_PAYLOAD_BYTES == 8 * 1024 * 1024  # 8 MB

    def test_globus_max_image_bytes_constant(self):
        """GLOBUS_MAX_IMAGE_BYTES should be 6 MB."""
        from stream.middleware.config import GLOBUS_MAX_IMAGE_BYTES

        assert GLOBUS_MAX_IMAGE_BYTES == 6 * 1024 * 1024  # 6 MB

    def test_image_budget_within_payload_limit(self):
        """Image budget must be less than the total payload limit."""
        from stream.middleware.config import GLOBUS_MAX_IMAGE_BYTES, GLOBUS_MAX_PAYLOAD_BYTES

        assert GLOBUS_MAX_IMAGE_BYTES < GLOBUS_MAX_PAYLOAD_BYTES
        headroom = GLOBUS_MAX_PAYLOAD_BYTES - GLOBUS_MAX_IMAGE_BYTES
        assert headroom >= 1 * 1024 * 1024  # At least 1 MB for text/serialization


# =============================================================================
# TESTS: strip_old_images()
# =============================================================================


class TestStripOldImages:
    """Tests for the strip_old_images utility function.

    strip_old_images removes image data from all messages EXCEPT the latest
    user message. This is critical for keeping Globus Compute payloads under
    the 10 MB limit during long conversations with multiple images.
    """

    def test_empty_messages_returns_empty(self):
        """Empty list should be returned as-is."""
        from stream.middleware.utils.multimodal import strip_old_images

        assert strip_old_images([]) == []

    def test_single_text_message_unchanged(self):
        """A single text-only message should pass through unchanged."""
        from stream.middleware.utils.multimodal import strip_old_images

        messages = [{"role": "user", "content": "Hello"}]
        result = strip_old_images(messages)
        assert result == messages

    def test_single_image_message_preserved(self):
        """The only user message (also the latest) should keep its images."""
        from stream.middleware.utils.multimodal import strip_old_images

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
                ],
            }
        ]
        result = strip_old_images(messages)
        assert len(result) == 1
        assert isinstance(result[0]["content"], list)
        assert any(b.get("type") == "image_url" for b in result[0]["content"])

    def test_latest_user_message_images_preserved(self):
        """Images in the latest user message must be kept intact."""
        from stream.middleware.utils.multimodal import strip_old_images

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,LATEST"}},
                ],
            },
        ]
        result = strip_old_images(messages)
        last_user = result[-1]
        assert isinstance(last_user["content"], list)
        image_blocks = [b for b in last_user["content"] if b.get("type") == "image_url"]
        assert len(image_blocks) == 1
        assert "LATEST" in image_blocks[0]["image_url"]["url"]

    def test_older_user_images_stripped(self):
        """Images in older user messages should be removed."""
        from stream.middleware.utils.multimodal import strip_old_images

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First image"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,OLD_IMAGE"}},
                ],
            },
            {"role": "assistant", "content": "I see a cat in the first image."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Second image"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,NEW_IMAGE"}},
                ],
            },
        ]
        result = strip_old_images(messages)

        # First user message: image stripped, text kept
        first_user = result[0]
        assert isinstance(first_user["content"], list)
        assert all(b.get("type") != "image_url" for b in first_user["content"])
        text_blocks = [b for b in first_user["content"] if b.get("type") == "text"]
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "First image"

        # Assistant message: unchanged (no images)
        assert result[1]["content"] == "I see a cat in the first image."

        # Last user message: image preserved
        last_user = result[2]
        assert isinstance(last_user["content"], list)
        image_blocks = [b for b in last_user["content"] if b.get("type") == "image_url"]
        assert len(image_blocks) == 1
        assert "NEW_IMAGE" in image_blocks[0]["image_url"]["url"]

    def test_image_only_older_message_becomes_placeholder(self):
        """An older message with only images (no text) should become '(image)'."""
        from stream.middleware.utils.multimodal import strip_old_images

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,ONLY_IMG"}},
                ],
            },
            {"role": "assistant", "content": "That's a photo of a sunset."},
            {"role": "user", "content": "Tell me more about the colors."},
        ]
        result = strip_old_images(messages)

        # First message had only images, so content becomes "(image)"
        assert result[0]["content"] == "(image)"
        assert result[0]["role"] == "user"

    def test_assistant_messages_never_modified(self):
        """Assistant messages (always text) should pass through unchanged."""
        from stream.middleware.utils.multimodal import strip_old_images

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": "What is AI?"},
        ]
        result = strip_old_images(messages)
        assert result[1]["content"] == "Hi! How can I help?"

    def test_does_not_modify_original_messages(self):
        """The function should return a new list, not modify the original."""
        from stream.middleware.utils.multimodal import strip_old_images

        original_image_url = "data:image/jpeg;base64,ORIGINAL"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Old image"},
                    {"type": "image_url", "image_url": {"url": original_image_url}},
                ],
            },
            {"role": "user", "content": "Latest text"},
        ]
        result = strip_old_images(messages)

        # Original should still have the image
        assert any(
            b.get("type") == "image_url"
            for b in messages[0]["content"]
            if isinstance(messages[0]["content"], list)
        )
        # Result should not
        assert result is not messages

    def test_multiple_images_in_older_message_all_stripped(self):
        """All images in an older message should be stripped, keeping text."""
        from stream.middleware.utils.multimodal import strip_old_images

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these two"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,IMG1"}},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,IMG2"}},
                ],
            },
            {"role": "assistant", "content": "The first shows a cat, the second a dog."},
            {"role": "user", "content": "Which is cuter?"},
        ]
        result = strip_old_images(messages)

        first_user = result[0]
        assert isinstance(first_user["content"], list)
        assert len(first_user["content"]) == 1  # Only the text block remains
        assert first_user["content"][0]["text"] == "Compare these two"

    def test_system_message_unchanged(self):
        """System messages should pass through unchanged."""
        from stream.middleware.utils.multimodal import strip_old_images

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        result = strip_old_images(messages)
        assert result[0]["content"] == "You are a helpful assistant."
        assert result[0]["role"] == "system"

    def test_no_user_messages(self):
        """If there are no user messages, all messages should pass through."""
        from stream.middleware.utils.multimodal import strip_old_images

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "assistant", "content": "How can I help?"},
        ]
        result = strip_old_images(messages)
        assert len(result) == 2
        assert result[0]["content"] == "You are helpful."

    def test_long_conversation_with_multiple_image_turns(self):
        """Realistic multi-turn conversation with images in multiple turns."""
        from stream.middleware.utils.multimodal import strip_old_images

        messages = [
            {"role": "system", "content": "You are a vision assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in image 1?"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,IMG_TURN1"}},
                ],
            },
            {"role": "assistant", "content": "Image 1 shows a bar chart of Q3 revenue."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "And image 2?"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,IMG_TURN2"}},
                ],
            },
            {"role": "assistant", "content": "Image 2 shows a pie chart of market share."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Now compare with image 3"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,IMG_TURN3"}},
                ],
            },
        ]
        result = strip_old_images(messages)

        # System message: unchanged
        assert result[0]["content"] == "You are a vision assistant."

        # Turn 1 user: image stripped, text kept
        assert isinstance(result[1]["content"], list)
        assert all(b.get("type") != "image_url" for b in result[1]["content"])

        # Turn 1 assistant: unchanged
        assert result[2]["content"] == "Image 1 shows a bar chart of Q3 revenue."

        # Turn 2 user: image stripped, text kept
        assert isinstance(result[3]["content"], list)
        assert all(b.get("type") != "image_url" for b in result[3]["content"])

        # Turn 2 assistant: unchanged
        assert result[4]["content"] == "Image 2 shows a pie chart of market share."

        # Turn 3 user (LATEST): image PRESERVED
        assert isinstance(result[5]["content"], list)
        image_blocks = [b for b in result[5]["content"] if b.get("type") == "image_url"]
        assert len(image_blocks) == 1
        assert "IMG_TURN3" in image_blocks[0]["image_url"]["url"]


# =============================================================================
# TESTS: Payload reduction via strip_old_images
# =============================================================================


class TestPayloadReduction:
    """Tests verifying that strip_old_images actually reduces payload size
    enough to fit within the Globus Compute limit."""

    def test_stripping_reduces_payload_size(self):
        """Stripping old images should significantly reduce payload size."""
        from stream.middleware.core.globus_compute_client import GlobusComputeClient
        from stream.middleware.utils.multimodal import strip_old_images

        client = GlobusComputeClient()

        # Simulate 3 turns, each with a ~2 MB image (6 MB total > 6 MB limit)
        fake_image = "A" * (2 * 1024 * 1024)
        messages = []
        for i in range(3):
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Turn {i+1} question"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{fake_image}"},
                        },
                    ],
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": f"Turn {i+1} response about the image.",
                }
            )

        # Add a final user message with image
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Final question"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{fake_image}"},
                    },
                ],
            }
        )

        size_before = client._estimate_payload_size(messages)
        stripped = strip_old_images(messages)
        size_after = client._estimate_payload_size(stripped)

        # Before: ~8 MB (4 images x 2 MB). After: ~2 MB (1 image)
        assert size_before > 7 * 1024 * 1024
        assert size_after < 3 * 1024 * 1024
        assert size_after < size_before / 2

    def test_stripped_payload_fits_within_globus_limit(self):
        """After stripping, a conversation with one current image should fit."""
        from stream.middleware.config import GLOBUS_MAX_PAYLOAD_BYTES
        from stream.middleware.core.globus_compute_client import GlobusComputeClient
        from stream.middleware.utils.multimodal import strip_old_images

        client = GlobusComputeClient()

        # 5 older turns with ~1 MB images + 1 current turn with ~1 MB image
        fake_image = "A" * (1 * 1024 * 1024)
        messages = []
        for i in range(5):
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Older turn {i+1}"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{fake_image}"},
                        },
                    ],
                }
            )
            messages.append({"role": "assistant", "content": f"Response {i+1}."})

        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Current question"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{fake_image}"},
                    },
                ],
            }
        )

        # Before stripping: ~6 MB (6 images x 1 MB) — might exceed 8 MB limit
        stripped = strip_old_images(messages)
        size_after = client._estimate_payload_size(stripped)

        # After stripping: ~1 MB (1 image) + text — well under 8 MB
        assert size_after < GLOBUS_MAX_PAYLOAD_BYTES

    def test_text_only_conversation_unaffected(self):
        """strip_old_images should not change text-only conversations."""
        from stream.middleware.core.globus_compute_client import GlobusComputeClient
        from stream.middleware.utils.multimodal import strip_old_images

        client = GlobusComputeClient()

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
            {"role": "user", "content": "Tell me more."},
        ]

        size_before = client._estimate_payload_size(messages)
        stripped = strip_old_images(messages)
        size_after = client._estimate_payload_size(stripped)

        assert size_before == size_after
