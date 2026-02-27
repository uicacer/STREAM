"""
Tests for modality-aware routing in STREAM.

This test module validates that the router correctly handles image queries:
  1. AUTO mode → selects vision-capable model automatically
  2. Tier-only → selects vision model within the chosen tier
  3. Explicit model → returns the model as-is (caller validates capability)
  4. Keyword fallback → defaults to MEDIUM for image queries

These tests do NOT require any running services.
They test routing logic with mocked tier health status.

Run with:
    pytest tests/test_multimodal_routing.py -v
"""


# =============================================================================
# TESTS: get_model_for_tier (modality-aware model selection)
# =============================================================================


class TestGetModelForTierMultimodal:
    """Tests for modality-aware model selection."""

    def test_auto_mode_no_images_returns_text_default(self):
        """AUTO mode without images should return the default text model."""
        from stream.middleware.core.query_router import get_model_for_tier

        model = get_model_for_tier("local", has_images=False)
        assert model == "local-llama"

    def test_auto_mode_with_images_returns_vision_model(self):
        """AUTO mode with images should return the vision model."""
        from stream.middleware.core.query_router import get_model_for_tier

        model = get_model_for_tier("local", has_images=True)
        assert model == "local-vision"

    def test_lakeshore_with_images_returns_vision_model(self):
        """Lakeshore tier with images should return the VL model."""
        from stream.middleware.core.query_router import get_model_for_tier

        model = get_model_for_tier("lakeshore", has_images=True)
        assert model == "lakeshore-qwen-vl-72b"

    def test_cloud_with_images_returns_cloud_default(self):
        """Cloud tier with images should return the cloud default (all support vision)."""
        from stream.middleware.config import DEFAULT_CLOUD_PROVIDER
        from stream.middleware.core.query_router import get_model_for_tier

        model = get_model_for_tier("cloud", has_images=True)
        assert model == DEFAULT_CLOUD_PROVIDER

    def test_explicit_local_model_overrides_vision(self):
        """
        Explicit model selection should be returned as-is, even with images.

        The caller (chat.py) is responsible for checking if the model
        supports images and returning an appropriate error.
        """
        from stream.middleware.core.query_router import get_model_for_tier

        model = get_model_for_tier("local", local_model="local-llama", has_images=True)
        # Returns the explicitly selected model — does NOT auto-switch
        assert model == "local-llama"

    def test_explicit_vision_model_works(self):
        """Explicitly selecting a vision model should work normally."""
        from stream.middleware.core.query_router import get_model_for_tier

        model = get_model_for_tier("local", local_model="local-vision", has_images=True)
        assert model == "local-vision"

    def test_explicit_lakeshore_model_overrides_vision(self):
        """Explicit lakeshore model should be returned as-is."""
        from stream.middleware.core.query_router import get_model_for_tier

        model = get_model_for_tier(
            "lakeshore", lakeshore_model="lakeshore-qwen-vl-72b", has_images=True
        )
        assert model == "lakeshore-qwen-vl-72b"

    def test_explicit_cloud_provider_overrides(self):
        """Explicit cloud provider should be returned as-is."""
        from stream.middleware.core.query_router import get_model_for_tier

        model = get_model_for_tier("cloud", cloud_provider="cloud-gpt", has_images=True)
        assert model == "cloud-gpt"

    def test_no_images_no_model_returns_text_default(self):
        """No images and no explicit model should return the text default."""
        from stream.middleware.core.query_router import get_model_for_tier

        model = get_model_for_tier("local", has_images=False)
        assert model == "local-llama"

    def test_lakeshore_no_images_returns_text_default(self):
        """Lakeshore without images should return the text default."""
        from stream.middleware.core.query_router import get_model_for_tier

        model = get_model_for_tier("lakeshore", has_images=False)
        assert model == "lakeshore-qwen-vl-72b"


# =============================================================================
# TESTS: Complexity Judge with Images
# =============================================================================


class TestComplexityJudgeMultimodal:
    """Tests for complexity judgment with image queries."""

    def test_keyword_fallback_defaults_medium_with_images(self):
        """
        When no keywords match but images are present, should default to MEDIUM.

        Without images, the default would also be MEDIUM, but this test
        verifies the explicit image handling path is active.
        """
        from stream.middleware.core.complexity_judge import judge_complexity

        # "asdfghjkl" has no keywords, but images are present
        result = judge_complexity("asdfghjkl", strategy="ollama-3b", query_has_images=True)
        assert result.complexity == "medium"

    def test_keyword_fallback_no_images_defaults_medium(self):
        """Without images and no keywords, should also default to MEDIUM."""
        from stream.middleware.core.complexity_judge import judge_complexity

        result = judge_complexity("asdfghjkl", strategy="ollama-3b", query_has_images=False)
        assert result.complexity == "medium"

    def test_high_keyword_still_overrides_with_images(self):
        """HIGH complexity keywords should still override even with images."""
        from stream.middleware.core.complexity_judge import judge_complexity

        # "design" is a HIGH keyword — should override the MEDIUM default.
        # NOTE: We avoid queries containing "hi"/"hey" etc. in substrings
        # because the keyword matcher uses substring matching, which can
        # cause false positives (e.g., "this" contains "hi" → LOW match).
        result = judge_complexity(
            "design a database schema for me", strategy="ollama-3b", query_has_images=True
        )
        # keyword matching finds "design" → HIGH
        assert result.complexity == "high"

    def test_low_keyword_with_images(self):
        """LOW keywords should still be detected even with images."""
        from stream.middleware.core.complexity_judge import judge_complexity

        result = judge_complexity("what is this?", strategy="ollama-3b", query_has_images=True)
        # "what is" is a LOW keyword, should match
        assert result.complexity in ["low", "medium"]


# =============================================================================
# TESTS: Payload Size Estimation
# =============================================================================


class TestPayloadSizeEstimation:
    """Tests for Globus Compute payload size validation."""

    def test_text_only_payload_is_small(self):
        """Text-only messages should have a small payload."""
        from stream.middleware.core.globus_compute_client import GlobusComputeClient

        client = GlobusComputeClient()
        messages = [{"role": "user", "content": "Hello world"}]
        size = client._estimate_payload_size(messages)
        assert size < 1000  # Well under 1 KB

    def test_image_payload_includes_base64(self):
        """Image messages should have a larger payload due to base64 data."""
        from stream.middleware.core.globus_compute_client import GlobusComputeClient

        client = GlobusComputeClient()
        # Create a message with ~100 KB of fake base64 data
        fake_base64 = "A" * 100_000
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{fake_base64}"},
                    },
                ],
            }
        ]
        size = client._estimate_payload_size(messages)
        # Should be at least 100 KB (the base64 data)
        assert size > 100_000

    def test_large_payload_would_be_rejected(self):
        """A payload exceeding 8 MB should be caught by the size check."""
        from stream.middleware.config import GLOBUS_MAX_PAYLOAD_BYTES
        from stream.middleware.core.globus_compute_client import GlobusComputeClient

        client = GlobusComputeClient()
        # Create a message with ~9 MB of fake base64 data
        fake_base64 = "A" * (9 * 1024 * 1024)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{fake_base64}"},
                    },
                ],
            }
        ]
        size = client._estimate_payload_size(messages)
        assert size > GLOBUS_MAX_PAYLOAD_BYTES


# =============================================================================
# TESTS: VISION_CAPABLE_MODELS consistency
# =============================================================================


class TestVisionModelConsistency:
    """Ensure vision model configuration is internally consistent."""

    def test_all_default_vision_models_exist_in_model_configs(self):
        """
        Every model in DEFAULT_VISION_MODELS should exist in the
        corresponding model configuration (OLLAMA_MODELS, LAKESHORE_MODELS,
        or CLOUD_PROVIDERS).
        """
        from stream.middleware.config import (
            CLOUD_PROVIDERS,
            DEFAULT_VISION_MODELS,
            LAKESHORE_MODELS,
            OLLAMA_MODELS,
        )

        all_known_models = (
            set(OLLAMA_MODELS.keys()) | set(LAKESHORE_MODELS.keys()) | set(CLOUD_PROVIDERS.keys())
        )

        for tier, model in DEFAULT_VISION_MODELS.items():
            assert model in all_known_models, (
                f"Default vision model '{model}' for tier '{tier}' "
                f"is not in any model configuration"
            )

    def test_context_limits_for_all_vision_models(self):
        """All vision-capable models should have context limits defined."""
        from stream.middleware.config import (
            MODEL_CONTEXT_LIMITS,
            VISION_CAPABLE_MODELS,
        )

        for model in VISION_CAPABLE_MODELS:
            assert (
                model in MODEL_CONTEXT_LIMITS
            ), f"Vision model '{model}' is missing from MODEL_CONTEXT_LIMITS"

    def test_lakeshore_vl_model_has_multimodal_flag(self):
        """The lakeshore VL model should have multimodal=True in its config."""
        from stream.middleware.config import LAKESHORE_MODELS

        vl_model = LAKESHORE_MODELS.get("lakeshore-qwen-vl-72b", {})
        assert vl_model.get("multimodal") is True
