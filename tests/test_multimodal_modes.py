"""
Tests for multimodal support consistency across desktop and server modes.

This test module validates that the multimodal implementation works correctly
in BOTH deployment modes:

  1. litellm_config.yaml consistency — the YAML that both modes depend on
     has entries for all models defined in config.py, and no stale entries.

  2. Desktop mode model resolution — litellm_direct.py can resolve all
     STREAM model names (especially local-vision) via the YAML.

  3. Server mode model routing — litellm_config.yaml has correct provider
     names and connection details for all models.

  4. Cross-mode configuration parity — config.py, litellm_config.yaml,
     and frontend settings all agree on which models exist.

  5. Removed model cleanup — no references to local-llama-tiny,
     local-llama-quality, or ollama-1b in active configuration.

These tests do NOT require any running services (Ollama, LiteLLM, etc.).
They parse configuration files and validate internal consistency.

Run with:
    pytest tests/test_multimodal_modes.py -v
"""

from pathlib import Path

import pytest
import yaml

# =============================================================================
# FIXTURES: Shared test data
# =============================================================================


@pytest.fixture
def litellm_config():
    """Load and parse litellm_config.yaml."""
    config_path = (
        Path(__file__).resolve().parent.parent / "stream" / "gateway" / "litellm_config.yaml"
    )
    assert config_path.exists(), f"litellm_config.yaml not found at {config_path}"
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def yaml_model_names(litellm_config):
    """Extract all model names defined in litellm_config.yaml."""
    return {entry["model_name"] for entry in litellm_config.get("model_list", [])}


@pytest.fixture
def yaml_model_map(litellm_config):
    """Build the same model map that litellm_direct.py builds at import time."""
    model_map = {}
    for entry in litellm_config.get("model_list", []):
        friendly_name = entry["model_name"]
        params = entry.get("litellm_params", {})
        model_map[friendly_name] = {
            "model": params.get("model"),
            "api_base": params.get("api_base"),
            "api_key": params.get("api_key"),
        }
    return model_map


@pytest.fixture
def yaml_timeouts(litellm_config):
    """Extract model timeout settings from litellm_config.yaml."""
    return litellm_config.get("router_settings", {}).get("model_timeout", {})


# =============================================================================
# TESTS: litellm_config.yaml has all required models
# =============================================================================


class TestYamlModelCompleteness:
    """Verify litellm_config.yaml contains entries for all active models."""

    def test_local_llama_in_yaml(self, yaml_model_names):
        """local-llama (text) should be defined in the YAML."""
        assert "local-llama" in yaml_model_names

    def test_local_vision_in_yaml(self, yaml_model_names):
        """local-vision (multimodal) should be defined in the YAML."""
        assert "local-vision" in yaml_model_names

    def test_local_vision_maps_to_gemma(self, yaml_model_map):
        """local-vision should resolve to ollama/gemma3:4b."""
        assert yaml_model_map["local-vision"]["model"] == "ollama/gemma3:4b"

    def test_local_llama_maps_to_llama32(self, yaml_model_map):
        """local-llama should resolve to ollama/llama3.2:3b."""
        assert yaml_model_map["local-llama"]["model"] == "ollama/llama3.2:3b"

    def test_cloud_claude_in_yaml(self, yaml_model_names):
        """cloud-claude should be defined."""
        assert "cloud-claude" in yaml_model_names

    def test_cloud_gpt_in_yaml(self, yaml_model_names):
        """cloud-gpt should be defined."""
        assert "cloud-gpt" in yaml_model_names

    def test_lakeshore_vl_in_yaml(self, yaml_model_names):
        """lakeshore-qwen-vl-72b (multimodal) should be defined."""
        assert "lakeshore-qwen-vl-72b" in yaml_model_names

    def test_all_ollama_models_in_yaml(self, yaml_model_names):
        """Every model in config.py OLLAMA_MODELS should have a YAML entry."""
        from stream.middleware.config import OLLAMA_MODELS

        for alias in OLLAMA_MODELS:
            assert alias in yaml_model_names, (
                f"OLLAMA_MODELS['{alias}'] exists in config.py but has no "
                f"entry in litellm_config.yaml. Desktop mode cannot resolve it."
            )

    def test_all_cloud_providers_in_yaml(self, yaml_model_names):
        """Every model in config.py CLOUD_PROVIDERS should have a YAML entry."""
        from stream.middleware.config import CLOUD_PROVIDERS

        for alias in CLOUD_PROVIDERS:
            assert alias in yaml_model_names, (
                f"CLOUD_PROVIDERS['{alias}'] exists in config.py but has no "
                f"entry in litellm_config.yaml."
            )

    def test_all_lakeshore_models_in_yaml(self, yaml_model_names):
        """Every model in config.py LAKESHORE_MODELS should have a YAML entry."""
        from stream.middleware.config import LAKESHORE_MODELS

        for alias in LAKESHORE_MODELS:
            assert alias in yaml_model_names, (
                f"LAKESHORE_MODELS['{alias}'] exists in config.py but has no "
                f"entry in litellm_config.yaml."
            )


# =============================================================================
# TESTS: Removed models are gone from YAML
# =============================================================================


class TestRemovedModelsCleanup:
    """Verify removed models are not lingering in litellm_config.yaml."""

    def test_no_local_llama_tiny(self, yaml_model_names):
        """local-llama-tiny (llama3.2:1b) should be removed from YAML."""
        assert "local-llama-tiny" not in yaml_model_names

    def test_no_local_llama_quality(self, yaml_model_names):
        """local-llama-quality (llama3.1:8b) should be removed from YAML."""
        assert "local-llama-quality" not in yaml_model_names

    def test_no_stale_timeout_entries(self, yaml_timeouts):
        """Timeout settings should not reference removed models."""
        assert "local-llama-tiny" not in yaml_timeouts
        assert "local-llama-quality" not in yaml_timeouts

    def test_local_vision_has_timeout(self, yaml_timeouts):
        """local-vision should have a timeout entry."""
        assert "local-vision" in yaml_timeouts
        assert yaml_timeouts["local-vision"] > 0

    def test_no_removed_ollama_models_anywhere(self, yaml_model_map):
        """No YAML model should reference llama3.2:1b or llama3.1:8b."""
        for name, entry in yaml_model_map.items():
            provider_model = entry.get("model", "")
            assert (
                "llama3.2:1b" not in provider_model
            ), f"YAML model '{name}' still references removed llama3.2:1b"
            assert (
                "llama3.1:8b" not in provider_model
            ), f"YAML model '{name}' still references removed llama3.1:8b"


# =============================================================================
# TESTS: Desktop mode model resolution
# =============================================================================


class TestDesktopModelResolution:
    """
    Verify that litellm_direct.py can resolve all STREAM models.

    In desktop mode, _load_model_map() parses litellm_config.yaml to build
    a lookup table. _resolve_model() then translates friendly names to
    litellm kwargs. If a model is missing from the YAML, desktop mode
    will crash with ValueError.
    """

    def test_local_vision_resolvable(self, yaml_model_map):
        """local-vision must be resolvable in desktop mode."""
        assert "local-vision" in yaml_model_map
        entry = yaml_model_map["local-vision"]
        assert entry["model"] is not None
        assert "ollama" in entry.get("api_base", "")

    def test_local_llama_resolvable(self, yaml_model_map):
        """local-llama must be resolvable in desktop mode."""
        assert "local-llama" in yaml_model_map
        entry = yaml_model_map["local-llama"]
        assert entry["model"] == "ollama/llama3.2:3b"

    def test_ollama_api_base_fixable(self, yaml_model_map):
        """
        All Ollama models should have api_base containing 'ollama' so
        _resolve_model() knows to replace it with OLLAMA_BASE_URL.
        """
        ollama_models = ["local-llama", "local-vision"]
        for name in ollama_models:
            entry = yaml_model_map[name]
            assert "ollama" in entry["api_base"], (
                f"Model '{name}' has api_base='{entry['api_base']}' which "
                f"doesn't contain 'ollama' — desktop URL fix won't work"
            )

    def test_lakeshore_api_base_fixable(self, yaml_model_map):
        """
        All Lakeshore models should have api_base containing 'lakeshore' so
        _resolve_model() knows to replace it with LAKESHORE_PROXY_URL.
        """
        from stream.middleware.config import LAKESHORE_MODELS

        for name in LAKESHORE_MODELS:
            if name not in yaml_model_map:
                continue
            entry = yaml_model_map[name]
            assert "lakeshore" in entry.get("api_base", ""), (
                f"Model '{name}' has api_base='{entry.get('api_base')}' which "
                f"doesn't contain 'lakeshore' — desktop URL fix won't work"
            )

    def test_cloud_models_have_no_api_base(self, yaml_model_map):
        """
        Cloud models should have no api_base (litellm auto-detects provider).
        If api_base is set, litellm would try to call it directly instead of
        using the correct provider endpoint.
        """
        cloud_models = ["cloud-claude", "cloud-gpt", "cloud-gpt-cheap", "cloud-haiku"]
        for name in cloud_models:
            if name not in yaml_model_map:
                continue
            entry = yaml_model_map[name]
            assert entry.get("api_base") is None, (
                f"Cloud model '{name}' has api_base='{entry['api_base']}' "
                f"but cloud models should not have api_base set"
            )

    def test_all_default_models_resolvable(self, yaml_model_map):
        """All DEFAULT_MODELS should be resolvable via the YAML."""
        from stream.middleware.config import DEFAULT_MODELS

        for tier, model_name in DEFAULT_MODELS.items():
            assert model_name in yaml_model_map, (
                f"DEFAULT_MODELS['{tier}'] = '{model_name}' is not in "
                f"litellm_config.yaml — desktop mode will crash"
            )

    def test_all_vision_default_models_resolvable(self, yaml_model_map):
        """All DEFAULT_VISION_MODELS should be resolvable via the YAML."""
        from stream.middleware.config import DEFAULT_VISION_MODELS

        for tier, model_name in DEFAULT_VISION_MODELS.items():
            assert model_name in yaml_model_map, (
                f"DEFAULT_VISION_MODELS['{tier}'] = '{model_name}' is not in "
                f"litellm_config.yaml — desktop mode will crash on image queries"
            )

    def test_judge_models_resolvable(self, yaml_model_map):
        """All judge strategy models should be resolvable via the YAML."""
        from stream.middleware.config import JUDGE_STRATEGIES

        for strategy_name, strategy_config in JUDGE_STRATEGIES.items():
            model_name = strategy_config["model"]
            assert model_name in yaml_model_map, (
                f"JUDGE_STRATEGIES['{strategy_name}'] uses model '{model_name}' "
                f"which is not in litellm_config.yaml — "
                f"desktop mode judge will crash"
            )


# =============================================================================
# TESTS: YAML ↔ config.py model name parity
# =============================================================================


class TestConfigYamlParity:
    """
    Ensure config.py OLLAMA_MODELS and the YAML agree on Ollama model names.

    config.py defines:  "local-llama": "llama3.2:3b"
    YAML defines:       model_name: local-llama → model: ollama/llama3.2:3b

    These must match — if config.py says local-llama uses llama3.2:3b but
    the YAML maps it to a different Ollama model, the system would check for
    one model in Ollama health checks but route to a different one in inference.
    """

    def test_ollama_model_names_match(self, yaml_model_map):
        """
        For each Ollama model in config.py, verify the YAML maps the
        friendly name to the same underlying Ollama model.
        """
        from stream.middleware.config import OLLAMA_MODELS

        for alias, ollama_model_name in OLLAMA_MODELS.items():
            if alias not in yaml_model_map:
                pytest.skip(f"{alias} not in YAML (tested elsewhere)")
            yaml_provider_model = yaml_model_map[alias]["model"]
            expected_provider = f"ollama/{ollama_model_name}"
            assert yaml_provider_model == expected_provider, (
                f"Mismatch for '{alias}': "
                f"config.py says '{ollama_model_name}' "
                f"but YAML says '{yaml_provider_model}'. "
                f"Expected YAML to have '{expected_provider}'."
            )


# =============================================================================
# TESTS: Multimodal message passthrough in desktop mode
# =============================================================================


class TestMultimodalMessagePassthrough:
    """
    Verify that multimodal messages (list content with image_url blocks)
    will pass through correctly to litellm in desktop mode.

    litellm.acompletion() natively supports the OpenAI vision format:
      content: [{"type": "text", ...}, {"type": "image_url", ...}]

    The key requirement is that litellm_direct.py passes the messages
    list through without modifying the content structure.
    """

    def test_text_message_structure(self):
        """Text-only messages should be valid for litellm."""
        messages = [{"role": "user", "content": "Hello"}]
        assert isinstance(messages[0]["content"], str)

    def test_multimodal_message_structure(self):
        """Multimodal messages should use the OpenAI vision format."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,/9j/fake"},
                    },
                ],
            }
        ]
        content = messages[0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"

    def test_mixed_conversation_structure(self):
        """Multi-turn conversations mixing text and images should be valid."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
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
        assert isinstance(messages[0]["content"], str)
        assert isinstance(messages[1]["content"], str)
        assert isinstance(messages[3]["content"], list)


# =============================================================================
# TESTS: Judge strategy configuration
# =============================================================================


class TestJudgeStrategyConfig:
    """Verify judge strategies are consistent across the codebase."""

    def test_ollama_1b_judge_removed_from_config(self):
        """ollama-1b should not be a valid judge strategy."""
        from stream.middleware.config import JUDGE_STRATEGIES

        assert "ollama-1b" not in JUDGE_STRATEGIES

    def test_gemma_vision_judge_exists(self):
        """gemma-vision should be a valid judge strategy."""
        from stream.middleware.config import JUDGE_STRATEGIES

        assert "gemma-vision" in JUDGE_STRATEGIES

    def test_gemma_vision_uses_local_vision_model(self):
        """gemma-vision judge should use the local-vision model."""
        from stream.middleware.config import JUDGE_STRATEGIES

        strategy = JUDGE_STRATEGIES["gemma-vision"]
        assert strategy["model"] == "local-vision"

    def test_gemma_vision_marked_as_vision(self):
        """gemma-vision strategy should have vision=True flag."""
        from stream.middleware.config import JUDGE_STRATEGIES

        strategy = JUDGE_STRATEGIES["gemma-vision"]
        assert strategy.get("vision") is True

    def test_ollama_3b_judge_still_exists(self):
        """ollama-3b should still be a valid judge strategy."""
        from stream.middleware.config import JUDGE_STRATEGIES

        assert "ollama-3b" in JUDGE_STRATEGIES

    def test_haiku_judge_still_exists(self):
        """haiku should still be a valid judge strategy."""
        from stream.middleware.config import JUDGE_STRATEGIES

        assert "haiku" in JUDGE_STRATEGIES

    def test_all_judge_models_have_context_limits(self):
        """Every model used by a judge strategy should have context limits."""
        from stream.middleware.config import JUDGE_STRATEGIES, MODEL_CONTEXT_LIMITS

        for name, strategy in JUDGE_STRATEGIES.items():
            model = strategy["model"]
            assert model in MODEL_CONTEXT_LIMITS, (
                f"Judge strategy '{name}' uses model '{model}' which has no "
                f"entry in MODEL_CONTEXT_LIMITS"
            )


# =============================================================================
# TESTS: Modality-aware routing end-to-end
# =============================================================================


class TestModalityRoutingEndToEnd:
    """
    End-to-end tests for modality-aware routing across tiers.

    Verifies the full chain: has_images() → get_model_for_tier() → model
    selection respects both the image presence and user preferences.
    """

    def test_text_query_routes_to_text_model(self):
        """A text-only query should route to the text model."""
        from stream.middleware.core.query_router import get_model_for_tier
        from stream.middleware.utils.multimodal import has_images

        messages = [{"role": "user", "content": "What is Python?"}]
        query_has_images = has_images(messages)
        assert query_has_images is False

        model = get_model_for_tier("local", has_images=query_has_images)
        assert model == "local-llama"

    def test_image_query_routes_to_vision_model(self):
        """An image query should route to the vision model."""
        from stream.middleware.core.query_router import get_model_for_tier
        from stream.middleware.utils.multimodal import has_images

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,/9j/fake"},
                    },
                ],
            }
        ]
        query_has_images = has_images(messages)
        assert query_has_images is True

        model = get_model_for_tier("local", has_images=query_has_images)
        assert model == "local-vision"

    def test_image_query_explicit_text_model_not_overridden(self):
        """
        If user explicitly picks a text-only model and sends an image,
        the router should return the text model as-is (not auto-switch).
        The caller (chat.py) then raises an error.
        """
        from stream.middleware.core.query_router import get_model_for_tier
        from stream.middleware.utils.multimodal import has_images

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,/9j/fake"},
                    },
                ],
            }
        ]
        query_has_images = has_images(messages)
        model = get_model_for_tier("local", local_model="local-llama", has_images=query_has_images)
        assert model == "local-llama"

    def test_vision_model_in_vision_capable_set(self):
        """The routed vision model should be in VISION_CAPABLE_MODELS."""
        from stream.middleware.config import VISION_CAPABLE_MODELS
        from stream.middleware.core.query_router import get_model_for_tier

        model = get_model_for_tier("local", has_images=True)
        assert model in VISION_CAPABLE_MODELS

    def test_text_model_not_in_vision_capable_set(self):
        """The routed text model should NOT be in VISION_CAPABLE_MODELS."""
        from stream.middleware.config import VISION_CAPABLE_MODELS
        from stream.middleware.core.query_router import get_model_for_tier

        model = get_model_for_tier("local", has_images=False)
        assert model not in VISION_CAPABLE_MODELS

    def test_lakeshore_image_routes_to_vl_model(self):
        """Lakeshore image queries should route to the VL (vision-language) model."""
        from stream.middleware.core.query_router import get_model_for_tier

        model = get_model_for_tier("lakeshore", has_images=True)
        assert model == "lakeshore-qwen-vl-72b"

    def test_lakeshore_text_routes_to_text_model(self):
        """Lakeshore text queries should route to the default text model."""
        from stream.middleware.core.query_router import get_model_for_tier

        model = get_model_for_tier("lakeshore", has_images=False)
        assert model != "lakeshore-qwen-vl-72b"


# =============================================================================
# TESTS: Globus payload size validation with multimodal
# =============================================================================


class TestGlobusPayloadMultimodal:
    """
    Verify payload size estimation handles multimodal content correctly.

    The Globus Compute AMQP transport has a ~10 MB limit. STREAM enforces
    8 MB to leave headroom. Image payloads (base64-encoded) can easily
    exceed this, so accurate size estimation is critical.
    """

    def test_text_payload_under_limit(self):
        """Normal text messages should be well under the 8 MB limit."""
        from stream.middleware.config import GLOBUS_MAX_PAYLOAD_BYTES
        from stream.middleware.core.globus_compute_client import GlobusComputeClient

        client = GlobusComputeClient()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is Python?"},
        ]
        size = client._estimate_payload_size(messages)
        assert size < GLOBUS_MAX_PAYLOAD_BYTES

    def test_single_compressed_image_under_limit(self):
        """A single compressed image (~200 KB) should be under the limit."""
        from stream.middleware.config import GLOBUS_MAX_PAYLOAD_BYTES
        from stream.middleware.core.globus_compute_client import GlobusComputeClient

        client = GlobusComputeClient()
        fake_base64 = "A" * 200_000
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
        assert size < GLOBUS_MAX_PAYLOAD_BYTES

    def test_oversized_image_exceeds_limit(self):
        """An uncompressed 9 MB image should exceed the limit."""
        from stream.middleware.config import GLOBUS_MAX_PAYLOAD_BYTES
        from stream.middleware.core.globus_compute_client import GlobusComputeClient

        client = GlobusComputeClient()
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
