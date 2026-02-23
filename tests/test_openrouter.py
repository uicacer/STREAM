"""
Tests for the OpenRouter cloud integration feature.

This module validates the full OpenRouter integration across multiple layers
of the STREAM middleware:

  - Config layer: CLOUD_PROVIDERS, key mappings, vision models, context limits
  - Request layer: user-provided API key fields on ChatCompletionRequest
  - Model resolution: dynamic OpenRouter model name translation
  - API endpoints: key validation and model catalog
  - Context window: fallback limits for dynamically-selected models

OpenRouter is an aggregator that gives users a single API key to access
500+ models from many providers (Anthropic, OpenAI, Google, Meta, etc.).
STREAM supports it alongside direct provider keys, so these tests verify
both paths work correctly.

Tests use mocking to avoid real HTTP/API calls, so they run fast and
don't require network access or API keys.
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from stream.middleware.config import (
    CLOUD_PROVIDER_KEY_MAPPING,
    CLOUD_PROVIDERS,
    DEFAULT_CLOUD_CONTEXT_LIMIT,
    MODEL_CONTEXT_LIMITS,
    VISION_CAPABLE_MODELS,
)
from stream.middleware.routes.chat import ChatCompletionRequest
from stream.middleware.utils.context_window import get_max_input_tokens

# =============================================================================
# FIXTURES
# =============================================================================
# Fixtures provide reusable test data and setup. pytest automatically injects
# them into any test function that declares them as a parameter.


@pytest.fixture
def openrouter_model_ids():
    """All STREAM model IDs that route through OpenRouter.

    These use the 'cloud-or-' prefix and require OPENROUTER_API_KEY.
    They're distinct from direct-provider models (cloud-claude, cloud-gpt)
    which call provider APIs without the aggregator.
    """
    return [
        model_id for model_id, info in CLOUD_PROVIDERS.items() if info["provider"] == "OpenRouter"
    ]


@pytest.fixture
def direct_model_ids():
    """All STREAM model IDs that call provider APIs directly.

    These bypass OpenRouter and use the provider's native API endpoint.
    Lower latency but requires separate keys for each provider.
    """
    return [
        model_id for model_id, info in CLOUD_PROVIDERS.items() if info["provider"] != "OpenRouter"
    ]


@pytest.fixture
def sample_messages():
    """Minimal valid message list for ChatCompletionRequest."""
    return [{"role": "user", "content": "Hello"}]


@pytest.fixture
def mock_openrouter_catalog_response():
    """Simulated response from OpenRouter's /api/v1/models endpoint.

    Includes a mix of model types: paid, free, vision-capable, and
    text-only. This exercises the catalog endpoint's categorization logic.
    """
    return {
        "data": [
            {
                "id": "anthropic/claude-sonnet-4",
                "name": "Claude Sonnet 4",
                "description": "Best for complex reasoning",
                "context_length": 200000,
                "pricing": {"prompt": "0.000003", "completion": "0.000015"},
                "architecture": {
                    "modality": "text+image->text",
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["text"],
                    "tokenizer": "Claude",
                    "instruct_type": None,
                },
                "top_provider": {},
            },
            {
                "id": "openai/gpt-4o",
                "name": "GPT-4o",
                "description": "Strong general-purpose",
                "context_length": 128000,
                "pricing": {"prompt": "0.0000025", "completion": "0.00001"},
                "architecture": {
                    "modality": "text+image+file->text",
                    "input_modalities": ["text", "image", "file"],
                    "output_modalities": ["text"],
                    "tokenizer": "GPT",
                    "instruct_type": None,
                },
                "top_provider": {},
            },
            {
                "id": "meta-llama/llama-3.1-70b-instruct:free",
                "name": "Llama 3.1 70B (free)",
                "description": "Open-source, free tier",
                "context_length": 128000,
                "pricing": {"prompt": "0", "completion": "0"},
                "architecture": {
                    "modality": "text->text",
                    "input_modalities": ["text"],
                    "output_modalities": ["text"],
                    "tokenizer": "Llama3",
                    "instruct_type": None,
                },
                "top_provider": {},
            },
            {
                "id": "google/gemini-2.0-flash-exp",
                "name": "Gemini 2.0 Flash",
                "description": "Google's fast model",
                "context_length": 1000000,
                "pricing": {"prompt": "0", "completion": "0"},
                "architecture": {
                    "modality": "text+image->text",
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["text"],
                    "tokenizer": "Gemini",
                    "instruct_type": None,
                },
                "top_provider": {},
            },
        ]
    }


@pytest.fixture
def test_client():
    """FastAPI TestClient for endpoint tests.

    We import the app inside the fixture (not at module level) to avoid
    triggering startup side effects (DB connections, health checks, etc.)
    during test collection.
    """
    from fastapi import FastAPI

    from stream.middleware.routes.models import router

    app = FastAPI()
    app.include_router(router, prefix="/v1")
    return TestClient(app)


# =============================================================================
# 1. CONFIG TESTS
# =============================================================================
# These verify that the static configuration in config.py is internally
# consistent. If someone adds a new OpenRouter model but forgets to update
# the vision set or context limits, these tests catch it.


class TestCloudProvidersConfig:
    """Verify CLOUD_PROVIDERS dict has the expected structure and entries."""

    def test_contains_openrouter_models(self, openrouter_model_ids):
        """CLOUD_PROVIDERS must include OpenRouter models (cloud-or-* prefix).

        OpenRouter models use the aggregator, so they all share one API key
        (OPENROUTER_API_KEY) and have provider='OpenRouter'.
        """
        assert len(openrouter_model_ids) > 0, "No OpenRouter models found in CLOUD_PROVIDERS"
        for model_id in openrouter_model_ids:
            assert model_id.startswith(
                "cloud-or-"
            ), f"OpenRouter model '{model_id}' should start with 'cloud-or-'"

    def test_contains_direct_provider_models(self, direct_model_ids):
        """CLOUD_PROVIDERS must include direct provider models (Anthropic, OpenAI).

        Direct models bypass OpenRouter and call the provider API directly.
        They use provider-specific API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY).
        """
        assert len(direct_model_ids) > 0, "No direct provider models found"
        providers = {CLOUD_PROVIDERS[m]["provider"] for m in direct_model_ids}
        assert "Anthropic" in providers, "Expected Anthropic in direct providers"
        assert "OpenAI" in providers, "Expected OpenAI in direct providers"

    def test_openrouter_models_use_openrouter_key(self, openrouter_model_ids):
        """All OpenRouter models should reference OPENROUTER_API_KEY.

        This ensures they all use the same aggregator key, not individual
        provider keys. The user only needs one key for all OR models.
        """
        for model_id in openrouter_model_ids:
            info = CLOUD_PROVIDERS[model_id]
            assert info["env_key"] == "OPENROUTER_API_KEY", (
                f"OpenRouter model '{model_id}' should use OPENROUTER_API_KEY, "
                f"got '{info['env_key']}'"
            )

    def test_all_cloud_providers_have_required_fields(self):
        """Every CLOUD_PROVIDERS entry must have the minimum required fields.

        Missing fields would cause KeyErrors at runtime when the routing
        or key-injection code tries to look them up.
        """
        required_fields = {"name", "provider", "description", "env_key", "key_source"}
        for model_id, info in CLOUD_PROVIDERS.items():
            missing = required_fields - set(info.keys())
            assert not missing, f"Model '{model_id}' is missing fields: {missing}"

    def test_key_source_is_user_for_all(self):
        """All current CLOUD_PROVIDERS use key_source='user'.

        In the BYOK (Bring Your Own Key) model, keys come from the user's
        browser, not from server environment variables. If we add server-side
        keys later, this test should be updated.
        """
        for model_id, info in CLOUD_PROVIDERS.items():
            assert (
                info["key_source"] == "user"
            ), f"Model '{model_id}' has key_source='{info['key_source']}', expected 'user'"


class TestCloudProviderKeyMapping:
    """Verify CLOUD_PROVIDER_KEY_MAPPING connects env vars to request fields."""

    def test_openrouter_mapping_exists(self):
        """OPENROUTER_API_KEY must map to the request body field name.

        This mapping tells the backend which ChatCompletionRequest field
        holds the user's OpenRouter key when resolving API credentials.
        """
        assert "OPENROUTER_API_KEY" in CLOUD_PROVIDER_KEY_MAPPING
        assert CLOUD_PROVIDER_KEY_MAPPING["OPENROUTER_API_KEY"] == "openrouter_api_key"

    def test_anthropic_mapping_exists(self):
        """ANTHROPIC_API_KEY must map to the anthropic_api_key field."""
        assert "ANTHROPIC_API_KEY" in CLOUD_PROVIDER_KEY_MAPPING
        assert CLOUD_PROVIDER_KEY_MAPPING["ANTHROPIC_API_KEY"] == "anthropic_api_key"

    def test_openai_mapping_exists(self):
        """OPENAI_API_KEY must map to the openai_api_key field."""
        assert "OPENAI_API_KEY" in CLOUD_PROVIDER_KEY_MAPPING
        assert CLOUD_PROVIDER_KEY_MAPPING["OPENAI_API_KEY"] == "openai_api_key"

    def test_all_provider_env_keys_are_mapped(self):
        """Every env_key used in CLOUD_PROVIDERS must appear in the mapping.

        If a model references an env_key that isn't mapped, the user's key
        would never be found — the model call would fail silently with no
        API key or use a stale env var instead.
        """
        env_keys_in_use = {info["env_key"] for info in CLOUD_PROVIDERS.values()}
        for env_key in env_keys_in_use:
            assert env_key in CLOUD_PROVIDER_KEY_MAPPING, (
                f"env_key '{env_key}' used in CLOUD_PROVIDERS but missing from "
                f"CLOUD_PROVIDER_KEY_MAPPING"
            )


class TestVisionCapableModels:
    """Verify VISION_CAPABLE_MODELS includes OpenRouter vision models."""

    def test_openrouter_vision_models_included(self):
        """OpenRouter models that support images must be in VISION_CAPABLE_MODELS.

        Without this, the router would reject image queries sent to these
        models, even though the underlying provider (GPT-4o, Claude, etc.)
        supports vision natively.
        """
        expected_vision_models = {
            "cloud-or-claude",
            "cloud-or-gpt4o",
            "cloud-or-gemini-pro",
            "cloud-or-gemini-flash",
            "cloud-or-llama-maverick",
        }
        for model in expected_vision_models:
            assert model in VISION_CAPABLE_MODELS, (
                f"Vision-capable OpenRouter model '{model}' missing from "
                f"VISION_CAPABLE_MODELS set"
            )

    def test_direct_vision_models_included(self):
        """Direct provider models with vision must also be listed.

        Claude and GPT-4o both support images via their native APIs.
        """
        expected = {"cloud-claude", "cloud-gpt", "cloud-gpt-cheap"}
        for model in expected:
            assert (
                model in VISION_CAPABLE_MODELS
            ), f"Direct vision model '{model}' missing from VISION_CAPABLE_MODELS"

    def test_text_only_models_excluded(self):
        """Text-only models (o3-mini, DeepSeek R1/V3) should NOT be in VISION_CAPABLE_MODELS.

        Adding a text-only model here would cause silent failures when
        users send images — the model can't process them.
        """
        for model in ["cloud-or-o3-mini", "cloud-or-deepseek-r1", "cloud-or-deepseek-v3"]:
            assert model not in VISION_CAPABLE_MODELS


class TestModelContextLimits:
    """Verify MODEL_CONTEXT_LIMITS has entries for all OpenRouter models."""

    def test_all_openrouter_models_have_context_limits(self, openrouter_model_ids):
        """Every statically-configured OpenRouter model needs context limits.

        Without explicit limits, the context window validator would fall back
        to DEFAULT_CLOUD_CONTEXT_LIMIT, which may not match the model's
        actual capacity (e.g., Gemini has 1M context, not 128K).
        """
        for model_id in openrouter_model_ids:
            assert (
                model_id in MODEL_CONTEXT_LIMITS
            ), f"OpenRouter model '{model_id}' missing from MODEL_CONTEXT_LIMITS"

    def test_context_limits_have_required_fields(self, openrouter_model_ids):
        """Each context limit entry must have 'total' and 'reserve_output'.

        These two fields are used to calculate max_input_tokens:
            max_input = total - reserve_output
        Missing either field causes a KeyError at runtime.
        """
        for model_id in openrouter_model_ids:
            limits = MODEL_CONTEXT_LIMITS[model_id]
            assert "total" in limits, f"'{model_id}' missing 'total' field"
            assert "reserve_output" in limits, f"'{model_id}' missing 'reserve_output' field"

    def test_gemini_flash_has_large_context(self):
        """Gemini 2.0 Flash supports 1M tokens — verify it's configured correctly.

        This is the highest context window among current models. If someone
        accidentally sets it to 128K (the default), users would get premature
        context-too-long errors.
        """
        limits = MODEL_CONTEXT_LIMITS.get("cloud-or-gemini-flash")
        assert limits is not None, "Gemini Flash missing from context limits"
        assert (
            limits["total"] == 1_000_000
        ), f"Gemini Flash should have 1M context, got {limits['total']}"

    def test_default_cloud_context_limit_structure(self):
        """DEFAULT_CLOUD_CONTEXT_LIMIT is the fallback for unknown models.

        When a user picks a model from the OpenRouter catalog that isn't in
        the static config (a 'dynamic' model), this provides safe defaults.
        128K total is the most common context window for modern models.
        """
        assert "total" in DEFAULT_CLOUD_CONTEXT_LIMIT
        assert "reserve_output" in DEFAULT_CLOUD_CONTEXT_LIMIT
        assert DEFAULT_CLOUD_CONTEXT_LIMIT["total"] == 128000
        assert DEFAULT_CLOUD_CONTEXT_LIMIT["reserve_output"] == 4000


# =============================================================================
# 2. API KEY THREADING TESTS
# =============================================================================
# These verify that the Pydantic request model accepts user-provided API keys.
# Keys travel from the browser → request body → litellm call, so the
# ChatCompletionRequest model must have the right fields to carry them.


class TestChatCompletionRequestApiKeys:
    """Verify ChatCompletionRequest accepts user-provided cloud API keys."""

    def test_openrouter_api_key_field_accepted(self, sample_messages):
        """The request model must accept an openrouter_api_key field.

        Users enter their OpenRouter key in the settings panel. The frontend
        stores it in localStorage and sends it with every chat request.
        """
        req = ChatCompletionRequest(
            messages=sample_messages,
            openrouter_api_key="sk-or-v1-test123",
        )
        assert req.openrouter_api_key == "sk-or-v1-test123"

    def test_anthropic_api_key_field_accepted(self, sample_messages):
        """The request model must accept an anthropic_api_key field."""
        req = ChatCompletionRequest(
            messages=sample_messages,
            anthropic_api_key="sk-ant-test456",
        )
        assert req.anthropic_api_key == "sk-ant-test456"

    def test_openai_api_key_field_accepted(self, sample_messages):
        """The request model must accept an openai_api_key field."""
        req = ChatCompletionRequest(
            messages=sample_messages,
            openai_api_key="sk-test789",
        )
        assert req.openai_api_key == "sk-test789"

    def test_all_keys_optional(self, sample_messages):
        """All API key fields should default to None when not provided.

        Users may only have one provider key, or none at all (using
        server-side env vars instead). Missing keys shouldn't cause errors.
        """
        req = ChatCompletionRequest(messages=sample_messages)
        assert req.openrouter_api_key is None
        assert req.anthropic_api_key is None
        assert req.openai_api_key is None

    def test_multiple_keys_simultaneously(self, sample_messages):
        """Users can provide keys for multiple providers at once.

        The frontend sends all stored keys with each request. The backend
        picks the right one based on the selected model's env_key.
        """
        req = ChatCompletionRequest(
            messages=sample_messages,
            openrouter_api_key="sk-or-v1-abc",
            anthropic_api_key="sk-ant-def",
            openai_api_key="sk-ghi",
        )
        assert req.openrouter_api_key == "sk-or-v1-abc"
        assert req.anthropic_api_key == "sk-ant-def"
        assert req.openai_api_key == "sk-ghi"


# =============================================================================
# 3. DYNAMIC MODEL RESOLUTION TESTS
# =============================================================================
# When a user picks a model from the OpenRouter catalog browser (e.g.,
# "anthropic/claude-sonnet-4"), it arrives as "cloud-or-dynamic-anthropic/claude-sonnet-4".
# _resolve_model() must strip the prefix and prepend "openrouter/" so
# LiteLLM knows to route through OpenRouter's API.


class TestDynamicModelResolution:
    """Verify _resolve_model handles dynamic OpenRouter model IDs."""

    def test_dynamic_model_prefix_stripped(self):
        """'cloud-or-dynamic-' prefix is removed and 'openrouter/' is prepended.

        Input:  'cloud-or-dynamic-anthropic/claude-sonnet-4'
        Output: {'model': 'openrouter/anthropic/claude-sonnet-4'}

        The 'openrouter/' prefix tells LiteLLM to route through OpenRouter's
        API at https://openrouter.ai/api/v1 instead of calling the provider
        directly.
        """
        from stream.middleware.core.litellm_direct import _resolve_model

        result = _resolve_model("cloud-or-dynamic-anthropic/claude-sonnet-4")
        assert result == {"model": "openrouter/anthropic/claude-sonnet-4"}

    def test_dynamic_model_with_nested_path(self):
        """Model IDs with slashes (provider/org/model) are preserved correctly.

        Some OpenRouter model IDs have three parts:
            meta-llama/llama-3.1-70b-instruct
        The prefix stripping must not eat into the actual model ID.
        """
        from stream.middleware.core.litellm_direct import _resolve_model

        result = _resolve_model("cloud-or-dynamic-meta-llama/llama-3.1-70b-instruct")
        assert result == {"model": "openrouter/meta-llama/llama-3.1-70b-instruct"}

    def test_dynamic_model_free_tier_suffix(self):
        """Free-tier models with ':free' suffix are handled correctly.

        OpenRouter appends ':free' to model IDs for free-tier access.
        The full ID must pass through to LiteLLM unchanged.
        """
        from stream.middleware.core.litellm_direct import _resolve_model

        result = _resolve_model("cloud-or-dynamic-meta-llama/llama-3.1-70b-instruct:free")
        assert result == {"model": "openrouter/meta-llama/llama-3.1-70b-instruct:free"}

    def test_unknown_model_raises_valueerror(self):
        """A model that isn't in the config and doesn't have the dynamic prefix should fail.

        This prevents typos from silently producing bad API calls. The error
        message lists available models to help the user fix the issue.
        """
        from stream.middleware.core.litellm_direct import _resolve_model

        with pytest.raises(ValueError, match="Unknown model"):
            _resolve_model("nonexistent-model-xyz")

    def test_dynamic_prefix_only_returns_empty_model_id(self):
        """Edge case: just the prefix with no model ID after it.

        This would produce 'openrouter/' which is technically invalid,
        but we don't validate at this layer — LiteLLM would reject it.
        This test documents the current behavior.
        """
        from stream.middleware.core.litellm_direct import _resolve_model

        result = _resolve_model("cloud-or-dynamic-")
        assert result == {"model": "openrouter/"}


# =============================================================================
# 4. KEY VALIDATION ENDPOINT TESTS
# =============================================================================
# The POST /v1/health/validate-key endpoint lets users verify their API key
# before attempting a chat. This avoids the frustrating experience of typing
# a long question, waiting, then getting an auth error.


class TestValidateKeyEndpoint:
    """Test the POST /v1/health/validate-key endpoint."""

    @patch("stream.middleware.routes.models.litellm.completion")
    def test_valid_openrouter_key(self, mock_completion, test_client):
        """A valid OpenRouter key should return {"valid": true}.

        The endpoint makes a 1-token test call to the cheapest model.
        If litellm.completion() succeeds, the key is valid.
        """
        mock_completion.return_value = MagicMock()

        response = test_client.post(
            "/v1/health/validate-key",
            json={"provider": "openrouter", "api_key": "sk-or-v1-valid"},
        )
        assert response.status_code == 200
        assert response.json()["valid"] is True

    @patch("stream.middleware.routes.models.litellm.completion")
    def test_valid_anthropic_key(self, mock_completion, test_client):
        """A valid Anthropic key should return {"valid": true}."""
        mock_completion.return_value = MagicMock()

        response = test_client.post(
            "/v1/health/validate-key",
            json={"provider": "anthropic", "api_key": "sk-ant-valid"},
        )
        assert response.status_code == 200
        assert response.json()["valid"] is True

    @patch("stream.middleware.routes.models.litellm.completion")
    def test_valid_openai_key(self, mock_completion, test_client):
        """A valid OpenAI key should return {"valid": true}."""
        mock_completion.return_value = MagicMock()

        response = test_client.post(
            "/v1/health/validate-key",
            json={"provider": "openai", "api_key": "sk-valid"},
        )
        assert response.status_code == 200
        assert response.json()["valid"] is True

    @patch("stream.middleware.routes.models.litellm.completion")
    def test_invalid_key_returns_false(self, mock_completion, test_client):
        """An invalid key should return {"valid": false} with an error message.

        When litellm gets a 401 from the provider, it raises AuthenticationError.
        The endpoint catches this and returns a user-friendly error.
        """
        import litellm

        mock_completion.side_effect = litellm.AuthenticationError(
            message="Invalid API key",
            llm_provider="openrouter",
            model="openrouter/openai/gpt-4o-mini",
        )

        response = test_client.post(
            "/v1/health/validate-key",
            json={"provider": "openrouter", "api_key": "sk-or-v1-invalid"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert "Invalid" in data["error"]

    @patch("stream.middleware.routes.models.litellm.completion")
    def test_rate_limited_key_still_valid(self, mock_completion, test_client):
        """A rate-limited key is still valid — just throttled.

        Free-tier OpenRouter keys have a 20 req/min limit. Hitting the
        limit during validation means the key works; the user just needs
        to wait before using it for chat.
        """
        import litellm

        mock_completion.side_effect = litellm.RateLimitError(
            message="Rate limited",
            llm_provider="openrouter",
            model="openrouter/openai/gpt-4o-mini",
        )

        response = test_client.post(
            "/v1/health/validate-key",
            json={"provider": "openrouter", "api_key": "sk-or-v1-free"},
        )
        assert response.status_code == 200
        assert response.json()["valid"] is True

    def test_unknown_provider_returns_400(self, test_client):
        """An unrecognized provider name should return 400 Bad Request.

        The endpoint only supports 'openrouter', 'anthropic', and 'openai'.
        Anything else is a client error (likely a frontend bug).
        """
        response = test_client.post(
            "/v1/health/validate-key",
            json={"provider": "fake-provider", "api_key": "sk-test"},
        )
        assert response.status_code == 400

    @patch("stream.middleware.routes.models.litellm.completion")
    def test_billing_issue_returns_helpful_error(self, mock_completion, test_client):
        """A key with billing issues should return a specific error message.

        Some providers return 400 with billing-related keywords when the
        account has no credits. We detect these and provide a clear message.
        """
        import litellm

        mock_completion.side_effect = litellm.BadRequestError(
            message="Account has exceeded billing quota",
            llm_provider="openai",
            model="gpt-4o-mini",
        )

        response = test_client.post(
            "/v1/health/validate-key",
            json={"provider": "openai", "api_key": "sk-no-credits"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert "billing" in data["error"].lower() or "credits" in data["error"].lower()


# =============================================================================
# 5. MODEL CATALOG ENDPOINT TESTS
# =============================================================================
# The GET /v1/models/catalog endpoint proxies OpenRouter's model list.
# We mock the external HTTP call to test the processing/categorization logic.


class TestModelCatalogEndpoint:
    """Test the GET /v1/models/catalog endpoint."""

    @patch("stream.middleware.routes.models._catalog_cache", {"data": None, "fetched_at": 0})
    @patch("stream.middleware.routes.models.httpx.AsyncClient")
    def test_fetches_and_categorizes_models(
        self, mock_client_cls, test_client, mock_openrouter_catalog_response
    ):
        """The catalog endpoint should fetch models and categorize them.

        It groups models into: all models, recommended, free, and by provider.
        The frontend uses these categories to populate different tabs in
        the model browser UI.
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_openrouter_catalog_response

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        response = test_client.get("/v1/models/catalog")
        assert response.status_code == 200

        data = response.json()
        assert "models" in data
        assert "recommended" in data
        assert "free" in data
        assert "categories" in data
        assert "total_count" in data
        assert "free_count" in data

    @patch("stream.middleware.routes.models._catalog_cache", {"data": None, "fetched_at": 0})
    @patch("stream.middleware.routes.models.httpx.AsyncClient")
    def test_identifies_free_models(
        self, mock_client_cls, test_client, mock_openrouter_catalog_response
    ):
        """Models with $0 pricing or ':free' suffix should appear in the free list.

        Free models are a key selling point for students — they can use
        cloud-quality models without any payment.
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_openrouter_catalog_response

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        response = test_client.get("/v1/models/catalog")
        data = response.json()

        assert data["free_count"] > 0
        free_ids = {m["id"] for m in data["free"]}
        assert "meta-llama/llama-3.1-70b-instruct:free" in free_ids

    @patch("stream.middleware.routes.models._catalog_cache", {"data": None, "fetched_at": 0})
    @patch("stream.middleware.routes.models.httpx.AsyncClient")
    def test_detects_vision_capable_models(
        self, mock_client_cls, test_client, mock_openrouter_catalog_response
    ):
        """Models with 'image' in their input modality should be flagged.

        The frontend shows a camera icon next to vision-capable models
        so users know which models can handle image queries.
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_openrouter_catalog_response

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        response = test_client.get("/v1/models/catalog")
        data = response.json()

        vision_models = [m for m in data["models"] if m["supports_vision"]]
        assert len(vision_models) >= 2, "Expected at least 2 vision-capable models"

        text_only = [m for m in data["models"] if not m["supports_vision"]]
        assert len(text_only) >= 1, "Expected at least 1 text-only model"

    @patch("stream.middleware.routes.models._catalog_cache")
    def test_returns_cached_data_when_fresh(self, mock_cache, test_client):
        """The catalog endpoint returns cached data if TTL hasn't expired.

        Caching avoids hammering OpenRouter's API on every settings panel
        open. The 1-hour TTL balances freshness with API courtesy.
        """
        cached_result = {
            "models": [{"id": "cached-model", "name": "Cached"}],
            "recommended": [],
            "free": [],
            "categories": {},
            "total_count": 1,
            "free_count": 0,
        }
        mock_cache.__getitem__ = lambda self, key: {
            "data": cached_result,
            "fetched_at": time.time(),
        }[key]

        response = test_client.get("/v1/models/catalog")
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 1

    @patch("stream.middleware.routes.models._catalog_cache", {"data": None, "fetched_at": 0})
    @patch("stream.middleware.routes.models.httpx.AsyncClient")
    def test_openrouter_api_error_with_no_cache_returns_502(self, mock_client_cls, test_client):
        """When OpenRouter fails and there's no cache, return a 502 error.

        This happens on the very first request if OpenRouter is down.
        Once cached data exists, subsequent failures fall back to stale cache.
        """
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("Connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        response = test_client.get("/v1/models/catalog")
        assert response.status_code == 502


# =============================================================================
# 6. CONTEXT WINDOW FALLBACK TESTS
# =============================================================================
# When a user picks a model from the OpenRouter catalog that isn't in the
# static MODEL_CONTEXT_LIMITS config, the system should use safe defaults
# instead of crashing. This is the "dynamic model" path.


class TestContextWindowFallback:
    """Verify context window handling for unknown/dynamic models."""

    def test_known_openrouter_model_uses_configured_limit(self):
        """Statically-configured OpenRouter models use their explicit limits.

        These are curated models with known context windows (e.g., Claude's
        200K, Gemini's 1M). Using the actual limit avoids unnecessarily
        rejecting long conversations.
        """
        max_input = get_max_input_tokens("cloud-or-claude")
        expected = (
            MODEL_CONTEXT_LIMITS["cloud-or-claude"]["total"]
            - MODEL_CONTEXT_LIMITS["cloud-or-claude"]["reserve_output"]
        )
        assert max_input == expected

    def test_unknown_dynamic_model_gets_default_limit(self):
        """Dynamically-selected models fall back to DEFAULT_CLOUD_CONTEXT_LIMIT.

        When a user picks 'cloud-or-dynamic-some-new-model' from the catalog,
        we don't have its context limit in config. The fallback (128K) is safe
        for most modern models and prevents context validation from breaking.
        """
        max_input = get_max_input_tokens("cloud-or-dynamic-some-unknown-model")
        expected = (
            DEFAULT_CLOUD_CONTEXT_LIMIT["total"] - DEFAULT_CLOUD_CONTEXT_LIMIT["reserve_output"]
        )
        assert max_input == expected

    def test_default_limit_is_128k_minus_4k(self):
        """The default context limit should be 128K total - 4K reserved = 124K input.

        128K is the most common context window size. 4K reserved output
        gives enough room for a detailed response (~3 pages).
        """
        max_input = get_max_input_tokens("cloud-or-dynamic-totally-new-model")
        assert max_input == 124000

    def test_gemini_flash_has_larger_limit_than_default(self):
        """Gemini Flash (1M context) should allow more input than the default.

        This validates that statically-configured models get their real
        limits, not the conservative default.
        """
        gemini_input = get_max_input_tokens("cloud-or-gemini-flash")
        default_input = get_max_input_tokens("cloud-or-dynamic-any-model")
        assert (
            gemini_input > default_input
        ), f"Gemini ({gemini_input}) should exceed default ({default_input})"
