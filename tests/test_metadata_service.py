"""Unit tests for :class:`ModelMetadataService`."""

from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List
from unittest.mock import MagicMock

import pytest
from hypothesis import given, strategies as st

# Ensure environment is ready before importing the module under test.
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

from agent_lab import ModelMetadataService, OpenRouterClient, OpenRouterConfig, OpenRouterError
from tests.fixtures.provider_models import (
    ANTHROPIC_MODEL,
    GOOGLE_MODEL,
    MARKDOWN_MODEL,
    OPENAI_MODEL,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
def config(monkeypatch: pytest.MonkeyPatch) -> OpenRouterConfig:
    """Provide a deterministic OpenRouter configuration for tests."""

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return OpenRouterConfig()


@pytest.fixture()
def client(config: OpenRouterConfig) -> MagicMock:
    """Create a MagicMock of :class:`OpenRouterClient`."""

    mocked = MagicMock(spec=OpenRouterClient)
    mocked.config = config
    return mocked


@pytest.fixture()
def metadata_service(client: MagicMock) -> ModelMetadataService:
    """Instantiate the metadata service using the mocked client."""

    return ModelMetadataService(client)


def _set_models(client: MagicMock, models: Iterable[Dict[str, Any]]) -> None:
    """Helper to set the return value for ``fetch_models``."""

    client.fetch_models.return_value = list(models)


# =============================================================================
# Cycle 6 – Model Choice Generation
# =============================================================================


def test_get_model_choices_filters_invalid_entries(
    metadata_service: ModelMetadataService,
    client: MagicMock,
) -> None:
    """Malformed entries should be ignored without raising errors."""

    _set_models(
        client,
        [
            None,
            "bad",  # type: ignore[list-item]
            {"id": "openai/gpt-4o-mini"},
            {"name": "openai/gpt-4o-mini"},
        ],
    )

    choices = metadata_service.get_model_choices()

    assert choices == [("openai/gpt-4o-mini", "openai/gpt-4o-mini")]


def test_get_model_choices_returns_default_when_empty(
    metadata_service: ModelMetadataService,
    client: MagicMock,
) -> None:
    """If no valid models are available, fall back to the configured default."""

    _set_models(client, [None, {}, {"id": ""}])

    choices = metadata_service.get_model_choices()

    default = metadata_service.client.config.default_model
    assert choices == [(default, default)]


@given(
    preferred=st.lists(
        st.text(min_size=1, max_size=12),
        min_size=1,
        max_size=5,
        unique=True,
    ),
    extras=st.lists(
        st.text(min_size=1, max_size=12),
        max_size=5,
        unique=True,
    ),
)
def test_sort_model_choices_respects_preferences(
    preferred: List[str],
    extras: List[str],
) -> None:
    """Preferred models must always appear first in their declared order."""

    config = OpenRouterConfig(api_key="test-key", preferred_models=tuple(preferred))
    client = MagicMock(spec=OpenRouterClient)
    client.config = config
    service = ModelMetadataService(client)

    ordered: List[str] = list(dict.fromkeys(preferred + extras))
    choices = [(identifier, identifier) for identifier in ordered]

    sorted_choices = service._sort_model_choices(choices)
    sorted_values = [value for _, value in sorted_choices]
    expected_prefix = [identifier for identifier in preferred if identifier in ordered]

    assert sorted_values[: len(expected_prefix)] == expected_prefix

    remainder = sorted_values[len(expected_prefix) :]
    expected_remainder = sorted(
        remainder,
        key=service._normalize_identifier,
    )
    assert remainder == expected_remainder


# =============================================================================
# Cycle 7 – Model Lookup & Schema Handling
# =============================================================================


def test_find_model_handles_identifier_variants(
    metadata_service: ModelMetadataService,
    client: MagicMock,
) -> None:
    """Model lookup should match on id fragments, case, and whitespace."""

    _set_models(client, [OPENAI_MODEL, GOOGLE_MODEL])

    model = metadata_service.find_model("openai/gpt-4o-mini")
    assert model is not None
    assert model["id"] == OPENAI_MODEL["id"]
    assert model["name"] == OPENAI_MODEL["name"]
    assert model["pricing"]["prompt"] == pytest.approx(0.0003)
    assert model["pricing"]["completion"] == pytest.approx(0.0006)

    assert metadata_service.find_model("GPT-4o Mini") == model
    assert metadata_service.find_model(" gpt-4o-mini ") == model
    assert metadata_service.find_model("GPT 4o Mini") == model


def test_find_model_returns_none_for_unknown(
    metadata_service: ModelMetadataService,
    client: MagicMock,
) -> None:
    """Unknown identifiers should return ``None`` without raising errors."""

    _set_models(client, [GOOGLE_MODEL])

    assert metadata_service.find_model("missing/model") is None


def test_get_valid_models_applies_schema_validation(
    metadata_service: ModelMetadataService,
    client: MagicMock,
) -> None:
    """Schema validation should filter malformed payloads and enrich metadata."""

    malformed: List[Any] = [None, "oops", {"id": ""}, {"name": ""}]
    _set_models(client, malformed + [ANTHROPIC_MODEL])

    models = metadata_service._get_valid_models()

    assert len(models) == 1
    normalized = models[0]
    pricing = normalized.get("pricing", {})

    assert pricing["prompt"] == pytest.approx(0.003)
    assert pricing["completion"] == pytest.approx(0.015)
    assert "prompt_caching" in normalized["supported_parameters"]


# =============================================================================
# Cycle 8 – Pricing & Capability Extraction
# =============================================================================


def test_format_pricing_supports_provider_aliases(metadata_service: ModelMetadataService) -> None:
    """Pricing should resolve provider-specific aliases and additional fields."""

    formatted = metadata_service._format_pricing(ANTHROPIC_MODEL["pricing"], "anthropic")

    assert "Input: $0.003/1M tok" in formatted
    assert "Output: $0.015/1M tok" in formatted
    assert "Cache: $0.001" in formatted

    tokens_pricing = {"input_tokens": "0.12", "output_tokens": "0.34"}
    formatted_alias = metadata_service._format_pricing(tokens_pricing, provider_key=None)

    assert "Input: $0.12/1M tok" in formatted_alias
    assert "Output: $0.34/1M tok" in formatted_alias


def test_extract_capabilities_merges_modality_and_flags(metadata_service: ModelMetadataService) -> None:
    """Capabilities should combine boolean flags and supported parameters."""

    model = metadata_service._normalize_model_schema(MARKDOWN_MODEL)
    assert model is not None

    capabilities = metadata_service._extract_capabilities(model)

    assert "Image-in" in capabilities
    assert "Func-calls" in capabilities
    assert "JSON-mode" in capabilities
    assert "Prompt-cache" in capabilities


@given(
    prompt=st.decimals(min_value="0", max_value="1000", places=4, allow_nan=False, allow_infinity=False),
    completion=st.decimals(min_value="0", max_value="1000", places=4, allow_nan=False, allow_infinity=False),
)
def test_format_pricing_preserves_numeric_precision(
    prompt: Decimal,
    completion: Decimal,
) -> None:
    """Formatted pricing strings must include the original numeric values."""

    pricing: Dict[str, Any] = {
        "prompt": str(prompt),
        "completion": str(completion),
    }

    config = OpenRouterConfig(api_key="test-key")
    client = MagicMock(spec=OpenRouterClient)
    client.config = config
    service = ModelMetadataService(client)

    formatted = service._format_pricing(pricing)

    def _extract(label: str) -> Decimal:
        token = next(
            (
                segment.split("$", 1)[1].split("/", 1)[0]
                for segment in formatted.split(", ")
                if segment.startswith(f"{label}: $")
            ),
            None,
        )
        assert token is not None, f"Missing {label} pricing segment"
        return Decimal(token.replace("E", "e"))

    assert _extract("Input") == Decimal(str(prompt)).normalize()
    assert _extract("Output") == Decimal(str(completion)).normalize()


# =============================================================================
# Cycle 9 – Markdown Generation
# =============================================================================


def test_build_model_markdown_matches_approval_fixture(
    metadata_service: ModelMetadataService,
    client: MagicMock,
) -> None:
    """Markdown output should remain stable to avoid UI regressions."""

    _set_models(client, [MARKDOWN_MODEL])

    markdown = metadata_service.build_model_markdown("openrouter/auto")
    expected = Path("tests/fixtures/expected_markdown.md").read_text(encoding="utf-8").strip()

    assert markdown.strip() == expected


def test_build_model_markdown_handles_missing_model(metadata_service: ModelMetadataService) -> None:
    """Unknown models should generate a clear placeholder message."""

    assert "No metadata found" in metadata_service.build_model_markdown("missing/model")


# =============================================================================
# Defensive Programming
# =============================================================================


def test_get_model_choices_handles_api_failure(
    metadata_service: ModelMetadataService,
    client: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failures in ``fetch_models`` should fall back gracefully."""

    client.fetch_models.side_effect = OpenRouterError("boom")
    warning_calls: List[str] = []
    monkeypatch.setattr("gradio.Warning", lambda message: warning_calls.append(str(message)))

    choices = metadata_service.get_model_choices()

    default = metadata_service.client.config.default_model
    assert choices == [(default, default)]
    assert warning_calls, "Expected a warning when API retrieval fails."

