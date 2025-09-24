"""Integration-oriented tests for :class:`GradioAgentLab`."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List
from unittest.mock import MagicMock

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

# Ensure environment readiness prior to importing agent_lab.
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

from agent_lab import (  # noqa: E402  pylint: disable=wrong-import-position
    APIConnectionError,
    AuthenticationError,
    ComponentUpdate,
    GradioAgentLab,
    ModelMetadataService,
    ModelNotFoundError,
    OpenRouterClient,
    OpenRouterConfig,
    OpenRouterError,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
def config(monkeypatch: pytest.MonkeyPatch) -> OpenRouterConfig:
    """Provide a deterministic OpenRouter configuration for the lab."""

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return OpenRouterConfig()


@pytest.fixture()
def client(config: OpenRouterConfig) -> MagicMock:
    """Create a client mock with cache management hooks."""

    mocked = MagicMock(spec=OpenRouterClient)
    mocked.config = config
    mocked.fetch_models.cache_clear = MagicMock()  # type: ignore[attr-defined]
    return mocked


@pytest.fixture()
def metadata_service() -> MagicMock:
    """Provide a metadata service mock."""

    service = MagicMock(spec=ModelMetadataService)
    service.get_model_choices.return_value = [("openrouter/auto", "openrouter/auto")]
    service.build_model_markdown.return_value = "### Model\n`openrouter/auto`"
    return service


@pytest.fixture()
def lab(
    config: OpenRouterConfig,
    client: MagicMock,
    metadata_service: MagicMock,
) -> GradioAgentLab:
    """Instantiate the lab with mocked dependencies."""

    instance = object.__new__(GradioAgentLab)
    instance.config = config
    instance.client = client
    instance.metadata_service = metadata_service
    instance.model_choices = [(config.default_model, config.default_model)]
    return instance


# =============================================================================
# Cycle 10 – Chat Message Building
# =============================================================================


def test_build_chat_messages_includes_system_prompt_and_history_order(lab: GradioAgentLab) -> None:
    """System prompts must be prepended and content sanitized."""

    history: List[Dict[str, Any]] = [
        {"role": "user", "content": "  hello\x00\n"},
        {"role": "assistant", "content": "  hi!  "},
    ]

    messages = lab._build_chat_messages(history, "  system role  ")

    assert messages[0] == {"role": "system", "content": "system role"}
    assert messages[1] == {"role": "user", "content": "hello"}
    assert messages[2] == {"role": "assistant", "content": "hi!"}
    # Ensure original history is not mutated by sanitization.
    assert history[0]["content"] == "  hello\x00\n"


def test_build_chat_messages_filters_invalid_history_entries(lab: GradioAgentLab) -> None:
    """Messages lacking valid roles or content should be discarded."""

    history: List[Any] = [
        {"role": "User", "content": "  keep me  "},
        {"role": "assistant", "content": ""},
        {"role": "system", "content": None},
        {"role": "moderator", "content": "ignore"},
        {"content": "missing role"},
        "bad",  # type: ignore[list-item]
    ]

    messages = lab._build_chat_messages(history, "")

    assert messages == [{"role": "user", "content": "keep me"}]


@st.composite
def _raw_message(draw: st.DrawFn) -> Dict[str, Any]:
    """Generate varied raw message payloads for property tests."""

    message: Dict[str, Any] = {}
    if draw(st.booleans()):
        message["role"] = draw(st.text(max_size=8))
    if draw(st.booleans()):
        content_strategy = st.one_of(
            st.text(
                min_size=0,
                max_size=32,
                alphabet=st.characters(min_codepoint=0, max_codepoint=0x7F),
            ),
            st.none(),
            st.integers(),
        )
        message["content"] = draw(content_strategy)
    if draw(st.booleans()):
        message["extra"] = draw(st.text(max_size=5))
    return message


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    system_prompt=st.text(
        min_size=0,
        max_size=64,
        alphabet=st.characters(min_codepoint=0, max_codepoint=0x7F),
    ),
    history=st.lists(
        st.one_of(_raw_message(), st.none(), st.integers()),
        max_size=6,
    ),
)
def test_build_chat_messages_property_sanitizes_content(
    lab: GradioAgentLab,
    system_prompt: str,
    history: List[Any],
) -> None:
    """All emitted messages must have valid roles and sanitized content."""

    messages = lab._build_chat_messages(history, system_prompt)

    if lab._sanitize_chat_content(system_prompt):
        assert messages[0]["role"] == "system"

    for message in messages:
        assert set(message.keys()) == {"role", "content"}
        assert message["role"] in lab._VALID_CHAT_ROLES
        assert isinstance(message["content"], str)
        assert message["content"]
        assert all(
            ord(char) >= 32 or char in "\n\t"
            for char in message["content"]
        )


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(raw_content=st.one_of(
    st.text(
        min_size=0,
        max_size=128,
        alphabet=st.characters(min_codepoint=0, max_codepoint=0x7F),
    ),
    st.none(),
    st.integers(),
))
def test_sanitize_chat_content_strips_control_characters(
    lab: GradioAgentLab,
    raw_content: Any,
) -> None:
    """Sanitization should remove ASCII control characters while returning text."""

    sanitized = lab._sanitize_chat_content(raw_content)

    assert isinstance(sanitized, str)
    assert all(ord(char) >= 32 or char in "\n\t" for char in sanitized)


# =============================================================================
# Cycle 11 – Stream Handling & Error Propagation
# =============================================================================


def test_handle_chat_stream_yields_tokens_and_sanitizes_request(
    lab: GradioAgentLab,
) -> None:
    """Streaming responses should be proxied with sanitized request payloads."""

    lab.client.stream_chat_completion.return_value = iter(["delta"])

    outputs = list(
        lab.handle_chat_stream(
            history=[{"role": "user", "content": "  hello  "}],
            model="  provider/model  ",
            system_prompt="  act cleanly  ",
            temperature=0.5,
            max_tokens=32,
            top_p=0.9,
            seed=7,
        )
    )

    assert outputs == ["delta"]
    lab.client.stream_chat_completion.assert_called_once()
    kwargs = lab.client.stream_chat_completion.call_args.kwargs
    assert kwargs["model"] == "provider/model"
    assert kwargs["messages"][0] == {"role": "system", "content": "act cleanly"}
    assert kwargs["messages"][1] == {"role": "user", "content": "hello"}


@pytest.mark.parametrize(
    "exception, expected",
    [
        (
            AuthenticationError("denied"),
            "❌ **Authentication Error**: Invalid OpenRouter API key. Please check your configuration.",
        ),
        (
            ModelNotFoundError("missing"),
            "❌ **Model Error**: Model 'provider/model' not found on OpenRouter.",
        ),
        (
            APIConnectionError("offline"),
            "❌ **Connection Error**: Unable to reach OpenRouter API. Please check your internet connection.",
        ),
        (
            OpenRouterError("API failure!\x00"),
            "❌ **API Error**: API failure!",
        ),
    ],
)
def test_handle_chat_stream_translates_known_errors(
    lab: GradioAgentLab,
    exception: Exception,
    expected: str,
) -> None:
    """Known exceptions should be converted into friendly error messages."""

    lab.client.stream_chat_completion.side_effect = exception

    outputs = list(
        lab.handle_chat_stream(
            history=[],
            model="provider/model",
            system_prompt="",
            temperature=0.0,
            max_tokens=0,
            top_p=1.0,
            seed=0,
        )
    )

    assert outputs == [expected]


def test_handle_chat_stream_handles_unexpected_exception(lab: GradioAgentLab) -> None:
    """Unexpected exceptions should be surfaced with sanitized diagnostics."""

    lab.client.stream_chat_completion.side_effect = ValueError("bad\x00 input")

    outputs = list(
        lab.handle_chat_stream(
            history=[],
            model=" provider/model ",
            system_prompt="",
            temperature=0.0,
            max_tokens=0,
            top_p=1.0,
            seed=0,
        )
    )

    assert outputs == ["❌ **Unexpected Error**: bad input"]


# =============================================================================
# Cycle 12 – UI Update Logic
# =============================================================================


def test_update_model_sidebar_returns_payload(lab: GradioAgentLab) -> None:
    """Sidebar updates should sanitize identifiers and forward metadata."""

    lab.metadata_service.build_model_markdown.return_value = "## Details"

    model_id, markdown = lab.update_model_sidebar("  openrouter/model  ")

    assert model_id == "openrouter/model"
    assert markdown == "## Details"
    lab.metadata_service.build_model_markdown.assert_called_once_with("openrouter/model")


def test_update_model_sidebar_handles_metadata_errors(lab: GradioAgentLab) -> None:
    """Failures when building metadata should fall back to safe markdown."""

    lab.metadata_service.build_model_markdown.side_effect = RuntimeError("boom")

    model_id, markdown = lab.update_model_sidebar(" broken/model\x00 ")

    assert model_id == "broken/model"
    assert markdown == "### Model\n`broken/model`\n\n*Error loading metadata.*"


def test_refresh_models_updates_dropdown_and_sidebar(
    lab: GradioAgentLab,
    metadata_service: MagicMock,
) -> None:
    """Refreshing models should yield sanitized component updates."""

    metadata_service.get_model_choices.return_value = [("  Label  ", " provider/model ")]
    metadata_service.build_model_markdown.return_value = "## Provider"

    dropdown_update, model_id, markdown = lab.refresh_models(" provider/model ")

    assert isinstance(dropdown_update, dict)
    assert dropdown_update["__type__"] == "update"
    assert dropdown_update["choices"] == [("Label", "provider/model")]
    assert dropdown_update["value"] == "provider/model"
    assert model_id == "provider/model"
    assert markdown == "## Provider"
    lab.client.fetch_models.cache_clear.assert_called_once()
    metadata_service.build_model_markdown.assert_called_once_with("provider/model")


def test_refresh_models_handles_empty_choice_list(
    lab: GradioAgentLab,
    metadata_service: MagicMock,
) -> None:
    """An empty model list should fall back to the configured default."""

    metadata_service.get_model_choices.return_value = []
    metadata_service.build_model_markdown.return_value = "## Default"

    dropdown_update, model_id, markdown = lab.refresh_models(" unknown/model ")

    default_model = lab.config.default_model
    assert dropdown_update["choices"] == [(default_model, default_model)]
    assert dropdown_update["value"] == default_model
    assert model_id == default_model
    assert markdown == "## Default"
    metadata_service.build_model_markdown.assert_called_once_with(default_model)

