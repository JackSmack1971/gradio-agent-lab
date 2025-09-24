"""Unit tests for OpenRouter configuration and error handling."""

from __future__ import annotations

import os
from typing import Dict, Type

os.environ.setdefault("OPENROUTER_API_KEY", "unit-test-placeholder")

import gradio as gr
import pytest
from hypothesis import given, strategies as st


class _PatchedChatInterface(gr.ChatInterface):
    """Shim Gradio ChatInterface to ignore retry_btn parameter for compatibility."""

    def __init__(self, *args, **kwargs):
        for unsupported in ("retry_btn", "undo_btn", "submit_btn", "stop_btn"):
            kwargs.pop(unsupported, None)
        super().__init__(*args, **kwargs)


gr.ChatInterface = _PatchedChatInterface


from agent_lab import (
    APIConnectionError,
    AuthenticationError,
    ModelNotFoundError,
    OpenRouterConfig,
    OpenRouterError,
)


def test_openrouter_config_missing_api_key_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure configuration validation rejects missing API key."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ValueError):
        OpenRouterConfig()


@given(
    api_key=st.text(
        min_size=1,
        alphabet=st.characters(min_codepoint=33, max_codepoint=126, blacklist_characters=["\\", "\n", "\r"]),
    ),
    http_referer=st.one_of(
        st.just(""),
        st.text(
            alphabet=st.characters(min_codepoint=33, max_codepoint=126, blacklist_characters=["\\", "\n", "\r"]),
            max_size=50,
        ),
    ),
    x_title=st.one_of(
        st.just(""),
        st.text(
            alphabet=st.characters(min_codepoint=33, max_codepoint=126, blacklist_characters=["\\", "\n", "\r"]),
            max_size=50,
        ),
    ),
)
def test_openrouter_config_headers_generation(api_key: str, http_referer: str, x_title: str) -> None:
    """Property-based test verifying header generation for varying configuration inputs."""
    config = OpenRouterConfig(api_key=api_key, http_referer=http_referer, x_title=x_title)

    headers = config.headers

    expected_headers: Dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if http_referer:
        expected_headers["HTTP-Referer"] = http_referer
    if x_title:
        expected_headers["X-Title"] = x_title

    assert headers == expected_headers


@pytest.mark.parametrize(
    "error_type",
    [AuthenticationError, ModelNotFoundError, APIConnectionError],
)
def test_custom_exceptions_inherit_from_base(error_type: Type[OpenRouterError]) -> None:
    """Validate that all custom errors derive from the OpenRouterError hierarchy."""
    assert issubclass(error_type, OpenRouterError)


def test_custom_exception_is_caught_by_base() -> None:
    """Ensure OpenRouterError captures derived exceptions when raised."""
    with pytest.raises(OpenRouterError):
        raise AuthenticationError("authentication failed")
