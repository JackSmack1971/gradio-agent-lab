"""Tests for the :mod:`agent_lab` OpenRouter client implementation."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable
from unittest.mock import MagicMock

import pytest
from hypothesis import given, strategies as st
from requests import Response
from requests.exceptions import (
    ConnectionError as RequestsConnectionError,
    HTTPError,
    RequestException,
    Timeout,
)
from urllib3.util.retry import Retry

from agent_lab import (
    APIConnectionError,
    AuthenticationError,
    ModelNotFoundError,
    OpenRouterClient,
    OpenRouterConfig,
    OpenRouterError,
    RETRY_ALLOWED_METHODS,
    RETRY_STATUS_FORCELIST,
    SSE_DATA_PREFIX,
    SSE_DONE_SENTINEL,
    STREAM_TIMEOUT_SECONDS,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
def config(monkeypatch: pytest.MonkeyPatch) -> OpenRouterConfig:
    """Provide a configuration object with a deterministic API key."""

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return OpenRouterConfig()


@pytest.fixture()
def client(config: OpenRouterConfig) -> OpenRouterClient:
    """Create a client instance for testing."""

    return OpenRouterClient(config)


# =============================================================================
# _create_session tests
# =============================================================================


def test_create_session_configures_retry_strategy(client: OpenRouterClient) -> None:
    """Validate retry configuration and default headers for the session."""

    adapter = client.session.get_adapter("https://")
    retry: Retry = adapter.max_retries

    assert retry.total == client.config.max_retries
    assert retry.backoff_factor == client.config.backoff_factor
    assert tuple(sorted(retry.status_forcelist)) == tuple(sorted(RETRY_STATUS_FORCELIST))
    assert retry.allowed_methods == frozenset(RETRY_ALLOWED_METHODS)

    # Security guardrail: ensure API key propagates to headers
    assert client.session.headers["Authorization"] == "Bearer test-key"


@given(
    attempts=st.integers(min_value=1, max_value=6),
    factor=st.floats(min_value=0.0, max_value=3.0, allow_nan=False, allow_infinity=False),
)
def test_retry_backoff_progression(attempts: int, factor: float) -> None:
    """Property-based test ensuring exponential backoff matches urllib3 rules."""

    retry = Retry(total=attempts + 1, backoff_factor=factor)

    for _ in range(attempts):
        retry = retry.increment(
            method="GET",
            url="https://example.test",
            error=RequestException("boom"),
        )

    consecutive_errors = len(retry.history)
    backoff_time = retry.get_backoff_time()

    if consecutive_errors <= 1:
        assert backoff_time == 0
    else:
        expected = min(
            Retry.DEFAULT_BACKOFF_MAX,
            factor * (2 ** (consecutive_errors - 1)),
        )
        assert backoff_time == pytest.approx(expected)


# =============================================================================
# _handle_api_errors tests
# =============================================================================


def _make_response(status_code: int | None) -> Response:
    response = Response()
    if status_code is not None:
        response.status_code = status_code
        response._content = b"{}"  # type: ignore[attr-defined]
    return response


def test_handle_api_errors_authentication(client: OpenRouterClient) -> None:
    """Ensure 401 and 403 responses raise :class:`AuthenticationError`."""

    response = _make_response(401)

    with pytest.raises(AuthenticationError) as excinfo:
        with client._handle_api_errors():
            raise HTTPError(response=response)

    assert "authentication failed" in str(excinfo.value).lower()


def test_handle_api_errors_model_not_found(client: OpenRouterClient) -> None:
    """Ensure a 404 response raises :class:`ModelNotFoundError`."""

    response = _make_response(404)

    with pytest.raises(ModelNotFoundError) as excinfo:
        with client._handle_api_errors():
            raise HTTPError(response=response)

    assert "requested model not found" in str(excinfo.value).lower()


def test_handle_api_errors_generic_http(client: OpenRouterClient) -> None:
    """Non-specific HTTP errors should surface with status context."""

    response = _make_response(500)

    with pytest.raises(OpenRouterError) as excinfo:
        with client._handle_api_errors():
            raise HTTPError(response=response)

    assert "http 500" in str(excinfo.value).lower()


def test_handle_api_errors_without_response(client: OpenRouterClient) -> None:
    """HTTP errors without a response should still raise :class:`OpenRouterError`."""

    with pytest.raises(OpenRouterError) as excinfo:
        with client._handle_api_errors():
            raise HTTPError("network exploded")

    assert "http error" in str(excinfo.value).lower()


def test_handle_api_errors_connection_issue(client: OpenRouterClient) -> None:
    """Connection errors should be mapped to :class:`APIConnectionError`."""

    with pytest.raises(APIConnectionError) as excinfo:
        with client._handle_api_errors():
            raise RequestsConnectionError("connection failed")

    assert "unable to connect" in str(excinfo.value).lower()


def test_handle_api_errors_timeout(client: OpenRouterClient) -> None:
    """Timeout exceptions are treated as connectivity problems."""

    with pytest.raises(APIConnectionError):
        with client._handle_api_errors():
            raise Timeout("took too long")


def test_handle_api_errors_generic_request_exception(client: OpenRouterClient) -> None:
    """Other request failures surface as :class:`OpenRouterError`."""

    with pytest.raises(OpenRouterError) as excinfo:
        with client._handle_api_errors():
            raise RequestException("unexpected")

    assert "request failed" in str(excinfo.value).lower()


# =============================================================================
# fetch_models tests
# =============================================================================


def test_fetch_models_success(client: OpenRouterClient) -> None:
    """Successful calls should return the parsed model list."""

    payload = {"data": [{"id": "model-a"}, {"name": "model-b"}]}
    response = MagicMock(spec=Response)
    response.json.return_value = payload
    response.raise_for_status.return_value = None

    session = MagicMock()
    session.get.return_value = response
    client.session = session

    models = client.fetch_models()

    assert models == payload["data"]
    session.get.assert_called_once()


def test_fetch_models_invalid_json(client: OpenRouterClient) -> None:
    """Invalid JSON payload should raise :class:`OpenRouterError`."""

    response = MagicMock(spec=Response)
    response.raise_for_status.return_value = None
    response.json.side_effect = ValueError("invalid JSON")

    client.session = MagicMock()
    client.session.get.return_value = response

    with pytest.raises(OpenRouterError) as excinfo:
        client.fetch_models()

    assert "invalid json" in str(excinfo.value).lower()


@given(
    data_field=st.one_of(
        st.none(),
        st.integers(),
        st.text(),
        st.dictionaries(keys=st.text(), values=st.integers()),
        st.lists(st.text(), min_size=1),
    )
)
def test_parse_models_response_rejects_invalid_payload(data_field: Any) -> None:
    """Property-based test ensuring malformed payloads raise errors."""

    client = OpenRouterClient(OpenRouterConfig(api_key="prop-key"))
    response = MagicMock(spec=Response)
    response.json.return_value = {"data": data_field}

    with pytest.raises(OpenRouterError):
        client._parse_models_response(response)


def test_parse_models_response_rejects_invalid_entries(client: OpenRouterClient) -> None:
    """Non-dict entries in the data list should raise :class:`OpenRouterError`."""

    response = MagicMock(spec=Response)
    response.json.return_value = {"data": [{"id": "ok"}, "bad"]}

    with pytest.raises(OpenRouterError):
        client._parse_models_response(response)


def test_fetch_models_http_error(client: OpenRouterClient) -> None:
    """HTTP failures are surfaced through the API error context manager."""

    error_response = _make_response(500)
    response = MagicMock(spec=Response)
    response.raise_for_status.side_effect = HTTPError(response=error_response)
    client.session = MagicMock()
    client.session.get.return_value = response

    with pytest.raises(OpenRouterError):
        client.fetch_models()


# =============================================================================
# stream_chat_completion tests
# =============================================================================


class _FakeStreamResponse:
    """Minimal object replicating the parts of :class:`Response` we exercise."""

    def __init__(self, lines: Iterable[str]) -> None:
        self.lines = list(lines)
        self.status_code = 200

    def iter_lines(self, decode_unicode: bool = True) -> Iterable[str]:
        return self.lines

    def raise_for_status(self) -> None:  # pragma: no cover - behaviour is simple
        return None


def _build_stream_line(payload: Dict[str, Any]) -> str:
    return f"{SSE_DATA_PREFIX}{json.dumps(payload)}"


def test_iter_stream_events_parses_valid_chunks(client: OpenRouterClient) -> None:
    """Validate SSE parsing yields delta content and respects DONE sentinel."""

    lines = [
        "garbage",
        f"{SSE_DATA_PREFIX}",
        _build_stream_line({"choices": [{"delta": {"content": "Hello"}}]}),
        _build_stream_line({"choices": [{"delta": {"content": " World"}}]}),
        f"{SSE_DATA_PREFIX}{SSE_DONE_SENTINEL}",
        _build_stream_line({"choices": [{"delta": {"content": "ignored"}}]}),
    ]

    response = _FakeStreamResponse(lines)

    assert list(client._iter_stream_events(response)) == ["Hello", " World"]


def test_iter_stream_events_skips_malformed_json(client: OpenRouterClient) -> None:
    """Malformed JSON lines should be ignored gracefully."""

    lines = [
        f"{SSE_DATA_PREFIX}{{not json}}",
        _build_stream_line({"choices": [{}]}),
        _build_stream_line({"choices": [{"delta": {"content": 123}}]}),
        _build_stream_line({"choices": [{"delta": {"content": "ok"}}]}),
        f"{SSE_DATA_PREFIX}{SSE_DONE_SENTINEL}",
    ]

    response = _FakeStreamResponse(lines)

    assert list(client._iter_stream_events(response)) == ["ok"]


def test_stream_chat_completion_invokes_session_post(client: OpenRouterClient) -> None:
    """Ensure streaming API call uses the configured timeout and yields tokens."""

    stream_lines = [
        _build_stream_line({"choices": [{"delta": {"content": "Hi"}}]}),
        f"{SSE_DATA_PREFIX}{SSE_DONE_SENTINEL}",
    ]

    response = _FakeStreamResponse(stream_lines)
    cm = MagicMock()
    cm.__enter__.return_value = response
    cm.__exit__.return_value = False

    session = MagicMock()
    session.post.return_value = cm
    client.session = session

    tokens = list(
        client.stream_chat_completion(
            model="model-x",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.5,
            max_tokens=128,
            top_p=0.9,
            seed=42,
        )
    )

    assert tokens == ["Hi"]

    session.post.assert_called_once()
    _, kwargs = session.post.call_args
    assert kwargs["timeout"] == STREAM_TIMEOUT_SECONDS
    assert kwargs["json"]["temperature"] == 0.5
    assert kwargs["json"]["max_tokens"] == 128
    assert kwargs["json"]["top_p"] == 0.9
    assert kwargs["json"]["seed"] == 42


def test_stream_chat_completion_handles_errors(client: OpenRouterClient) -> None:
    """Streaming failures bubble up through the API error context manager."""

    response = MagicMock()
    response.raise_for_status.side_effect = HTTPError(response=_make_response(500))
    cm = MagicMock()
    cm.__enter__.return_value = response
    cm.__exit__.return_value = False

    session = MagicMock()
    session.post.return_value = cm
    client.session = session

    with pytest.raises(OpenRouterError):
        list(
            client.stream_chat_completion(
                model="model-x",
                messages=[{"role": "user", "content": "Hello"}],
            )
        )

