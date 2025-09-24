"""
Enterprise-grade Gradio Agent Lab for OpenRouter API integration.

This module provides a production-ready, type-safe implementation of a chat interface
that integrates with OpenRouter's AI model API, featuring comprehensive error handling,
modern Python patterns, and enterprise architecture standards.

Note:
    Importing this module requires the ``OPENROUTER_API_KEY`` environment variable
    to be configured. The module-level singleton validates credential availability
    before exposing helpers to guard against unauthenticated API usage.

Author: Senior Python Developer
Version: 2.0.0
Python: 3.9+
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
import re
from decimal import Decimal, InvalidOperation
from textwrap import dedent
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypedDict,
    Union,
    cast,
)
from urllib.parse import urljoin

import gradio as gr
import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError, HTTPError, RequestException, Timeout
from urllib3.util.retry import Retry


# =============================================================================
# Constants
# =============================================================================

RETRY_STATUS_FORCELIST: Tuple[int, ...] = (429, 500, 502, 503, 504)
RETRY_ALLOWED_METHODS: Tuple[str, ...] = ("HEAD", "GET", "OPTIONS", "POST")
STREAM_TIMEOUT_SECONDS: int = 300
SSE_DATA_PREFIX: str = "data: "
SSE_DONE_SENTINEL: str = "[DONE]"
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# =============================================================================
# Type Definitions & Data Structures
# =============================================================================

class ModelInfo(TypedDict, total=False):
    """Type definition for OpenRouter model information."""
    id: str
    name: str
    owned_by: Optional[str]
    organization: Optional[str] 
    provider: Optional[str]
    context_length: Optional[int]
    max_context: Optional[int]
    context: Optional[int]
    pricing: Optional[Dict[str, float]]
    input_modalities: Optional[List[str]]
    output_modalities: Optional[List[str]]
    modality: Optional[List[str]]
    supported_parameters: Optional[List[str]]
    supports_images: Optional[bool]
    tools: Optional[bool]
    function_calling: Optional[bool]
    json_output: Optional[bool]
    description: Optional[str]
    permalink: Optional[str]


class ChatMessage(TypedDict):
    """Type definition for chat message structure."""
    role: str  # "user", "assistant", or "system"
    content: str


class ModelChoice(TypedDict):
    """Type definition for model dropdown choices."""
    label: str
    value: str


class ComponentUpdate(TypedDict, total=False):
    """Typed representation of a Gradio component update payload."""

    __type__: str
    choices: List[Tuple[str, str]]
    value: Optional[str]


# =============================================================================
# Configuration Management
# =============================================================================

@dataclass(frozen=True)
class OpenRouterConfig:
    """Centralized configuration for OpenRouter API integration."""
    
    api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    base_url: str = "https://openrouter.ai/api/v1"
    timeout: int = 30
    max_retries: int = 3
    backoff_factor: float = 0.3
    
    # Attribution headers (optional)
    http_referer: str = "http://localhost:7860"
    x_title: str = "Agent Lab (Gradio 5)"
    
    # Default model settings
    default_model: str = "openrouter/auto"
    preferred_models: Tuple[str, ...] = (
        "anthropic/claude-3.5-sonnet",
        "google/gemini-2.0-flash", 
        "openai/gpt-4o-mini",
        "meta-llama/llama-3.1-70b-instruct",
        "deepseek/deepseek-chat",
        "qwen/qwen2.5-72b-instruct",
        "openrouter/auto",
    )
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
    
    @property
    def headers(self) -> Dict[str, str]:
        """Generate HTTP headers for API requests."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # Add optional attribution headers
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.x_title:
            headers["X-Title"] = self.x_title
            
        return headers


# =============================================================================
# Custom Exceptions
# =============================================================================

class OpenRouterError(Exception):
    """Base exception for OpenRouter API errors."""
    pass


class AuthenticationError(OpenRouterError):
    """Raised when API authentication fails."""
    pass


class ModelNotFoundError(OpenRouterError):
    """Raised when a requested model is not available."""
    pass


class APIConnectionError(OpenRouterError):
    """Raised when unable to connect to OpenRouter API."""
    pass


# =============================================================================
# OpenRouter API Client
# =============================================================================

class OpenRouterClient:
    """Production-ready OpenRouter API client with comprehensive error handling."""
    
    def __init__(self, config: OpenRouterConfig) -> None:
        """Initialize the OpenRouter client with configuration."""
        self.config = config
        self.session = self._create_session()
        
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy and proper configuration."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.backoff_factor,
            status_forcelist=RETRY_STATUS_FORCELIST,
            allowed_methods=frozenset(RETRY_ALLOWED_METHODS),
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set default headers
        session.headers.update(self.config.headers)
        
        return session
    
    @contextmanager
    def _handle_api_errors(self) -> Iterator[None]:
        """Context manager for consistent API error handling."""
        try:
            yield
        except HTTPError as exc:
            if exc.response is not None:
                status_code = exc.response.status_code
                if status_code in (401, 403):
                    raise AuthenticationError(
                        "OpenRouter API authentication failed. Check your API key."
                    ) from exc
                elif status_code == 404:
                    raise ModelNotFoundError(
                        "Requested model not found on OpenRouter."
                    ) from exc
                else:
                    raise OpenRouterError(
                        f"OpenRouter API error (HTTP {status_code}): {exc}"
                    ) from exc
            else:
                raise OpenRouterError(f"HTTP error: {exc}") from exc
        except (ConnectionError, Timeout) as exc:
            raise APIConnectionError(
                "Unable to connect to OpenRouter API. Check your internet connection."
            ) from exc
        except RequestException as exc:
            raise OpenRouterError(f"Request failed: {exc}") from exc

    def _parse_models_response(self, response: requests.Response) -> List[ModelInfo]:
        """Validate and extract model data from the API response."""
        try:
            payload = response.json()
        except ValueError as exc:
            raise OpenRouterError(
                "Invalid JSON response from OpenRouter models endpoint"
            ) from exc

        data = payload.get("data")
        if not isinstance(data, list):
            raise OpenRouterError(
                "Invalid payload structure from OpenRouter models endpoint"
            )

        models: List[ModelInfo] = []
        for entry in data:
            if isinstance(entry, dict):
                models.append(cast(ModelInfo, entry))
            else:
                raise OpenRouterError(
                    "Invalid model entry format from OpenRouter models endpoint"
                )

        return models

    def _iter_stream_events(self, response: requests.Response) -> Iterator[str]:
        """Yield text deltas from a streaming chat completion response."""
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith(SSE_DATA_PREFIX):
                continue

            data_content = raw_line[len(SSE_DATA_PREFIX) :].strip()
            if not data_content:
                continue

            if data_content == SSE_DONE_SENTINEL:
                break

            try:
                chunk = json.loads(data_content)
            except json.JSONDecodeError:
                # Skip malformed JSON chunks
                continue

            delta = (
                chunk.get("choices", [{}])[0]
                .get("delta", {})
                .get("content")
            )

            if isinstance(delta, str) and delta:
                yield delta

    @lru_cache(maxsize=1)
    def fetch_models(self) -> List[ModelInfo]:
        """
        Fetch available models from OpenRouter API with caching.
        
        Returns:
            List of model information dictionaries.
            
        Raises:
            OpenRouterError: If the API request fails.
        """
        url = urljoin(self.config.base_url, "models")

        with self._handle_api_errors():
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            return self._parse_models_response(response)
    
    def stream_chat_completion(
        self,
        model: str,
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """
        Stream a chat completion from OpenRouter API.
        
        Args:
            model: Model identifier to use for completion.
            messages: List of chat messages.
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens to generate.
            top_p: Nucleus sampling parameter.
            seed: Random seed for reproducible outputs.
            
        Yields:
            Individual content tokens from the streaming response.
            
        Raises:
            OpenRouterError: If the streaming request fails.
        """
        url = urljoin(self.config.base_url, "chat/completions")
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        
        # Add optional parameters if provided
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if seed is not None and seed != 0:
            payload["seed"] = int(seed)
        
        with self._handle_api_errors():
            with self.session.post(
                url,
                json=payload,
                stream=True,
                timeout=STREAM_TIMEOUT_SECONDS,  # Longer timeout for streaming
            ) as response:
                response.raise_for_status()
                yield from self._iter_stream_events(response)


# =============================================================================
# Model Metadata Service
# =============================================================================

class ModelMetadataService:
    """Service for processing and formatting model metadata."""

    _DEFAULT_PRIORITY_RANK: int = 10_000
    _PRICING_KEY_ALIASES: Dict[str, Tuple[str, ...]] = {
        "prompt": ("prompt", "input", "input_token", "input_tokens"),
        "completion": (
            "completion",
            "output",
            "output_token",
            "output_tokens",
        ),
    }
    _PROVIDER_PRICING_OVERRIDES: Dict[str, Dict[str, Tuple[str, ...]]] = {
        "anthropic": {
            "prompt": ("cache", "cache_prompt"),
            "completion": ("cache_completion",),
        },
        "google": {
            "prompt": ("cached_prompt",),
            "completion": ("cached_completion",),
        },
    }
    _MARKDOWN_TEMPLATE: str = dedent(
        """\
        ### Model
        [`{name}`]({permalink})

        **Provider:** `{provider}`

        **Context length:** `{context_length}` tokens

        **Pricing:** {pricing}

        **Modalities:**
        - Input: `{input_modalities}`
        - Output: `{output_modalities}`

        **Supported params:** {supported_parameters}

        **Capabilities:** {capabilities}

        **Notes:**
        {description}

        > Data from OpenRouter **Models API**. Fields vary by provider.
        """
    )

    def __init__(self, client: OpenRouterClient) -> None:
        """Initialize with OpenRouter client."""
        self.client = client
    
    def get_model_choices(self) -> List[Tuple[str, str]]:
        """
        Get formatted model choices for dropdown UI component.

        Returns:
            List of (label, value) tuples for model selection.
        """
        try:
            models = self._get_valid_models()
        except OpenRouterError:
            gr.Warning("Unable to fetch models from OpenRouter. Using default.")
            fallback = self.client.config.default_model
            return [(fallback, fallback)]

        choices: List[Tuple[str, str]] = []
        seen: Set[str] = set()
        for model in models:
            identifier = (model.get("id") or model.get("name") or "").strip()
            if not identifier or identifier in seen:
                continue
            seen.add(identifier)
            choices.append((identifier, identifier))

        if not choices:
            fallback = self.client.config.default_model
            return [(fallback, fallback)]

        return self._sort_model_choices(choices)
    
    def find_model(self, model_id: str) -> Optional[ModelInfo]:
        """
        Find model information by ID using normalized comparison rules.

        Args:
            model_id: The model identifier to search for.

        Returns:
            Model information dictionary if found, None otherwise.
        """
        search_tokens = self._identifier_tokens(model_id)
        if not search_tokens:
            return None

        try:
            models = self._get_valid_models()
        except OpenRouterError:
            return None

        for model in models:
            candidate_tokens: Set[str] = set()
            for key in ("id", "name"):
                value = model.get(key)
                if isinstance(value, str):
                    candidate_tokens.update(self._identifier_tokens(value))

            if search_tokens & candidate_tokens:
                return model

        return None

    def _get_valid_models(self) -> List[ModelInfo]:
        """Fetch and normalize models, filtering malformed entries."""
        raw_models = self.client.fetch_models()
        normalized: List[ModelInfo] = []
        for raw_model in raw_models:
            normalized_model = self._normalize_model_schema(raw_model)
            if normalized_model:
                normalized.append(normalized_model)
        return normalized

    def _normalize_model_schema(self, raw_model: Any) -> Optional[ModelInfo]:
        """Validate and normalize the structure of a raw model entry."""
        if not isinstance(raw_model, dict):
            return None

        identifier = raw_model.get("id") or raw_model.get("name")
        if not isinstance(identifier, str) or not identifier.strip():
            return None

        normalized: ModelInfo = {}

        for key in (
            "id",
            "name",
            "owned_by",
            "organization",
            "provider",
            "permalink",
        ):
            value = raw_model.get(key)
            if isinstance(value, str) and value.strip():
                normalized[key] = value.strip()

        description = raw_model.get("description")
        if isinstance(description, str) and description.strip():
            normalized["description"] = description.strip()

        for key in ("context_length", "max_context", "context"):
            value = raw_model.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                normalized[key] = int(value)

        for key in (
            "input_modalities",
            "output_modalities",
            "modality",
            "supported_parameters",
        ):
            value = raw_model.get(key)
            if isinstance(value, (list, tuple, set)):
                cleaned = [
                    str(item).strip()
                    for item in value
                    if isinstance(item, str) and item.strip()
                ]
                if cleaned:
                    normalized[key] = cleaned

        for key in ("supports_images", "tools", "function_calling", "json_output"):
            value = raw_model.get(key)
            if isinstance(value, bool):
                normalized[key] = value

        provider_key = self._determine_provider_key(raw_model)
        pricing = self._normalize_pricing_data(
            raw_model.get("pricing"),
            provider_key,
        )
        if pricing:
            normalized["pricing"] = pricing

        normalized = self._apply_provider_adapters(normalized, provider_key)
        return normalized

    def _determine_provider_key(self, model: Dict[str, Any]) -> str:
        """Derive a normalized provider key used for adapter lookups."""
        provider_value = (
            model.get("provider")
            or model.get("owned_by")
            or model.get("organization")
            or ""
        )
        if isinstance(provider_value, str):
            return provider_value.strip().lower()
        return ""

    def _apply_provider_adapters(
        self,
        model: ModelInfo,
        provider_key: str,
    ) -> ModelInfo:
        """Apply provider-specific schema adjustments."""
        adapter = self._provider_schema_adapters().get(provider_key)
        if not adapter:
            return model
        return adapter(model)

    def _provider_schema_adapters(
        self,
    ) -> Dict[str, Callable[[ModelInfo], ModelInfo]]:
        """Expose schema adapter registry for provider normalization."""
        return {
            "anthropic": self._anthropic_adapter,
            "google": self._google_adapter,
        }

    @staticmethod
    def _anthropic_adapter(model: ModelInfo) -> ModelInfo:
        """Adapt Anthropic metadata to expose prompt caching support."""
        adapted = cast(ModelInfo, dict(model))
        pricing = adapted.get("pricing", {})
        params = list(
            dict.fromkeys(adapted.get("supported_parameters", []) or [])
        )

        if any(key.startswith("cache") for key in pricing):
            if "prompt_caching" not in params:
                params.append("prompt_caching")

        if params:
            adapted["supported_parameters"] = params

        return adapted

    @staticmethod
    def _google_adapter(model: ModelInfo) -> ModelInfo:
        """Ensure Google modality data is reflected in output modalities."""
        adapted = cast(ModelInfo, dict(model))
        modalities = adapted.get("modality", []) or []
        outputs = list(
            dict.fromkeys(adapted.get("output_modalities", []) or [])
        )

        for modality in modalities:
            if modality not in outputs:
                outputs.append(modality)

        if outputs:
            adapted["output_modalities"] = outputs

        return adapted

    def _normalize_pricing_data(
        self,
        pricing: Any,
        provider_key: Optional[str] = None,
    ) -> Dict[str, float]:
        """Normalize pricing values across provider schemas."""
        if not isinstance(pricing, dict):
            return {}

        normalized: Dict[str, float] = {}
        alias_map = dict(self._PRICING_KEY_ALIASES)

        if provider_key:
            provider_aliases = self._PROVIDER_PRICING_OVERRIDES.get(provider_key, {})
            for target, aliases in provider_aliases.items():
                existing = alias_map.get(target, ())
                alias_map[target] = existing + tuple(
                    alias for alias in aliases if alias not in existing
                )

        for target_field, aliases in alias_map.items():
            for alias in aliases:
                if alias in pricing and target_field not in normalized:
                    decimal_value = self._coerce_decimal(pricing[alias])
                    if decimal_value is not None:
                        normalized[target_field] = float(decimal_value)
                        break

        for key, value in pricing.items():
            if key in alias_map:
                continue
            decimal_value = self._coerce_decimal(value)
            if decimal_value is not None:
                normalized[key] = float(decimal_value)

        return normalized

    def _coerce_decimal(self, value: Any) -> Optional[Decimal]:
        """Convert pricing data into a :class:`Decimal` when possible."""
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, Decimal):
            return value
        if isinstance(value, (int, float)):
            return Decimal(str(value))
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return None
            match = re.search(r"(\d+(?:\.\d+)?)", cleaned.replace(",", ""))
            if not match:
                return None
            cleaned_value = match.group(1)
            try:
                return Decimal(cleaned_value)
            except InvalidOperation:
                return None
        return None

    def _format_decimal(self, value: float) -> str:
        """Format a numeric value without introducing exponent notation."""
        decimal_value = self._coerce_decimal(value)
        if decimal_value is None:
            return "0"

        normalized = decimal_value.normalize()
        if normalized == normalized.to_integral():
            return str(normalized.to_integral())

        text = format(normalized, "f").rstrip("0").rstrip(".")
        return text or "0"
    
    def _format_pricing(
        self,
        pricing: Optional[Dict[str, Any]],
        provider_key: Optional[str] = None,
    ) -> str:
        """Format pricing information for display."""
        normalized = self._normalize_pricing_data(pricing, provider_key)
        if not normalized:
            return "—"

        parts: List[str] = []
        for field in ("prompt", "completion"):
            if field in normalized:
                label = "Input" if field == "prompt" else "Output"
                formatted = self._format_decimal(normalized[field])
                parts.append(f"{label}: ${formatted}/1M tok")

        for key in sorted(normalized):
            if key in ("prompt", "completion"):
                continue
            formatted = self._format_decimal(normalized[key])
            parts.append(f"{key.capitalize()}: ${formatted}")

        return ", ".join(parts) if parts else "—"

    def _identifier_tokens(self, value: str) -> Set[str]:
        """Create a set of comparable tokens for identifier matching."""
        normalized = self._normalize_identifier(value)
        if not normalized:
            return set()

        tokens = {normalized}
        canonical = normalized.replace("_", "-")
        tokens.add(canonical)
        tokens.add(canonical.replace("-", " "))
        tokens.add(normalized.replace(" ", ""))

        if "/" in normalized:
            _, remainder = normalized.split("/", 1)
            tokens.add(remainder)
            remainder_dash = remainder.replace("_", "-")
            tokens.add(remainder_dash)
            tokens.add(remainder_dash.replace("-", " "))
            tokens.add(remainder.replace(" ", ""))

        return {token for token in tokens if token}

    @staticmethod
    def _normalize_identifier(value: str) -> str:
        """Normalize an identifier for comparison."""
        return value.strip().lower()

    def _sort_model_choices(
        self,
        choices: Sequence[Tuple[str, str]],
    ) -> List[Tuple[str, str]]:
        """Sort model choices by preferred priority and then identifier."""
        preferred = self.client.config.preferred_models
        priority_map = {
            self._normalize_identifier(model): index
            for index, model in enumerate(preferred)
        }

        sorted_choices = sorted(
            choices,
            key=lambda item: (
                priority_map.get(
                    self._normalize_identifier(item[1]),
                    self._DEFAULT_PRIORITY_RANK,
                ),
                self._normalize_identifier(item[1]),
            ),
        )
        return list(sorted_choices)

    def _render_markdown_template(self, context: Dict[str, str]) -> str:
        """Render the markdown template with the provided context."""
        return self._MARKDOWN_TEMPLATE.format(**context)

    @staticmethod
    def _escape_markdown(value: str) -> str:
        """Escape markdown-sensitive characters."""
        escaped = value.replace("\\", "\\\\")
        for char in ("`", "[", "]"):
            escaped = escaped.replace(char, f"\\{char}")
        return escaped
    
    def _extract_capabilities(self, model: ModelInfo) -> List[str]:
        """Extract model capabilities for display."""
        capabilities: List[str] = []

        input_modalities = (
            model.get("input_modalities") or model.get("modality") or []
        )
        if model.get("supports_images") or "image" in [
            entry.lower() for entry in input_modalities
        ]:
            capabilities.append("Image-in")

        params = model.get("supported_parameters", []) or []
        normalized_params = [param.lower() for param in params]
        if model.get("tools") or "tools" in normalized_params:
            capabilities.append("Tools")
        if model.get("function_calling") or "function_calling" in normalized_params:
            capabilities.append("Func-calls")
        if model.get("json_output") or "response_format" in normalized_params:
            capabilities.append("JSON-mode")
        if "prompt_caching" in normalized_params:
            capabilities.append("Prompt-cache")

        return capabilities
    
    def build_model_markdown(self, model_id: str) -> str:
        """
        Generate comprehensive markdown documentation for a model.

        Args:
            model_id: The model identifier to document.

        Returns:
            Formatted markdown string with model information.
        """
        model = self.find_model(model_id)
        if not model:
            return (
                "### Model\n"
                f"`{self._escape_markdown(model_id)}`\n\n"
                "*No metadata found from `/models` endpoint.*"
            )

        provider_key = self._determine_provider_key(model)

        name = model.get("name") or model.get("id") or model_id
        provider_display = (
            model.get("owned_by")
            or model.get("organization")
            or model.get("provider")
            or "—"
        )

        permalink = model.get("permalink") or (
            f"https://openrouter.ai/models/{model.get('id', model_id)}"
        )

        context_length = (
            model.get("context_length")
            or model.get("max_context")
            or model.get("context")
            or "—"
        )

        pricing_display = self._format_pricing(
            model.get("pricing"),
            provider_key,
        )

        input_modalities = ", ".join(
            model.get("input_modalities")
            or model.get("modality")
            or ["text"]
        )
        output_modalities = ", ".join(
            model.get("output_modalities") or ["text"]
        )

        params = model.get("supported_parameters") or []
        if params:
            sorted_params = sorted(dict.fromkeys(params))
            params_display = ", ".join(sorted_params[:12])
            if len(sorted_params) > 12:
                params_display += f" (+{len(sorted_params) - 12} more)"
        else:
            params_display = "—"

        capabilities = self._extract_capabilities(model)
        capabilities_display = ", ".join(capabilities) if capabilities else "—"

        description = (model.get("description") or "").strip()
        if description:
            if len(description) > 560:
                description = description[:560].rstrip() + "…"
        else:
            description = "—"

        context = {
            "name": self._escape_markdown(name),
            "permalink": permalink,
            "provider": self._escape_markdown(str(provider_display)),
            "context_length": self._escape_markdown(str(context_length)),
            "pricing": pricing_display,
            "input_modalities": self._escape_markdown(
                input_modalities or "text"
            ),
            "output_modalities": self._escape_markdown(
                output_modalities or "text"
            ),
            "supported_parameters": self._escape_markdown(params_display),
            "capabilities": self._escape_markdown(capabilities_display),
            "description": self._escape_markdown(description),
        }

        return self._render_markdown_template(context)


# =============================================================================
# Main Application Class
# =============================================================================

class GradioAgentLab:
    """Main application class for the Gradio Agent Lab interface."""

    _VALID_CHAT_ROLES: Tuple[str, ...] = ("system", "user", "assistant", "tool")

    def __init__(self, config: Optional[OpenRouterConfig] = None) -> None:
        """
        Initialize the Gradio Agent Lab application.

        Args:
            config: OpenRouter configuration. If None, creates default config.
        """
        self.config = config or OpenRouterConfig()
        self.client = OpenRouterClient(self.config)
        self.metadata_service = ModelMetadataService(self.client)
        
        # Cache initial model choices
        self.model_choices = self.metadata_service.get_model_choices()
    
    def _sanitize_chat_content(self, content: Any) -> str:
        """Normalize chat content by stripping control characters and whitespace."""

        if content is None:
            return ""

        text = str(content)
        # Normalize carriage returns before stripping control characters.
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = CONTROL_CHAR_PATTERN.sub("", normalized)
        return cleaned.strip()

    def _normalize_chat_message(self, message: Any) -> Optional[ChatMessage]:
        """Validate and sanitize a raw chat message payload."""

        if not isinstance(message, dict):
            return None

        role_value = message.get("role")
        if not isinstance(role_value, str):
            return None

        role = role_value.strip().lower()
        if role not in self._VALID_CHAT_ROLES:
            return None

        sanitized_content = self._sanitize_chat_content(message.get("content"))
        if not sanitized_content:
            return None

        return {"role": role, "content": sanitized_content}

    def _build_chat_messages(
        self,
        history: List[ChatMessage],
        system_prompt: str
    ) -> List[ChatMessage]:
        """Build the complete message list including sanitized system prompt."""

        messages: List[ChatMessage] = []

        sanitized_system = self._sanitize_chat_content(system_prompt)
        if sanitized_system:
            messages.append({"role": "system", "content": sanitized_system})

        for raw_message in history:
            normalized = self._normalize_chat_message(raw_message)
            if normalized:
                messages.append(normalized)

        return messages

    def _format_error_message(self, icon: str, category: str, detail: str) -> str:
        """Generate a user-facing, sanitized error message for the chat stream."""

        sanitized_category = self._sanitize_chat_content(category) or "Error"
        sanitized_detail = self._sanitize_chat_content(detail) or "An unknown error occurred."
        icon_display = icon or "❌"
        return f"{icon_display} **{sanitized_category}**: {sanitized_detail}"

    def _build_sidebar_error_markdown(self, model_id: str) -> str:
        """Produce a safe fallback markdown payload for sidebar failures."""

        safe_model_id = self._sanitize_chat_content(model_id) or self.config.default_model
        return f"### Model\n`{safe_model_id}`\n\n*Error loading metadata.*"

    @staticmethod
    def _build_sidebar_payload(model_id: str, markdown: str) -> Tuple[str, str]:
        """Package sidebar updates in a consistent tuple format."""

        return model_id, markdown

    def _build_dropdown_update(
        self,
        choices: List[Tuple[str, str]],
        value: str,
    ) -> ComponentUpdate:
        """Create a sanitized Gradio component update payload for dropdowns."""

        sanitized_choices = self._sanitize_model_choices(choices)
        if not sanitized_choices:
            sanitized_choices = [(self.config.default_model, self.config.default_model)]

        sanitized_value = self._sanitize_chat_content(value)
        if not sanitized_value:
            sanitized_value = sanitized_choices[0][1]

        valid_values = {choice[1] for choice in sanitized_choices}
        if sanitized_value not in valid_values:
            sanitized_value = sanitized_choices[0][1]

        update_payload = gr.update(choices=sanitized_choices, value=sanitized_value)
        return cast(ComponentUpdate, update_payload)

    def _sanitize_model_choices(
        self, choices: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        """Remove unsafe entries from model dropdown choices."""

        sanitized: List[Tuple[str, str]] = []
        for label, value in choices:
            sanitized_value = self._sanitize_chat_content(value)
            if not sanitized_value:
                continue

            sanitized_label = self._sanitize_chat_content(label) or sanitized_value
            sanitized.append((sanitized_label, sanitized_value))

        return sanitized
    
    def handle_chat_stream(
        self,
        history: List[ChatMessage],
        model: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        seed: int,
    ) -> Generator[str, None, None]:
        """
        Handle streaming chat completion with comprehensive error handling.
        
        Args:
            history: Chat message history.
            model: Model identifier to use.
            system_prompt: System instructions.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            top_p: Nucleus sampling parameter.
            seed: Random seed for reproducible outputs.
            
        Yields:
            Individual content tokens from the streaming response.
        """
        safe_model = self._sanitize_chat_content(model) or self.config.default_model

        try:
            messages = self._build_chat_messages(history, system_prompt)

            yield from self.client.stream_chat_completion(
                model=safe_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                seed=seed,
            )

        except AuthenticationError:
            yield self._format_error_message(
                "❌",
                "Authentication Error",
                "Invalid OpenRouter API key. Please check your configuration.",
            )
        except ModelNotFoundError:
            yield self._format_error_message(
                "❌",
                "Model Error",
                f"Model '{safe_model}' not found on OpenRouter.",
            )
        except APIConnectionError:
            yield self._format_error_message(
                "❌",
                "Connection Error",
                "Unable to reach OpenRouter API. Please check your internet connection.",
            )
        except OpenRouterError as exc:
            yield self._format_error_message("❌", "API Error", str(exc))
        except Exception as exc:
            detail = str(exc) or exc.__class__.__name__
            yield self._format_error_message("❌", "Unexpected Error", detail)

    def update_model_sidebar(self, model_id: str) -> Tuple[str, str]:
        """
        Update the model metadata sidebar.

        Args:
            model_id: The selected model identifier.

        Returns:
            Tuple of (model_id, markdown_content) for UI updates.
        """
        safe_model_id = self._sanitize_chat_content(model_id) or self.config.default_model

        try:
            markdown_content = self.metadata_service.build_model_markdown(safe_model_id)
        except Exception:
            markdown_content = self._build_sidebar_error_markdown(safe_model_id)

        return self._build_sidebar_payload(safe_model_id, markdown_content)
    
    def refresh_models(self, current_model: str) -> Tuple[ComponentUpdate, str, str]:
        """Refresh the model list from OpenRouter API.

        Args:
            current_model: Currently selected model.

        Returns:
            Tuple of (dropdown_update, model_id, markdown_content) for UI
            updates. The first element is a Gradio component update payload
            that exposes ``choices`` and ``value`` for downstream consumers
            (UI bindings, validation utilities, and tests).
        """
        # Clear the cache to force fresh fetch
        self.client.fetch_models.cache_clear()

        # Get fresh model choices
        new_choices = self.metadata_service.get_model_choices()
        sanitized_choices = self._sanitize_model_choices(new_choices)
        if not sanitized_choices:
            sanitized_choices = [(self.config.default_model, self.config.default_model)]

        self.model_choices = sanitized_choices

        safe_current = self._sanitize_chat_content(current_model) or self.config.default_model

        # Determine new selected value
        available_values = {choice[1] for choice in sanitized_choices}
        if safe_current in available_values:
            new_value = safe_current
        elif sanitized_choices:
            new_value = sanitized_choices[0][1]
        else:
            new_value = self.config.default_model

        new_value = self._sanitize_chat_content(new_value) or self.config.default_model

        # Generate metadata for the selected model
        markdown_content = self.metadata_service.build_model_markdown(new_value)

        # Use helper to ensure consistent update payload.
        dropdown_update = self._build_dropdown_update(sanitized_choices, new_value)

        return dropdown_update, new_value, markdown_content
    
    def create_interface(self) -> gr.Blocks:
        """
        Create the complete Gradio interface.
        
        Returns:
            Configured Gradio Blocks interface.
        """
        with gr.Blocks(title="Agent Lab · OpenRouter", theme="soft") as demo:
            gr.Markdown("# Agent Lab · OpenRouter\nQuickly A/B system prompts and models (streaming)")
            
            # Model metadata sidebar
            with gr.Sidebar(label="Model Metadata", open=True, width=320):
                gr.Markdown("## Model Metadata")
                sidebar_model_id = gr.Textbox(
                    label="Model ID (readonly)", 
                    interactive=False
                )
                sidebar_markdown = gr.Markdown(
                    value="*(select a model to load metadata)*"
                )
                refresh_button = gr.Button("🔄 Refresh Models")
            
            # Main control panel
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=self.model_choices,
                    value=self.model_choices[0][1] if self.model_choices else self.config.default_model,
                    label="Model (OpenRouter)"
                )
                temperature_slider = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.7,
                    step=0.05,
                    label="temperature"
                )
                top_p_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=1.0,
                    step=0.01,
                    label="top_p"
                )
            
            with gr.Row():
                max_tokens_input = gr.Number(
                    value=1024,
                    precision=0,
                    label="max_tokens (response cap)"
                )
                seed_input = gr.Number(
                    value=0,
                    precision=0,
                    label="seed (0 = random)"
                )
            
            system_prompt_input = gr.Textbox(
                label="System Instructions",
                placeholder="e.g., You are a meticulous code-review copilot. Use evidence-tagged findings.",
                lines=5,
            )
            
            # Chat interface
            chat_interface = gr.ChatInterface(
                fn=lambda messages, model, system, temp, max_tok, top_p_val, seed_val:
                    self.handle_chat_stream(
                        messages, model, system, temp, max_tok, top_p_val, seed_val
                    ),
                type="messages",
                fill_height=True,
                autofocus=True,
                submit_btn="Send",
                stop_btn="Stop",
                additional_inputs=[
                    model_dropdown,
                    system_prompt_input,
                    temperature_slider,
                    max_tokens_input,
                    top_p_slider,
                    seed_input,
                ],
            )
            
            # Event handlers
            model_dropdown.change(
                fn=self.update_model_sidebar,
                inputs=model_dropdown,
                outputs=[sidebar_model_id, sidebar_markdown]
            )
            
            refresh_button.click(
                fn=self.refresh_models,
                inputs=model_dropdown,
                outputs=[model_dropdown, sidebar_model_id, sidebar_markdown]
            )
        
        return demo


# =============================================================================
# Module-Level Singleton & Public API
# =============================================================================

_app: GradioAgentLab | None = None


def _get_app() -> GradioAgentLab:
    """Return a lazily-instantiated :class:`GradioAgentLab` singleton."""

    global _app

    if _app is None:
        try:
            # Security: fail fast when the API key is missing to avoid
            # accidental unauthenticated traffic to OpenRouter.
            _app = GradioAgentLab()
        except ValueError as exc:
            raise RuntimeError(
                "OPENROUTER_API_KEY environment variable must be set before "
                "importing or using agent_lab."
            ) from exc

    return _app


demo = _get_app().create_interface()


def stream_openrouter(
    model: str,
    system_prompt: str,
    history: List[ChatMessage],
    temperature: float,
    max_tokens: int,
    top_p: float,
    seed: int,
) -> Generator[str, None, None]:
    """Stream chat completions for the provided conversation context."""

    return _get_app().handle_chat_stream(
        history=history,
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        seed=seed,
    )


def reply_fn(
    history: List[ChatMessage],
    model: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    seed: int,
) -> Generator[str, None, None]:
    """Wrapper for Gradio ChatInterface callback signature."""

    return stream_openrouter(
        model=model,
        system_prompt=system_prompt,
        history=history,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        seed=seed,
    )


@lru_cache
def fetch_models_raw() -> List[ModelInfo]:
    """Fetch the raw OpenRouter model list with memoization."""

    return _get_app().client.fetch_models()


def list_model_choices() -> List[Tuple[str, str]]:
    """Return dropdown-ready model choices."""

    return _get_app().metadata_service.get_model_choices()


MODEL_CHOICES = list_model_choices()


def find_model(model_id: str) -> Optional[ModelInfo]:
    """Locate metadata for the requested model identifier."""

    return _get_app().metadata_service.find_model(model_id)


def build_model_markdown(model_id: str) -> str:
    """Generate Markdown describing the selected model."""

    return _get_app().metadata_service.build_model_markdown(model_id)


def update_sidebar(model_id: str) -> Tuple[str, str]:
    """Return sidebar updates for a newly selected model."""

    return _get_app().update_model_sidebar(model_id)


def refresh_models(current_model: str) -> Tuple[ComponentUpdate, str, str]:
    """Refresh dropdown choices and sidebar metadata from OpenRouter.

    The returned ``ComponentUpdate`` preserves ``choices`` and ``value`` keys
    so downstream callers (UI bindings, tests, or validation scripts) can rely
    on a dictionary-like contract without needing to inspect Gradio internals.
    """

    dropdown_update, model_id, markdown = _get_app().refresh_models(current_model)

    if isinstance(dropdown_update, dict):
        dropdown_payload = cast(ComponentUpdate, dropdown_update)
    elif hasattr(dropdown_update, "get_config"):
        config = dropdown_update.get_config()  # type: ignore[call-arg]
        dropdown_payload = cast(
            ComponentUpdate,
            {
                "__type__": "update",
                "choices": config.get("choices", []),
                "value": config.get("value"),
            },
        )
    else:
        dropdown_payload = cast(
            ComponentUpdate,
            {
                "__type__": "update",
                "choices": getattr(dropdown_update, "choices", []),
                "value": getattr(dropdown_update, "value", None),
            },
        )

    if "__type__" not in dropdown_payload:
        dropdown_payload["__type__"] = "update"

    return dropdown_payload, model_id, markdown


# =============================================================================
# Application Entry Point
# =============================================================================

def main() -> None:
    """Main entry point for the application."""
    try:
        config = OpenRouterConfig()
        app = GradioAgentLab(config)
        demo = app.create_interface()
        
        print(f"🚀 Starting Agent Lab with model: {config.default_model}")
        demo.launch()
        
    except ValueError as exc:
        print(f"❌ Configuration Error: {exc}")
        print("Please set the OPENROUTER_API_KEY environment variable.")
        raise SystemExit(1) from exc
    except Exception as exc:
        print(f"❌ Startup Error: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
