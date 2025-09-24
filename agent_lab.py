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
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, TypedDict, Union, cast
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
    
    def __init__(self, client: OpenRouterClient) -> None:
        """Initialize with OpenRouter client."""
        self.client = client
    
    def get_model_choices(self) -> List[Tuple[str, str]]:
        """
        Get formatted model choices for dropdown UI component.
        
        Returns:
            List of (display_name, model_id) tuples sorted by preference.
        """
        try:
            models = self.client.fetch_models()
        except OpenRouterError:
            # Fall back to default if API fails
            gr.Warning("Unable to fetch models from OpenRouter. Using default.")
            return [(self.client.config.default_model, self.client.config.default_model)]
        
        choices = []
        for model in models:
            model_id = model.get("id") or model.get("name")
            if model_id:
                choices.append((model_id, model_id))
        
        if not choices:
            return [(self.client.config.default_model, self.client.config.default_model)]
        
        # Sort by preference
        preferred = self.client.config.preferred_models
        priority_map = {model: idx for idx, model in enumerate(preferred)}
        
        choices.sort(key=lambda x: (priority_map.get(x[1], 10_000), x[1]))
        
        return choices
    
    def find_model(self, model_id: str) -> Optional[ModelInfo]:
        """
        Find model information by ID.
        
        Args:
            model_id: The model identifier to search for.
            
        Returns:
            Model information dictionary if found, None otherwise.
        """
        try:
            models = self.client.fetch_models()
            for model in models:
                if model.get("id") == model_id or model.get("name") == model_id:
                    return model
        except OpenRouterError:
            pass
        
        return None
    
    def _format_pricing(self, pricing: Optional[Dict[str, Any]]) -> str:
        """Format pricing information for display."""
        if not isinstance(pricing, dict):
            return "—"
        
        parts = []
        if "prompt" in pricing:
            parts.append(f"Input: ${pricing['prompt']}/1M tok")
        if "completion" in pricing:
            parts.append(f"Output: ${pricing['completion']}/1M tok")
        
        # Add other pricing fields if present
        for key, value in pricing.items():
            if key not in ("prompt", "completion"):
                parts.append(f"{key.capitalize()}: ${value}")
        
        return ", ".join(parts) if parts else "—"
    
    def _extract_capabilities(self, model: ModelInfo) -> List[str]:
        """Extract model capabilities for display."""
        capabilities = []
        
        if (model.get("supports_images") or 
            "image" in (model.get("input_modalities") or [])):
            capabilities.append("Image-in")
        
        params = model.get("supported_parameters", [])
        if model.get("tools") or "tools" in params:
            capabilities.append("Tools")
        if model.get("function_calling") or "function_calling" in params:
            capabilities.append("Func-calls")
        if model.get("json_output") or "response_format" in params:
            capabilities.append("JSON-mode")
        
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
            return f"### Model\n`{model_id}`\n\n*No metadata found from `/models` endpoint.*"
        
        # Extract basic information
        name = model.get("name") or model.get("id") or model_id
        provider = (
            model.get("owned_by") or 
            model.get("organization") or 
            model.get("provider") or 
            "—"
        )
        
        # Build model URL
        permalink = model.get("permalink")
        if not permalink:
            permalink = f"https://openrouter.ai/models/{model.get('id', model_id)}"
        
        # Extract technical specifications
        context_length = (
            model.get("context_length") or 
            model.get("max_context") or 
            model.get("context") or 
            "—"
        )
        
        pricing = self._format_pricing(model.get("pricing"))
        
        # Extract modalities
        input_modalities = ", ".join(
            model.get("input_modalities", []) or 
            model.get("modality", []) or 
            ["text"]
        )
        output_modalities = ", ".join(
            model.get("output_modalities", []) or ["text"]
        )
        
        # Format supported parameters
        params = model.get("supported_parameters", [])
        if params:
            sorted_params = sorted(set(params))
            params_display = ", ".join(sorted_params[:12])
            if len(sorted_params) > 12:
                params_display += f" (+{len(sorted_params) - 12} more)"
        else:
            params_display = "—"
        
        # Extract capabilities
        capabilities = self._extract_capabilities(model)
        capabilities_display = ", ".join(capabilities) if capabilities else "—"
        
        # Format description
        description = model.get("description", "").strip()
        if description:
            if len(description) > 560:
                description = description[:560].rstrip() + "…"
        else:
            description = "—"
        
        # Build markdown sections
        sections = [
            f"### Model\n[`{name}`]({permalink})",
            "",
            f"**Provider:** `{provider}`",
            "",
            f"**Context length:** `{context_length}` tokens",
            "",
            f"**Pricing:** {pricing}",
            "",
            "**Modalities:**",
            f"- Input: `{input_modalities}`",
            f"- Output: `{output_modalities}`",
            "",
            f"**Supported params:** {params_display}",
            "",
            f"**Capabilities:** {capabilities_display}",
            "",
            "**Notes:**",
            description,
            "",
            "> Data from OpenRouter **Models API**. Fields vary by provider.",
        ]
        
        return "\n".join(sections)


# =============================================================================
# Main Application Class
# =============================================================================

class GradioAgentLab:
    """Main application class for the Gradio Agent Lab interface."""
    
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
    
    def _build_chat_messages(
        self, 
        history: List[ChatMessage], 
        system_prompt: str
    ) -> List[ChatMessage]:
        """
        Build the complete message list including system prompt.
        
        Args:
            history: Chat message history from Gradio.
            system_prompt: System instructions to prepend.
            
        Returns:
            Complete list of chat messages.
        """
        messages: List[ChatMessage] = []
        
        # Add system prompt if provided
        if system_prompt and system_prompt.strip():
            messages.append({
                "role": "system", 
                "content": system_prompt.strip()
            })
        
        # Add conversation history
        messages.extend(history)
        
        return messages
    
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
        try:
            messages = self._build_chat_messages(history, system_prompt)
            
            yield from self.client.stream_chat_completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                seed=seed,
            )
            
        except AuthenticationError:
            yield "❌ **Authentication Error**: Invalid OpenRouter API key. Please check your configuration."
        except ModelNotFoundError:
            yield f"❌ **Model Error**: Model '{model}' not found on OpenRouter."
        except APIConnectionError:
            yield "❌ **Connection Error**: Unable to reach OpenRouter API. Please check your internet connection."
        except OpenRouterError as exc:
            yield f"❌ **API Error**: {exc}"
        except Exception as exc:
            yield f"❌ **Unexpected Error**: {exc}"
    
    def update_model_sidebar(self, model_id: str) -> Tuple[str, str]:
        """
        Update the model metadata sidebar.
        
        Args:
            model_id: The selected model identifier.
            
        Returns:
            Tuple of (model_id, markdown_content) for UI updates.
        """
        try:
            markdown_content = self.metadata_service.build_model_markdown(model_id)
        except Exception:
            markdown_content = f"### Model\n`{model_id}`\n\n*Error loading metadata.*"
        
        return model_id, markdown_content
    
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
        
        # Determine new selected value
        if any(choice[1] == current_model for choice in new_choices):
            new_value = current_model
        else:
            new_value = new_choices[0][1] if new_choices else self.config.default_model
        
        # Generate metadata for the selected model
        markdown_content = self.metadata_service.build_model_markdown(new_value)

        # Use gr.update to ensure a dictionary-like payload compatible with
        # event handlers and validation helpers that rely on ``choices`` and
        # ``value`` subscripting.
        dropdown_update = cast(
            ComponentUpdate,
            gr.update(choices=new_choices, value=new_value),
        )

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
                retry_btn=None,
                undo_btn="Delete last",
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
