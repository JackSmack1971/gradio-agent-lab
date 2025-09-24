"""Provider-specific model fixtures for metadata service tests."""

from __future__ import annotations

from typing import Any, Dict

ANTHROPIC_MODEL: Dict[str, Any] = {
    "id": "anthropic/claude-3.5-sonnet",
    "name": "Claude 3.5 Sonnet",
    "owned_by": "Anthropic",
    "pricing": {
        "input": "0.003",
        "output": "0.015",
        "cache": "0.001",
    },
    "supported_parameters": ["temperature", "top_p"],
    "tools": True,
    "context_length": 200_000,
    "input_modalities": ["text"],
    "output_modalities": ["text"],
    "description": "Enterprise assistant with prompt caching support.",
}

GOOGLE_MODEL: Dict[str, Any] = {
    "id": "google/gemini-2.0-flash",
    "name": "Gemini 2.0 Flash",
    "organization": "Google",
    "modality": ["text", "image"],
    "output_modalities": ["text"],
    "pricing": {
        "prompt": 0.0005,
        "completion": 0.00075,
        "cache": "0.0001",
    },
    "supported_parameters": ["function_calling"],
    "function_calling": True,
    "description": "Fast multimodal generation.",
}

OPENAI_MODEL: Dict[str, Any] = {
    "id": "openai/gpt-4o-mini",
    "name": "GPT-4o Mini",
    "provider": "OpenAI",
    "pricing": {
        "prompt": "0.0003",
        "completion": "0.0006",
    },
    "supported_parameters": ["json_output", "response_format"],
    "json_output": True,
    "input_modalities": ["text"],
    "output_modalities": ["text"],
    "description": "Compact reasoning model for evaluation suites.",
}

MARKDOWN_MODEL: Dict[str, Any] = {
    "id": "openrouter/auto",
    "name": "Router Auto",
    "owned_by": "OpenRouter",
    "context_length": 32000,
    "pricing": {
        "prompt": "0.0001",
        "completion": "0.0002",
        "special": "0.5",
    },
    "input_modalities": ["text", "image"],
    "output_modalities": ["text"],
    "supported_parameters": [
        "temperature",
        "top_p",
        "function_calling",
        "prompt_caching",
        "json_output",
    ],
    "function_calling": True,
    "json_output": True,
    "supports_images": True,
    "permalink": "https://openrouter.ai/models/openrouter/auto",
    "description": "Multi-provider router with `backtick` safe description.",
}
