"""Model client backends and factory."""

from __future__ import annotations

from evalkit.models.base import (
    GenerationConfig,
    ModelClient,
    ModelError,
    RetryableModelError,
)
from evalkit.models.mock import MockClient


def create_client(spec: str, config: GenerationConfig | None = None) -> ModelClient:
    """Build a client from ``provider:model`` (e.g. ``openai:gpt-4o-mini``, ``mock``)."""
    provider, _, model = spec.partition(":")
    provider = provider.lower().strip()
    if provider == "mock":
        return MockClient(model or "mock-1", config=config)
    if provider == "openai":
        from evalkit.models.openai_client import OpenAIClient

        return OpenAIClient(model or "gpt-4o-mini", config=config)
    if provider == "anthropic":
        from evalkit.models.anthropic_client import AnthropicClient

        return AnthropicClient(model or "claude-sonnet-4-5", config=config)
    raise ValueError(f"unknown model provider {provider!r} (expected mock, openai or anthropic)")


__all__ = [
    "GenerationConfig",
    "MockClient",
    "ModelClient",
    "ModelError",
    "RetryableModelError",
    "create_client",
]
