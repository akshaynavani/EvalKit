"""Anthropic Messages API client."""

from __future__ import annotations

import os

from evalkit.models._http import post_json
from evalkit.models.base import GenerationConfig, ModelClient, ModelError
from evalkit.types import ModelResponse

API_VERSION = "2023-06-01"


class AnthropicClient(ModelClient):
    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        config: GenerationConfig | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(model, config)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ModelError("ANTHROPIC_API_KEY is not set")
        self.base_url = (base_url or os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip("/")

    def _complete(self, system: str, user: str) -> ModelResponse:
        payload: dict = {
            "model": self.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [{"role": "user", "content": user}],
        }
        if system:
            payload["system"] = system
        data = post_json(
            f"{self.base_url}/v1/messages",
            headers={"x-api-key": self.api_key, "anthropic-version": API_VERSION},
            payload=payload,
            timeout_s=self.config.timeout_s,
        )
        blocks = data.get("content", [])
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        usage = data.get("usage", {})
        return ModelResponse(
            text=text,
            model=data.get("model", self.model),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            raw=data,
        )
