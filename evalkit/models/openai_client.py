"""OpenAI-compatible Chat Completions client (also works with local servers such as vLLM)."""

from __future__ import annotations

import os

from evalkit.models._http import post_json
from evalkit.models.base import GenerationConfig, ModelClient, ModelError
from evalkit.types import ModelResponse


class OpenAIClient(ModelClient):
    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        config: GenerationConfig | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(model, config)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ModelError("OPENAI_API_KEY is not set")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")

    def _complete(self, system: str, user: str) -> ModelResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        data = post_json(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            payload={
                "model": self.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            },
            timeout_s=self.config.timeout_s,
        )
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise ModelError(f"unexpected OpenAI response shape: {data}") from exc
        usage = data.get("usage", {})
        return ModelResponse(
            text=text,
            model=data.get("model", self.model),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            raw=data,
        )
