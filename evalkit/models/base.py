"""Model client protocol and shared helpers."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from evalkit.types import ModelResponse


class ModelError(RuntimeError):
    """Raised when a model call fails permanently."""


class RetryableModelError(ModelError):
    """Raised for transient failures (rate limits, 5xx, timeouts)."""


@dataclass
class GenerationConfig:
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_s: float = 60.0


class ModelClient(ABC):
    """Minimal interface every backend implements."""

    name: str = "model"

    def __init__(self, model: str, config: GenerationConfig | None = None) -> None:
        self.model = model
        self.config = config or GenerationConfig()

    @abstractmethod
    def _complete(self, system: str, user: str) -> ModelResponse:
        ...

    def complete(self, system: str, user: str, retries: int = 3, backoff_s: float = 1.0) -> ModelResponse:
        """Call the model with exponential backoff on retryable errors."""
        attempt = 0
        while True:
            start = time.perf_counter()
            try:
                response = self._complete(system, user)
            except RetryableModelError:
                attempt += 1
                if attempt > retries:
                    raise
                time.sleep(backoff_s * (2 ** (attempt - 1)))
                continue
            if not response.latency_s:
                response.latency_s = time.perf_counter() - start
            return response

    def describe(self) -> dict[str, object]:
        return {
            "provider": self.name,
            "model": self.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
