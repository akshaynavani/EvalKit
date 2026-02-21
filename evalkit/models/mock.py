"""Deterministic offline model used for tests, demos and dry runs."""

from __future__ import annotations

import re
from typing import Callable

from evalkit.models.base import GenerationConfig, ModelClient
from evalkit.types import ModelResponse

Responder = Callable[[str, str], str]


class MockClient(ModelClient):
    """Returns scripted responses.

    Resolution order for each call:
    1. ``responder(system, user)`` if provided.
    2. The first ``(regex, response)`` rule whose pattern matches the user prompt.
       ``response`` may reference regex groups via ``\\1`` / ``\\g<name>``.
    3. ``default`` (``"echo"`` returns the user prompt verbatim).
    """

    name = "mock"

    def __init__(
        self,
        model: str = "mock-1",
        *,
        rules: list[tuple[str, str]] | None = None,
        responder: Responder | None = None,
        default: str = "echo",
        config: GenerationConfig | None = None,
    ) -> None:
        super().__init__(model, config)
        self.rules = [(re.compile(p, re.DOTALL), r) for p, r in (rules or [])]
        self.responder = responder
        self.default = default
        self.calls: list[tuple[str, str]] = []

    def _complete(self, system: str, user: str) -> ModelResponse:
        self.calls.append((system, user))
        text = self._resolve(system, user)
        return ModelResponse(
            text=text,
            model=self.model,
            latency_s=0.0,
            input_tokens=len((system + user).split()),
            output_tokens=len(text.split()),
        )

    def _resolve(self, system: str, user: str) -> str:
        if self.responder is not None:
            return self.responder(system, user)
        for pattern, response in self.rules:
            match = pattern.search(user)
            if match:
                return match.expand(response)
        return user if self.default == "echo" else self.default
