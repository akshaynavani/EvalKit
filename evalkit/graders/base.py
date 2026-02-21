"""Grader base class, answer extraction and registry."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Callable

from evalkit.types import GradeResult, Task

_REGISTRY: dict[str, type["Grader"]] = {}

NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?|-?\.\d+")
FENCE_RE = re.compile(r"```(?:[\w+-]+)?\n(.*?)```", re.DOTALL)


def register(name: str) -> Callable[[type["Grader"]], type["Grader"]]:
    def decorator(cls: type["Grader"]) -> type["Grader"]:
        _REGISTRY[name] = cls
        cls.kind = name
        return cls

    return decorator


def available_graders() -> list[str]:
    return sorted(_REGISTRY)


def extract_answer(output: str, how: str | None) -> str:
    """Pull the gradable part out of a raw model output.

    ``how`` may be ``None`` (whole output), ``last_line``, ``last_number``,
    ``code_block`` or ``regex:<pattern>`` (first capture group, else whole match).
    """
    text = output.strip()
    if not how:
        return text
    if how == "last_line":
        lines = [ln for ln in text.splitlines() if ln.strip()]
        return lines[-1].strip() if lines else ""
    if how == "last_number":
        numbers = NUMBER_RE.findall(text)
        return numbers[-1] if numbers else ""
    if how == "code_block":
        blocks = FENCE_RE.findall(text)
        return blocks[-1].strip() if blocks else text
    if how.startswith("regex:"):
        match = re.search(how[len("regex:"):], text, re.DOTALL | re.IGNORECASE)
        if not match:
            return ""
        return (match.group(1) if match.groups() else match.group(0)).strip()
    raise ValueError(f"unknown extraction mode {how!r}")


class Grader(ABC):
    """Scores a model output for a task. Subclasses implement :meth:`score`."""

    kind: str = "base"

    def __init__(self, extract: str | None = None, **options: Any) -> None:
        self.extract = extract
        self.options = options

    def grade(self, task: Task, output: str) -> GradeResult:
        answer = extract_answer(output, self.extract)
        if self.extract and not answer:
            return GradeResult.fail(f"could not extract answer ({self.extract})", extracted="")
        result = self.score(task, answer)
        result.details.setdefault("extracted", answer[:500])
        return result

    @abstractmethod
    def score(self, task: Task, answer: str) -> GradeResult:
        ...


def build_grader(spec: dict[str, Any] | str, **context: Any) -> Grader:
    """Instantiate a grader from a spec like ``{"type": "numeric", "tolerance": 0.01}``.

    ``context`` carries runtime dependencies (e.g. ``judge_client`` for LLM judges).
    """
    if isinstance(spec, str):
        spec = {"type": spec}
    spec = dict(spec)
    kind = spec.pop("type", "exact")
    if kind not in _REGISTRY:
        raise ValueError(f"unknown grader {kind!r}; available: {', '.join(available_graders())}")
    cls = _REGISTRY[kind]
    if getattr(cls, "needs_context", False):
        spec.update({k: v for k, v in context.items() if v is not None})
    return cls(**spec)
