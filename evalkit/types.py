"""Core data structures shared across EvalKit."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Task:
    """A single evaluation item.

    ``input`` holds the template variables rendered into the prompt, ``expected`` is the
    reference solution, and ``grader`` is a spec dict such as ``{"type": "exact"}``.
    """

    id: str
    category: str
    input: dict[str, Any]
    expected: Any = None
    grader: dict[str, Any] = field(default_factory=lambda: {"type": "exact"})
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelResponse:
    text: str
    model: str
    latency_s: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    raw: dict[str, Any] | None = None


@dataclass
class GradeResult:
    passed: bool
    score: float
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, reason: str = "match", **details: Any) -> "GradeResult":
        return cls(True, 1.0, reason, details)

    @classmethod
    def fail(cls, reason: str, score: float = 0.0, **details: Any) -> "GradeResult":
        return cls(False, score, reason, details)


@dataclass
class TaskResult:
    """Outcome of running and grading one task."""

    task_id: str
    category: str
    prompt: str
    output: str | None
    expected: Any
    passed: bool
    score: float
    reason: str
    error: str | None = None
    latency_s: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    grade_details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskResult":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})
