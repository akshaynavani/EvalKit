"""Numeric answer grader with absolute/relative tolerance."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

from evalkit.graders.base import NUMBER_RE, Grader, register
from evalkit.types import GradeResult, Task


def parse_number(text: str) -> float | None:
    text = text.strip().replace("$", "").replace("%", "")
    try:
        return float(text.replace(",", ""))
    except ValueError:
        pass
    try:
        return float(Fraction(text.replace(" ", "")))
    except (ValueError, ZeroDivisionError):
        pass
    found = NUMBER_RE.findall(text)
    if found:
        try:
            return float(found[-1].replace(",", ""))
        except ValueError:
            return None
    return None


@register("numeric")
class NumericGrader(Grader):
    """Compares numbers using ``math.isclose(abs_tol, rel_tol)``."""

    def __init__(self, abs_tol: float = 1e-6, rel_tol: float = 1e-6, **kw: Any) -> None:
        super().__init__(**kw)
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol

    def score(self, task: Task, answer: str) -> GradeResult:
        got = parse_number(answer)
        expected = task.expected if isinstance(task.expected, (int, float)) else parse_number(str(task.expected))
        if expected is None:
            return GradeResult.fail(f"task expected value {task.expected!r} is not numeric")
        if got is None:
            return GradeResult.fail(f"no number found in output {answer[:60]!r}")
        if math.isclose(got, float(expected), abs_tol=self.abs_tol, rel_tol=self.rel_tol):
            return GradeResult.ok("numeric match", value=got)
        return GradeResult.fail(f"expected {expected}, got {got}", value=got, error=got - float(expected))
