"""Structured-output grader: parse JSON and compare fields."""

from __future__ import annotations

import json
import re
from typing import Any

from evalkit.graders.base import Grader, register
from evalkit.graders.text import normalize
from evalkit.types import GradeResult, Task

_OBJ_RE = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)


def parse_json_loose(text: str) -> Any:
    """Parse JSON, tolerating code fences and leading/trailing prose."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```\w*\n?|```$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = _OBJ_RE.search(stripped)
        if not match:
            raise
        return json.loads(match.group(0))


def _values_equal(a: Any, b: Any, loose_strings: bool) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))
    if isinstance(a, str) and isinstance(b, str) and loose_strings:
        return normalize(a) == normalize(b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_values_equal(a[k], b[k], loose_strings) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_values_equal(x, y, loose_strings) for x, y in zip(a, b))
    return a == b


@register("json")
class JSONMatchGrader(Grader):
    """Checks a JSON output against ``task.expected`` (a dict).

    ``fields`` restricts comparison to a subset of keys; score is the fraction of matching
    fields and the task passes only when all compared fields match.
    """

    def __init__(self, fields: list[str] | None = None, loose_strings: bool = True, allow_extra: bool = True, **kw: Any) -> None:
        super().__init__(**kw)
        self.fields = fields
        self.loose_strings = loose_strings
        self.allow_extra = allow_extra

    def score(self, task: Task, answer: str) -> GradeResult:
        try:
            got = parse_json_loose(answer)
        except json.JSONDecodeError as exc:
            return GradeResult.fail(f"invalid JSON: {exc.msg}")
        expected = task.expected
        if not isinstance(expected, dict):
            ok = _values_equal(got, expected, self.loose_strings)
            return GradeResult.ok("json equal") if ok else GradeResult.fail("json value mismatch")
        if not isinstance(got, dict):
            return GradeResult.fail(f"expected a JSON object, got {type(got).__name__}")

        keys = self.fields or list(expected)
        wrong = [k for k in keys if k not in got or not _values_equal(got[k], expected.get(k), self.loose_strings)]
        extra = [] if self.allow_extra else sorted(set(got) - set(expected))
        frac = (len(keys) - len(wrong)) / len(keys) if keys else 1.0
        if not wrong and not extra:
            return GradeResult.ok("all fields match")
        reasons = []
        if wrong:
            reasons.append(f"mismatched fields: {wrong}")
        if extra:
            reasons.append(f"unexpected fields: {extra}")
        return GradeResult.fail("; ".join(reasons), score=frac, wrong_fields=wrong, extra_fields=extra)
