"""String-comparison graders."""

from __future__ import annotations

import re
import string
import unicodedata
from typing import Any

from evalkit.graders.base import Grader, register
from evalkit.types import GradeResult, Task


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _expected_list(expected: Any) -> list[str]:
    if expected is None:
        return []
    if isinstance(expected, (list, tuple)):
        return [str(e) for e in expected]
    return [str(expected)]


@register("exact")
class ExactMatchGrader(Grader):
    """Exact string equality (after strip). Accepts a list of acceptable answers."""

    def __init__(self, case_sensitive: bool = True, **kw: Any) -> None:
        super().__init__(**kw)
        self.case_sensitive = case_sensitive

    def score(self, task: Task, answer: str) -> GradeResult:
        candidates = _expected_list(task.expected)
        a = answer if self.case_sensitive else answer.lower()
        for exp in candidates:
            e = exp.strip() if self.case_sensitive else exp.strip().lower()
            if a == e:
                return GradeResult.ok("exact match")
        return GradeResult.fail(f"expected {candidates[0] if candidates else None!r}, got {answer[:80]!r}")


@register("normalized")
class NormalizedMatchGrader(Grader):
    """Case/punctuation/article-insensitive match, SQuAD-style, with token F1 partial credit."""

    def __init__(self, f1_threshold: float = 1.0, **kw: Any) -> None:
        super().__init__(**kw)
        self.f1_threshold = f1_threshold

    @staticmethod
    def token_f1(pred: str, gold: str) -> float:
        p, g = normalize(pred).split(), normalize(gold).split()
        if not p or not g:
            return float(p == g)
        common = sum(min(p.count(t), g.count(t)) for t in set(p))
        if common == 0:
            return 0.0
        precision, recall = common / len(p), common / len(g)
        return 2 * precision * recall / (precision + recall)

    def score(self, task: Task, answer: str) -> GradeResult:
        candidates = _expected_list(task.expected)
        best = max((self.token_f1(answer, c) for c in candidates), default=0.0)
        if best >= self.f1_threshold:
            return GradeResult(True, best, "normalized match", {"f1": best})
        return GradeResult.fail(f"normalized mismatch (f1={best:.2f})", score=best, f1=best)


@register("contains")
class ContainsGrader(Grader):
    """Pass if the output contains the expected string(s).

    ``mode="all"`` requires every expected item; ``mode="any"`` requires at least one.
    ``forbidden`` lists substrings that must not appear.
    """

    def __init__(self, mode: str = "all", case_sensitive: bool = False, forbidden: list[str] | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        if mode not in {"all", "any"}:
            raise ValueError("contains grader mode must be 'all' or 'any'")
        self.mode = mode
        self.case_sensitive = case_sensitive
        self.forbidden = forbidden or []

    def _fold(self, s: str) -> str:
        return s if self.case_sensitive else s.lower()

    def score(self, task: Task, answer: str) -> GradeResult:
        hay = self._fold(answer)
        bad = [f for f in self.forbidden if self._fold(f) in hay]
        if bad:
            return GradeResult.fail(f"contains forbidden text: {bad}", forbidden=bad)
        needles = _expected_list(task.expected)
        hits = [n for n in needles if self._fold(n) in hay]
        missing = [n for n in needles if n not in hits]
        frac = len(hits) / len(needles) if needles else 1.0
        passed = bool(hits) if self.mode == "any" else not missing
        if passed:
            return GradeResult(True, frac if self.mode == "all" else 1.0, "contains expected text")
        return GradeResult.fail(f"missing expected text: {missing}", score=frac, missing=missing)


@register("regex")
class RegexGrader(Grader):
    """Pass if ``pattern`` (option, or ``task.expected``) matches the output."""

    def __init__(self, pattern: str | None = None, fullmatch: bool = False, ignore_case: bool = True, **kw: Any) -> None:
        super().__init__(**kw)
        self.pattern = pattern
        self.fullmatch = fullmatch
        self.flags = re.IGNORECASE if ignore_case else 0

    def score(self, task: Task, answer: str) -> GradeResult:
        pattern = self.pattern or str(task.expected)
        rx = re.compile(pattern, self.flags | re.DOTALL)
        matched = rx.fullmatch(answer) if self.fullmatch else rx.search(answer)
        if matched:
            return GradeResult.ok("regex matched")
        return GradeResult.fail(f"output did not match /{pattern}/")
