"""LLM-as-judge grader for open-ended answers."""

from __future__ import annotations

import json
import re
from typing import Any

from evalkit.graders.base import Grader, register
from evalkit.models.base import ModelClient
from evalkit.types import GradeResult, Task

DEFAULT_RUBRIC = (
    "The response is correct if it conveys the same final answer as the reference "
    "solution. Ignore formatting and minor wording differences."
)

JUDGE_SYSTEM = (
    "You are a strict, impartial grader. Compare a candidate response to a reference "
    "solution using the rubric. Reply with a single JSON object: "
    '{"score": <number 0-1>, "passed": <true|false>, "reason": "<one sentence>"}'
)

JUDGE_TEMPLATE = """## Task
{task}

## Reference solution
{reference}

## Rubric
{rubric}

## Candidate response
{candidate}

Return only the JSON object."""


@register("llm_judge")
class LLMJudgeGrader(Grader):
    needs_context = True

    def __init__(self, judge_client: ModelClient | None = None, rubric: str = DEFAULT_RUBRIC, pass_threshold: float = 0.5, **kw: Any) -> None:
        super().__init__(**kw)
        if judge_client is None:
            raise ValueError("llm_judge grader requires a judge model (use --judge provider:model)")
        self.judge = judge_client
        self.rubric = rubric
        self.pass_threshold = pass_threshold

    def score(self, task: Task, answer: str) -> GradeResult:
        task_text = task.input.get("question") or json.dumps(task.input, ensure_ascii=False)
        prompt = JUDGE_TEMPLATE.format(
            task=task_text, reference=task.expected, rubric=self.rubric, candidate=answer
        )
        response = self.judge.complete(JUDGE_SYSTEM, prompt)
        verdict = self._parse(response.text)
        if verdict is None:
            return GradeResult.fail("judge returned unparseable verdict", judge_raw=response.text[:300])
        score = max(0.0, min(1.0, float(verdict.get("score", 0.0))))
        passed = bool(verdict.get("passed", score >= self.pass_threshold)) and score >= self.pass_threshold
        reason = str(verdict.get("reason", "")) or ("judge accepted" if passed else "judge rejected")
        return GradeResult(passed, score, reason, {"judge_model": response.model})

    @staticmethod
    def _parse(text: str) -> dict[str, Any] | None:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None
