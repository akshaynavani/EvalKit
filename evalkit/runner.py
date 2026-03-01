"""Runs a prompt + model over a suite and grades each output."""

from __future__ import annotations

import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from evalkit.graders import build_grader
from evalkit.graders.base import Grader
from evalkit.models.base import ModelClient, ModelError
from evalkit.prompts import PromptError, PromptTemplate
from evalkit.types import Task, TaskResult

log = logging.getLogger(__name__)

ProgressFn = Callable[[int, int, TaskResult], None]


@dataclass
class EvalRun:
    run_id: str
    created_at: str
    suite: str
    prompt: PromptTemplate
    model: dict[str, Any]
    results: list[TaskResult] = field(default_factory=list)
    tags: dict[str, Any] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return sum(r.passed for r in self.results) / len(self.results) if self.results else 0.0

    def failures(self) -> list[TaskResult]:
        return [r for r in self.results if not r.passed]


def new_run_id(prefix: str = "run") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:6]}"


class Runner:
    def __init__(
        self,
        client: ModelClient,
        *,
        judge_client: ModelClient | None = None,
        concurrency: int = 4,
        retries: int = 3,
    ) -> None:
        self.client = client
        self.judge_client = judge_client
        self.concurrency = max(1, concurrency)
        self.retries = retries
        self._graders: dict[str, Grader] = {}

    def _grader_for(self, task: Task) -> Grader:
        key = json.dumps(task.grader, sort_keys=True, default=str)
        if key not in self._graders:
            self._graders[key] = build_grader(task.grader, judge_client=self.judge_client)
        return self._graders[key]

    def run_task(self, task: Task, prompt: PromptTemplate) -> TaskResult:
        try:
            system, user = prompt.render(task.input)
        except PromptError as exc:
            return self._errored(task, "", f"prompt error: {exc}")

        try:
            response = self.client.complete(system, user, retries=self.retries)
        except ModelError as exc:
            return self._errored(task, user, f"model error: {exc}")

        try:
            grade = self._grader_for(task).grade(task, response.text)
        except Exception as exc:  # grader bugs should not kill the run
            log.exception("grader failed on task %s", task.id)
            return self._errored(task, user, f"grader error: {type(exc).__name__}: {exc}", output=response.text)

        return TaskResult(
            task_id=task.id,
            category=task.category,
            prompt=user,
            output=response.text,
            expected=task.expected,
            passed=grade.passed,
            score=grade.score,
            reason=grade.reason,
            latency_s=response.latency_s,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            grade_details=_jsonable(grade.details),
        )

    def run(
        self,
        tasks: list[Task],
        prompt: PromptTemplate,
        *,
        suite_name: str = "suite",
        run_id: str | None = None,
        progress: ProgressFn | None = None,
        tags: dict[str, Any] | None = None,
    ) -> EvalRun:
        run = EvalRun(
            run_id=run_id or new_run_id(),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            suite=suite_name,
            prompt=prompt,
            model=self.client.describe(),
            tags=dict(tags or {}),
        )
        order = {t.id: i for i, t in enumerate(tasks)}
        results: list[TaskResult] = []
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {pool.submit(self.run_task, t, prompt): t for t in tasks}
            for done, fut in enumerate(as_completed(futures), start=1):
                result = fut.result()
                results.append(result)
                if progress:
                    progress(done, len(tasks), result)
        run.results = sorted(results, key=lambda r: order[r.task_id])
        return run

    @staticmethod
    def _errored(task: Task, prompt: str, error: str, output: str | None = None) -> TaskResult:
        return TaskResult(
            task_id=task.id,
            category=task.category,
            prompt=prompt,
            output=output,
            expected=task.expected,
            passed=False,
            score=0.0,
            reason=error,
            error=error,
        )


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return json.loads(json.dumps(value, default=str))
