"""Executes generated Python code against test cases in an isolated subprocess.

This is *process* isolation (separate interpreter, temp dir, timeout, stripped env), not a
security sandbox. Only run untrusted model code inside a container or VM.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from evalkit.graders.base import Grader, register
from evalkit.types import GradeResult, Task

_HARNESS = r'''
import json, sys, traceback
results = []
ns = {}
try:
    with open("candidate.py", encoding="utf-8") as fh:
        exec(compile(fh.read(), "candidate.py", "exec"), ns)
except BaseException as exc:
    print("__EVALKIT__" + json.dumps({"load_error": f"{type(exc).__name__}: {exc}"}))
    sys.exit(0)
with open("tests.json", encoding="utf-8") as fh:
    tests = json.load(fh)
for test in tests:
    try:
        if "assert" in test:
            exec(test["assert"], ns)
            results.append({"name": test["name"], "passed": True})
        else:
            got = eval(test["call"], ns)
            ok = got == test["expected"]
            results.append({"name": test["name"], "passed": ok,
                            "got": repr(got)[:200], "expected": repr(test["expected"])[:200]})
    except BaseException as exc:
        results.append({"name": test["name"], "passed": False,
                        "error": f"{type(exc).__name__}: {exc}"[:300]})
print("__EVALKIT__" + json.dumps({"results": results}))
'''


def _normalize_tests(raw: Any) -> list[dict[str, Any]]:
    """Accept assert strings or ``{"call": ..., "expected": ...}`` dicts."""
    if raw is None:
        return []
    if isinstance(raw, (str, dict)):
        raw = [raw]
    tests = []
    for i, item in enumerate(raw):
        if isinstance(item, str):
            tests.append({"name": f"test_{i}", "assert": item})
        elif isinstance(item, dict) and ("call" in item or "assert" in item):
            tests.append({"name": item.get("name", f"test_{i}"), **item})
        else:
            raise ValueError(f"invalid code test case: {item!r}")
    return tests


@register("code")
class CodeExecutionGrader(Grader):
    """Grades code by running ``tests`` (grader option) or ``task.expected``.

    Defaults to extracting the last fenced code block from the output.
    """

    def __init__(self, tests: Any = None, timeout_s: float = 10.0, python: str | None = None, extract: str | None = "code_block", **kw: Any) -> None:
        super().__init__(extract=extract, **kw)
        self.tests = tests
        self.timeout_s = timeout_s
        self.python = python or sys.executable

    def score(self, task: Task, answer: str) -> GradeResult:
        tests = _normalize_tests(self.tests if self.tests is not None else task.expected)
        if not tests:
            return GradeResult.fail("no test cases configured for code grader")

        with tempfile.TemporaryDirectory(prefix="evalkit_code_") as tmp:
            workdir = Path(tmp)
            (workdir / "candidate.py").write_text(answer, encoding="utf-8")
            (workdir / "tests.json").write_text(json.dumps(tests), encoding="utf-8")
            (workdir / "harness.py").write_text(_HARNESS, encoding="utf-8")
            env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"}
            try:
                proc = subprocess.run(
                    [self.python, "-I", "harness.py"],
                    cwd=workdir,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                )
            except subprocess.TimeoutExpired:
                return GradeResult.fail(f"timeout after {self.timeout_s}s")

        payload = None
        for line in proc.stdout.splitlines():
            if line.startswith("__EVALKIT__"):
                payload = json.loads(line[len("__EVALKIT__"):])
        if payload is None:
            return GradeResult.fail(f"harness crashed: {proc.stderr.strip()[-300:]}")
        if "load_error" in payload:
            return GradeResult.fail(f"code failed to load: {payload['load_error']}")

        results = payload["results"]
        passed = sum(r["passed"] for r in results)
        frac = passed / len(results)
        if passed == len(results):
            return GradeResult.ok(f"all {passed} tests passed", tests=results)
        first = next(r for r in results if not r["passed"])
        why = first.get("error") or f"got {first.get('got')} expected {first.get('expected')}"
        return GradeResult.fail(f"{passed}/{len(results)} tests passed; {first['name']}: {why}", score=frac, tests=results)
