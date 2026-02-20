"""Loading evaluation suites from YAML or JSONL files."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml

from evalkit.types import Task


class SuiteError(ValueError):
    """Raised when a suite file is malformed."""


def _task_from_dict(raw: dict[str, Any], defaults: dict[str, Any], source: str) -> Task:
    if not isinstance(raw, dict):
        raise SuiteError(f"{source}: each task must be a mapping, got {type(raw).__name__}")
    if "id" not in raw:
        raise SuiteError(f"{source}: task missing required field 'id'")

    task_input = raw.get("input")
    if task_input is None:
        raise SuiteError(f"{source}: task {raw['id']!r} missing 'input'")
    if not isinstance(task_input, dict):
        # Allow shorthand ``input: "question text"`` -> {"question": ...}
        task_input = {"question": task_input}

    grader = raw.get("grader", defaults.get("grader", {"type": "exact"}))
    if isinstance(grader, str):
        grader = {"type": grader}

    return Task(
        id=str(raw["id"]),
        category=str(raw.get("category", defaults.get("category", "default"))),
        input=task_input,
        expected=raw.get("expected"),
        grader=dict(grader),
        metadata=dict(raw.get("metadata", {})),
    )


def _load_yaml(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data, {}
    if isinstance(data, dict) and "tasks" in data:
        return list(data["tasks"]), dict(data.get("defaults", {}))
    raise SuiteError(f"{path}: expected a list of tasks or a mapping with 'tasks'")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SuiteError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc
    return rows


def load_suite(path: str | Path) -> list[Task]:
    """Load a suite file (``.yaml``/``.yml``/``.jsonl``/``.json``) or a directory of them."""
    path = Path(path)
    if path.is_dir():
        tasks: list[Task] = []
        for child in sorted(path.iterdir()):
            if child.suffix in {".yaml", ".yml", ".jsonl", ".json"}:
                tasks.extend(load_suite(child))
        _check_unique(tasks, str(path))
        return tasks

    if not path.exists():
        raise SuiteError(f"suite not found: {path}")

    defaults: dict[str, Any] = {}
    if path.suffix in {".yaml", ".yml"}:
        rows, defaults = _load_yaml(path)
    elif path.suffix == ".jsonl":
        rows = _load_jsonl(path)
    elif path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data["tasks"] if isinstance(data, dict) else data
        defaults = data.get("defaults", {}) if isinstance(data, dict) else {}
    else:
        raise SuiteError(f"unsupported suite format: {path.suffix}")

    tasks = [_task_from_dict(r, defaults, str(path)) for r in rows]
    _check_unique(tasks, str(path))
    return tasks


def _check_unique(tasks: Iterable[Task], source: str) -> None:
    dupes = [tid for tid, n in Counter(t.id for t in tasks).items() if n > 1]
    if dupes:
        raise SuiteError(f"{source}: duplicate task ids: {', '.join(sorted(dupes))}")


def filter_tasks(
    tasks: list[Task], categories: list[str] | None = None, limit: int | None = None
) -> list[Task]:
    selected = [t for t in tasks if not categories or t.category in categories]
    return selected[:limit] if limit else selected
