"""Persist runs to disk as ``runs/<run_id>/{config.json,results.jsonl,summary.json}``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evalkit.metrics import summarize
from evalkit.prompts import PromptTemplate
from evalkit.runner import EvalRun
from evalkit.types import TaskResult


class RunStore:
    def __init__(self, root: str | Path = "runs") -> None:
        self.root = Path(root)

    def path_for(self, run_id: str) -> Path:
        return self.root / run_id

    def save(self, run: EvalRun, extra: dict[str, Any] | None = None) -> Path:
        out = self.path_for(run.run_id)
        out.mkdir(parents=True, exist_ok=True)
        config = {
            "run_id": run.run_id,
            "created_at": run.created_at,
            "suite": run.suite,
            "model": run.model,
            "prompt": run.prompt.to_dict(),
            "prompt_fingerprint": run.prompt.fingerprint,
            "tags": run.tags,
        }
        (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        with (out / "results.jsonl").open("w", encoding="utf-8") as fh:
            for r in run.results:
                fh.write(json.dumps(r.to_dict(), ensure_ascii=False, default=str) + "\n")
        summary = summarize(run.results)
        if extra:
            summary.update(extra)
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return out

    def resolve(self, ref: str | Path) -> Path:
        """Accept a run id, a run directory path, or ``latest``."""
        if str(ref) == "latest":
            runs = self.list_runs()
            if not runs:
                raise FileNotFoundError(f"no runs found under {self.root}")
            return self.path_for(runs[-1]["run_id"])
        path = Path(ref)
        if path.is_dir():
            return path
        candidate = self.path_for(str(ref))
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(f"run not found: {ref}")

    def load(self, ref: str | Path) -> EvalRun:
        path = self.resolve(ref)
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        results = [
            TaskResult.from_dict(json.loads(line))
            for line in (path / "results.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return EvalRun(
            run_id=config["run_id"],
            created_at=config["created_at"],
            suite=config["suite"],
            prompt=PromptTemplate.from_dict(config["prompt"]),
            model=config["model"],
            results=results,
            tags=config.get("tags", {}),
        )

    def list_runs(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        rows = []
        for cfg_path in self.root.glob("*/config.json"):
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            summary_path = cfg_path.parent / "summary.json"
            acc = json.loads(summary_path.read_text())["accuracy"] if summary_path.exists() else None
            rows.append({
                "run_id": cfg["run_id"],
                "created_at": cfg["created_at"],
                "suite": cfg["suite"],
                "model": f"{cfg['model'].get('provider')}:{cfg['model'].get('model')}",
                "prompt": f"{cfg['prompt']['name']}@v{cfg['prompt'].get('version', 1)}",
                "accuracy": acc,
            })
        return sorted(rows, key=lambda r: (r["created_at"], r["run_id"]))
