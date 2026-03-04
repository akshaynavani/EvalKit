"""Render run summaries and comparisons as Markdown."""

from __future__ import annotations

from typing import Any

from evalkit.metrics import summarize
from evalkit.runner import EvalRun


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _signed_pct(x: float) -> str:
    return f"{'+' if x >= 0 else ''}{x * 100:.1f}pp"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _short(text: Any, n: int = 80) -> str:
    s = " ".join(str(text).split())
    s = s.replace("|", "\\|")
    return s if len(s) <= n else s[: n - 1] + "…"


def render_run(run: EvalRun, max_failures: int = 15) -> str:
    s = summarize(run.results)
    out = [
        f"# EvalKit run `{run.run_id}`",
        "",
        f"- **Suite:** {run.suite}",
        f"- **Model:** {run.model.get('provider')}:{run.model.get('model')}",
        f"- **Prompt:** {run.prompt.label} (`{run.prompt.fingerprint}`)",
        f"- **Accuracy:** {_pct(s['accuracy'])} ({s['passed']}/{s['n']}), "
        f"mean score {s['mean_score']:.3f}, errors {s['errors']}",
        f"- **Latency:** p50 {s['latency_s']['p50']:.2f}s, p95 {s['latency_s']['p95']:.2f}s",
        f"- **Tokens:** {s['tokens']['input']} in / {s['tokens']['output']} out",
        "",
        "## By category",
        "",
        _table(
            ["Category", "N", "Passed", "Accuracy", "Mean score"],
            [
                [cat, b["n"], b["passed"], _pct(b["accuracy"]), f"{b['mean_score']:.3f}"]
                for cat, b in s["categories"].items()
            ],
        ),
    ]
    failures = run.failures()
    if failures:
        out += ["", f"## Failures ({len(failures)})", ""]
        out.append(
            _table(
                ["Task", "Category", "Reason", "Output"],
                [[f.task_id, f.category, _short(f.reason), _short(f.output or "", 60)] for f in failures[:max_failures]],
            )
        )
        if len(failures) > max_failures:
            out.append(f"\n…and {len(failures) - max_failures} more.")
    return "\n".join(out) + "\n"


def render_comparison(baseline: EvalRun, candidate: EvalRun, diff: dict[str, Any]) -> str:
    out = [
        f"# Comparison: `{baseline.run_id}` → `{candidate.run_id}`",
        "",
        f"- **Baseline:** {baseline.prompt.label} on {baseline.model.get('model')} — {_pct(diff['baseline_accuracy'])}",
        f"- **Candidate:** {candidate.prompt.label} on {candidate.model.get('model')} — {_pct(diff['candidate_accuracy'])}",
        f"- **Delta:** {_signed_pct(diff['delta'])}",
        "",
        _table(
            ["Category", "Baseline", "Candidate", "Delta"],
            [[c, _pct(v["baseline"]), _pct(v["candidate"]), _signed_pct(v["delta"])] for c, v in diff["categories"].items()],
        ),
        "",
        f"**Fixed ({len(diff['fixed'])}):** {', '.join(diff['fixed']) or '—'}",
        "",
        f"**Regressed ({len(diff['regressed'])}):** {', '.join(diff['regressed']) or '—'}",
    ]
    return "\n".join(out) + "\n"
