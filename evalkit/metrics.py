"""Aggregate metrics and run-to-run comparison."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from evalkit.types import TaskResult


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _bucket(results: list[TaskResult]) -> dict[str, Any]:
    n = len(results)
    passed = sum(r.passed for r in results)
    return {
        "n": n,
        "passed": passed,
        "failed": n - passed,
        "errors": sum(r.error is not None for r in results),
        "accuracy": passed / n if n else 0.0,
        "mean_score": sum(r.score for r in results) / n if n else 0.0,
    }


def summarize(results: list[TaskResult]) -> dict[str, Any]:
    by_cat: dict[str, list[TaskResult]] = defaultdict(list)
    for r in results:
        by_cat[r.category].append(r)
    latencies = [r.latency_s for r in results if r.error is None]
    summary = _bucket(results)
    summary["categories"] = {cat: _bucket(rs) for cat, rs in sorted(by_cat.items())}
    summary["latency_s"] = {
        "p50": _percentile(latencies, 0.5),
        "p95": _percentile(latencies, 0.95),
        "mean": sum(latencies) / len(latencies) if latencies else 0.0,
    }
    summary["tokens"] = {
        "input": sum(r.input_tokens or 0 for r in results),
        "output": sum(r.output_tokens or 0 for r in results),
    }
    return summary


def compare(baseline: list[TaskResult], candidate: list[TaskResult]) -> dict[str, Any]:
    """Diff two runs over the same suite: accuracy deltas plus fixed/regressed tasks."""
    base = {r.task_id: r for r in baseline}
    cand = {r.task_id: r for r in candidate}
    shared = sorted(set(base) & set(cand))

    fixed = [t for t in shared if not base[t].passed and cand[t].passed]
    regressed = [t for t in shared if base[t].passed and not cand[t].passed]

    s_base, s_cand = summarize(baseline), summarize(candidate)
    categories = {}
    for cat in sorted(set(s_base["categories"]) | set(s_cand["categories"])):
        a = s_base["categories"].get(cat, {}).get("accuracy", 0.0)
        b = s_cand["categories"].get(cat, {}).get("accuracy", 0.0)
        categories[cat] = {"baseline": a, "candidate": b, "delta": b - a}

    return {
        "baseline_accuracy": s_base["accuracy"],
        "candidate_accuracy": s_cand["accuracy"],
        "delta": s_cand["accuracy"] - s_base["accuracy"],
        "categories": categories,
        "fixed": fixed,
        "regressed": regressed,
        "only_in_baseline": sorted(set(base) - set(cand)),
        "only_in_candidate": sorted(set(cand) - set(base)),
    }
