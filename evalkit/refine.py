"""Iterative prompt refinement driven by clustered failures.

Loop:
    1. Evaluate the current best prompt on the dev split.
    2. Cluster its failures and show representative examples to a *refiner* model.
    3. The refiner proposes a revised template; it is validated and evaluated.
    4. Keep the candidate if it beats the best dev accuracy by ``min_delta``.
    5. Stop after ``iterations`` rounds, ``patience`` rounds without improvement,
       or when the dev split has no failures left.

An optional holdout split is evaluated for the baseline and final best prompt only, so
reported gains are not just overfitting to the examples the refiner saw.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from evalkit.clustering import FailureCluster, cluster_failures
from evalkit.models.base import ModelClient
from evalkit.prompts import PromptError, PromptTemplate
from evalkit.runner import EvalRun, Runner
from evalkit.storage import RunStore
from evalkit.types import Task

REFINER_SYSTEM = """You are an expert prompt engineer improving a prompt template used to \
evaluate a language model. You will see the current template, its accuracy and clusters \
of failures with representative examples. Diagnose the systematic causes and rewrite the \
template to fix them without overfitting to the specific examples.

Rules:
- Keep every template variable in {braces} that the current template uses; do not invent new ones.
- Literal braces in the template must be doubled ({{ and }}).
- Be concrete about output format when failures are formatting-related.
- Do not include the example answers in the template.

Respond in exactly this format:
<analysis>short diagnosis</analysis>
<system>new system prompt (may be empty)</system>
<user>new user prompt template</user>"""


@dataclass
class RefinementStep:
    iteration: int
    prompt_label: str
    fingerprint: str
    run_id: str
    dev_accuracy: float
    accepted: bool
    analysis: str = ""
    n_clusters: int = 0
    error: str | None = None


@dataclass
class RefinementResult:
    baseline: PromptTemplate
    best: PromptTemplate
    baseline_dev_accuracy: float
    best_dev_accuracy: float
    baseline_holdout_accuracy: float | None = None
    best_holdout_accuracy: float | None = None
    steps: list[RefinementStep] = field(default_factory=list)
    stop_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.to_dict(),
            "best": self.best.to_dict(),
            "baseline_dev_accuracy": self.baseline_dev_accuracy,
            "best_dev_accuracy": self.best_dev_accuracy,
            "baseline_holdout_accuracy": self.baseline_holdout_accuracy,
            "best_holdout_accuracy": self.best_holdout_accuracy,
            "steps": [asdict(s) for s in self.steps],
            "stop_reason": self.stop_reason,
        }


def split_tasks(tasks: list[Task], holdout_frac: float, seed: str = "evalkit") -> tuple[list[Task], list[Task]]:
    """Deterministic, stratification-free split by hashing task ids."""
    if holdout_frac <= 0:
        return list(tasks), []
    dev, holdout = [], []
    for t in tasks:
        h = int(hashlib.sha256(f"{seed}:{t.id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        (holdout if h < holdout_frac else dev).append(t)
    if not dev:  # tiny suites: keep at least one dev task
        dev, holdout = holdout[:1], holdout[1:]
    return dev, holdout


def build_refiner_prompt(prompt: PromptTemplate, run: EvalRun, clusters: list[FailureCluster], examples_per_cluster: int = 2) -> str:
    total = len(run.results)
    lines = [
        "## Current template",
        "<system>",
        prompt.system,
        "</system>",
        "<user>",
        prompt.user,
        "</user>",
        "",
        f"Template variables: {', '.join(sorted(prompt.variables())) or '(none)'}",
        f"Accuracy: {run.accuracy * 100:.1f}% ({total - len(run.failures())}/{total})",
        "",
        "## Failure clusters",
    ]
    for c in clusters:
        lines.append(f"\n### Cluster {c.cluster_id} — {c.size} failures — {c.label}")
        lines.append(f"Categories: {json.dumps(c.categories)}")
        for ex in c.representatives[:examples_per_cluster]:
            lines += [
                f"- Task `{ex['task_id']}`",
                f"  Rendered prompt: {ex['prompt'][:300]!r}",
                f"  Model output: {ex['output'][:300]!r}",
                f"  Expected: {str(ex['expected'])[:200]!r}",
                f"  Grader: {ex['reason'][:200]}",
            ]
    return "\n".join(lines)


_TAG_RE = {tag: re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.DOTALL) for tag in ("analysis", "system", "user")}


def parse_refiner_output(text: str) -> dict[str, str]:
    out = {}
    for tag, rx in _TAG_RE.items():
        m = rx.search(text)
        if m:
            out[tag] = m.group(1)
    if "user" not in out:
        raise PromptError("refiner response missing <user> block")
    return out


def validate_candidate(candidate: PromptTemplate, original: PromptTemplate, sample: dict[str, Any] | None) -> None:
    missing = original.variables() - candidate.variables()
    extra = candidate.variables() - original.variables()
    if missing:
        raise PromptError(f"refined prompt dropped variables: {sorted(missing)}")
    if extra:
        raise PromptError(f"refined prompt introduced unknown variables: {sorted(extra)}")
    if sample is not None:
        candidate.render(sample)  # raises PromptError on malformed braces


class PromptRefiner:
    def __init__(self, refiner: ModelClient, examples_per_cluster: int = 2) -> None:
        self.refiner = refiner
        self.examples_per_cluster = examples_per_cluster

    def propose(self, prompt: PromptTemplate, run: EvalRun, clusters: list[FailureCluster], sample_input: dict[str, Any] | None = None) -> tuple[PromptTemplate, str]:
        response = self.refiner.complete(
            REFINER_SYSTEM, build_refiner_prompt(prompt, run, clusters, self.examples_per_cluster)
        )
        parsed = parse_refiner_output(response.text)
        analysis = parsed.get("analysis", "")
        candidate = prompt.derive(system=parsed.get("system", prompt.system), user=parsed["user"], notes=analysis[:500])
        try:
            validate_candidate(candidate, prompt, sample_input)
        except (ValueError, IndexError, KeyError) as exc:
            raise PromptError(f"invalid refined prompt: {exc}") from exc
        return candidate, analysis


def refine_prompt(
    tasks: list[Task],
    prompt: PromptTemplate,
    runner: Runner,
    refiner: PromptRefiner,
    *,
    iterations: int = 3,
    patience: int = 2,
    min_delta: float = 0.0,
    holdout_frac: float = 0.0,
    store: RunStore | None = None,
    suite_name: str = "suite",
    log: Callable[[str], None] = lambda _msg: None,
) -> RefinementResult:
    dev, holdout = split_tasks(tasks, holdout_frac)
    sample_input = dev[0].input if dev else None

    def evaluate(p: PromptTemplate, split: list[Task], tag: str) -> EvalRun:
        run = runner.run(split, p, suite_name=suite_name, tags={"refine": tag, "prompt": p.label})
        if store is not None:
            store.save(run)
        return run

    base_run = evaluate(prompt, dev, "baseline-dev")
    log(f"baseline {prompt.label}: dev accuracy {base_run.accuracy:.3f}")
    result = RefinementResult(
        baseline=prompt,
        best=prompt,
        baseline_dev_accuracy=base_run.accuracy,
        best_dev_accuracy=base_run.accuracy,
    )
    result.steps.append(RefinementStep(0, prompt.label, prompt.fingerprint, base_run.run_id, base_run.accuracy, True))

    best_run, stale = base_run, 0
    seen = {prompt.fingerprint}
    version = prompt.version
    result.stop_reason = "max iterations reached"
    for i in range(1, iterations + 1):
        if not best_run.failures():
            result.stop_reason = "no failures left on dev split"
            break
        clusters = cluster_failures(best_run.results)
        try:
            candidate, analysis = refiner.propose(result.best, best_run, clusters, sample_input)
        except Exception as exc:  # refiner errors count as a non-improving round
            log(f"iteration {i}: refiner failed: {exc}")
            result.steps.append(RefinementStep(i, "-", "-", "-", 0.0, False, error=str(exc), n_clusters=len(clusters)))
            stale += 1
            if stale >= patience:
                result.stop_reason = f"no improvement for {patience} rounds"
                break
            continue

        version += 1
        candidate.version = version  # every proposal gets its own version, accepted or not
        if candidate.fingerprint in seen:
            log(f"iteration {i}: refiner repeated an earlier prompt")
        seen.add(candidate.fingerprint)

        run = evaluate(candidate, dev, f"iter-{i}")
        accepted = run.accuracy > result.best_dev_accuracy + min_delta
        log(f"iteration {i} {candidate.label}: dev accuracy {run.accuracy:.3f} ({'accepted' if accepted else 'rejected'})")
        result.steps.append(
            RefinementStep(i, candidate.label, candidate.fingerprint, run.run_id, run.accuracy, accepted, analysis, len(clusters))
        )
        if accepted:
            result.best, result.best_dev_accuracy, best_run, stale = candidate, run.accuracy, run, 0
        else:
            stale += 1
            if stale >= patience:
                result.stop_reason = f"no improvement for {patience} rounds"
                break

    if holdout:
        result.baseline_holdout_accuracy = evaluate(prompt, holdout, "baseline-holdout").accuracy
        if result.best is prompt:
            result.best_holdout_accuracy = result.baseline_holdout_accuracy
        else:
            result.best_holdout_accuracy = evaluate(result.best, holdout, "best-holdout").accuracy
        log(f"holdout: {result.baseline_holdout_accuracy:.3f} -> {result.best_holdout_accuracy:.3f}")
    return result


def save_refinement(result: RefinementResult, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "refinement.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    result.best.save(out / f"{result.best.name}.v{result.best.version}.yaml")
    return out
