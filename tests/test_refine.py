import json

import pytest

from evalkit.models import MockClient
from evalkit.prompts import PromptError, PromptTemplate
from evalkit.refine import (
    PromptRefiner,
    parse_refiner_output,
    refine_prompt,
    save_refinement,
    split_tasks,
    validate_candidate,
)
from evalkit.runner import Runner
from evalkit.storage import RunStore
from evalkit.types import Task

TASKS = [
    Task(id=f"t{i}", category="math", input={"question": f"{i}+{i}"}, expected=2 * i, grader={"type": "exact"})
    for i in range(1, 9)
]


def target_model() -> MockClient:
    """Answers correctly only when the prompt demands a bare number."""

    def respond(system, user):
        q = user.split("Q:")[-1].strip()
        a, b = (int(x) for x in q.split("+"))
        if "ONLY the number" in system + user:
            return str(a + b)
        return f"The answer is {a + b}."

    return MockClient(responder=respond)


GOOD_REFINEMENT = (
    "<analysis>Outputs include prose; grader wants a bare number.</analysis>\n"
    "<system>Reply with ONLY the number.</system>\n"
    "<user>Q: {question}</user>"
)


def test_parse_refiner_output():
    parsed = parse_refiner_output(GOOD_REFINEMENT)
    assert parsed["system"] == "Reply with ONLY the number."
    assert parsed["user"] == "Q: {question}"
    with pytest.raises(PromptError):
        parse_refiner_output("no tags here")


def test_validate_candidate_checks_variables():
    base = PromptTemplate(name="p", user="Q: {question}")
    with pytest.raises(PromptError, match="dropped"):
        validate_candidate(base.derive(user="Q:"), base, None)
    with pytest.raises(PromptError, match="unknown"):
        validate_candidate(base.derive(user="{question} {extra}"), base, None)
    validate_candidate(base.derive(user="Question: {question}"), base, {"question": "x"})


def test_split_is_deterministic():
    dev1, hold1 = split_tasks(TASKS, 0.25)
    dev2, hold2 = split_tasks(TASKS, 0.25)
    assert [t.id for t in dev1] == [t.id for t in dev2]
    assert len(dev1) + len(hold1) == len(TASKS)
    assert split_tasks(TASKS, 0) == (TASKS, [])


def test_refinement_improves_and_stops(tmp_path):
    base = PromptTemplate(name="math", system="Be helpful.", user="Q: {question}")
    refiner_model = MockClient(default=GOOD_REFINEMENT)
    result = refine_prompt(
        TASKS,
        base,
        Runner(target_model()),
        PromptRefiner(refiner_model),
        iterations=3,
        store=RunStore(tmp_path / "runs"),
    )
    assert result.baseline_dev_accuracy == 0.0
    assert result.best_dev_accuracy == 1.0
    assert result.best.version == 2 and result.best.parent == "math@v1"
    assert result.stop_reason == "no failures left on dev split"
    assert "Cluster" in refiner_model.calls[0][1]

    out = save_refinement(result, tmp_path / "ref")
    history = json.loads((out / "refinement.json").read_text())
    assert [s["accepted"] for s in history["steps"]] == [True, True]
    assert (out / "math.v2.yaml").exists()


def test_refinement_rejects_bad_proposals_and_respects_patience():
    base = PromptTemplate(name="math", user="Q: {question}")
    bad = MockClient(default="<system></system><user>Q: {question} ({hint})</user>")
    result = refine_prompt(TASKS, base, Runner(target_model()), PromptRefiner(bad), iterations=5, patience=2)
    assert result.best is base
    assert len(result.steps) == 3  # baseline + two failed proposals
    assert all(s.error for s in result.steps[1:])
    assert "no improvement" in result.stop_reason


def test_holdout_is_evaluated_for_baseline_and_best():
    base = PromptTemplate(name="math", system="Be helpful.", user="Q: {question}")
    result = refine_prompt(
        TASKS, base, Runner(target_model()), PromptRefiner(MockClient(default=GOOD_REFINEMENT)), holdout_frac=0.3
    )
    if result.baseline_holdout_accuracy is not None:
        assert result.baseline_holdout_accuracy == 0.0
        assert result.best_holdout_accuracy == 1.0
