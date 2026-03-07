import json

import pytest

from evalkit.prompts import PromptError, PromptTemplate, load_prompt
from evalkit.tasks import SuiteError, filter_tasks, load_suite


def test_load_yaml_with_defaults(tmp_path):
    suite = tmp_path / "s.yaml"
    suite.write_text(
        "defaults:\n  category: math\n  grader: numeric\n"
        "tasks:\n  - id: t1\n    input: {question: '1+1'}\n    expected: 2\n"
        "  - id: t2\n    category: other\n    input: shorthand\n    expected: x\n    grader: {type: exact}\n"
    )
    tasks = load_suite(suite)
    assert [t.id for t in tasks] == ["t1", "t2"]
    assert tasks[0].category == "math" and tasks[0].grader == {"type": "numeric"}
    assert tasks[1].input == {"question": "shorthand"}
    assert tasks[1].category == "other"


def test_load_jsonl_and_directory(tmp_path):
    (tmp_path / "a.jsonl").write_text(json.dumps({"id": "j1", "input": {"q": 1}}) + "\n\n")
    (tmp_path / "b.yaml").write_text("- id: y1\n  input: {q: 2}\n")
    tasks = load_suite(tmp_path)
    assert {t.id for t in tasks} == {"j1", "y1"}


def test_duplicate_ids_rejected(tmp_path):
    (tmp_path / "a.jsonl").write_text('{"id": "x", "input": {}}\n{"id": "x", "input": {}}\n')
    with pytest.raises(SuiteError, match="duplicate"):
        load_suite(tmp_path / "a.jsonl")


def test_missing_input_rejected(tmp_path):
    (tmp_path / "a.jsonl").write_text('{"id": "x"}\n')
    with pytest.raises(SuiteError, match="input"):
        load_suite(tmp_path / "a.jsonl")


def test_filter_tasks(arithmetic_tasks):
    assert [t.id for t in filter_tasks(arithmetic_tasks, ["math"])] == ["a1", "a2"]
    assert len(filter_tasks(arithmetic_tasks, limit=1)) == 1


def test_prompt_render_and_variables(qa_prompt):
    assert qa_prompt.variables() == {"question"}
    assert qa_prompt.render({"question": "why?"}) == ("Answer briefly.", "Q: why?")
    with pytest.raises(PromptError):
        qa_prompt.render({})


def test_prompt_derive_and_roundtrip(tmp_path, qa_prompt):
    child = qa_prompt.derive(user="Question: {question}\nAnswer:")
    assert child.version == 2 and child.parent == "qa@v1"
    assert child.fingerprint != qa_prompt.fingerprint
    child.save(tmp_path / "p.yaml")
    loaded = load_prompt(tmp_path / "p.yaml")
    assert loaded.user == child.user and loaded.version == 2


def test_literal_braces_are_preserved():
    p = PromptTemplate(name="j", user='Return JSON like {{"a": 1}} for {question}')
    assert p.variables() == {"question"}
    assert p.render({"question": "x"})[1] == 'Return JSON like {"a": 1} for x'
