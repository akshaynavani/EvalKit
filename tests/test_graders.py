import sys

import pytest

from evalkit.graders import available_graders, build_grader, extract_answer
from evalkit.models import MockClient
from evalkit.types import Task


def task(expected, grader="exact", **kw):
    return Task(id="t", category="c", input={"question": "q"}, expected=expected, grader={"type": grader}, **kw)


def test_registry_has_builtins():
    assert {"exact", "normalized", "contains", "regex", "numeric", "json", "code", "llm_judge"} <= set(available_graders())


def test_unknown_grader():
    with pytest.raises(ValueError, match="unknown grader"):
        build_grader({"type": "nope"})


@pytest.mark.parametrize(
    "how,text,expected",
    [
        (None, "  hi  ", "hi"),
        ("last_line", "work\n\nAnswer\n", "Answer"),
        ("last_number", "3 apples and 1,200 pears", "1,200"),
        ("code_block", "x\n```python\nprint(1)\n```", "print(1)"),
        ("regex:answer:\\s*(\\w+)", "Answer: blue", "blue"),
    ],
)
def test_extract_answer(how, text, expected):
    assert extract_answer(text, how) == expected


def test_exact_and_list_of_answers():
    g = build_grader("exact")
    assert g.grade(task(["a", "b"]), "b").passed
    assert not g.grade(task("a"), "A").passed
    assert build_grader({"type": "exact", "case_sensitive": False}).grade(task("a"), "A").passed


def test_normalized_ignores_articles_and_punctuation():
    g = build_grader("normalized")
    assert g.grade(task("Paris"), "the paris.").passed
    res = g.grade(task("New York City"), "New York")
    assert not res.passed and 0 < res.score < 1


def test_contains_modes_and_forbidden():
    t = task(["alpha", "beta"])
    assert build_grader("contains").grade(t, "ALPHA and beta").passed
    assert not build_grader("contains").grade(t, "alpha only").passed
    assert build_grader({"type": "contains", "mode": "any"}).grade(t, "alpha only").passed
    assert not build_grader({"type": "contains", "forbidden": ["sorry"]}).grade(t, "sorry alpha beta").passed


def test_regex_grader():
    assert build_grader({"type": "regex", "pattern": r"\b\d{3}-\d{4}\b"}).grade(task(None), "call 555-1234").passed


def test_numeric_grader():
    g = build_grader({"type": "numeric", "abs_tol": 0.01, "extract": "last_number"})
    assert g.grade(task(42.5), "so it's 42.50").passed
    assert g.grade(task(1250), "$1,250").passed
    res = g.grade(task(10), "answer 11")
    assert not res.passed and "expected 10" in res.reason
    assert not g.grade(task(1), "none").passed


def test_numeric_parses_fractions():
    assert build_grader("numeric").grade(task(0.5), "1/2").passed


def test_json_grader():
    g = build_grader("json")
    exp = {"name": "Tom", "age": 41}
    assert g.grade(task(exp), 'Sure: {"name": "tom", "age": 41.0}').passed
    res = g.grade(task(exp), '{"name": "Tom", "age": "41"}')
    assert not res.passed and res.details["wrong_fields"] == ["age"] and res.score == 0.5
    assert not g.grade(task(exp), "not json").passed
    strict = build_grader({"type": "json", "allow_extra": False})
    assert not strict.grade(task(exp), '{"name": "Tom", "age": 41, "x": 1}').passed


def test_code_grader_pass_and_fail():
    g = build_grader({"type": "code", "timeout_s": 10, "python": sys.executable})
    tests = ["assert add(1, 2) == 3", {"call": "add(2, 2)", "expected": 4}]
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    bad = "```python\ndef add(a, b):\n    return a - b\n```"
    assert g.grade(task(tests), good).passed
    res = g.grade(task(tests), bad)
    assert not res.passed and res.score == 0.0


def test_code_grader_syntax_error_and_timeout():
    tests = ["assert f() == 1"]
    assert "failed to load" in build_grader("code").grade(task(tests), "```python\ndef f(:\n```").reason
    slow = build_grader({"type": "code", "timeout_s": 1}).grade(task(tests), "```python\nwhile True: pass\n```")
    assert not slow.passed and "timeout" in slow.reason


def test_llm_judge_uses_judge_client():
    judge = MockClient(default='{"score": 0.9, "passed": true, "reason": "same answer"}')
    g = build_grader("llm_judge", judge_client=judge)
    res = g.grade(task("Paris"), "It's Paris")
    assert res.passed and res.score == 0.9
    assert "Paris" in judge.calls[0][1]


def test_llm_judge_unparseable_and_missing_client():
    g = build_grader("llm_judge", judge_client=MockClient(default="looks fine"))
    assert not g.grade(task("x"), "y").passed
    with pytest.raises(ValueError):
        build_grader("llm_judge")
