from evalkit.metrics import compare, summarize
from evalkit.models import MockClient
from evalkit.models.base import ModelError, RetryableModelError
from evalkit.prompts import PromptTemplate
from evalkit.runner import Runner
from evalkit.storage import RunStore
from evalkit.types import ModelResponse, Task


def math_client(wrong: set[str] = frozenset()) -> MockClient:
    answers = {"2+2": "4", "3*3": "9", "capital of France": "Paris"}

    def respond(system, user):
        q = user.removeprefix("Q: ")
        return "nope" if q in wrong else answers[q]

    return MockClient(responder=respond)


def test_runner_grades_and_preserves_order(arithmetic_tasks, qa_prompt):
    run = Runner(math_client({"3*3"}), concurrency=3).run(arithmetic_tasks, qa_prompt, suite_name="demo")
    assert [r.task_id for r in run.results] == ["a1", "a2", "s1"]
    assert [r.passed for r in run.results] == [True, False, True]
    assert run.accuracy == 2 / 3
    assert run.results[0].prompt == "Q: 2+2"
    assert run.model["provider"] == "mock"


def test_runner_captures_model_and_prompt_errors(qa_prompt):
    class Broken(MockClient):
        def _complete(self, system, user):
            raise ModelError("boom")

    tasks = [Task(id="x", category="c", input={"question": "q"}, expected="a")]
    run = Runner(Broken()).run(tasks, qa_prompt)
    assert run.results[0].error and "boom" in run.results[0].error

    bad_tasks = [Task(id="y", category="c", input={"other": "q"}, expected="a")]
    run = Runner(MockClient()).run(bad_tasks, qa_prompt)
    assert "prompt error" in run.results[0].error


def test_retry_then_success():
    class Flaky(MockClient):
        attempts = 0

        def _complete(self, system, user):
            Flaky.attempts += 1
            if Flaky.attempts < 3:
                raise RetryableModelError("429")
            return ModelResponse(text="ok", model="m")

    assert Flaky().complete("", "hi", retries=3, backoff_s=0).text == "ok"
    assert Flaky.attempts == 3


def test_summarize_by_category(arithmetic_tasks, qa_prompt):
    run = Runner(math_client({"3*3"})).run(arithmetic_tasks, qa_prompt)
    s = summarize(run.results)
    assert s["n"] == 3 and s["passed"] == 2
    assert s["categories"]["math"]["accuracy"] == 0.5
    assert s["categories"]["text"]["accuracy"] == 1.0


def test_compare_runs(arithmetic_tasks, qa_prompt):
    a = Runner(math_client({"3*3"})).run(arithmetic_tasks, qa_prompt)
    b = Runner(math_client({"2+2"})).run(arithmetic_tasks, qa_prompt)
    diff = compare(a.results, b.results)
    assert diff["fixed"] == ["a2"] and diff["regressed"] == ["a1"]
    assert diff["delta"] == 0


def test_store_roundtrip(tmp_path, arithmetic_tasks, qa_prompt):
    store = RunStore(tmp_path / "runs")
    run = Runner(math_client()).run(arithmetic_tasks, qa_prompt, suite_name="demo")
    path = store.save(run)
    assert (path / "results.jsonl").exists() and (path / "summary.json").exists()
    loaded = store.load(run.run_id)
    assert loaded.accuracy == 1.0
    assert loaded.prompt.fingerprint == qa_prompt.fingerprint
    assert store.resolve("latest") == path
    assert store.list_runs()[0]["accuracy"] == 1.0


def test_prompt_template_type_is_reused():
    p = PromptTemplate(name="x", user="{question}")
    assert p.label == "x@v1"
