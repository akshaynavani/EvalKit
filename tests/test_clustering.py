from evalkit.clustering import cluster_failures, failure_document
from evalkit.types import TaskResult


def result(tid, category, reason, output, passed=False):
    return TaskResult(task_id=tid, category=category, prompt="p", output=output, expected=None,
                      passed=passed, score=float(passed), reason=reason)


def test_no_failures_gives_no_clusters():
    assert cluster_failures([result("a", "c", "ok", "x", passed=True)]) == []


def test_small_failure_sets_form_one_cluster():
    clusters = cluster_failures([result("a", "c", "expected 1, got 2", "2")])
    assert len(clusters) == 1 and clusters[0].task_ids == ["a"]


def test_numbers_are_masked_in_documents():
    doc = failure_document(result("a", "math", "expected 12, got 13", "13"))
    assert "12" not in doc and "NUM" in doc


def test_distinct_error_modes_are_separated():
    results = []
    for i in range(6):
        results.append(result(f"m{i}", "math", f"expected {i}, got {i + 1}", f"The answer is {i + 1}"))
    for i in range(6):
        results.append(result(f"j{i}", "extraction", "invalid JSON: Expecting value", "Sure, here is the data you asked for"))
    clusters = cluster_failures(results, max_k=4)
    assert len(clusters) >= 2
    groups = [set(c.task_ids) for c in clusters]
    assert {f"m{i}" for i in range(6)} in groups
    assert {f"j{i}" for i in range(6)} in groups
    assert all(c.representatives for c in clusters)
    assert sum(c.size for c in clusters) == 12
