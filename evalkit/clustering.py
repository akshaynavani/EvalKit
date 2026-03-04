"""Group failed tasks into clusters of similar error modes.

Each failure is turned into a short document combining the grader's reason, the task
category and a slice of the model output. Numbers and quoted literals are masked so that
"expected 12, got 13" and "expected 7, got 9" land together. Documents are embedded with
TF-IDF and clustered with k-means, choosing ``k`` by silhouette score.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from evalkit.types import TaskResult

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_QUOTED_RE = re.compile(r"'[^']*'|\"[^\"]*\"")


@dataclass
class FailureCluster:
    cluster_id: int
    label: str
    size: int
    top_terms: list[str]
    task_ids: list[str]
    categories: dict[str, int]
    representatives: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def failure_document(result: TaskResult, output_chars: int = 200) -> str:
    reason = _QUOTED_RE.sub(" QUOTED ", result.reason or "")
    reason = _NUM_RE.sub(" NUM ", reason)
    output = _NUM_RE.sub(" NUM ", (result.output or "")[:output_chars])
    kind = "error" if result.error else "wrong"
    return f"{kind} category_{result.category} {reason} {reason} {output}"


def _representatives(results: list[TaskResult], idx: list[int], dist: np.ndarray, k: int) -> list[dict[str, Any]]:
    ordered = [idx[i] for i in np.argsort(dist)[:k]]
    return [
        {
            "task_id": results[i].task_id,
            "category": results[i].category,
            "reason": results[i].reason,
            "output": (results[i].output or "")[:400],
            "expected": results[i].expected,
            "prompt": results[i].prompt[:600],
        }
        for i in ordered
    ]


def _single_cluster(failures: list[TaskResult], n_examples: int) -> list[FailureCluster]:
    reasons = Counter(_NUM_RE.sub("N", f.reason.split(":")[0]) for f in failures)
    label = reasons.most_common(1)[0][0] if reasons else "failures"
    return [
        FailureCluster(
            cluster_id=0,
            label=label,
            size=len(failures),
            top_terms=[],
            task_ids=[f.task_id for f in failures],
            categories=dict(Counter(f.category for f in failures)),
            representatives=_representatives(failures, list(range(len(failures))), np.zeros(len(failures)), n_examples),
        )
    ]


def cluster_failures(
    results: list[TaskResult],
    *,
    max_k: int = 8,
    n_terms: int = 5,
    n_examples: int = 3,
    random_state: int = 0,
) -> list[FailureCluster]:
    """Cluster failed results; returns clusters sorted by size (largest first)."""
    failures = [r for r in results if not r.passed]
    if not failures:
        return []
    if len(failures) < 3:
        return _single_cluster(failures, n_examples)

    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import silhouette_score

    docs = [failure_document(f) for f in failures]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, token_pattern=r"(?u)\b[a-zA-Z_][a-zA-Z_]+\b")
    try:
        X = vectorizer.fit_transform(docs)
    except ValueError:  # empty vocabulary
        return _single_cluster(failures, n_examples)
    terms = np.array(vectorizer.get_feature_names_out())

    n_unique = len({d for d in docs})
    best_k, best_score, best_model = 1, -1.0, None
    for k in range(2, min(max_k, len(failures) - 1, n_unique) + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X)
        if len(set(model.labels_)) < 2:
            continue
        score = silhouette_score(X, model.labels_, metric="cosine")
        if score > best_score:
            best_k, best_score, best_model = k, score, model
    if best_model is None or best_score < 0.05:
        return _single_cluster(failures, n_examples)

    dense = X.toarray()
    clusters = []
    for cid in range(best_k):
        idx = [i for i, lab in enumerate(best_model.labels_) if lab == cid]
        if not idx:
            continue
        centroid = best_model.cluster_centers_[cid]
        top = [t for t in terms[np.argsort(centroid)[::-1]] if not t.startswith("category_")][:n_terms]
        dist = np.linalg.norm(dense[idx] - centroid, axis=1)
        members = [failures[i] for i in idx]
        clusters.append(
            FailureCluster(
                cluster_id=cid,
                label=", ".join(top[:3]) or f"cluster {cid}",
                size=len(idx),
                top_terms=[str(t) for t in top],
                task_ids=[m.task_id for m in members],
                categories=dict(Counter(m.category for m in members)),
                representatives=_representatives(failures, idx, dist, n_examples),
            )
        )
    clusters.sort(key=lambda c: c.size, reverse=True)
    for new_id, c in enumerate(clusters):
        c.cluster_id = new_id
    return clusters
