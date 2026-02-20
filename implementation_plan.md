# EvalKit — Implementation Plan

**EvalKit** is a Python evaluation harness for benchmarking LLM outputs against expected
solutions across task categories. It ships automated graders with pass/fail scoring,
clusters failures to surface systematic error modes, and drives an iterative prompt
refinement loop that measures accuracy across evaluation runs.

## Design decisions

- **Pure Python 3.10+ package** (`evalkit/`) with a CLI entry point (`evalkit`).
- **Provider-agnostic model layer.** A small `ModelClient` protocol with three backends:
  `mock` (deterministic, offline — used by tests and demos), `openai` and `anthropic`
  (plain HTTPS via `httpx`, keys read from environment variables). No SDK lock-in.
- **Tasks are data.** Suites live in YAML/JSONL files: each task has an `id`, `category`,
  `input` (template variables), `expected` solution and a `grader` spec.
- **Prompts are versioned templates** (`str.format`-style with a system + user part) so
  refinement produces a new prompt version rather than mutating state.
- **Graders are pluggable** via a registry: exact match, normalized match, contains,
  regex, numeric tolerance, JSON field match, Python code execution against test cases
  (sandboxed subprocess with timeout), and LLM-as-judge. Every grader returns a
  `GradeResult(passed, score, reason)`.
- **Runs are reproducible artifacts.** Each run writes `runs/<run_id>/` containing
  `config.json`, `results.jsonl` and `summary.json`; reports compare runs.
- **Failure clustering** uses TF-IDF over failure text (output + grader reason) with
  agglomerative / k-means clustering from scikit-learn, auto-selecting `k` by silhouette
  score, and labels each cluster with its top terms.
- **Refinement loop**: run -> cluster failures -> ask a "refiner" model to rewrite the
  prompt given representative failures -> re-run -> keep the best prompt by accuracy,
  stopping on plateau or max iterations.

## Phase 1 — Core models, task loading and model clients

Deliverables:
- Package skeleton, `pyproject.toml`, dataclasses for `Task`, `PromptTemplate`,
  `ModelResponse`, `GradeResult`, `TaskResult`.
- YAML/JSONL suite loader with validation.
- `ModelClient` protocol + `MockClient`, `OpenAIClient`, `AnthropicClient`, factory.

Files: `pyproject.toml`, `evalkit/__init__.py`, `evalkit/types.py`, `evalkit/tasks.py`,
`evalkit/prompts.py`, `evalkit/models/{__init__,base,mock,openai_client,anthropic_client}.py`,
`.env.example`.

## Phase 2 — Graders

Deliverables:
- Grader base class and registry with `build_grader(spec)`.
- String graders (exact, normalized, contains, regex), numeric tolerance, JSON match.
- Code-execution grader running candidate code against test cases in a subprocess.
- LLM-as-judge grader using any `ModelClient`.

Files: `evalkit/graders/{__init__,base,text,numeric,json_match,code_exec,llm_judge}.py`.

## Phase 3 — Runner, storage, reporting and CLI

Deliverables:
- Concurrent runner (thread pool) with retries and per-task error capture.
- Run storage (`runs/<run_id>/`), summary metrics per category, run comparison.
- Plain-text / Markdown report rendering.
- `evalkit` CLI: `run`, `report`, `compare`.

Files: `evalkit/runner.py`, `evalkit/storage.py`, `evalkit/metrics.py`,
`evalkit/report.py`, `evalkit/cli.py`.

## Phase 4 — Failure clustering

Deliverables:
- Failure text extraction, TF-IDF vectorization, k selection via silhouette score.
- Cluster labels (top terms), representative examples, category breakdown.
- `evalkit cluster` CLI command and cluster section in reports.

Files: `evalkit/clustering.py`, `evalkit/cli.py`, `evalkit/report.py`.

## Phase 5 — Iterative prompt refinement, example suites and tests

Deliverables:
- Refiner that builds a meta-prompt from failure clusters and parses a new template.
- Refinement loop with history (`refinement.json`), best-prompt tracking, plateau stop.
- `evalkit refine` CLI command.
- Example suites (arithmetic, string transforms, code, extraction) and prompts.
- pytest suite covering loaders, graders, runner, clustering and refinement with the
  mock client.

Files: `evalkit/refine.py`, `evalkit/cli.py`, `examples/`, `tests/`.
