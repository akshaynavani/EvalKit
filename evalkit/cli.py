"""Command-line interface: ``evalkit run|report|compare|list``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from evalkit.metrics import compare
from evalkit.models import GenerationConfig, MockClient, ModelClient, create_client
from evalkit.prompts import load_prompt
from evalkit.report import render_comparison, render_run
from evalkit.runner import Runner
from evalkit.storage import RunStore
from evalkit.tasks import filter_tasks, load_suite


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader; never overrides variables already set."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key.strip(), value)


def build_client(spec: str, args: argparse.Namespace) -> ModelClient:
    config = GenerationConfig(temperature=args.temperature, max_tokens=args.max_tokens)
    client = create_client(spec, config)
    rules_path = getattr(args, "mock_rules", None)
    if isinstance(client, MockClient) and rules_path:
        data = yaml.safe_load(Path(rules_path).read_text(encoding="utf-8")) or {}
        client.rules = MockClient(rules=[(r["pattern"], r["response"]) for r in data.get("rules", [])]).rules
        client.default = data.get("default", client.default)
    return client


def _progress(done: int, total: int, result) -> None:
    mark = "." if result.passed else ("E" if result.error else "F")
    sys.stderr.write(mark)
    if done == total:
        sys.stderr.write("\n")
    sys.stderr.flush()


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", "-m", default="mock", help="provider:model, e.g. openai:gpt-4o-mini")
    p.add_argument("--judge", default=None, help="provider:model used by llm_judge graders")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--concurrency", "-j", type=int, default=4)
    p.add_argument("--mock-rules", default=None, help="YAML file of regex rules for the mock provider")


def _add_suite_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--suite", "-s", required=True, help="suite file or directory")
    p.add_argument("--prompt", "-p", required=True, help="prompt template YAML")
    p.add_argument("--category", "-c", action="append", help="only run these categories")
    p.add_argument("--limit", type=int, default=None)


def cmd_run(args: argparse.Namespace) -> int:
    tasks = filter_tasks(load_suite(args.suite), args.category, args.limit)
    prompt = load_prompt(args.prompt)
    client = build_client(args.model, args)
    judge = build_client(args.judge, args) if args.judge else None
    runner = Runner(client, judge_client=judge, concurrency=args.concurrency)
    run = runner.run(tasks, prompt, suite_name=Path(args.suite).stem, progress=_progress)
    store = RunStore(args.runs_dir)
    path = store.save(run)
    print(render_run(run))
    print(f"Saved run to {path}")
    if args.fail_under is not None and run.accuracy < args.fail_under:
        print(f"Accuracy {run.accuracy:.3f} below threshold {args.fail_under}", file=sys.stderr)
        return 1
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    run = RunStore(args.runs_dir).load(args.run)
    text = render_run(run, max_failures=args.max_failures)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(text)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    store = RunStore(args.runs_dir)
    base, cand = store.load(args.baseline), store.load(args.candidate)
    diff = compare(base.results, cand.results)
    print(json.dumps(diff, indent=2) if args.json else render_comparison(base, cand, diff))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    rows = RunStore(args.runs_dir).list_runs()
    if not rows:
        print("No runs yet.")
        return 0
    for r in rows:
        acc = f"{r['accuracy'] * 100:5.1f}%" if r["accuracy"] is not None else "   n/a"
        print(f"{r['run_id']:<34} {acc}  {r['suite']:<16} {r['model']:<28} {r['prompt']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evalkit", description="LLM task evaluation harness")
    parser.add_argument("--runs-dir", default="runs", help="where runs are stored (default: runs/)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="evaluate a prompt + model on a suite")
    _add_suite_args(p)
    _add_model_args(p)
    p.add_argument("--fail-under", type=float, default=None, help="exit 1 if accuracy is below this")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("report", help="render a stored run as Markdown")
    p.add_argument("run", nargs="?", default="latest")
    p.add_argument("--output", "-o", default=None)
    p.add_argument("--max-failures", type=int, default=25)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("compare", help="diff two stored runs")
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("list", help="list stored runs")
    p.set_defaults(func=cmd_list)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
