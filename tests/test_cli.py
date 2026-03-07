from pathlib import Path

from evalkit.cli import main

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def test_cli_run_report_cluster_compare(tmp_path, capsys):
    runs = str(tmp_path / "runs")
    common = [
        "--runs-dir", runs, "run",
        "--suite", str(EXAMPLES / "suites" / "arithmetic.yaml"),
        "--model", "mock",
        "--mock-rules", str(EXAMPLES / "mock_rules.yaml"),
        "-j", "2",
    ]
    assert main(common + ["--prompt", str(EXAMPLES / "prompts" / "baseline.yaml")]) == 0
    assert main(common + ["--prompt", str(EXAMPLES / "prompts" / "structured.yaml")]) == 0
    out = capsys.readouterr().out
    assert "By category" in out and "arithmetic" in out

    run_dirs = sorted(p for p in Path(runs).iterdir() if p.is_dir())
    assert len(run_dirs) == 2

    assert main(["--runs-dir", runs, "report", "latest", "--clusters"]) == 0
    assert "Failure clusters" in capsys.readouterr().out

    assert main(["--runs-dir", runs, "cluster", str(run_dirs[0])]) == 0
    assert (run_dirs[0] / "clusters.json").exists()

    assert main(["--runs-dir", runs, "compare", str(run_dirs[0]), str(run_dirs[1])]) == 0
    assert "Comparison" in capsys.readouterr().out

    assert main(["--runs-dir", runs, "list"]) == 0


def test_cli_fail_under(tmp_path):
    code = main([
        "--runs-dir", str(tmp_path), "run",
        "--suite", str(EXAMPLES / "suites" / "arithmetic.yaml"),
        "--prompt", str(EXAMPLES / "prompts" / "baseline.yaml"),
        "--model", "mock",
        "--fail-under", "0.99",
    ])
    assert code == 1
