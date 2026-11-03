import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


def _script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sol_cli_usage_separates_cached_input_and_reasoning():
    totals = _script("benchmark_pr1_sol").usage_totals({
        "input_tokens": 3400, "cached_input_tokens": 2000,
        "cache_write_input_tokens": 400, "output_tokens": 200,
        "reasoning_output_tokens": 100})
    assert totals["total"] == 3700
    assert totals["effective"] == pytest.approx(3100)
    assert totals["api_equivalent_usd"] == pytest.approx(0.0062)


def test_invalid_usage_is_not_counted_as_savings():
    with pytest.raises(ValueError, match="Invalid CLI usage"):
        _script("benchmark_pr1_sol").usage_totals({
            "input_tokens": 1, "cached_input_tokens": 2, "output_tokens": 0})


def test_coding_oracles_reject_all_original_bugs(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    tasks = _script("benchmark_sol_tasks").TASKS
    for task in tasks:
        (tmp_path / "solution.py").write_text(task["code"])
        result = subprocess.run([sys.executable, "-B", "-c", "from solution import *\n" + task["checks"]],
                                cwd=tmp_path, capture_output=True, text=True, timeout=10)
        assert result.returncode != 0, task["id"]
