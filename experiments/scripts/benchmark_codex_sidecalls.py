"""Compare actual Codex side-call implementations using the same compact judge prompts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

from benchmark_pr1_sol import TASKS, usage_totals


def worker(source, target):
    sys.path.insert(0, source)
    from tokeneconomizer.plugin import sidecall

    prompt = sys.stdin.read()
    target = Path(target)
    real_run = subprocess.run
    observations = []

    def record(cmd, **kwargs):
        result = real_run(cmd, **kwargs)
        if len(cmd) > 1 and cmd[1] == "exec" and "--json" in cmd:
            observations.append({"command": cmd, "returncode": result.returncode,
                                 "stdout": result.stdout, "stderr": result.stderr})
        return result

    sidecall.subprocess.run = record
    started = time.monotonic()
    reply, charged = sidecall.codex_call(prompt, model="gpt-6.1-sol")
    elapsed = time.monotonic() - started
    (target / "calls.json").write_text(json.dumps(observations, indent=2) + "\n")
    if len(observations) != 1 or observations[0]["returncode"] != 0:
        raise RuntimeError("Expected exactly one successful paid invocation")
    usage, unexpected = None, []
    for line in observations[0]["stdout"].splitlines():
        event = json.loads(line)
        if event["type"] == "turn.completed":
            usage = event["usage"]
        elif event["type"] == "item.completed" and event.get("item", {}).get("type") not in (
                "agent_message", "reasoning", "error"):
            unexpected.append(event["item"])
    if usage is None or unexpected or not reply:
        raise RuntimeError("Invalid text-only completion; do not count as savings")
    row = {"reply": reply, "usage": usage, "seconds": elapsed,
           "reported_effective": charged, **usage_totals(usage)}
    (target / "usage.json").write_text(json.dumps(row, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    from tokeneconomizer.toolprune.tools import judge_tools_for_task, PROTECTED_TOOLS

    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if list(root.glob("*/usage.json")):
        raise ValueError("Choose a fresh output directory")
    catalog = json.loads((args.source / "tokeneconomizer/trace/claude_tool_catalog.json").read_text())
    selected = [t for t in TASKS if t[0] in ("cache", "retry", "web_docs", "notebook")]
    cases = [(repeat, task, arm) for repeat in range(args.repeats) for task in selected
             for arm in ("reference", "lean")]
    random.Random(20261103).shuffle(cases)
    diff = subprocess.check_output(["git", "-C", str(args.source), "diff", "HEAD"], text=True)
    (root / "source-diff.patch").write_text(diff)
    manifest = {"model": "gpt-6.1-sol", "effort": "low", "seed": 20261103,
                "source_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
                "reference_commit": subprocess.check_output(["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True).strip(),
                "order": [[r, t[0], a] for r, t, a in cases],
                "scope": "Actual sidecall.codex_call paths, shared isolated config/auth; compact judge classification only; no API calls outside CLI"}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    with tempfile.TemporaryDirectory(prefix="te-side-benchmark-") as home:
        auth_home = Path(home)
        (auth_home / "auth.json").symlink_to(Path.home() / ".codex/auth.json")
        (auth_home / "config.toml").write_text('model_reasoning_effort = "low"\n')
        env = dict(os.environ, CODEX_HOME=home)
        for repeat, (name, issue, required), arm in cases:
            target = root / f"{name}-{repeat}-{arm}"
            target.mkdir()
            captured = []

            def capture(prompt, model):
                captured.append(prompt)
                return '{"drop": []}', 0

            judge_tools_for_task(issue, catalog, capture, compact=True)
            prompt = captured[0]
            (target / "prompt.txt").write_text(prompt)
            source = args.reference if arm == "reference" else args.source
            proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker",
                                   str(source.resolve()), str(target)], input=prompt, capture_output=True,
                                  text=True, env=env, cwd=target, timeout=240)
            if proc.returncode:
                (target / "worker-error.txt").write_text(proc.stderr)
                raise RuntimeError(f"Side-call benchmark failed: {target}")
            row = json.loads((target / "usage.json").read_text())
            verdict, _ = judge_tools_for_task(issue, catalog, lambda *a: (row["reply"], row["effective"]), compact=True)
            passed = bool(verdict) and all(verdict.get(t) for t in required) and all(
                verdict.get(t) for t in catalog if t in PROTECTED_TOOLS)
            row.update(task=name, repeat=repeat, arm=arm, verdict=verdict, required_tools_retained=passed)
            (target / "row.json").write_text(json.dumps(row, indent=2) + "\n")
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("task", "repeat", "arm", "input", "output", "effective", "required_tools_retained")}), flush=True)
    summary = {}
    for arm in ("reference", "lean"):
        group = [r for r in rows if r["arm"] == arm]
        summary[arm] = {"runs": len(group), "quality_pass": sum(r["required_tools_retained"] for r in group),
                        **{k: sum(r[k] for r in group) for k in ("input", "cached", "cache_write", "output", "total", "effective", "seconds")}}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3])
    else:
        main()
