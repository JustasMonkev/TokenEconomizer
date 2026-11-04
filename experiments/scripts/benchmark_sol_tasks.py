"""Small local coding-task smoke benchmark of TE's native Codex pruning controller."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

from benchmark_pr1_sol import usage_totals


TASKS = [
    {"id": "cache", "issue": "Fix lookup so a cached False, 0, empty string, or None is returned unchanged; use the fallback only for an absent key.",
     "code": "def lookup(cache, key, fallback):\n    return cache.get(key) or fallback\n",
     "smoke": "assert lookup({'a': 1}, 'a', 2) == 1",
     "checks": "assert lookup({'a': False}, 'a', 7) is False\nassert lookup({'a': 0}, 'a', 7) == 0\nassert lookup({'a': ''}, 'a', 7) == ''\nassert lookup({'a': None}, 'a', 7) is None\nassert lookup({}, 'a', 7) == 7"},
    {"id": "pagination", "issue": "Fix pages: return all input items in nonempty pages of at most size items. Empty input returns []. Reject size <= 0 with ValueError.",
     "code": "def pages(items, size):\n    return [items[i:i + size] for i in range(0, len(items) - 1, size)]\n",
     "smoke": "assert pages([1, 2, 3], 2) == [[1, 2], [3]]",
     "checks": "assert pages([1], 1) == [[1]]\nassert pages([1,2,3,4],1)==[[1],[2],[3],[4]]\nassert pages([],2)==[]\nassert pages([1,2,3,4],2)==[[1,2],[3,4]]\nfor size in (0,-1):\n    try: pages([1],size)\n    except ValueError: pass\n    else: raise AssertionError('size accepted')"},
    {"id": "unicode", "issue": "Fix same_name to compare Unicode names case-insensitively, including Straße versus STRASSE and Greek sigma forms. Do not ignore whitespace or accents.",
     "code": "def same_name(left, right):\n    return left.lower() == right.lower()\n",
     "smoke": "assert same_name('Alice', 'ALICE')",
     "checks": "assert same_name('Straße','STRASSE')\nassert same_name('Σ','ς')\nassert not same_name(' alice','alice')\nassert not same_name('café','cafe')\nassert same_name('','')"},
    {"id": "retry", "issue": "Fix request: retries is the number of additional attempts after the first. Retry only ValueError, return the first successful result, re-raise the final ValueError, and reject negative retries with ValueError before calling send.",
     "code": "def request(send, retries):\n    for _ in range(retries):\n        try:\n            return send()\n        except ValueError:\n            pass\n",
     "smoke": "assert request(lambda: 'ok', 1) == 'ok'",
     "checks": "assert request(lambda: False,0) is False\ncalls=[]\ndef send():\n    calls.append(1)\n    raise ValueError('last')\ntry: request(send,2)\nexcept ValueError as e: assert str(e)=='last'\nelse: raise AssertionError('swallowed')\nassert len(calls)==3\ncalls.clear()\ntry: request(send,-1)\nexcept ValueError: pass\nelse: raise AssertionError('negative')\nassert calls==[]\ndef fatal(): raise TypeError('fatal')\ntry: request(fatal,3)\nexcept TypeError: pass\nelse: raise AssertionError('wrong exception')"},
    {"id": "csv", "issue": "Fix encode_row to emit one CSV record that round-trips arbitrary strings including commas, quotes, newlines and empty strings using Python csv.reader. Return a str.",
     "code": "def encode_row(values):\n    return ','.join(values) + '\\n'\n",
     "smoke": "assert encode_row(['a','b']).strip() == 'a,b'",
     "checks": "import csv,io\nfor values in (['a,b','c'],['a\\nb','c'],['a\\rb','c'],['a\\\"b',''],[''],[]):\n    encoded=encode_row(values)\n    assert isinstance(encoded,str)\n    assert list(csv.reader(io.StringIO(encoded,newline='')))==[values]"},
    {"id": "dedupe", "issue": "Fix unique to remove duplicates while retaining first-occurrence order. Support unhashable values such as lists and dictionaries. Equality, not object identity, defines duplicates. Do not mutate the input.",
     "code": "def unique(values):\n    return list(set(values))\n",
     "smoke": "assert unique([1,1]) == [1]",
     "checks": "assert unique([3,1,3,2,1])==[3,1,2]\nassert unique([[1],[1],[2]])==[[1],[2]]\nassert unique([{'a':1},{'a':1}])==[{'a':1}]\na=[3,2,3];assert unique(a)==[3,2];assert a==[3,2,3]\nassert unique([])==[]"},
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261103)
    ap.add_argument("--cycles", type=int, default=1)
    args = ap.parse_args()
    sys.path[:0] = [str(args.source.resolve()), str(args.source.resolve() / "experiments")]
    from tokeneconomizer.toolprune.codex import CodexUntoolProfile, inspect_usage
    from te_harness.codex import config_toggles

    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if list(root.glob('*/row.json')):
        raise RuntimeError("Use a fresh output directory to preserve causal learning")
    rng = random.Random(args.seed)
    if args.cycles < 1:
        raise ValueError("cycles must be positive")
    tasks = []
    for cycle in range(args.cycles):
        for task in rng.sample(TASKS, len(TASKS)):
            tasks.append(dict(task, id=task["id"] if args.cycles == 1 else f"{task['id']}-{cycle}"))
    source_diff = subprocess.check_output(["git", "-C", str(args.source), "diff", "HEAD"], text=True)
    (root / "source-diff.patch").write_text(source_diff)
    (root / "runner-source.py").write_text(Path(__file__).read_text())
    profile = CodexUntoolProfile("local/python-smoke", str(root / "state"), fresh=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix="te-sol-auth-") as auth_dir:
        auth_home = Path(auth_dir)
        (auth_home / "auth.json").symlink_to(Path.home() / ".codex/auth.json")
        for task in tasks:
            selection_kwargs = {}
            if "available_capabilities" in inspect.signature(profile.begin_task).parameters:
                selection_kwargs["available_capabilities"] = ["collaboration", "web", "goals", "skills", "image_media"]
            selection = profile.begin_task(task["id"], problem_statement=task["issue"], **selection_kwargs)
            arms = ["baseline", "te_native_pruning"]
            rng.shuffle(arms)
            for arm in arms:
                target = root / f"{task['id']}-{arm}"
                workspace = target / "workspace"
                workspace.mkdir(parents=True)
                (workspace / "solution.py").write_text(task["code"])
                smoke = "from solution import *\n" + task["smoke"] + "\n"
                (workspace / "smoke.py").write_text(smoke)
                prompt = (task["issue"] + "\nEdit solution.py in this directory. Keep the public function signature. "
                          "Run python3 smoke.py and add any useful edge checks. Do not modify smoke.py. "
                          "Do not access files outside this working directory or use the network. No delegation. Finish when fixed.")
                cmd = ["codex", "exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check",
                       "--disable", "hooks", "-s", "workspace-write", "--json", "-m", "gpt-6.1-sol",
                       "-c", 'model_reasoning_effort="low"', "-c", "project_doc_max_bytes=0"]
                applied, unsupported = [], []
                if arm != "baseline":
                    capabilities = selection["config_capabilities"]
                    applied = [c for c in capabilities if c != "interactive_guidance"]
                    unsupported = [c for c in selection["logical_capabilities"] if c not in applied]
                    top, tables = config_toggles(applied)
                    for line in top:
                        cmd += ["-c", line]
                    for table, line in tables:
                        cmd += ["-c", f"{table}.{line}"]
                    if "skills" in applied:
                        paths = sorted((auth_home / "skills").rglob("SKILL.md"))
                        cmd += ["-c", "skills.config=[" + ",".join(
                            "{path=" + json.dumps(str(p)) + ",enabled=false}" for p in paths) + "]"]
                cmd.append("-")
                (target / "prompt.txt").write_text(prompt)
                started = time.monotonic()
                env = dict(os.environ, CODEX_HOME=auth_dir, TE_SIDECALL="1")
                proc = subprocess.run(cmd, input=prompt, text=True, capture_output=True,
                                      cwd=workspace, env=env, timeout=180)
                elapsed = time.monotonic() - started
                trace = target / "transcript.jsonl"
                trace.write_text(proc.stdout)
                (target / "stderr.txt").write_text(proc.stderr)
                usage = None
                for line in proc.stdout.splitlines():
                    event = json.loads(line)
                    if event["type"] == "turn.completed":
                        usage = event["usage"]
                if proc.returncode or usage is None:
                    raise RuntimeError(f"Invalid run: {target}; not counted as saving")
                grader = "from solution import *\n" + task["checks"] + "\n"
                grade = subprocess.run([sys.executable, "-c", grader], cwd=workspace,
                                       capture_output=True, text=True, timeout=10)
                passed = grade.returncode == 0 and (workspace / "smoke.py").read_text() == smoke
                (target / "grade.txt").write_text(grade.stdout + grade.stderr)
                row = {"task": task["id"], "arm": arm, "resolved": passed, "seconds": elapsed,
                       "usage": usage, "command": cmd, "applied_config_capabilities": applied,
                       "unsupported_capabilities": unsupported, **usage_totals(usage)}
                if arm != "baseline":
                    row["selection"] = selection
                    cost_kwargs = {}
                    if "cost_row" in inspect.signature(profile.observe).parameters:
                        cost_kwargs["cost_row"] = {
                            "model": "gpt-6.1-sol", "input_tokens": row["input"] - row["cached"] - row["cache_write"],
                            "cache_read_input_tokens": row["cached"], "cache_creation_input_tokens": row["cache_write"],
                            "output_tokens": row["output"]}
                    row["observation"] = profile.observe(inspect_usage(str(trace)), resolved=passed, **cost_kwargs)
                (target / "row.json").write_text(json.dumps(row, indent=2) + "\n")
                rows.append(row)
                print(json.dumps({k: row[k] for k in ("task", "arm", "resolved", "input", "output", "total", "applied_config_capabilities")}), flush=True)
    summary = {"model": "gpt-6.1-sol", "effort": "low", "seed": args.seed,
               "cycles": args.cycles, "source_diff_sha256": hashlib.sha256(source_diff.encode()).hexdigest(),
               "source_commit": subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip(),
               "scope": "6 synthetic Python bug fixes per cycle, cold-start native TE capability pruning only; no micro-skills, loop guard or relay rewriting; not SWE-bench",
               "task_order": [t["id"] for t in tasks], "arms": {}}
    for arm in ("baseline", "te_native_pruning"):
        group = [r for r in rows if r["arm"] == arm]
        summary["arms"][arm] = {"runs": len(group), "resolved": sum(r["resolved"] for r in group),
                               **{k: sum(r[k] for r in group) for k in ("input", "cached", "cache_write", "output", "total", "effective", "api_equivalent_usd", "seconds")}}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
