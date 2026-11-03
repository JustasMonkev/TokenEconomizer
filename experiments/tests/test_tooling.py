"""Tooling: generic evidence untooling + Codex-native capability untooling."""
import os

from tokeneconomizer.toolprune import ToolingStage
from tokeneconomizer.toolprune import codex as cu
from tokeneconomizer.config import ToolingConfig
from te_bench.adapters import SwebenchAdapter
from fixtures import write_codex_transcript


# ---- Codex-native capability untool controller ----

def test_codex_profile_task1_full_contract(tmp_path):
    p = cu.CodexUntoolProfile("pydata/xarray", str(tmp_path), fresh=True)
    sel = p.begin_task("x-1", problem_statement="fix concat")
    assert sel["audit"] is True                 # task 1 observes the complete contract
    assert sel["relay_capabilities"] == []       # nothing pruned yet
    assert "deferred_wait" in sel["core_capabilities"]


def test_codex_profile_never_prunes_core(tmp_path):
    p = cu.CodexUntoolProfile("pydata/xarray", str(tmp_path), fresh=True)
    # drive several tasks where nothing completes; deferred_wait must never be demoted.
    for i in range(6):
        sel = p.begin_task(f"x-{i}")
        assert "deferred_wait" not in sel["relay_capabilities"]
        usage = {"attempted": {}, "completed": {}, "failed": {}}
        p.observe(usage, resolved=True)
    assert "deferred_wait" not in p.state["current_pruned"]


def _priced_baseline(profile):
    for i in range(cu.COST_WINDOW):
        profile.begin_task(f"baseline-{i}")
        profile.observe({"attempted": {}, "completed": {}, "failed": {}}, resolved=True,
                        cost_row={"model": "gpt-5.6-sol", "input_tokens": 1000,
                                  "cache_read_input_tokens": 0,
                                  "cache_creation_input_tokens": 0, "output_tokens": 100})


def test_codex_profile_trials_unused_web_after_priced_baseline(tmp_path):
    p = cu.CodexUntoolProfile("pydata/xarray", str(tmp_path), fresh=True)
    _priced_baseline(p)
    sel2 = p.begin_task("trial-1")
    assert "web" in sel2["relay_capabilities"]


def test_codex_profile_completed_use_protects(tmp_path):
    p = cu.CodexUntoolProfile("pydata/xarray", str(tmp_path), fresh=True)
    p.begin_task("x-1")
    # a completed web use permanently protects it.
    p.observe({"attempted": {"web": 1}, "completed": {"web": 1}, "failed": {}},
              resolved=True)
    assert "web" in p.state["ever_completed"]
    sel2 = p.begin_task("x-2")
    assert "web" not in sel2["relay_capabilities"]


def test_inspect_usage_counts_web(tmp_path):
    tr = str(tmp_path / "t.jsonl")
    write_codex_transcript(tr, [
        {"type": "web_search", "name": "web_search", "status": "failed"},
        {"type": "command_execution", "command": "pytest"}])
    usage = cu.inspect_usage(tr)
    assert usage["attempted"]["web"] >= 1
    assert usage["completed"]["web"] == 0        # failed status is not a completed use


# ---- Codex ToolingStage wiring ----

def test_tooling_stage_codex_mask(tmp_path):
    cfg = ToolingConfig(codex_untool=True)
    stage = ToolingStage(cfg, "codex", "agent",
                         adapter=SwebenchAdapter(), repo="pydata/xarray",
                         results_root=str(tmp_path), fresh=True)
    task = {"instance_id": "x-1", "problem_statement": "fix"}
    mask = stage.build_tools(task, 0)
    assert mask == []                            # task 1 = full contract
    rd = str(tmp_path / "run"); os.makedirs(rd)
    write_codex_transcript(os.path.join(rd, "transcript.jsonl"),
                           [{"type": "command_execution", "command": "pytest"}])
    srow = {"resolved": True}
    stage.after_task(task, {}, rd, srow)
    assert srow["codex_untool_active"] is True
    assert srow["codex_untool_profile"] == cu.PROFILE_VERSION


# ---- generic evidence untooling ----

class _StubAdapter:
    """Minimal adapter: fixed catalog, no tool used, protects nothing."""
    name = "stub"
    def tool_usage(self, path):
        return {}
    def protected_tools(self):
        return frozenset()
    def parse_catalog(self, run_dir_path, agent):
        return {"web_fetch": "fetch a url", "ask_user": "ask the human"}


def test_evidence_untool_drops_after_streak(tmp_path):
    cfg = ToolingConfig(evidence=2)
    stage = ToolingStage(cfg, "claude", "agent", adapter=_StubAdapter())
    rd = str(tmp_path / "run"); os.makedirs(rd)
    open(os.path.join(rd, "transcript.jsonl"), "w").write("")
    task = {"instance_id": "w-1"}
    # task 0: catalog gets learned, streaks start
    stage.build_tools(task, 0)
    stage.after_task(task, {}, rd, {"resolved": True})
    # after 2 unused tasks, the tools cross the evidence threshold
    stage.after_task(task, {}, rd, {"resolved": True})
    excl = stage.build_tools(task, 3)
    assert "web_fetch" in excl and "ask_user" in excl


# ---- the mask must be EARNED, never a prior ---------------------------------
# The mask comes only from observed disuse: none before the first task has completed, and
# never one drawn from the adapters' static `untool_exclusions()` / INAPPLICABLE_TOOLS
# surface, which is still exported but not reachable from the stage. A static prior would be
# live from the first rollout, cost zero LLM tokens (so the honest ledger would charge nothing
# for it) and give a byte-identical keep-set in every cell -- an unpaid, unlearned saving
# rather than the evidence-gated mechanism. These tests pin that in BOTH domains, because the
# difference is invisible in aggregate numbers and shows up only as a suspiciously large cost
# win.
from te_bench.adapters import WebBenchAdapter                    # noqa: E402
from tokeneconomizer.toolprune import tools as reflect                  # noqa: E402
from fixtures import write_claude_transcript_with_catalog               # noqa: E402

_CATALOG = ["Bash", "Edit", "Read", "Write", "WebFetch", "WebSearch", "Workflow",
            "Skill", "NotebookEdit", "CronCreate", "TaskCreate"]


def _stage(tmp_path, adapter, evidence=3):
    return ToolingStage(ToolingConfig(evidence=evidence, judge=False), "claude", "agent",
                        adapter=adapter, repo="acme/widget",
                        results_root=str(tmp_path), fresh=True)


def _finish(stage, tmp_path, i, used, catalog=_CATALOG):
    """Run one task to completion so the stage can learn from its transcript."""
    rd = os.path.join(str(tmp_path), f"run{i}")
    os.makedirs(rd, exist_ok=True)
    write_claude_transcript_with_catalog(
        os.path.join(rd, "transcript.jsonl"), catalog,
        [(t, {"command": "x"} if t == "Bash" else {"file_path": "a.py"}, "ok")
         for t in used])
    stage.after_task({"instance_id": f"w-{i}"}, {}, rd, {})


def test_first_task_of_a_stream_is_never_masked_swebench(tmp_path):
    """No evidence can exist before the first task, so the agent must see the full catalog."""
    stage = _stage(tmp_path, SwebenchAdapter(tasks_path=None))
    assert stage.build_tools({"instance_id": "w-0"}, 0) == []


def test_first_task_of_a_stream_is_never_masked_webbench(tmp_path):
    """Same guarantee on the Web-Bench adapter."""
    stage = _stage(tmp_path, WebBenchAdapter("zustand"))
    assert stage.build_tools({"instance_id": "task-1"}, 0) == []


def test_mask_appears_only_after_the_catalog_is_learned(tmp_path):
    """The trim must come from observed disuse: full catalog on task 0, then a mask."""
    stage = _stage(tmp_path, WebBenchAdapter("zustand"))
    assert stage.build_tools({"instance_id": "task-1"}, 0) == []
    for i in range(3):                       # three tasks using only Bash/Edit/Read/Write
        _finish(stage, tmp_path, i, ["Bash", "Edit", "Read", "Write"])
    mask = stage.build_tools({"instance_id": "task-4"}, 3)
    assert mask, "after 3 tasks of disuse the unused tools should be dropped"
    assert not (set(mask) & reflect.PROTECTED_TOOLS), "protected tools must survive"
    for t in ("WebFetch", "Workflow", "CronCreate"):
        assert t in mask, f"{t} was loaded and unused for 3 tasks; expected it dropped"


def test_static_inapplicable_prior_is_not_applied(tmp_path):
    """An adapter that declares EVERYTHING inapplicable must not change the mask.

    The stage must consult observed usage, never the adapter's static
    `inapplicable_tools` / `hardcoded_exclusions` surface.
    """
    class _GreedyPrior(WebBenchAdapter):
        def inapplicable_tools(self):
            return tuple(_CATALOG)
        def hardcoded_exclusions(self, used_tools):
            return list(_CATALOG)

    stage = _stage(tmp_path, _GreedyPrior("zustand"))
    assert stage.build_tools({"instance_id": "task-1"}, 0) == []      # still unmasked
    _finish(stage, tmp_path, 0, ["Bash", "Edit", "Read", "Write"])
    mask = stage.build_tools({"instance_id": "task-2"}, 1)
    assert not (set(mask) & reflect.PROTECTED_TOOLS)
    # one task of disuse is below the evidence threshold of 3 -> nothing earned yet
    assert mask == [], f"a static prior leaked into the mask: {mask}"


def test_untooling_is_off_entirely_without_an_evidence_threshold(tmp_path):
    """`evidence=0` must mean NO mask, not a static one."""
    stage = _stage(tmp_path, WebBenchAdapter("zustand"), evidence=0)
    _finish(stage, tmp_path, 0, ["Bash"])
    assert stage.build_tools({"instance_id": "task-2"}, 1) is None


# ---- network tools are blocked on every arm ---------------------------------------------
# Both benchmarks' answers -- gold patches, grading specs -- are on the public internet, a
# fetch away from the agent. These pin the block so it cannot be quietly dropped, and pin the
# escape hatch so it stays deliberate.

def test_network_tools_are_blocked_by_default(monkeypatch):
    from te_harness import webbench as wb, swebench as sb
    monkeypatch.delenv("TE_ALLOW_NETWORK_TOOLS", raising=False)
    for mod in (wb, sb):
        assert set(mod.network_tool_block()) == {"WebSearch", "WebFetch"}, mod.__name__


def test_network_block_is_liftable_but_only_explicitly(monkeypatch):
    from te_harness import webbench as wb
    for val, blocked in (("1", False), ("true", False), ("yes", False),
                         ("0", True), ("", True), ("no", True)):
        monkeypatch.setenv("TE_ALLOW_NETWORK_TOOLS", val)
        assert bool(wb.network_tool_block()) is blocked, val


def test_network_tools_are_not_protected():
    """PROTECTED_TOOLS survive every mask, so a network tool must never be listed there."""
    from tokeneconomizer.toolprune import tools as reflect
    assert not ({"WebSearch", "WebFetch"} & set(reflect.PROTECTED_TOOLS))


# ---- the Web-Bench PreToolUse network guard ----------------------------------------------
# Derived from the traces, not from taste: across 774 stored transcripts the network calls a
# task needs (335: 226 package-manager, 109 loopback) never mention GitHub, and every route
# to the benchmark's answer key does. These pin both halves of that separation.

def test_net_guard_blocks_answer_key_fetches():
    from te_harness.guards.net_guard import decide
    for cmd in (
        "curl -sL https://raw.githubusercontent.com/bytedance/web-bench/main/"
        "projects/threejs/test/task-7.spec.js",
        "curl https://api.github.com/repos/bytedance/web-bench/git/trees/main?recursive=1",
        "wget https://github.com/bytedance/web-bench/archive/main.tar.gz",
    ):
        blocked, reason = decide(cmd)
        assert blocked and reason, cmd


def test_net_guard_allows_every_legitimate_shape_seen_in_the_stored_runs():
    from te_harness.guards.net_guard import decide
    for cmd in ("npm install", "npx vite build", "yarn add three",
                "curl -s http://localhost:3211/", "curl -s http://127.0.0.1:3005/index.html",
                "ls -la src/", "npm run test -- -a"):
        assert not decide(cmd)[0], cmd


def test_net_guard_reads_both_agents_payload_spellings():
    """Claude sends `tool_input`, the Copilot CLI sends `toolArgs`.

    Reading only one is silent: the hook installs, fires on every call, finds no command
    and exits 0 -- guarded-looking and guarding nothing. Measured off a live Copilot
    preToolUse hook (CLI 1.0.83, 2026-09-08).
    """
    from te_harness.guards.net_guard import hook_command
    cmd = "curl https://github.com/x/y.patch"
    assert hook_command({"tool_name": "Bash", "tool_input": {"command": cmd}}) == cmd
    assert hook_command({"toolName": "bash", "toolArgs": {"command": cmd}}) == cmd
    # Nothing usable must degrade to "" rather than raising -- the hook fails open.
    for junk in ({}, {"toolArgs": {}}, {"tool_input": None}, [], "nope", None):
        assert hook_command(junk) == ""


def test_net_guard_blocks_end_to_end_on_a_copilot_shaped_payload():
    """End to end, the hook exits rc=2 on the Copilot spelling, not just the Claude one."""
    import json, subprocess, sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {k: v for k, v in os.environ.items() if k != "TE_ALLOW_NETWORK_TOOLS"}
    for key in ("tool_input", "toolArgs"):
        payload = json.dumps({key: {"command": "curl https://github.com/x"}})
        r = subprocess.run([sys.executable, "-m", "te_harness.guards.net_guard"],
                           input=payload, capture_output=True, text=True,
                           cwd=root, env=env)
        assert r.returncode == 2, f"{key}: rc={r.returncode} -- would block nothing"


def test_net_guard_is_liftable_by_the_same_switch_as_the_tool_block(monkeypatch, tmp_path):
    import json, subprocess, sys
    payload = json.dumps({"tool_input": {"command": "curl https://github.com/x"}})
    run = lambda env: subprocess.run(
        [sys.executable, "-m", "te_harness.guards.net_guard"], input=payload,
        capture_output=True, text=True, env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    base = dict(os.environ); base.pop("TE_ALLOW_NETWORK_TOOLS", None)
    assert run(base).returncode == 2                      # blocks, and tells the model why
    assert run({**base, "TE_ALLOW_NETWORK_TOOLS": "1"}).returncode == 0


def test_net_guard_fails_open_on_unreadable_input(tmp_path):
    """A malformed hook payload must never wedge the agent."""
    import subprocess, sys
    r = subprocess.run([sys.executable, "-m", "te_harness.guards.net_guard"],
                       input="not json", capture_output=True, text=True,
                       cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert r.returncode == 0


# ---- the guard is installed on BOTH harnesses, not just Web-Bench ------------------------
# Masking WebSearch/WebFetch does not cover the agent's main route to the network on
# SWE-bench, which is `Bash` (across 1,824 stored rows, 120 reached the network through it
# against 47 through a web tool). `Bash` is protected and can never be masked, so the
# PreToolUse hook is the only seam -- and for SWE-bench it has to be installed INSIDE the
# task container, where the agent actually runs.

def _method_calls(cls_name, method, path="te_harness/swebench.py"):
    """The attribute names `cls_name.method` calls on self, via AST (no Docker needed)."""
    import ast
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tree = ast.parse(open(os.path.join(root, path), encoding="utf-8").read())
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == cls_name)
    fn = next(n for n in cls.body
              if isinstance(n, ast.FunctionDef) and n.name == method)
    return {n.func.attr for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}


def test_the_swebench_runner_installs_and_reads_the_container_side_guard():
    calls = _method_calls("ClaudeSwebenchRunner", "run_task")
    assert "_install_net_guard" in calls, "SWE-bench tasks would run unguarded"
    assert "_net_guard_blocks" in calls, "a guard you cannot watch fire is not evidence"


def test_the_swebench_guard_honours_the_same_escape_hatch(monkeypatch):
    """`TE_ALLOW_NETWORK_TOOLS` lifts both halves or neither -- never one silently."""
    from te_harness import swebench as sb
    monkeypatch.setenv("TE_ALLOW_NETWORK_TOOLS", "1")
    runner = sb.ClaudeSwebenchRunner.__new__(sb.ClaudeSwebenchRunner)
    assert runner._install_net_guard("no-such-container") is False


def test_the_recorded_tool_mask_is_the_one_the_cli_was_given(monkeypatch):
    """`run_config.tools.excluded_applied` must include the network block: without it a
    guarded run is byte-identical to an unguarded one in its own provenance."""
    import ast
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for path in ("te_harness/swebench.py", "te_harness/webbench.py"):
        tree = ast.parse(open(os.path.join(root, path), encoding="utf-8").read())
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_write_run_config")
        names = {n.func.id for n in ast.walk(fn)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert "network_tool_block" in names, path


# ---------------------------------------------------------------------------------------
# `goals` as a config-prunable package: the product-goal tool docs in the code-mode exec
# description, removable with `[features] goals = false`.
# ---------------------------------------------------------------------------------------
def test_codex_profile_trials_unused_goals_after_priced_baseline(tmp_path):
    p = cu.CodexUntoolProfile("sympy/sympy", str(tmp_path), fresh=True)
    sel1 = p.begin_task("s-1")
    assert "goals" not in sel1["config_capabilities"]          # task 1: complete contract
    p.observe({"attempted": {}, "completed": {}, "failed": {}}, resolved=True)
    _priced_baseline(p)
    sel3 = p.begin_task("s-3")
    assert "goals" in sel3["config_capabilities"]
    assert "goals" in sel3["logical_capabilities"]
    assert "goals" not in sel3["relay_capabilities"]           # config-only, no relay id


def test_codex_profile_completed_goal_use_protects(tmp_path):
    p = cu.CodexUntoolProfile("sympy/sympy", str(tmp_path), fresh=True)
    p.begin_task("s-1")
    p.observe({"attempted": {"goals": 1}, "completed": {"goals": 1}, "failed": {}},
              resolved=True)
    assert "goals" in p.state["ever_completed"]
    for i in range(2, 6):
        sel = p.begin_task(f"s-{i}")
        assert "goals" not in sel["config_capabilities"]
        p.observe({"attempted": {}, "completed": {}, "failed": {}}, resolved=True)


def test_inspect_usage_counts_goal_items(tmp_path):
    path = str(tmp_path / "t.jsonl")
    write_codex_transcript(path, [
        {"type": "create_goal", "id": "g1", "status": "completed", "name": "create_goal"},
        {"type": "custom_tool_call", "id": "c1", "status": "completed", "name": "exec",
         "input": "await tools.update_goal({id: 1});"},
        {"type": "command_execution", "command": "ls", "aggregated_output": "", "exit_code": 0},
    ])
    u = cu.inspect_usage(path)
    assert u["attempted"]["goals"] == 2 and u["completed"]["goals"] == 2


def test_harness_emits_the_goals_toggle_and_the_cli_accepts_the_whole_mask():
    from te_harness import codex
    top, tables = codex.config_toggles(["goals"])
    assert ("features", "goals = false") in tables
    assert codex.CONFIG_CAPABILITIES == cu.CONFIG_CAPABILITIES
    # Only worth running where the harness's CLI is installed: prove every toggle the full
    # mask can emit is a field 0.153.1+ recognises (`--strict-config`, offline, unbilled).
    import shutil, pytest
    if not (os.environ.get("TE_CODEX_BIN") or shutil.which("codex")):
        pytest.skip("codex-cli not installed")
    routing = {"base_url": "http://127.0.0.1:9/v1", "api_key": "k"}
    txt = codex.config_toml(routing=routing, model="gpt-5.6-terra", effort="high",
                            capabilities=sorted(codex.CONFIG_CAPABILITIES))
    assert "goals = false" in txt
    err = codex.validate_config(txt)
    assert not err, err


def test_codex_profile_trials_interactive_guidance_after_priced_baseline(tmp_path):
    p = cu.CodexUntoolProfile("sympy/sympy", str(tmp_path), fresh=True)
    _priced_baseline(p)
    sel = p.begin_task("trial-1")
    assert "interactive_guidance" in sel["config_capabilities"]
    assert "interactive_guidance" not in sel["relay_capabilities"]  # config-only package
