"""Facade: the Economizer drives all three mechanisms end-to-end (offline)."""
import os

from te_bench.facade import Economizer
from tokeneconomizer import EconomizerConfig
from tokeneconomizer.config import InjectionConfig, ToolingConfig, SpiralConfig
from te_bench.adapters import SwebenchAdapter
from fixtures import write_claude_transcript, write_codex_transcript


def _runner(prompt, model="agent"):
    return ('{"extractors":[],"facts":[]}', 5.0)


def test_generic_trimech_end_to_end(tmp_path):
    cfg = EconomizerConfig(
        injection=InjectionConfig(enabled=True, propose_every=5),
        tooling=ToolingConfig(evidence=3, judge=False),
        spiral=SpiralConfig(guard=True, nudge=True))
    econ = Economizer(cfg, agent="claude", repo="acme/widget",
                      results_root=str(tmp_path), fresh=True, runner=_runner)
    task = {"instance_id": "w-1", "problem_statement": "fix"}
    assert econ.injection_prefix(task) == ""       # nothing formed yet
    assert econ.build_tools(task, 0) == []          # no catalog learned yet
    rk = econ.run_kwargs(task)
    assert rk["spiral_guard"] is True

    rd = str(tmp_path / "run"); os.makedirs(rd)
    write_claude_transcript(os.path.join(rd, "transcript.jsonl"),
                            [("bash", {"command": "pytest"}, "ok")])
    row = {"resolved": True, "num_turns": 10, "tokens_total": 1000,
           "input_tokens": 100, "cache_read_input_tokens": 800,
           "cache_creation_input_tokens": 50, "output_tokens": 50}
    srow = {"resolved": True, "num_turns": 10}
    opt, reason = econ.after_task(task, row, rd, srow)
    assert reason == "manual-distill"
    assert "manual_facts" in srow and "aborted" in srow


def test_codex_arm_end_to_end(tmp_path):
    cfg = EconomizerConfig(
        injection=InjectionConfig(enabled=False),
        tooling=ToolingConfig(codex_untool=True),
        spiral=SpiralConfig(codex_spiral=True))
    econ = Economizer(cfg, agent="codex", repo="pydata/xarray",
                      results_root=str(tmp_path), fresh=True)
    task = {"instance_id": "x-1", "problem_statement": "fix concat"}
    assert econ.build_tools(task, 0) == []          # task 1 full contract
    rk = econ.run_kwargs(task)
    assert "codex_spiral_policy" in rk

    rd = str(tmp_path / "run"); os.makedirs(rd)
    write_codex_transcript(os.path.join(rd, "transcript.jsonl"),
                           [{"type": "command_execution", "command": "pytest"}])
    trial_masks = []
    for i in range(6):
        task = {"instance_id": f"x-{i}", "problem_statement": "fix concat"}
        mask = econ.build_tools(task, i)
        if i < 3:
            assert mask == []
        else:
            assert "web" in mask
            trial_masks.append(mask)
        row = {"resolved": True, "model": "gpt-6.1-sol", "requests": 4,
               "num_turns": 30, "tokens_total": 5000 if i < 3 else 4750,
               "input_tokens": 500 if i < 3 else 250, "cache_read_input_tokens": 4000,
               "cache_creation_input_tokens": 200, "output_tokens": 300,
               "codex_spiral_nudges": 0}
        srow = {"resolved": True, "num_turns": 30}
        opt, reason = econ.after_task(task, row, rd, srow)
        assert reason is None
        assert srow["codex_untool_active"] is True
        assert srow["codex_spiral_active"] is True
        assert srow["codex_untool_cost_guard"]["phase"] == (
            "baseline" if i < 2 else "trial" if i < 5 else "retained")
    assert all(mask == trial_masks[0] for mask in trial_masks)
    guard = srow["codex_untool_cost_guard"]
    assert guard["baseline"]["effective_tokens"] == 3 * 2450
    assert guard["trial"]["effective_tokens"] == 3 * 2200
    assert guard["basis"]["per_request"] is True
    assert "web" in econ.build_tools(task, 6)


def test_codex_trimech_end_to_end(tmp_path):
    """All three Codex mechanisms driven through the facade at once, offline.

    Built from the REGISTRY rather than a hand-written config, so this is the arm a lane
    actually runs. What it guards is that the three do not contend: injection renders a
    prompt prefix, untooling freezes a capability mask, and the spiral stage hands the
    runner its policy -- on the same task, in one `run_kwargs`, with all three telemetry
    families landing on one stream row.
    """
    from te_bench.arms import ARMS
    calls = []

    def runner(prompt, model="agent"):
        calls.append(prompt)
        return '{"extractors":[],"facts":[]}', 3.0

    econ = Economizer(ARMS["codex-trimech"].cfg(), agent="codex",
                      repo="pydata/xarray", results_root=str(tmp_path), fresh=True,
                      runner=runner, adapter=SwebenchAdapter())
    task = {"instance_id": "x-1", "problem_statement": "fix concat"}

    # one run_kwargs carries BOTH codex mechanisms' runner arguments, and they do not
    # collide -- the capability mask and the spiral policy are separate keys.
    rk = econ.run_kwargs(task)
    assert "codex_config_capabilities" in rk and "codex_spiral_policy" in rk
    assert econ.build_tools(task, 0) == []          # task 1 runs the full contract

    # ten tasks, because the composed profile proposes at cadence 10 -- fewer would leave
    # the paid half of the arm untested.
    total = 0
    for i in range(10):
        task = {"instance_id": f"x-{i}", "problem_statement": "fix concat"}
        # the per-task order the driver uses: prefix -> mask -> run_kwargs -> run -> hook
        econ.injection_prefix(task)
        mask = econ.build_tools(task, i)
        assert ("web" in mask) == (i >= 3)
        econ.run_kwargs(task)
        rd = str(tmp_path / f"run{i}"); os.makedirs(rd)
        write_codex_transcript(os.path.join(rd, "transcript.jsonl"), [
            {"type": "file_change", "changes": [{"path": "xarray/core/concat.py",
                                                 "kind": "update"}]},
            {"type": "command_execution",
             "command": '/bin/bash -lc "python -m pytest xarray/tests/test_concat.py -q"',
             "aggregated_output": "1 passed", "exit_code": 0}])
        row = {"resolved": True, "model": "gpt-6.1-sol", "requests": 4,
               "num_turns": 30, "tokens_total": 5000 if i < 3 else 4750,
               "input_tokens": 500 if i < 3 else 250, "cache_read_input_tokens": 4000,
               "cache_creation_input_tokens": 200, "output_tokens": 300,
               "codex_spiral_nudges": 0, "codex_spiral_tool_turns": 12}
        srow = {"resolved": True, "num_turns": 30}
        opt, reason = econ.after_task(task, row, rd, srow)
        total += opt
        # all three mechanisms reported on the same row
        assert srow["codex_untool_active"] is True
        assert srow["codex_spiral_active"] is True
        assert srow["codex_untool_cost_guard"]["phase"] == (
            "baseline" if i < 2 else "trial" if i < 5 else "retained")
        assert "manual_facts" in srow
        assert reason and reason.startswith("envfacts")      # the composed profile

    # the proposer's spend crossed the ledger seam rather than being invisible
    assert calls and total == 3.0 * len(calls)
    # untooling still learns across tasks with the other two active
    assert "web" in econ.build_tools(task, 10)
