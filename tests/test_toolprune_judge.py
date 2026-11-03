"""The per-task tool judge, in its full and its compact answer forms."""
import json
from dataclasses import replace

import pytest

from tokeneconomizer.config import EconomizerConfig, ToolingConfig
from tokeneconomizer.toolprune import tools
from tokeneconomizer.toolprune.stage import ToolingStage

CATALOG = {"Bash": "run a shell command", "WebFetch": "fetch a URL",
           "NotebookEdit": "edit a notebook", "Write": "write a file"}


def _runner(answer, seen=None):
    def run(prompt, model):
        if seen is not None:
            seen.append(prompt)
        return answer, 123.0
    return run


def test_compact_judge_maps_the_drop_list_onto_the_catalog():
    seen = []
    out, tok = tools.judge_tools_for_task(
        "Fix the parser.", CATALOG, _runner('{"drop": ["WebFetch", "NotebookEdit"]}', seen),
        compact=True)
    assert out == {"Bash": True, "WebFetch": False, "NotebookEdit": False, "Write": True}
    assert tok == 123.0
    assert tools.TOOL_JUDGE_COMPACT_ANSWER in seen[0]
    assert '"reason"' not in seen[0] and '"load"' not in seen[0]


def test_compact_judge_never_drops_a_protected_tool():
    out, _ = tools.judge_tools_for_task(
        "Fix the parser.", CATALOG, _runner('{"drop": ["Bash", "Write", "WebFetch"]}'),
        compact=True)
    assert out["Bash"] is True and out["Write"] is True and out["WebFetch"] is False


def test_compact_judge_ignores_names_outside_the_catalog():
    out, _ = tools.judge_tools_for_task(
        "Fix the parser.", CATALOG, _runner('{"drop": ["Telepathy", "WebFetch"]}'),
        compact=True)
    assert set(out) == set(CATALOG) and out["WebFetch"] is False


@pytest.mark.parametrize("raw", [
    "", "no json here", '{"load": ["Bash"]}', '{"drop": "WebFetch"}',
    '["WebFetch"]', '{"drop": null}', '{"drop": {"WebFetch": true}}',
])
def test_an_unusable_compact_answer_keeps_everything_loaded(raw):
    out, tok = tools.judge_tools_for_task("Fix it.", CATALOG, _runner(raw), compact=True)
    assert out == {}
    assert tok == 123.0


def test_the_full_form_is_unchanged():
    answer = json.dumps({"Bash": {"load": True, "reason": "x"},
                         "WebFetch": {"load": False, "reason": "y"}})
    seen = []
    out, _ = tools.judge_tools_for_task("Fix it.", CATALOG, _runner(answer, seen))
    assert out == {"Bash": True, "WebFetch": False}
    assert seen[0].startswith(tools.TOOL_JUDGE_DOMAIN)


@pytest.mark.parametrize("domain", [
    "Classify tools. Return ONLY JSON. Never drop WebFetch: documentation verification is mandatory.",
    "Return ONLY the classification. Keep WebFetch. Return ONLY JSON. Keep NotebookEdit too.",
    "Keep WebFetch; this custom policy has no response-format marker.",
])
def test_compact_judge_preserves_every_custom_domain_policy(domain):
    seen = []
    tools.judge_tools_for_task("Fix it.", CATALOG, _runner('{"drop": []}', seen),
                               domain=domain, compact=True)
    assert seen[0].startswith(domain + "\n")
    assert "overrides only earlier response-format instructions" in seen[0]
    assert tools.TOOL_JUDGE_COMPACT_ANSWER in seen[0]


def test_full_judge_preserves_custom_domain_unchanged():
    domain = "Return ONLY JSON. Keep WebFetch: external verification is mandatory."
    seen = []
    tools.judge_tools_for_task("Fix it.", CATALOG, _runner('{}', seen), domain=domain)
    assert seen[0].startswith(domain + "\nISSUE:")


def test_compact_flag_preserves_existing_positional_tooling_config():
    cfg = ToolingConfig(3, 0, True, True)
    assert cfg.codex_untool is True
    assert cfg.judge_compact is False


def test_lean_arm_changes_only_the_judge_response_format():
    from experiments.te_bench.arms import ARMS, ARM_ORDER

    full = ARMS["trimech"].cfg()
    lean = ARMS["trimech-lean"].cfg()
    assert "trimech-lean" in ARM_ORDER
    assert lean.injection == full.injection
    assert lean.spiral == full.spiral
    assert lean.tooling == replace(full.tooling, judge_compact=True)
    assert ARMS["trimech-lean"].env == ARMS["trimech"].env
    assert ARMS["trimech-lean"].scaffold == ARMS["trimech"].scaffold


def test_judge_compact_is_read_from_args_and_reaches_the_judge(monkeypatch, tmp_path):
    class Args:
        untool_evidence = 3
        untool_judge = True
        untool_judge_compact = True
    cfg = EconomizerConfig.from_args(Args())
    assert cfg.tooling.judge_compact is True
    assert ToolingConfig().judge_compact is False

    calls = []

    def fake_judge(problem, cand, runner, model="agent", compact=False):
        calls.append(compact)
        return {}, 0

    class Adapter:
        def problem_statements(self):
            return {"t-1": "Fix it."}

        def protected_tools(self):
            return tools.PROTECTED_TOOLS

    monkeypatch.setattr("tokeneconomizer.toolprune.stage.judge_tools_for_task", fake_judge)
    stage = ToolingStage(cfg.tooling, "claude", "agent", judge_tool_runner=_runner(""),
                         adapter=Adapter(), repo="a/b", results_root=str(tmp_path), fresh=True)
    stage.catalog_tools = set(CATALOG)
    stage.build_tools({"instance_id": "t-1"}, 1)
    assert calls == [True]
