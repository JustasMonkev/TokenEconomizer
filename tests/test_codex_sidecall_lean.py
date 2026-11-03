import json
import os
from pathlib import Path
import subprocess

import pytest

from tokeneconomizer.plugin import sidecall


pytestmark = pytest.mark.real_sidecalls


def _reply(text="OK", usage=None):
    return "\n".join(json.dumps(event) for event in (
        {"type": "item.completed", "item": {"type": "agent_message", "text": text}},
        {"type": "turn.completed", "usage": usage or {
            "input_tokens": 3400, "cached_input_tokens": 2000,
            "cache_write_input_tokens": 400, "output_tokens": 200,
            "reasoning_output_tokens": 100}},
    ))


def _cli(monkeypatch, *, supported=True, validation=0, servers=None, infer=None):
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if cmd[1:] == ["app-server", "--help"]:
            text = "--strict-config --print-startup-config" if supported else "--listen"
        elif cmd[1:] == ["features", "list"]:
            text = "\n".join(f"{key} stable true" for key in (
                "shell_tool", "apps", "plugins", "multi_agent", "goals", "view_image",
                "skip_host_skill_discovery", "memories", "image_generation"))
        elif cmd[1:4] == ["mcp", "list", "--json"]:
            assert "features.apps=false" in _overrides(cmd)
            assert "features.plugins=false" in _overrides(cmd)
            text = json.dumps(servers or [])
        elif cmd[1:4] == ["app-server", "--strict-config", "--print-startup-config"]:
            return subprocess.CompletedProcess(cmd, validation, stdout="{}", stderr="")
        elif cmd[1] == "exec":
            return infer(cmd, kwargs) if infer else subprocess.CompletedProcess(
                cmd, 0, stdout=_reply(), stderr="")
        else:
            raise AssertionError(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=text, stderr="")

    monkeypatch.setattr(sidecall.subprocess, "run", run)
    return calls


def _overrides(cmd):
    return [cmd[i + 1] for i, arg in enumerate(cmd[:-1]) if arg == "-c"]


def test_lean_call_preserves_routing_and_uses_only_temporary_overrides(monkeypatch, tmp_path):
    home = tmp_path / "codex home"
    home.mkdir()
    config = home / "config.toml"
    original = 'model_provider = "company"\nmodel = "configured-model"\n'
    config.write_text(original)
    skill = home / "skills" / ".system" / "example" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: example\n---\nExample skill")
    monkeypatch.setenv("CODEX_HOME", str(home))
    monkeypatch.setenv("COMPANY_API_KEY", "test-routing-key")
    seen = {}

    def infer(cmd, kwargs):
        overrides = _overrides(cmd)
        path = json.loads(next(x.split("=", 1)[1] for x in overrides
                               if x.startswith("model_instructions_file=")))
        seen.update(path=path, cwd=kwargs["cwd"])
        assert "do not use tools" in Path(path).read_text().lower()
        assert kwargs["env"]["CODEX_HOME"] == str(home)
        assert kwargs["env"]["COMPANY_API_KEY"] == "test-routing-key"
        assert kwargs["env"]["TE_SIDECALL"] == "1"
        assert "--ignore-user-config" not in cmd and "-m" not in cmd
        assert not any(x.startswith(("model_provider=", "model_providers.", "openai_base_url=")) for x in overrides)
        assert "features.shell_tool=false" in overrides
        assert "features.plugins=false" in overrides and "features.apps=false" in overrides
        assert "project_doc_max_bytes=0" in overrides and 'web_search="disabled"' in overrides
        assert any(str(skill) in x and "enabled=false" in x for x in overrides)
        assert "mcp_servers.acme-db.enabled=false" in overrides
        assert "--disable" in cmd and "hooks" in cmd
        assert cmd[cmd.index("-s") + 1] == "read-only"
        return subprocess.CompletedProcess(cmd, 0, stdout=_reply(), stderr="")

    calls = _cli(monkeypatch, servers=[{"name": "acme-db", "enabled": True}], infer=infer)
    text, cost = sidecall.codex_call("Return JSON", timeout=17)
    assert text == "OK" and cost is not None
    assert not Path(seen["path"]).exists() and not Path(seen["cwd"]).exists()
    assert config.read_text() == original
    assert sum(cmd[1] == "exec" for cmd, _ in calls) == 1
    assert 0 < calls[-1][1]["timeout"] <= 17


@pytest.mark.parametrize("supported,validation", [(False, 0), (True, 1)])
def test_unsupported_cli_falls_back_before_the_single_model_call(monkeypatch, supported, validation):
    calls = _cli(monkeypatch, supported=supported, validation=validation)
    assert sidecall.codex_call("Reply OK", model="gpt-6.1-sol") == ("OK", 3100)
    command = next(cmd for cmd, _ in calls if cmd[1] == "exec")
    assert _overrides(command) == []
    assert command[command.index("-m") + 1] == "gpt-6.1-sol"
    assert sum(cmd[1] == "exec" for cmd, _ in calls) == 1


def test_lean_call_prices_all_reported_usage_buckets(monkeypatch):
    _cli(monkeypatch)
    assert sidecall.codex_call("Reply OK", model="gpt-6.1-sol") == ("OK", 3100)


def test_server_names_that_cannot_be_overridden_safely_use_legacy_call(monkeypatch):
    calls = _cli(monkeypatch, servers=[{"name": "acme.db", "enabled": True}])
    assert sidecall.codex_call("Reply OK")[0] == "OK"
    command = next(cmd for cmd, _ in calls if cmd[1] == "exec")
    assert _overrides(command) == []


def test_model_failure_is_not_retried_and_is_reported(monkeypatch):
    noted = []
    monkeypatch.setattr(sidecall, "_no_reply", lambda host, out: noted.append((host, out.returncode)))
    calls = _cli(monkeypatch, infer=lambda cmd, kwargs: subprocess.CompletedProcess(
        cmd, 1, stdout="", stderr="Not logged in"))
    assert sidecall.codex_call("Reply OK") == ("", None)
    assert noted == [("codex", 1)]
    assert sum(cmd[1] == "exec" for cmd, _ in calls) == 1


def test_timeout_removes_temporary_instructions_and_is_not_retried(monkeypatch):
    directories = []

    def infer(cmd, kwargs):
        directories.append(kwargs["cwd"])
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    calls = _cli(monkeypatch, infer=infer)
    with pytest.raises(subprocess.TimeoutExpired):
        sidecall.codex_call("Reply OK", timeout=1)
    assert all(not os.path.exists(p) for p in directories)
    assert sum(cmd[1] == "exec" for cmd, _ in calls) == 1


def test_probes_and_inference_share_the_callers_timeout(monkeypatch):
    now = [0]
    monkeypatch.setattr(sidecall.time, "monotonic", lambda: now[0])
    calls = _cli(monkeypatch)
    original = sidecall.subprocess.run

    def elapsed(cmd, **kwargs):
        now[0] += 2
        return original(cmd, **kwargs)

    monkeypatch.setattr(sidecall.subprocess, "run", elapsed)
    assert sidecall.codex_call("Reply OK", timeout=12)[0] == "OK"
    assert calls[-1][1]["timeout"] == 4


def test_exhausted_probe_budget_cannot_start_a_model_call(monkeypatch):
    now = [0]
    monkeypatch.setattr(sidecall.time, "monotonic", lambda: now[0])
    commands = []

    def timed_out(cmd, **kwargs):
        commands.append(cmd)
        now[0] += kwargs["timeout"]
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(sidecall.subprocess, "run", timed_out)
    with pytest.raises(subprocess.TimeoutExpired):
        sidecall.codex_call("Reply OK", timeout=1)
    assert len(commands) == 1 and commands[0][1] != "exec"


def test_probe_diagnostics_are_not_printed_or_forwarded_to_model(monkeypatch, capsys):
    calls = _cli(monkeypatch)
    original = sidecall.subprocess.run

    def failed_validation(cmd, **kwargs):
        if cmd[1:4] == ["app-server", "--strict-config", "--print-startup-config"]:
            return subprocess.CompletedProcess(cmd, 1, stdout='{"detail":"private-config"}',
                                               stderr="private-diagnostic")
        return original(cmd, **kwargs)

    monkeypatch.setattr(sidecall.subprocess, "run", failed_validation)
    assert sidecall.codex_call("Reply OK")[0] == "OK"
    command = next(cmd for cmd, _ in calls if cmd[1] == "exec")
    assert command[-1] == "Reply OK" and _overrides(command) == []
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_skill_discovery_handles_symlinks_without_revisiting_cycles(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    home = tmp_path / "codex"
    root = home / "skills"
    skill = root / "example" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("Example skill")
    try:
        (root / "alias").symlink_to(skill.parent, target_is_directory=True)
        (skill.parent / "cycle").symlink_to(root, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink unavailable: {error}")
    assert sidecall._codex_skill_paths({"CODEX_HOME": str(home)}) == [str(skill)]
