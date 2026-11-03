import pytest

from tokeneconomizer.plugin import denylist
from tests.test_live_hooks import call, env, repo, store_for


def _start(env, repo, sid, source="startup", host="codex"):
    return call(env, {"hook_event_name": "SessionStart", "session_id": sid,
                      "source": source, "cwd": repo}, host)


def _own_mask(env, repo, mask):
    store = store_for(env, repo, "codex")
    denylist._record(store, mask, denylist.settings_path(repo, "codex"))
    return store


@pytest.mark.parametrize("mask", [[], ["goals", "collaboration"]])
def test_codex_start_freezes_owned_mask_not_later_settings(env, repo, mask):
    store = _own_mask(env, repo, mask)
    _start(env, repo, "s")
    launched = store.read("sessions/s.json")
    assert launched["codex_launch_mask"] == sorted(mask)
    assert launched["codex_mask_mixed"] is False
    _own_mask(env, repo, ["image_media"])
    assert store.read("sessions/s.json")["codex_launch_mask"] == sorted(mask)


def test_same_mask_resume_preserves_history_and_comparability(env, repo):
    store = _own_mask(env, repo, ["goals"])
    _start(env, repo, "s")
    session = store.read("sessions/s.json")
    session.update(turns=[{"number": 1}], injected_ids=["fact"], judge_tokens=23)
    store.write("sessions/s.json", session)
    _start(env, repo, "s", source="resume")
    resumed = store.read("sessions/s.json")
    assert resumed["codex_launch_mask"] == ["goals"]
    assert resumed["codex_mask_mixed"] is False
    assert resumed["started"] == session["started"]
    assert resumed["turns"] == [{"number": 1}]
    assert resumed["injected_ids"] == ["fact"]
    assert resumed["judge_tokens"] == 23


def test_changed_mask_resume_is_permanently_incomparable(env, repo):
    store = _own_mask(env, repo, [])
    _start(env, repo, "s")
    _own_mask(env, repo, ["goals"])
    _start(env, repo, "s", source="resume")
    assert store.read("sessions/s.json")["codex_mask_mixed"] is True
    _own_mask(env, repo, [])
    _start(env, repo, "s", source="resume")
    resumed = store.read("sessions/s.json")
    assert resumed["codex_launch_mask"] == []
    assert resumed["codex_mask_mixed"] is True


@pytest.mark.parametrize("source", ["startup", "resume", "compact"])
def test_legacy_session_never_gains_a_retroactive_launch_mask(env, repo, source):
    store = _own_mask(env, repo, ["goals"])
    store.write("sessions/legacy.json", {"session_id": "legacy", "started": 1,
                                         "turns": [{"number": 1}]})
    _start(env, repo, "legacy", source=source)
    session = store.read("sessions/legacy.json")
    assert "codex_launch_mask" not in session
    assert session["codex_mask_mixed"] is True
    assert session["started"] == 1
    assert session["turns"] == [{"number": 1}]


@pytest.mark.parametrize("source", ["resume", "fork", "compact"])
def test_unknown_existing_conversation_has_no_assumed_launch_mask(env, repo, source):
    store = _own_mask(env, repo, ["goals"])
    _start(env, repo, "unknown", source=source)
    session = store.read("sessions/unknown.json")
    assert "codex_launch_mask" not in session
    assert session["codex_mask_mixed"] is True


def test_other_hosts_do_not_gain_codex_mask_metadata(env, repo):
    _start(env, repo, "c", host="claude")
    session = store_for(env, repo, "claude").read("sessions/c.json")
    assert "codex_launch_mask" not in session
    assert "codex_mask_mixed" not in session
