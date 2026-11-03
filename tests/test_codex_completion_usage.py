import json

import pytest

from tokeneconomizer.trace import codex_session_usage


@pytest.mark.parametrize("tail,completed", [
    ([], False),
    (["task_complete"], True),
    (["task_complete", "task_started"], False),
    (["task_complete", "turn_aborted"], False),
    (["task_complete", "turn_context"], False),
    (["turn_aborted", "task_started", "task_complete"], True),
])
def test_completion_is_last_turn_evidence_not_a_success_grade(tmp_path, tail, completed):
    rows = [{"type": "turn_context", "payload": {"model": "gpt-6.1-sol"}},
            {"type": "event_msg", "payload": {"type": "token_count", "info": {
                "total_token_usage": {"input_tokens": 1500, "cached_input_tokens": 500,
                                      "output_tokens": 100}}}}]
    for event in tail:
        rows.append({"type": "turn_context", "payload": {}} if event == "turn_context" else
                    {"type": "event_msg", "payload": {"type": event}})
    path = tmp_path / "rollout.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    usage = codex_session_usage(str(path))
    assert usage["session_completed"] is completed
    assert usage["input_tokens"] == 1000
    assert usage["cache_read_input_tokens"] == 500
    assert "resolved" not in usage


@pytest.mark.parametrize("usage,models", [
    ({"input_tokens": 10, "cached_input_tokens": 100, "output_tokens": 1}, ["gpt-6.1-sol"]),
    ({"input_tokens": 100, "cached_input_tokens": 10}, ["gpt-6.1-sol"]),
    ({"input_tokens": 100, "cached_input_tokens": 10, "output_tokens": True}, ["gpt-6.1-sol"]),
    ({"input_tokens": 100, "cached_input_tokens": 10, "output_tokens": 1}, ["gpt-5.6-sol", "gpt-6.1-sol"]),
])
def test_mixed_or_incomplete_costs_cannot_authorize_pruning(tmp_path, usage, models):
    rows = [{"type": "turn_context", "payload": {"model": model}} for model in models]
    rows += [{"type": "event_msg", "payload": {"type": "token_count", "info": {
        "total_token_usage": usage}}}, {"type": "event_msg", "payload": {"type": "task_complete"}}]
    path = tmp_path / "rollout.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    measured = codex_session_usage(str(path))
    assert measured["session_completed"] is True
    assert measured["usage_incomplete"] is True
    assert measured["input_tokens"] is None
