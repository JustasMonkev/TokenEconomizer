import copy

import pytest

from tokeneconomizer.toolprune import codex as cu


UNUSED = {"attempted": {}, "completed": {}, "failed": {}}


def cost(fresh=1000, cached=0, output=0, **extra):
    return {"model": "gpt-6.1-sol", "input_tokens": fresh,
            "cache_read_input_tokens": cached, "cache_creation_input_tokens": 0,
            "output_tokens": output, **extra}


def finish(profile, name, row=None, resolved=True, **kwargs):
    selection = profile.begin_task(name, **kwargs)
    record = profile.observe(UNUSED, resolved=resolved, cost_row=row)
    return selection, record


def warm(profile, row=None, **kwargs):
    for i in range(3):
        selection, record = finish(profile, f"baseline-{i}", row or cost(), **kwargs)
        assert selection["logical_capabilities"] == []
    return record


def test_missing_cost_never_promotes_unused_capabilities(tmp_path):
    profile = cu.CodexUntoolProfile("a/b", str(tmp_path), fresh=True)
    for i in range(6):
        selected, record = finish(profile, str(i))
        assert selected["logical_capabilities"] == []
    assert record["cost_guard"]["reason"] == "incomplete-cost-usage"


def test_trial_batches_the_mask_then_keeps_it_stable(tmp_path):
    profile = cu.CodexUntoolProfile("a/b", str(tmp_path), fresh=True)
    warm(profile)
    masks = []
    for i in range(6):
        selected, record = finish(profile, f"trial-{i}", cost(fresh=500))
        masks.append(selected["logical_capabilities"])
    assert masks[0]
    assert all(mask == masks[0] for mask in masks)
    assert "deferred_wait" not in masks[0]
    assert record["cost_guard"]["phase"] == "retained"


def test_lower_raw_tokens_with_higher_cached_cost_roll_back(tmp_path):
    profile = cu.CodexUntoolProfile("a/b", str(tmp_path), fresh=True)
    warm(profile, cost(fresh=100, cached=1900))
    for i in range(3):
        _, record = finish(profile, f"trial-{i}", cost(fresh=600, cached=400))
    assert record["cost_guard"]["decision"] == "rollback"
    assert record["cost_guard"]["reason"] == "no-observed-cost-improvement"
    assert record["mask_after"] == []
    for i in range(4):
        selected, record = finish(profile, f"after-{i}", cost(fresh=1))
        assert selected["logical_capabilities"] == []
        assert record["cost_guard"]["decision"] == "rollback"


@pytest.mark.parametrize("bad", [
    {}, cost(model="unknown-model"), cost(input_tokens=-1), cost(output_tokens=float("nan")),
    cost(requests=0), cost(usage_incomplete=True), cost(is_error=True),
])
def test_invalid_or_unpriced_cost_cannot_form_a_baseline(tmp_path, bad):
    profile = cu.CodexUntoolProfile("a/b", str(tmp_path), fresh=True)
    for i in range(4):
        selected, record = finish(profile, str(i), bad)
        assert selected["logical_capabilities"] == []
    assert record["cost_guard"]["phase"] == "baseline"


def test_failed_trial_restores_immediately_without_good_usage(tmp_path):
    profile = cu.CodexUntoolProfile("a/b", str(tmp_path), fresh=True)
    warm(profile)
    _, record = finish(profile, "failed", resolved=False)
    assert record["cost_guard"]["reason"] == "task-failed"
    assert record["mask_after"] == []


def test_request_normalized_regression_prevents_retention(tmp_path):
    profile = cu.CodexUntoolProfile("a/b", str(tmp_path), fresh=True)
    warm(profile, cost(requests=10))
    for i in range(3):
        _, record = finish(profile, f"trial-{i}", cost(fresh=900, requests=1))
    assert record["mask_after"] == []
    assert record["cost_guard"]["decision"] == "rollback"


@pytest.mark.parametrize("row,resolved", [
    (cost(model="gpt-5.6-sol"), True),
    (cost(session_completed=True), None),
    (cost(requests=1), True),
])
def test_model_and_quality_basis_changes_are_skipped_not_mixed(tmp_path, row, resolved):
    profile = cu.CodexUntoolProfile("a/b", str(tmp_path), fresh=True)
    warm(profile)
    candidate = profile.state["cost_guard"]["candidate"]
    _, record = finish(profile, "changed-basis", row, resolved=resolved)
    assert record["mask_after"] == candidate
    assert record["cost_guard"]["phase"] == "trial"
    assert record["cost_guard"]["reason"] == "cost-basis-changed"
    assert record["cost_guard"]["trial"]["samples"] == 0


@pytest.mark.parametrize("bad,resolved", [
    (cost(usage_incomplete=True), True),          # a resumed or mixed-mask session
    (cost(session_completed=False), None),        # a turn left unfinished
    ({k: v for k, v in cost().items() if k != "model"}, True),
])
def test_one_unclean_trial_session_is_skipped_and_the_trial_goes_on(tmp_path, bad, resolved):
    profile = cu.CodexUntoolProfile("a/b", str(tmp_path), fresh=True)
    warm(profile)
    candidate = profile.state["cost_guard"]["candidate"]
    _, record = finish(profile, "unclean", bad, resolved=resolved)
    assert record["cost_guard"]["phase"] == "trial"
    assert record["mask_after"] == candidate
    for i in range(3):
        _, record = finish(profile, f"trial-{i}", cost(fresh=500))
    assert record["cost_guard"]["phase"] == "retained"
    assert record["mask_after"] == candidate


def test_a_trial_that_stays_unmeasured_is_restored(tmp_path):
    profile = cu.CodexUntoolProfile("a/b", str(tmp_path), fresh=True)
    warm(profile)
    for i in range(cu.MAX_TRIAL_SKIPS - 1):
        _, record = finish(profile, f"unclean-{i}", cost(usage_incomplete=True))
        assert record["cost_guard"]["phase"] == "trial"
    _, record = finish(profile, "last", cost(usage_incomplete=True))
    assert record["cost_guard"]["reason"] == "trial-unmeasured"
    assert record["mask_after"] == []


def test_a_failed_baseline_session_keeps_the_earlier_samples(tmp_path):
    profile = cu.CodexUntoolProfile("a/b", str(tmp_path), fresh=True)
    for i in range(2):
        finish(profile, f"baseline-{i}", cost())
    _, record = finish(profile, "failed", cost(), resolved=False)
    assert record["cost_guard"]["reason"] == "task-failed"
    assert record["cost_guard"]["baseline"]["samples"] == 2
    _, record = finish(profile, "baseline-2", cost())
    assert record["cost_guard"]["phase"] == "trial"


def test_genuinely_completed_ungraded_sessions_have_a_distinct_basis(tmp_path):
    profile = cu.CodexUntoolProfile("a/b", str(tmp_path), fresh=True)
    for i in range(3):
        finish(profile, f"baseline-{i}", cost(session_completed=True), resolved=None)
    for i in range(3):
        _, record = finish(profile, f"trial-{i}", cost(fresh=500, session_completed=True), resolved=None)
    assert record["cost_guard"]["phase"] == "retained"
    assert record["cost_guard"]["basis"]["quality"] == "completed-session"


def test_actual_mask_must_match_candidate_before_counting_trial(tmp_path):
    profile = cu.CodexUntoolProfile("a/b", str(tmp_path), fresh=True)
    warm(profile, available_capabilities=["goals", "collaboration"])
    profile.begin_task("not-applied", available_capabilities=["goals", "collaboration"])
    profile._pending["logical_capabilities"] = []
    record = profile.observe(UNUSED, resolved=True, cost_row=cost(fresh=1))
    assert record["cost_guard"]["reason"] == "candidate-not-applied"
    assert record["cost_guard"]["trial"]["samples"] == 0
    assert set(record["mask_after"]) == {"goals", "collaboration"}


def test_hints_and_completed_uses_still_restore_tools(tmp_path):
    profile = cu.CodexUntoolProfile("a/b", str(tmp_path), fresh=True)
    warm(profile)
    selected = profile.begin_task("needs-image", problem_statement="Fix screenshot rendering")
    assert "image_media" not in selected["logical_capabilities"]
    used = copy.deepcopy(UNUSED)
    used["attempted"]["image_media"] = used["completed"]["image_media"] = 1
    record = profile.observe(used, resolved=True, cost_row=cost(fresh=500))
    assert "image_media" not in record["mask_after"]
    assert "image_media" in profile.state["ever_completed"]


def test_window_survives_reload_but_old_policy_state_is_invalidated(tmp_path):
    profile = cu.CodexUntoolProfile("a/b", str(tmp_path), fresh=True)
    warm(profile)
    finish(profile, "trial-0", cost(fresh=500))
    restored = cu.CodexUntoolProfile("a/b", str(tmp_path))
    assert restored.state["cost_guard"]["trial"]["samples"] == 1
    for i in range(2):
        _, record = finish(restored, f"trial-{i+1}", cost(fresh=500))
    assert record["cost_guard"]["phase"] == "retained"
    restored.state["version"] = "codex-recent-experience-untool-indirect-use-core"
    restored.save()
    reset = cu.CodexUntoolProfile("a/b", str(tmp_path))
    assert reset.state["current_pruned"] == []
    assert reset.state["cost_guard"]["phase"] == "baseline"


def test_retained_mask_is_restored_after_a_later_failure(tmp_path):
    profile = cu.CodexUntoolProfile("a/b", str(tmp_path), fresh=True)
    warm(profile)
    for i in range(3):
        finish(profile, f"trial-{i}", cost(fresh=500))
    _, record = finish(profile, "later-failure", cost(), resolved=False)
    assert record["mask_after"] == []
    assert record["cost_guard"]["decision"] == "rollback"


def test_stage_forwards_actual_cost_and_available_capabilities(tmp_path):
    from tokeneconomizer.config import ToolingConfig
    from tokeneconomizer.toolprune.stage import ToolingStage

    stage = ToolingStage(ToolingConfig(codex_untool=True), "codex", "agent",
                         repo="a/b", results_root=str(tmp_path), fresh=True)
    (tmp_path / "transcript.jsonl").write_text("")
    for i in range(3):
        task = {"instance_id": f"task-{i}", "codex_available_capabilities": ["goals"]}
        assert stage.build_tools(task, i) == []
        result = {"resolved": True}
        stage.after_task(task, cost(), str(tmp_path), result)
    assert result["codex_untool_cost_guard"]["phase"] == "trial"
    stage.build_tools({"instance_id": "next", "codex_available_capabilities": ["goals"]}, 3)
    assert stage.run_kwargs({})["codex_config_capabilities"] == ["goals"]


@pytest.mark.parametrize("row", [cost(), None])
def test_existing_applied_mask_requests_baseline_restore_without_counting_it(tmp_path, row):
    profile = cu.CodexUntoolProfile("a/b", str(tmp_path), fresh=True)
    profile.begin_task("old-applied-mask")
    profile._pending["logical_capabilities"] = ["goals"]
    record = profile.observe(UNUSED, resolved=True, cost_row=row)
    assert record["cost_guard"]["decision"] == "restore-baseline"
    assert record["cost_guard"]["phase"] == "baseline"
    assert record["cost_guard"]["baseline"]["samples"] == 0
    _, next_record = finish(profile, "restored", cost())
    assert next_record["cost_guard"]["baseline"]["samples"] == 1


def test_scheduled_probes_wait_until_the_stable_trial_finishes(tmp_path):
    profile = cu.CodexUntoolProfile("a/b", str(tmp_path), fresh=True)
    warm(profile)
    profile.state["next_probe_at"] = 0
    for i in range(3):
        selected, _ = finish(profile, f"trial-{i}", cost(fresh=500))
        assert selected["probe_capability"] is None
    assert profile.begin_task("retained-probe")["probe_capability"] is not None


def test_a_trial_mask_applied_in_part_shrinks_the_trial_to_what_was_applied(tmp_path):
    profile = cu.CodexUntoolProfile("a/b", str(tmp_path), fresh=True)
    warm(profile)
    finish(profile, "trial-0", cost(fresh=500))
    selection = profile.begin_task("given-back")
    assert len(selection["logical_capabilities"]) > 1
    kept = [selection["logical_capabilities"][0]]
    profile._pending["logical_capabilities"] = kept
    record = profile.observe(UNUSED, resolved=True, cost_row=cost(fresh=500))
    assert record["cost_guard"]["phase"] == "trial"
    assert record["cost_guard"]["candidate"] == kept
    assert record["cost_guard"]["trial"]["samples"] == 1    # earlier samples ran a larger mask
    assert record["mask_after"] == kept
    for i in range(2):
        _, record = finish(profile, f"trial-{i}", cost(fresh=500))
    assert record["cost_guard"]["phase"] == "retained" and record["mask_after"] == kept


def test_a_mask_the_trial_did_not_choose_is_skipped(tmp_path):
    profile = cu.CodexUntoolProfile("a/b", str(tmp_path), fresh=True)
    warm(profile)
    candidate = profile.state["cost_guard"]["candidate"]
    profile.begin_task("foreign")
    profile._pending["logical_capabilities"] = ["deferred_wait"]
    record = profile.observe(UNUSED, resolved=True, cost_row=cost())
    assert record["cost_guard"]["reason"] == "candidate-mask-changed"
    assert record["cost_guard"]["phase"] == "trial"
    assert record["cost_guard"]["trial"]["samples"] == 0
    assert record["mask_after"] == candidate


def test_using_a_package_that_was_given_back_does_not_end_the_trial(tmp_path):
    profile = cu.CodexUntoolProfile("a/b", str(tmp_path), fresh=True)
    warm(profile)
    candidate = profile.state["cost_guard"]["candidate"]
    given = candidate[0]
    rest = [c for c in candidate if c != given]
    profile.begin_task("given-back-and-used",
                       available_capabilities=[c for c in cu.CANDIDATE_CAPABILITIES if c != given])
    profile._pending["logical_capabilities"] = rest
    used = {"attempted": {given: 1}, "completed": {given: 1}, "failed": {}}
    record = profile.observe(used, resolved=True, cost_row=cost(fresh=500))
    assert record["cost_guard"]["phase"] == "trial"
    assert record["cost_guard"]["candidate"] == rest
    assert record["mask_after"] == rest


def test_a_trial_given_back_whole_starts_a_fresh_baseline(tmp_path):
    profile = cu.CodexUntoolProfile("a/b", str(tmp_path), fresh=True)
    warm(profile)
    candidate = profile.state["cost_guard"]["candidate"]
    profile.begin_task("all-given-back", available_capabilities=[
        c for c in cu.CANDIDATE_CAPABILITIES if c not in candidate])
    record = profile.observe(UNUSED, resolved=True, cost_row=cost())
    assert record["cost_guard"]["phase"] == "baseline"
    assert record["cost_guard"]["reason"] == "candidate-given-back"
    assert record["cost_guard"]["baseline"]["samples"] == 0
    assert record["mask_after"] == []


@pytest.mark.parametrize("field", ["version", "contract_fingerprint"])
def test_policy_migration_preserves_only_known_completed_use_protections(tmp_path, field):
    profile = cu.CodexUntoolProfile("a/b", str(tmp_path), fresh=True)
    warm(profile)
    profile.state[field] = "older-policy"
    profile.state["ever_completed"] = ["goals", "deferred_wait", "unknown", {"bad": "shape"}, 4]
    profile.save()
    restored = cu.CodexUntoolProfile("a/b", str(tmp_path))
    assert restored.state["ever_completed"] == ["deferred_wait", "goals"]
    assert restored.state["current_pruned"] == []
    assert restored.state["tasks_completed"] == 0
    assert not any(restored.state["loaded_exposures"].values())
    assert restored.state["cost_guard"]["baseline"]["samples"] == 0
    warm(restored)
    assert "goals" not in restored.begin_task("new-trial")["logical_capabilities"]


def test_policy_migration_never_imports_another_repositories_protections(tmp_path):
    profile = cu.CodexUntoolProfile("a/b", str(tmp_path), fresh=True)
    profile.state.update(repo="other/repository", version="older-policy",
                         ever_completed=["goals"], current_pruned=["web"])
    profile.save()
    restored = cu.CodexUntoolProfile("a/b", str(tmp_path))
    assert restored.state["ever_completed"] == []
    assert restored.state["current_pruned"] == []


def test_a_task_that_asks_for_a_trial_package_is_skipped_not_shrunk(tmp_path):
    profile = cu.CodexUntoolProfile("a/b", str(tmp_path), fresh=True)
    warm(profile)
    candidate = profile.state["cost_guard"]["candidate"]
    assert "image_media" in candidate
    selection, record = finish(profile, "hinted", cost(fresh=500),
                               problem_statement="Fix the screenshot in the attached image.png")
    assert "image_media" not in selection["logical_capabilities"]
    assert record["cost_guard"]["reason"] == "candidate-hinted"
    assert record["cost_guard"]["candidate"] == candidate
    assert record["mask_after"] == candidate
