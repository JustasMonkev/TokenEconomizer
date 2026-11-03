# Tool pruning

How tool pruning (see the [main README](../../README.md)) decides what to drop, per agent.

- **Evidence pruning** (Claude Code, Copilot CLI; `stage.ToolingStage`). After each session, a
  loaded tool's unused streak grows or resets. In the experiments a tool is proposed once its
  streak reaches `evidence` sessions. The plugin asks for more (`plugin/fold.py`): unused over
  `toolprune_evidence` prompts across at least `toolprune_min_sessions` sessions, or over
  `toolprune_sessions` sessions, and three times as long for `NotebookEdit`. A protected
  work-surface tool (`tools.PROTECTED_TOOLS`) or one ever seen in use is never proposed. In the
  experiments only, a per-task judge (`tools.judge_tools_for_task`) can also drop tools the
  task text makes irrelevant, its cost charged to the ledger; the plugin has no task text and
  never calls a model to prune. `ToolingConfig.judge_compact=True` requests a bare DROP list;
  unnamed and protected tools stay loaded. Custom-domain policies are preserved, and the
  full response format remains the default. The `trimech-lean` experiment arm enables this
  option; it is not a paper arm or a Codex-native capability judge.
- **Capability pruning** (Codex; `codex.CodexUntoolProfile`). Codex's unit is a capability
  package (collaboration, goals, image/media, ...) switched off in its config. A package is
  eligible only after enough loaded-but-unused sessions; a single completed use protects it.
  Three comparable, priced baseline sessions precede one stable three-session trial of the
  eligible batch. The mask is retained only when average cache-weighted cost improves, both
  per session and per request when actual request counts exist. Otherwise it is restored,
  with no further demotions in that profile. This is an observational guard, not a paired
  counterfactual or a guarantee of future savings. The plugin also requires the prompt-and-session
  evidence above (three times as long for image/media).

The cost guard needs an explicit priced model and complete token buckets. Graded successes
and completed-but-ungraded interactive sessions are separate comparison bases. A failed task
or use of a trial capability restores tools. A session that cannot be compared (mixed-mask
resume, mixed-model or incomplete usage, an unfinished turn, another price basis) is skipped,
not counted; a trial unmeasured for `MAX_TRIAL_SKIPS` sessions is restored. A trial mask
applied in part, as when a package is given back, shrinks the trial and restarts its samples. Old policy masks are restored before a new baseline is collected, but historical
completed-use protections survive. Restoration only removes settings owned by TokenEconomizer
and retries after temporary write failures. `untool.json` exposes the guard's phase and reason.

Direct callers pass normalized fresh input, cached input, cache writes, output, and `model`
through `observe(..., cost_row=row)`. For ungraded sessions, `session_completed=True` must be
backed by a terminal event. `ToolingStage.after_task` forwards runner rows automatically.
`begin_task(..., available_capabilities=...)` restricts trials to capabilities the host can
actually apply. See [the decision record](../../docs/decisions/001-cache-aware-codex.md).

This package decides *what* to drop. Writing the decision into an agent's own settings file is
the plugin's job (`tokeneconomizer/plugin/denylist.py`): by itself under `toolprune=auto` (the
default), once the repository's sessions have been idle `toolprune_quiet_minutes`, or on
`toolprune apply` under `propose`; either way it takes effect from the next session.

| Module | |
|---|---|
| `stage.py` | `ToolingStage`: evidence across sessions, the proposal, the Codex mask |
| `codex.py` | `CodexUntoolProfile`: Codex's evidence-gated capability controller |
| `tools.py` | the protected and inapplicable tool sets; the experiments' per-task tool judge |
