# ADR 001 Stable cache aware Codex trials

## Status

Accepted for the local implementation, 2026-11-03.

## Context

The [SOL review](../../experiments/PR1_SOL_REVIEW.md) measured 26.9% fewer raw tokens but
39.8% higher cache-weighted cost with capability pruning on a small task stream. Successive
mask changes can undermine prefix reuse. Unused-tool evidence alone does not establish that
removing a capability saves money.

## Decision

Use three comparable priced baseline sessions, then freeze one eligible candidate batch for
three valid trial sessions. Retain only if average effective cost improves; where request
counts are available, the per-request average must improve too. Otherwise restore the mask
and stop further demotions for that profile. Keep exact model/pricing, grading basis, and
request-count availability consistent. Unknown costs cannot authorize pruning.

Interactive completion is not a success grade. Completed-ungraded sessions are compared only
with that same basis. Record the launch mask and do not count mixed-mask resumes,
mixed-model cumulative usage, malformed counters, unfinished sessions, or a session priced on
another basis, or a mask the trial did not choose: such a session is skipped, and only a
measured signal ends a trial. A trial with nine skipped sessions in all is restored. Count a
trial only when its mask was actually applied; a mask applied in part, or a package given
back, shrinks the trial to what remains and restarts its samples, and a trial given back
whole starts a fresh baseline. A failed baseline session is not counted but keeps earlier samples. Task hints and completed-use protections remain effective; known
failures restore trial/retained tools. Old policy state loses masks and cost evidence but
retains validated historical use protections. Restore only TE-owned settings, retrying
temporary failures.

Separately, reduce Codex optimizer context using validated per-call overrides, not
`--ignore-user-config`: ignoring user config can discard provider/authentication routing.
Offline compatibility checks precede one inference call and use at most a quarter of its
deadline; their answer is remembered per CLI binary, Codex config and skill set (a day, or
an hour after a failed probe). Unsupported
configurations keep the old path rather than failing or retrying paid inference.

## Alternatives

- Keeping the raw-token controller ignores measured cache costs.
- Repeatedly searching different masks introduces more prefix churn and optimizer complexity.
- Requiring paired reruns would give stronger evidence but doubles real work and its cost;
  the live plugin has no independent grader or controlled duplicate task.
- Requiring a grader for every session would disable optimization for interactive users.

## Consequences

Adjacent sessions vary in difficulty, so the guard is observational and can still make a
wrong decision. The trial uses three comparable sessions, not a monetary budget or statistical
guarantee; partial masks are restored rather than run indefinitely without measurement.
A retained mask stops expanding, and a rejected profile does not automatically
retry. This trades potential later savings for bounded experimentation and stable prompts.
Benchmark current behavior separately from historical paper arms, preserve raw usage and
grades, and report raw tokens and cache-weighted estimates separately.
