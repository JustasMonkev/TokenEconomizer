# TokenEconomizer PR 1 implementation follow up

All three recommendations from [the review](PR1_SOL_REVIEW.md) are implemented locally:
cache-aware Codex pruning, lean Codex optimizer calls, and preservation of compact-judge
custom policies. Three implementation agents worked on separate modules; integration review
added protections for migration, restoration, and session accounting.

## Implemented behavior

### Cache aware pruning

The controller collects three priced baseline sessions, then holds one eligible capability
batch stable for a three-session trial. It retains the mask only if observed average
cache-weighted cost falls. Where real request counts are available, the per-request average
must also fall. Otherwise it restores the mask and stops further demotions in that profile.

Costs must share a known model, pricing basis, grading basis, and request-count availability.
Live completed-but-ungraded sessions are not labelled successful. Incomplete or mixed-model
usage, unfinished sessions, mixed-mask resumes, and partial trial masks cannot establish an
improvement. A pending mask that has not been applied contributes no trial samples.

SessionStart records the actual launch mask; fold-time settings cannot retroactively turn an
old full-context session into a pruned sample. Migration restores old TE-owned masks before
collecting a baseline while retaining valid historical use protections. Failed restorations
are retried. Completed capability use restores that capability even after retention.

This is a bounded observational safeguard, not a causal estimator: adjacent tasks can have
different difficulty. A retained mask stops expanding, and a rejected profile does not retry
automatically. [Decision and alternatives](../docs/decisions/001-cache-aware-codex.md).

### Lean Codex side calls

`sidecall.codex_call` keeps the current provider configuration, authentication, model routing,
and Codex home. Supported CLIs receive a short temporary helper prompt and per-call overrides
that suppress project documents, discovered skills, supported tool features, plugins/apps,
and MCP servers. The implementation does not use `--ignore-user-config` or copy credentials.

Offline configuration validation precedes inference. Unsupported CLI/configuration cases
use the previous invocation; MCP names containing dots or quotes currently take that fallback.
All probes and the single inference call share a deadline. Paid failures are not retried.
User settings and interactive sessions are unchanged. Native schemas without a supported
toggle can remain, so this is lean supported context rather than a universal tool-free API.

### Compact judge policies

The default policy and output format are separate constants. Custom-domain text is preserved
instead of being split at `Return ONLY`; a final instruction overrides response formatting
without removing tool-selection policy. The full default prompt remains unchanged, catalog
ordering stays deterministic, and compact mode remains opt-in through `judge_compact` or the
`trimech-lean` experiment arm. It is not used by the Codex-native capability controller.

## New SOL measurements

Model: `gpt-6.1-sol`, low reasoning effort, Codex CLI 0.166.0, measured 2026-11-03.
These are **40 new completed model calls**, separate from the earlier review's 60 calls.

### Actual side call implementation

Four compact judge tasks, each repeated twice per arm, were randomly interleaved. Both arms
used the same isolated configuration and existing login; the old and new production
`codex_call` functions executed the prompts. Checks required protected work tools plus
task-specific capabilities, including notebook editing and fetching documentation.

| Metric | Previous side calls | Implemented lean calls |
|---|---:|---:|
| Required-tool checks passed | 8 / 8 | 8 / 8 |
| Input tokens | 143,684 | 39,950 |
| Output including reasoning | 808 | 657 |
| Raw total tokens | 144,492 | 40,607 |
| Cache-weighted equivalents | 54,335.2 | 29,372.6 |

**71.9% fewer raw tokens and 45.9% fewer cache-weighted equivalents.** These component checks
do not establish unchanged end-to-end coding quality. Authentication/provider preservation
and unsupported-version behavior have separate regression tests.

Evidence: the locally retained side-call manifest,
[summary](reports/pr1-sol/followup-sidecalls.json). Each call directory retains its exact
prompt, command, raw CLI response, usage, and parsed verdict. Startup configuration probe
output is not saved, avoiding disclosure of provider configuration.

### Cache guard on actual coding tasks

Six synthetic Python bug fixes were repeated in two shuffled cycles of one continuous
learning stream. Each task had an independently graded baseline and guarded run, with paired
execution order randomized. New source workspaces started from the original bug every time.

| Metric | Pruning disabled | Cache-aware pruning |
|---|---:|---:|
| Independent grades passed | 12 / 12 | 12 / 12 |
| Raw total tokens | 778,401 | 636,677 |
| Cache-weighted equivalents | 150,971.4 | 174,777.8 |
| Standard API-equivalent estimate | $0.3019 | $0.3496 |

Raw tokens fell **18.2%**, but the weighted estimate was still **15.8% higher** than disabled
pruning. There is no net cost-saving claim for this coding run.

The important observed guard behavior was:

1. Tasks 1–3 collected an unpruned baseline: 42,062.2 effective tokens.
2. Tasks 4–6 held one native capability mask stable: 48,898.2 effective tokens.
3. The trial cost was 16.3% higher per task, so the guard recorded
   `no-observed-cost-improvement` and rolled back.
4. Tasks 7–12 used no pruning overrides and remained resolved.

The harness intentionally supplied no fabricated request count: CLI tool-step counts are
not model requests. It measured native pruning only, without micro-skills, the loop guard,
relay rewriting, or unsupported base-prompt removal. This validates detection and rollback
in a small stream, not statistical superiority over the old controller or a SWE-bench result.

Evidence: [coding summary](reports/pr1-sol/followup-coding.json),
[combined follow-up data](reports/pr1-sol/followup-summary.json). Each guarded row preserves the
baseline/trial windows, decision, actual command, code workspace, transcript, and grading.

Dollar estimates use the [published SOL rates](https://developers.openai.com/api/docs/models/gpt-6.1-sol).
Calls used ChatGPT authentication, so these are API-equivalent estimates, not subscription
invoices or a measured quota formula. Benchmark directories preserve the measured source
diffs. Later migration/partial-mask edge refinements were regression-tested; they do not
change the normal paths exercised by these paid runs.

## Verification and local status

- Full Python 3.14 suite on the combined PR: **970 passed, 9 skipped**.
- Custom-domain regressions fail with the original PR helper and pass with the fix.
- Cost-guard tests cover stable trials, retain/rollback, missing and mismatched costs,
  grading bases, model changes, partial masks, task hints, migration, and use protection.
- Live tests cover launch-mask accounting, resumed histories, unfinished sessions, rollback
  retries, preserving user configuration, and restoring newly used retained capabilities.
- Side-call tests cover provider/auth preservation, compatibility fallback, MCP isolation,
  skills, shared deadlines, cleanup, accounting, and no paid retry.
- Benchmark scripts compile; `git diff --check` passes.

These changes update PR 1. The original review remains a record of the earlier measurements;
its compact-policy finding is fixed. Aggregate benchmark summaries are committed alongside this
report; raw transcripts and workspaces remain local and can be regenerated with the commands below.

## Reproduce

Use a fresh result directory, Python 3.10+, and an authenticated Codex CLI with access to SOL.
The commands consume model quota. The reference checkout must contain PR commit
`3772cb1df998707ea33b71fdc6fc87e63be1d21d`; the source checkout must contain these changes.

```bash
python3 experiments/scripts/benchmark_codex_sidecalls.py --source . --reference /path/to/pr1-checkout --output /tmp/te-sidecalls-new --repeats 2
python3 experiments/scripts/benchmark_sol_tasks.py --source . --output /tmp/te-guard-new --seed 20261103 --cycles 2
```
