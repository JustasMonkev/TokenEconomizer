# TokenEconomizer PR 1 review and GPT 6.1 SOL benchmark

The JSON summaries named below (`experiments/reports/pr1-sol/*.json`) are run outputs, so they stay out of git (`CLAUDE.md`); they were recorded in commit `b410c9d`.

Implementation status: [the follow-up](PR1_SOL_FOLLOWUP.md) implements all three recommendations
locally, fixes the compact-policy finding, and records new SOL measurements. Results below
are the original review's measurements and are retained unchanged.

## Conclusion

TokenEconomizer's native Codex capability pruning reduced raw tokens by **26.9%** on six small Python bug fixes repeated in two randomized orders. Both arms passed **12 of 12** independent grading runs. However, pruning increased the cache-weighted cost estimate by **39.8%**. The baseline reused more cached input. Fewer raw tokens did not mean a cheaper workload in this experiment.

PR 1's compact tool judge reduced output tokens by **87.0%** across 12 paired judge calls, but total raw tokens fell only **4.5%**. A separate lean Codex context prototype reduced total judge tokens by **49.2%**. These judge results are component measurements, not coding-task completion results.

Review target: [PR 1](https://github.com/JustasMonkev/TokenEconomizer/pull/1), commit `3772cb1df998707ea33b71fdc6fc87e63be1d21d`. Measured on 2026-11-03 using `gpt-6.1-sol`, low reasoning effort, Codex CLI 0.166.0. There were **60 completed benchmark calls**: 24 coding runs and 36 judge calls, excluding one availability probe.

## Review finding

### P2 Compact mode can delete custom domain policy

[`_compact_brief`](https://github.com/JustasMonkev/TokenEconomizer/blob/3772cb1df998707ea33b71fdc6fc87e63be1d21d/tokeneconomizer/toolprune/tools.py#L76-L79) splits the entire domain string at `Return ONLY` and discards everything after it. With `compact=True`, this custom domain loses its required-tool policy:

```text
Classify tools. Return ONLY JSON. Never drop WebFetch: documentation verification is mandatory.
```

The generated prompt no longer contains `documentation verification is mandatory`. Since WebFetch is not a protected work-surface tool, the policy loss can affect pruning. This is an opt-in custom-domain problem; the default domain is unaffected and no in-repository custom-domain caller was found.

**Fix:** separate policy text from response-format text. Replace only the known output directive, not arbitrary trailing domain instructions. Add a regression covering policy after the format directive. This finding is reported, not patched into the GitHub PR.

The PR's Claude side-call change does not affect Codex side calls. The compact judge also is not used by the Codex-native pruning controller, whose capability-selection path returns before generic judging. Consequently, this PR cannot be credited with the coding-task reductions below.

## Coding task results

The tasks covered false-valued cache entries, pagination boundaries, CSV escaping, stable deduplication of unhashable values, Unicode case comparison, and retry counts/error propagation. Each arm started from the same buggy source. Graders ran outside the agent's task workspace; editing the supplied smoke check also failed grading. All six original buggy implementations were verified to fail their graders.

| Metric across both task orders | Pruning disabled | Native TE pruning |
|---|---:|---:|
| Passed independent grading | 12 / 12 | 12 / 12 |
| Input tokens including cached input | 790,400 | 576,131 |
| Cached input tokens | 711,808 | 425,984 |
| Output including reasoning tokens | 6,222 | 6,323 |
| Total raw tokens | 796,622 | 582,454 |
| Cache-weighted input-token equivalents | 145,292.4 | 203,061.2 |
| Standard API-equivalent estimate | $0.2906 | $0.4061 |

The first order reduced raw tokens by 26.8% but increased effective cost by 46.1%. The second reduced raw tokens by 26.9% but increased effective cost by 33.5%. The baseline's aggregate input-cache hit rate was 90.1%, versus 73.9% with pruning. Changing capability masks is a plausible contributor to lost prefix reuse; these observations do not isolate it from server cache behavior or differences in generated trajectories.

Each order used a fresh, cold-start `CodexUntoolProfile`, learned only from completed tasks, and randomized the baseline/TE execution order within each pair. Existing authentication was reused through a temporary isolated Codex directory. The benchmark applied supported native config toggles. It did **not** enable micro-skills, the loop guard, relay rewriting, or version-specific base-prompt pruning unavailable for CLI 0.166.0. Unsupported selections are recorded per run. This is a small synthetic component smoke benchmark, not SWE-bench or a validation of the complete three-mechanism system.

Evidence: combined totals (`coding-combined.json`), first order (`coding-order1.json`), second order (`coding-order2.json`). Each task directory contains its transcript, source workspace, usage row, prompt, command, and grading result.

## Compact judge results

Six issue descriptions were sampled from a pool of twelve with seed 20261103. Each was judged twice in each of three arms; all 36 calls were randomly interleaved. Full and compact arms used the PR's prompt construction and parsing against its 24-tool Claude catalog, with SOL acting as the judge. This does not turn the catalog into Codex-native tools.

| Metric across 12 calls per arm | Full judge | Compact judge | Compact plus lean context prototype |
|---|---:|---:|---:|
| Required-tool checks passed | 12 / 12 | 12 / 12 | 12 / 12 |
| Input tokens | 184,855 | 181,866 | 96,389 |
| Output including reasoning tokens | 6,476 | 843 | 732 |
| Total raw tokens | 191,331 | 182,709 | 97,121 |
| Raw token reduction versus full | — | 4.5% | 49.2% |
| Effective-token reduction versus full | — | 39.0% | 54.4% |

Full versus compact tool decisions matched exactly in **8 of 12** pairs; full versus the lean prototype matched in **7 of 12**. All retained protected work tools and the explicitly required WebFetch capability. This checks required-tool retention, not full semantic equivalence or successful completion of those issue descriptions.

The prototype replaces the Codex base instructions with a short classifier instruction, suppresses project documents, and disables discovered personal skills. It is not shipped by PR 1 and is not a drop-in production side-call implementation. Common benchmark controls disabled hooks, shell execution, web search, and agent orchestration while retaining the signed-in CLI.

Cache-hit rates differed: 13.3% full, 39.2% compact, and 12.7% lean. Host context also varied by roughly 3,169 input tokens in several runs. Therefore the 39.0% weighted reduction is an observed total, not a clean estimate of the compact format's causal cost saving. The exploratory paired-bootstrap interval for compact total-token savings spans approximately **−2.3% to +10.8%**. The much larger output reduction is the clearest component result.

Evidence: the locally retained manifest and randomized schedule, totals (`judge-summary.json`), analysis and limitations (`judge-analysis.json`), structured review (`review.json`). Per-call directories retain exact prompts, raw responses, usage, commands, and verdicts.

## Accounting

Raw tokens are total input plus output and separately reported reasoning output. Cached input is already included in CLI input, so it is subtracted before pricing fresh input. The effective-token formula is:

```text
fresh_input + 0.05 × cached_input + 1.25 × cache_write + 5 × output
```

The standard short-context rates are $2 per million fresh input tokens, $0.10 cached input, $2.50 cache writes, and $10 output. These ratios and prices were checked against the [official GPT 6.1 SOL model documentation](https://developers.openai.com/api/docs/models/gpt-6.1-sol). Runs used ChatGPT authentication: dollar values are API-equivalent estimates, **not actual subscription invoices or a measured quota formula**. No cache-write tokens were reported in these runs. All completed raw usage rows reconcile with the calculated effective costs.

## Improvements in priority order

1. **Make pruning decisions cache-aware.** Compare stable masks over longer task streams, separate cold/warm cache cases, and optimize total cost per successfully completed task. Both short task orders reduced raw tokens but increased the weighted estimate.
2. **Reduce Codex side-call context.** The lean classifier prototype is promising. A production implementation must preserve custom provider/authentication routing, validate supported CLI options, suppress irrelevant tools and skills, and benchmark judge quality before becoming default.
3. **Preserve custom domain policy.** Fix the P2 finding before advertising compact mode as interchangeable for custom domains.
4. **Stabilize prompts and price the exact model.** The local follow-ups add a SOL 6.1 pricing entry and sort tool-catalog rendering so equivalent catalogs produce identical prompts. A further experiment should put stable catalog content before variable task text to improve prefix reuse.
5. **Strengthen published evidence.** Preserve raw usage and grades; use original benchmark issue text, not descriptions reconstructed from fix tests. The PR's existing measurements cannot establish general coding savings for SOL or exact decision equivalence.

## Local changes and verification

Local production changes are two lines: the SOL pricing entry in `tokeneconomizer/pricing.py` and stable catalog ordering in `tokeneconomizer/toolprune/tools.py`. Regression tests first failed on the old behavior and then passed. The benchmark ran against the **unmodified PR checkout**, not those local improvements.

- Unmodified PR suite on Python 3.14.6: **885 passed, 9 skipped**.
- Local follow-up suite: **884 passed, 9 skipped**; its base is main and does not include the PR's additional tests.
- Final targeted pricing, prompt-ordering, and benchmark checks: **6 passed**.
- Both benchmark scripts compile; `git diff --check` passes.
- Judge resume was exercised against all 36 saved rows and made no new model calls.
- Early full-suite attempts used unsupported macOS Python 3.9 or hit sandboxed CLI probes; the counts above are the completed supported-environment runs.

No changes were committed, pushed, merged, or posted as a GitHub review. The compact-domain issue remains a proposed PR fix. Raw run artifacts stay local under the repository's ignored `experiments/results/` directory.

## Reproduce

Use Python 3.10 or newer and a Codex CLI already authenticated with access to `gpt-6.1-sol`. The examples make paid/quota-consuming model calls. Set `--source` to a checkout of the reviewed PR commit, not this workspace's main branch.

```bash
python3 experiments/scripts/benchmark_pr1_sol.py --source /path/to/pr1-checkout --output /tmp/te-judge-new --tasks 6 --repeats 2
python3 experiments/scripts/benchmark_sol_tasks.py --source /path/to/pr1-checkout --output /tmp/te-coding-order1 --seed 20261103
python3 experiments/scripts/benchmark_sol_tasks.py --source /path/to/pr1-checkout --output /tmp/te-coding-order2 --seed 20261104
```

Seeds control task sampling/order, not deterministic model output. Use fresh directories for coding runs so learning state cannot be silently reused. The judge runner rejects mismatched experiment manifests and can resume matching saved rows.
