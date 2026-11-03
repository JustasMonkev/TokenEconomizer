"""Arms that involve the mechanisms — including the two scaffold integrations.

The `trimech-rtk` / `trimech-repomap` arms live **here, beside `trimech`**, not in
`baselines/arms.py`, because that is what they are: `trimech` with one external tool
co-activated. Filing them under the baselines would make them read as scaffold arms that
happen to have mechanisms, which is backwards — the mechanisms are the subject and the
scaffold is the comparison. They name their scaffold with a string, so this module still
imports nothing from `te_bench.baselines`.

`baselines/arms.py` holds only the arms that are *purely* an external tool with no
mechanism active (`rtk`, `serena`, `repomap`).
"""
from ._armspec import arm, cfg
from tokeneconomizer.config import InjectionConfig, ToolingConfig, SpiralConfig

__all__ = ["MECHANISM_ARMS", "INJECT", "ENVFACTS", "ENVFACTS_PLUS", "UNTOOL", "SPIRAL",
           "SPIRAL_WEB", "UNTOOL_LEAN", "CODEX_UNTOOL", "CODEX_SPIRAL", "CODEX_SPIRAL_POSTGREEN"]

# The proven single-mechanism flag sets, split out of `trimech` so each mechanism can be
# measured on its own. They are the SAME parameters trimech co-activates -- an arm that
# retuned its mechanism would not be an ablation of trimech, it would be a different arm.
INJECT = dict(enabled=True, propose_every=5)       # micro-skill synthesis + injection
# The LLM-free environment-fact layer alone (microskills/envfacts). No paid call, so
# `needs_optimizer` is False. An ablation of the default below, not the default itself.
ENVFACTS = dict(enabled=True, profile="envfacts")
# THE CODEX INJECTION DEFAULT (2026-09-12): the free layer plus the paid proposer at half
# the generic cadence -- the environment facts are already stated, so the proposer's calls
# are rarer and its budget goes to facts beyond them.
#
# Why the paid layer is the default. The harness activates the project environment on
# every lane, so `python` resolves to the project interpreter and the free layer forms NO
# interpreter fact -- over a baseline it states only the network-boundary line. The free
# layer therefore mostly abstains, and the open-vocabulary proposer is what is left to find
# repository facts; its spend is charged to the arm.
ENVFACTS_PLUS = dict(enabled=True, profile="envfacts+distiller", propose_every=10)
# The Codex lane's injection profiles by name: the default, and its LLM-free ablation
# (arm `codex-inject-envfacts`).
CODEX_DEFAULT_PROFILE = ENVFACTS_PLUS["profile"]
CODEX_FREE_PROFILE = ENVFACTS["profile"]
# The Codex-native flag sets, split out for exactly the reason the generic ones above are:
# `codex-trimech` co-activates THESE, so each Codex single-mechanism arm is a true ablation
# of it rather than a separately tuned arm.
CODEX_UNTOOL = dict(codex_untool=True)      # config-toggle capability mask + lean prompt
# The spiral half of `codex-trimech` is the DEFENSIVE profile, not the shipped post-green
# one, and that is a deliberate choice rather than an upgrade-by-default. The defensive
# profile NEVER aborts, which is what makes it safe to stack on top of two mechanisms that already
# change the request
# contract and the prompt. Its ablation is
# `codex-spiral-defensive`, not `codex-spiral`.
CODEX_SPIRAL = dict(codex_spiral=True, codex_spiral_profile="defensive")
# There is deliberately NO `codex-*-web` variant, and the reason is a property of the Codex
# controller rather than an omission. `trimech-web` / `spiral-only-web` exist because the
# GENERIC envelope is a within-stream p90 that never warms up over a 20-task Web-Bench
# stream, so those arms pre-register one and write the nudge with an LLM.
# `CodexEnvelopeTracker` already carries a cold start (24 tool events, used until five
# resolved tasks have been seen) and the Codex nudges are deterministic text, so the same
# nine Codex arms are well-defined on both domains and mean the same thing on each. Expect
# the spiral half to be near-dormant on Web-Bench all the same: its tasks are small
# increments and a 24-event cold start is above most of them.
# The SHIPPED post-green policy, kept reachable and named. It is not what the spiral arms
# carry any more, for the reason above; it is what `codex-spiral-postgreen` is for, and what
# every pre-2026-09-12 `codex_spiral_profile` in a stored run_config resolves to.
CODEX_SPIRAL_POSTGREEN = dict(codex_spiral=True)     # SpiralConfig's own default profile
UNTOOL = dict(evidence=3, judge=True)              # evidence untooling + tool judge
UNTOOL_LEAN = dict(UNTOOL, judge_compact=True)
SPIRAL = dict(guard=True, judge=True, nudge=True)  # spiral guard -> judge -> nudge
# Web-Bench's streams are 20-21 tasks, too short for the within-stream p90 envelope to warm
# up, so its spiral arms pre-register one and write the nudge with the LLM.
SPIRAL_WEB = dict(SPIRAL, nudge_llm=True, frozen_envelope=30.0)


def _trimech(spiral=SPIRAL, tooling=UNTOOL, **kw):
    """All three mechanisms at trimech's own parameters. `kw` carries the scaffold, if any.

    `spiral` selects the spiral tuning: SPIRAL for SWE-bench's long streams, SPIRAL_WEB for
    Web-Bench's 20-21 task ones (LLM nudge + frozen envelope, because a within-stream p90
    never warms up that fast). Nothing else differs between the two domains' variants.
    """
    return arm("manual4",
               lambda: cfg(injection=InjectionConfig(**INJECT),
                           tooling=ToolingConfig(**tooling),
                           spiral=SpiralConfig(**spiral)), **kw)


def _codex_trimech(**kw):
    """All three Codex mechanisms at their single-arm parameters. `kw` carries the scaffold."""
    return arm("manual4",
               lambda: cfg(injection=InjectionConfig(**ENVFACTS_PLUS),
                           tooling=ToolingConfig(**CODEX_UNTOOL),
                           spiral=SpiralConfig(**CODEX_SPIRAL)), **kw)


MECHANISM_ARMS = {
    # ---- naive (no mechanisms) -----------------------------------------------------------
    "baseline": arm(
        "standalone",
        lambda: cfg()),
    # Reasoning-effort variants of naive. The LEVEL IS PART OF THE ARM NAME, and the arm sets
    # TE_EFFORT_LEVEL itself rather than trusting the caller to export it: the level is
    # invisible in an EconomizerConfig, so a mislabelled cell would be indistinguishable from
    # a correct one after the fact, and the whole point of the comparison is that the level
    # differs. `baseline` itself is deliberately NOT pinned here -- it inherits the harness
    # default (high), which is what every stored row was run at.
    "baseline-medium": arm(
        "standalone",
        lambda: cfg(), {"TE_EFFORT_LEVEL": "medium"}),
    "baseline-low": arm(
        "standalone",
        lambda: cfg(), {"TE_EFFORT_LEVEL": "low"}),

    # ---- one mechanism active ------------------------------------------------------------
    "inject-only": arm(
        "manual4",
        lambda: cfg(injection=InjectionConfig(**INJECT))),
    "untool-only": arm(
        "standalone",
        lambda: cfg(tooling=ToolingConfig(**UNTOOL))),
    "spiral-only": arm(
        "standalone",
        lambda: cfg(spiral=SpiralConfig(**SPIRAL))),
    "spiral-only-web": arm(
        "standalone",
        lambda: cfg(spiral=SpiralConfig(**SPIRAL_WEB))),

    # ---- all three active ----------------------------------------------------------------
    # trimech = manual4 injection + evidence untooling(3)+judge + spiral guard/judge/nudge
    "trimech": _trimech(),
    "trimech-lean": _trimech(tooling=UNTOOL_LEAN),
    # short Web-Bench streams add the LLM-written nudge + a cold-start frozen envelope
    "trimech-web": arm(
        "manual4",
        lambda: cfg(injection=InjectionConfig(**INJECT),
                    tooling=ToolingConfig(**UNTOOL),
                    spiral=SpiralConfig(**SPIRAL_WEB))),

    # ---- all three, co-activated with an external tool baseline --------------------------
    # Siblings of `trimech`, and identical to it apart from the scaffold they name. Both
    # combinations are well-defined: rtk hooks Bash (always in PROTECTED_TOOLS, so untooling
    # can never drop it) and the repo map writes a file and touches no tool.
    #
    # There is deliberately NO `trimech-serena`. Serena's contract is that the agent
    # navigates through its MCP tools instead of the native ones, which is the very thing
    # evidence untooling prunes when they go unused -- the combined arm would measure that
    # contention rather than either baseline.
    "trimech-rtk": _trimech(scaffold="rtk"),
    "trimech-repomap": _trimech(scaffold="repomap"),
    # Web-Bench's variant of `trimech-rtk`, for the same reason `trimech-web` exists: on a
    # 20-task stream the within-stream p90 envelope never warms up. Pairing rtk with plain
    # `trimech` there would spiral-tune it differently from every other spiral-bearing
    # Web-Bench arm, so `trimech-rtk` would not be comparable to `trimech-web` -- which is
    # the single comparison the arm exists to make.
    "trimech-rtk-web": _trimech(spiral=SPIRAL_WEB, scaffold="rtk"),

    # ---- Codex-native --------------------------------------------------------------------
    "codex-untool": arm(
        "standalone",
        lambda: cfg(tooling=ToolingConfig(**CODEX_UNTOOL))),
    # THE CODEX SPIRAL ARM, and since 2026-09-12 it carries the DEFENSIVE profile -- the
    # same move, for the same reason, that made `codex-inject` the composed profile: the
    # plain name should be the current configuration, and the superseded one should be
    # reachable under a name that says which it is. The defensive profile answers every
    # deterministic spiral signature with a same-thread nudge, never aborts, and never
    # touches reasoning effort.
    #
    # This is also what makes the Codex family analogous to the generic one: `spiral-only`
    # IS trimech's spiral, and now `codex-spiral` IS codex-trimech's.
    "codex-spiral": arm(
        "standalone",
        lambda: cfg(spiral=SpiralConfig(**CODEX_SPIRAL))),
    # The shipped post-green finalize alone -- the named ablation, the twin of
    # `codex-inject-envfacts`. Run it to attribute a defensive result between the finalize
    # rule and the signatures added on top of it.
    "codex-spiral-postgreen": arm(
        "standalone",
        lambda: cfg(spiral=SpiralConfig(**CODEX_SPIRAL_POSTGREEN))),
    # The name the defensive profile shipped under on 2026-09-12, kept so the stored cell of
    # that name and any pinned script still resolve to the same configuration as
    # `codex-spiral`. The twin of `codex-inject-distiller`.
    "codex-spiral-defensive": arm(
        "standalone",
        lambda: cfg(spiral=SpiralConfig(**CODEX_SPIRAL))),
    # Untooling + the spiral, i.e. `codex-trimech` minus injection. It reads the same two
    # dicts trimech does, so it stays that as the parameters move; a spiral half pinned
    # independently would make it a different arm from the one its name promises.
    "codex-combo": arm(
        "standalone",
        lambda: cfg(tooling=ToolingConfig(**CODEX_UNTOOL),
                    spiral=SpiralConfig(**CODEX_SPIRAL))),
    # Codex injection, alone and stacked with capability untooling. `manual4` is the mode
    # label every injection arm carries (aggregate.py recovers it from the run-dir name);
    # the PROFILE in the config is what tells the prefix schemes apart, and it is what
    # changed on 2026-09-12: `codex-inject` is now the composed `envfacts+distiller`
    # (see ENVFACTS_PLUS above for why), which makes it a PAID arm -- `needs_optimizer` is
    # True, the endpoint preflight fires, and the proposer's spend is charged to the arm
    # through `optimizer_tokens`. Cells run before that date carry `profile: envfacts` in
    # their run_config and are the ablation, not the default.
    "codex-inject": arm(
        "manual4",
        lambda: cfg(injection=InjectionConfig(**ENVFACTS_PLUS))),
    "codex-inject-untool": arm(
        "manual4",
        lambda: cfg(injection=InjectionConfig(**ENVFACTS_PLUS),
                    tooling=ToolingConfig(**CODEX_UNTOOL))),
    # The LLM-free ablation: the environment-fact layer with no proposer, zero overhead.
    # Run it to attribute a composed result between the two layers, or on a lane that must
    # make no paid call at all.
    "codex-inject-envfacts": arm(
        "manual4",
        lambda: cfg(injection=InjectionConfig(**ENVFACTS))),
    # The name the composed profile shipped under between 2026-09-11 and 2026-09-12, kept
    # so the stored cell of that name and any launch script that pins it still resolve to
    # the same configuration as `codex-inject`.
    "codex-inject-distiller": arm(
        "manual4",
        lambda: cfg(injection=InjectionConfig(**ENVFACTS_PLUS))),

    # ---- all three Codex mechanisms active ----------------------------------------------
    # The Codex sibling of `trimech`, and it composes the same way: the three flag sets
    # above, unretuned, so `codex-inject`, `codex-untool` and `codex-spiral-defensive` are
    # each a true ablation of this arm. It is `manual4` because injection is on -- that is
    # the mode label `aggregate.py` recovers from the run-dir name.
    #
    # The three act on three different surfaces and do not contend:
    #   * injection writes a prefix into the task prompt (`build_prompt`);
    #   * untooling writes the per-task `config.toml` (capability toggles) and the lean
    #     `model_instructions_file` beside it;
    #   * the spiral guard reads the live `--json` stream and, when it fires, resumes the
    #     SAME thread in the SAME container -- which still has that config.toml and that
    #     instructions file, so the resumed segment inherits the untooled contract and the
    #     prompt prefix already in its history. Nothing is re-sent and nothing is re-paid.
    #
    # Two compounding effects to expect in the row, both by construction:
    #   * the proposer's spend lands in `optimizer_tokens` (this is a PAID arm);
    #   * a task the guard nudges reports no usage for the killed segment, so
    #     `codex.merge_usage` poisons the merge and the row is flagged `usage_incomplete`.
    #     It still counts for resolve and drops out of the paired cost figure, exactly as
    #     in `codex-spiral-defensive`. With the defensive profile that is a few percent of
    #     tasks on terra.
    "codex-trimech": _codex_trimech(),
    # The Codex sibling of `trimech-rtk`: the same three mechanisms with rtk co-activated.
    # On Codex the scaffold is INSTRUCTION-based -- `rtk init -g --codex` writes AGENTS.md +
    # RTK.md and the agent is told to prefix its commands; there is no hook this CLI honours
    # -- so the row carries `rtk_compliance` and an arm at zero compliance is native. rtk
    # touches no tool schema and the capability mask touches no shell command, so the two
    # do not contend, exactly as on the generic lane.
    "codex-trimech-rtk": _codex_trimech(scaffold="rtk"),
    # The Codex sibling of `trimech-repomap`: the same three mechanisms with aider's repo map
    # co-activated. The map is written as the PROJECT `AGENTS.md` (/testbed), which Codex
    # reads at session start and carries in every request. None of the three mechanisms
    # touches that file -- injection rides the task prompt, untooling the config and the
    # `model_instructions_file`, the spiral the live stream -- and no capability toggle in
    # the mask disables project instructions, so map and mask coexist however far the mask
    # has grown. SWE-bench only, like `repomap` (the generator walks `*.py`).
    "codex-trimech-repomap": _codex_trimech(scaffold="repomap"),
}
