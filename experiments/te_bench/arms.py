"""The arm registry: every proven arm, mapped to an exact `EconomizerConfig`.

This is the single source of truth. It assembles two tables that live next to the code they
configure:

* `mechanism_arms.py` -- everything the economizer drives: naive, the effort variants, the
  single-mechanism ablations, `trimech`, and the two scaffold **integrations**, which sit
  beside `trimech` because that is what they are.
* `baselines/arms.py`  -- arms that are only an external tool, with no mechanism active.

`experiments/scripts/run_experiment.py` re-exports `ARMS` and `SCAFFOLD_ARMS` from here, so the
driver
remains the entry point the docs point at and every existing import keeps working.

**Importing this module does not pull a baseline's dependencies.** The arm tables name a
scaffold with a string and the harness resolves it lazily, so `import tokeneconomizer` and
`import te_bench.arms` both stay light -- and `tokeneconomizer/__init__.py`
deliberately does not import this module, so the zero-dependency core import is unchanged.
"""
from ._armspec import Arm, arm, cfg
from .baselines.arms import BASELINE_ARMS
from .mechanism_arms import MECHANISM_ARMS

__all__ = ["ARMS", "SCAFFOLD_ARMS", "ARM_ORDER", "Arm", "arm", "cfg"]

# Canonical order, stated rather than left to emerge from a dict merge: it is what a plan
# lists and what a reader compares against the tables above, and the two source dicts
# interleave here (the pure-baseline arms sit between `trimech-web` and the integrations).
ARM_ORDER = (
    "baseline", "baseline-medium", "baseline-low",
    "inject-only", "untool-only", "spiral-only", "spiral-only-web",
    "trimech", "trimech-lean", "trimech-web",
    "codex-untool", "codex-spiral", "codex-spiral-postgreen", "codex-spiral-defensive",
    "codex-combo", "codex-inject", "codex-inject-untool",
    "codex-inject-envfacts", "codex-inject-distiller", "codex-trimech", "codex-trimech-rtk",
    "codex-trimech-repomap",
    "rtk", "serena", "repomap",
    "trimech-rtk", "trimech-rtk-web", "trimech-repomap",
)

_MERGED = {**MECHANISM_ARMS, **BASELINE_ARMS}

_missing = set(_MERGED) - set(ARM_ORDER)
_unknown = set(ARM_ORDER) - set(_MERGED)
if _missing or _unknown:                    # a new arm must be placed, not silently appended
    raise RuntimeError(
        f"ARM_ORDER is out of step with the arm tables: "
        f"missing from order {sorted(_missing)}, not defined {sorted(_unknown)}")

ARMS = {name: _MERGED[name] for name in ARM_ORDER}

# Arms carrying an external tool scaffold, derived rather than restated so it cannot fall
# out of step with ARMS. `tests/test_sweep_arms.py` pins this against
# `te_bench.baselines.ARM_SCAFFOLDS`, which is what the harness reads.
SCAFFOLD_ARMS = {name: a.scaffold for name, a in ARMS.items() if a.scaffold}
