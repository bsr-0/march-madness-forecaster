"""Researcher degrees of freedom in the pool-strategy search.

WHY THIS EXISTS
---------------
The published figure is that ``meta_region_poolaware`` finishes first in about
12% of simulated 30-person pools over 14 seasons. Audit finding **H2, "Garden
of forking paths with no untouched holdout"**, is that the figure was produced
by a search nobody had counted: one mode chosen as best of 79, over knobs whose
values are mostly undocumented literals, at least two of which were set
*because of their effect on this very number*, with two whole candidate
families deleted for scoring worse on the evaluation seasons -- and no
correction, no sensitivity analysis and no clean holdout anywhere.

Audit recommendation 12 asked to "bring the pool-strategy search under the RDoF
audit". There was such an audit -- ``src/ml/evaluation/rdof_audit.py`` -- but it
was deleted with the whole ML pipeline in ``ea06a40`` (finding H10), and it had
never had a single reference to the pool search in any case. So this is a
rebuild, scoped to the pool product, keeping what that module got right:

  * a **tier** taxonomy (1 externally derived, 2 structurally constrained,
    3 freely tuned) with ``valid_range`` doing double duty as sweep bounds;
  * a **DoF/sample-size ratio** gate, and a flatness discount so a knob whose
    entire grid gives the same answer is not counted as a spent degree of
    freedom;
  * the **freeze/lockfile pair**, whose thesis transfers verbatim: *once a
    holdout year has been evaluated, any later change to the configuration that
    produced those results is researcher-degrees-of-freedom contamination*;
  * one report object emitting both JSON and text, with recommendation strings
    prefix-tagged so they can be grepped.

and fixing what it got wrong. Its registry was a hand-maintained literal that
silently drifted from the code it claimed to describe -- which is why it
accumulated a section literally titled "Previously Unregistered Constants" and
why ``loyo_protocol.py`` carried a hand-copied ``_N_TUNED_CONSTANTS = 58``.
Here, every entry that can name a live symbol does, ``live_value`` resolves it
by import, and ``tests/test_pool_rdof_audit.py`` fails the build when the two
disagree. A registry that cannot be wrong about the code is the only kind worth
having.

One thing is deliberately **not** ported: ``adopt_sensitivity_optima()``.
Auto-adopting a sweep optimum is itself a fresh degree of freedom, and it was
the most dangerous function in the original. The sweep in
``scripts/pool_rdof_audit.py`` is diagnostic and read-only.

THE TWO HOLDOUT TIERS, AND WHY 2026 CANNOT BE THE FIRST ONE
------------------------------------------------------------
Recommendation 12's second half asks to "keep one never-touched holdout year".
No season in history qualifies, and 2026 -- the obvious candidate, since it is
already barred from the aggregates -- cannot be promoted to one. The evidence,
checked rather than assumed:

* **Training is already clean.** ``TRAIN_YEARS`` contains 2026
  (``src/prediction/noseed_model.py:72-79``) but ``train_noseed_model``'s
  walk-forward filter is ``y < max_year``, so no model evaluated on 2026 ever
  trains on 2026. There is no training leakage to repair.
* **The search is contaminated, irreversibly.** Two candidate families were
  deleted *because they lowered aggregate P(1st)*, and the code records the
  dates and the window: ``scripts/mc_pool_backtest.py:4026-4042``, family (d)
  removed **2026-05-03** ("10.93% WITH upset vs 11.20% WITHOUT... selected in
  only 1/15 years") and family (e) removed **2026-05-16** ("7.1% WITH vs 11.9%
  WITHOUT"). A 15-season window is ``BACKTEST_YEARS`` = 2011-2026 excluding
  2020 -- it *includes* 2026, whose tournament had ended that April.
  ``CONTAMINATED_EVAL_YEARS = {2026}``, the guard that now strips 2026 from
  aggregates, was not added until **2026-09-09** (``a3fb412``), four months
  after those prunings. The removed families no longer exist as code, so those
  decisions cannot be re-run on a 2026-free window.
* **Promoting it in the frozen spec would cost the 2027 claim.**
  ``holdout.contaminated_seasons: [2026]`` lives inside the hashed
  ``configs/frozen/prospective_2027_v2_scoped.json``. Changing it requires a
  ``SPEC_VERSION`` bump, which by ``src/governance/frozen_spec.py``'s own rule
  "invalidates the prospective claim for this version" -- trading a genuinely
  untouched future holdout for a partly-decontaminated past one.

So there are two tiers, and the weaker one says so on its face:

  ``SEQUESTERED_YEARS``        Level 1. Untouched, prospective, enforced.
  ``PARAMETER_CLEAN_HOLDOUT``  Level 2.5. Structurally contaminated (the
                               candidate-family set was pruned on a window
                               containing it -- a sunk cost) but
                               parameter-clean, because every parametric knob
                               is swept on 2011-2025 only. Better than
                               in-sample; not a substitute for Level 1.

"Level 2.5" is not invented here: it is the deleted ``loyo_protocol.py``'s own
term for "architecture-contaminated but parameter-clean", and it is the most
this season can honestly be called.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Holdout tiers
# ---------------------------------------------------------------------------

# Level 1. Genuinely untouched: 2027 has not been played, no outcome exists,
# and the frozen 2027.v2 spec pins the methodology that will predict it. The
# only clean holdout this project has or can have without waiting another year.
SEQUESTERED_YEARS: frozenset[int] = frozenset({2027})

# Level 2.5. See the module docstring for why this is not Level 1 and cannot
# be made Level 1.
PARAMETER_CLEAN_HOLDOUT: int = 2026

# Deliberately inside the artifacts tree rather than /tmp or .git, so it
# persists across sessions and is committed alongside the measurements it
# protects. Same reasoning as the deleted module's holdout_evaluation.lock.json.
HOLDOUT_LOCKFILE = Path("artifacts/headline_measurement/holdout_evaluation.lock.json")

# The unit of independence for every ratio computed here. Repeats within a
# season share one seed layout and one pick distribution, so counting
# season x repeat trials as observations overstates the sample several-fold --
# audit finding H1, which measured the inflation at roughly 3x. 14 is
# len(EVALUATION_YEARS): 2011-2025 excluding 2020.
N_EVALUATION_SEASONS: int = 14


class HoldoutContaminationError(RuntimeError):
    """A sequestered season was about to be used for search or evaluation."""


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

# Tiers, following the deleted module's taxonomy exactly so the two are
# comparable:
TIER_EXTERNAL = 1  # value comes from outside this repo (literature, ESPN's own rules)
TIER_STRUCTURAL = 2  # bounded or pinned by structure; not a free choice
TIER_FREE = 3  # freely tuned -- a researcher degree of freedom

# Status vocabulary. This is the field the deleted module lacked, and the one
# finding H2 actually needs: the problem is not that constants exist, it is
# *how* their values were arrived at.
STATUS_SEARCHED = "searched"  # swept, and the winner is what ships
STATUS_FROZEN_AFTER_SEARCH = "frozen_after_search"  # swept once, then pinned to a literal
STATUS_NEVER_SEARCHED = "never_searched"  # a literal with no sweep behind it
STATUS_EXTERNAL = "external"  # taken from a citation or an external rule
STATUS_TUNED_ON_METRIC = "tuned_on_metric"  # set BECAUSE of its effect on the reported number
STATUS_REMOVED_AFTER_MEASURING = "removed_after_measuring"  # deleted for scoring worse

ALL_STATUSES = (
    STATUS_SEARCHED,
    STATUS_FROZEN_AFTER_SEARCH,
    STATUS_NEVER_SEARCHED,
    STATUS_EXTERNAL,
    STATUS_TUNED_ON_METRIC,
    STATUS_REMOVED_AFTER_MEASURING,
)

# Statuses that represent a degree of freedom actually spent on this metric.
# `external` and `structural` choices are not researcher choices;
# `never_searched` is a choice but an unexercised one -- it consumed no
# comparison against this number, which is a different (and smaller) problem
# than having optimised it.
_SPENDING_STATUSES = (
    STATUS_SEARCHED,
    STATUS_FROZEN_AFTER_SEARCH,
    STATUS_TUNED_ON_METRIC,
    STATUS_REMOVED_AFTER_MEASURING,
)

_TRANSFORMS = {
    None: lambda v: v,
    "len": len,
    "sorted": lambda v: sorted(v),
}


@dataclass(frozen=True)
class PoolDegreeOfFreedom:
    """One knob in the pool-strategy search, with how its value was arrived at.

    Attributes:
        name: Stable identifier, used as the registry key.
        tier: ``TIER_EXTERNAL`` / ``TIER_STRUCTURAL`` / ``TIER_FREE``.
        current_value: The value as registered. Compared against the live code
            by :func:`live_value`; a mismatch fails the build.
        code_path: Human-readable ``file:line`` pointer, including
            ``(inline literal)`` where the value is not bound to a name. The
            deleted module's ``config_path`` field, same purpose.
        live_symbol: ``"module.path:attr"``, or ``"module.path:callable.param"``
            to read a parameter default. ``None`` when the value is an inline
            literal with no importable binding -- those are the entries the
            drift test cannot protect, and it reports how many there are.
        live_transform: Applied to the live value before comparison. ``"len"``
            lets a large collection be registered by size rather than
            transcribed.
        valid_range: Plausible bounds, doubling as sweep bounds.
        derivation: Provenance. Quoted from the code comment where one exists,
            because the code's own account is the evidence.
        status: One of :data:`ALL_STATUSES`.
        contaminated_by: Seasons whose *outcomes* were known and in scope when
            this value was chosen. Non-empty is the forking-paths debt.
        swept_by: Name of the axis in ``scripts/pool_rdof_audit.py``'s
            ``SWEEP_AXES`` that measures this knob's sensitivity, if any.
    """

    name: str
    tier: int
    current_value: Any
    code_path: str
    derivation: str
    status: str
    live_symbol: Optional[str] = None
    live_transform: Optional[str] = None
    valid_range: Optional[Tuple[float, float]] = None
    contaminated_by: Tuple[int, ...] = ()
    swept_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tier": self.tier,
            "current_value": _jsonable(self.current_value),
            "code_path": self.code_path,
            "live_symbol": self.live_symbol,
            "live_transform": self.live_transform,
            "valid_range": list(self.valid_range) if self.valid_range else None,
            "derivation": self.derivation,
            "status": self.status,
            "contaminated_by": list(self.contaminated_by),
            "swept_by": self.swept_by,
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, frozenset):
        return sorted(value)
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


# The registry. Ordered by subsystem, mirroring where a reader would go looking.
#
# Every `derivation` that appears in quotes is copied from the code or from
# FINDINGS.md, not paraphrased -- the point of this field is to be evidence,
# and a paraphrase of a provenance claim is not one.
REGISTRY: Tuple[PoolDegreeOfFreedom, ...] = (
    # --- The measurement contract -------------------------------------------
    PoolDegreeOfFreedom(
        name="n_opponents",
        tier=TIER_FREE,
        current_value=29,
        code_path="scripts/mc_pool_backtest.py:154 — N_OPPONENTS",
        live_symbol="scripts.mc_pool_backtest:N_OPPONENTS",
        valid_range=(9, 999),
        derivation=(
            "29 (a 30-person pool) since 2026-09-10. 'THIS WAS 999 UNTIL 2026-09-10, and the "
            "change is a bug fix rather than a preference... the old default silently measured "
            "a pool 33x larger than any that exists here for every pre-2023 season.' P(1st) is "
            "mechanically pool-size dependent (~2.5x worse at 1000 entries), so this is part "
            "of the claim, not a detail."
        ),
        status=STATUS_NEVER_SEARCHED,
    ),
    PoolDegreeOfFreedom(
        name="espn_scoring",
        tier=TIER_EXTERNAL,
        current_value={"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320},
        code_path="scripts/mc_pool_backtest.py:139 — ESPN_SCORING",
        live_symbol="scripts.mc_pool_backtest:ESPN_SCORING",
        derivation=(
            "ESPN's published round values, 'verified against real brackets rather than "
            "documentation'. Not a choice this project makes; changing it would model a "
            "different pool."
        ),
        status=STATUS_EXTERNAL,
    ),
    PoolDegreeOfFreedom(
        name="n_repeats",
        tier=TIER_STRUCTURAL,
        current_value=50,
        code_path="scripts/mc_pool_backtest.py:243 — N_REPEATS",
        live_symbol="scripts.mc_pool_backtest:N_REPEATS",
        valid_range=(10, 1000),
        derivation=(
            "Monte-Carlo budget, not a modelling choice: more repeats reduce opponent-sampling "
            "variance and converge on the same estimand. The canonical published contract "
            "overrides it to 100. Tier 2 because the direction is monotone and the limit is "
            "the true value."
        ),
        status=STATUS_NEVER_SEARCHED,
    ),
    PoolDegreeOfFreedom(
        name="n_model_brackets",
        tier=TIER_STRUCTURAL,
        current_value=50,
        code_path="scripts/mc_pool_backtest.py:244 — N_MODEL_BRACKETS",
        live_symbol="scripts.mc_pool_backtest:N_MODEL_BRACKETS",
        valid_range=(1, 500),
        derivation="Stochastic brackets sampled per mode per repeat. A variance budget, like n_repeats.",
        status=STATUS_NEVER_SEARCHED,
    ),
    PoolDegreeOfFreedom(
        name="referee_noise_std",
        tier=TIER_EXTERNAL,
        current_value=0.16,
        code_path="scripts/mc_pool_backtest.py:180 — REFEREE_NOISE_STD",
        live_symbol="scripts.mc_pool_backtest:REFEREE_NOISE_STD",
        valid_range=(0.02, 0.30),
        derivation=(
            "Lopez & Matthews (2015). Registered tier 1 by citation, but the code itself says "
            "it is 'NOT FIT TO THIS REPO'S DATA, and audit finding M2 is right to flag that' -- "
            "the source is a point-spread paper, and the one attempt to fit it locally "
            "(src/simulation/mc_calibration.py) produced 16-seeds rated above 1-seeds and was "
            "retired on 2026-09-11 rather than repaired. A citation is provenance for the "
            "magnitude, not for this application."
        ),
        status=STATUS_EXTERNAL,
        swept_by="referee_noise_std",
    ),
    PoolDegreeOfFreedom(
        name="pool_factor_threshold",
        tier=TIER_FREE,
        current_value=50,
        code_path="scripts/mc_pool_backtest.py:187 — _POOL_FACTOR_THRESHOLD",
        live_symbol="scripts.mc_pool_backtest:_POOL_FACTOR_THRESHOLD",
        valid_range=(0, 1000),
        derivation=(
            "Pool size above which the EV scorer engages its pool factor. 'Which is why the H6 "
            "field-size defect was invisible at every real pool size' -- i.e. at the canonical "
            "pool of 30 this threshold means construction ignores pool size entirely."
        ),
        status=STATUS_NEVER_SEARCHED,
    ),
    # --- The strategy family ------------------------------------------------
    PoolDegreeOfFreedom(
        name="all_modes_family_size",
        tier=TIER_FREE,
        current_value=79,
        code_path="scripts/mc_pool_backtest.py:344 — ALL_MODES",
        live_symbol="scripts.mc_pool_backtest:ALL_MODES",
        live_transform="len",
        valid_range=(1, 200),
        derivation=(
            "The size of the family the headline mode was selected from. This is the single "
            "largest degree of freedom in the search and the direct subject of finding H2. "
            "Measured, not merely counted, by scripts/pool_rdof_audit.py --multiplicity."
        ),
        status=STATUS_SEARCHED,
        contaminated_by=(2026,),
    ),
    PoolDegreeOfFreedom(
        name="removed_candidate_family_upset_aware",
        tier=TIER_FREE,
        current_value="removed 2026-05-03",
        code_path="scripts/mc_pool_backtest.py:4026-4034 — comment only; the code is gone",
        derivation=(
            "'Upset-aware candidates — REMOVED 2026-05-03... Result: 10.93% P(1st) WITH upset "
            "vs 11.20% WITHOUT.' Deleted because it lowered the reported number on the "
            "evaluation seasons. A 15-season aggregate is BACKTEST_YEARS, which includes 2026, "
            "whose tournament had finished that April -- so this decision saw the holdout. Not "
            "correctable by resampling: the specification no longer exists as code."
        ),
        status=STATUS_REMOVED_AFTER_MEASURING,
        contaminated_by=(2026,),
    ),
    PoolDegreeOfFreedom(
        name="removed_candidate_family_confidence_routed",
        tier=TIER_FREE,
        current_value="removed 2026-05-16",
        code_path="scripts/mc_pool_backtest.py:4036-4042 — comment only; the code is gone",
        derivation=(
            "'Confidence-routed candidates — REMOVED 2026-05-16... Result: 7.1% P(1st) WITH vs "
            "11.9% WITHOUT — severe regression.' Same contamination as the upset family: the "
            "decision date is after the 2026 final and the window included 2026."
        ),
        status=STATUS_REMOVED_AFTER_MEASURING,
        contaminated_by=(2026,),
    ),
    PoolDegreeOfFreedom(
        name="poolaware_selector_estimator",
        tier=TIER_FREE,
        current_value="binary_p1",
        code_path="scripts/mc_pool_backtest.py:4107-4113 — inline literal (the selection loop)",
        derivation=(
            "'Rank-based estimator (fraction-beaten) was tested 2026-05-16 and caused a severe "
            "regression: 7.1% vs 11.9% baseline.' The estimator was chosen by comparing the two "
            "on this metric, on this window."
        ),
        status=STATUS_TUNED_ON_METRIC,
        contaminated_by=(2026,),
    ),
    PoolDegreeOfFreedom(
        name="small_pool_mode_pruning",
        tier=TIER_FREE,
        current_value="opt_seed, opt_blend, opt_torvik, hedge_tv removed",
        code_path="scripts/mc_pool_backtest.py:426-433 — comment; SMALL_POOL_MODES aliases ALL_MODES",
        derivation=(
            "'opt_seed, opt_blend, opt_torvik, hedge_tv removed. 13-year backtest (N=1000): "
            "opt_* statistically significantly worse... Council decision 2026-04-12.' Dated "
            "2026-04-12, which is after the 2026 final."
        ),
        status=STATUS_REMOVED_AFTER_MEASURING,
        contaminated_by=(2026,),
    ),
    # --- The meta_region_poolaware recipe -----------------------------------
    PoolDegreeOfFreedom(
        name="poolaware_risk_levels",
        tier=TIER_FREE,
        current_value=[0.1, 0.3, 0.5, 0.7, 0.9],
        code_path="src/optimization/poolaware_recipe.py:30 — POOLAWARE_RISK_LEVELS",
        live_symbol="src.optimization.poolaware_recipe:POOLAWARE_RISK_LEVELS",
        live_transform="sorted",
        valid_range=(0.0, 1.0),
        derivation="No numeric provenance is recorded for this grid anywhere in the repo.",
        status=STATUS_NEVER_SEARCHED,
        swept_by="poolaware_risk_levels",
    ),
    PoolDegreeOfFreedom(
        name="poolaware_exhaustive_risks",
        tier=TIER_FREE,
        current_value=[0.3, 0.5, 0.7],
        code_path="src/optimization/poolaware_recipe.py:34 — POOLAWARE_EXHAUSTIVE_RISKS",
        live_symbol="src.optimization.poolaware_recipe:POOLAWARE_EXHAUSTIVE_RISKS",
        live_transform="sorted",
        valid_range=(0.0, 1.0),
        derivation=(
            "'far more expensive per candidate, so it gets 3 risk levels rather than 5' -- a "
            "compute justification for the grid's size, and none at all for its values."
        ),
        status=STATUS_NEVER_SEARCHED,
    ),
    PoolDegreeOfFreedom(
        name="tv_mass80_torvik_weight",
        tier=TIER_FREE,
        current_value=0.8,
        code_path="src/optimization/poolaware_recipe.py:37 — TV_MASS80_TORVIK_WEIGHT",
        live_symbol="src.optimization.poolaware_recipe:TV_MASS80_TORVIK_WEIGHT",
        valid_range=(0.0, 1.0),
        derivation=(
            "'Weight on torvik in the tv_mass80 mixed base.' That is the entire justification "
            "in the repo -- the audit's '0.8/0.2 blend with no provenance'. The base it defines "
            "was selected in 2 of 15 seasons, so the weight is load-bearing for those."
        ),
        status=STATUS_NEVER_SEARCHED,
    ),
    PoolDegreeOfFreedom(
        name="poolaware_base_order",
        tier=TIER_FREE,
        current_value=["tv", "mass_avg", "mass_best", "blend", "tv_mass80"],
        code_path="src/optimization/poolaware_recipe.py:70-89 — build_poolaware_prob_bases (inline order)",
        derivation=(
            "'ORDER IS LOAD-BEARING. The selector keeps the first candidate that achieves the "
            "best P(1st) (p1 > best_p1, strictly greater), so ties are broken by position. "
            "Reordering this list can change which bracket ships without changing any "
            "probability.' An undocumented degree of freedom the recipe itself flags."
        ),
        status=STATUS_NEVER_SEARCHED,
        swept_by="poolaware_base_order",
    ),
    PoolDegreeOfFreedom(
        name="poolaware_forced_champion_risk",
        tier=TIER_FREE,
        current_value=0.5,
        code_path="scripts/mc_pool_backtest.py:4001 — inline literal (candidate family (a))",
        valid_range=(0.0, 1.0),
        derivation=(
            "risk_level for the forced-1-seed-champion candidates. A bare literal in the middle "
            "of the candidate loop, with no comment and no sweep."
        ),
        status=STATUS_NEVER_SEARCHED,
    ),
    PoolDegreeOfFreedom(
        name="pa_trials",
        tier=TIER_FREE,
        current_value=500,
        code_path="scripts/mc_pool_backtest.py:2729, 4786, 5040 — three unshared literals",
        live_symbol="scripts.mc_pool_backtest:run_backtest.pa_trials",
        valid_range=(50, 5000),
        derivation=(
            "'This is why pa_trials was later raised 200->500 (the one accepted change in that "
            "family...)' (FINDINGS.md:137-139); the audit records it as 'raised 200->500 because "
            "it helped'. Provenance by its own effect on the number being reported, which is "
            "the definition of a researcher degree of freedom. Also duplicated across three "
            "call sites with no shared constant, so the three can drift apart."
        ),
        status=STATUS_TUNED_ON_METRIC,
        contaminated_by=(2026,),
        swept_by="pa_trials",
    ),
    PoolDegreeOfFreedom(
        name="blend_alpha",
        tier=TIER_FREE,
        current_value=0.5,
        code_path="scripts/mc_pool_backtest.py:440 — PoolHyperparameters.blend_alpha",
        live_symbol="scripts.mc_pool_backtest:PoolHyperparameters.blend_alpha",
        valid_range=(0.0, 1.0),
        derivation=(
            "One of only two fields in PoolHyperparameters, the dataclass whose docstring "
            "promises 'every field here must be fittable from train_years alone'. In the "
            "canonical run it is not fitted at all: default_pool_hyperparameters does "
            "'del train_years  # baseline is year-independent by design'. So the walk-forward "
            "firewall for pool-layer knobs exists and is unused."
        ),
        status=STATUS_NEVER_SEARCHED,
    ),
    PoolDegreeOfFreedom(
        name="fixed_alpha_grid_at_risk_35",
        tier=TIER_FREE,
        current_value=[0.0, 0.25, 0.5, 0.75, 1.0],
        code_path="scripts/mc_pool_backtest.py:3212-3230 — inline literal (0, 25, 50, 75, 100)",
        valid_range=(0.0, 1.0),
        derivation=(
            "The alpha sweep for the fixed_blend* modes, pinned at risk 0.35. The run log at "
            ":3183-3212 records the sweep and admits the plateau: 'alpha=1.0 tops the P(1st) "
            "column and does not survive'."
        ),
        status=STATUS_SEARCHED,
        contaminated_by=(2026,),
    ),
    PoolDegreeOfFreedom(
        name="fixed_risk_grid",
        tier=TIER_FREE,
        current_value=[0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9],
        code_path="scripts/mc_pool_backtest.py:3235 — inline literal (10, 20, 30, 40, 50, 70, 90)",
        valid_range=(0.0, 1.0),
        derivation=(
            "Carries the sharpest self-reported RDoF confession in the repo "
            "(sample_fixed_region_risk, :1183-1225): 'CAVEAT ON THIS COMPARISON. The frozen "
            "levels were chosen after looking at which levels meta selects across these same "
            "15 seasons, so the margins are optimistic.'"
        ),
        status=STATUS_TUNED_ON_METRIC,
        contaminated_by=(2026,),
    ),
    PoolDegreeOfFreedom(
        name="meta_region_blend_weights",
        tier=TIER_FREE,
        current_value=[0.9, 0.1],
        code_path="scripts/mc_pool_backtest.py:4200-4204 — inline literal",
        valid_range=(0.0, 1.0),
        derivation="'Light blend: 90% torvik + 10% GBM' — the comment restates the numbers, it does not justify them.",
        status=STATUS_NEVER_SEARCHED,
    ),
    # --- Opponent model -----------------------------------------------------
    PoolDegreeOfFreedom(
        name="pool_chalk_noise_std",
        tier=TIER_FREE,
        current_value=0.0,
        code_path="scripts/mc_pool_backtest.py:2670 — local initial value in _run_one_year",
        valid_range=(0.0, 0.5),
        derivation=(
            "Bracket-level correlation for synthetic opponents, zero on the canonical path. "
            "Zero means opponents are independent draws with no chalk clustering -- audit "
            "finding H3. src/simulation/pool_history_opponent_model.py uses 0.15 as a "
            "'conservative default' and pool_competition.compute_bracket_win_probability "
            "defaults to 0.4, so the repo holds three different answers."
        ),
        status=STATUS_NEVER_SEARCHED,
    ),
    PoolDegreeOfFreedom(
        name="pool_blend_weight",
        tier=TIER_FREE,
        current_value=0.7,
        code_path="scripts/mc_pool_backtest.py:2728 — run_backtest.pool_blend_weight",
        live_symbol="scripts.mc_pool_backtest:run_backtest.pool_blend_weight",
        valid_range=(0.0, 1.0),
        derivation=(
            "Only read on the --opponent pool_calibrated path, which the canonical contract "
            "does not use. FINDINGS records the sweep: 'blend weights 0.0-0.7 of a cross-year "
            "pool model: all degrade P(1st) monotonically.'"
        ),
        status=STATUS_SEARCHED,
        contaminated_by=(2026,),
    ),
    PoolDegreeOfFreedom(
        name="pool_competition_noise_std",
        tier=TIER_FREE,
        current_value=0.16,
        code_path="src/simulation/pool_competition.py:95 — compute_bracket_win_probability.noise_std",
        live_symbol="src.simulation.pool_competition:compute_bracket_win_probability.noise_std",
        valid_range=(0.02, 0.30),
        derivation=(
            "Its own docstring says it 'should match the main MC engine (default 0.12)' while "
            "the value is 0.16 and the main engine is also 0.16. The documentation is wrong "
            "about both; registered so the contradiction is visible rather than latent."
        ),
        status=STATUS_NEVER_SEARCHED,
    ),
    PoolDegreeOfFreedom(
        name="seed_outcome_window",
        tier=TIER_FREE,
        current_value="recent",
        code_path="src/prediction/seed_probabilities.py:33 — OUTCOME_WINDOW",
        live_symbol="src.prediction.seed_probabilities:OUTCOME_WINDOW",
        derivation=(
            "'THIS MODULE IS AN OUTCOME MODEL, SO IT USES THE RECENT WINDOW... Keeping both on "
            "one window was the previous behaviour and it quietly cancelled itself.' Sets the "
            "referee to 2010-2025 seed head-to-heads -- a window that overlaps every backtested "
            "season, which is the circularity of audit finding C2."
        ),
        status=STATUS_SEARCHED,
        contaminated_by=(2026,),
    ),
    PoolDegreeOfFreedom(
        name="seed_recent_first_season",
        tier=TIER_FREE,
        current_value=2010,
        code_path="src/data/seed_pick_model.py:131 — RECENT_FIRST_SEASON",
        live_symbol="src.data.seed_pick_model:RECENT_FIRST_SEASON",
        valid_range=(1985, 2020),
        derivation=(
            "The cut defining 'recent'. Justified by an effect size — '6-11: the favourite wins "
            "62.2% across the full history and 48.3% since 2010' — which is a reason the cut "
            "matters, not a reason it falls in 2010."
        ),
        status=STATUS_NEVER_SEARCHED,
    ),
    PoolDegreeOfFreedom(
        name="seed_recent_prior_strength",
        tier=TIER_FREE,
        current_value=8,
        code_path="src/data/seed_pick_model.py — _RECENT_PRIOR_STRENGTH",
        live_symbol="src.data.seed_pick_model:_RECENT_PRIOR_STRENGTH",
        valid_range=(0, 50),
        derivation="Shrinkage strength on the recent-window seed table. No provenance recorded.",
        status=STATUS_NEVER_SEARCHED,
    ),
    PoolDegreeOfFreedom(
        name="seed_recent_min_games",
        tier=TIER_FREE,
        current_value=8,
        code_path="src/data/seed_pick_model.py — _RECENT_MIN_GAMES",
        live_symbol="src.data.seed_pick_model:_RECENT_MIN_GAMES",
        valid_range=(0, 50),
        derivation="Minimum games before the recent window is trusted over the prior. No provenance recorded.",
        status=STATUS_NEVER_SEARCHED,
    ),
    # --- The shipped candidate bank (a different recipe again; finding C3) ---
    PoolDegreeOfFreedom(
        name="candidate_bank_pool_size",
        tier=TIER_FREE,
        current_value=30,
        code_path="scripts/experiments/build_candidate_artifact.py:84 — DEFAULT_POOL_SIZE",
        live_symbol="scripts.experiments.build_candidate_artifact:DEFAULT_POOL_SIZE",
        valid_range=(2, 1000),
        derivation="'opponent field assumed by every P(1st) in the artifact' — 30 entries, i.e. 29 opponents.",
        status=STATUS_NEVER_SEARCHED,
    ),
    PoolDegreeOfFreedom(
        name="candidate_bank_risk_grid",
        tier=TIER_FREE,
        current_value=[0.1, 0.2, 0.35, 0.5, 0.7],
        code_path="scripts/experiments/build_candidate_artifact.py:810 — inline literal",
        valid_range=(0.0, 1.0),
        derivation=(
            "'The risk grid spans the measured plateau (0.2-0.5) plus its edges.' A DIFFERENT "
            "grid from the backtest's (0.1, 0.3, 0.5, 0.7, 0.9), and the plateau it spans was "
            "measured on the evaluation seasons."
        ),
        status=STATUS_TUNED_ON_METRIC,
        contaminated_by=(2026,),
    ),
    PoolDegreeOfFreedom(
        name="blend_region_35_risk",
        tier=TIER_FREE,
        current_value=0.35,
        code_path="scripts/experiments/build_candidate_artifact.py:503 — _blend_region_bracket.risk",
        live_symbol="scripts.experiments.build_candidate_artifact:_blend_region_bracket.risk",
        valid_range=(0.0, 1.0),
        derivation=(
            "'0.35 is the middle of that plateau, picked for being unremarkable.' The shipped "
            "'Maximise chance of winning' strategy, and NOT meta_region_poolaware -- its own "
            "docstring records the opposite finding: 'CHOOSING THE LEVEL PER SEASON IS WORSE "
            "THAN FIXING IT. Walk-forward selection over the 21-config grid scores 0.1092 "
            "against 0.1317 for a fixed mid-plateau level.'"
        ),
        status=STATUS_FROZEN_AFTER_SEARCH,
        contaminated_by=(2026,),
    ),
    PoolDegreeOfFreedom(
        name="candidate_bank_min_per_champion",
        tier=TIER_FREE,
        current_value=8,
        code_path="scripts/experiments/build_candidate_artifact.py:275 — stratified_sample.min_per_champion",
        live_symbol="scripts.experiments.build_candidate_artifact:stratified_sample.min_per_champion",
        valid_range=(0, 100),
        derivation="'The floor is what keeps unlikely-but-plausible champions in the artifact.' Value unjustified.",
        status=STATUS_NEVER_SEARCHED,
    ),
    PoolDegreeOfFreedom(
        name="candidate_bank_ev_strata",
        tier=TIER_FREE,
        current_value=10,
        code_path="scripts/experiments/build_candidate_artifact.py:276 — stratified_sample.ev_strata",
        live_symbol="scripts.experiments.build_candidate_artifact:stratified_sample.ev_strata",
        valid_range=(1, 100),
        derivation=(
            "'the EV strata are what keep the low-EV / high-P(1st) region.' Value unjustified. "
            "Both this and min_per_champion ARE inside the hashed frozen 2027 spec, so they "
            "cannot be changed without a SPEC_VERSION bump."
        ),
        status=STATUS_NEVER_SEARCHED,
    ),
    # --- The diagnostic fitter that is not wired in -------------------------
    PoolDegreeOfFreedom(
        name="recency_alpha_grid",
        tier=TIER_FREE,
        current_value=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        code_path="src/optimization/recency_hparam_fitter.py:66 — _DEFAULT_ALPHA_GRID",
        live_symbol="src.optimization.recency_hparam_fitter:_DEFAULT_ALPHA_GRID",
        live_transform="sorted",
        valid_range=(0.0, 1.0),
        derivation=(
            "The one real walk-forward fitter in the repo, and it is 'Diagnostic only: this "
            "module is NOT wired into scripts/generate_poolaware_bracket.py'. Registered "
            "because the walk-forward discipline the headline claims is implemented here and "
            "not used."
        ),
        status=STATUS_NEVER_SEARCHED,
    ),
    PoolDegreeOfFreedom(
        name="recency_window_years",
        tier=TIER_FREE,
        current_value=3,
        code_path="src/optimization/recency_hparam_fitter.py:120 — RecencyAlphaFitter.window_years",
        live_symbol="src.optimization.recency_hparam_fitter:RecencyAlphaFitter.window_years",
        valid_range=(1, 15),
        derivation="Trailing seasons the fitter tunes on. No provenance recorded.",
        status=STATUS_NEVER_SEARCHED,
    ),
    PoolDegreeOfFreedom(
        name="recency_fitter_n_opponents",
        tier=TIER_FREE,
        current_value=30,
        code_path="src/optimization/recency_hparam_fitter.py — RecencyAlphaFitter.n_opponents",
        live_symbol="src.optimization.recency_hparam_fitter:RecencyAlphaFitter.n_opponents",
        valid_range=(9, 999),
        derivation=(
            "30, against the canonical contract's 29 — a 31-person pool, not a 30-person one. "
            "This is audit recommendation 15, and the module's own LOAD-BEARING CAVEAT says "
            "why it matters: 'the fitter silently tunes blend_alpha against a different "
            "opponent field than the one actually used for selection. Nothing in the "
            "HparamFitter protocol enforces this — it is pure documentation discipline.'"
        ),
        status=STATUS_NEVER_SEARCHED,
    ),
    PoolDegreeOfFreedom(
        name="recency_pa_trials_fit",
        tier=TIER_FREE,
        current_value=100,
        code_path="src/optimization/recency_hparam_fitter.py:127 — RecencyAlphaFitter.pa_trials_fit",
        live_symbol="src.optimization.recency_hparam_fitter:RecencyAlphaFitter.pa_trials_fit",
        valid_range=(10, 5000),
        derivation="100 against production's 500, so the fitter selects under 5x the selection noise production has.",
        status=STATUS_NEVER_SEARCHED,
    ),
    PoolDegreeOfFreedom(
        name="recency_risk_grids_duplicated",
        tier=TIER_STRUCTURAL,
        current_value=[0.1, 0.3, 0.5, 0.7, 0.9],
        code_path="src/optimization/recency_hparam_fitter.py:64 — _REGION_RISK_LEVELS",
        live_symbol="src.optimization.recency_hparam_fitter:_REGION_RISK_LEVELS",
        live_transform="sorted",
        valid_range=(0.0, 1.0),
        derivation=(
            "A hand-copy of poolaware_recipe.POOLAWARE_RISK_LEVELS rather than an import, so "
            "the two can drift silently. Tier 2 because it is not an independent choice — it "
            "is supposed to equal the recipe's grid, and a test now asserts that."
        ),
        status=STATUS_NEVER_SEARCHED,
    ),
    # --- Pool-tool inputs, added under recommendation 13 (2026-09-13) -------
    #
    # These are user-facing inputs rather than researcher choices -- a pool
    # either pays its top three or it does not. But each has a DEFAULT, and a
    # default is a choice, so they are registered like any other. All three
    # were chosen to preserve existing behaviour exactly, which is why the
    # headline is unchanged by their existence.
    PoolDegreeOfFreedom(
        name="payout_structure",
        tier=TIER_STRUCTURAL,
        current_value="winner_take_all",
        code_path="scripts/mc_pool_backtest.py — run_backtest.payout",
        live_symbol="scripts.mc_pool_backtest:run_backtest.payout",
        derivation=(
            "The pool's own rules, not a modelling choice -- tier 2 for the same reason "
            "ESPN_SCORING is tier 1. The default is winner_take_all because that is what the "
            "published headline measures and what the real pool behind pool_hist_results.json "
            "pays. Selection maximises expected share of the pot under any other setting; "
            "measured 2026-09-13 to change the selected bracket in 1 of 14 seasons under "
            "top_3 and 11 of 14 under top_25pct."
        ),
        status=STATUS_NEVER_SEARCHED,
    ),
    PoolDegreeOfFreedom(
        name="n_entries",
        tier=TIER_STRUCTURAL,
        current_value=1,
        code_path="scripts/mc_pool_backtest.py — run_backtest.n_entries",
        live_symbol="scripts.mc_pool_backtest:run_backtest.n_entries",
        valid_range=(1, 25),
        derivation=(
            "How many brackets the user enters. One by default, matching every published "
            "figure. Measured 2026-09-13: the 2nd and 3rd entries raise P(winning the pool) "
            "by +4.21pp (p=.018) and +5.00pp (p=.010); the 4th by +1.86pp (p=.45), which is "
            "not distinguishable from zero. Expected prize per entry falls throughout."
        ),
        status=STATUS_NEVER_SEARCHED,
    ),
    PoolDegreeOfFreedom(
        name="pool_factor_mode",
        tier=TIER_FREE,
        current_value="threshold",
        code_path="scripts/mc_pool_backtest.py — run_backtest.pool_factor_mode",
        live_symbol="scripts.mc_pool_backtest:run_backtest.pool_factor_mode",
        derivation=(
            "Whether pool size reaches bracket construction below 51 entries. SWEPT under "
            "recommendation 13 and the production value KEPT: 'continuous' scored +1.93pp on "
            "the mean, clearing the 1.5pp season-level SE, but won only 6 of 14 seasons and "
            "nearly half its gain is 2011 alone (.020 -> .150). The pre-registered rule "
            "required both a mean gain and 9 of 14 seasons; adopting on the mean alone would "
            "have been a forking path on the very metric H2 is about. 'off' scores 0.1200, "
            "identical to 'threshold' -- direct proof the gate never fires at a real pool size."
        ),
        status=STATUS_SEARCHED,
        contaminated_by=(),
        swept_by="pool_factor_mode",
    ),
    # --- Evaluation window --------------------------------------------------
    PoolDegreeOfFreedom(
        name="evaluation_years",
        tier=TIER_STRUCTURAL,
        current_value=[2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025],
        code_path="scripts/mc_pool_backtest.py:117 — EVALUATION_YEARS",
        live_symbol="scripts.mc_pool_backtest:EVALUATION_YEARS",
        live_transform="sorted",
        derivation=(
            "2011 floor because TRAIN_YEARS starts 2008 and the walk-forward contract needs at "
            "least 3 prior seasons; 2020 absent (no tournament); 2026 stripped as contaminated. "
            "Tier 2: every boundary is forced by something other than preference."
        ),
        status=STATUS_NEVER_SEARCHED,
    ),
    PoolDegreeOfFreedom(
        name="contaminated_eval_years",
        tier=TIER_STRUCTURAL,
        current_value=[2026],
        code_path="scripts/mc_pool_backtest.py:116 — CONTAMINATED_EVAL_YEARS",
        live_symbol="scripts.mc_pool_backtest:CONTAMINATED_EVAL_YEARS",
        live_transform="sorted",
        derivation=(
            "Added 2026-09-09 in a3fb412 — four months AFTER the candidate-family prunings of "
            "2026-05-03 and 2026-05-16 that used a 15-season aggregate including 2026. The "
            "guard is correct and arrived too late to have prevented what it guards against."
        ),
        status=STATUS_NEVER_SEARCHED,
    ),
    PoolDegreeOfFreedom(
        name="per_year_rng_seeds",
        tier=TIER_STRUCTURAL,
        current_value=[42, 77777, 54321, 99999],
        code_path="scripts/mc_pool_backtest.py:3342, 3610, 3927, 3960, 4156 — inline literals (seed + year)",
        derivation=(
            "Five independent per-year streams. Tier 2: arbitrary but inconsequential, since "
            "each is offset by `year` and the estimand does not depend on the seed. The 77777 "
            "stream is regression-pinned by tests/test_parallel_run_backtest.py, so it is now "
            "load-bearing for reproducibility even though it was arbitrary when chosen."
        ),
        status=STATUS_NEVER_SEARCHED,
    ),
)

# Module-level constants in the searched files that are deliberately NOT
# degrees of freedom. Enumerated explicitly so the completeness test can tell
# "not a knob" from "forgotten" -- the distinction the deleted module lost.
_NOT_A_DOF: Dict[str, str] = {
    "SEED_MATCHUP_ORDER": "the NCAA's bracket structure, not a choice",
    "REGION_ORDER": "fallback only; the real F4 pairing is derived per year by derive_f4_region_pairing",
    "ROUND_NAMES": "the tournament's rounds",
    "BACKTEST_YEARS": "registered via evaluation_years, which is the window aggregates use",
    "ALL_MODES": "registered as all_modes_family_size",
    "SMALL_POOL_MODES": "a no-op alias of ALL_MODES since 2026-04-12",
    "LEGACY_MODE_MAP": "documentation/back-compat mapping; not the dispatch table that builds brackets",
    "PROBABILITY_BASES": "the --bases/--construction-modes cross-product interface, not the poolaware recipe",
    "CONSTRUCTION_MODES": "as PROBABILITY_BASES",
    "LOG_DIR": "an output path",
    "HIST_DIR": "an input path",
    "POOL_HIST_PATH": "an input path",
    "PROJECT_ROOT": "an input path",
    "_TOP_QUADRANT_SEEDS": "bracket geometry",
    "_BOTTOM_QUADRANT_SEEDS": "bracket geometry",
    "_LOCK_ROUND_INDEX": "bracket geometry",
    "_VALID_PRETOURNAMENT_TYPES": "a leakage guard on input data, not a modelling knob",
    "N_EVALUATION_SEASONS": "defined in this module, not in the search",
}


def registry() -> Tuple[PoolDegreeOfFreedom, ...]:
    return REGISTRY


def by_tier(tier: int) -> List[PoolDegreeOfFreedom]:
    return [d for d in REGISTRY if d.tier == tier]


def by_status(status: str) -> List[PoolDegreeOfFreedom]:
    if status not in ALL_STATUSES:
        raise ValueError(f"unknown status {status!r}; known: {ALL_STATUSES}")
    return [d for d in REGISTRY if d.status == status]


# ---------------------------------------------------------------------------
# Drift: does the registry still describe the code?
# ---------------------------------------------------------------------------


class UnresolvableSymbol(Exception):
    """A ``live_symbol`` could not be imported or found."""


def live_value(dof: PoolDegreeOfFreedom) -> Any:
    """Resolve ``dof.live_symbol`` against the live code.

    Accepts ``"module.path:attr"`` and ``"module.path:callable.param"``; the
    second form reads a parameter default via :mod:`inspect`, which also works
    for dataclass fields because a dataclass's signature carries its defaults.

    Raises:
        UnresolvableSymbol: if the entry has no symbol, or the symbol does not
            resolve. Callers distinguish "cannot check" from "checked and
            wrong" -- conflating the two is how the original registry drifted.
    """
    if not dof.live_symbol:
        raise UnresolvableSymbol(f"{dof.name} has no live_symbol (inline literal at {dof.code_path})")
    module_path, _, attr_path = dof.live_symbol.partition(":")
    if not attr_path:
        raise UnresolvableSymbol(f"{dof.name}: malformed live_symbol {dof.live_symbol!r}, expected 'module:attr'")
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise UnresolvableSymbol(f"{dof.name}: cannot import {module_path}: {exc}") from exc

    parts = attr_path.split(".")
    if len(parts) == 1:
        try:
            raw = getattr(module, parts[0])
        except AttributeError as exc:
            raise UnresolvableSymbol(f"{dof.name}: {module_path} has no attribute {parts[0]}") from exc
    elif len(parts) == 2:
        owner_name, param_name = parts
        try:
            owner = getattr(module, owner_name)
        except AttributeError as exc:
            raise UnresolvableSymbol(f"{dof.name}: {module_path} has no attribute {owner_name}") from exc
        try:
            params = inspect.signature(owner).parameters
        except (TypeError, ValueError) as exc:
            raise UnresolvableSymbol(f"{dof.name}: {owner_name} has no inspectable signature") from exc
        if param_name not in params:
            raise UnresolvableSymbol(f"{dof.name}: {owner_name} has no parameter {param_name}")
        default = params[param_name].default
        if default is inspect.Parameter.empty:
            raise UnresolvableSymbol(f"{dof.name}: {owner_name}.{param_name} has no default")
        raw = default
    else:
        raise UnresolvableSymbol(f"{dof.name}: live_symbol {dof.live_symbol!r} is too deeply nested")

    transform = _TRANSFORMS.get(dof.live_transform, _TRANSFORMS[None])
    return transform(raw)


def drift_report() -> Dict[str, Any]:
    """Compare every resolvable registry entry against the live code.

    Shape deliberately mirrors :func:`src.governance.frozen_spec.diff_against_frozen`
    so the two gates read alike.
    """
    drifted: List[Dict[str, Any]] = []
    unresolvable: List[Dict[str, str]] = []
    checked = 0
    for dof in REGISTRY:
        try:
            live = live_value(dof)
        except UnresolvableSymbol as exc:
            unresolvable.append({"name": dof.name, "reason": str(exc)})
            continue
        checked += 1
        if _jsonable(live) != _jsonable(dof.current_value):
            drifted.append(
                {
                    "name": dof.name,
                    "registered": _jsonable(dof.current_value),
                    "live": _jsonable(live),
                    "code_path": dof.code_path,
                }
            )
    return {
        "drifted": drifted,
        "unresolvable": unresolvable,
        "n_checked": checked,
        "n_registered": len(REGISTRY),
        "coverage": round(checked / len(REGISTRY), 3) if REGISTRY else 0.0,
    }


def registry_hash() -> str:
    """SHA-256 over the canonical registry, for holdout lockfiles.

    Only the fields that describe the *configuration* are hashed --
    ``derivation`` and ``code_path`` are prose and line numbers, and editing a
    comment must not read as a configuration change.
    """
    body = [
        {
            "name": d.name,
            "tier": d.tier,
            "current_value": _jsonable(d.current_value),
            "status": d.status,
        }
        for d in sorted(REGISTRY, key=lambda d: d.name)
    ]
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# DoF accounting
# ---------------------------------------------------------------------------


def dof_summary(effective_dof_by_knob: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Count the degrees of freedom the search spent, and the DoF/sample ratio.

    Args:
        effective_dof_by_knob: Optional ``{registry name: effective DoF}`` from
            the specification curve. Knobs measured flat are discounted toward
            zero; knobs not measured are counted at full weight, because an
            unmeasured knob cannot be assumed harmless.

    Returns:
        Counts per tier and status, the raw and effective DoF totals, and the
        ratio against :data:`N_EVALUATION_SEASONS`.
    """
    spending = [d for d in REGISTRY if d.status in _SPENDING_STATUSES]
    raw_dof = len(spending)

    measured = effective_dof_by_knob or {}
    effective = 0.0
    for d in spending:
        key = d.swept_by or d.name
        effective += measured.get(key, measured.get(d.name, 1.0))

    contaminated = [d for d in REGISTRY if d.contaminated_by]
    return {
        "n_registered": len(REGISTRY),
        "by_tier": {
            "1_external": len(by_tier(TIER_EXTERNAL)),
            "2_structural": len(by_tier(TIER_STRUCTURAL)),
            "3_freely_tuned": len(by_tier(TIER_FREE)),
        },
        "by_status": {s: len(by_status(s)) for s in ALL_STATUSES},
        "raw_dof_spent": raw_dof,
        "effective_dof_spent": round(effective, 2),
        "effective_dof_is_measured": bool(measured),
        "n_evaluation_seasons": N_EVALUATION_SEASONS,
        "dof_per_season": round(raw_dof / N_EVALUATION_SEASONS, 2),
        "effective_dof_per_season": round(effective / N_EVALUATION_SEASONS, 2),
        "n_contaminated_by_2026": len(contaminated),
        "contaminated_names": [d.name for d in contaminated],
        "note": (
            "The denominator is seasons, not season x repeat trials. Audit finding H1 "
            "established the season as the unit of independence and measured the inflation "
            "from using repeats at roughly 3x."
        ),
    }


# ---------------------------------------------------------------------------
# Holdout sequestration
# ---------------------------------------------------------------------------


def assert_not_sequestered(years: Iterable[int], context: str) -> None:
    """Refuse to let a sequestered season into a search or aggregate path.

    The point of a holdout is that it is enforceable rather than asserted. A
    comment saying "2027 is the holdout" is worth nothing the first time
    someone passes ``--years 2027`` to a sweep; this raises.

    A season stops being sequestered only when it has been evaluated and
    recorded, at which point the lockfile exists and the season is spent.

    Raises:
        HoldoutContaminationError: if any sequestered, unspent season appears.
    """
    offending = sorted(set(int(y) for y in years) & SEQUESTERED_YEARS)
    if not offending:
        return
    spent = {entry["year"] for entry in _read_lockfile().get("evaluations", [])}
    still_sealed = [y for y in offending if y not in spent]
    if still_sealed:
        raise HoldoutContaminationError(
            f"{context}: season(s) {still_sealed} are sequestered holdouts and must not enter "
            f"a search, sweep or aggregate. They are the only untouched evaluation this "
            f"project has. If you genuinely intend to spend one, evaluate it ONCE through "
            f"record_holdout_evaluation() and accept that it is spent thereafter."
        )


def _read_lockfile() -> Dict[str, Any]:
    if not HOLDOUT_LOCKFILE.exists():
        return {"evaluations": []}
    try:
        with open(HOLDOUT_LOCKFILE) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"evaluations": []}
    data.setdefault("evaluations", [])
    return data


def _registry_values() -> Dict[str, str]:
    """Per-knob values, for telling a behaviour change from a documentation one."""
    return {d.name: repr(_jsonable(d.current_value)) for d in REGISTRY}


def record_holdout_evaluation(year: int, evidence_level: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    """Record that a holdout season has been evaluated, and against what config.

    The registry hash is the payload that matters. Once a holdout has been
    scored, any later change to a registered knob means the configuration that
    produced the holdout result no longer exists -- so the result has stopped
    describing the shipped system, and a later re-evaluation is not
    out-of-sample any more. :func:`check_holdout_contamination` detects exactly
    that.
    """
    data = _read_lockfile()
    entry = {
        "year": int(year),
        "evidence_level": evidence_level,
        "registry_hash": registry_hash(),
        # Per-knob values as well as the hash, so a later check can tell a
        # knob that CHANGED (the holdout result no longer describes the
        # system) from a knob that was merely newly REGISTERED (the audit got
        # more complete; nothing about the system moved).
        "registry_values": _registry_values(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": summary,
    }
    # Re-evaluating a season replaces its entry but keeps the history, so a
    # quietly repeated "holdout" evaluation is visible rather than overwritten.
    data.setdefault("superseded", []).extend(e for e in data["evaluations"] if e["year"] == int(year))
    data["evaluations"] = [e for e in data["evaluations"] if e["year"] != int(year)] + [entry]
    HOLDOUT_LOCKFILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = HOLDOUT_LOCKFILE.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    tmp.replace(HOLDOUT_LOCKFILE)
    return entry


def check_holdout_contamination() -> List[Dict[str, Any]]:
    """Holdout evaluations whose CONFIGURATION no longer matches the live code.

    Distinguishes a behaviour change from a documentation one, the way
    ``frozen_spec.verify_freeze`` does for the methodology spec. A knob whose
    value changed means the holdout result has stopped describing the shipped
    system, and that is contamination. A knob that was merely added to the
    registry means the audit got more complete while the system stood still,
    and flagging that as contamination would punish improving the audit --
    which would quickly teach everyone not to improve it.

    Returns one record per genuinely stale evaluation; empty means every
    recorded holdout still describes the current system. Newly registered
    knobs are reported inside the record as ``newly_registered`` when a real
    change is present, and are otherwise silent.
    """
    current_values = _registry_values()
    stale = []
    for entry in _read_lockfile().get("evaluations", []):
        recorded = entry.get("registry_values")
        if recorded is None:
            # Pre-dates value recording. Fall back to the hash, which cannot
            # tell the two cases apart -- so say so rather than implying it can.
            if entry.get("registry_hash") != registry_hash():
                stale.append(
                    {
                        "year": entry.get("year"),
                        "evaluated_at": entry.get("timestamp"),
                        "changed_knobs": None,
                        "message": (
                            f"The {entry.get('year')} holdout was recorded before per-knob values "
                            "were stored, so only the registry hash can be compared and it "
                            "differs. That may be a real configuration change or merely a newly "
                            "registered knob; re-record the evaluation to get a precise answer."
                        ),
                    }
                )
            continue

        changed = {
            name: {"at_evaluation": recorded[name], "now": current_values[name]}
            for name in set(recorded) & set(current_values)
            if recorded[name] != current_values[name]
        }
        added = sorted(set(current_values) - set(recorded))
        removed = sorted(set(recorded) - set(current_values))
        if not changed and not removed:
            continue
        stale.append(
            {
                "year": entry.get("year"),
                "evaluated_at": entry.get("timestamp"),
                "changed_knobs": changed,
                "removed_knobs": removed,
                "newly_registered": added,
                "message": (
                    f"The {entry.get('year')} holdout was evaluated against a different "
                    f"configuration than the one now in the code: {sorted(changed) + removed}. "
                    "That result no longer describes the shipped system, and re-running it is "
                    "not an out-of-sample evaluation."
                ),
            }
        )
    return stale


def holdout_status() -> Dict[str, Any]:
    """The two tiers, their current state, and why the weaker one is weaker."""
    spent = {e["year"] for e in _read_lockfile().get("evaluations", [])}
    return {
        "level_1_sequestered": {
            "years": sorted(SEQUESTERED_YEARS),
            "spent": sorted(SEQUESTERED_YEARS & spent),
            "enforced_by": "assert_not_sequestered(), called by scripts/pool_rdof_audit.py --sweep",
            "status": "untouched" if not (SEQUESTERED_YEARS & spent) else "spent",
        },
        "level_2_5_parameter_clean": {
            "year": PARAMETER_CLEAN_HOLDOUT,
            "evaluated": PARAMETER_CLEAN_HOLDOUT in spent,
            "why_not_level_1": (
                "The candidate-family set was pruned using aggregates that included it: "
                "scripts/mc_pool_backtest.py:4026-4042 records family (d) removed 2026-05-03 "
                "('selected in only 1/15 years') and family (e) removed 2026-05-16, where a "
                "15-season window is BACKTEST_YEARS = 2011-2026 excluding 2020. The 2026 "
                "tournament ended that April, so its outcome was known. CONTAMINATED_EVAL_YEARS "
                "was not added until 2026-09-09 (a3fb412). The removed families no longer exist "
                "as code, so those decisions cannot be re-run on a 2026-free window."
            ),
            "what_is_clean": (
                "The prediction model never trains on it (train_noseed_model filters y < "
                "max_year) and every parametric knob is swept on 2011-2025 only."
            ),
        },
        "no_historical_holdout": (
            "2011-2025 were all both trained on (as priors for later seasons) and searched "
            "over, so none is a holdout. 2008-2010 are training-only and never evaluated, "
            "which is not the same thing. 2020 has no tournament. A never-touched holdout "
            "cannot be manufactured retroactively; it can only be sequestered going forward."
        ),
        "contamination_checks": check_holdout_contamination(),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class PoolRDOFReport:
    """Assembles the registry and whatever measurements exist into one report.

    Every measurement section is optional: a registry-only run must render, and
    so must a run where one long measurement has completed and the other has
    not. ``to_dict`` is a pure serialiser -- it does NOT generate
    recommendations as a side effect, which the deleted module's did.
    """

    multiplicity: Optional[Dict[str, Any]] = None
    sensitivity: Optional[Dict[str, Any]] = None
    holdout: Optional[Dict[str, Any]] = None
    _generated_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def effective_dof_by_knob(self) -> Dict[str, float]:
        if not self.sensitivity:
            return {}
        return {axis: spec["effective_dof"] for axis, spec in self.sensitivity.get("axes", {}).items()}

    def recommendations(self) -> List[str]:
        """Prefix-tagged findings, so they can be grepped out of the text report."""
        recs: List[str] = []
        drift = drift_report()
        if drift["drifted"]:
            for d in drift["drifted"]:
                recs.append(f"DRIFT: {d['name']} registered as {d['registered']!r} but the code says {d['live']!r}")
        else:
            recs.append(
                f"DRIFT: none. {drift['n_checked']}/{drift['n_registered']} entries verified against live code "
                f"({drift['coverage']:.0%} coverage); {len(drift['unresolvable'])} are inline literals with no "
                f"importable binding and cannot be machine-checked."
            )

        summary = dof_summary(self.effective_dof_by_knob())
        recs.append(
            f"DoF RATIO: {summary['raw_dof_spent']} degrees of freedom spent against "
            f"{summary['n_evaluation_seasons']} independent seasons = "
            f"{summary['dof_per_season']} per season. That is close to one free choice per "
            f"observation, i.e. a design with roughly as many tuning decisions as data "
            f"points, and it needs no external benchmark to be a problem: at this ratio the "
            f"in-sample fit of the search is uninformative about out-of-sample performance "
            f"almost regardless of what the fit is. Only time fixes the denominator -- one "
            f"season per year."
        )
        if summary["effective_dof_is_measured"]:
            recs.append(
                f"SENSITIVITY: after flatness discounting, {summary['effective_dof_spent']} effective DoF "
                f"({summary['effective_dof_per_season']} per season). Knobs measured flat cost ~0; "
                f"unmeasured knobs are counted at full weight because an unmeasured knob cannot be "
                f"assumed harmless."
            )
        else:
            recs.append("SENSITIVITY: not measured. Run --sweep; until then every knob counts at full weight.")

        removed = by_status(STATUS_REMOVED_AFTER_MEASURING)
        tuned = by_status(STATUS_TUNED_ON_METRIC)
        recs.append(
            f"CIRCULARITY: {len(tuned)} knob(s) were set because of their effect on this metric and "
            f"{len(removed)} candidate family/families were deleted for scoring worse on the evaluation "
            f"seasons. No resampling scheme can correct for the removed ones -- their code is gone. This "
            f"is the irreducible core of finding H2."
        )

        if self.multiplicity:
            m = self.multiplicity
            recs.append(
                f"MULTIPLICITY: {m['headline_mode']} aggregate P(1st) {m['headline_aggregate_p_first']:.4f} vs "
                f"baseline {m['baseline_aggregate_p_first']:.4f}; best-of-family p = {m['p_max_statistic']}, "
                f"stepdown-adjusted p = {m['headline_p_adjusted']} over {m['n_modes_in_family']} modes. "
                f"VERDICT {m['verdict']}."
            )
        else:
            recs.append("MULTIPLICITY: not measured. Run --multiplicity.")

        status = holdout_status()
        recs.append(
            f"HOLDOUT: Level 1 = {status['level_1_sequestered']['years']} "
            f"({status['level_1_sequestered']['status']}); Level 2.5 = {PARAMETER_CLEAN_HOLDOUT} "
            f"(evaluated={status['level_2_5_parameter_clean']['evaluated']}). No historical season is a "
            f"holdout and none can be made one retroactively."
        )
        for stale in status["contamination_checks"]:
            recs.append(f"HOLDOUT: CONTAMINATED — {stale['message']}")
        return recs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": {
                "report_type": "pool_strategy_researcher_degrees_of_freedom_audit",
                "generated_at": self._generated_at,
                "registry_hash": registry_hash(),
                "audit_finding": "H2",
                "audit_recommendation": 12,
            },
            "registry": [d.to_dict() for d in REGISTRY],
            "dof_summary": dof_summary(self.effective_dof_by_knob()),
            "drift": drift_report(),
            "holdout": holdout_status(),
            "multiplicity": self.multiplicity,
            "sensitivity": self.sensitivity,
            "holdout_evaluation": self.holdout,
            "recommendations": self.recommendations(),
        }

    def to_text(self) -> str:
        lines: List[str] = []
        w = 100
        lines.append("=" * w)
        lines.append("RESEARCHER DEGREES OF FREEDOM — POOL-STRATEGY SEARCH")
        lines.append(f"audit finding H2 / recommendation 12    generated {self._generated_at}")
        lines.append(f"registry hash {registry_hash()[:16]}")
        lines.append("=" * w)

        summary = dof_summary(self.effective_dof_by_knob())
        lines.append("\n1. INVENTORY")
        lines.append(f"   registered knobs: {summary['n_registered']}")
        for label, count in summary["by_tier"].items():
            lines.append(f"     tier {label:<16} {count}")
        lines.append("   by how the value was arrived at:")
        for status, count in summary["by_status"].items():
            if count:
                lines.append(f"     {status:<28} {count}")
        lines.append(f"   degrees of freedom spent: {summary['raw_dof_spent']}")
        if summary["effective_dof_is_measured"]:
            lines.append(f"   effective (flatness-discounted): {summary['effective_dof_spent']}")
        lines.append(f"   independent seasons: {summary['n_evaluation_seasons']}")
        lines.append(f"   DoF per season: {summary['dof_per_season']}")
        lines.append(f"   knobs whose choice saw 2026's outcome: {summary['n_contaminated_by_2026']}")

        lines.append("\n2. REGISTRY")
        for tier, label in ((TIER_FREE, "tier 3 — freely tuned"), (TIER_STRUCTURAL, "tier 2 — structural"), (TIER_EXTERNAL, "tier 1 — external")):
            entries = by_tier(tier)
            if not entries:
                continue
            lines.append(f"\n   {label} ({len(entries)})")
            for d in entries:
                flag = " [saw 2026]" if d.contaminated_by else ""
                lines.append(f"     {d.name}  =  {_jsonable(d.current_value)!r}   [{d.status}]{flag}")
                lines.append(f"       {d.code_path}")

        lines.append("\n3. DRIFT")
        drift = drift_report()
        lines.append(f"   verified against live code: {drift['n_checked']}/{drift['n_registered']} ({drift['coverage']:.0%})")
        lines.append(f"   drifted: {len(drift['drifted'])}")
        for d in drift["drifted"]:
            lines.append(f"     {d['name']}: registered {d['registered']!r}, live {d['live']!r}")
        lines.append(f"   not machine-checkable (inline literals): {len(drift['unresolvable'])}")

        lines.append("\n4. MULTIPLICITY — is the headline just the best of many?")
        if self.multiplicity:
            m = self.multiplicity
            lines.append(f"   family size:        {m['n_modes_in_family']} modes, {len(m['seasons'])} seasons")
            lines.append(f"   headline mode:      {m['headline_mode']}")
            lines.append(f"   aggregate P(1st):   {m['headline_aggregate_p_first']:.4f}")
            lines.append(f"   baseline P(1st):    {m['baseline_aggregate_p_first']:.4f}")
            lines.append(f"   best by t-stat:     {m.get('best_mode_by_test_statistic')}")
            circ = " [CIRCULAR — graded by its own probability source, see H11]" if m.get("best_mode_is_circular") else ""
            lines.append(
                f"   best by P(1st):     {m.get('best_mode_by_aggregate_p_first')} "
                f"({m.get('best_aggregate_p_first', float('nan')):.4f}, "
                f"{m.get('mean_gap_to_best_pp', 0.0):+.2f}pp vs headline){circ}"
            )
            if m.get("best_non_circular_mode"):
                lines.append(
                    f"   best non-circular:  {m['best_non_circular_mode']} "
                    f"({m.get('best_non_circular_p_first', float('nan')):.4f})"
                )
            for mode, why in (m.get("circular_modes") or {}).items():
                lines.append(f"     circular: {mode} — {why}")
            lines.append(f"   modes beating seed at FWER<0.05: {m.get('n_modes_surviving_fwer_05')}")
            lines.append(f"   p, uncorrected:     {m['headline_p_unadjusted']}")
            lines.append(f"   p, stepdown:        {m['headline_p_adjusted']}")
            lines.append(f"   p, best-of-family:  {m['p_max_statistic']}")
            lines.append(f"   VERDICT:            {m['verdict']}")
            lines.append(f"   not corrected:      {m['residual_not_corrected']}")
        else:
            lines.append("   not measured — run scripts/pool_rdof_audit.py --multiplicity")

        lines.append("\n5. SENSITIVITY — does the headline depend on the unjustified knobs?")
        if self.sensitivity:
            s = self.sensitivity
            lines.append(f"   swept on seasons: {s['n_seasons']} (2026 and 2027 excluded by construction)")
            lines.append(
                f"   {'knob':<28} {'spread':>9}  {'category':<14} {'eff DoF':>7} {'gap to best':>12}  prod best?"
            )
            for axis, spec in s["axes"].items():
                gap = spec.get("gap_to_best_pp")
                gap_s = "—" if gap is None else f"{gap:+.2f}pp"
                lines.append(
                    f"   {axis:<28} {spec['spread_pp']:>7.2f}pp  {spec['category']:<14} "
                    f"{spec['effective_dof']:>7.1f} {gap_s:>12}  {spec['production_is_best']}"
                )
            lines.append(f"   total effective DoF over swept knobs: {s['total_effective_dof']}")
            lines.append(f"   {s['design_note']}")
        else:
            lines.append("   not measured — run scripts/pool_rdof_audit.py --sweep")

        lines.append("\n6. HOLDOUT")
        status = holdout_status()
        lines.append(f"   Level 1 (untouched):        {status['level_1_sequestered']['years']} — {status['level_1_sequestered']['status']}")
        lines.append(f"     enforced by: {status['level_1_sequestered']['enforced_by']}")
        lines.append(f"   Level 2.5 (param-clean):    {PARAMETER_CLEAN_HOLDOUT} — evaluated={status['level_2_5_parameter_clean']['evaluated']}")
        if self.holdout:
            lines.append(f"     P(1st): {self.holdout.get('p_first')}")
        lines.append(f"     why not Level 1: {status['level_2_5_parameter_clean']['why_not_level_1']}")
        lines.append(f"   {status['no_historical_holdout']}")

        lines.append("\n7. FINDINGS")
        for rec in self.recommendations():
            lines.append(f"   {rec}")

        lines.append("\n" + "=" * w)
        lines.append("DISCLOSURE")
        lines.append(
            "   This audit counts and measures the search that produced the headline; it does "
            "not repair it. The strategy was selected on the same 14 seasons it is reported on, "
            "so the figure remains in-sample for strategy choice however it is corrected. The "
            "first genuinely prospective evaluation is 2027, n=1."
        )
        lines.append("=" * w)
        return "\n".join(lines)
