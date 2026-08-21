"""Frozen prospective specification — the 2027 holdout contract, in code.

WHY THIS EXISTS
---------------
2026 is contaminated as a research benchmark: it sits inside ``BACKTEST_YEARS``,
the production strategy was selected on a window containing it, and CLAUDE.md
records a modelling conclusion drawn from its outcome ("2026 upset-year anomaly
blocks further gains with current feature set"). No amount of care now makes a
2026 evaluation out-of-sample.

2027 has not happened. It is therefore the first genuinely prospective
evaluation this project can run — but only if the system is pinned *before* any
2027 outcome exists, and pinned in a way that can be audited afterwards rather
than asserted.

This module captures the frozen parameters from the live code, hashes them
canonically, and provides the comparison that CI uses to detect drift. The point
is that "frozen" becomes a checkable fact instead of an intention.

WHAT FREEZING DOES AND DOES NOT MEAN
------------------------------------
It does NOT mean ignoring 2027 data. Seeds, ratings and public pick percentages
that a user would genuinely have had before tip-off are exactly what the system
is supposed to consume. The freeze is on the *system*, not the inputs:

    allowed      information available before the stated prediction cutoff
    allowed      historical seasons through 2026, per the frozen training spec
    NOT allowed  2027 tournament outcomes
    NOT allowed  post-cutoff information of any kind
    NOT allowed  tuning any frozen parameter in response to a 2027 result

Changing a frozen value is permitted. Doing it silently is not: bump
``SPEC_VERSION``, which invalidates the prospective claim for this version and
starts a new one.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

SPEC_VERSION = "2027.v1"
FREEZE_DATE = "2026-08-20"
FROZEN_SPEC_PATH = Path("configs/frozen/prospective_2027.json")


def capture_live_spec() -> Dict[str, Any]:
    """Read the frozen parameters out of the live code.

    Deliberately introspective rather than transcribed: a hand-copied spec
    drifts silently from the system it claims to describe, which is the failure
    mode this whole exercise exists to prevent.
    """
    import inspect

    from scripts.experiments.build_candidate_artifact import (
        DEFAULT_POOL_SIZE,
        stratified_sample,
    )
    from scripts.mc_pool_backtest import ESPN_SCORING
    from src.prediction.noseed_model import REQUIRED_FEATURE_KEYS, TRAIN_YEARS
    from src.prediction.pairwise import log5

    sampler_defaults = {
        k: v.default
        for k, v in inspect.signature(stratified_sample).parameters.items()
        if v.default is not inspect.Parameter.empty
    }

    return {
        "spec_version": SPEC_VERSION,
        "freeze_date": FREEZE_DATE,
        "model": {
            "training_cutoff_season": max(TRAIN_YEARS),
            "train_years": sorted(TRAIN_YEARS),
            "feature_count": len(REQUIRED_FEATURE_KEYS),
            "feature_keys": sorted(REQUIRED_FEATURE_KEYS),
            "blend_alpha_default": 0.5,
            "noseed_architecture": "logistic + GBM ensemble, 50/50, symmetric augmentation",
        },
        "tournament_engine": {
            "pairwise_construction": "log5 over barthag-equivalent ratings",
            "log5_probe_0.9_vs_0.5": log5(0.9, 0.5),
            "scenario_bank_noise_std": 0.0,
            "scenario_bank_size": 150000,
            "propagation": "src.prediction.pairwise.simulate_bracket_outcomes",
            "marginals_direction": "pairwise -> simulator -> marginals (never reversed)",
        },
        "candidate_selection": {
            "sampler": "champion quotas (proportional with floor) then EV strata within champion",
            "min_per_champion": sampler_defaults.get("min_per_champion"),
            "ev_strata": sampler_defaults.get("ev_strata"),
            "target_candidates": 3000,
            "p1_trials": 2000,
            "p1_pool_size": DEFAULT_POOL_SIZE,
            "p1_opponent_model": "ESPN public pick distribution + seed pairwise referee",
            "common_random_numbers": True,
            "objectives": ["ev", "p1"],
            "ev_definition": "sum_R pts_R * sum_{t in picked_R} P(t wins R), marginals from the unconditional bank",
            "p1_definition": "P(bracket score >= max opponent score) over shared trials",
            "diversity_algorithm": "hierarchical: distinct champion -> distinct Final Four -> points-weighted distance",
            "k_returned": 3,
            "scoring_system": dict(ESPN_SCORING),
            "scoring_mode": "team_identity (never shape-encoded)",
        },
        "preferences": {
            "predicates": [
                "f4_at_least_1_two_three",
                "f4_at_least_2_two_three",
                "f4_mostly_favorites",
                "s16_at_least_1_double_digit",
                "s16_at_least_2_double_digit",
                "s16_no_double_digit",
                "team_reaches_final_four",
            ],
            "frequencies_source": "full scenario bank, never the candidate artifact",
        },
        "product": {
            "strategies": [
                {"name": "Trust the Model", "objective": "ev", "constraint": None},
                {"name": "Win My Pool", "objective": "p1", "constraint": None},
                {"name": "Your Preference", "objective": "user-selected", "constraint": "user-selected"},
            ],
            "excluded_from_v1": ["balanced blend", "contrarian ownership penalty", "configurable pool size"],
            "p1_disclosure_required": (
                "P(1st) assumes a 30-opponent pool with ESPN public pick behaviour. "
                "It is not a universal probability of winning any pool."
            ),
        },
        "holdout": {
            "contaminated_seasons": [2026],
            "contamination_reason": (
                "2026 is inside BACKTEST_YEARS, the production strategy was selected on a "
                "window containing it, and a documented modelling conclusion was drawn from "
                "its outcome. It is an integration/regression season, not a benchmark."
            ),
            "prospective_season": 2027,
            "outcomes_available_at_freeze": False,
        },
    }


def canonical_hash(spec: Dict[str, Any]) -> str:
    """SHA-256 over canonical JSON (sorted keys, no whitespace)."""
    blob = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def load_frozen_spec(path: Path = FROZEN_SPEC_PATH) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def diff_against_frozen(path: Path = FROZEN_SPEC_PATH) -> Dict[str, Any]:
    """Compare live code against the frozen spec.

    Returns ``{"drifted": [...], "frozen_hash": ..., "live_hash": ...}``. Empty
    ``drifted`` means the system still matches what was frozen.
    """
    frozen = load_frozen_spec(path)
    live = capture_live_spec()
    frozen_body = {k: v for k, v in frozen.items() if k != "spec_hash"}

    drifted = []

    def walk(a, b, trail):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                walk(a.get(key), b.get(key), trail + [key])
        elif a != b:
            drifted.append({"path": ".".join(trail), "frozen": a, "live": b})

    walk(frozen_body, live, [])
    return {
        "drifted": drifted,
        "frozen_hash": frozen.get("spec_hash"),
        "live_hash": canonical_hash(live),
        "spec_version": frozen_body.get("spec_version"),
    }
