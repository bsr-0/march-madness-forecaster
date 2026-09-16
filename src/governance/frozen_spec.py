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

SPEC_VERSION = "2027.v3"
FREEZE_DATE = "2026-09-15"
PROSPECTIVE_DOC = Path("artifacts/methodology_audit/step18/PROSPECTIVE_2027_v3.md")

# 2027.v3 -- the post-audit freeze. Supersedes 2027.v2 (scoped) because the
# methodology audit of 2026-09-15 (artifacts/methodology_audit/) corrected the
# methodology v2 described: the Final Four topology, the P(1st) definition,
# the walk-forward seed and noseed tables, the scoring default, the opponent
# fallback and the artifact pool size. v1 and v2 files are immutable history.
SUPERSEDES_V2 = {
    "version": "2027.v2",
    "spec_path": "configs/frozen/prospective_2027_v2_scoped.json",
    "spec_hash": "c7d1c67663601bd4b28758ca368d3de70a18daaa15d60d18b74fa500d3109018",
    "reason_superseded": (
        "Methodology audit 2026-09-15 (artifacts/methodology_audit/, Steps 1-12, R-4): "
        "construction and scoring walked different Final Four pairings (hardcoded East-West/"
        "South-Midwest vs the real per-season pairing, wrong in 9 of 15 seasons); P(1st) meant "
        "three different quantities (>= in selection, rank==1 in reporting, split in payout) and is "
        "now the expected first-place share with ties split; the seed table and the noseed marginals "
        "were not walk-forward; the shape-encoded scorer was the harness default; 2012 opponents came "
        "from 2023-26 pool data; the artifact's P(1st) pool had 31 entries. All corrected, "
        "regression-tested, artifacts rebuilt. Decided after the corrections and before any 2027 "
        "information existed; NOT on the basis of any 2027 performance."
    ),
}

# The methodology spec the drift gate checks.
#
# SCOPE CORRECTION, 2026-08-21. The original file below carried three
# selection-owned fields -- candidate_selection.diversity_algorithm,
# candidate_selection.k_returned and product.strategies -- as hardcoded literals.
# capture_live_spec() transcribed the same literals, so the gate compared a
# constant to itself and reported "no drift" while all three had stopped
# describing the product. A gate that cannot fail is not a gate.
#
# Those fields moved to src/governance/product_spec.py (product.v3), where they
# are derived from the live implementation. THE METHODOLOGY DID NOT CHANGE: no
# model, simulation, objective, scoring rule or P(1st) definition moved. The hash
# differs only because the specification boundary was corrected, and
# test_methodology_values_are_unchanged_by_the_boundary_correction proves that
# field-by-field against the original.
FROZEN_SPEC_PATH = Path("configs/frozen/prospective_2027_v3_audited.json")
V2_SCOPED_SPEC_PATH = Path("configs/frozen/prospective_2027_v2_scoped.json")

# Kept byte-identical as the original prospective record. Never rewritten.
ORIGINAL_V2_SPEC_PATH = Path("configs/frozen/prospective_2027_v2.json")

SCOPE_CORRECTION = {
    "date": "2026-08-21",
    "supersedes_scope_of": str(ORIGINAL_V2_SPEC_PATH),
    # FROZEN VALUE -- do not "fix" this path. configs/frozen/product_v3.json,
    # docs/build.js (the UI it described) and tests/test_spec_boundary.py (its
    # drift test) were all deleted in 32f860e ("Remove the UI layer and its
    # contracts ahead of a rebuild") on 2026-08-21 or after, and this field is
    # part of the hashed, frozen spec (configs/frozen/prospective_2027_v2_scoped.json)
    # that CI checks for drift -- changing its value here would fail that check
    # for describing reality more accurately, which is exactly backwards. The
    # staleness is real and is documented instead in PROSPECTIVE_2027_v2.md's
    # 2026-09-12 correction (audit recommendation 10): there is currently no
    # live drift gate for the presentation-selection layer this path once named.
    "moved_to": "configs/frozen/product_v3.json",
    "fields_moved": [
        "candidate_selection.diversity_algorithm",
        "candidate_selection.k_returned",
        "product.strategies",
    ],
    "methodology_unchanged": True,
    "reason": (
        "The moved fields describe how candidates become the displayed brackets, not "
        "how candidates were produced. They were hardcoded literals on both sides of "
        "the comparison, so the drift gate could not detect a change in them. The "
        "2027.v2 methodology itself is untouched; the hash changed only because the "
        "specification boundary was corrected."
    ),
}

# v1 is retained verbatim as the original prospective specification. Its file and
# document are immutable; `test_v1_specification_is_immutable` pins the hash.
SUPERSEDED = {
    "version": "2027.v1",
    "spec_path": "configs/frozen/prospective_2027.json",
    "doc": "PROSPECTIVE_2027.md",
    "spec_hash": "557d5fd54a198933b0bf3e5466c9cc874956b07087b650a8822ea6d0fab1dcf6",
    "commit": "10c8a66223c9b77a22e92aa8ec059379cb20812c",
    "reason_superseded": (
        "TRAIN_YEARS extended through 2026. The 2026 season concluded in April 2026, "
        "so it is ordinary historical data for a 2027 prediction. Decided ex ante on "
        "2026-08-20, before any 2027 information existed, and NOT on the basis of any "
        "2026 performance comparison. 2026 remains permanently barred as an evaluation "
        "season; this concerns training data only."
    ),
}


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

    from src.optimization.poolaware_recipe import POOLAWARE_EXHAUSTIVE_RISKS, POOLAWARE_RISK_LEVELS
    from src.evaluation.referee_audit import QUALIFICATION, REFEREE_NOISE_STD_DOC
    from src.simulation.pool_competition import simulate_tournament_outcomes  # noqa: F401 - the one simulator

    return {
        "spec_version": SPEC_VERSION,
        "freeze_date": FREEZE_DATE,
        "supersedes": SUPERSEDES_V2,
        "lineage": {"v1": SUPERSEDED, "scope_correction_v2": SCOPE_CORRECTION},
        "audit": {
            "evidence": "artifacts/methodology_audit/ (Steps 1-12, R-4)",
            "commit": "75b97e4",
            "validation_result": (
                "Step 9: production strategy edge over the seed bracket, one fixed bracket per season, "
                "+7.1/+7.5/+6.2/+5.6 pp under seed/torvik/blend/pit and +4.0 pp [+2.5, +5.6] under market_v2 "
                "(the pre-registered independent referee); LORO all positive; registered verdict ROBUST. "
                "Historical validation only; not evidence of future or real-world performance; seed, torvik, "
                "blend and pit are one data family, so this is one independent confirmation, not five."
            ),
        },
        "topology": {
            "source": "src.simulation.bracket_topology.resolve_region_order",
            "rule": "played F4 games if present, else seeds.f4_pairing from the announced bracket; never a default",
            "prospective_requirement": "tournament_context_2027.json seeds block MUST carry f4_pairing = [[A,B],[C,D]]",
            "projection": "strict: a picks dict must name exactly one team of every game, else TopologyMismatch",
            "play_ins": "resolve_first_four removes First Four losers before the 64-team draw is built",
        },
        "referee": {
            "outcome_table": "seed-vs-seed win rates, Kaggle 2010..Y-1 (as_of=Y), cells < 8 games -> logistic(0.175*dseed)",
            "walk_forward": True,
            "simulator": "src.simulation.pool_competition.simulate_tournament_outcomes (raises on missing pair / non-64 tree)",
            "logit_noise_std": 0.16,
            "logit_noise_effect": REFEREE_NOISE_STD_DOC,
            "probability_cap": [0.01, 0.99],
            "qualified_set_step8": ["seed", "torvik", "blend", "pit", "market_v2"],
            "independent_referee_step8": "market_v2",
            "qualification_rule": QUALIFICATION,
        },
        "opponents": {
            "n_opponents": "pool size - 1 (real pool size for seasons with pool history, else 29)",
            "pick_model": "per-team round pick shares: the real pool's entries where available, else ESPN archive, else static seed pick rates",
            "per_game_rule": "P(pick t1) = share(t1)/(share(t1)+share(t2)); path-consistent walk; independent opponents; chalk noise 0",
            "known_limitation_R3": "the per-game ratio rule does not reproduce deep-round shares (up to 12 pp at E8 in 2026)",
        },
        "objective": {
            "p1": "expected first-place share: 1 if strictly top, 1/(1+k) if tied with k opponents at the top score, else 0; "
                  "mean over shared CRN trials; equals the winner-take-all prize",
            "ev": "sum_R pts_R * P(picked team wins R), Torvik marginals",
            "ties": "split, everywhere (selection, reporting, payout)",
            "scoring_default": "team_identity; shape-encoded only via --shape-encoded",
        },
        "production_candidates": {
            "strategy": "meta_region_poolaware",
            "families": "region_top_n x prob bases x risk grid; exhaustive_champion x prob bases x exhaustive risks; byte-dedup, first label kept",
            "prob_bases": ["tv", "mass_avg", "mass_best", "blend", "tv_mass80"],
            "risk_levels": list(POOLAWARE_RISK_LEVELS),
            "exhaustive_risks": list(POOLAWARE_EXHAUSTIVE_RISKS),
            "selection": "first argmax of P(1st) on 500 CRN selection trials (seed 77777+year)",
            "known_limitation_R2": "a wider legal candidate bank scored +1.2 pp out-of-sample (4/15 seasons > 2 SE)",
            "forced_champion_family": "removed 2026-09-15 (was inert: region_top_n ignores forced_champion)",
        },
        "future_research_not_in_production": {
            "R-1": "noseed advancement marginals are a compounding heuristic, not propagated from pairwise",
            "R-2": "candidate-space expansion",
            "R-3": "opponent sampler joint structure / real-entry resampling",
            "R-4": "CLOSED negative: recency/upset training weights fail the frozen gates on both populations",
            "R-5": "production calibration slope 1.33 on 2023-25 (observation only)",
            "rule": "any change to model selection, weighting, candidate space, referee or objective after this freeze "
                    "requires a separately dated pre-registration; none may be applied to 2027 production",
        },
        "scope_correction": SCOPE_CORRECTION,
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
            "p1_pool_semantics": "entries in the pool (us + DEFAULT_POOL_SIZE-1 opponents); was 31 entries before 2026-09-15",
            "p1_opponent_model": "ESPN public pick distribution + seed pairwise referee",
            "common_random_numbers": True,
            "objectives": ["ev", "p1"],
            "ev_definition": "sum_R pts_R * sum_{t in picked_R} P(t wins R), marginals from the unconditional bank",
            "p1_definition": "expected first-place share over shared trials (ties split); was 'score >= max opponent' before 2026-09-15",
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
        # Selection-owned fields (diversity_algorithm, k_returned, strategies)
        # are NOT here; see src/governance/product_spec.py. What remains are
        # constraints the methodology genuinely owns.
        "product": {
            "excluded_from_v1": ["balanced blend", "contrarian ownership penalty", "configurable pool size"],
            "p1_disclosure_required": (
                "P(1st) is the expected share of first place (a tie for the top score is split among the tied "
                "entries) in a 30-entry pool (you plus 29 opponents) with ESPN public pick behaviour. "
                "It is NOT a universal probability of winning any pool."
            ),
        },
        "holdout": {
            "contaminated_seasons": [2026],
            "contaminated_for_evaluation_only": True,
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
