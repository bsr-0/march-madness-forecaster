"""Integrity checks for the dated 2027 ESPN-points selector re-freeze."""

from __future__ import annotations

import json
from pathlib import Path

from src.governance.frozen_spec import canonical_hash

REPO = Path(__file__).resolve().parent.parent
SPEC_PATH = REPO / "configs" / "frozen" / "prospective_2027_v4_points.json"


def _spec() -> dict:
    return json.loads(SPEC_PATH.read_text())


def test_2027_points_freeze_hash_is_self_consistent():
    spec = _spec()
    body = {key: value for key, value in spec.items() if key != "spec_hash"}
    assert spec["spec_hash"] == canonical_hash(body)


def test_2027_points_freeze_pins_nested_selector_and_evaluation_window():
    spec = _spec()
    assert spec["spec_version"] == "2027.v4"
    assert spec["supersedes"]["version"] == "2027.v3"
    assert spec["supersedes"]["spec_hash"] == "183f90796d527f0f254cc087c349bd72852c43865676d5d175d79a122e89974a"
    assert set(spec["selector"]["candidate_rules"]) == {"blend_region_35", "ev_optimal", "highest_p1"}
    assert spec["selector"]["inter_rule_tie_break"] == "rule ID ascending alphabetically"
    assert spec["selector"]["no_prior_eligible_seasons"] == "seed_only"
    assert spec["selector"]["production_training_seasons"] == "2011-2025 inclusive, excluding 2020"
    assert spec["release"]["historical_gate"]["target_seasons"] == [
        *range(2011, 2020),
        *range(2021, 2026),
    ]


def test_2027_points_freeze_pins_cutoffs_generation_and_missing_result_policy():
    spec = _spec()
    assert spec["source_cutoffs"]["field"]["selection_sunday"] == "2027-03-14"
    assert spec["source_cutoffs"]["torvik"]["cutoff_date_exclusive_before"] == "2027-03-16"
    assert spec["source_cutoffs"]["public_picks"]["capture_at_or_before"] == "2027-03-18T12:00:00-04:00"
    assert spec["candidate_generation"]["defaults"] == {
        "n_sims_total": 150000,
        "p1_trials": 2000,
        "pool_size": 30,
        "rng_seed": 20260820,
        "scoring_id": "espn_standard",
    }
    assert spec["scoring"]["required_game_counts"] == {
        "E8": 4,
        "F4": 2,
        "FF": 4,
        "NCG": 1,
        "R32": 16,
        "R64": 32,
        "S16": 8,
    }
    assert spec["scoring"]["missing_or_incomplete_results"].startswith("no score")
