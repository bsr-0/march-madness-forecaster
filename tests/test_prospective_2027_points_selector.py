from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.evaluation import prospective_2027_points as selector


def _score_table() -> dict[int, dict[str, int]]:
    spec = selector.load_spec()
    return {
        year: {
            "seed_only": 100,
            "blend_region_35": 110,
            "ev_optimal": 120,
            "highest_p1": 130,
        }
        for year in spec["release"]["historical_gate"]["target_seasons"]
    }


def _seed_bracket() -> list[list[int]]:
    return [
        list(range(32)),
        list(range(16)),
        list(range(8)),
        list(range(4)),
        [0, 1],
        [0],
    ]


def test_nested_selector_uses_prior_seasons_only_and_seed_only_in_2011():
    scores = _score_table()
    scores[2012] = {
        "seed_only": 100,
        "blend_region_35": 140,
        "ev_optimal": 130,
        "highest_p1": 120,
    }
    result = selector.evaluate_nested_scores(scores)
    assert result["nested_folds"]["2011"]["selected_rule"] == "seed_only"
    assert result["nested_folds"]["2011"]["paired_gain"] == 0
    assert result["nested_folds"]["2012"]["selected_rule"] == "highest_p1"
    assert result["nested_folds"]["2012"]["training_seasons"] == [2011]
    assert result["nested_folds"]["2012"]["selected_points"] == scores[2012]["highest_p1"]
    assert 2012 not in result["nested_folds"]["2012"]["training_seasons"]


def test_positive_walk_forward_gains_pass_both_frozen_intervals():
    result = selector.evaluate_nested_scores(_score_table())
    assert result["promotion_gate"]["status"] == "PASS"
    assert result["release"]["selected_rule"] == "highest_p1"
    assert result["promotion_gate"]["bootstrap_95_lower"] > 0
    assert result["promotion_gate"]["paired_t_95_lower"] > 0


def test_rule_ties_are_broken_by_ascending_rule_id():
    gains = {
        2011: {
            "blend_region_35": 4.0,
            "ev_optimal": 4.0,
            "highest_p1": 4.0,
        }
    }
    targets = selector.load_spec()["release"]["historical_gate"]["target_seasons"]
    assert selector._pick_rule(2012, gains, targets) == "blend_region_35"


def test_selector_requires_every_frozen_year_and_every_rule():
    scores = _score_table()
    del scores[2012]
    with pytest.raises(ValueError, match="must match the frozen target set"):
        selector.evaluate_nested_scores(scores)

    scores = _score_table()
    del scores[2012]["highest_p1"]
    with pytest.raises(ValueError, match="expected score columns"):
        selector.evaluate_nested_scores(scores)


def test_failed_promotion_gate_falls_back_to_seed_only():
    scores = _score_table()
    for row in scores.values():
        row.update(blend_region_35=90, ev_optimal=80, highest_p1=70)
    result = selector.evaluate_nested_scores(scores)
    assert result["promotion_gate"]["status"] == "FAIL"
    assert result["release"] == {
        "selected_rule": "seed_only",
        "fallback": True,
        "reason": "promotion gate is FAIL",
    }


def test_disagreement_between_primary_and_sensitivity_intervals_is_indeterminate(monkeypatch):
    monkeypatch.setattr(
        selector,
        "_confidence_interval",
        lambda *_args, **_kwargs: {
            "mean_gain": 1.0,
            "bootstrap_95_lower": 0.1,
            "bootstrap_95_upper": 2.0,
            "paired_t_95_lower": -0.1,
            "paired_t_95_upper": 2.1,
            "resamples": 5000,
            "seed": 42,
        },
    )
    result = selector.evaluate_nested_scores(_score_table())
    assert result["promotion_gate"]["status"] == "INDETERMINATE"
    assert result["release"]["selected_rule"] == "seed_only"


def test_bootstrap_interval_is_reproducible():
    import numpy as np

    gains = np.asarray([10, 20, -10, 0, 40, 10], dtype=float)
    first = selector._confidence_interval(gains, resamples=5000, seed=42)
    second = selector._confidence_interval(gains, resamples=5000, seed=42)
    assert first == second


def test_candidate_rules_resolve_frozen_brackets_and_highest_p1_tie_order():
    second_bracket = _seed_bracket()
    second_bracket[0][0] = 31
    artifact = {
        "teams": [{"id": f"t{i}"} for i in range(64)],
        "named_strategies": {
            "blend_region_35": {"w": _seed_bracket()},
            "ev_optimal": {"w": _seed_bracket()},
        },
        "candidates": [
            {"w": _seed_bracket(), "p1": 0.5},
            {"w": second_bracket, "p1": 0.5},
        ],
    }
    assert selector.candidate_bracket(artifact, "blend_region_35") == _seed_bracket()
    assert selector.candidate_bracket(artifact, "ev_optimal") == _seed_bracket()
    assert selector.candidate_bracket(artifact, "highest_p1") == _seed_bracket()
    with pytest.raises(ValueError, match="no required rule"):
        selector.candidate_bracket({"teams": [], "named_strategies": {}}, "ev_optimal")


def test_seed_only_builder_uses_walk_forward_probabilities_and_no_public_picks(monkeypatch):
    from src.optimization import bracket_construction
    from src.prediction import seed_probabilities
    from src.simulation import bracket_topology

    teams = [
        {"id": f"{region}_{seed}", "seed": seed, "region": region}
        for region in ("East", "West", "South", "Midwest")
        for seed in range(1, 17)
    ]
    seeds = {team["id"]: team["seed"] for team in teams}
    regions = {team["id"]: team["region"] for team in teams}
    region_order = ("East", "West", "South", "Midwest")
    first_round = bracket_topology.build_bracket_order(seeds, regions, region_order=region_order)
    index = {team["id"]: i for i, team in enumerate(teams)}
    for entrant in range(12):
        teams.append({
            "id": f"play_in_{entrant}",
            "seed": 16,
            "region": region_order[entrant % len(region_order)],
        })
    as_of_values = []
    construction_args = {}

    def fake_probabilities(_seeds, as_of=None):
        as_of_values.append(as_of)
        return {}

    def fake_construct(**kwargs):
        construction_args.update(kwargs)
        winners_by_round = []
        current = list(first_round)
        picks = {}
        for round_name in ("R64", "R32", "S16", "E8", "F4", "CHAMP"):
            winners = []
            for game_index in range(0, len(current), 2):
                team1, team2 = current[game_index : game_index + 2]
                winner = team1
                winners.append(winner)
                picks[f"{round_name}_{game_index // 2}"] = winner
            winners_by_round.append(winners)
            current = winners
        return picks, None, None, None, None

    monkeypatch.setattr(seed_probabilities, "build_seed_round_probabilities", fake_probabilities)
    monkeypatch.setattr(bracket_construction, "construct_bracket", fake_construct)
    artifact = {
        "teams": teams,
        "first_round": [index[team_id] for team_id in first_round],
    }
    result = selector.build_seed_only_bracket(artifact, 2025)
    assert as_of_values == [2025]
    assert construction_args["mode"] == "forward_greedy"
    assert construction_args["risk_level"] == 0.0
    assert construction_args["public_picks"] == {}
    assert tuple(map(len, result)) == (32, 16, 8, 4, 2, 1)
    assert all(index < 64 for round_picks in result for index in round_picks)


def _valid_outcome(year: int) -> tuple[list[str], list[dict]]:
    first_round = [f"team_{i}" for i in range(60)] + ["ff_0", "ff_2", "ff_4", "ff_6"]
    games = []
    for i in range(0, 8, 2):
        games.append({
            "year": year,
            "round_name": "FF",
            "team1_id": f"ff_{i}",
            "team2_id": f"ff_{i + 1}",
            "team1_won": True,
            "team1_score": 70,
            "team2_score": 60,
        })
    current = first_round
    for round_name in ("R64", "R32", "S16", "E8", "F4", "NCG"):
        next_round = []
        for game_index in range(0, len(current), 2):
            team1, team2 = current[game_index : game_index + 2]
            games.append({
                "year": year,
                "round_name": round_name,
                "team1_id": team1,
                "team2_id": team2,
                "team1_won": True,
                "team1_score": 70,
                "team2_score": 60,
            })
            next_round.append(team1)
        current = next_round
    return first_round, games


def test_seed_comparison_uses_canonical_team_identity_espn_scorer():
    first_round, games = _valid_outcome(2011)
    artifact = {
        "teams": [{"id": team_id} for team_id in first_round],
        "first_round": list(range(64)),
    }
    perfect_bracket = [
        first_round[::2],
        first_round[::4],
        first_round[::8],
        first_round[::16],
        [first_round[0], first_round[32]],
        [first_round[0]],
    ]
    team_index = {team_id: index for index, team_id in enumerate(first_round)}
    perfect_bracket = [[team_index[team_id] for team_id in rnd] for rnd in perfect_bracket]
    assert selector._points_for_w(artifact, perfect_bracket, games, 2011) == 1920


def _source_artifacts(tmp_path: Path) -> tuple[Path, dict]:
    from src.data.season_calendar import TOURNAMENT_START_DATES

    spec = selector.load_spec()
    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
    for year in spec["release"]["historical_gate"]["target_seasons"]:
        public_path = tmp_path / f"public_{year}.json"
        captured = datetime(year, 1, 1, tzinfo=timezone.utc).isoformat()
        public_path.write_text(json.dumps({"captured_at": captured}))
        torvik_path = tmp_path / f"torvik_{year}.json"
        start = TOURNAMENT_START_DATES[year]
        cutoff = (start - timedelta(days=1)).isoformat()
        torvik_path.write_text(json.dumps({
            "data_type": "pre_tournament",
            "cutoff_date": cutoff,
            "tournament_start": start.isoformat(),
        }))
        field_seed_path = tmp_path / f"field_seeds_{year}.json"
        field_seed_path.write_text("{}")
        field_results_path = tmp_path / f"field_results_{year}.json"
        field_results_path.write_text("{}")
        referee_seed_path = tmp_path / f"referee_seeds_{year}.csv"
        referee_seed_path.write_text("seed\n")
        referee_results_path = tmp_path / f"referee_results_{year}.csv"
        referee_results_path.write_text("winner\n")

        def source_record(path: Path) -> dict:
            return {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

        artifact = {
            "meta": {"pool_settings": {"pool_size": 30, "scoring_id": "espn_standard"}},
            "provenance": {
                "public_picks": {
                    **source_record(public_path),
                    "capture_time_verified": True,
                    "captured_at": captured,
                    "declared_cutoff": None,
                },
                "torvik": {
                    **source_record(torvik_path),
                    "cutoff_date": cutoff,
                    "tournament_start": start.isoformat(),
                },
                "field": {
                    "input_files": [source_record(field_seed_path), source_record(field_results_path)],
                },
                "seed_head_to_head": {
                    "point_in_time": True,
                    "as_of": year,
                    "input_files": [source_record(referee_seed_path), source_record(referee_results_path)],
                },
                "rating_sources": {
                    "used": ["torvik"],
                    "inputs": {
                        "torvik": source_record(torvik_path),
                        "massey_avg": {
                            "file": f"external_ratings_{year}.json",
                            "sha256": None,
                            "status": "unavailable",
                        },
                        "elo": {
                            "file": f"historical_games_{year}.json",
                            "sha256": None,
                            "status": "unavailable",
                        },
                    },
                },
            },
        }
        path = candidates_dir / f"candidates_{year}.json"
        raw = json.dumps(artifact).encode()
        path.write_bytes(raw)
        path.with_suffix(".sha256").write_text(f"{hashlib.sha256(raw).hexdigest()}  {path.name}\n")
    return candidates_dir, spec


def test_source_gate_verifies_each_file_hash_timestamp_and_torvik_cutoff(tmp_path):
    candidates_dir, spec = _source_artifacts(tmp_path)
    result = selector.inspect_historical_sources(spec, candidates_dir)
    assert result["status"] == "PASS"
    assert len(result["artifacts"]) == 14


def test_source_gate_rejects_required_source_without_path_or_hash(tmp_path):
    candidates_dir, spec = _source_artifacts(tmp_path)
    path = candidates_dir / "candidates_2011.json"
    artifact = json.loads(path.read_text())
    artifact["provenance"]["field"]["input_files"][0].pop("file")
    artifact["provenance"]["seed_head_to_head"]["input_files"][0].pop("sha256")
    raw = json.dumps(artifact).encode()
    path.write_bytes(raw)
    path.with_suffix(".sha256").write_text(f"{hashlib.sha256(raw).hexdigest()}  {path.name}\n")

    result = selector.inspect_historical_sources(spec, candidates_dir)
    assert result["status"] == "INDETERMINATE"
    reasons = [issue["reason"] for issue in result["issues"] if issue["year"] == "2011"]
    assert any("field/results input 1 source path is absent" in reason for reason in reasons)
    assert any("seed-referee input 1 source SHA-256 is absent" in reason for reason in reasons)


@pytest.mark.parametrize(
    ("section", "record_key", "reason"),
    [
        ("public_picks", None, "public-picks source path is absent"),
        ("torvik", None, "Torvik source path is absent"),
        ("rating_sources", "torvik", "torvik rating source SHA-256 is absent or invalid"),
    ],
)
def test_source_gate_requires_path_and_hash_for_public_torvik_and_rating_inputs(
    tmp_path, section, record_key, reason
):
    candidates_dir, spec = _source_artifacts(tmp_path)
    path = candidates_dir / "candidates_2011.json"
    artifact = json.loads(path.read_text())
    record = artifact["provenance"][section]
    if record_key is not None:
        record = record["inputs"][record_key]
        record.pop("sha256")
    else:
        record.pop("file")
    raw = json.dumps(artifact).encode()
    path.write_bytes(raw)
    path.with_suffix(".sha256").write_text(f"{hashlib.sha256(raw).hexdigest()}  {path.name}\n")

    result = selector.inspect_historical_sources(spec, candidates_dir)
    assert result["status"] == "INDETERMINATE"
    assert any(issue["year"] == "2011" and reason in issue["reason"] for issue in result["issues"])


def test_source_gate_requires_optional_rating_availability_to_be_explicit(tmp_path):
    candidates_dir, spec = _source_artifacts(tmp_path)
    path = candidates_dir / "candidates_2011.json"
    artifact = json.loads(path.read_text())
    del artifact["provenance"]["rating_sources"]["inputs"]["massey_avg"]
    raw = json.dumps(artifact).encode()
    path.write_bytes(raw)
    path.with_suffix(".sha256").write_text(f"{hashlib.sha256(raw).hexdigest()}  {path.name}\n")

    result = selector.inspect_historical_sources(spec, candidates_dir)
    assert result["status"] == "INDETERMINATE"
    assert any(
        issue["year"] == "2011" and "massey_avg rating-source availability" in issue["reason"]
        for issue in result["issues"]
    )


def test_source_gate_does_not_infer_missing_public_pick_timestamps(tmp_path):
    candidates_dir, spec = _source_artifacts(tmp_path)
    path = candidates_dir / "candidates_2011.json"
    artifact = json.loads(path.read_text())
    artifact["provenance"]["public_picks"]["capture_time_verified"] = False
    raw = json.dumps(artifact).encode()
    path.write_bytes(raw)
    path.with_suffix(".sha256").write_text(f"{hashlib.sha256(raw).hexdigest()}  {path.name}\n")
    result = selector.inspect_historical_sources(spec, candidates_dir)
    assert result["status"] == "INDETERMINATE"
    assert any("capture time is not verified" in item["reason"] for item in result["issues"])


def test_repository_report_refuses_to_score_without_all_source_evidence(tmp_path):
    result = selector.build_repository_result(candidates_dir=tmp_path)
    assert result["status"] == "INDETERMINATE"
    assert result["release"]["selected_rule"] == "seed_only"
    assert "paired_points" not in result["inputs"]
    assert result["promotion_gate"]["status"] == "INDETERMINATE"


def test_source_gate_fails_candidate_artifact_hash_mismatch(tmp_path):
    candidates_dir, spec = _source_artifacts(tmp_path)
    path = candidates_dir / "candidates_2011.json"
    path.with_suffix(".sha256").write_text("0" * 64 + f"  {path.name}\n")
    result = selector.inspect_historical_sources(spec, candidates_dir)
    assert result["status"] == "FAIL"
    assert any("sidecar" in item["reason"] for item in result["issues"])


def test_source_gate_fails_public_picks_captured_after_r64_lock(tmp_path):
    from src.data.season_calendar import get_round_of_64_tip

    candidates_dir, spec = _source_artifacts(tmp_path)
    path = candidates_dir / "candidates_2011.json"
    artifact = json.loads(path.read_text())
    public_path = Path(artifact["provenance"]["public_picks"]["file"])
    late_capture = (get_round_of_64_tip(2011) + timedelta(hours=1)).isoformat()
    public_path.write_text(json.dumps({"captured_at": late_capture}))
    artifact["provenance"]["public_picks"].update({
        "sha256": hashlib.sha256(public_path.read_bytes()).hexdigest(),
        "captured_at": late_capture,
    })
    raw = json.dumps(artifact).encode()
    path.write_bytes(raw)
    path.with_suffix(".sha256").write_text(f"{hashlib.sha256(raw).hexdigest()}  {path.name}\n")
    result = selector.inspect_historical_sources(spec, candidates_dir)
    assert result["status"] == "FAIL"
    assert any("after the R64 lock" in item["reason"] for item in result["issues"])


def test_source_gate_rejects_torvik_snapshot_at_or_after_tournament_start(tmp_path):
    from src.data.season_calendar import TOURNAMENT_START_DATES

    candidates_dir, spec = _source_artifacts(tmp_path)
    path = candidates_dir / "candidates_2011.json"
    artifact = json.loads(path.read_text())
    torvik_path = Path(artifact["provenance"]["torvik"]["file"])
    start = TOURNAMENT_START_DATES[2011]
    torvik_payload = {
        "data_type": "pre_tournament",
        "cutoff_date": start.isoformat(),
        "tournament_start": start.isoformat(),
    }
    torvik_path.write_text(json.dumps(torvik_payload))
    artifact["provenance"]["torvik"].update({
        "sha256": hashlib.sha256(torvik_path.read_bytes()).hexdigest(),
        "cutoff_date": start.isoformat(),
    })
    raw = json.dumps(artifact).encode()
    path.write_bytes(raw)
    path.with_suffix(".sha256").write_text(f"{hashlib.sha256(raw).hexdigest()}  {path.name}\n")
    result = selector.inspect_historical_sources(spec, candidates_dir)
    assert result["status"] == "FAIL"
    assert any("Torvik cutoff is not before" in item["reason"] for item in result["issues"])


def _prospective_artifact(tmp_path: Path) -> tuple[dict, Path, dict]:
    from src.data.season_calendar import TOURNAMENT_START_DATES, get_round_of_64_tip
    from src.simulation import bracket_topology

    spec = selector.load_spec()
    year = spec["release"]["prospective_season"]
    public_path = tmp_path / "public_2027.json"
    public_cutoff = datetime.fromisoformat(spec["source_cutoffs"]["public_picks"]["capture_at_or_before"])
    captured = (public_cutoff - timedelta(hours=1)).isoformat()
    public_path.write_text(json.dumps({"captured_at": captured}))
    start = TOURNAMENT_START_DATES[year]
    torvik_path = tmp_path / "torvik_2027.json"
    torvik_payload = {
        "data_type": "pre_tournament",
        "cutoff_date": (start - timedelta(days=1)).isoformat(),
        "tournament_start": start.isoformat(),
    }
    torvik_path.write_text(json.dumps(torvik_payload))
    source_paths = [
        tmp_path / "field_seeds.json",
        tmp_path / "field_results.json",
        tmp_path / "referee_seeds.csv",
        tmp_path / "referee_results.csv",
        tmp_path / "massey.json",
        tmp_path / "elo.json",
    ]
    for path in source_paths:
        path.write_text("{}")
    teams = [
        {"id": f"{region}_{seed}", "region": region, "seed": seed}
        for region in ("East", "West", "South", "Midwest")
        for seed in range(1, 17)
    ]
    team_seeds = {team["id"]: team["seed"] for team in teams}
    team_regions = {team["id"]: team["region"] for team in teams}
    pairing = ["East", "West", "South", "Midwest"]
    ordered_ids = bracket_topology.build_bracket_order(
        team_seeds, team_regions, region_order=pairing
    )
    team_index = {team["id"]: i for i, team in enumerate(teams)}
    teams.extend(
        {"id": f"play_in_{entrant}", "region": pairing[entrant % len(pairing)], "seed": 16}
        for entrant in range(12)
    )

    def source_record(path: Path) -> dict:
        return {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    defaults = spec["candidate_generation"]["defaults"]
    artifact = {
        "teams": teams,
        "first_round": [team_index[team_id] for team_id in ordered_ids],
        "meta": {
            "n_sims": defaults["n_sims_total"],
            "candidate_target": spec["candidate_generation"]["candidate_count_target"],
            "p1_trials": defaults["p1_trials"],
            "rng_seed": defaults["rng_seed"],
            "n_candidates": spec["candidate_generation"]["candidate_count_target"],
            "pool_settings": {
                "pool_size": defaults["pool_size"],
                "scoring_id": defaults["scoring_id"],
            },
            "p1_pool_size": defaults["pool_size"],
            "generated_at": "2027-03-14T16:00:00+00:00",
        },
        "provenance": {
            "public_picks": {
                **source_record(public_path),
                "capture_time_verified": True,
                "captured_at": captured,
                "declared_cutoff": spec["source_cutoffs"]["public_picks"]["capture_at_or_before"],
            },
            "torvik": {
                **source_record(torvik_path),
                "data_type": "pre_tournament",
                "cutoff_date": torvik_payload["cutoff_date"],
                "tournament_start": torvik_payload["tournament_start"],
            },
            "field": {
                "main_draw": 64,
                "play_in_games": 12,
                "slots_resolved": 12,
                "input_files": [source_record(path) for path in source_paths[:2]],
            },
            "f4_pairing": pairing,
            "seed_head_to_head": {
                "point_in_time": True,
                "as_of": year,
                "input_files": [source_record(path) for path in source_paths[2:4]],
            },
            "rating_sources": {
                "used": ["torvik"],
                "missing_torvik_team_ratings": [],
                "inputs": {
                    "torvik": source_record(torvik_path),
                    "massey_avg": source_record(source_paths[4]),
                    "elo": source_record(source_paths[5]),
                },
            },
        },
    }
    artifact_path = tmp_path / "candidates_2027.json"
    raw = json.dumps(artifact).encode()
    artifact_path.write_bytes(raw)
    artifact_path.with_suffix(".sha256").write_text(
        f"{hashlib.sha256(raw).hexdigest()}  {artifact_path.name}\n"
    )
    assert datetime.fromisoformat(captured) <= get_round_of_64_tip(year)
    return artifact, artifact_path, spec


def test_prospective_artifact_must_match_frozen_settings_and_source_cutoffs(tmp_path):
    artifact, path, spec = _prospective_artifact(tmp_path)
    assert selector.validate_prospective_artifact(artifact, path, 2027, spec)["status"] == "PASS"
    artifact["meta"]["rng_seed"] += 1
    result = selector.validate_prospective_artifact(artifact, path, 2027, spec)
    assert result["status"] == "FAIL"
    assert any("rng_seed" in issue for issue in result["failures"])


def test_prospective_artifact_requires_hashes_for_each_consumed_source(tmp_path):
    artifact, path, spec = _prospective_artifact(tmp_path)
    artifact["provenance"]["field"]["input_files"][0].pop("sha256")
    raw = json.dumps(artifact).encode()
    path.write_bytes(raw)
    path.with_suffix(".sha256").write_text(f"{hashlib.sha256(raw).hexdigest()}  {path.name}\n")
    result = selector.validate_prospective_artifact(artifact, path, 2027, spec)
    assert result["status"] == "INDETERMINATE"
    assert any("field/results input 1 source SHA-256 is absent or invalid" in item for item in result["unknowns"])


def test_payload_recommendation_is_seed_only_without_a_verified_result(monkeypatch, tmp_path):
    from scripts import build_ui_payload

    monkeypatch.setattr(selector, "build_seed_only_bracket", lambda _art, _year: _seed_bracket())
    artifact = {
        "teams": [{"id": f"t{i}"} for i in range(64)],
        "named_strategies": {},
        "candidates": [],
    }
    recommendation = build_ui_payload._recommended_strategy(
        2027, artifact, None, "artifact-hash", tmp_path / "artifact.json"
    )
    assert recommendation["id"] == "recommended"
    assert recommendation["picks"] == _seed_bracket()
    assert recommendation["p1"] is None and recommendation["ev"] is None
    assert recommendation["selector"]["fallback"] is True
    assert recommendation["selector"]["source_gate"] == "INDETERMINATE"


def test_payload_recommendation_uses_v4_rule_when_both_gates_pass(monkeypatch, tmp_path):
    from scripts import build_ui_payload

    artifact = {
        "teams": [{"id": f"t{i}"} for i in range(64)],
        "named_strategies": {"ev_optimal": {"w": _seed_bracket(), "ev": 888, "p1": 0.12}},
        "candidates": [],
    }
    report = {
        "source_gate": {"status": "PASS"},
        "promotion_gate": {"status": "PASS"},
        "nested_folds": {"2025": {"selected_rule": "ev_optimal"}},
    }
    recommendation = build_ui_payload._recommended_strategy(
        2025, artifact, report, "artifact-hash", tmp_path / "artifact.json"
    )
    assert recommendation["picks"] == _seed_bracket()
    assert recommendation["ev"] == 888
    assert recommendation["p1"] == 0.12
    assert recommendation["selector"]["rule_id"] == "ev_optimal"
    assert recommendation["selector"]["fallback"] is False
