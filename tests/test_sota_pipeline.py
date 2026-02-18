"""Integration tests for the SOTA pipeline with real historical data."""

import json
import os

import pytest

from src.models.team import Team
from src.pipeline.sota import DataRequirementError, SOTAPipeline, SOTAPipelineConfig

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
_GENERATED = os.path.join(_DATA_DIR, "generated_strict_2025")

REAL_DATA_PATHS = {
    "teams": os.path.join(_GENERATED, "teams_2025.json"),
    "torvik": os.path.join(_GENERATED, "torvik_2025.json"),
    "rosters": os.path.join(_GENERATED, "rosters_2025.json"),
    "public_picks": os.path.join(_GENERATED, "public_picks_2025.json"),
    "historical_games": os.path.join(_DATA_DIR, "historical", "historical_games_2025.json"),
}

_REAL_DATA_AVAILABLE = all(os.path.exists(p) for p in REAL_DATA_PATHS.values())


@pytest.mark.skipif(not _REAL_DATA_AVAILABLE, reason="Real data files not present")
def test_sota_pipeline_produces_rubric_artifacts():
    config = SOTAPipelineConfig(
        year=2025,
        num_simulations=120,
        pool_size=64,
        calibration_method="isotonic",
        enforce_feed_freshness=False,
        teams_json=REAL_DATA_PATHS["teams"],
        torvik_json=REAL_DATA_PATHS["torvik"],
        roster_json=REAL_DATA_PATHS["rosters"],
        public_picks_json=REAL_DATA_PATHS["public_picks"],
        historical_games_json=REAL_DATA_PATHS["historical_games"],
    )
    pipeline = SOTAPipeline(config)

    report = pipeline.run()

    assert "rubric_evaluation" in report
    assert "artifacts" in report

    adjacency = report["artifacts"]["adjacency_matrix"]
    assert len(adjacency) == 64
    assert len(adjacency[0]) == 64

    sim = report["artifacts"]["simulation"]
    assert sim["num_simulations"] == 120
    assert sim["injury_noise_samples_per_matchup"] == 10000

    ev_bracket = report["artifacts"]["ev_max_bracket"]
    assert "champion" in ev_bracket
    assert len(ev_bracket["final_four"]) <= 4
    assert len(ev_bracket["picks"]) >= 63

    baseline = report["artifacts"]["baseline_training"]
    assert baseline["model"] in {"lightgbm", "logistic_regression", "none"}
    assert "model_uncertainty" in report["artifacts"]
    assert sorted(report["artifacts"]["public_pick_sources"]) == ["cbs", "espn", "yahoo"]
    for pick in report["artifacts"]["top_leverage_picks"][:10]:
        assert 0.0 <= pick["public_pick_percentage"] <= 1.0


@pytest.mark.skipif(not _REAL_DATA_AVAILABLE, reason="Real data files not present")
def test_sota_pipeline_output_file(tmp_path):
    output_path = tmp_path / "sota_report.json"

    config = SOTAPipelineConfig(
        year=2025,
        num_simulations=80,
        pool_size=20,
        enforce_feed_freshness=False,
        teams_json=REAL_DATA_PATHS["teams"],
        torvik_json=REAL_DATA_PATHS["torvik"],
        roster_json=REAL_DATA_PATHS["rosters"],
        public_picks_json=REAL_DATA_PATHS["public_picks"],
        historical_games_json=REAL_DATA_PATHS["historical_games"],
    )
    pipeline = SOTAPipeline(config)
    report = pipeline.run()

    with open(output_path, "w") as f:
        json.dump(report, f)

    with open(output_path, "r") as f:
        restored = json.load(f)

    assert restored["artifacts"]["simulation"]["num_simulations"] == 80


def test_public_pick_loader_supports_explicit_multi_source_payload(tmp_path):
    payload = {
        "timestamp": "2026-03-17T12:00:00Z",
        "espn": {
            "teams": {
                "duke": {
                    "team_name": "Duke",
                    "seed": 1,
                    "region": "East",
                    "round_of_64_pct": 98.0,
                    "round_of_32_pct": 90.0,
                    "sweet_16_pct": 70.0,
                    "elite_8_pct": 50.0,
                    "final_four_pct": 30.0,
                    "champion_pct": 20.0,
                }
            }
        },
        "yahoo": {
            "teams": {
                "duke": {
                    "team_name": "Duke",
                    "seed": 1,
                    "region": "East",
                    "round_of_64_pct": 97.0,
                    "round_of_32_pct": 88.0,
                    "sweet_16_pct": 68.0,
                    "elite_8_pct": 48.0,
                    "final_four_pct": 28.0,
                    "champion_pct": 10.0,
                }
            }
        },
        "cbs": {
            "teams": {
                "duke": {
                    "team_name": "Duke",
                    "seed": 1,
                    "region": "East",
                    "round_of_64_pct": 96.0,
                    "round_of_32_pct": 87.0,
                    "sweet_16_pct": 66.0,
                    "elite_8_pct": 46.0,
                    "final_four_pct": 26.0,
                    "champion_pct": 5.0,
                }
            }
        },
    }
    picks_path = tmp_path / "picks.json"
    with open(picks_path, "w") as f:
        json.dump(payload, f)

    pipeline = SOTAPipeline(SOTAPipelineConfig(public_picks_json=str(picks_path)))
    pipeline.team_struct["duke"] = Team(name="Duke", seed=1, region="East")

    public = pipeline._load_public_picks({"duke": {"CHAMP": 0.1}})
    assert sorted(pipeline.public_pick_sources) == ["cbs", "espn", "yahoo"]
    assert abs(public["duke"]["CHAMP"] - 0.14) < 1e-9


def test_sota_pipeline_requires_real_data():
    config = SOTAPipelineConfig(num_simulations=20, pool_size=10)
    pipeline = SOTAPipeline(config)

    try:
        pipeline.run()
        assert False, "Expected DataRequirementError"
    except DataRequirementError:
        assert True


def test_sota_pipeline_rejects_stale_public_feed(tmp_path):
    payload = {
        "timestamp": "2020-01-01T00:00:00Z",
        "teams": {
            "duke": {
                "team_name": "Duke",
                "seed": 1,
                "region": "East",
                "round_of_64_pct": 98.0,
                "round_of_32_pct": 90.0,
                "sweet_16_pct": 70.0,
                "elite_8_pct": 50.0,
                "final_four_pct": 30.0,
                "champion_pct": 20.0,
            }
        },
        "sources": ["espn", "yahoo"],
    }
    picks_path = tmp_path / "stale_picks.json"
    with open(picks_path, "w") as f:
        json.dump(payload, f)

    pipeline = SOTAPipeline(
        SOTAPipelineConfig(
            public_picks_json=str(picks_path),
            max_feed_age_hours=1,
            min_public_sources=2,
        )
    )
    pipeline.team_struct["duke"] = Team(name="Duke", seed=1, region="East")

    try:
        pipeline._load_public_picks({"duke": {"CHAMP": 0.1}})
        assert False, "Expected stale feed rejection"
    except DataRequirementError:
        assert True


def test_rapm_enrichment_from_stints_backfills_missing_player_rapm():
    pipeline = SOTAPipeline(
        SOTAPipelineConfig(
            enforce_feed_freshness=False,
            min_rapm_players_per_team=3,
        )
    )
    players = [
        pipeline._player_from_dict(
            "duke",
            {
                "player_id": f"duke_p{i}",
                "name": f"P{i}",
                "position": "PG",
                "minutes_per_game": 30 - i,
                "games_played": 30,
                "usage_rate": 20,
            },
        )
        for i in range(5)
    ]
    team_block = {
        "stints": [
            {"players": ["duke_p0", "duke_p1", "duke_p2"], "plus_minus": 4, "possessions": 10},
            {"players": ["duke_p1", "duke_p2", "duke_p3"], "plus_minus": -2, "possessions": 8},
            {"players": ["duke_p0", "duke_p3", "duke_p4"], "plus_minus": 3, "possessions": 9},
            {"players": ["duke_p2", "duke_p3", "duke_p4"], "plus_minus": -1, "possessions": 7},
        ]
    }

    pipeline._enrich_roster_rapm(players, team_block)
    non_zero = sum(1 for p in players if abs(p.rapm_total) > 1e-8)
    assert non_zero >= 3
