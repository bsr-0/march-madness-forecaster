"""Integration tests for the SOTA 2026 pipeline with real-data inputs."""

import json

from src.models.team import Team
from src.pipeline.sota import DataRequirementError, SOTAPipeline, SOTAPipelineConfig


def _build_fixture_payloads():
    regions = ["East", "West", "South", "Midwest"]
    teams = []

    for region in regions:
        for seed in range(1, 17):
            team_name = f"{region} Seed {seed}"
            team_id = team_name.lower().replace(" ", "_")
            teams.append(
                {
                    "name": team_name,
                    "seed": seed,
                    "region": region,
                    "elo_rating": 1820 - seed * 24,
                    "stats": {
                        "offensive_efficiency": 102 - seed,
                        "defensive_efficiency": 96 + seed,
                        "strength_of_schedule": 80 - seed * 1.2,
                        "recent_performance": 70 + (17 - seed),
                        "tempo": 68,
                        "experience": 2.4,
                    },
                }
            )

    torvik = {
        "timestamp": "2026-03-17T12:00:00Z",
        "teams": [
            {
                "team_id": t["name"].lower().replace(" ", "_"),
                "name": t["name"],
                "conference": "TestConf",
                "t_rank": t["seed"],
                "barthag": 0.5 + (17 - t["seed"]) / 100,
                "adj_offensive_efficiency": 105 + (17 - t["seed"]),
                "adj_defensive_efficiency": 95 + t["seed"] * 0.7,
                "adj_tempo": 67.5,
                "effective_fg_pct": 0.50,
                "turnover_rate": 0.17,
                "offensive_reb_rate": 0.30,
                "free_throw_rate": 0.31,
                "opp_effective_fg_pct": 0.48,
                "opp_turnover_rate": 0.19,
                "defensive_reb_rate": 0.71,
                "opp_free_throw_rate": 0.29,
                "wab": 4.5,
                "wins": 24,
                "losses": 8,
            }
            for t in teams
        ]
    }

    roster_payload = {"timestamp": "2026-03-17T12:00:00Z", "teams": []}
    for t in teams:
        team_id = t["name"].lower().replace(" ", "_")
        roster_payload["teams"].append(
            {
                "team_id": team_id,
                "players": [
                    {
                        "player_id": f"{team_id}_p{i}",
                        "name": f"{t['name']} P{i}",
                        "position": ["PG", "SG", "SF", "PF", "C"][i % 5],
                        "minutes_per_game": 32 - i,
                        "games_played": 30,
                        "games_started": max(0, 30 - i),
                        "rapm_offensive": 0.8 - i * 0.03,
                        "rapm_defensive": 0.6 - i * 0.02,
                        "warp": 0.12 - i * 0.005,
                        "box_plus_minus": 3.0 - i * 0.2,
                        "usage_rate": 22 - i,
                        "injury_status": "healthy",
                        "is_transfer": i < 2,
                        "eligibility_year": 3,
                    }
                    for i in range(10)
                ],
            }
        )

    public_picks = {
        "teams": {
            t["name"].lower().replace(" ", "_"): {
                "team_name": t["name"],
                "seed": t["seed"],
                "region": t["region"],
                "round_of_64_pct": 90 - t["seed"],
                "round_of_32_pct": 70 - t["seed"],
                "sweet_16_pct": 45 - t["seed"],
                "elite_8_pct": 25 - t["seed"] * 0.8,
                "final_four_pct": 12 - t["seed"] * 0.5,
                "champion_pct": max(0.1, 6 - t["seed"] * 0.25),
            }
            for t in teams
        },
        "sources": ["espn", "yahoo", "cbs"],
        "timestamp": "2026-03-17T12:00:00Z",
    }

    return {
        "teams": {"teams": teams},
        "torvik": torvik,
        "rosters": roster_payload,
        "public_picks": public_picks,
    }


def _write_payloads(tmp_path):
    payloads = _build_fixture_payloads()
    paths = {}
    for key, payload in payloads.items():
        p = tmp_path / f"{key}.json"
        with open(p, "w") as f:
            json.dump(payload, f)
        paths[key] = str(p)
    return paths


def test_sota_pipeline_produces_rubric_artifacts(tmp_path):
    paths = _write_payloads(tmp_path)

    config = SOTAPipelineConfig(
        num_simulations=120,
        pool_size=64,
        calibration_method="isotonic",
        teams_json=paths["teams"],
        torvik_json=paths["torvik"],
        roster_json=paths["rosters"],
        public_picks_json=paths["public_picks"],
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


def test_sota_pipeline_output_file(tmp_path):
    paths = _write_payloads(tmp_path)
    output_path = tmp_path / "sota_report.json"

    config = SOTAPipelineConfig(
        num_simulations=80,
        pool_size=20,
        teams_json=paths["teams"],
        torvik_json=paths["torvik"],
        roster_json=paths["rosters"],
        public_picks_json=paths["public_picks"],
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
