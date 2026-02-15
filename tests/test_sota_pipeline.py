"""Integration tests for the SOTA 2026 pipeline with real-data inputs."""

import json

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

    kenpom = {
        "teams": [
            {
                "team_id": t["name"].lower().replace(" ", "_"),
                "name": t["name"],
                "conference": "TestConf",
                "adj_efficiency_margin": (17 - t["seed"]) * 1.5,
                "adj_offensive_efficiency": 105 + (17 - t["seed"]),
                "adj_defensive_efficiency": 95 + t["seed"] * 0.7,
                "adj_tempo": 67.5,
                "overall_rank": t["seed"],
                "offensive_rank": t["seed"] + 5,
                "defensive_rank": t["seed"] + 6,
                "luck": 0.01,
                "sos_adj_em": 5.0,
                "sos_opp_o": 103.0,
                "sos_opp_d": 100.0,
                "ncsos_adj_em": 1.0,
                "wins": 24,
                "losses": 8,
            }
            for t in teams
        ]
    }

    torvik = {
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

    shotquality_teams = {
        "teams": [
            {
                "team_id": t["name"].lower().replace(" ", "_"),
                "team_name": t["name"],
                "offensive_xp_per_possession": 1.05,
                "defensive_xp_per_possession": 0.97,
                "rim_rate": 0.33,
                "three_rate": 0.38,
                "midrange_rate": 0.29,
            }
            for t in teams
        ]
    }

    roster_payload = {"teams": []}
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

    games = []
    for idx in range(0, len(teams), 2):
        t1 = teams[idx]["name"].lower().replace(" ", "_")
        t2 = teams[idx + 1]["name"].lower().replace(" ", "_")
        possessions = []
        for p in range(50):
            offense = t1 if p % 2 == 0 else t2
            points = 2 if p % 3 == 0 else 0
            possessions.append(
                {
                    "possession_id": f"g{idx:03d}_p{p}",
                    "game_id": f"g{idx:03d}",
                    "team_id": offense,
                    "period": 1 if p < 25 else 2,
                    "game_clock": float(1200 - (p % 25) * 40),
                    "shot_type": "above_break_three" if p % 4 == 0 else "rim",
                    "is_contested": False,
                    "xp": 1.1,
                    "actual_points": points,
                    "outcome": "made" if points > 0 else "missed",
                }
            )
        games.append(
            {
                "game_id": f"g{idx:03d}",
                "team_id": t1,
                "opponent_id": t2,
                "possessions": possessions,
            }
        )

    shotquality_games = {"games": games}

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
        "sources": ["espn"],
        "timestamp": "2026-03-17T12:00:00Z",
    }

    return {
        "teams": {"teams": teams},
        "kenpom": kenpom,
        "torvik": torvik,
        "shotquality_teams": shotquality_teams,
        "rosters": roster_payload,
        "shotquality_games": shotquality_games,
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
        kenpom_json=paths["kenpom"],
        torvik_json=paths["torvik"],
        shotquality_teams_json=paths["shotquality_teams"],
        shotquality_games_json=paths["shotquality_games"],
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

    baseline = report["artifacts"]["baseline_training"]
    assert baseline["model"] in {"lightgbm", "logistic_regression", "none"}
    assert "model_uncertainty" in report["artifacts"]
    assert report["artifacts"]["public_pick_sources"] == ["espn"]
    for pick in report["artifacts"]["top_leverage_picks"][:10]:
        assert 0.0 <= pick["public_pick_percentage"] <= 1.0


def test_sota_pipeline_output_file(tmp_path):
    paths = _write_payloads(tmp_path)
    output_path = tmp_path / "sota_report.json"

    config = SOTAPipelineConfig(
        num_simulations=80,
        pool_size=20,
        teams_json=paths["teams"],
        kenpom_json=paths["kenpom"],
        torvik_json=paths["torvik"],
        shotquality_teams_json=paths["shotquality_teams"],
        shotquality_games_json=paths["shotquality_games"],
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


def test_sota_pipeline_requires_real_data():
    config = SOTAPipelineConfig(num_simulations=20, pool_size=10)
    pipeline = SOTAPipeline(config)

    try:
        pipeline.run()
        assert False, "Expected DataRequirementError"
    except DataRequirementError:
        assert True
