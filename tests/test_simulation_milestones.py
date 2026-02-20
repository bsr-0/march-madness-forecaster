"""Regression tests for tournament round milestone accounting."""

from src.simulation.monte_carlo import MonteCarloEngine, SimulationConfig, TournamentBracket, TournamentTeam


def _seed_from_team_id(team_id: str) -> int:
    return int(team_id.split("_")[-1])


def _build_standard_teams():
    teams_by_region = {}
    for region in ("East", "West", "South", "Midwest"):
        region_teams = []
        for seed in range(1, 17):
            team_id = f"{region.lower()}_{seed}"
            region_teams.append(
                TournamentTeam(
                    team_id=team_id,
                    seed=seed,
                    region=region,
                    strength=100.0 - seed,
                )
            )
        teams_by_region[region] = region_teams
    return teams_by_region


def test_simulation_tracks_correct_round_sizes():
    teams_by_region = _build_standard_teams()
    bracket = TournamentBracket.create_standard_bracket(teams_by_region)

    def predict_fn(team1_id: str, team2_id: str) -> float:
        return 0.8 if _seed_from_team_id(team1_id) < _seed_from_team_id(team2_id) else 0.2

    engine = MonteCarloEngine(
        predict_fn,
        config=SimulationConfig(
            num_simulations=1,
            noise_std=0.0,
            injury_probability=0.0,
            random_seed=7,
            batch_size=1,
        ),
    )

    results = engine.simulate_tournament(bracket, show_progress=False)

    assert len(results.round_of_32_odds) == 32
    assert len(results.sweet_sixteen_odds) == 16
    assert len(results.elite_eight_odds) == 8
    assert len(results.final_four_odds) == 4
    assert len(results.championship_odds) == 1
    assert abs(sum(results.championship_odds.values()) - 1.0) < 1e-9


def test_simulation_se_and_ci_populated():
    """Fix 13: Verify SE and Wilson CI fields are populated in results."""
    teams_by_region = _build_standard_teams()
    bracket = TournamentBracket.create_standard_bracket(teams_by_region)

    def predict_fn(team1_id: str, team2_id: str) -> float:
        return 0.7 if _seed_from_team_id(team1_id) < _seed_from_team_id(team2_id) else 0.3

    engine = MonteCarloEngine(
        predict_fn,
        config=SimulationConfig(
            num_simulations=100,
            noise_std=0.04,
            injury_probability=0.0,
            random_seed=42,
            batch_size=100,
        ),
    )

    results = engine.simulate_tournament(bracket, show_progress=False)

    # SE and CI should be populated for teams that appear in results
    assert len(results.simulation_se) > 0, "simulation_se should be populated"
    assert len(results.ci_lower) > 0, "ci_lower should be populated"
    assert len(results.ci_upper) > 0, "ci_upper should be populated"

    # For each team with championship odds, verify SE and CI consistency
    for team, p in results.championship_odds.items():
        assert team in results.simulation_se, f"Missing SE for {team}"
        assert "CHAMP" in results.simulation_se[team], f"Missing CHAMP SE for {team}"
        se = results.simulation_se[team]["CHAMP"]
        lo = results.ci_lower[team]["CHAMP"]
        hi = results.ci_upper[team]["CHAMP"]
        assert se >= 0, f"SE should be non-negative for {team}"
        assert lo <= hi, f"CI lower should <= upper for {team}"
        assert lo >= 0, f"CI lower should be >= 0 for {team}"
        assert hi <= 1, f"CI upper should be <= 1 for {team}"
