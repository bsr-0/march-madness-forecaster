"""Data contract tests for Four Factors repair pipeline.

Validates that the enrichment functions correctly compute and apply
defensive and offensive Four Factors from game box scores, and that
the validation gate catches bad data.
"""

import pytest

from src.conference_tournament.data_enrichment import (
    compute_defensive_four_factors_from_games,
    compute_offensive_four_factors_from_games,
)
from src.data.ingestion.validators import validate_four_factors
from src.data.team_id_resolver import GameToTorvikResolver


def _make_game_pair(game_id, team_a, team_b, a_stats, b_stats):
    """Create a paired game record (two per-team rows)."""
    base_a = {
        "game_id": game_id, "team_id": team_a, "team_name": team_a,
        "fgm": 25, "fga": 60, "fg3m": 8, "fg3a": 20, "fta": 15,
        "turnovers": 12, "orb": 10, "drb": 25,
    }
    base_b = {
        "game_id": game_id, "team_id": team_b, "team_name": team_b,
        "fgm": 22, "fga": 55, "fg3m": 6, "fg3a": 18, "fta": 12,
        "turnovers": 14, "orb": 8, "drb": 22,
    }
    base_a.update(a_stats)
    base_b.update(b_stats)
    return [base_a, base_b]


def _make_mock_games(n_games=10):
    """Generate mock paired game records for testing."""
    records = []
    for i in range(n_games):
        records.extend(_make_game_pair(
            f"game_{i}", "team_alpha", "team_beta",
            {"fgm": 25 + i, "fga": 60, "fg3m": 8, "orb": 10, "drb": 25, "turnovers": 12},
            {"fgm": 22, "fga": 55, "fg3m": 6, "orb": 8, "drb": 22, "turnovers": 14},
        ))
    return records


@pytest.mark.data_contract
class TestDefensiveFourFactorsComputation:
    def test_computes_from_paired_records(self):
        games = _make_mock_games(5)
        result = compute_defensive_four_factors_from_games(games)
        assert len(result) >= 2
        for tid in result:
            d = result[tid]
            assert 0.3 <= d["opp_effective_fg_pct"] <= 0.65
            assert 0.05 <= d["opp_turnover_rate"] <= 0.40
            assert 0.1 <= d["opp_free_throw_rate"] <= 0.60

    def test_skips_unpaired_games(self):
        games = [{"game_id": "solo", "team_id": "lonely", "fga": 60, "fgm": 25}]
        result = compute_defensive_four_factors_from_games(games)
        assert "lonely" not in result

    def test_skips_low_fga(self):
        games = _make_game_pair("g1", "a", "b", {"fga": 0}, {"fga": 0})
        result = compute_defensive_four_factors_from_games(games)
        assert len(result) == 0


@pytest.mark.data_contract
class TestOffensiveFourFactorsComputation:
    def test_computes_from_paired_records(self):
        games = _make_mock_games(5)
        result = compute_offensive_four_factors_from_games(games)
        assert len(result) >= 2
        for tid in result:
            d = result[tid]
            assert 0.05 <= d["turnover_rate"] <= 0.35
            assert 0.10 <= d["offensive_reb_rate"] <= 0.50
            assert 0.40 <= d["defensive_reb_rate"] <= 0.90

    def test_requires_minimum_games(self):
        # Only 2 games — below the 3-game threshold
        games = _make_mock_games(2)
        result = compute_offensive_four_factors_from_games(games)
        assert len(result) == 0


@pytest.mark.data_contract
class TestResolverRemapIntegration:
    def test_remap_resolves_mascot_ids(self):
        resolver = GameToTorvikResolver({"michigan", "duke"})
        raw = {
            "michigan_wolverines": {"to_rate": 0.15},
            "duke_blue_devils": {"to_rate": 0.12},
        }
        remapped = resolver.remap_dict(raw)
        assert "michigan" in remapped
        assert "duke" in remapped

    def test_end_to_end_compute_and_remap(self):
        games = _make_mock_games(5)
        resolver = GameToTorvikResolver({"team_alpha", "team_beta"})
        def_ff = resolver.remap_dict(
            compute_defensive_four_factors_from_games(games)
        )
        assert "team_alpha" in def_ff or "team_beta" in def_ff


@pytest.mark.data_contract
class TestCutoffDateFilterLogging:
    def test_defensive_ff_logs_excluded_games(self, caplog):
        import logging
        games = _make_mock_games(5)
        # Add dates: 3 pre-cutoff, 2 post-cutoff
        for i, g in enumerate(games):
            day = 10 + (i // 2)  # games at days 10..14
            g["date"] = f"2026-03-{day:02d}"
        with caplog.at_level(logging.INFO):
            compute_defensive_four_factors_from_games(games, cutoff_date="2026-03-13")
        assert "excluded" in caplog.text.lower()

    def test_offensive_ff_logs_excluded_games(self, caplog):
        import logging
        games = _make_mock_games(5)
        for i, g in enumerate(games):
            day = 10 + (i // 2)
            g["date"] = f"2026-03-{day:02d}"
        with caplog.at_level(logging.INFO):
            compute_offensive_four_factors_from_games(games, cutoff_date="2026-03-13")
        assert "excluded" in caplog.text.lower()


@pytest.mark.data_contract
class TestValidateFourFactors:
    def _make_torvik_payload(self, n_teams=20, def_ff_zero=False, orb_bias=False):
        teams = []
        for i in range(n_teams):
            t = {
                "team_id": f"team_{i}",
                "opp_effective_fg_pct": 0.0 if def_ff_zero else 0.45 + i * 0.005,
                "opp_turnover_rate": 0.0 if def_ff_zero else 0.18,
                "opp_free_throw_rate": 0.0 if def_ff_zero else 0.30,
                "offensive_reb_rate": 0.08 if orb_bias else 0.29 + i * 0.002,
                "defensive_reb_rate": 0.19 if orb_bias else 0.72,
            }
            teams.append(t)
        return {"teams": teams}

    def test_passes_good_data(self):
        payload = self._make_torvik_payload()
        errors = validate_four_factors(payload)
        assert errors == []

    def test_catches_defensive_zeros(self):
        payload = self._make_torvik_payload(def_ff_zero=True)
        errors = validate_four_factors(payload)
        assert any("defensive FF = 0.0" in e for e in errors)

    def test_catches_orb_bias(self):
        payload = self._make_torvik_payload(orb_bias=True)
        errors = validate_four_factors(payload)
        assert any("ORB% median" in e for e in errors)

    def test_empty_payload(self):
        errors = validate_four_factors({"teams": []})
        assert any("no teams" in e for e in errors)
