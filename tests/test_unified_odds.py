"""Tests for the unified betting odds data layer."""

import pytest
from src.data.scrapers.unified_odds import (
    UnifiedGameOdds,
    compute_team_market_features,
    normalize_from_covers,
    normalize_from_sbr,
    normalize_from_sbro,
)
from src.data.features.feature_engineering import TEAM_FEATURE_DIM, TeamFeatures


class TestNormalizers:
    def test_normalize_sbro(self):
        raw = {
            "season": 2019,
            "game_date": "2019-03-21",
            "home_team_id": "virginia",
            "away_team_id": "towson",
            "spread": 26.0,
            "spread_open": 25.0,
            "total": 129.0,
            "total_open": 127.5,
            "moneyline_home": -16500.0,
            "moneyline_away": 6500.0,
            "implied_prob_home": 0.9940,
            "implied_prob_away": 0.0060,
            "home_score": 73,
            "away_score": 42,
            "is_neutral": False,
        }
        g = normalize_from_sbro(raw)
        assert g.source == "sbro"
        assert g.spread == 26.0
        assert g.spread_open == 25.0
        assert g.moneyline_home == -16500.0
        assert g.home_score == 73

    def test_normalize_covers(self):
        raw = {
            "season": 2023,
            "game_date": "2023-03-16",
            "home_team_id": "maryland",
            "away_team_id": "west_virginia",
            "spread": -2.5,
            "total": 136.5,
            "implied_prob_home": 0.6,
            "implied_prob_away": 0.4,
            "home_score": 67,
            "away_score": 65,
            "is_neutral": True,
        }
        g = normalize_from_covers(raw)
        assert g.source == "covers"
        assert g.spread == -2.5
        assert g.moneyline_home == 0.0  # Covers has no moneylines
        assert g.spread_open == 0.0

    def test_normalize_sbr(self):
        raw = {
            "season": 2025,
            "game_date": "2025-03-18",
            "home_team_id": "alabama_state_hornets",
            "away_team_id": "st_francis_pa_red_flash",
            "home_team_name": "Alabama State Hornets",
            "away_team_name": "St. Francis (PA) Red Flash",
            "spread": -4.5,
            "moneyline_home": -174.0,
            "moneyline_away": 164.0,
            "total": 136.5,
            "implied_prob_home": 0.626,
            "implied_prob_away": 0.374,
        }
        g = normalize_from_sbr(raw)
        assert g.source == "sbr"
        assert g.home_score == 0  # SBR has no scores
        assert g.is_neutral is False


class TestComputeTeamMarketFeatures:
    def _make_odds(self, dates_and_spreads, team="duke"):
        """Helper: create game odds for a team."""
        games = []
        for i, (d, spread) in enumerate(dates_and_spreads):
            imp = 1.0 / (1.0 + 10.0 ** (spread / 10.0))
            games.append(UnifiedGameOdds(
                season=2024,
                game_date=d,
                home_team_id=team,
                away_team_id=f"opp_{i}",
                spread=spread,
                implied_prob_home=imp,
                implied_prob_away=1 - imp,
            ))
        return games

    def test_basic(self):
        games = self._make_odds([
            ("2024-01-10", -5.0),
            ("2024-01-15", -3.0),
        ])
        result = compute_team_market_features("duke", "2024-02-01", games)
        assert result["market_implied_prob"] > 0.5  # favored in both
        assert result["market_spread"] < 0  # negative = favored

    def test_pit_constraint(self):
        """Games on or after as_of_date must be excluded."""
        games = self._make_odds([
            ("2024-01-10", -5.0),
            ("2024-03-15", -10.0),  # after cutoff
        ])
        result = compute_team_market_features("duke", "2024-03-15", games)
        # Only the Jan 10 game should count
        assert result["market_spread"] == pytest.approx(-5.0)

    def test_empty_returns_zeros(self):
        result = compute_team_market_features("duke", "2024-03-15", [])
        assert result["market_implied_prob"] == 0.0
        assert result["market_spread"] == 0.0

    def test_away_team_perspective(self):
        """When team is away, spread should be flipped."""
        games = [UnifiedGameOdds(
            season=2024,
            game_date="2024-01-10",
            home_team_id="unc",
            away_team_id="duke",
            spread=-5.0,  # UNC favored by 5
            implied_prob_home=0.65,
            implied_prob_away=0.35,
        )]
        result = compute_team_market_features("duke", "2024-02-01", games)
        assert result["market_spread"] == pytest.approx(5.0)  # +5 from Duke's perspective
        assert result["market_implied_prob"] == pytest.approx(0.35)


class TestFeatureVectorDim:
    def test_dim_is_54(self):
        assert TEAM_FEATURE_DIM == 56

    def test_feature_names_include_market(self):
        names = TeamFeatures.get_feature_names()
        assert len(names) == 56
        assert "market_implied_prob" in names
        assert "market_spread" in names
        assert names[52] == "market_implied_prob"
        assert names[53] == "market_spread"
