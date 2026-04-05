"""Tests for Phase 5: Betting Market Integration."""

import pytest
import json
import tempfile
import os
from src.data.scrapers.betting_markets import (
    BettingMarketOdds,
    MarketConsensus,
    TheOddsAPIScraper,
    american_to_probability,
    decimal_to_probability,
    remove_vig,
    compute_market_consensus,
    blend_with_model,
)


class TestAmericanToProbability:
    def test_plus_500(self):
        """'+500 underdog → ~16.7% implied probability."""
        prob = american_to_probability(500)
        assert abs(prob - 100 / 600) < 0.001

    def test_minus_150(self):
        """-150 favorite → 60% implied probability."""
        prob = american_to_probability(-150)
        assert abs(prob - 150 / 250) < 0.001

    def test_even_odds(self):
        """+100 → 50%."""
        prob = american_to_probability(100)
        assert abs(prob - 0.5) < 0.001

    def test_heavy_favorite(self):
        """-1000 → ~90.9%."""
        prob = american_to_probability(-1000)
        assert prob > 0.9

    def test_heavy_underdog(self):
        """+10000 → ~1%."""
        prob = american_to_probability(10000)
        assert prob < 0.02

    def test_zero_odds(self):
        prob = american_to_probability(0)
        assert prob == 0.5


class TestDecimalToProbability:
    def test_decimal_2(self):
        """Decimal 2.0 → 50%."""
        assert abs(decimal_to_probability(2.0) - 0.5) < 0.001

    def test_decimal_6(self):
        """Decimal 6.0 → ~16.7%."""
        assert abs(decimal_to_probability(6.0) - 1 / 6) < 0.001

    def test_decimal_zero(self):
        assert decimal_to_probability(0) == 0.0

    def test_decimal_negative(self):
        assert decimal_to_probability(-1.0) == 0.0


class TestRemoveVig:
    def test_already_normalized(self):
        probs = {"A": 0.5, "B": 0.3, "C": 0.2}
        result = remove_vig(probs)
        assert abs(sum(result.values()) - 1.0) < 0.001

    def test_with_overround(self):
        """Typical bookmaker overround: probabilities sum to ~1.05."""
        probs = {"A": 0.525, "B": 0.315, "C": 0.21}  # sum = 1.05
        result = remove_vig(probs)
        assert abs(sum(result.values()) - 1.0) < 0.001
        assert result["A"] < 0.525  # Should decrease

    def test_empty_dict(self):
        result = remove_vig({})
        assert result == {}

    def test_preserves_relative_order(self):
        probs = {"A": 0.6, "B": 0.3, "C": 0.2}
        result = remove_vig(probs)
        assert result["A"] > result["B"] > result["C"]


class TestComputeMarketConsensus:
    def _make_source(self, probs, source_name="fanduel"):
        return {
            team_id: BettingMarketOdds(
                team_id=team_id,
                team_name=team_id,
                season=2026,
                source=source_name,
                championship_odds=0,
                implied_probability=prob,
            )
            for team_id, prob in probs.items()
        }

    def test_single_source(self):
        source = self._make_source({"duke": 0.20, "unc": 0.10})
        consensus = compute_market_consensus([source])
        assert len(consensus.team_probabilities) == 2
        assert "fanduel" in consensus.sources

    def test_multi_source_averages(self):
        s1 = self._make_source({"duke": 0.20, "unc": 0.10}, "fanduel")
        s2 = self._make_source({"duke": 0.30, "unc": 0.10}, "draftkings")
        consensus = compute_market_consensus([s1, s2], adjust_vig=False)
        # Average of 0.20 and 0.30 = 0.25
        assert abs(consensus.team_probabilities["duke"] - 0.25) < 0.01

    def test_vig_adjustment(self):
        source = self._make_source({"duke": 0.60, "unc": 0.50})  # sum = 1.10
        consensus = compute_market_consensus([source], adjust_vig=True)
        total = sum(consensus.team_probabilities.values())
        assert abs(total - 1.0) < 0.01
        assert consensus.vig_adjusted is True

    def test_no_vig_adjustment(self):
        source = self._make_source({"duke": 0.60, "unc": 0.50})
        consensus = compute_market_consensus([source], adjust_vig=False)
        assert consensus.vig_adjusted is False

    def test_empty_sources(self):
        consensus = compute_market_consensus([])
        assert consensus.team_probabilities == {}
        assert consensus.sources == []

    def test_confidence_weighting(self):
        """Stale data should have less influence."""
        fresh = {
            "duke": BettingMarketOdds("duke", "Duke", 2026, "fd", 0, 0.30, confidence=1.0),
        }
        stale = {
            "duke": BettingMarketOdds("duke", "Duke", 2026, "dk", 0, 0.10, confidence=0.1),
        }
        consensus = compute_market_consensus([fresh, stale], adjust_vig=False)
        # Fresh (0.30, w=1.0) vs stale (0.10, w=0.1) → weighted avg closer to 0.30
        assert consensus.team_probabilities["duke"] > 0.25

    def test_source_weights(self):
        s1 = self._make_source({"duke": 0.20}, "fanduel")
        s2 = self._make_source({"duke": 0.40}, "draftkings")
        consensus = compute_market_consensus(
            [s1, s2],
            source_weights={"fanduel": 2.0, "draftkings": 1.0},
            adjust_vig=False,
        )
        # Weighted avg: (0.20*2 + 0.40*1) / 3 ≈ 0.267
        assert consensus.team_probabilities["duke"] < 0.30


class TestMarketConsensus:
    def test_get_top_teams(self):
        consensus = MarketConsensus(
            team_probabilities={"A": 0.30, "B": 0.20, "C": 0.50},
            sources=["test"],
        )
        top = consensus.get_top_teams(2)
        assert len(top) == 2
        assert top[0][0] == "C"
        assert top[1][0] == "A"


class TestBlendWithModel:
    def test_pure_model(self):
        model = {"duke": 0.20, "unc": 0.10}
        market = {"duke": 0.30, "unc": 0.05}
        result = blend_with_model(model, market, market_weight=0.0)
        assert result["duke"] == 0.20
        assert result["unc"] == 0.10

    def test_pure_market(self):
        model = {"duke": 0.20, "unc": 0.10}
        market = {"duke": 0.30, "unc": 0.05}
        result = blend_with_model(model, market, market_weight=1.0)
        assert abs(result["duke"] - 0.30) < 0.001

    def test_equal_blend(self):
        model = {"duke": 0.20}
        market = {"duke": 0.40}
        result = blend_with_model(model, market, market_weight=0.5)
        assert abs(result["duke"] - 0.30) < 0.001

    def test_missing_market_team(self):
        model = {"duke": 0.20, "unc": 0.10}
        market = {"duke": 0.30}  # unc missing
        result = blend_with_model(model, market, market_weight=0.5)
        # unc: 0.5 * 0.10 + 0.5 * 0.0 = 0.05
        assert abs(result["unc"] - 0.05) < 0.001

    def test_missing_model_team(self):
        model = {"duke": 0.20}
        market = {"duke": 0.30, "unc": 0.15}
        result = blend_with_model(model, market, market_weight=0.5)
        # unc: 0.5 * 0.0 + 0.5 * 0.15 = 0.075
        assert abs(result["unc"] - 0.075) < 0.001


class TestTheOddsAPIScraper:
    """Tests for The Odds API scraper."""

    def test_scrape_no_api_key(self):
        """Without API key, scraper should return empty dict (or raise LeakageError post-tournament)."""
        from datetime import date
        from src.pipeline.config import TOURNAMENT_START_DATES
        from src.exceptions import LeakageError

        scraper = TheOddsAPIScraper()
        season = 2026
        cutoff = TOURNAMENT_START_DATES.get(season)
        # Ensure env var is not set
        old_val = os.environ.pop("THE_ODDS_API_KEY", None)
        try:
            if cutoff and date.today() >= cutoff:
                with pytest.raises(LeakageError):
                    scraper.scrape(season)
            else:
                result = scraper.scrape(season)
                assert result == {}
        finally:
            if old_val is not None:
                os.environ["THE_ODDS_API_KEY"] = old_val

    def test_parse_outrights_response(self):
        """Parse a championship futures response."""
        scraper = TheOddsAPIScraper()
        data = [
            {
                "id": "abc123",
                "sport_key": "basketball_ncaab",
                "bookmakers": [
                    {
                        "key": "fanduel",
                        "title": "FanDuel",
                        "markets": [
                            {
                                "key": "outrights",
                                "outcomes": [
                                    {"name": "Duke Blue Devils", "price": 500},
                                    {"name": "North Carolina Tar Heels", "price": 1000},
                                    {"name": "Kansas Jayhawks", "price": -150},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
        result = scraper._parse_response(data, 2026, "outrights")
        assert len(result) >= 3
        # Duke at +500 → ~16.7%
        duke = [v for v in result.values() if "Duke" in v.team_name]
        assert len(duke) == 1
        assert abs(duke[0].implied_probability - 100 / 600) < 0.01
        assert duke[0].source == "the_odds_api"

    def test_parse_h2h_response(self):
        """Parse a game-level moneyline response."""
        scraper = TheOddsAPIScraper()
        data = [
            {
                "id": "game1",
                "home_team": "Duke Blue Devils",
                "away_team": "UNC Tar Heels",
                "bookmakers": [
                    {
                        "key": "draftkings",
                        "title": "DraftKings",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Duke Blue Devils", "price": -200},
                                    {"name": "UNC Tar Heels", "price": 170},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
        result = scraper._parse_response(data, 2026, "h2h")
        assert len(result) == 2

    def test_parse_empty_response(self):
        scraper = TheOddsAPIScraper()
        result = scraper._parse_response([], 2026, "outrights")
        assert result == {}

    def test_parse_keeps_best_odds(self):
        """When multiple bookmakers list the same team, keep highest implied prob."""
        scraper = TheOddsAPIScraper()
        data = [
            {
                "id": "abc",
                "bookmakers": [
                    {
                        "key": "book1",
                        "title": "Book1",
                        "markets": [
                            {
                                "key": "outrights",
                                "outcomes": [
                                    {"name": "Duke Blue Devils", "price": 500},
                                ],
                            }
                        ],
                    },
                    {
                        "key": "book2",
                        "title": "Book2",
                        "markets": [
                            {
                                "key": "outrights",
                                "outcomes": [
                                    {"name": "Duke Blue Devils", "price": 400},
                                ],
                            }
                        ],
                    },
                ],
            }
        ]
        result = scraper._parse_response(data, 2026, "outrights")
        # +400 → 20%, +500 → 16.7% — should keep +400 (higher implied prob)
        duke_odds = [v for v in result.values() if "Duke" in v.team_name]
        assert len(duke_odds) == 1
        assert abs(duke_odds[0].implied_probability - 100 / 500) < 0.01

    def test_load_from_json_cache(self):
        """Test loading from JSON cache (inherits from base class)."""
        data = {
            "source": "the_odds_api",
            "season": 2026,
            "timestamp": "2026-03-17T12:00:00Z",
            "teams": {
                "duke": {
                    "team_name": "Duke",
                    "championship_odds": 500,
                    "implied_probability": 0.167,
                    "source": "the_odds_api",
                },
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            tmppath = f.name

        try:
            scraper = TheOddsAPIScraper()
            result = scraper.load_from_json(tmppath)
            assert "duke" in result
            assert result["duke"].implied_probability == 0.167
        finally:
            os.unlink(tmppath)
