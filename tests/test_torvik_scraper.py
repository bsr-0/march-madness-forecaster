"""Unit tests for the BartTorvik scraper hardening (Part 1–5).

Covers:
- TorVikValidator range checks
- strategy_used telemetry
- _cache_has_valid_four_factors
- _aggregate_player_csv
- _dict_to_team round-trip
"""

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data.scrapers.torvik import BartTorvikScraper, TorVikTeam, TorVikValidator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scraper(tmp_path):
    return BartTorvikScraper(cache_dir=str(tmp_path))


@pytest.fixture
def scraper_no_cache():
    return BartTorvikScraper(cache_dir=None)


def _make_team(**kwargs) -> TorVikTeam:
    defaults = dict(
        team_id="duke",
        name="Duke",
        conference="ACC",
        t_rank=1,
        barthag=0.97,
        adj_offensive_efficiency=120.5,
        adj_defensive_efficiency=91.2,
        adj_tempo=70.3,
        effective_fg_pct=0.57,
        turnover_rate=0.15,
        offensive_reb_rate=0.32,
        free_throw_rate=0.35,
        opp_effective_fg_pct=0.45,
        opp_turnover_rate=0.18,
        defensive_reb_rate=0.72,
        opp_free_throw_rate=0.31,
        wins=30,
        losses=5,
    )
    defaults.update(kwargs)
    return TorVikTeam(**defaults)


# ---------------------------------------------------------------------------
# TorVikValidator
# ---------------------------------------------------------------------------


class TestTorVikValidator:
    def test_valid_team_no_warnings(self):
        team = _make_team()
        warnings = TorVikValidator.validate_team(team)
        assert warnings == [], f"Unexpected warnings: {warnings}"

    def test_barthag_out_of_range_triggers_warning(self):
        team = _make_team(barthag=1.5)
        warnings = TorVikValidator.validate_team(team)
        assert any("barthag" in w for w in warnings)

    def test_negative_barthag_triggers_warning(self):
        team = _make_team(barthag=-0.1)
        warnings = TorVikValidator.validate_team(team)
        assert any("barthag" in w for w in warnings)

    def test_adjoe_too_low_triggers_warning(self):
        team = _make_team(adj_offensive_efficiency=50.0)
        warnings = TorVikValidator.validate_team(team)
        assert any("adj_offensive_efficiency" in w for w in warnings)

    def test_adjoe_too_high_triggers_warning(self):
        team = _make_team(adj_offensive_efficiency=150.0)
        warnings = TorVikValidator.validate_team(team)
        assert any("adj_offensive_efficiency" in w for w in warnings)

    def test_efg_out_of_range_triggers_warning(self):
        team = _make_team(effective_fg_pct=0.90)
        warnings = TorVikValidator.validate_team(team)
        assert any("effective_fg_pct" in w for w in warnings)

    def test_nan_values_skipped(self):
        """NaN means 'not scraped from this source' — should not warn."""
        team = _make_team(
            effective_fg_pct=math.nan,
            turnover_rate=math.nan,
        )
        warnings = TorVikValidator.validate_team(team)
        # Should not have warnings about the NaN fields
        assert not any("effective_fg_pct" in w for w in warnings)
        assert not any("turnover_rate" in w for w in warnings)

    def test_zero_values_skipped(self):
        """Zero means 'not available from this source' — should not warn."""
        team = _make_team(three_pt_pct=0.0, ft_pct=0.0)
        warnings = TorVikValidator.validate_team(team)
        assert not any("three_pt_pct" in w or "ft_pct" in w for w in warnings)

    def test_empty_conference_triggers_warning(self):
        team = _make_team(conference="")
        warnings = TorVikValidator.validate_team(team)
        assert any("conference" in w for w in warnings)

    def test_unreasonable_game_count_triggers_warning(self):
        team = _make_team(wins=50, losses=2)
        warnings = TorVikValidator.validate_team(team)
        assert any("wins+losses" in w for w in warnings)

    def test_validate_teams_returns_dict(self):
        teams = [_make_team(), _make_team(team_id="kansas", name="Kansas", barthag=99.0)]
        result = TorVikValidator.validate_teams(teams)
        assert isinstance(result, dict)
        assert "kansas" in result
        assert "duke" not in result

    def test_validate_four_factors_range_check(self):
        data = {
            "duke": {
                "effective_fg_pct": 0.57,
                "turnover_rate": 0.15,
                "offensive_reb_rate": 0.32,
                "free_throw_rate": 0.35,
                "opp_effective_fg_pct": 0.45,
                "opp_turnover_rate": 0.18,
                "defensive_reb_rate": 0.72,
                "opp_free_throw_rate": 0.31,
            },
            "bad_team": {
                "effective_fg_pct": 1.5,  # out of range
                "turnover_rate": 0.15,
                "offensive_reb_rate": 0.32,
                "free_throw_rate": 0.35,
                "opp_effective_fg_pct": 0.45,
                "opp_turnover_rate": 0.18,
                "defensive_reb_rate": 0.72,
                "opp_free_throw_rate": 0.31,
            },
        }
        result = TorVikValidator.validate_four_factors(data)
        assert "bad_team" in result
        assert "duke" not in result


# ---------------------------------------------------------------------------
# strategy_used telemetry
# ---------------------------------------------------------------------------


class TestStrategyTelemetry:
    def test_fetch_strategy_attribute_exists(self, scraper):
        assert hasattr(scraper, "_fetch_strategy")
        assert isinstance(scraper._fetch_strategy, dict)

    def test_fetch_strategy_property_returns_copy(self, scraper):
        prop = scraper.fetch_strategy
        assert isinstance(prop, dict)
        # Modifying the copy should not affect internal state
        prop["test"] = "value"
        assert "test" not in scraper._fetch_strategy

    def test_cbbstat_api_strategy_recorded(self, scraper):
        """When cbbstat API succeeds, strategy_used is 'cbbstat_api'."""
        fake_team = _make_team()
        with patch.object(scraper, "_rankings_from_cbbstat_api", return_value=[fake_team]):
            with patch.object(scraper, "_save_to_cache"):
                teams = scraper.fetch_current_rankings(year=2024)
        assert len(teams) == 1
        assert scraper._fetch_strategy.get("rankings") == "cbbstat_api"

    def test_csv_fallback_strategy_recorded(self, scraper):
        """When CSV fallback is used for four factors, strategy is recorded."""
        fake_ff = {"duke": {"effective_fg_pct": 0.57, "turnover_rate": 0.15,
                            "offensive_reb_rate": 0.32, "free_throw_rate": 0.35,
                            "opp_effective_fg_pct": 0.45, "opp_turnover_rate": 0.18,
                            "defensive_reb_rate": 0.72, "opp_free_throw_rate": 0.31}}
        with patch.object(scraper, "_four_factors_from_cbbstat_api", return_value={}):
            with patch.object(scraper, "_four_factors_from_player_csv", return_value=fake_ff):
                with patch.object(scraper, "_save_to_cache"):
                    ff = scraper.fetch_four_factors(year=2024)
        assert ff == fake_ff
        assert scraper._fetch_strategy.get("four_factors") == "csv_fallback"


# ---------------------------------------------------------------------------
# _cache_has_valid_four_factors
# ---------------------------------------------------------------------------


class TestCacheValidation:
    def test_valid_cache_accepted(self, scraper):
        cache = {
            "duke": {"effective_fg_pct": 0.57, "offensive_reb_rate": 0.32, "turnover_rate": 0.15},
            "kansas": {"effective_fg_pct": 0.54, "offensive_reb_rate": 0.35, "turnover_rate": 0.18},
        }
        assert scraper._cache_has_valid_four_factors(cache) is True

    def test_all_zero_orb_and_to_rejected(self, scraper):
        cache = {
            "duke": {"effective_fg_pct": 0.57, "offensive_reb_rate": 0.0, "turnover_rate": 0.0},
            "kansas": {"effective_fg_pct": 0.54, "offensive_reb_rate": 0.0, "turnover_rate": 0.0},
        }
        assert scraper._cache_has_valid_four_factors(cache) is False

    def test_empty_cache_rejected(self, scraper):
        assert scraper._cache_has_valid_four_factors({}) is False

    def test_none_cache_rejected(self, scraper):
        assert scraper._cache_has_valid_four_factors(None) is False  # type: ignore


# ---------------------------------------------------------------------------
# _aggregate_player_csv
# ---------------------------------------------------------------------------


class TestAggregatePlayerCsv:
    def _make_csv_row(self, player, team, conf, min_pct, ftm, fta, fg2m, fg2a, fg3m, fg3a):
        # Columns: [0]=player [1]=team [2]=conf [3]=gp [4]=min_pct
        # [7]=efg [9]=orb [10]=drb [12]=to [13]=ftm [14]=fta [15]=ft%
        # [16]=2pm [17]=2pa [18]=? [19]=3pm [20]=3pa
        cols = [""] * 22
        cols[0] = player
        cols[1] = team
        cols[2] = conf
        cols[3] = "30"       # gp
        cols[4] = str(min_pct)
        cols[7] = "0.5"      # efg (unused in aggregate)
        cols[9] = "0.04"     # orb_pct
        cols[10] = "0.20"    # drb_pct
        cols[12] = "0.15"    # to_pct
        cols[13] = str(ftm)
        cols[14] = str(fta)
        cols[15] = str(ftm / fta if fta else 0)
        cols[16] = str(fg2m)
        cols[17] = str(fg2a)
        cols[18] = ""
        cols[19] = str(fg3m)
        cols[20] = str(fg3a)
        return ",".join(cols)

    def test_basic_aggregation(self, scraper):
        rows = [
            self._make_csv_row("Player A", "Duke", "ACC", 25.0, 50, 60, 100, 200, 30, 80),
            self._make_csv_row("Player B", "Duke", "ACC", 20.0, 40, 50, 80, 160, 25, 70),
        ]
        csv_text = "\n".join(rows)
        result = scraper._aggregate_player_csv(csv_text)
        assert "duke" in result or any("duke" in k for k in result)
        # Find the duke entry
        duke_key = next(k for k in result if "duke" in k.lower())
        t = result[duke_key]
        assert t["ftm"] == 90.0
        assert t["fta"] == 110.0
        assert t["fg2m"] == 180.0
        assert t["fg3m"] == 55.0

    def test_short_row_skipped(self, scraper):
        csv_text = "Player A,Duke\n"  # too short
        result = scraper._aggregate_player_csv(csv_text)
        # Should not crash; duke entry may not exist or be empty
        assert isinstance(result, dict)

    def test_multiple_teams(self, scraper):
        rows = [
            self._make_csv_row("P1", "Duke", "ACC", 25.0, 50, 60, 100, 200, 30, 80),
            self._make_csv_row("P2", "Kansas", "B12", 25.0, 45, 55, 90, 180, 28, 75),
        ]
        result = scraper._aggregate_player_csv("\n".join(rows))
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _dict_to_team round-trip
# ---------------------------------------------------------------------------


class TestDictToTeam:
    def test_round_trip(self, scraper):
        team = _make_team()
        d = team.to_dict()
        reconstructed = scraper._dict_to_team(d)
        assert reconstructed.team_id == team.team_id
        assert reconstructed.barthag == team.barthag
        assert reconstructed.adj_offensive_efficiency == team.adj_offensive_efficiency

    def test_missing_fields_use_defaults(self, scraper):
        minimal = {"team_id": "test_team", "name": "Test", "conference": "SEC"}
        team = scraper._dict_to_team(minimal)
        assert team.t_rank == 999
        assert team.barthag == 0.5
        assert team.adj_offensive_efficiency == 100.0


# ---------------------------------------------------------------------------
# Rankings CSV fallback
# ---------------------------------------------------------------------------


class TestRankingsCsvFallback:
    def test_rankings_csv_fallback_parses_teams(self, scraper):
        """When cbbstat fails, _rankings_from_csv returns valid TorVikTeam objects."""
        csv_content = (
            "rank,team,conf,adj_o,adj_d,barthag,adj_t,wab,wins,losses\n"
            "1,Duke,ACC,122.0,93.0,0.97,70.0,8.5,30,5\n"
            "2,Kansas,B12,118.0,95.0,0.94,68.5,7.2,28,7\n"
        )
        fake_resp = MagicMock()
        fake_resp.text = csv_content

        with patch.object(scraper, "_get_with_retry", return_value=fake_resp):
            teams = scraper._rankings_from_csv(2026)

        assert len(teams) == 2
        assert any("duke" in t.team_id.lower() for t in teams)
        # Four Factors should be NaN
        duke = next(t for t in teams if "duke" in t.team_id.lower())
        assert math.isnan(duke.effective_fg_pct)
        assert math.isnan(duke.turnover_rate)

    def test_rankings_csv_fallback_strategy_recorded(self, scraper):
        """When cbbstat API fails, rankings should fall through to CSV."""
        fake_team = _make_team()
        with patch.object(scraper, "_rankings_from_cbbstat_api", return_value=[]):
            with patch.object(scraper, "_rankings_from_csv", return_value=[fake_team]):
                with patch.object(scraper, "_save_to_cache"):
                    teams = scraper.fetch_current_rankings(year=2024)

        assert len(teams) == 1
        assert scraper._fetch_strategy.get("rankings") == "csv_fallback"


# ---------------------------------------------------------------------------
# Structural: HTML methods removed
# ---------------------------------------------------------------------------


class TestNoHtmlMethods:
    def test_no_html_parse_methods_exist(self):
        """HTML parsing methods should not exist after refactor."""
        scraper = BartTorvikScraper()
        assert not hasattr(scraper, "_parse_rankings_page")
        assert not hasattr(scraper, "_parse_four_factors_page")
        assert not hasattr(scraper, "_parse_shooting_page")
        assert not hasattr(scraper, "_extract_team_id")
