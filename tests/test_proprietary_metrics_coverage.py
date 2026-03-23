"""Comprehensive unit tests for src.data.features.proprietary_metrics.

Covers all public functions, classes, dataclasses, and internal methods
with mocking for I/O dependencies.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.data.features.proprietary_metrics import (
    GameRecord,
    IncrementalMetricsEngine,
    ProprietaryMetricsEngine,
    ProprietaryTeamMetrics,
    _team_id,
    _to_float,
    team_games_to_game_records,
    torvik_to_game_records,
)


# ---------------------------------------------------------------------------
# Helpers: build game records for testing
# ---------------------------------------------------------------------------

def _make_game(
    game_id="g1",
    game_date="2025-01-15",
    team_id="team_a",
    team_name="Team A",
    opponent_id="team_b",
    points=75.0,
    opp_points=70.0,
    possessions=68.0,
    fga=60.0,
    fgm=28.0,
    fg3a=20.0,
    fg3m=8.0,
    fta=15.0,
    ftm=11.0,
    tov=12.0,
    orb=10.0,
    drb=25.0,
    opp_fga=58.0,
    opp_fgm=26.0,
    opp_fg3a=18.0,
    opp_fg3m=6.0,
    opp_fta=12.0,
    opp_ftm=9.0,
    opp_tov=14.0,
    opp_orb=8.0,
    opp_drb=22.0,
    ast=15.0,
    stl=7.0,
    blk=4.0,
    pf=16.0,
    opp_ast=12.0,
    opp_stl=5.0,
    opp_blk=3.0,
    opp_pf=14.0,
    is_home=False,
    is_neutral=True,
) -> GameRecord:
    return GameRecord(
        game_id=game_id,
        game_date=game_date,
        team_id=team_id,
        team_name=team_name,
        opponent_id=opponent_id,
        points=points,
        opp_points=opp_points,
        possessions=possessions,
        fga=fga, fgm=fgm, fg3a=fg3a, fg3m=fg3m,
        fta=fta, ftm=ftm, tov=tov, orb=orb, drb=drb,
        opp_fga=opp_fga, opp_fgm=opp_fgm, opp_fg3a=opp_fg3a, opp_fg3m=opp_fg3m,
        opp_fta=opp_fta, opp_ftm=opp_ftm, opp_tov=opp_tov, opp_orb=opp_orb, opp_drb=opp_drb,
        ast=ast, stl=stl, blk=blk, pf=pf,
        opp_ast=opp_ast, opp_stl=opp_stl, opp_blk=opp_blk, opp_pf=opp_pf,
        is_home=is_home, is_neutral=is_neutral,
    )


def _make_season_games(
    team_id="team_a",
    opponent_id="team_b",
    n_games=15,
    start_date="2025-01-01",
    win_ratio=0.6,
) -> List[GameRecord]:
    """Generate n_games for a single team across a season."""
    games = []
    for i in range(n_games):
        day = int(start_date.split("-")[2]) + i
        month = int(start_date.split("-")[1])
        if day > 28:
            day = day - 28
            month += 1
        date_str = f"2025-{month:02d}-{day:02d}"
        is_win = i < int(n_games * win_ratio)
        pts = 80.0 if is_win else 65.0
        opp_pts = 70.0 if is_win else 75.0
        games.append(_make_game(
            game_id=f"g_{team_id}_{i}",
            game_date=date_str,
            team_id=team_id,
            team_name=team_id.replace("_", " ").title(),
            opponent_id=f"{opponent_id}_{i % 5}",
            points=pts,
            opp_points=opp_pts,
            possessions=68.0 + i * 0.5,
            fg3a=20.0 + i % 3,
            fg3m=7.0 + (i % 4),
            is_home=i % 3 == 0,
            is_neutral=i % 3 != 0,
        ))
    return games


def _make_two_team_season(n_games=15):
    """Build a minimal two-team season suitable for full engine compute()."""
    games = []
    for i in range(n_games):
        day = 1 + i
        month = 1
        if day > 28:
            day -= 28
            month += 1
        date_str = f"2025-{month:02d}-{day:02d}"
        pts_a = 75.0 + (i % 5)
        pts_b = 70.0 + (i % 4)
        gid = f"game_{i}"
        # Team A side
        games.append(_make_game(
            game_id=gid, game_date=date_str,
            team_id="team_a", team_name="Team A",
            opponent_id="team_b",
            points=pts_a, opp_points=pts_b,
        ))
        # Team B side (mirror)
        games.append(_make_game(
            game_id=gid, game_date=date_str,
            team_id="team_b", team_name="Team B",
            opponent_id="team_a",
            points=pts_b, opp_points=pts_a,
        ))
    return games


# ---------------------------------------------------------------------------
# Tests: utility functions
# ---------------------------------------------------------------------------

class TestTeamId:
    def test_lowercase_and_strip(self):
        assert _team_id("Duke Blue Devils") == "duke_blue_devils"

    def test_special_chars(self):
        assert _team_id("Texas A&M") == "texas_a_m"

    def test_empty(self):
        assert _team_id("") == ""

    def test_numbers(self):
        assert _team_id("Team123") == "team123"

    def test_strips_leading_trailing_underscores(self):
        assert _team_id("  Spaces  ") == "spaces"


class TestToFloat:
    def test_int(self):
        assert _to_float(42) == 42.0

    def test_string_float(self):
        assert _to_float("3.14") == 3.14

    def test_none(self):
        assert _to_float(None) == 0.0

    def test_invalid_string(self):
        assert _to_float("abc") == 0.0

    def test_bool(self):
        assert _to_float(True) == 1.0


# ---------------------------------------------------------------------------
# Tests: GameRecord dataclass
# ---------------------------------------------------------------------------

class TestGameRecord:
    def test_defaults(self):
        g = GameRecord(
            game_id="g1", game_date="2025-01-01",
            team_id="a", team_name="A", opponent_id="b",
            points=80, opp_points=70, possessions=68,
        )
        assert g.fga == 0.0
        assert g.is_home is False
        assert g.is_neutral is True

    def test_custom_fields(self):
        g = _make_game(is_home=True, is_neutral=False, ast=20)
        assert g.is_home is True
        assert g.is_neutral is False
        assert g.ast == 20.0


# ---------------------------------------------------------------------------
# Tests: ProprietaryTeamMetrics dataclass
# ---------------------------------------------------------------------------

class TestProprietaryTeamMetrics:
    def test_defaults(self):
        m = ProprietaryTeamMetrics(team_id="t1", team_name="Team 1")
        assert m.adj_offensive_efficiency == 100.0
        assert m.adj_defensive_efficiency == 100.0
        assert m.elo_rating == 1500.0

    def test_to_dict(self):
        m = ProprietaryTeamMetrics(team_id="t1", team_name="T1")
        d = m.to_dict()
        assert d["team_id"] == "t1"
        assert "adj_efficiency_margin" in d
        assert isinstance(d, dict)

    def test_to_dict_preserves_all_fields(self):
        m = ProprietaryTeamMetrics(team_id="t1", team_name="T1", elo_rating=1600.0)
        d = m.to_dict()
        assert d["elo_rating"] == 1600.0


# ---------------------------------------------------------------------------
# Tests: ProprietaryMetricsEngine helpers
# ---------------------------------------------------------------------------

class TestRawEfficiency:
    def test_basic(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = _make_season_games(n_games=5)
        by_team = {"team_a": games}
        raw_off, raw_def, tempo, names = engine._raw_efficiency(by_team)
        assert "team_a" in raw_off
        assert raw_off["team_a"] > 0
        assert tempo["team_a"] > 0
        assert names["team_a"] != ""

    def test_zero_possessions(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = [_make_game(possessions=0.0, points=0, opp_points=0)]
        by_team = {"team_a": games}
        raw_off, raw_def, tempo, names = engine._raw_efficiency(by_team)
        # max(total_poss, 1.0) prevents division by zero
        assert raw_off["team_a"] == 0.0


class TestFourFactors:
    def test_basic(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = [_make_game()]
        ff = engine._four_factors(games)
        assert "effective_fg_pct" in ff
        assert "turnover_rate" in ff
        assert "offensive_reb_rate" in ff
        assert "free_throw_rate" in ff
        assert "opp_effective_fg_pct" in ff
        assert "opp_turnover_rate" in ff
        assert "defensive_reb_rate" in ff
        assert "opp_free_throw_rate" in ff

    def test_efg_calculation(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        # fgm=28, fg3m=8, fga=60 → eFG% = (28+0.5*8)/60 = 32/60 ≈ 0.533
        games = [_make_game(fgm=28, fg3m=8, fga=60)]
        ff = engine._four_factors(games)
        assert abs(ff["effective_fg_pct"] - (28 + 0.5 * 8) / 60) < 1e-6

    def test_zero_fga(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = [_make_game(fga=0, fgm=0, fg3m=0, fta=0, tov=0, orb=0, opp_drb=0)]
        ff = engine._four_factors(games)
        # Should not raise; max(fga, 1.0) protects division
        assert ff["effective_fg_pct"] == 0.0


class TestSupplementaryShooting:
    def test_basic(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = [_make_game()]
        ss = engine._supplementary_shooting(games)
        assert "two_pt_pct" in ss
        assert "three_pt_pct" in ss
        assert "three_pt_rate" in ss
        assert "ft_pct" in ss

    def test_three_pt_rate(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = [_make_game(fg3a=20, fga=60)]
        ss = engine._supplementary_shooting(games)
        assert abs(ss["three_pt_rate"] - 20.0 / 60.0) < 1e-6


class TestStrengthOfSchedule:
    def test_basic(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = [_make_game(opponent_id="opp1")]
        adj_off = {"team_a": 105.0, "opp1": 102.0}
        adj_def = {"team_a": 98.0, "opp1": 100.0}
        sos = engine._strength_of_schedule(games, adj_off, adj_def)
        assert "sos_adj_em" in sos
        assert "ncsos_adj_em" in sos

    def test_with_conference_map(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = [
            _make_game(opponent_id="opp1"),
            _make_game(game_id="g2", opponent_id="opp2"),
        ]
        adj_off = {"team_a": 105.0, "opp1": 102.0, "opp2": 100.0}
        adj_def = {"team_a": 98.0, "opp1": 100.0, "opp2": 101.0}
        conf_map = {"team_a": "ACC", "opp1": "ACC", "opp2": "Big12"}
        sos = engine._strength_of_schedule(games, adj_off, adj_def, conf_map)
        # opp2 is non-conference, ncsos should be based on opp2 only
        assert sos["ncsos_adj_em"] == pytest.approx(100.0 - 101.0, abs=0.01)

    def test_no_conference_fallback(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        # When no conference map, use neutral/away heuristic
        games = [_make_game(opponent_id="opp1", is_neutral=True, is_home=False)]
        adj_off = {"team_a": 105.0, "opp1": 102.0}
        adj_def = {"team_a": 98.0, "opp1": 100.0}
        sos = engine._strength_of_schedule(games, adj_off, adj_def, {})
        assert sos["ncsos_adj_em"] != 0.0


class TestCorrelatedGaussianLuck:
    def test_too_few_games(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = [_make_game()] * 5
        assert engine._correlated_gaussian_luck(games) == 0.0

    def test_with_enough_games(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = _make_season_games(n_games=20)
        luck = engine._correlated_gaussian_luck(games)
        assert isinstance(luck, float)
        assert -1.0 <= luck <= 1.0

    def test_zero_std(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        # All identical margins → std ≈ 0 → return 0.0
        games = [_make_game(points=80, opp_points=75, game_id=f"g{i}",
                            game_date=f"2025-01-{i+1:02d}") for i in range(15)]
        luck = engine._correlated_gaussian_luck(games)
        assert luck == 0.0


class TestPythagoreanWinPct:
    def test_even_matchup(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        assert engine._pythagorean_win_pct(100.0, 100.0) == pytest.approx(0.5, abs=1e-6)

    def test_strong_offense(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        wp = engine._pythagorean_win_pct(110.0, 100.0)
        assert 0.8 < wp < 0.9

    def test_weak_team(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        wp = engine._pythagorean_win_pct(95.0, 105.0)
        assert wp < 0.5


class TestBoxScoreXP:
    def test_default_factors(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        ff = {
            "effective_fg_pct": 0.50,
            "turnover_rate": 0.18,
            "offensive_reb_rate": 0.30,
            "free_throw_rate": 0.30,
        }
        xp = engine._box_score_xp(ff)
        assert 0.5 <= xp <= 1.8

    def test_clipping(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        # Extreme values should be clipped to [0.5, 1.8]
        ff = {"effective_fg_pct": 1.0, "turnover_rate": 0.0,
              "offensive_reb_rate": 1.0, "free_throw_rate": 1.0}
        xp = engine._box_score_xp(ff, ft_pct=1.0)
        assert xp == 1.8

    def test_low_values(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        ff = {"effective_fg_pct": 0.0, "turnover_rate": 1.0,
              "offensive_reb_rate": 0.0, "free_throw_rate": 0.0}
        xp = engine._box_score_xp(ff, ft_pct=0.0)
        assert xp == 0.5


class TestShotDistributionScore:
    def test_basic(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = [_make_game()]
        score = engine._shot_distribution_score(games)
        assert isinstance(score, float)

    def test_zero_fga(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = [_make_game(fga=0, fg3a=0, fta=0)]
        assert engine._shot_distribution_score(games) == 0.0


class TestThreePointVariance:
    def test_too_few_qualifying(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        # fg3a < 5 means games are excluded → return prior
        games = [_make_game(fg3a=3, fg3m=1, game_id=f"g{i}",
                            game_date=f"2025-01-{i+1:02d}") for i in range(10)]
        var = engine._three_point_variance(games)
        assert var == pytest.approx(0.095, abs=1e-6)

    def test_with_enough_games(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = []
        for i in range(10):
            games.append(_make_game(
                fg3a=20, fg3m=5 + i % 5,
                game_id=f"g{i}", game_date=f"2025-01-{i+1:02d}",
            ))
        var = engine._three_point_variance(games)
        assert var > 0


class TestMomentum:
    def test_too_few_games(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = [_make_game(game_id=f"g{i}", game_date=f"2025-01-{i+1:02d}",
                            team_id="team_a", opponent_id="team_b")
                 for i in range(5)]
        adj_off = {"team_a": 105.0, "team_b": 100.0}
        adj_def = {"team_a": 98.0, "team_b": 100.0}
        mom, recent = engine._momentum(games, adj_off, adj_def)
        assert mom == 0.0

    def test_with_enough_games(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = _make_season_games(n_games=20)
        adj_off = {"team_a": 105.0}
        adj_def = {"team_a": 98.0}
        for i in range(5):
            adj_off[f"team_b_{i}"] = 100.0
            adj_def[f"team_b_{i}"] = 100.0
        mom, recent = engine._momentum(games, adj_off, adj_def)
        assert isinstance(mom, float)
        assert isinstance(recent, float)


class TestPaceAdjustedVariance:
    def test_single_game(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = [_make_game()]
        var = engine._pace_adjusted_variance(games)
        assert var == 11.0  # PACE_VAR_PRIOR_STD returned for n < 2

    def test_multiple_games(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = [_make_game(points=70+i*3, opp_points=68+i*2, possessions=65+i,
                            game_id=f"g{i}", game_date=f"2025-01-{i+1:02d}")
                 for i in range(10)]
        var = engine._pace_adjusted_variance(games)
        assert var > 0


class TestConsistency:
    def test_too_few_games(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = [_make_game()] * 3
        assert engine._consistency(games) == 0.5

    def test_consistent_team(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        # All same margin → low stdev → high consistency
        games = [_make_game(points=80, opp_points=75, game_id=f"g{i}",
                            game_date=f"2025-01-{i+1:02d}") for i in range(10)]
        c = engine._consistency(games)
        # With zero sample variance, shrunk variance is dominated by prior
        assert c > 0.0

    def test_inconsistent_team(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = []
        for i in range(10):
            margin = 30 if i % 2 == 0 else -20
            games.append(_make_game(points=70+margin, opp_points=70,
                                    game_id=f"g{i}", game_date=f"2025-01-{i+1:02d}"))
        c = engine._consistency(games)
        assert 0 < c < 0.5


class TestSOSAdjustedConsistency:
    def test_too_few_games(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = [_make_game()] * 3
        assert engine._sos_adjusted_consistency(games, {}, {}) == 0.5

    def test_with_enough_games(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = _make_season_games(n_games=10)
        adj_off = {"team_a": 105.0}
        adj_def = {"team_a": 98.0}
        for i in range(5):
            adj_off[f"team_b_{i}"] = 100.0
            adj_def[f"team_b_{i}"] = 100.0
        c = engine._sos_adjusted_consistency(games, adj_off, adj_def)
        assert 0 < c < 1


class TestExtendedBoxScoreMetrics:
    def test_basic(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = [_make_game()]
        ext = engine._extended_box_score_metrics(games)
        assert "free_throw_pct" in ext
        assert "assist_to_turnover_ratio" in ext
        assert "steal_rate" in ext
        assert "block_rate" in ext
        assert "defensive_disruption_rate" in ext

    def test_ft_pct_calculation(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = [_make_game(ftm=9, fta=12)]
        ext = engine._extended_box_score_metrics(games)
        assert ext["free_throw_pct"] == pytest.approx(9.0 / 12.0, abs=1e-6)


class TestOpponentShotSelection:
    def test_basic(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = [_make_game()]
        opp = engine._opponent_shot_selection(games)
        assert "opp_two_pt_pct_allowed" in opp
        assert "opp_three_pt_attempt_rate" in opp


class TestFoulRate:
    def test_basic(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = [_make_game(pf=16, possessions=68)]
        rate = engine._foul_rate(games)
        assert rate == pytest.approx(16.0 / 68.0, abs=1e-6)

    def test_zero_possessions(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        # possessions=0 → max(0, 1.0) = 1.0 per game, so pf=0/1.0=0.0
        games = [_make_game(pf=0, possessions=0)]
        rate = engine._foul_rate(games)
        assert rate == 0.0


class TestLog5WinProb:
    def test_equal_teams(self):
        p = ProprietaryMetricsEngine._log5_win_prob(10.0, 10.0)
        assert p == pytest.approx(0.5, abs=1e-6)

    def test_strong_favorite(self):
        p = ProprietaryMetricsEngine._log5_win_prob(20.0, 0.0)
        assert p > 0.9

    def test_underdog(self):
        p = ProprietaryMetricsEngine._log5_win_prob(0.0, 20.0)
        assert p < 0.1

    def test_clipping(self):
        # Extreme values should be clipped to [-40, 40]
        p = ProprietaryMetricsEngine._log5_win_prob(100.0, 0.0)
        assert 0 < p <= 1.0


class TestClassifyQuadrantByEM:
    def test_neutral_q1(self):
        q = ProprietaryMetricsEngine._classify_quadrant_by_em(12.0, is_home=False, is_neutral=True)
        assert q == 1

    def test_neutral_q2(self):
        q = ProprietaryMetricsEngine._classify_quadrant_by_em(5.0, is_home=False, is_neutral=True)
        assert q == 2

    def test_neutral_q3(self):
        q = ProprietaryMetricsEngine._classify_quadrant_by_em(-5.0, is_home=False, is_neutral=True)
        assert q == 3

    def test_neutral_q4(self):
        q = ProprietaryMetricsEngine._classify_quadrant_by_em(-10.0, is_home=False, is_neutral=True)
        assert q == 4

    def test_home_q1(self):
        q = ProprietaryMetricsEngine._classify_quadrant_by_em(16.0, is_home=True, is_neutral=False)
        assert q == 1

    def test_home_q2(self):
        q = ProprietaryMetricsEngine._classify_quadrant_by_em(10.0, is_home=True, is_neutral=False)
        assert q == 2

    def test_home_q3(self):
        q = ProprietaryMetricsEngine._classify_quadrant_by_em(0.0, is_home=True, is_neutral=False)
        assert q == 3

    def test_home_q4(self):
        q = ProprietaryMetricsEngine._classify_quadrant_by_em(-5.0, is_home=True, is_neutral=False)
        assert q == 4

    def test_away_q1(self):
        q = ProprietaryMetricsEngine._classify_quadrant_by_em(6.0, is_home=False, is_neutral=False)
        assert q == 1

    def test_away_q2(self):
        q = ProprietaryMetricsEngine._classify_quadrant_by_em(0.0, is_home=False, is_neutral=False)
        assert q == 2

    def test_away_q3(self):
        q = ProprietaryMetricsEngine._classify_quadrant_by_em(-8.0, is_home=False, is_neutral=False)
        assert q == 3

    def test_away_q4(self):
        q = ProprietaryMetricsEngine._classify_quadrant_by_em(-15.0, is_home=False, is_neutral=False)
        assert q == 4


class TestTrueShootingPct:
    def test_basic(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = [_make_game(points=80, fga=60, fta=15, opp_points=70, opp_fga=58, opp_fta=12)]
        ts, opp_ts = engine._true_shooting_pct(games)
        expected_tsa = 2.0 * (60 + 0.44 * 15)
        assert ts == pytest.approx(80.0 / expected_tsa, abs=1e-6)


class TestNeutralSiteRecord:
    def test_no_neutral_games(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = [_make_game(is_neutral=False, is_home=True)]
        pct, n = engine._neutral_site_record(games)
        assert pct == pytest.approx(0.5, abs=1e-6)  # Beta prior
        assert n == 0

    def test_with_neutral_games(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = [_make_game(is_neutral=True, points=80, opp_points=70, game_id=f"g{i}",
                            game_date=f"2025-01-{i+1:02d}")
                 for i in range(6)]
        pct, n = engine._neutral_site_record(games)
        assert n == 6
        # All wins: (6 + 2) / (6 + 4) = 0.8
        assert pct == pytest.approx(0.8, abs=1e-6)


class TestHomeAwaySplits:
    def test_basic(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = []
        for i in range(10):
            games.append(_make_game(
                game_id=f"g{i}", game_date=f"2025-01-{i+1:02d}",
                opponent_id=f"opp_{i}",
                is_home=i < 5, is_neutral=False,
                points=80 if i < 5 else 68,
                opp_points=70,
            ))
        adj_off = {f"opp_{i}": 100.0 for i in range(10)}
        adj_def = {f"opp_{i}": 100.0 for i in range(10)}
        home_em, away_em, dep = engine._home_away_splits(games, adj_off, adj_def)
        assert isinstance(home_em, float)
        assert isinstance(away_em, float)
        assert isinstance(dep, float)
        # Home should be better than away (home wins by 10, away by -2)
        assert home_em > away_em


class TestMomentum5g:
    def test_too_few_games(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = [_make_game(game_id=f"g{i}", game_date=f"2025-01-{i+1:02d}")
                 for i in range(5)]
        adj_off = {"team_a": 105.0, "team_b": 100.0}
        adj_def = {"team_a": 98.0, "team_b": 100.0}
        assert engine._momentum_5g(games, adj_off, adj_def) == 0.0

    def test_enough_games(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = _make_season_games(n_games=15)
        adj_off = {"team_a": 105.0}
        adj_def = {"team_a": 98.0}
        for i in range(5):
            adj_off[f"team_b_{i}"] = 100.0
            adj_def[f"team_b_{i}"] = 100.0
        result = engine._momentum_5g(games, adj_off, adj_def)
        assert isinstance(result, float)


class TestPaceVariance:
    def test_too_few_games(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = [_make_game()] * 3
        assert engine._pace_variance(games) == 0.0

    def test_with_enough_games(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = [_make_game(possessions=60+i*3, game_id=f"g{i}",
                            game_date=f"2025-01-{i+1:02d}")
                 for i in range(10)]
        var = engine._pace_variance(games)
        assert var > 0

    def test_low_possession_filtered(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        # Games with possessions <= 20 are filtered out
        games = [_make_game(possessions=10, game_id=f"g{i}",
                            game_date=f"2025-01-{i+1:02d}")
                 for i in range(10)]
        assert engine._pace_variance(games) == 0.0


class TestPoissonBinomial:
    def test_cdf_empty(self):
        assert ProprietaryMetricsEngine._poisson_binomial_cdf(0, []) == 1.0

    def test_cdf_single_game_certain(self):
        # P(X <= 1) with p=0.99 should be ~1.0
        cdf = ProprietaryMetricsEngine._poisson_binomial_cdf(1, [0.99])
        assert cdf == pytest.approx(1.0, abs=1e-4)

    def test_cdf_single_game_zero_wins(self):
        # P(X <= 0) with p=0.5 should be 0.5
        cdf = ProprietaryMetricsEngine._poisson_binomial_cdf(0, [0.5])
        assert cdf == pytest.approx(0.5, abs=1e-4)

    def test_cdf_multiple_games(self):
        probs = [0.8, 0.6, 0.7]
        cdf = ProprietaryMetricsEngine._poisson_binomial_cdf(3, probs)
        assert cdf == pytest.approx(1.0, abs=1e-4)

    def test_mean(self):
        probs = [0.8, 0.6, 0.7]
        mean = ProprietaryMetricsEngine._poisson_binomial_mean(probs)
        assert mean == pytest.approx(2.1, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests: compute() full pipeline (integration-level, no I/O)
# ---------------------------------------------------------------------------

class TestComputeFullPipeline:
    def test_empty_records(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        assert engine.compute([], cutoff_date=None) == {}

    def test_cutoff_date_required(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=True)
        with pytest.raises(ValueError, match="cutoff_date is required"):
            engine.compute([_make_game()])

    def test_cutoff_filters_games(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = _make_two_team_season(n_games=15)
        results = engine.compute(games, cutoff_date="2025-01-10")
        # Only games on or before Jan 10 are included
        for tid, m in results.items():
            assert m.wins + m.losses <= 10

    def test_cutoff_filters_all(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = _make_two_team_season(n_games=5)
        # Cutoff before any games
        results = engine.compute(games, cutoff_date="2024-12-01")
        assert results == {}

    def test_basic_two_team(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = _make_two_team_season(n_games=15)
        results = engine.compute(games)
        assert "team_a" in results
        assert "team_b" in results
        assert results["team_a"].adj_offensive_efficiency > 0
        assert results["team_b"].adj_defensive_efficiency > 0
        assert results["team_a"].wins + results["team_a"].losses == 15
        assert results["team_a"].elo_rating != 1500.0  # Elo should be updated

    def test_conference_map(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = _make_two_team_season(n_games=15)
        conf_map = {"team_a": "ACC", "team_b": "ACC"}
        results = engine.compute(games, conference_map=conf_map)
        assert results["team_a"].conference == "ACC"

    def test_wab_computed(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = _make_two_team_season(n_games=15)
        results = engine.compute(games)
        # WAB should be a non-default value for at least one team
        assert results["team_a"].wab != 0.0 or results["team_b"].wab != 0.0

    def test_sor_computed(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = _make_two_team_season(n_games=15)
        results = engine.compute(games)
        # SOR should be between 0 and 1
        for tid, m in results.items():
            assert 0.0 <= m.sor <= 1.0

    def test_end_of_season_elo_stored(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = _make_two_team_season(n_games=15)
        engine.compute(games)
        assert engine._end_of_season_elo is not None
        assert "team_a" in engine._end_of_season_elo


class TestComputeRestDays:
    def test_with_reference_date(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = _make_two_team_season(n_games=5)
        results = engine.compute(games)
        # rest_days should be set for both teams
        assert results["team_a"].rest_days >= 0

    def test_no_reference_date(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        results = {
            "team_a": ProprietaryTeamMetrics(team_id="team_a", team_name="Team A")
        }
        games = _make_season_games(n_games=5, team_id="team_a")
        by_team = {"team_a": games}
        engine._compute_rest_days(results, by_team, reference_date=None)
        assert results["team_a"].rest_days >= 0

    def test_single_game_no_ref(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        results = {
            "team_a": ProprietaryTeamMetrics(team_id="team_a", team_name="Team A")
        }
        games = [_make_game(team_id="team_a")]
        by_team = {"team_a": games}
        engine._compute_rest_days(results, by_team, reference_date=None)
        assert results["team_a"].rest_days == 5.0  # D1 average

    def test_invalid_date(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        results = {
            "team_a": ProprietaryTeamMetrics(team_id="team_a", team_name="Team A")
        }
        games = [_make_game(team_id="team_a", game_date="invalid-date")]
        by_team = {"team_a": games}
        engine._compute_rest_days(results, by_team, reference_date="2025-03-15")
        assert results["team_a"].rest_days == 5.0


class TestComputeConferenceStrength:
    def test_with_conference_map(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        results = {
            "t1": ProprietaryTeamMetrics(team_id="t1", team_name="T1", adj_efficiency_margin=10.0),
            "t2": ProprietaryTeamMetrics(team_id="t2", team_name="T2", adj_efficiency_margin=5.0),
        }
        by_team = {"t1": [], "t2": []}
        conf_map = {"t1": "ACC", "t2": "ACC"}
        engine._compute_conference_strength(results, by_team, conf_map)
        # Average AdjEM of ACC: (10+5)/2 = 7.5
        assert results["t1"].conference_adj_em == pytest.approx(7.5, abs=0.01)

    def test_without_conference_map(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        results = {
            "t1": ProprietaryTeamMetrics(team_id="t1", team_name="T1",
                                         adj_efficiency_margin=10.0, sos_adj_em=3.0),
            "t2": ProprietaryTeamMetrics(team_id="t2", team_name="T2",
                                         adj_efficiency_margin=5.0, sos_adj_em=2.0),
        }
        # t1 plays t2 twice (conference-like)
        games_t1 = [
            _make_game(game_id="g1", team_id="t1", opponent_id="t2"),
            _make_game(game_id="g2", team_id="t1", opponent_id="t2"),
        ]
        by_team = {"t1": games_t1, "t2": []}
        engine._compute_conference_strength(results, by_team, {})
        # t2 is played 2x → conference peer → conference_adj_em = t2.adj_em = 5.0
        assert results["t1"].conference_adj_em == pytest.approx(5.0, abs=0.01)

    def test_no_games_fallback(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        results = {
            "t1": ProprietaryTeamMetrics(team_id="t1", team_name="T1", sos_adj_em=3.0),
        }
        by_team = {"t1": []}
        engine._compute_conference_strength(results, by_team, {})
        assert results["t1"].conference_adj_em == 3.0  # SOS fallback


class TestH2HAndCommonOpponent:
    def test_h2h_no_by_team(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        assert engine.compute_h2h_record("a", "b") == 0.5

    def test_h2h_no_games(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        engine._by_team = {"a": [], "b": []}
        assert engine.compute_h2h_record("a", "b") == 0.5

    def test_h2h_with_games(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = [
            _make_game(team_id="a", opponent_id="b", points=80, opp_points=70, game_id="g1"),
            _make_game(team_id="a", opponent_id="b", points=60, opp_points=70, game_id="g2"),
        ]
        engine._by_team = {"a": games, "b": []}
        h2h = engine.compute_h2h_record("a", "b")
        # 2 games, 1 win → weight=0.5, raw=0.5 → 0.5*0.5 + 0.5*0.5 = 0.5
        assert h2h == pytest.approx(0.5, abs=0.01)

    def test_h2h_four_games(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = [
            _make_game(team_id="a", opponent_id="b", points=80, opp_points=70, game_id=f"g{i}")
            for i in range(4)
        ]
        engine._by_team = {"a": games}
        h2h = engine.compute_h2h_record("a", "b")
        # 4 games, all wins → weight=1.0, raw=1.0 → 1.0
        assert h2h == pytest.approx(1.0, abs=0.01)

    def test_common_opponent_no_by_team(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        assert engine.compute_common_opponent_margin("a", "b") == 0.0

    def test_common_opponent_no_common(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        engine._by_team = {
            "a": [_make_game(team_id="a", opponent_id="c")],
            "b": [_make_game(team_id="b", opponent_id="d")],
        }
        assert engine.compute_common_opponent_margin("a", "b") == 0.0

    def test_common_opponent_with_common(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        engine._by_team = {
            "a": [_make_game(team_id="a", opponent_id="c", points=80, opp_points=70)],
            "b": [_make_game(team_id="b", opponent_id="c", points=70, opp_points=70)],
        }
        margin = engine.compute_common_opponent_margin("a", "b")
        # team_a vs c: +10, team_b vs c: 0 → diff = 10, /20 = 0.5
        assert margin == pytest.approx(0.5, abs=0.01)


class TestIterativeSOSAdjust:
    def test_basic(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = _make_two_team_season(n_games=10)
        by_team = defaultdict(list)
        for g in games:
            by_team[g.team_id].append(g)
        raw_off = {"team_a": 110.0, "team_b": 103.0}
        raw_def = {"team_a": 95.0, "team_b": 103.0}
        adj_off, adj_def = engine._iterative_sos_adjust(by_team, raw_off, raw_def)
        assert "team_a" in adj_off
        assert "team_b" in adj_def

    def test_warm_start(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = _make_two_team_season(n_games=10)
        by_team = defaultdict(list)
        for g in games:
            by_team[g.team_id].append(g)
        raw_off = {"team_a": 110.0, "team_b": 103.0}
        raw_def = {"team_a": 95.0, "team_b": 103.0}
        adj_off, adj_def = engine._iterative_sos_adjust(
            by_team, raw_off, raw_def,
            initial_adj_off={"team_a": 108.0, "team_b": 102.0},
            initial_adj_def={"team_a": 96.0, "team_b": 104.0},
            n_iterations=5,
        )
        assert "team_a" in adj_off

    def test_n_iterations_override(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = _make_two_team_season(n_games=5)
        by_team = defaultdict(list)
        for g in games:
            by_team[g.team_id].append(g)
        raw_off = {"team_a": 110.0, "team_b": 103.0}
        raw_def = {"team_a": 95.0, "team_b": 103.0}
        # Should run without error with 1 iteration
        adj_off, adj_def = engine._iterative_sos_adjust(
            by_team, raw_off, raw_def, n_iterations=1,
        )
        assert len(adj_off) == 2


class TestComputeEloRatings:
    def test_with_prior(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = _make_two_team_season(n_games=5)
        results = {
            "team_a": ProprietaryTeamMetrics(team_id="team_a", team_name="A"),
            "team_b": ProprietaryTeamMetrics(team_id="team_b", team_name="B"),
        }
        by_team = defaultdict(list)
        for g in games:
            by_team[g.team_id].append(g)
        prior = {"team_a": 1600.0, "team_b": 1400.0}
        end_elo = engine._compute_elo_ratings(results, by_team, prior_elo=prior)
        # Team A started higher and mostly wins → should still be higher
        assert end_elo["team_a"] > end_elo["team_b"]

    def test_without_prior(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = _make_two_team_season(n_games=5)
        results = {
            "team_a": ProprietaryTeamMetrics(team_id="team_a", team_name="A"),
            "team_b": ProprietaryTeamMetrics(team_id="team_b", team_name="B"),
        }
        by_team = defaultdict(list)
        for g in games:
            by_team[g.team_id].append(g)
        end_elo = engine._compute_elo_ratings(results, by_team)
        assert "team_a" in end_elo
        assert "team_b" in end_elo


class TestComputeEliteSosAndQuadrants:
    def test_basic(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        results = {
            "team_a": ProprietaryTeamMetrics(team_id="team_a", team_name="A"),
        }
        games = [
            _make_game(opponent_id="elite_opp", points=80, opp_points=70,
                       is_neutral=True),
        ]
        by_team = {"team_a": games}
        adj_off = {"team_a": 110, "elite_opp": 115}
        adj_def = {"team_a": 95, "elite_opp": 95}
        engine._compute_elite_sos_and_quadrants(results, by_team, adj_off, adj_def)
        # elite_opp has AdjEM = 20 >= 15 → counted in elite SOS
        assert results["team_a"].elite_sos == pytest.approx(20.0, abs=0.01)
        assert results["team_a"].q1_wins == 1


# ---------------------------------------------------------------------------
# Tests: team_games_to_game_records
# ---------------------------------------------------------------------------

class TestTeamGamesToGameRecords:
    def test_basic_conversion(self):
        rows = [
            {"game_id": "g1", "team_id": "a", "opponent_id": "b",
             "team_score": 80, "opponent_score": 70, "possessions": 68,
             "fgm": 30, "fga": 60, "fg3m": 8, "fg3a": 20,
             "fta": 10, "turnovers": 12, "orb": 8, "drb": 22,
             "date": "2025-01-15"},
            {"game_id": "g1", "team_id": "b", "opponent_id": "a",
             "team_score": 70, "opponent_score": 80, "possessions": 68,
             "fgm": 26, "fga": 58, "fg3m": 6, "fg3a": 18,
             "fta": 8, "turnovers": 14, "orb": 6, "drb": 20,
             "date": "2025-01-15"},
        ]
        records = team_games_to_game_records(rows, 2025)
        assert len(records) == 2
        r = records[0]
        assert r.points == 80
        assert r.opp_fga == 58  # From mirror row

    def test_skip_forfeits(self):
        rows = [
            {"game_id": "g1", "team_id": "a", "opponent_id": "b",
             "team_score": 0, "opponent_score": 0, "date": "2025-01-01"},
        ]
        records = team_games_to_game_records(rows, 2025)
        assert len(records) == 0

    def test_missing_fields(self):
        rows = [
            {"game_id": "g1", "team_id": "a", "opponent_id": "b",
             "team_score": 80, "opponent_score": 70, "date": "2025-01-01"},
        ]
        records = team_games_to_game_records(rows, 2025)
        assert len(records) == 1
        # Possessions derived from score when missing
        assert records[0].possessions > 0

    def test_ftm_derivation(self):
        rows = [
            {"game_id": "g1", "team_id": "a", "opponent_id": "b",
             "team_score": 80, "opponent_score": 70,
             "fgm": 28, "fg3m": 8, "fga": 60, "fg3a": 20,
             "fta": 10, "possessions": 68, "date": "2025-01-01"},
        ]
        records = team_games_to_game_records(rows, 2025)
        # ftm = points - 2*fgm - fg3m = 80 - 56 - 8 = 16
        assert records[0].ftm == 16.0

    def test_deduplication(self):
        rows = [
            {"game_id": "g1", "team_id": "a", "opponent_id": "b",
             "team_score": 80, "opponent_score": 70, "date": "2025-01-01"},
            {"game_id": "g1", "team_id": "a", "opponent_id": "b",
             "team_score": 80, "opponent_score": 70, "date": "2025-01-01"},
        ]
        records = team_games_to_game_records(rows, 2025)
        assert len(records) == 1

    def test_possession_estimation_from_fga(self):
        rows = [
            {"game_id": "g1", "team_id": "a", "opponent_id": "b",
             "team_score": 80, "opponent_score": 70,
             "fga": 60, "orb": 10, "turnovers": 12, "fta": 15,
             "fgm": 28, "fg3m": 8, "fg3a": 20,
             "date": "2025-01-01"},
        ]
        records = team_games_to_game_records(rows, 2025)
        # poss = fga - orb + tov + 0.475 * fta = 60 - 10 + 12 + 7.125 = 69.125
        assert records[0].possessions == pytest.approx(69.125, abs=0.01)


# ---------------------------------------------------------------------------
# Tests: torvik_to_game_records
# ---------------------------------------------------------------------------

class TestTorvikToGameRecords:
    @patch("src.data.features.proprietary_metrics._load_cbbpy_team_map", return_value={})
    def test_basic(self, mock_map):
        torvik_teams = [
            {"team_id": "duke", "name": "Duke"},
            {"team_id": "unc", "name": "North Carolina"},
        ]
        games = [
            {"game_id": "g1", "team_id": "duke", "opponent_id": "unc",
             "team_score": 80, "opponent_score": 70,
             "game_date": "2025-01-15", "team_name": "Duke"},
        ]
        records = torvik_to_game_records(torvik_teams, games, 2025)
        assert len(records) == 1
        assert records[0].team_id == "duke"

    @patch("src.data.features.proprietary_metrics._load_cbbpy_team_map", return_value={})
    def test_skip_non_dict(self, mock_map):
        torvik_teams = [{"team_id": "duke", "name": "Duke"}]
        games = ["not_a_dict", {"game_id": "g1", "team_id": "duke",
                                "opponent_id": "unc", "team_score": 80,
                                "opponent_score": 70, "game_date": "2025-01-15",
                                "team_name": "Duke"}]
        records = torvik_to_game_records(torvik_teams, games, 2025)
        assert len(records) == 1

    @patch("src.data.features.proprietary_metrics._load_cbbpy_team_map", return_value={})
    def test_possession_estimation(self, mock_map):
        torvik_teams = [{"team_id": "a", "name": "A"}, {"team_id": "b", "name": "B"}]
        games = [
            {"game_id": "g1", "team_id": "a", "opponent_id": "b",
             "team_score": 80, "opponent_score": 70,
             "fga": 60, "orb": 10, "turnovers": 12, "fta": 15,
             "opp_fga": 55, "opp_orb": 8, "opp_tov": 14, "opp_fta": 10,
             "game_date": "2025-01-15", "team_name": "A"},
        ]
        records = torvik_to_game_records(torvik_teams, games, 2025)
        assert records[0].possessions > 0


# ---------------------------------------------------------------------------
# Tests: IncrementalMetricsEngine
# ---------------------------------------------------------------------------

class TestIncrementalMetricsEngine:
    def _make_engine(self, n_games=15):
        games = _make_two_team_season(n_games=n_games)
        return IncrementalMetricsEngine(games)

    def test_init(self):
        engine = self._make_engine()
        assert len(engine._unique_dates) > 0
        assert len(engine._elo_snapshots) > 0

    def test_compute_as_of_cached(self):
        # Need > 50 records before date: 28 games * 2 sides = 56
        games = _make_two_team_season(n_games=28)
        engine = IncrementalMetricsEngine(games)
        r1 = engine.compute_as_of("2025-03-01")
        r2 = engine.compute_as_of("2025-03-01")
        assert r1 is r2  # Same object from cache

    def test_compute_as_of_too_few_games(self):
        games = _make_two_team_season(n_games=3)
        engine = IncrementalMetricsEngine(games)
        result = engine.compute_as_of("2025-01-10")
        # Less than 50 records before that date
        assert result == {}

    def test_compute_as_of_returns_metrics(self):
        engine = self._make_engine(n_games=15)
        results = engine.compute_as_of("2025-02-15")
        if results:  # May be empty if not enough games
            for tid, m in results.items():
                assert isinstance(m, ProprietaryTeamMetrics)

    def test_get_end_of_season_elo(self):
        engine = self._make_engine()
        elo = engine.get_end_of_season_elo()
        assert isinstance(elo, dict)
        assert len(elo) > 0

    def test_games_played_before(self):
        engine = self._make_engine(n_games=10)
        count = engine.games_played_before("team_a", "2025-01-05")
        assert isinstance(count, int)
        assert count >= 0

    def test_games_played_before_unknown_team(self):
        engine = self._make_engine()
        assert engine.games_played_before("unknown_team", "2025-01-15") == 0

    def test_with_prior_elo(self):
        games = _make_two_team_season(n_games=10)
        prior = {"team_a": 1600.0, "team_b": 1400.0}
        engine = IncrementalMetricsEngine(games, prior_elo=prior)
        elo = engine.get_end_of_season_elo()
        # Priors should influence final Elo
        assert elo["team_a"] != elo["team_b"]

    def test_with_conference_map(self):
        games = _make_two_team_season(n_games=15)
        conf = {"team_a": "ACC", "team_b": "ACC"}
        engine = IncrementalMetricsEngine(games, conference_map=conf)
        assert engine._conference_map == conf


class TestMetricsToTeamVector:
    def test_basic(self):
        m = ProprietaryTeamMetrics(
            team_id="t1", team_name="T1",
            adj_offensive_efficiency=110.0,
            adj_defensive_efficiency=95.0,
            adj_tempo=68.0,
        )
        try:
            v = IncrementalMetricsEngine.metrics_to_team_vector(m, seed=3)
            assert isinstance(v, np.ndarray)
            assert v[0] == 110.0
            assert v[1] == 95.0
            assert v[2] == 68.0
            # seed > 0 → non-zero seed strength (index 73)
            assert v[73] > 0
        except (ImportError, ModuleNotFoundError):
            pytest.skip("feature_engineering or tournament_features not importable")

    def test_zero_seed(self):
        m = ProprietaryTeamMetrics(team_id="t1", team_name="T1")
        try:
            v = IncrementalMetricsEngine.metrics_to_team_vector(m, seed=0)
            assert v[73] == 0.0
        except (ImportError, ModuleNotFoundError):
            pytest.skip("feature_engineering not importable")

    def test_nan_inf_guard(self):
        m = ProprietaryTeamMetrics(team_id="t1", team_name="T1",
                                   adj_offensive_efficiency=float('nan'))
        try:
            v = IncrementalMetricsEngine.metrics_to_team_vector(m)
            # Inf values must never appear in the vector
            assert not np.isinf(v).any()
            # NaN is expected at indices 71-72 (external ratings) and 74-85
            # (massey features) when no external data is provided.
            # Non-rating features (indices 0-70, 73) should not have
            # unexpected NaN beyond the one we injected (adj_off at idx 0).
            non_rating_indices = list(range(0, 71)) + [73]
            nan_in_non_rating = np.isnan(v[non_rating_indices]).sum()
            # Only the injected NaN at index 0 (adj_offensive_efficiency)
            assert nan_in_non_rating <= 1, (
                f"Found {nan_in_non_rating} NaN in non-rating features, "
                f"expected at most 1 (the injected adj_offensive_efficiency)"
            )
        except (ImportError, ModuleNotFoundError):
            pytest.skip("feature_engineering not importable")


class TestBuildMatchupVector:
    def test_basic(self):
        try:
            from src.data.features.feature_engineering import TEAM_FEATURE_DIM
            v1 = np.ones(TEAM_FEATURE_DIM) * 110
            v2 = np.ones(TEAM_FEATURE_DIM) * 100
            mv = IncrementalMetricsEngine.build_matchup_vector(v1, v2, seed1=1, seed2=16)
            assert isinstance(mv, np.ndarray)
            assert len(mv) > TEAM_FEATURE_DIM
        except (ImportError, ModuleNotFoundError):
            pytest.skip("feature_engineering not importable")

    def test_no_seeds(self):
        try:
            from src.data.features.feature_engineering import TEAM_FEATURE_DIM
            v1 = np.ones(TEAM_FEATURE_DIM) * 105
            v2 = np.ones(TEAM_FEATURE_DIM) * 100
            mv = IncrementalMetricsEngine.build_matchup_vector(v1, v2)
            assert isinstance(mv, np.ndarray)
        except (ImportError, ModuleNotFoundError):
            pytest.skip("feature_engineering not importable")


# ---------------------------------------------------------------------------
# Tests: Elo cross-season carryover (D2)
# ---------------------------------------------------------------------------

class TestEloCrossSeason:
    def test_elo_prior_sets_initial(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        engine._elo_prior = {"team_a": 1700.0, "team_b": 1300.0}
        games = _make_two_team_season(n_games=10)
        results = engine.compute(games)
        # With prior, team_a should start higher
        assert engine._end_of_season_elo is not None


class TestComputeWAB:
    def test_wab_home_away(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        # Create games with home/away to test D4 HCA correction
        games_a = [
            _make_game(game_id="g1", team_id="team_a", opponent_id="team_b",
                       points=80, opp_points=70, is_home=True, is_neutral=False),
            _make_game(game_id="g2", team_id="team_a", opponent_id="team_b",
                       points=65, opp_points=70, is_home=False, is_neutral=False),
        ]
        results = {
            "team_a": ProprietaryTeamMetrics(team_id="team_a", team_name="A",
                                             adj_efficiency_margin=10.0),
            "team_b": ProprietaryTeamMetrics(team_id="team_b", team_name="B",
                                             adj_efficiency_margin=0.0),
        }
        by_team = {"team_a": games_a}
        engine._compute_wab(results, by_team)
        assert results["team_a"].wab != 0.0


class TestValidateFourFactorsWeights:
    def test_insufficient_data(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        game_records = {"t1": [_make_game()]}
        result = engine.validate_four_factors_weights(game_records)
        assert result["recommendation"] == "insufficient_data"
        assert result["n_matchups"] < 50

    def test_with_enough_data(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        # Build enough game pairs for logistic regression
        games_a = []
        games_b = []
        for i in range(60):
            gid = f"g{i}"
            ga = _make_game(game_id=gid, team_id="a", opponent_id="b",
                            points=70+i%20, opp_points=65+i%15,
                            fgm=25+i%5, fga=55+i%10, fg3m=5+i%4, fg3a=18+i%5,
                            fta=10+i%5, tov=10+i%3, orb=8+i%3, opp_drb=20+i%4,
                            opp_fgm=24+i%4, opp_fga=56+i%8, opp_fg3m=5+i%3,
                            opp_fg3a=17+i%4, opp_fta=9+i%3, opp_tov=11+i%3,
                            opp_orb=7+i%3)
            gb = _make_game(game_id=gid, team_id="b", opponent_id="a",
                            points=65+i%15, opp_points=70+i%20,
                            fgm=24+i%4, fga=56+i%8, fg3m=5+i%3, fg3a=17+i%4,
                            fta=9+i%3, tov=11+i%3, orb=7+i%3, opp_drb=22+i%4,
                            opp_fgm=25+i%5, opp_fga=55+i%10, opp_fg3m=5+i%4,
                            opp_fg3a=18+i%5, opp_fta=10+i%5, opp_tov=10+i%3,
                            opp_orb=8+i%3)
            games_a.append(ga)
            games_b.append(gb)
        game_records = {"a": games_a, "b": games_b}
        try:
            result = engine.validate_four_factors_weights(game_records, n_bootstrap=5)
            assert result["n_matchups"] >= 50
            assert "fitted_weights" in result
            assert "sensitivity_brier_deltas" in result
            assert result["recommendation"] in ("keep_priors", "consider_update")
        except ImportError:
            pytest.skip("sklearn not available")


# ---------------------------------------------------------------------------
# Tests: _load_cbbpy_team_map
# ---------------------------------------------------------------------------

class TestLoadCBBpyTeamMap:
    @patch("builtins.open", create=True)
    @patch("os.path.exists", return_value=True)
    def test_loads_csv(self, mock_exists, mock_open):
        import src.data.features.proprietary_metrics as pm
        # Reset cache
        pm._CBBPY_TEAM_MAP_CACHE = None
        from io import StringIO
        csv_content = "season,id,team,location,conference,conference_abb\n2025,1,Duke Blue Devils,Duke,ACC,ACC\n"
        mock_open.return_value.__enter__ = lambda s: StringIO(csv_content)
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        result = pm._load_cbbpy_team_map("/fake/path.csv")
        assert "Duke Blue Devils" in result
        assert result["Duke Blue Devils"] == "Duke"
        # Reset cache after test
        pm._CBBPY_TEAM_MAP_CACHE = None

    @patch("os.path.exists", return_value=False)
    def test_missing_file(self, mock_exists):
        import src.data.features.proprietary_metrics as pm
        pm._CBBPY_TEAM_MAP_CACHE = None
        result = pm._load_cbbpy_team_map("/nonexistent/path.csv")
        assert result == {}
        pm._CBBPY_TEAM_MAP_CACHE = None

    def test_cache(self):
        import src.data.features.proprietary_metrics as pm
        pm._CBBPY_TEAM_MAP_CACHE = {"cached": "value"}
        result = pm._load_cbbpy_team_map()
        assert result == {"cached": "value"}
        pm._CBBPY_TEAM_MAP_CACHE = None


# ---------------------------------------------------------------------------
# Tests: Date inference in team_games_to_game_records
# ---------------------------------------------------------------------------

class TestDateInference:
    def test_single_date_triggers_inference(self):
        """When all games have same date and >50 records, dates are inferred."""
        rows = []
        for i in range(60):
            tid = f"team_{i % 10}"
            oid = f"team_{(i + 1) % 10}"
            rows.append({
                "game_id": f"g{i}",
                "team_id": tid,
                "opponent_id": oid,
                "team_score": 75,
                "opponent_score": 70,
                "date": "2025-01-01",
            })
        records = team_games_to_game_records(rows, 2025)
        # Dates should be spread across the season, not all 2025-01-01
        unique_dates = set(r.game_date for r in records)
        assert len(unique_dates) > 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_game_record_with_zero_box_scores(self):
        g = GameRecord(
            game_id="g1", game_date="2025-01-01",
            team_id="a", team_name="A", opponent_id="b",
            points=70, opp_points=65, possessions=65,
        )
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        ff = engine._four_factors([g])
        assert ff["effective_fg_pct"] == 0.0  # fga=0 → max(0,1)=1 → 0/1=0

    def test_efficiency_ratio_near_zero_def(self):
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = _make_two_team_season(n_games=15)
        results = engine.compute(games)
        for tid, m in results.items():
            assert m.efficiency_ratio > 0  # Never divide by zero

    def test_compute_with_single_team_opponent_missing(self):
        """When opponent is not in results, WAB uses opp_em=0.0."""
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = [_make_game(team_id="solo", opponent_id="ghost",
                            game_id=f"g{i}", game_date=f"2025-01-{i+1:02d}")
                 for i in range(15)]
        results = engine.compute(games)
        assert "solo" in results

    def test_three_pt_regression_shrinkage(self):
        """B2: Bayesian 3PT shrinkage toward D1 average."""
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        games = _make_two_team_season(n_games=15)
        results = engine.compute(games)
        for tid, m in results.items():
            # three_pt_regression_signal should be close to 0 for average shooters
            assert -0.5 < m.three_pt_regression_signal < 0.5
