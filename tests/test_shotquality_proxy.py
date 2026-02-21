"""Tests for proprietary metrics engine (replaces ShotQuality proxy tests)."""

from src.data.features.proprietary_metrics import (
    GameRecord,
    ProprietaryMetricsEngine,
    ProprietaryTeamMetrics,
)


def _make_game_pair(
    game_id: str,
    game_date: str,
    team_id: str,
    team_name: str,
    opp_id: str,
    opp_name: str,
    pts: float,
    opp_pts: float,
    poss: float = 69.0,
    *,
    fga=56, fgm=28, fg3a=20, fg3m=8, fta=18, ftm=14, tov=9, orb=10, drb=20,
    ast=14, stl=6, blk=3, pf=16,
    opp_fga=58, opp_fgm=25, opp_fg3a=22, opp_fg3m=6, opp_fta=14, opp_ftm=10,
    opp_tov=11, opp_orb=9, opp_drb=18,
    opp_ast=12, opp_stl=5, opp_blk=2, opp_pf=18,
    is_neutral=True,
):
    """Create both sides of a game record pair."""
    return [
        GameRecord(
            game_id=game_id, game_date=game_date,
            team_id=team_id, team_name=team_name, opponent_id=opp_id,
            points=pts, opp_points=opp_pts, possessions=poss,
            fga=fga, fgm=fgm, fg3a=fg3a, fg3m=fg3m, fta=fta, ftm=ftm,
            tov=tov, orb=orb, drb=drb, ast=ast, stl=stl, blk=blk, pf=pf,
            opp_fga=opp_fga, opp_fgm=opp_fgm, opp_fg3a=opp_fg3a, opp_fg3m=opp_fg3m,
            opp_fta=opp_fta, opp_ftm=opp_ftm, opp_tov=opp_tov, opp_orb=opp_orb,
            opp_drb=opp_drb, opp_ast=opp_ast, opp_stl=opp_stl, opp_blk=opp_blk,
            opp_pf=opp_pf,
            is_neutral=is_neutral,
        ),
        GameRecord(
            game_id=game_id, game_date=game_date,
            team_id=opp_id, team_name=opp_name, opponent_id=team_id,
            points=opp_pts, opp_points=pts, possessions=poss,
            fga=opp_fga, fgm=opp_fgm, fg3a=opp_fg3a, fg3m=opp_fg3m,
            fta=opp_fta, ftm=opp_ftm, tov=opp_tov, orb=opp_orb, drb=opp_drb,
            ast=opp_ast, stl=opp_stl, blk=opp_blk, pf=opp_pf,
            opp_fga=fga, opp_fgm=fgm, opp_fg3a=fg3a, opp_fg3m=fg3m,
            opp_fta=fta, opp_ftm=ftm, opp_tov=tov, opp_orb=orb, opp_drb=drb,
            opp_ast=ast, opp_stl=stl, opp_blk=blk,
            opp_pf=pf,
            is_neutral=is_neutral,
        ),
    ]


def test_proprietary_engine_computes_metrics_from_game_records():
    records = _make_game_pair(
        "g1", "2026-01-10", "duke", "Duke", "unc", "UNC", 78, 70,
    )
    engine = ProprietaryMetricsEngine()
    results = engine.compute(records)

    assert len(results) == 2
    assert "duke" in results
    assert "unc" in results

    duke = results["duke"]
    assert isinstance(duke, ProprietaryTeamMetrics)
    assert duke.adj_offensive_efficiency > 0
    assert duke.adj_defensive_efficiency > 0
    assert duke.offensive_xp_per_possession > 0
    assert duke.wins == 1
    assert duke.losses == 0

    unc = results["unc"]
    assert unc.wins == 0
    assert unc.losses == 1
    # Duke won, so Duke should have higher AdjEM
    assert duke.adj_efficiency_margin > unc.adj_efficiency_margin


def test_proprietary_engine_handles_empty_records():
    engine = ProprietaryMetricsEngine()
    results = engine.compute([])
    assert results == {}


def test_elo_ratings_computed():
    """Elo ratings should favor the winning team after a game."""
    records = _make_game_pair(
        "g1", "2026-01-10", "duke", "Duke", "unc", "UNC", 78, 70,
    )
    engine = ProprietaryMetricsEngine()
    results = engine.compute(records)

    duke = results["duke"]
    unc = results["unc"]
    # Winner should have higher Elo
    assert duke.elo_rating > unc.elo_rating
    # Both should be within reasonable range (MOV-adjusted K=38 can produce ~±150 swings)
    assert 1350 < duke.elo_rating < 1700
    assert 1300 < unc.elo_rating < 1650


def test_extended_box_score_metrics():
    """FT%, A/TO ratio, steal/block rates should be computed from box-score fields."""
    records = _make_game_pair(
        "g1", "2026-01-10", "duke", "Duke", "unc", "UNC", 78, 70,
        ast=14, stl=6, blk=3, opp_ast=12, opp_stl=5, opp_blk=2,
    )
    engine = ProprietaryMetricsEngine()
    results = engine.compute(records)

    duke = results["duke"]
    # FT%: ftm=14, fta=18 → ~0.778
    assert abs(duke.free_throw_pct - 14.0 / 18.0) < 1e-6
    # A/TO: ast=14, tov=9 → ~1.556
    assert abs(duke.assist_to_turnover_ratio - 14.0 / 9.0) < 1e-6
    # Assist rate: ast=14, fgm=28 → 0.5
    assert abs(duke.assist_rate - 14.0 / 28.0) < 1e-6
    # Steal rate: stl=6, opp_poss=69 → 6/69*100 ≈ 8.7
    assert duke.steal_rate > 0
    # Block rate: blk=3, opp_poss=69 → 3/69*100 ≈ 4.35
    assert duke.block_rate > 0
    # Defensive disruption = steal_rate + block_rate (approximately)
    assert abs(duke.defensive_disruption_rate - (duke.steal_rate + duke.block_rate)) < 1e-6


def test_opponent_shot_selection_metrics():
    """Opponent 2P% and 3PA rate should reflect defensive shot selection."""
    records = _make_game_pair(
        "g1", "2026-01-10", "duke", "Duke", "unc", "UNC", 78, 70,
    )
    engine = ProprietaryMetricsEngine()
    results = engine.compute(records)

    duke = results["duke"]
    # Opp 2P%: (opp_fgm - opp_fg3m) / (opp_fga - opp_fg3a) = (25-6)/(58-22) = 19/36
    expected_opp_2p = (25 - 6) / (58 - 22)
    assert abs(duke.opp_two_pt_pct_allowed - expected_opp_2p) < 1e-6
    # Opp 3PA rate: opp_fg3a / opp_fga = 22/58
    expected_opp_3pa_rate = 22 / 58
    assert abs(duke.opp_three_pt_attempt_rate - expected_opp_3pa_rate) < 1e-6


def test_conference_strength_with_map():
    """Conference AdjEM should average same-conference teams."""
    records = _make_game_pair(
        "g1", "2026-01-10", "duke", "Duke", "unc", "UNC", 78, 70,
    )
    conf_map = {"duke": "ACC", "unc": "ACC"}
    engine = ProprietaryMetricsEngine()
    results = engine.compute(records, conference_map=conf_map)

    # Both are in ACC, so conference AdjEM should be the average of their AdjEM
    avg_em = (results["duke"].adj_efficiency_margin + results["unc"].adj_efficiency_margin) / 2.0
    assert abs(results["duke"].conference_adj_em - avg_em) < 1e-6
    assert abs(results["unc"].conference_adj_em - avg_em) < 1e-6


def test_conference_strength_without_map():
    """Without a conference map, conference_adj_em should fall back to SOS."""
    records = _make_game_pair(
        "g1", "2026-01-10", "duke", "Duke", "unc", "UNC", 78, 70,
    )
    engine = ProprietaryMetricsEngine()
    results = engine.compute(records, conference_map=None)

    # Should fall back to SOS
    assert abs(results["duke"].conference_adj_em - results["duke"].sos_adj_em) < 1e-6


def test_to_dict_includes_all_new_fields():
    """to_dict() should include every new metric field."""
    records = _make_game_pair(
        "g1", "2026-01-10", "duke", "Duke", "unc", "UNC", 78, 70,
        ast=14, stl=6, blk=3,
    )
    engine = ProprietaryMetricsEngine()
    results = engine.compute(records)
    d = results["duke"].to_dict()

    required_keys = [
        "elo_rating", "free_throw_pct", "opp_free_throw_pct",
        "assist_to_turnover_ratio", "assist_rate",
        "steal_rate", "block_rate", "defensive_disruption_rate",
        "opp_two_pt_pct_allowed", "opp_three_pt_attempt_rate",
        "conference_adj_em", "seed_efficiency_residual",
        # Exhaustive audit additions
        "win_pct", "elite_sos", "q1_wins", "q1_losses", "q1_win_pct",
        "efficiency_ratio", "foul_rate", "three_pt_regression_signal",
        "barthag", "two_pt_pct", "three_pt_pct", "three_pt_rate",
        "defensive_xp_per_possession",
        # Schedule/context features
        "rest_days", "top5_minutes_share", "preseason_ap_rank",
        "coach_tournament_appearances", "conf_tourney_champion",
    ]
    for key in required_keys:
        assert key in d, f"Missing key: {key}"


def test_multi_game_elo_convergence():
    """Elo should separate teams over multiple games."""
    records = []
    for i in range(5):
        records.extend(_make_game_pair(
            f"g{i+1}", f"2026-01-{10+i:02d}",
            "duke", "Duke", "unc", "UNC",
            pts=75 + i, opp_pts=70 - i,  # Duke wins by increasing margins
        ))
    engine = ProprietaryMetricsEngine()
    results = engine.compute(records)

    # After 5 straight wins, Duke's Elo should be meaningfully above UNC's
    assert results["duke"].elo_rating > results["unc"].elo_rating + 50


def test_foul_rate_computed():
    """Foul rate should be personal fouls per possession."""
    records = _make_game_pair(
        "g1", "2026-01-10", "duke", "Duke", "unc", "UNC", 78, 70,
        pf=16, opp_pf=18,
    )
    engine = ProprietaryMetricsEngine()
    results = engine.compute(records)

    duke = results["duke"]
    # pf=16, possessions=69 → 16/69 ≈ 0.2319
    expected_foul_rate = 16.0 / 69.0
    assert abs(duke.foul_rate - expected_foul_rate) < 1e-6

    unc = results["unc"]
    # opp_pf=18 becomes UNC's pf → 18/69
    expected_unc_foul_rate = 18.0 / 69.0
    assert abs(unc.foul_rate - expected_unc_foul_rate) < 1e-6


def test_win_pct_and_efficiency_ratio():
    """Win% and efficiency ratio should be computed correctly."""
    records = _make_game_pair(
        "g1", "2026-01-10", "duke", "Duke", "unc", "UNC", 78, 70,
    )
    engine = ProprietaryMetricsEngine()
    results = engine.compute(records)

    duke = results["duke"]
    unc = results["unc"]

    # Duke won 1 of 1 game
    assert abs(duke.win_pct - 1.0) < 1e-6
    assert abs(unc.win_pct - 0.0) < 1e-6

    # Efficiency ratio = AdjO / AdjD (should be > 1 for winning team)
    assert duke.efficiency_ratio > 1.0
    assert unc.efficiency_ratio < 1.0


def test_three_pt_regression_signal():
    """3-Point regression signal should reflect deviation from D1 average."""
    records = _make_game_pair(
        "g1", "2026-01-10", "duke", "Duke", "unc", "UNC", 78, 70,
        fg3a=20, fg3m=8,  # Duke 3P% = 0.400
    )
    engine = ProprietaryMetricsEngine()
    results = engine.compute(records)

    duke = results["duke"]
    # B2: Bayesian shrinkage — shrunk_3p = (20*0.4 + 100*0.345) / 120 = 0.3542
    # regression signal = 0.3542 - 0.345 ≈ 0.0092
    assert abs(duke.three_pt_regression_signal - 0.0092) < 2e-3


def test_barthag_computed():
    """Barthag (Pythagorean win%) should favor higher-efficiency teams."""
    records = _make_game_pair(
        "g1", "2026-01-10", "duke", "Duke", "unc", "UNC", 78, 70,
    )
    engine = ProprietaryMetricsEngine()
    results = engine.compute(records)

    duke = results["duke"]
    unc = results["unc"]

    # Duke has better efficiency → higher barthag
    assert duke.barthag > unc.barthag
    # Both should be in [0, 1]
    assert 0.0 <= duke.barthag <= 1.0
    assert 0.0 <= unc.barthag <= 1.0


def test_rest_days_computed():
    """Rest days should reflect days from last game to tournament start."""
    records = _make_game_pair(
        "g1", "2026-03-15", "duke", "Duke", "unc", "UNC", 78, 70,
    )
    engine = ProprietaryMetricsEngine()
    results = engine.compute(records)

    duke = results["duke"]
    # Last game 2026-03-15, tournament ~2026-03-20 → ~5 days
    assert 3.0 <= duke.rest_days <= 7.0


def test_torvik_to_dict_includes_all_fields():
    """TorVikTeam.to_dict() should roundtrip all scraped fields."""
    from src.data.scrapers.torvik import TorVikTeam

    team = TorVikTeam(
        team_id="duke", name="Duke", conference="ACC",
        t_rank=5, barthag=0.95,
        adj_offensive_efficiency=118.0, adj_defensive_efficiency=92.0,
        adj_tempo=70.0,
        effective_fg_pct=0.55, turnover_rate=0.16,
        offensive_reb_rate=0.32, free_throw_rate=0.35,
        opp_effective_fg_pct=0.48, opp_turnover_rate=0.20,
        defensive_reb_rate=0.72, opp_free_throw_rate=0.28,
        two_pt_pct=0.52, three_pt_pct=0.37, three_pt_rate=0.40,
        ft_pct=0.75, block_pct=0.08, steal_pct=0.09,
        opp_two_pt_pct=0.46, opp_three_pt_pct=0.32, opp_three_pt_rate=0.38,
        wab=5.2, wins=25, losses=5, conf_wins=14, conf_losses=4,
    )
    d = team.to_dict()

    # Verify ALL fields roundtrip
    assert d["two_pt_pct"] == 0.52
    assert d["three_pt_pct"] == 0.37
    assert d["three_pt_rate"] == 0.40
    assert d["ft_pct"] == 0.75
    assert d["block_pct"] == 0.08
    assert d["steal_pct"] == 0.09
    assert d["opp_two_pt_pct"] == 0.46
    assert d["opp_three_pt_pct"] == 0.32
    assert d["opp_three_pt_rate"] == 0.38
    assert d["conf_wins"] == 14
    assert d["conf_losses"] == 4
    assert d["conference"] == "ACC"


def test_feature_vector_dimension_matches_names():
    """to_vector() length must equal get_feature_names() length."""
    from src.data.features.feature_engineering import TeamFeatures

    features = TeamFeatures(team_id="test", team_name="Test", seed=5, region="East")
    vec = features.to_vector(include_embeddings=False)
    names = TeamFeatures.get_feature_names(include_embeddings=False)

    assert len(vec) == len(names), (
        f"Vector length {len(vec)} != names length {len(names)}. "
        f"Names: {names}"
    )
