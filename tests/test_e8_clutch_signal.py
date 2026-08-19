"""Tests for the clutch-composure signal added to E8MatchupInteractionScorer."""

from types import SimpleNamespace

from src.optimization.e8_matchup_scorer import _SIGNAL_WEIGHTS, E8MatchupInteractionScorer


def test_signal_weights_sum_to_one():
    assert abs(sum(_SIGNAL_WEIGHTS.values()) - 1.0) < 1e-9


def test_composed_team_gets_positive_adjustment():
    scorer = E8MatchupInteractionScorer()
    composed = SimpleNamespace(blown_10pt_lead_rate=0.1, close_game_win_rate=0.8)
    shaky = SimpleNamespace(blown_10pt_lead_rate=0.6, close_game_win_rate=0.2)

    signal = scorer._clutch_composure_interaction(composed, shaky)
    assert signal > 0.5  # composed team favored

    result = scorer.score(composed, shaky)
    assert -0.08 <= result.adjustment <= 0.08


def test_missing_clutch_data_falls_back_to_neutral_default():
    scorer = E8MatchupInteractionScorer()
    no_data_a = SimpleNamespace()
    no_data_b = SimpleNamespace()
    signal = scorer._clutch_composure_interaction(no_data_a, no_data_b)
    assert signal == 0.5  # both neutral -> no asymmetry


def test_score_still_bounded_with_full_signal_set():
    scorer = E8MatchupInteractionScorer()
    team_a = SimpleNamespace(
        opp_turnover_rate=0.25,
        turnover_rate=0.12,
        offensive_reb_rate=0.35,
        defensive_reb_rate=0.78,
        adj_tempo=72.0,
        three_pt_pct=0.40,
        opp_effective_fg_pct=0.42,
        coach_e8_appearances=5,
        coach_deep_run_rate=0.6,
        blown_10pt_lead_rate=0.05,
        close_game_win_rate=0.9,
    )
    team_b = SimpleNamespace(
        opp_turnover_rate=0.14,
        turnover_rate=0.22,
        offensive_reb_rate=0.20,
        defensive_reb_rate=0.65,
        adj_tempo=64.0,
        three_pt_pct=0.28,
        opp_effective_fg_pct=0.55,
        coach_e8_appearances=0,
        coach_deep_run_rate=0.0,
        blown_10pt_lead_rate=0.7,
        close_game_win_rate=0.1,
    )
    result = scorer.score(team_a, team_b)
    assert -0.08 <= result.adjustment <= 0.08
    assert set(result.signals) == set(_SIGNAL_WEIGHTS)
