"""Tests for the outcome logger module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

from src.data.historical_tournament_results import TournamentGame
from src.evaluation.outcome_logger import (
    OutcomeLog,
    PredictionOutcomeRecord,
    build_outcome_log,
    format_summary,
    load_outcome_log,
    load_predictions_from_report,
    save_outcome_log,
    _lookup_prediction,
    _infer_team_ids,
    _split_matchup_key,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_game(
    team1: str = "duke",
    team2: str = "vermont",
    seed1: int = 1,
    seed2: int = 16,
    score1: int = 78,
    score2: int = 54,
    team1_won: bool = True,
    round_name: str = "R64",
    region: str = "East",
    year: int = 2026,
) -> TournamentGame:
    return TournamentGame(
        year=year,
        round_name=round_name,
        team1_id=team1,
        team2_id=team2,
        team1_seed=seed1,
        team2_seed=seed2,
        team1_score=score1,
        team2_score=score2,
        team1_won=team1_won,
        region=region,
    )


def _sample_games() -> List[TournamentGame]:
    return [
        _make_game("duke", "vermont", 1, 16, 78, 54, True, "R64", "East"),
        _make_game("unc", "nova", 2, 7, 65, 70, False, "R64", "East"),
        _make_game("gonzaga", "memphis", 1, 8, 80, 75, True, "R64", "West"),
        _make_game("duke", "unc", 1, 2, 72, 68, True, "E8", "East"),
        _make_game("gonzaga", "kansas", 1, 3, 60, 65, False, "S16", "West"),
    ]


def _sample_predictions() -> Dict[Tuple[str, str], float]:
    return {
        ("duke", "vermont"): 0.97,
        ("unc", "nova"): 0.65,
        ("gonzaga", "memphis"): 0.70,
        ("duke", "unc"): 0.55,
        ("gonzaga", "kansas"): 0.60,
    }


# ---------------------------------------------------------------------------
# PredictionOutcomeRecord tests
# ---------------------------------------------------------------------------

class TestPredictionOutcomeRecord:
    def test_correct_when_predicted_winner_wins(self):
        r = PredictionOutcomeRecord(
            round_name="R64", region="East",
            team1_id="duke", team2_id="vermont",
            team1_seed=1, team2_seed=16,
            predicted_prob_team1=0.97,
            team1_score=78, team2_score=54, team1_won=True,
        )
        assert r.correct is True
        assert r.upset is False

    def test_incorrect_when_predicted_winner_loses(self):
        r = PredictionOutcomeRecord(
            round_name="R64", region="East",
            team1_id="unc", team2_id="nova",
            team1_seed=2, team2_seed=7,
            predicted_prob_team1=0.65,
            team1_score=65, team2_score=70, team1_won=False,
        )
        assert r.correct is False
        assert r.upset is True  # 7-seed beats 2-seed

    def test_brier_component_perfect_prediction(self):
        r = PredictionOutcomeRecord(
            round_name="R64", region="",
            team1_id="a", team2_id="b",
            team1_seed=1, team2_seed=16,
            predicted_prob_team1=1.0,
            team1_score=80, team2_score=60, team1_won=True,
        )
        assert r.brier_component == pytest.approx(0.0)

    def test_brier_component_wrong_prediction(self):
        r = PredictionOutcomeRecord(
            round_name="R64", region="",
            team1_id="a", team2_id="b",
            team1_seed=1, team2_seed=16,
            predicted_prob_team1=0.9,
            team1_score=60, team2_score=80, team1_won=False,
        )
        assert r.brier_component == pytest.approx(0.81)

    def test_log_loss_component(self):
        r = PredictionOutcomeRecord(
            round_name="R64", region="",
            team1_id="a", team2_id="b",
            team1_seed=1, team2_seed=16,
            predicted_prob_team1=0.8,
            team1_score=80, team2_score=60, team1_won=True,
        )
        expected = -np.log(0.8)
        assert r.log_loss_component == pytest.approx(expected, rel=1e-6)

    def test_to_dict_includes_derived_fields(self):
        r = PredictionOutcomeRecord(
            round_name="R64", region="East",
            team1_id="duke", team2_id="vermont",
            team1_seed=1, team2_seed=16,
            predicted_prob_team1=0.97,
            team1_score=78, team2_score=54, team1_won=True,
        )
        d = r.to_dict()
        assert "correct" in d
        assert "upset" in d
        assert "brier_component" in d
        assert "log_loss_component" in d
        assert d["correct"] is True
        assert d["upset"] is False


# ---------------------------------------------------------------------------
# _lookup_prediction tests
# ---------------------------------------------------------------------------

class TestLookupPrediction:
    def test_direct_match(self):
        preds = {("duke", "vermont"): 0.95}
        assert _lookup_prediction(preds, "duke", "vermont") == 0.95

    def test_reverse_match(self):
        preds = {("vermont", "duke"): 0.05}
        result = _lookup_prediction(preds, "duke", "vermont")
        assert result == pytest.approx(0.95)

    def test_no_match(self):
        preds = {("duke", "vermont"): 0.95}
        assert _lookup_prediction(preds, "unc", "nova") is None


# ---------------------------------------------------------------------------
# build_outcome_log tests
# ---------------------------------------------------------------------------

class TestBuildOutcomeLog:
    def test_basic_join(self):
        games = _sample_games()
        preds = _sample_predictions()
        log = build_outcome_log(preds, games, year=2026)

        assert log.year == 2026
        assert log.n_games_with_predictions == 5
        assert log.n_games_total == 5
        assert log.n_predictions_unmatched == 0
        assert len(log.games) == 5

    def test_aggregate_metrics(self):
        games = _sample_games()
        preds = _sample_predictions()
        log = build_outcome_log(preds, games, year=2026)

        m = log.aggregate_metrics
        assert "brier_score" in m
        assert "log_loss" in m
        assert "accuracy" in m
        assert "round_weighted_brier" in m
        assert 0.0 <= m["brier_score"] <= 1.0
        assert 0.0 <= m["accuracy"] <= 1.0

    def test_per_round_metrics(self):
        games = _sample_games()
        preds = _sample_predictions()
        log = build_outcome_log(preds, games, year=2026)

        assert "R64" in log.per_round_metrics
        assert "E8" in log.per_round_metrics
        assert "S16" in log.per_round_metrics
        assert log.per_round_metrics["R64"]["n_games"] == 3

    def test_unmatched_games(self):
        games = _sample_games()
        preds = {("duke", "vermont"): 0.97}  # only 1 of 5 games
        log = build_outcome_log(preds, games, year=2026)

        assert log.n_games_with_predictions == 1
        assert log.n_predictions_unmatched == 4

    def test_reverse_lookup(self):
        """Predictions stored as (vermont, duke) should still match."""
        games = [_make_game("duke", "vermont", 1, 16, 78, 54, True)]
        preds = {("vermont", "duke"): 0.03}  # reverse order
        log = build_outcome_log(preds, games, year=2026)

        assert log.n_games_with_predictions == 1
        assert log.n_predictions_unmatched == 0
        # pred should be flipped: 1 - 0.03 = 0.97
        assert log.games[0]["predicted_prob_team1"] == pytest.approx(0.97)

    def test_empty_games(self):
        preds = _sample_predictions()
        log = build_outcome_log(preds, [], year=2026)

        assert log.n_games_with_predictions == 0
        assert log.n_games_total == 0

    def test_accuracy_calculation(self):
        # 3 correct, 2 incorrect
        games = _sample_games()
        preds = _sample_predictions()
        log = build_outcome_log(preds, games, year=2026)

        # duke beats vermont (pred 0.97 > 0.5, correct)
        # nova beats unc (pred 0.65 for unc, but unc lost -> incorrect)
        # gonzaga beats memphis (pred 0.70, correct)
        # duke beats unc (pred 0.55, correct)
        # kansas beats gonzaga (pred 0.60 for gonzaga, but gonzaga lost -> incorrect)
        # => 3 correct / 5 total = 0.6
        assert log.aggregate_metrics["accuracy"] == pytest.approx(0.6)
        assert log.aggregate_metrics["n_correct"] == 3

    def test_upset_tracking(self):
        games = _sample_games()
        preds = _sample_predictions()
        log = build_outcome_log(preds, games, year=2026)

        # Upsets: nova (7) beats unc (2), kansas (3) beats gonzaga (1)
        assert log.aggregate_metrics["n_upsets_total"] == 2


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_save_and_load_roundtrip(self, tmp_path):
        games = _sample_games()
        preds = _sample_predictions()
        log = build_outcome_log(preds, games, year=2026, prediction_source="test")

        path = str(tmp_path / "test_log.json")
        save_outcome_log(log, path)

        loaded = load_outcome_log(path)
        assert loaded.year == 2026
        assert loaded.n_games_with_predictions == 5
        assert loaded.prediction_source == "test"
        assert len(loaded.games) == 5

    def test_json_is_valid(self, tmp_path):
        games = _sample_games()
        preds = _sample_predictions()
        log = build_outcome_log(preds, games, year=2026)

        path = str(tmp_path / "test_log.json")
        save_outcome_log(log, path)

        with open(path) as f:
            data = json.load(f)

        assert data["year"] == 2026
        assert "aggregate_metrics" in data
        assert "per_round_metrics" in data
        assert "games" in data


# ---------------------------------------------------------------------------
# load_predictions_from_report tests
# ---------------------------------------------------------------------------

class TestLoadPredictionsFromReport:
    def test_load_underscore_format(self, tmp_path):
        report = {
            "predictions_2026": {
                "duke_vermont": 0.97,
                "unc_nova": 0.65,
            }
        }
        path = tmp_path / "report.json"
        path.write_text(json.dumps(report))

        preds = load_predictions_from_report(str(path))
        assert ("duke", "vermont") in preds
        assert preds[("duke", "vermont")] == pytest.approx(0.97)

    def test_load_vs_format(self, tmp_path):
        report = {
            "predictions_2026": {
                "duke__vs__vermont": 0.97,
            }
        }
        path = tmp_path / "report.json"
        path.write_text(json.dumps(report))

        preds = load_predictions_from_report(str(path))
        assert ("duke", "vermont") in preds

    def test_missing_predictions_key_raises(self, tmp_path):
        report = {"metrics": {}}
        path = tmp_path / "report.json"
        path.write_text(json.dumps(report))

        with pytest.raises(ValueError, match="No predictions_"):
            load_predictions_from_report(str(path))

    def test_multi_word_team_names_with_known_ids(self, tmp_path):
        """Keys like 'duke_michigan_state' are ambiguous without known IDs."""
        report = {
            "predictions_2026": {
                "duke_michigan_state": 0.65,
                "duke_ohio_state": 0.70,
                "michigan_state_ohio_state": 0.55,
            }
        }
        path = tmp_path / "report.json"
        path.write_text(json.dumps(report))

        known = ["duke", "michigan_state", "ohio_state"]
        preds = load_predictions_from_report(str(path), known_team_ids=known)

        assert ("duke", "michigan_state") in preds
        assert preds[("duke", "michigan_state")] == pytest.approx(0.65)
        assert ("duke", "ohio_state") in preds
        assert ("michigan_state", "ohio_state") in preds

    def test_multi_word_inferred_ids(self, tmp_path):
        """Without known IDs, inference should still work for common teams."""
        # Build enough matchups that "michigan_state" appears many times
        preds_dict = {}
        opponents = ["duke", "unc", "kansas", "kentucky", "gonzaga"]
        for opp in opponents:
            preds_dict[f"{opp}_michigan_state"] = 0.55
            preds_dict[f"{opp}_ohio_state"] = 0.60
        report = {"predictions_2026": preds_dict}
        path = tmp_path / "report.json"
        path.write_text(json.dumps(report))

        preds = load_predictions_from_report(str(path))
        assert ("duke", "michigan_state") in preds
        assert ("duke", "ohio_state") in preds

    def test_double_underscore_team_names(self, tmp_path):
        """Team IDs with double underscores like st__john_s__ny."""
        report = {
            "predictions_2026": {
                "duke_st__john_s__ny": 0.77,
            }
        }
        path = tmp_path / "report.json"
        path.write_text(json.dumps(report))

        known = ["duke", "st__john_s__ny"]
        preds = load_predictions_from_report(str(path), known_team_ids=known)
        assert ("duke", "st__john_s__ny") in preds


# ---------------------------------------------------------------------------
# _infer_team_ids tests
# ---------------------------------------------------------------------------

class TestInferTeamIds:
    def test_basic_inference(self):
        keys = [
            "duke_michigan_state",
            "duke_ohio_state",
            "duke_kansas",
            "michigan_state_ohio_state",
            "michigan_state_kansas",
            "ohio_state_kansas",
        ]
        ids = _infer_team_ids(keys)
        assert "duke" in ids
        assert "michigan_state" in ids
        assert "ohio_state" in ids
        assert "kansas" in ids

    def test_empty_input(self):
        ids = _infer_team_ids([])
        assert ids == set()


# ---------------------------------------------------------------------------
# _split_matchup_key tests
# ---------------------------------------------------------------------------

class TestSplitMatchupKey:
    def test_simple_two_part(self):
        assert _split_matchup_key("duke_kansas", set()) == ("duke", "kansas")

    def test_known_multi_word(self):
        team_set = {"duke", "michigan_state"}
        assert _split_matchup_key("duke_michigan_state", team_set) == ("duke", "michigan_state")

    def test_both_multi_word(self):
        team_set = {"michigan_state", "ohio_state"}
        assert _split_matchup_key("michigan_state_ohio_state", team_set) == (
            "michigan_state", "ohio_state"
        )

    def test_fallback_one_side_known(self):
        team_set = {"duke"}
        result = _split_matchup_key("duke_michigan_state", team_set)
        assert result is not None
        assert result[0] == "duke"
        assert result[1] == "michigan_state"


# ---------------------------------------------------------------------------
# format_summary tests
# ---------------------------------------------------------------------------

class TestFormatSummary:
    def test_summary_contains_key_info(self):
        games = _sample_games()
        preds = _sample_predictions()
        log = build_outcome_log(preds, games, year=2026)

        summary = format_summary(log)
        assert "2026" in summary
        assert "Brier score" in summary
        assert "Accuracy" in summary
        assert "PER-ROUND BREAKDOWN" in summary

    def test_summary_with_unmatched_warning(self):
        games = _sample_games()
        preds = {("duke", "vermont"): 0.97}
        log = build_outcome_log(preds, games, year=2026)

        summary = format_summary(log)
        assert "WARNING" in summary
        assert "no matching prediction" in summary

    def test_empty_log_summary(self):
        log = build_outcome_log({}, [], year=2026)
        summary = format_summary(log)
        assert "2026" in summary
        assert "0 / 0" in summary
