"""Tests for Kaggle backtest evaluation (WS6)."""

import numpy as np
import pandas as pd
import pytest

from src.ml.evaluation.kaggle_backtest import (
    BacktestReport,
    KaggleBacktester,
    YearBacktestResult,
    _get_kaggle_rank,
    validate_submission,
)


class TestGetKaggleRank:

    def test_top_1_percent(self):
        assert _get_kaggle_rank(0.410, 2024) == "top 1%"

    def test_top_5_percent(self):
        assert _get_kaggle_rank(0.435, 2024) == "top 5%"

    def test_top_50_percent(self):
        assert _get_kaggle_rank(0.460, 2024) == "top 50%"

    def test_bottom_half(self):
        assert _get_kaggle_rank(0.500, 2024) == "bottom 50%"

    def test_default_thresholds(self):
        """Unknown years should use defaults."""
        assert _get_kaggle_rank(0.410, 2010) == "top 1%"
        assert _get_kaggle_rank(0.500, 2010) == "bottom 50%"


class TestYearBacktestResult:

    def test_brier_skill_score_good(self):
        """BSS > 0 means better than seed baseline."""
        result = YearBacktestResult(
            year=2024, brier_score=0.20, log_loss=0.5,
            n_games=63, n_correct=50, accuracy=0.79,
            upset_accuracy=0.3, n_upsets_predicted=5, n_upsets_total=15,
        )
        assert result.brier_skill_score > 0

    def test_brier_skill_score_bad(self):
        """BSS < 0 means worse than seed baseline."""
        result = YearBacktestResult(
            year=2024, brier_score=0.30, log_loss=0.8,
            n_games=63, n_correct=40, accuracy=0.63,
            upset_accuracy=0.1, n_upsets_predicted=1, n_upsets_total=15,
        )
        assert result.brier_skill_score < 0


class TestKaggleBacktester:

    def test_evaluate_perfect_predictions(self):
        bt = KaggleBacktester()
        predictions = {("a", "b"): 1.0, ("c", "d"): 0.0}
        games = [
            {"team1": "a", "team2": "b", "team1_won": True,
             "team1_seed": 1, "team2_seed": 16, "round": "R64"},
            {"team1": "c", "team2": "d", "team1_won": False,
             "team1_seed": 8, "team2_seed": 9, "round": "R64"},
        ]
        result = bt.evaluate_predictions(predictions, games, 2024)
        assert result.brier_score < 0.01
        assert result.accuracy == 1.0
        assert result.n_games == 2

    def test_evaluate_all_wrong_predictions(self):
        bt = KaggleBacktester()
        predictions = {("a", "b"): 0.0, ("c", "d"): 1.0}
        games = [
            {"team1": "a", "team2": "b", "team1_won": True,
             "team1_seed": 1, "team2_seed": 16, "round": "R64"},
            {"team1": "c", "team2": "d", "team1_won": False,
             "team1_seed": 8, "team2_seed": 9, "round": "R64"},
        ]
        result = bt.evaluate_predictions(predictions, games, 2024)
        assert result.brier_score > 0.9
        assert result.accuracy == 0.0

    def test_evaluate_missing_prediction_defaults_to_half(self):
        bt = KaggleBacktester()
        predictions = {}  # No predictions
        games = [
            {"team1": "a", "team2": "b", "team1_won": True,
             "team1_seed": 1, "team2_seed": 16, "round": "R64"},
        ]
        result = bt.evaluate_predictions(predictions, games, 2024)
        # Default 0.5 prediction
        assert result.brier_score == pytest.approx(0.25, abs=0.01)

    def test_evaluate_flipped_prediction_key(self):
        """Prediction with (t2, t1) should be correctly flipped."""
        bt = KaggleBacktester()
        predictions = {("b", "a"): 0.3}  # Stored as (b, a), game is (a, b)
        games = [
            {"team1": "a", "team2": "b", "team1_won": True,
             "team1_seed": 1, "team2_seed": 16, "round": "R64"},
        ]
        result = bt.evaluate_predictions(predictions, games, 2024)
        # P(a wins) = 1 - 0.3 = 0.7
        expected_brier = (0.7 - 1.0) ** 2  # 0.09
        assert result.brier_score == pytest.approx(expected_brier, abs=0.01)

    def test_evaluate_per_round_tracking(self):
        bt = KaggleBacktester()
        predictions = {("a", "b"): 0.8, ("c", "d"): 0.8}
        games = [
            {"team1": "a", "team2": "b", "team1_won": True,
             "team1_seed": 1, "team2_seed": 16, "round": "R64"},
            {"team1": "c", "team2": "d", "team1_won": True,
             "team1_seed": 2, "team2_seed": 15, "round": "R32"},
        ]
        result = bt.evaluate_predictions(predictions, games, 2024)
        assert "R64" in result.per_round_brier
        assert "R32" in result.per_round_brier

    def test_evaluate_upset_tracking(self):
        bt = KaggleBacktester()
        # An upset: 16 seed beats 1 seed
        predictions = {("a", "b"): 0.4}  # We predict b wins (upset pred)
        games = [
            {"team1": "a", "team2": "b", "team1_won": False,
             "team1_seed": 1, "team2_seed": 16, "round": "R64"},
        ]
        result = bt.evaluate_predictions(predictions, games, 2024)
        assert result.n_upsets_total == 1
        assert result.n_upsets_predicted == 1

    def test_aggregate_results(self):
        bt = KaggleBacktester()
        results = [
            YearBacktestResult(
                year=2023, brier_score=0.20, log_loss=0.5,
                n_games=63, n_correct=50, accuracy=0.79,
                upset_accuracy=0.3, n_upsets_predicted=5, n_upsets_total=15,
                estimated_kaggle_rank="top 1%",
            ),
            YearBacktestResult(
                year=2024, brier_score=0.25, log_loss=0.6,
                n_games=63, n_correct=45, accuracy=0.71,
                upset_accuracy=0.2, n_upsets_predicted=3, n_upsets_total=15,
                estimated_kaggle_rank="top 50%",
            ),
        ]
        report = bt.aggregate_results(results)
        assert report.mean_brier == pytest.approx(0.225, abs=0.001)
        assert report.best_year == 2023
        assert report.worst_year == 2024
        assert report.years_top_1pct == 1

    def test_aggregate_empty(self):
        bt = KaggleBacktester()
        report = bt.aggregate_results([])
        assert report.mean_brier == 0.0


class TestBacktestReport:

    def test_summary_format(self):
        report = BacktestReport(
            year_results=[
                YearBacktestResult(
                    year=2024, brier_score=0.22, log_loss=0.5,
                    n_games=63, n_correct=50, accuracy=0.79,
                    upset_accuracy=0.3, n_upsets_predicted=5, n_upsets_total=15,
                    estimated_kaggle_rank="top 1%",
                ),
            ],
            mean_brier=0.22,
            std_brier=0.0,
            best_year=2024,
            worst_year=2024,
            years_top_1pct=1,
            years_top_5pct=1,
        )
        summary = report.summary()
        assert "KAGGLE BACKTEST REPORT" in summary
        assert "2024" in summary
        assert "top 1%" in summary


class TestValidateSubmission:

    def test_valid_submission(self):
        df = pd.DataFrame({
            "ID": ["2026_1001_1002", "2026_1003_1004"],
            "Pred": [0.7, 0.3],
        })
        issues = validate_submission(df, expected_rows=2)
        errors = [i for i in issues if i.startswith("ERROR")]
        assert len(errors) == 0

    def test_missing_id_column(self):
        df = pd.DataFrame({"Pred": [0.5]})
        issues = validate_submission(df)
        assert any("Missing 'ID'" in i for i in issues)

    def test_missing_pred_column(self):
        df = pd.DataFrame({"ID": ["2026_1001_1002"]})
        issues = validate_submission(df)
        assert any("Missing 'Pred'" in i for i in issues)

    def test_nan_predictions(self):
        df = pd.DataFrame({
            "ID": ["2026_1001_1002", "2026_1003_1004"],
            "Pred": [0.5, float("nan")],
        })
        issues = validate_submission(df)
        assert any("NaN" in i for i in issues)

    def test_out_of_range_predictions(self):
        df = pd.DataFrame({
            "ID": ["2026_1001_1002", "2026_1003_1004"],
            "Pred": [-0.1, 1.5],
        })
        issues = validate_submission(df)
        assert any("below 0" in i for i in issues)
        assert any("above 1" in i for i in issues)

    def test_row_count_mismatch(self):
        df = pd.DataFrame({
            "ID": ["2026_1001_1002"],
            "Pred": [0.5],
        })
        issues = validate_submission(df, expected_rows=100)
        assert any("Row count" in i for i in issues)

    def test_duplicate_ids(self):
        df = pd.DataFrame({
            "ID": ["2026_1001_1002", "2026_1001_1002"],
            "Pred": [0.5, 0.7],
        })
        issues = validate_submission(df)
        assert any("duplicate" in i for i in issues)

    def test_suspicious_half_predictions(self):
        n = 100
        df = pd.DataFrame({
            "ID": [f"2026_1001_{1002+i}" for i in range(n)],
            "Pred": [0.5] * n,
        })
        issues = validate_submission(df)
        assert any("0.5" in i and "unmapped" in i for i in issues)

    def test_womens_all_half_detected(self):
        """Detect when all women's predictions are 0.5."""
        df = pd.DataFrame({
            "ID": ["2026_3001_3002", "2026_3003_3004"],
            "Pred": [0.5, 0.5],
        })
        issues = validate_submission(df)
        assert any("women's" in i.lower() for i in issues)
