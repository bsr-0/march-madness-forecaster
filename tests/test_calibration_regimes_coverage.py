"""Tests for src/evaluation/calibration_regimes.py."""

import numpy as np
import pytest

from src.evaluation.calibration_regimes import (
    CalibrationRegimeAnalyzer,
    CalibrationRegimeReport,
    FullRegimeReport,
    MethodRegimeComparison,
    PROBABILITY_REGIMES,
    ROUND_REGIMES,
    _compute_regime_metrics,
)
from src.evaluation.bootstrap_metrics import BootstrapResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_data(n=100, seed=42):
    """Generate synthetic predictions and outcomes for testing."""
    rng = np.random.RandomState(seed)
    predictions = rng.uniform(0.45, 1.0, size=n)
    outcomes = (rng.rand(n) < predictions).astype(float)
    return predictions, outcomes


# ---------------------------------------------------------------------------
# Dataclass to_dict tests
# ---------------------------------------------------------------------------

class TestCalibrationRegimeReportToDict:
    def test_to_dict_with_none_bootstrap(self):
        report = CalibrationRegimeReport(regime_name="test", n_games=0)
        d = report.to_dict()
        assert d["regime_name"] == "test"
        assert d["n_games"] == 0
        assert d["brier"] is None
        assert d["log_loss"] is None
        assert d["ece"] == 0.0
        assert d["mean_predicted"] == 0.0
        assert d["mean_actual"] == 0.0
        assert d["calibration_error"] == 0.0

    def test_to_dict_with_bootstrap_results(self):
        br = BootstrapResult(
            estimate=0.2, ci_lower=0.1, ci_upper=0.3,
            ci_level=0.95, n_bootstrap=100, n_samples=50,
        )
        report = CalibrationRegimeReport(
            regime_name="heavy_favorites",
            n_games=50,
            brier=br,
            log_loss_val=br,
            ece=0.05,
            mean_predicted=0.85,
            mean_actual=0.90,
            calibration_error=0.05,
        )
        d = report.to_dict()
        assert d["regime_name"] == "heavy_favorites"
        assert d["n_games"] == 50
        assert isinstance(d["brier"], dict)
        assert isinstance(d["log_loss"], dict)
        assert d["ece"] == 0.05


class TestMethodRegimeComparisonToDict:
    def test_to_dict_empty_regimes(self):
        comp = MethodRegimeComparison(method_name="platt")
        d = comp.to_dict()
        assert d["method_name"] == "platt"
        assert d["regimes"] == []

    def test_to_dict_with_regimes(self):
        r = CalibrationRegimeReport(regime_name="near_pickem", n_games=10)
        comp = MethodRegimeComparison(method_name="isotonic", regimes=[r])
        d = comp.to_dict()
        assert len(d["regimes"]) == 1
        assert d["regimes"][0]["regime_name"] == "near_pickem"


class TestFullRegimeReportToDict:
    def test_to_dict_empty(self):
        report = FullRegimeReport()
        d = report.to_dict()
        assert d["n_total"] == 0
        assert d["probability_regimes"] == []
        assert d["round_regimes"] == []
        assert d["method_comparisons"] == []

    def test_to_dict_populated(self):
        pr = CalibrationRegimeReport(regime_name="heavy_favorites", n_games=20)
        rr = CalibrationRegimeReport(regime_name="round_S16", n_games=8)
        mc = MethodRegimeComparison(method_name="platt", regimes=[pr])
        report = FullRegimeReport(
            probability_regimes=[pr],
            round_regimes=[rr],
            method_comparisons=[mc],
            n_total=100,
        )
        d = report.to_dict()
        assert d["n_total"] == 100
        assert len(d["probability_regimes"]) == 1
        assert len(d["round_regimes"]) == 1
        assert len(d["method_comparisons"]) == 1
        assert d["method_comparisons"][0]["method_name"] == "platt"


# ---------------------------------------------------------------------------
# _compute_regime_metrics tests
# ---------------------------------------------------------------------------

class TestComputeRegimeMetrics:
    def test_empty_data(self):
        preds = np.array([])
        outs = np.array([])
        report = _compute_regime_metrics(preds, outs, "empty", n_bootstrap=100)
        assert report.regime_name == "empty"
        assert report.n_games == 0
        assert report.brier is None
        assert report.log_loss_val is None

    def test_normal_data(self):
        preds, outs = _make_test_data(n=50, seed=42)
        report = _compute_regime_metrics(
            preds, outs, "test_regime", n_bootstrap=100, random_seed=42,
        )
        assert report.regime_name == "test_regime"
        assert report.n_games == 50
        assert report.brier is not None
        assert report.log_loss_val is not None
        assert 0.0 <= report.ece <= 1.0
        assert 0.0 <= report.mean_predicted <= 1.0
        assert 0.0 <= report.mean_actual <= 1.0
        assert report.calibration_error >= 0.0
        # Bootstrap results should have CIs
        assert report.brier.ci_lower <= report.brier.estimate <= report.brier.ci_upper

    def test_small_data_no_bootstrap(self):
        """With fewer than 3 samples, bootstrap should not be computed."""
        preds = np.array([0.7, 0.8])
        outs = np.array([1.0, 1.0])
        report = _compute_regime_metrics(preds, outs, "tiny", n_bootstrap=100)
        assert report.n_games == 2
        assert report.brier is None
        assert report.log_loss_val is None
        assert report.mean_predicted == pytest.approx(0.75)
        assert report.mean_actual == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# CalibrationRegimeAnalyzer.analyze tests
# ---------------------------------------------------------------------------

class TestAnalyze:
    def test_analyze_without_round_names(self):
        preds, outs = _make_test_data(n=200, seed=42)
        analyzer = CalibrationRegimeAnalyzer(n_bootstrap=100)
        report = analyzer.analyze(preds, outs, round_names=None, random_seed=42)

        assert isinstance(report, FullRegimeReport)
        assert report.n_total == 200
        assert len(report.probability_regimes) == len(PROBABILITY_REGIMES)
        assert report.round_regimes == []

        # Each probability regime report should have a valid regime name
        regime_names = {r.regime_name for r in report.probability_regimes}
        assert regime_names == set(PROBABILITY_REGIMES.keys())

        # All n_games should sum to <= total (some predictions may not fall in any regime)
        total_regime_games = sum(r.n_games for r in report.probability_regimes)
        assert total_regime_games <= 200

    def test_analyze_with_round_names(self):
        preds, outs = _make_test_data(n=40, seed=42)
        round_names = ["S16"] * 10 + ["E8"] * 10 + ["F4"] * 10 + ["NCG"] * 10
        analyzer = CalibrationRegimeAnalyzer(n_bootstrap=100)
        report = analyzer.analyze(preds, outs, round_names=round_names, random_seed=42)

        assert report.n_total == 40
        assert len(report.round_regimes) == len(ROUND_REGIMES)

        round_regime_names = [r.regime_name for r in report.round_regimes]
        for rr in ROUND_REGIMES:
            assert f"round_{rr}" in round_regime_names

        # Each round regime should have 10 games
        for rr in report.round_regimes:
            assert rr.n_games == 10

    def test_analyze_custom_regimes(self):
        preds, outs = _make_test_data(n=100, seed=42)
        custom_regimes = {"high": (0.70, 1.00), "low": (0.45, 0.70)}
        analyzer = CalibrationRegimeAnalyzer(
            probability_regimes=custom_regimes, n_bootstrap=100,
        )
        report = analyzer.analyze(preds, outs, random_seed=42)
        assert len(report.probability_regimes) == 2
        names = {r.regime_name for r in report.probability_regimes}
        assert names == {"high", "low"}

    def test_analyze_to_dict_roundtrip(self):
        preds, outs = _make_test_data(n=50, seed=42)
        analyzer = CalibrationRegimeAnalyzer(n_bootstrap=100)
        report = analyzer.analyze(preds, outs, random_seed=42)
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "n_total" in d
        assert "probability_regimes" in d


# ---------------------------------------------------------------------------
# CalibrationRegimeAnalyzer.compare_methods tests
# ---------------------------------------------------------------------------

class TestCompareMethods:
    def test_compare_methods_basic(self):
        rng = np.random.RandomState(42)
        n = 100
        preds_a = rng.uniform(0.45, 1.0, size=n)
        preds_b = rng.uniform(0.45, 1.0, size=n)
        outcomes = (rng.rand(n) < preds_a).astype(float)

        methods = {"method_a": preds_a, "method_b": preds_b}
        analyzer = CalibrationRegimeAnalyzer(n_bootstrap=100)
        report = analyzer.compare_methods(methods, outcomes, random_seed=42)

        assert isinstance(report, FullRegimeReport)
        assert report.n_total == n
        assert len(report.method_comparisons) == 2

        method_names = {mc.method_name for mc in report.method_comparisons}
        assert method_names == {"method_a", "method_b"}

        # Each method comparison should have one report per probability regime
        for mc in report.method_comparisons:
            assert len(mc.regimes) == len(PROBABILITY_REGIMES)

    def test_compare_methods_with_round_names(self):
        rng = np.random.RandomState(42)
        n = 40
        preds = rng.uniform(0.45, 1.0, size=n)
        outcomes = (rng.rand(n) < preds).astype(float)
        round_names = ["S16"] * 10 + ["E8"] * 10 + ["F4"] * 10 + ["NCG"] * 10

        methods = {"platt": preds}
        analyzer = CalibrationRegimeAnalyzer(n_bootstrap=100)
        report = analyzer.compare_methods(
            methods, outcomes, round_names=round_names, random_seed=42,
        )

        assert len(report.round_regimes) == len(ROUND_REGIMES)
        assert len(report.method_comparisons) == 1
        assert report.method_comparisons[0].method_name == "platt"

    def test_compare_methods_to_dict(self):
        rng = np.random.RandomState(42)
        n = 60
        preds = rng.uniform(0.50, 0.95, size=n)
        outcomes = (rng.rand(n) < preds).astype(float)

        methods = {"iso": preds}
        analyzer = CalibrationRegimeAnalyzer(n_bootstrap=100)
        report = analyzer.compare_methods(methods, outcomes, random_seed=42)
        d = report.to_dict()

        assert isinstance(d, dict)
        assert len(d["method_comparisons"]) == 1
        assert d["method_comparisons"][0]["method_name"] == "iso"
        assert len(d["method_comparisons"][0]["regimes"]) == len(PROBABILITY_REGIMES)
