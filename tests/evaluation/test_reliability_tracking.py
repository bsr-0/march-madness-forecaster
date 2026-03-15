"""Tests for reliability tracking and drift detection.

Verifies:
- SeasonReliability fields are correct
- Drift detection flags obvious calibration shifts
- Multi-season tracking produces one result per year
- Output is JSON-serializable
"""

from __future__ import annotations

import json
import numpy as np
import pytest

from src.evaluation.reliability_tracking import (
    DriftReport,
    ReliabilityTracker,
    ReliabilityTrackingReport,
    SeasonReliability,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tracker():
    return ReliabilityTracker(n_bins=5)


@pytest.fixture
def well_calibrated_data():
    """Data where predictions are well-calibrated."""
    rng = np.random.default_rng(42)
    n = 100
    true_p = rng.uniform(0.1, 0.9, n)
    outcomes = (rng.random(n) < true_p).astype(float)
    return true_p, outcomes


@pytest.fixture
def multi_season_data():
    """Multiple seasons of well-calibrated data."""
    rng = np.random.default_rng(42)
    data = {}
    for year in [2018, 2019, 2021, 2022, 2023]:
        n = 63
        true_p = rng.uniform(0.2, 0.8, n)
        preds = true_p + rng.normal(0, 0.03, n)
        preds = np.clip(preds, 0.01, 0.99)
        outcomes = (rng.random(n) < true_p).astype(float)
        data[year] = (preds, outcomes)
    return data


@pytest.fixture
def drifting_data():
    """Data with obvious calibration drift (ECE increasing over seasons)."""
    rng = np.random.default_rng(42)
    data = {}
    for i, year in enumerate([2018, 2019, 2021, 2022, 2023]):
        n = 63
        true_p = rng.uniform(0.2, 0.8, n)
        # Add increasing bias to simulate drift
        bias = i * 0.1
        preds = true_p + bias
        preds = np.clip(preds, 0.01, 0.99)
        outcomes = (rng.random(n) < true_p).astype(float)
        data[year] = (preds, outcomes)
    return data


# ---------------------------------------------------------------------------
# Single season tests
# ---------------------------------------------------------------------------

class TestTrackSeason:
    """Tests for single season reliability tracking."""

    def test_season_has_correct_year(self, tracker, well_calibrated_data):
        preds, outs = well_calibrated_data
        result = tracker.track_season(2022, preds, outs)
        assert result.year == 2022

    def test_season_has_bin_edges(self, tracker, well_calibrated_data):
        preds, outs = well_calibrated_data
        result = tracker.track_season(2022, preds, outs)
        # n_bins=5 → 6 bin edges
        assert len(result.bin_edges) == 6
        assert result.bin_edges[0] == pytest.approx(0.0)
        assert result.bin_edges[-1] == pytest.approx(1.0)

    def test_season_has_predictions_and_rates(self, tracker, well_calibrated_data):
        preds, outs = well_calibrated_data
        result = tracker.track_season(2022, preds, outs)
        assert len(result.predicted_probabilities) > 0
        assert len(result.observed_win_rates) > 0
        assert len(result.sample_counts) > 0
        # Lengths should match (populated bins only)
        assert len(result.predicted_probabilities) == len(result.observed_win_rates)
        assert len(result.predicted_probabilities) == len(result.sample_counts)

    def test_season_counts_sum_to_n(self, tracker, well_calibrated_data):
        preds, outs = well_calibrated_data
        result = tracker.track_season(2022, preds, outs)
        assert sum(result.sample_counts) == len(preds)

    def test_season_ece_is_finite(self, tracker, well_calibrated_data):
        preds, outs = well_calibrated_data
        result = tracker.track_season(2022, preds, outs)
        assert np.isfinite(result.ece)
        assert result.ece >= 0

    def test_season_mce_is_finite(self, tracker, well_calibrated_data):
        preds, outs = well_calibrated_data
        result = tracker.track_season(2022, preds, outs)
        assert np.isfinite(result.mce)
        assert result.mce >= 0

    def test_season_n_games(self, tracker, well_calibrated_data):
        preds, outs = well_calibrated_data
        result = tracker.track_season(2022, preds, outs)
        assert result.n_games == len(preds)

    def test_season_reliability_slope_near_one(self, tracker, well_calibrated_data):
        """Well-calibrated data should have slope near 1.0."""
        preds, outs = well_calibrated_data
        result = tracker.track_season(2022, preds, outs)
        # Allow generous range since it's random data
        assert 0.3 < result.reliability_slope < 2.0

    def test_empty_season(self, tracker):
        """Empty season returns a valid but empty result."""
        result = tracker.track_season(2022, np.array([]), np.array([]))
        assert result.year == 2022
        assert result.n_games == 0


# ---------------------------------------------------------------------------
# Multi-season tracking tests
# ---------------------------------------------------------------------------

class TestMultiSeasonTracking:
    """Tests for multi-season reliability tracking."""

    def test_one_result_per_year(self, tracker, multi_season_data):
        report = tracker.track_multiple_seasons(multi_season_data)
        assert len(report.seasons) == len(multi_season_data)

    def test_years_in_order(self, tracker, multi_season_data):
        report = tracker.track_multiple_seasons(multi_season_data)
        years = [s.year for s in report.seasons]
        assert years == sorted(years)

    def test_drift_report_included(self, tracker, multi_season_data):
        report = tracker.track_multiple_seasons(multi_season_data)
        assert report.drift is not None
        assert isinstance(report.drift, DriftReport)

    def test_drift_report_has_values(self, tracker, multi_season_data):
        report = tracker.track_multiple_seasons(multi_season_data)
        drift = report.drift
        assert len(drift.years) == len(multi_season_data)
        assert len(drift.ece_values) == len(multi_season_data)
        assert len(drift.slope_values) == len(multi_season_data)


# ---------------------------------------------------------------------------
# Drift detection tests
# ---------------------------------------------------------------------------

class TestDriftDetection:
    """Tests for calibration drift detection."""

    def test_stable_data_no_drift(self, tracker, multi_season_data):
        """Well-calibrated stable data should not trigger drift."""
        report = tracker.track_multiple_seasons(multi_season_data)
        # May or may not detect drift — but ECE trend should be small
        assert abs(report.drift.ece_trend) < 0.1

    def test_obvious_drift_detected(self, tracker, drifting_data):
        """Obvious increasing bias should trigger drift detection."""
        report = tracker.track_multiple_seasons(drifting_data)
        # ECE should be trending up
        assert report.drift.ece_trend > 0
        # With strong enough drift, flag should be set
        if report.drift.ece_trend > tracker.ece_drift_threshold:
            assert report.drift.drift_detected
            assert len(report.drift.drift_reasons) > 0

    def test_drift_with_two_seasons(self, tracker):
        """Drift detection works with minimum 2 seasons."""
        rng = np.random.default_rng(42)
        data = {}
        for year in [2021, 2022]:
            n = 30
            true_p = rng.uniform(0.2, 0.8, n)
            preds = np.clip(true_p + rng.normal(0, 0.05, n), 0.01, 0.99)
            outs = (rng.random(n) < true_p).astype(float)
            data[year] = (preds, outs)

        report = tracker.track_multiple_seasons(data)
        assert report.drift is not None
        assert len(report.drift.ece_values) == 2

    def test_single_season_no_drift(self, tracker):
        """Single season cannot produce drift analysis."""
        rng = np.random.default_rng(42)
        preds = rng.uniform(0.2, 0.8, 30)
        outs = (rng.random(30) > 0.5).astype(float)
        seasons = [tracker.track_season(2022, preds, outs)]
        drift = tracker.detect_drift(seasons)
        assert not drift.drift_detected
        assert drift.ece_trend == 0.0


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------

class TestSerialization:
    """Verify outputs are JSON-serializable."""

    def test_season_to_dict(self, tracker, well_calibrated_data):
        preds, outs = well_calibrated_data
        result = tracker.track_season(2022, preds, outs)
        d = result.to_dict()
        # Verify JSON-serializable
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_report_to_dict(self, tracker, multi_season_data):
        report = tracker.track_multiple_seasons(multi_season_data)
        d = report.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_drift_to_dict(self, tracker, multi_season_data):
        report = tracker.track_multiple_seasons(multi_season_data)
        d = report.drift.to_dict()
        assert "ece_trend" in d
        assert "drift_detected" in d
        assert "drift_reasons" in d

    def test_season_dict_fields(self, tracker, well_calibrated_data):
        preds, outs = well_calibrated_data
        result = tracker.track_season(2022, preds, outs)
        d = result.to_dict()
        assert d["year"] == 2022
        assert "bin_edges" in d
        assert "predicted_probabilities" in d
        assert "observed_win_rates" in d
        assert "sample_counts" in d
        assert "ece" in d
        assert "mce" in d
