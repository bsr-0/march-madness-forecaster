"""Tests for evaluation integrity framework (Phase 1 lockdown).

Tests cover:
1. YearSplitPolicy — hard dev/holdout split enforcement
2. Freeze-before-predict gate — require_freeze_for_season
3. CanonicalLeaderboard — all five metric families
4. Metric computation helpers — probability, upset, bracket-pool
"""

import json
import math
import os
import tempfile
import time

import numpy as np
import pytest

from src.ml.evaluation.evaluation_integrity import (
    CANONICAL_DEV_YEARS,
    CANONICAL_HOLDOUT_YEARS,
    DEFAULT_BRACKET_POOL_POINTS,
    PROSPECTIVE_CANDIDATE_YEARS,
    BracketPoolResult,
    CanonicalLeaderboard,
    FreezeRequiredError,
    HoldoutContaminationError,
    LeaderboardEntry,
    UpsetMetrics,
    YearSplitPolicy,
    build_leaderboard_entry,
    compute_bracket_pool_score,
    compute_probability_metrics,
    compute_upset_metrics,
    require_freeze_for_season,
)


# ───────────────────────────────────────────────────────────────────────
# 1. YearSplitPolicy tests
# ───────────────────────────────────────────────────────────────────────

class TestYearSplitPolicy:
    """Tests for the hard dev/holdout year split enforcement."""

    def test_default_construction(self):
        policy = YearSplitPolicy()
        assert list(policy.dev_years) == CANONICAL_DEV_YEARS
        assert list(policy.holdout_years) == CANONICAL_HOLDOUT_YEARS
        assert list(policy.prospective_years) == PROSPECTIVE_CANDIDATE_YEARS

    def test_no_overlap_dev_holdout(self):
        """Dev and holdout years must not overlap."""
        with pytest.raises(ValueError, match="Dev/holdout overlap"):
            YearSplitPolicy(
                dev_years=(2021, 2022, 2023),
                holdout_years=(2023, 2024),
            )

    def test_no_overlap_dev_prospective(self):
        with pytest.raises(ValueError, match="Dev/prospective overlap"):
            YearSplitPolicy(
                dev_years=(2026,),
                holdout_years=(2025,),
                prospective_years=(2026,),
            )

    def test_no_overlap_holdout_prospective(self):
        with pytest.raises(ValueError, match="Holdout/prospective overlap"):
            YearSplitPolicy(
                dev_years=(2021,),
                holdout_years=(2026,),
                prospective_years=(2026,),
            )

    def test_covid_year_rejected(self):
        """2020 must not appear in any partition."""
        with pytest.raises(ValueError, match="COVID year 2020"):
            YearSplitPolicy(dev_years=(2019, 2020, 2021), holdout_years=(2025,))
        with pytest.raises(ValueError, match="COVID year 2020"):
            YearSplitPolicy(dev_years=(2019,), holdout_years=(2020,))

    def test_immutable(self):
        """YearSplitPolicy is frozen and cannot be mutated."""
        policy = YearSplitPolicy()
        with pytest.raises(AttributeError):
            policy.dev_years = (2021,)

    def test_assert_dev_only_passes(self):
        policy = YearSplitPolicy()
        policy.assert_dev_only([2016, 2017, 2018], context="training")

    def test_assert_dev_only_rejects_holdout_year(self):
        policy = YearSplitPolicy()
        with pytest.raises(HoldoutContaminationError, match="NOT in the dev set"):
            policy.assert_dev_only([2016, 2025], context="training")

    def test_assert_dev_only_rejects_unknown_year(self):
        policy = YearSplitPolicy()
        with pytest.raises(HoldoutContaminationError):
            policy.assert_dev_only([2030])

    def test_assert_holdout_allowed(self):
        policy = YearSplitPolicy()
        policy.assert_holdout_allowed([2025])

    def test_assert_holdout_allowed_rejects_dev_year(self):
        policy = YearSplitPolicy()
        with pytest.raises(HoldoutContaminationError):
            policy.assert_holdout_allowed([2018])

    def test_evidence_level_prospective_with_freeze(self):
        policy = YearSplitPolicy()
        assert policy.evidence_level(2026, has_freeze=True) == 1

    def test_evidence_level_holdout_with_freeze(self):
        policy = YearSplitPolicy()
        assert policy.evidence_level(2025, has_freeze=True) == 2

    def test_evidence_level_holdout_without_freeze(self):
        policy = YearSplitPolicy()
        assert policy.evidence_level(2025, has_freeze=False) == 3

    def test_evidence_level_dev_year(self):
        policy = YearSplitPolicy()
        assert policy.evidence_level(2018, has_freeze=True) == 3

    def test_to_dict(self):
        policy = YearSplitPolicy()
        d = policy.to_dict()
        assert "dev_years" in d
        assert "holdout_years" in d
        assert "prospective_years" in d

    def test_from_config(self):
        """from_config should build a policy from a config-like object."""
        class FakeConfig:
            dev_years = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]
            holdout_years = [2025]

        policy = YearSplitPolicy.from_config(FakeConfig())
        assert 2025 in policy.holdout_years
        assert 2025 not in policy.dev_years
        assert 2020 not in policy.dev_years

    def test_from_config_generates_prospective(self):
        class FakeConfig:
            dev_years = [2016, 2017, 2018]
            holdout_years = [2019]

        policy = YearSplitPolicy.from_config(FakeConfig())
        # Prospective should start after max(dev ∪ holdout) = 2019
        assert 2020 not in policy.prospective_years  # COVID excluded
        assert 2021 in policy.prospective_years


# ───────────────────────────────────────────────────────────────────────
# 2. Freeze-before-predict gate tests
# ───────────────────────────────────────────────────────────────────────

class TestFreezeGate:
    """Tests for require_freeze_for_season."""

    def test_missing_freeze_strict_raises(self):
        class FakeConfig:
            freeze_file = None

        with pytest.raises(FreezeRequiredError, match="No freeze artifact"):
            require_freeze_for_season(2026, FakeConfig(), strict=True)

    def test_missing_freeze_nonstrict_returns_dict(self):
        class FakeConfig:
            freeze_file = None

        result = require_freeze_for_season(2026, FakeConfig(), strict=False)
        assert result["verified"] is False
        assert "error" in result

    def test_nonexistent_file_strict_raises(self):
        class FakeConfig:
            freeze_file = "/nonexistent/freeze.json"

        with pytest.raises(FreezeRequiredError, match="No freeze artifact"):
            require_freeze_for_season(2026, FakeConfig(), strict=True)

    def test_explicit_path_overrides_config(self):
        class FakeConfig:
            freeze_file = "/nonexistent/config_freeze.json"

        result = require_freeze_for_season(
            2026, FakeConfig(), freeze_path="/also/nonexistent.json", strict=False,
        )
        assert result["verified"] is False


# ───────────────────────────────────────────────────────────────────────
# 3. Probability metrics tests
# ───────────────────────────────────────────────────────────────────────

class TestProbabilityMetrics:
    """Tests for compute_probability_metrics."""

    def test_perfect_predictions(self):
        probs = np.array([1.0, 0.0, 1.0, 0.0])
        outcomes = np.array([1, 0, 1, 0])
        metrics = compute_probability_metrics(probs, outcomes)
        assert metrics["brier_score"] < 1e-6
        assert metrics["log_loss"] < 0.01  # clipped, not exactly 0

    def test_uniform_predictions(self):
        probs = np.array([0.5, 0.5, 0.5, 0.5])
        outcomes = np.array([1, 0, 1, 0])
        metrics = compute_probability_metrics(probs, outcomes)
        assert abs(metrics["brier_score"] - 0.25) < 1e-6

    def test_all_wrong_predictions(self):
        probs = np.array([0.0, 1.0, 0.0, 1.0])
        outcomes = np.array([1, 0, 1, 0])
        metrics = compute_probability_metrics(probs, outcomes)
        assert metrics["brier_score"] > 0.9

    def test_returns_all_three_metrics(self):
        probs = np.random.RandomState(42).random(100)
        outcomes = (probs > 0.5).astype(float)
        metrics = compute_probability_metrics(probs, outcomes)
        assert "brier_score" in metrics
        assert "log_loss" in metrics
        assert "calibration_error" in metrics

    def test_calibration_error_range(self):
        probs = np.random.RandomState(42).random(100)
        outcomes = np.random.RandomState(42).randint(0, 2, 100).astype(float)
        metrics = compute_probability_metrics(probs, outcomes)
        assert 0.0 <= metrics["calibration_error"] <= 1.0


# ───────────────────────────────────────────────────────────────────────
# 4. Upset metrics tests
# ───────────────────────────────────────────────────────────────────────

class TestUpsetMetrics:
    """Tests for compute_upset_metrics."""

    def test_no_upsets(self):
        """Higher seed always wins, model always predicts higher seed."""
        probs = np.array([0.9, 0.8, 0.7])
        outcomes = np.array([1, 1, 1])
        seed_hi = np.array([1, 2, 3])
        seed_lo = np.array([16, 15, 14])

        results = compute_upset_metrics(probs, outcomes, seed_hi, seed_lo)
        for u in results:
            assert u.n_actual_upsets == 0

    def test_all_upsets_correctly_predicted(self):
        """Lower seed always wins, model predicts lower seed."""
        probs = np.array([0.3, 0.2])
        outcomes = np.array([0, 0])
        seed_hi = np.array([5, 5])
        seed_lo = np.array([12, 12])

        results = compute_upset_metrics(probs, outcomes, seed_hi, seed_lo)
        gap7_bucket = [u for u in results if u.seed_gap == 7]
        assert len(gap7_bucket) == 1
        u = gap7_bucket[0]
        assert u.n_actual_upsets == 2
        assert u.n_predicted_upsets == 2
        assert u.n_correct_upset_picks == 2
        assert u.precision == 1.0
        assert u.recall == 1.0

    def test_upset_metrics_dataclass(self):
        u = UpsetMetrics(seed_gap=5, n_actual_upsets=10, n_predicted_upsets=8, n_correct_upset_picks=6)
        assert u.precision == 6 / 8
        assert u.recall == 6 / 10
        assert u.f1 > 0

    def test_zero_division_safe(self):
        u = UpsetMetrics(seed_gap=1)
        assert u.precision == 0.0
        assert u.recall == 0.0
        assert u.f1 == 0.0

    def test_to_dict_has_all_fields(self):
        u = UpsetMetrics(seed_gap=3, n_actual_upsets=5, n_predicted_upsets=3, n_correct_upset_picks=2)
        d = u.to_dict()
        assert "seed_gap" in d
        assert "precision" in d
        assert "recall" in d
        assert "f1" in d


# ───────────────────────────────────────────────────────────────────────
# 5. Bracket-pool utility tests
# ───────────────────────────────────────────────────────────────────────

class TestBracketPool:
    """Tests for compute_bracket_pool_score."""

    def test_perfect_bracket(self):
        actuals = [
            {"round": "R64", "winner": "Duke"},
            {"round": "R32", "winner": "Duke"},
            {"round": "S16", "winner": "UNC"},
        ]
        picks = [
            {"round": "R64", "winner": "Duke"},
            {"round": "R32", "winner": "Duke"},
            {"round": "S16", "winner": "UNC"},
        ]
        result = compute_bracket_pool_score(picks, actuals)
        assert result.total_points == 10 + 20 + 40
        assert result.efficiency == 1.0

    def test_empty_bracket(self):
        actuals = [{"round": "R64", "winner": "Duke"}]
        picks = [{"round": "R64", "winner": "Kentucky"}]
        result = compute_bracket_pool_score(picks, actuals)
        assert result.total_points == 0

    def test_partial_correct(self):
        actuals = [
            {"round": "R64", "winner": "Duke"},
            {"round": "R64", "winner": "UNC"},
        ]
        picks = [
            {"round": "R64", "winner": "Duke"},
            {"round": "R64", "winner": "Kentucky"},
        ]
        result = compute_bracket_pool_score(picks, actuals)
        assert result.total_points == 10
        assert result.correct_by_round["R64"] == 1

    def test_custom_points(self):
        actuals = [{"round": "R1", "winner": "A"}]
        picks = [{"round": "R1", "winner": "A"}]
        result = compute_bracket_pool_score(picks, actuals, {"R1": 100})
        assert result.total_points == 100

    def test_bracket_pool_result_to_dict(self):
        r = BracketPoolResult(total_points=50, max_possible_points=100)
        d = r.to_dict()
        assert d["efficiency"] == 0.5


# ───────────────────────────────────────────────────────────────────────
# 6. LeaderboardEntry and CanonicalLeaderboard tests
# ───────────────────────────────────────────────────────────────────────

class TestLeaderboardEntry:
    def test_brier_skill_score(self):
        e = LeaderboardEntry(year=2025, evidence_level=2, brier_score=0.2, seed_baseline_brier=0.25)
        assert abs(e.brier_skill_score - 0.2) < 1e-6  # 1 - 0.2/0.25 = 0.2

    def test_to_dict_structure(self):
        e = LeaderboardEntry(year=2025, evidence_level=2, n_games=63,
                             brier_score=0.18, log_loss=0.5, calibration_error=0.03)
        d = e.to_dict()
        assert d["year"] == 2025
        assert "probability_metrics" in d
        assert "brier_score" in d["probability_metrics"]
        assert "log_loss" in d["probability_metrics"]
        assert "calibration_error_ece" in d["probability_metrics"]
        assert "provenance" in d


class TestCanonicalLeaderboard:
    def _make_entry(self, year, level, brier=0.2, n_games=63):
        return LeaderboardEntry(
            year=year, evidence_level=level, n_games=n_games,
            brier_score=brier, log_loss=0.5, calibration_error=0.03,
        )

    def test_add_entry_idempotent(self):
        lb = CanonicalLeaderboard()
        lb.add_entry(self._make_entry(2025, 2, brier=0.20))
        lb.add_entry(self._make_entry(2025, 2, brier=0.18))
        assert len(lb.entries) == 1
        assert lb.entries[0].brier_score == 0.18

    def test_entries_sorted_by_year(self):
        lb = CanonicalLeaderboard()
        lb.add_entry(self._make_entry(2025, 2))
        lb.add_entry(self._make_entry(2023, 3))
        lb.add_entry(self._make_entry(2024, 3))
        assert [e.year for e in lb.entries] == [2023, 2024, 2025]

    def test_by_evidence_level(self):
        lb = CanonicalLeaderboard()
        lb.add_entry(self._make_entry(2023, 3))
        lb.add_entry(self._make_entry(2024, 3))
        lb.add_entry(self._make_entry(2025, 2))
        assert len(lb.by_evidence_level(3)) == 2
        assert len(lb.by_evidence_level(2)) == 1
        assert len(lb.by_evidence_level(1)) == 0

    def test_aggregate_brier(self):
        lb = CanonicalLeaderboard()
        lb.add_entry(self._make_entry(2023, 3, brier=0.20, n_games=63))
        lb.add_entry(self._make_entry(2024, 3, brier=0.22, n_games=63))
        agg = lb.aggregate_brier()
        assert abs(agg - 0.21) < 1e-6  # equal weights

    def test_aggregate_brier_by_level(self):
        lb = CanonicalLeaderboard()
        lb.add_entry(self._make_entry(2023, 3, brier=0.22))
        lb.add_entry(self._make_entry(2025, 2, brier=0.18))
        assert abs(lb.aggregate_brier(3) - 0.22) < 1e-6
        assert abs(lb.aggregate_brier(2) - 0.18) < 1e-6

    def test_summary_output(self):
        lb = CanonicalLeaderboard()
        lb.add_entry(self._make_entry(2025, 2))
        text = lb.summary()
        assert "CANONICAL EVALUATION LEADERBOARD" in text
        assert "QUASI-PROSPECTIVE" in text
        assert "2025" in text

    def test_to_dict_structure(self):
        lb = CanonicalLeaderboard()
        lb.add_entry(self._make_entry(2025, 2))
        d = lb.to_dict()
        assert "entries" in d
        assert "aggregates" in d
        assert "level_2_quasi_prospective" in d["aggregates"]

    def test_roundtrip_to_file(self):
        lb = CanonicalLeaderboard()
        lb.add_entry(self._make_entry(2025, 2, brier=0.185))
        lb.add_entry(self._make_entry(2024, 3, brier=0.210))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            lb.to_file(path)
            lb2 = CanonicalLeaderboard.from_file(path)
            assert len(lb2.entries) == 2
            assert lb2.entries[0].year == 2024
            assert abs(lb2.entries[1].brier_score - 0.185) < 1e-6
        finally:
            os.unlink(path)

    def test_empty_leaderboard_aggregates(self):
        lb = CanonicalLeaderboard()
        assert lb.aggregate_brier() == 0.0
        assert lb.aggregate_log_loss() == 0.0


# ───────────────────────────────────────────────────────────────────────
# 7. build_leaderboard_entry convenience function
# ───────────────────────────────────────────────────────────────────────

class TestBuildLeaderboardEntry:
    def test_basic_entry(self):
        rng = np.random.RandomState(42)
        probs = rng.random(63)
        outcomes = (rng.random(63) > 0.5).astype(float)
        entry = build_leaderboard_entry(
            year=2025,
            probs=probs,
            outcomes=outcomes,
            evidence_level=2,
        )
        assert entry.year == 2025
        assert entry.evidence_level == 2
        assert entry.n_games == 63
        assert entry.brier_score > 0
        assert entry.log_loss > 0
        assert entry.calibration_error >= 0

    def test_with_seed_info(self):
        probs = np.array([0.8, 0.3, 0.6])
        outcomes = np.array([1, 0, 1])
        seed_hi = np.array([1, 5, 3])
        seed_lo = np.array([16, 12, 14])
        entry = build_leaderboard_entry(
            year=2025, probs=probs, outcomes=outcomes,
            seed_higher=seed_hi, seed_lower=seed_lo,
            evidence_level=3,
        )
        assert len(entry.upset_metrics) > 0

    def test_with_bracket(self):
        probs = np.array([0.9])
        outcomes = np.array([1.0])
        picks = [{"round": "R64", "winner": "Duke"}]
        actuals = [{"round": "R64", "winner": "Duke"}]
        entry = build_leaderboard_entry(
            year=2025, probs=probs, outcomes=outcomes,
            bracket_picks=picks, bracket_actuals=actuals,
        )
        assert entry.bracket_pool is not None
        assert entry.bracket_pool.total_points == 10


# ───────────────────────────────────────────────────────────────────────
# 8. Canonical constants tests
# ───────────────────────────────────────────────────────────────────────

class TestCanonicalConstants:
    def test_canonical_dev_years_exclude_covid(self):
        assert 2020 not in CANONICAL_DEV_YEARS

    def test_canonical_holdout_years(self):
        assert CANONICAL_HOLDOUT_YEARS == [2025]

    def test_no_overlap_between_canonical_sets(self):
        assert set(CANONICAL_DEV_YEARS).isdisjoint(set(CANONICAL_HOLDOUT_YEARS))
        assert set(CANONICAL_DEV_YEARS).isdisjoint(set(PROSPECTIVE_CANDIDATE_YEARS))
        assert set(CANONICAL_HOLDOUT_YEARS).isdisjoint(set(PROSPECTIVE_CANDIDATE_YEARS))

    def test_default_bracket_points(self):
        assert DEFAULT_BRACKET_POOL_POINTS["R64"] == 10
        assert DEFAULT_BRACKET_POOL_POINTS["NCG"] == 320
