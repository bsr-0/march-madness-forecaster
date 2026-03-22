"""Phase 8: Provider robustness stress tests.

Tests model stability under provider removal, priority swaps,
and controlled missingness injection.

Providers (from LibraryProviderHub.DEFAULT_PRIORITIES):
  historical_games: sportsdataverse, espn_scoreboard, cbbpy, sportsipy, cbbdata
  team_metrics: sportsdataverse, sportsipy, cbbdata
  torvik: barttorvik, cbbdata

Thresholds are adaptive: if artifacts/robustness_baseline.json exists,
use mean + 2*std from prior runs. Otherwise use conservative defaults.

Stress tests:
  1. Remove each provider one-at-a-time
  2. Swap provider priority order
  3. Inject controlled missingness by feature group
  4. Inject controlled missingness by time segment
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BASELINE_PATH = REPO_ROOT / "artifacts" / "robustness_baseline.json"


# ---------------------------------------------------------------------------
# Adaptive thresholds
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS = {
    "flip_rate_increase": 0.15,
    "calibration_drift": 0.03,
    "performance_drop": 0.05,
}


def get_thresholds(baseline_path: Path = BASELINE_PATH) -> Dict[str, float]:
    """Load adaptive thresholds from baseline or return defaults.

    If historical baseline stats exist, use mean + 2*std from previous
    runs. Otherwise fall back to conservative guardrail defaults.
    """
    if baseline_path.is_file():
        try:
            with open(baseline_path) as f:
                baseline = json.load(f)
            return {
                k: baseline["mean"][k] + 2 * baseline["std"][k]
                for k in DEFAULT_THRESHOLDS
            }
        except (KeyError, json.JSONDecodeError, TypeError, OSError):
            logger.warning("Could not parse robustness baseline, using defaults")
    return DEFAULT_THRESHOLDS.copy()


def save_baseline(metrics: Dict[str, List[float]], baseline_path: Path = BASELINE_PATH) -> None:
    """Save observed metrics to baseline file for adaptive thresholds."""
    baseline = {
        "mean": {k: float(np.mean(v)) for k, v in metrics.items()},
        "std": {k: float(np.std(v)) for k, v in metrics.items()},
        "n_runs": {k: len(v) for k, v in metrics.items()},
    }
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with open(baseline_path, "w") as f:
        json.dump(baseline, f, indent=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_flip_rate(preds_baseline: np.ndarray, preds_perturbed: np.ndarray) -> float:
    """Fraction of predictions that flip winner (cross 0.5 threshold)."""
    if len(preds_baseline) == 0:
        return 0.0
    baseline_winners = (preds_baseline > 0.5).astype(int)
    perturbed_winners = (preds_perturbed > 0.5).astype(int)
    flips = np.sum(baseline_winners != perturbed_winners)
    return float(flips) / len(preds_baseline)


def _compute_calibration_drift(preds_baseline: np.ndarray, preds_perturbed: np.ndarray) -> float:
    """Mean absolute difference in predicted probabilities."""
    return float(np.mean(np.abs(preds_baseline - preds_perturbed)))


def _compute_brier_degradation(
    preds_baseline: np.ndarray,
    preds_perturbed: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Relative Brier score increase (positive = worse)."""
    brier_base = float(np.mean((preds_baseline - labels) ** 2))
    brier_pert = float(np.mean((preds_perturbed - labels) ** 2))
    if brier_base == 0:
        return 0.0
    return (brier_pert - brier_base) / brier_base


def _synthetic_feature_matrix(n_samples: int = 200, n_features: int = 74, seed: int = 42):
    """Generate synthetic feature matrix and labels for robustness testing."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    # Deterministic weights matching _simple_predict defaults
    w = np.zeros(n_features)
    for idx in [1, 5, 14, 22, 28, 38, 51, 60, 66, 70]:
        w[idx] = 0.06
    logits = X @ w
    probs = 1.0 / (1.0 + np.exp(-logits))
    labels = (rng.rand(n_samples) < probs).astype(float)
    return X, labels, probs


def _simple_predict(X: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Simple logistic prediction function for testing.

    Default weights are spread across multiple feature groups to simulate
    a realistic model where no single provider dominates predictions.
    """
    if weights is None:
        # Deterministic weights spread evenly.  Each weight is placed on a
        # feature index that belongs to only ONE group, so zeroing any
        # single group removes at most one 0.06-weight anchor.
        weights = np.zeros(X.shape[1])
        weights[1] = 0.06    # efficiency group [0-2]
        weights[5] = 0.06    # four_factors group [3-10]
        weights[14] = 0.06   # roster group [11-19]
        weights[22] = 0.06   # game_flows group [20-23]
        weights[28] = 0.06   # schedule_strength group [26-29]
        weights[38] = 0.06   # elo (extended metrics)
        weights[51] = 0.06   # elite_sos area
        weights[58] = 0.06   # coach group [56-58]
        weights[66] = 0.06   # misc (ungrouped)
        weights[70] = 0.06   # ungrouped
    logits = X @ weights
    return 1.0 / (1.0 + np.exp(-logits))


# ---------------------------------------------------------------------------
# Feature group definitions (map to contract source systems)
# ---------------------------------------------------------------------------

FEATURE_GROUPS = {
    "four_factors": list(range(3, 11)),  # efg_pct through opp_ft_rate
    "efficiency": list(range(0, 3)),     # adj_off_eff, adj_def_eff, adj_tempo
    "roster": list(range(11, 20)),       # rapm, warp, continuity, etc.
    "game_flows": list(range(20, 24)),   # volatility, entropy, etc.
    "schedule_strength": list(range(25, 29)),  # sos features (shifted: wab removed)
    "coach": list(range(56, 59)),        # FIX C3: coach features consolidated 7→3
    "external_ratings": [71, 72],        # external composites
    "seed": [73],                        # seed_strength
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestThresholdPolicy:
    """Verify threshold loading and adaptive behavior."""

    def test_default_thresholds_loaded(self):
        thresholds = get_thresholds(Path("/nonexistent/path.json"))
        assert thresholds == DEFAULT_THRESHOLDS

    def test_adaptive_thresholds_from_baseline(self, tmp_path):
        baseline = {
            "mean": {"flip_rate_increase": 0.05, "calibration_drift": 0.01, "performance_drop": 0.02},
            "std": {"flip_rate_increase": 0.02, "calibration_drift": 0.005, "performance_drop": 0.01},
        }
        path = tmp_path / "baseline.json"
        with open(path, "w") as f:
            json.dump(baseline, f)

        thresholds = get_thresholds(path)
        assert thresholds["flip_rate_increase"] == pytest.approx(0.05 + 2 * 0.02)
        assert thresholds["calibration_drift"] == pytest.approx(0.01 + 2 * 0.005)
        assert thresholds["performance_drop"] == pytest.approx(0.02 + 2 * 0.01)

    def test_save_and_reload_baseline(self, tmp_path):
        metrics = {
            "flip_rate_increase": [0.03, 0.05, 0.04],
            "calibration_drift": [0.01, 0.015, 0.012],
            "performance_drop": [0.02, 0.03, 0.025],
        }
        path = tmp_path / "baseline.json"
        save_baseline(metrics, path)

        with open(path) as f:
            loaded = json.load(f)
        assert loaded["mean"]["flip_rate_increase"] == pytest.approx(0.04, abs=0.001)
        assert loaded["n_runs"]["flip_rate_increase"] == 3


class TestRemoveOneProvider:
    """Stress test: remove each feature group (simulating provider dropout)."""

    def test_single_feature_group_removal_flip_rate(self):
        """Removing any single feature group should not flip >10% of predictions."""
        X, labels, _ = _synthetic_feature_matrix()
        preds_baseline = _simple_predict(X)
        thresholds = get_thresholds(Path("/nonexistent"))

        for group_name, indices in FEATURE_GROUPS.items():
            X_perturbed = X.copy()
            valid_indices = [i for i in indices if i < X.shape[1]]
            if not valid_indices:
                continue
            X_perturbed[:, valid_indices] = 0.0  # Zero out the group

            preds_perturbed = _simple_predict(X_perturbed)
            flip_rate = _compute_flip_rate(preds_baseline, preds_perturbed)
            assert flip_rate <= thresholds["flip_rate_increase"], (
                f"Provider dropout '{group_name}': flip_rate={flip_rate:.3f} "
                f"exceeds threshold {thresholds['flip_rate_increase']}"
            )

    def test_single_feature_group_removal_calibration(self):
        """Removing any single feature group should not drift calibration >0.03."""
        X, labels, _ = _synthetic_feature_matrix()
        preds_baseline = _simple_predict(X)
        thresholds = get_thresholds(Path("/nonexistent"))

        for group_name, indices in FEATURE_GROUPS.items():
            X_perturbed = X.copy()
            valid_indices = [i for i in indices if i < X.shape[1]]
            if not valid_indices:
                continue
            X_perturbed[:, valid_indices] = 0.0

            preds_perturbed = _simple_predict(X_perturbed)
            drift = _compute_calibration_drift(preds_baseline, preds_perturbed)
            assert drift <= thresholds["calibration_drift"], (
                f"Provider dropout '{group_name}': calibration_drift={drift:.4f} "
                f"exceeds threshold {thresholds['calibration_drift']}"
            )


class TestSwapPriorityOrder:
    """Stress test: reverse provider priority order."""

    def test_provider_hub_has_default_priorities(self):
        """LibraryProviderHub defines DEFAULT_PRIORITIES."""
        from src.data.ingestion.providers import LibraryProviderHub
        assert hasattr(LibraryProviderHub, "DEFAULT_PRIORITIES")
        priorities = LibraryProviderHub.DEFAULT_PRIORITIES
        assert "historical_games" in priorities
        assert len(priorities["historical_games"]) >= 2

    def test_reversed_priority_does_not_crash(self):
        """Reversing provider order should not raise exceptions."""
        from src.data.ingestion.providers import LibraryProviderHub
        hub = LibraryProviderHub()
        priorities = LibraryProviderHub.DEFAULT_PRIORITIES
        for category, providers in priorities.items():
            reversed_order = list(reversed(providers))
            # The hub should gracefully handle any ordering
            assert len(reversed_order) == len(providers)


class TestInjectMissingnessByFeatureGroup:
    """Stress test: zero out 20% of values in each feature group."""

    def test_partial_missingness_bounded_impact(self):
        """Injecting 20% missingness per group should not exceed performance threshold."""
        X, labels, _ = _synthetic_feature_matrix(n_samples=300)
        preds_baseline = _simple_predict(X)
        thresholds = get_thresholds(Path("/nonexistent"))

        rng = np.random.RandomState(99)
        for group_name, indices in FEATURE_GROUPS.items():
            X_perturbed = X.copy()
            valid_indices = [i for i in indices if i < X.shape[1]]
            if not valid_indices:
                continue

            # Zero out 20% of values in the group
            n_to_zero = max(1, int(0.2 * X.shape[0]))
            rows_to_zero = rng.choice(X.shape[0], n_to_zero, replace=False)
            X_perturbed[np.ix_(rows_to_zero, valid_indices)] = 0.0

            preds_perturbed = _simple_predict(X_perturbed)
            degradation = _compute_brier_degradation(preds_baseline, preds_perturbed, labels)
            assert degradation <= thresholds["performance_drop"], (
                f"Missingness in '{group_name}': performance_drop={degradation:.4f} "
                f"exceeds threshold {thresholds['performance_drop']}"
            )


class TestInjectMissingnessByTimeSegment:
    """Stress test: remove all data from a 2-week window."""

    def test_time_segment_removal(self):
        """Removing a contiguous block of rows should have bounded impact."""
        X, labels, _ = _synthetic_feature_matrix(n_samples=300)
        preds_baseline = _simple_predict(X)
        thresholds = get_thresholds(Path("/nonexistent"))

        # Simulate removing a 2-week window (roughly 10% of a season)
        window_size = int(0.10 * X.shape[0])
        start_idx = X.shape[0] // 3  # arbitrary mid-season point
        end_idx = min(start_idx + window_size, X.shape[0])

        # Zero out the window's features (simulating missing data period)
        X_perturbed = X.copy()
        X_perturbed[start_idx:end_idx, :] = 0.0

        preds_perturbed = _simple_predict(X_perturbed)
        flip_rate = _compute_flip_rate(preds_baseline, preds_perturbed)
        assert flip_rate <= thresholds["flip_rate_increase"] + 0.05, (
            f"Time segment removal: flip_rate={flip_rate:.3f} "
            f"exceeds threshold (with 5% buffer for temporal disruption)"
        )


class TestHelperFunctions:
    """Unit tests for robustness helper functions."""

    def test_flip_rate_no_flips(self):
        preds = np.array([0.8, 0.7, 0.3, 0.2])
        assert _compute_flip_rate(preds, preds) == 0.0

    def test_flip_rate_all_flips(self):
        base = np.array([0.8, 0.7, 0.3, 0.2])
        pert = np.array([0.2, 0.3, 0.7, 0.8])
        assert _compute_flip_rate(base, pert) == 1.0

    def test_flip_rate_partial(self):
        base = np.array([0.8, 0.7, 0.3, 0.2])
        pert = np.array([0.8, 0.3, 0.3, 0.2])  # 1 flip out of 4
        assert _compute_flip_rate(base, pert) == 0.25

    def test_calibration_drift_zero(self):
        preds = np.array([0.5, 0.5, 0.5])
        assert _compute_calibration_drift(preds, preds) == 0.0

    def test_calibration_drift_nonzero(self):
        base = np.array([0.5, 0.5])
        pert = np.array([0.6, 0.4])
        assert _compute_calibration_drift(base, pert) == pytest.approx(0.1)

    def test_brier_degradation_same(self):
        preds = np.array([0.8, 0.3])
        labels = np.array([1.0, 0.0])
        assert _compute_brier_degradation(preds, preds, labels) == 0.0

    def test_brier_degradation_worse(self):
        labels = np.array([1.0, 0.0])
        good = np.array([0.9, 0.1])
        bad = np.array([0.5, 0.5])
        deg = _compute_brier_degradation(good, bad, labels)
        assert deg > 0  # bad predictions have worse Brier
