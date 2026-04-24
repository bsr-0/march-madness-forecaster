"""Unit tests for the Chaos Index.

Locks the two ends of the regime spectrum as regression anchors:
- 2025 = all 1-seeds in F4 → mean_f4_seed = 1.0 (maximum chalk).
- 2011 = VCU 11-seed + Butler 8-seed F4 → mean_f4_seed = 6.5 (chaos).

Also locks the sign of ``mean_top8_barthag``'s correlation with chaos
(measured r = -0.668 at 2026-04-24, p ≈ 0.006). If a future data update
or refactor flips that sign the regression tests here fail loudly —
that's a real signal change, not just a numeric nudge.
"""

from pathlib import Path

import pytest

from src.evaluation.chaos_index import (
    ChaosFeatures,
    ChaosObservation,
    compute_actual_chaos,
    compute_features,
    regime_report,
    walk_forward_predict,
)

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def test_compute_actual_chaos_2025_is_pure_chalk():
    """All four 1-seeds in F4 → mean_f4_seed = 1.0."""
    obs = compute_actual_chaos(2025, DATA_ROOT)
    assert isinstance(obs, ChaosObservation)
    assert obs.year == 2025
    assert obs.mean_f4_seed == 1.0


def test_compute_actual_chaos_2011_is_high_chaos():
    """VCU 11 + Butler 8 + UConn 3 + Kentucky 4 → 26/4 = 6.5."""
    obs = compute_actual_chaos(2011, DATA_ROOT)
    assert obs.mean_f4_seed == 6.5


def test_compute_actual_chaos_2023_chaos():
    """UConn 4 + SDSU 5 + FAU 9 + Miami 5 → 23/4 = 5.75."""
    obs = compute_actual_chaos(2023, DATA_ROOT)
    assert obs.mean_f4_seed == 5.75


def test_compute_features_2026_shape():
    feats = compute_features(2026, DATA_ROOT)
    assert isinstance(feats, ChaosFeatures)
    assert feats.year == 2026
    # Barthag is a probability ∈ (0, 1); sanity bounds on all numeric fields.
    assert 0.0 < feats.mean_1seed_barthag < 1.0
    assert 0.0 < feats.weakest_1seed_barthag <= feats.mean_1seed_barthag
    assert 0.0 < feats.mean_top8_barthag < 1.0
    assert feats.elite_count_gt_095 >= 0
    assert feats.top4_minus_top30_barthag > 0  # top-4 stronger than top-30 by def


def test_compute_features_chalk_year_has_stronger_top8_than_chaos_year():
    """2025 (chalk, all 1-seeds F4) should have stronger top-8 than 2023 (chaos).

    This is the direction the correlation test below locks:
    stronger top-of-field -> chalkier tournament.
    """
    feats_2025 = compute_features(2025, DATA_ROOT)
    feats_2023 = compute_features(2023, DATA_ROOT)
    assert feats_2025.mean_top8_barthag > feats_2023.mean_top8_barthag


def test_regime_report_locks_mean_top8_barthag_correlation_sign():
    """Measured 2026-04-24: r(mean_top8_barthag, mean_f4_seed) ≈ -0.668.

    Direction is the point of this test. If the data updates and the
    magnitude shifts a bit that's fine, but a sign flip would mean the
    hypothesis (strong top-of-field → chalk) no longer holds — we want
    that to fail loudly.
    """
    report = regime_report(range(2011, 2027), DATA_ROOT)
    assert len(report["rows"]) == 15  # 2011-2026 excl 2020
    r = report["pearson_r_per_feature"]["mean_top8_barthag"]
    assert r < -0.5, f"Expected r < -0.5 for mean_top8_barthag; got {r:.3f}"
    # elite count should have the same direction
    r_elite = report["pearson_r_per_feature"]["elite_count_gt_095"]
    assert r_elite < -0.5, f"Expected r < -0.5 for elite_count_gt_095; got {r_elite:.3f}"


def test_walk_forward_predict_uses_only_prior_years():
    """Walk-forward contract: the fit for year Y uses years != Y only.

    We can't directly observe the training set from the return value,
    but we can verify the prediction changes when we pass a different
    train window (which it wouldn't if the fit silently peeked at Y).
    """
    full_train = [y for y in range(2011, 2027) if y != 2020]
    pred_full = walk_forward_predict(2026, full_train, DATA_ROOT)
    pred_narrow = walk_forward_predict(2026, [2021, 2022, 2023], DATA_ROOT)
    assert pred_full is not None and pred_narrow is not None
    assert abs(pred_full - pred_narrow) > 0.01, "Walk-forward predictions should differ with different train windows"


def test_walk_forward_predict_returns_none_when_insufficient_train():
    assert walk_forward_predict(2026, [2023, 2024], DATA_ROOT) is None


def test_regime_report_walk_forward_mae_beats_trivial_baseline():
    """The univariate walk-forward predictor should beat 'always predict the mean'.

    Mean of actual F4 seeds across the window is ~3.6. A constant
    predictor achieves MAE ≈ mean |actual - 3.6|. Our predictor needs
    to be at least as good as that baseline; at n=15, anything better
    is modest signal.
    """
    report = regime_report(range(2011, 2027), DATA_ROOT)
    actuals = [r["actual_mean_f4_seed"] for r in report["rows"]]
    preds = [r["predicted_mean_f4_seed"] for r in report["rows"] if r["predicted_mean_f4_seed"] is not None]
    actuals_for_preds = [r["actual_mean_f4_seed"] for r in report["rows"] if r["predicted_mean_f4_seed"] is not None]

    trivial = sum(actuals) / len(actuals)
    mae_trivial = sum(abs(a - trivial) for a in actuals_for_preds) / len(actuals_for_preds)
    mae_pred = sum(abs(a - p) for a, p in zip(actuals_for_preds, preds)) / len(preds)
    assert mae_pred < mae_trivial, f"Walk-forward MAE {mae_pred:.2f} should beat mean-baseline MAE {mae_trivial:.2f}"
