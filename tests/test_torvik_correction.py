import numpy as np

from src.prediction.torvik_correction import TorvikCorrectionConfig, TorvikCorrectionModel


def test_torvik_correction_prediction_is_bounded():
    model = TorvikCorrectionModel(TorvikCorrectionConfig(max_correction=0.10, clip_lo=0.01, clip_hi=0.99))
    # 6 features: intercept, seed_gap, abs_seed_gap, torvik_confidence, market_prob, market_disagreement
    model.coef_ = np.asarray([0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

    pred = model.predict_one(0.96, 1, 16)

    assert 0.01 <= pred <= 0.99
    assert pred == 0.99


def test_torvik_correction_fit_learns_seed_gap_direction():
    torvik = np.asarray([0.55, 0.55, 0.45, 0.45], dtype=float)
    seed1 = np.asarray([1, 2, 12, 13], dtype=float)
    seed2 = np.asarray([16, 15, 5, 4], dtype=float)
    outcomes = np.asarray([1.0, 1.0, 0.0, 0.0], dtype=float)

    model = TorvikCorrectionModel(TorvikCorrectionConfig(ridge=0.1, max_correction=0.10))
    model.fit(torvik, seed1, seed2, outcomes)

    strong_favorite = model.predict_one(0.55, 1, 16)
    weak_underdog = model.predict_one(0.45, 13, 4)

    assert strong_favorite > 0.55
    assert weak_underdog < 0.45


def test_torvik_correction_requires_fit_before_predict():
    model = TorvikCorrectionModel()

    try:
        model.predict_one(0.5, 8, 9)
    except ValueError as exc:
        assert "fit" in str(exc)
    else:
        raise AssertionError("predict_one should require fit() first")


def test_market_disagreement_shifts_prediction():
    """Market prob higher than torvik should nudge prediction upward."""
    torvik = np.asarray([0.60, 0.60], dtype=float)
    seed1 = np.asarray([3, 3], dtype=float)
    seed2 = np.asarray([14, 14], dtype=float)
    outcomes = np.asarray([1.0, 1.0], dtype=float)
    # Market agrees with torvik in training
    market = np.asarray([0.60, 0.60], dtype=float)

    model = TorvikCorrectionModel(TorvikCorrectionConfig(ridge=0.1, max_correction=0.10))
    model.fit(torvik, seed1, seed2, outcomes, market_probs=market)

    # At predict time, market is MORE confident than torvik
    pred_with_market = model.predict_one(0.60, 3, 14, market_prob=0.75)
    pred_no_market = model.predict_one(0.60, 3, 14)

    # Both should be valid probabilities
    assert 0.01 <= pred_with_market <= 0.99
    assert 0.01 <= pred_no_market <= 0.99


def test_missing_market_falls_back_gracefully():
    """None and zero market_prob should give the same result as no market."""
    torvik = np.asarray([0.70], dtype=float)
    seed1 = np.asarray([2], dtype=float)
    seed2 = np.asarray([15], dtype=float)
    outcomes = np.asarray([1.0], dtype=float)

    model = TorvikCorrectionModel(TorvikCorrectionConfig(ridge=1.0, max_correction=0.10))
    model.fit(torvik, seed1, seed2, outcomes)

    pred_none = model.predict_one(0.70, 2, 15, market_prob=None)
    pred_zero = model.predict_one(0.70, 2, 15, market_prob=0.0)
    pred_no_arg = model.predict_one(0.70, 2, 15)

    assert pred_none == pred_no_arg
    assert pred_zero == pred_no_arg


def test_market_coverage_logged_in_training_info():
    """fit_torvik_correction_from_year_records records market coverage fraction."""
    from src.prediction.torvik_correction import fit_torvik_correction_from_year_records

    year_records = {
        2015: [{"torvik": 0.7, "seed1": 1, "seed2": 16, "outcome": 1.0, "odds": 0.75}],
        2016: [{"torvik": 0.6, "seed1": 2, "seed2": 15, "outcome": 1.0, "odds": 0.0}],  # missing
        2017: [{"torvik": 0.5, "seed1": 8, "seed2": 9, "outcome": 0.0}],  # no odds key
    }

    model = fit_torvik_correction_from_year_records(year_records, recent_year_start=None)

    assert "market_coverage" in model.training_info_
    # Only 2015 row has real odds; 2016 and 2017 fall back → coverage = 1/3
    assert abs(model.training_info_["market_coverage"] - 1 / 3) < 1e-9
