import numpy as np

from src.prediction.torvik_correction import (
    TorvikCorrectionConfig,
    TorvikCorrectionModel,
    round_to_num,
)


def test_torvik_correction_prediction_is_bounded():
    model = TorvikCorrectionModel(TorvikCorrectionConfig(max_correction=0.10, clip_lo=0.01, clip_hi=0.99))
    # 8 features: intercept, seed_gap, abs_seed_gap, torvik_confidence,
    #             market_prob, market_disagree, elo_disagree, round_num
    model.coef_ = np.asarray([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

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
    """Market prob higher than torvik should produce a valid prediction."""
    torvik = np.asarray([0.60, 0.60], dtype=float)
    seed1 = np.asarray([3, 3], dtype=float)
    seed2 = np.asarray([14, 14], dtype=float)
    outcomes = np.asarray([1.0, 1.0], dtype=float)
    market = np.asarray([0.60, 0.60], dtype=float)

    model = TorvikCorrectionModel(TorvikCorrectionConfig(ridge=0.1, max_correction=0.10))
    model.fit(torvik, seed1, seed2, outcomes, market_probs=market)

    pred_with_market = model.predict_one(0.60, 3, 14, market_prob=0.75)
    pred_no_market = model.predict_one(0.60, 3, 14)

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


def test_missing_elo_falls_back_gracefully():
    """None and zero elo_prob should give the same result as no elo."""
    torvik = np.asarray([0.65], dtype=float)
    seed1 = np.asarray([4], dtype=float)
    seed2 = np.asarray([13], dtype=float)
    outcomes = np.asarray([1.0], dtype=float)

    model = TorvikCorrectionModel(TorvikCorrectionConfig(ridge=1.0, max_correction=0.10))
    model.fit(torvik, seed1, seed2, outcomes)

    pred_none = model.predict_one(0.65, 4, 13, elo_prob=None)
    pred_zero = model.predict_one(0.65, 4, 13, elo_prob=0.0)
    pred_no_arg = model.predict_one(0.65, 4, 13)

    assert pred_none == pred_no_arg
    assert pred_zero == pred_no_arg


def test_round_num_encoding():
    assert round_to_num("R64") == 0.0
    assert round_to_num("R32") == 0.2
    assert round_to_num("S16") == 0.4
    assert round_to_num("E8") == 0.6
    assert round_to_num("FF") == 0.8
    assert round_to_num("CH") == 1.0
    assert round_to_num(None) == 0.0
    assert round_to_num("UNKNOWN") == 0.0


def test_round_feature_accepted_in_fit_and_predict():
    torvik = np.asarray([0.70, 0.55], dtype=float)
    seed1 = np.asarray([1, 5], dtype=float)
    seed2 = np.asarray([16, 12], dtype=float)
    outcomes = np.asarray([1.0, 1.0], dtype=float)
    rounds = np.asarray([round_to_num("R64"), round_to_num("S16")], dtype=float)

    model = TorvikCorrectionModel(TorvikCorrectionConfig(ridge=1.0, max_correction=0.10))
    model.fit(torvik, seed1, seed2, outcomes, round_nums=rounds)

    pred = model.predict_one(0.70, 1, 16, round_num=round_to_num("E8"))
    assert 0.01 <= pred <= 0.99


def test_market_and_elo_coverage_logged_in_training_info():
    """fit_torvik_correction_from_year_records logs market and elo coverage."""
    from src.prediction.torvik_correction import fit_torvik_correction_from_year_records

    year_records = {
        2015: [{"torvik": 0.7, "seed1": 1, "seed2": 16, "outcome": 1.0, "odds": 0.75, "elo": 0.80, "round": "R64"}],
        2016: [{"torvik": 0.6, "seed1": 2, "seed2": 15, "outcome": 1.0, "odds": 0.0, "elo": 0.0}],  # both missing
        2017: [{"torvik": 0.5, "seed1": 8, "seed2": 9, "outcome": 0.0}],  # no keys at all
    }

    model = fit_torvik_correction_from_year_records(year_records, recent_year_start=None)

    assert "market_coverage" in model.training_info_
    assert "elo_coverage" in model.training_info_
    # Only 2015 row has real odds/elo → coverage = 1/3
    assert abs(model.training_info_["market_coverage"] - 1 / 3) < 1e-9
    assert abs(model.training_info_["elo_coverage"] - 1 / 3) < 1e-9
