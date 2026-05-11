from src.prediction.kaggle_recency import resolve_recent_weighting
from src.prediction.torvik_correction import fit_torvik_correction_from_year_records


def test_resolve_recent_weighting_balances_recent_block_against_older_years():
    info = resolve_recent_weighting(
        [2012, 2013, 2014, 2015, 2016, 2017, 2018],
        recent_year_count=5,
        recent_total_ratio=1.0,
    )

    assert info["mode"] == "recent_block"
    assert info["recent_year_start"] == 2014
    assert info["recent_years"] == [2014, 2015, 2016, 2017, 2018]
    assert info["older_years"] == [2012, 2013]
    assert info["recent_year_weight"] == 0.4


def test_torvik_correction_records_policy_training_metadata():
    year_records = {
        2018: [{"torvik": 0.60, "seed1": 1, "seed2": 16, "outcome": 1.0}],
        2019: [{"torvik": 0.58, "seed1": 2, "seed2": 15, "outcome": 1.0}],
        2021: [{"torvik": 0.52, "seed1": 8, "seed2": 9, "outcome": 1.0}],
        2022: [{"torvik": 0.48, "seed1": 9, "seed2": 8, "outcome": 0.0}],
        2023: [{"torvik": 0.55, "seed1": 4, "seed2": 13, "outcome": 1.0}],
        2024: [{"torvik": 0.45, "seed1": 13, "seed2": 4, "outcome": 0.0}],
    }

    model = fit_torvik_correction_from_year_records(
        year_records,
        recent_year_count=5,
        recent_total_ratio=1.0,
    )

    assert model.training_info_ is not None
    assert model.training_info_["weighting_mode"] == "recent_block"
    assert model.training_info_["recent_year_start"] == 2019
    assert model.training_info_["recent_years"] == [2019, 2021, 2022, 2023, 2024]
    assert model.training_info_["older_years"] == [2018]
    assert model.training_info_["recent_year_weight"] == 0.2
