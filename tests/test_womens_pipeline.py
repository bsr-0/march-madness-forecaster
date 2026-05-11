import pytest

from src.pipeline.womens import WomensPipeline, WomensPipelineConfig


def test_womens_pipeline_default_config():
    config = WomensPipelineConfig()
    assert config.year == 2026
    assert config.clip_lo == 0.005
    assert config.tournament_shrinkage == 0.02


def test_womens_pipeline_rejects_bad_probability_profile():
    with pytest.raises(ValueError):
        WomensPipelineConfig(probability_profile="bad")


def test_womens_pipeline_seed_only_predictions_are_bounded():
    pipeline = WomensPipeline(WomensPipelineConfig(seed_only_mode=True))
    report = pipeline.run()
    assert report["status"] == "ready"

    pipeline.set_team_seeds({"uconn": 1, "fairleigh": 16})
    prob = pipeline.predict_probability("uconn", "fairleigh")

    assert 0.005 <= prob <= 0.995
    assert prob > 0.80


def test_womens_pipeline_unknown_teams_fall_back_near_half():
    pipeline = WomensPipeline(WomensPipelineConfig(seed_only_mode=True))
    pipeline.run()
    prob = pipeline.predict_probability("unknown_a", "unknown_b")
    assert 0.30 <= prob <= 0.70


def test_womens_pipeline_symmetry_in_seed_only_mode():
    pipeline = WomensPipeline(WomensPipelineConfig(seed_only_mode=True))
    pipeline.run()
    pipeline.set_team_seeds({"team_a": 3, "team_b": 14})

    p_ab = pipeline.predict_probability("team_a", "team_b")
    p_ba = pipeline.predict_probability("team_b", "team_a")
    assert abs((p_ab + p_ba) - 1.0) < 0.15


def test_womens_pipeline_backfills_missing_feature_rows_from_seeds():
    pipeline = WomensPipeline(WomensPipelineConfig(seed_only_mode=True))
    pipeline.run()
    pipeline.set_team_seeds({"w_team_3101": 1, "w_team_3481": 16})

    assert "w_team_3101" in pipeline.feature_engineer.team_features
    assert "w_team_3481" in pipeline.feature_engineer.team_features


def test_womens_pipeline_loads_kaggle_regular_season_fallback():
    pipeline = WomensPipeline(WomensPipelineConfig(year=2026, seed_only_mode=True))
    report = pipeline.run()

    assert report["status"] == "ready"
    assert report["kaggle_teams_loaded"] >= 300
    assert report["features_built"] >= 300

    ranked = sorted(pipeline.team_stats.values(), key=lambda stats: stats.adj_efficiency_margin, reverse=True)
    assert ranked[0].seed <= ranked[-1].seed

    prob = pipeline.predict_probability(ranked[0].team_id, ranked[-1].team_id)
    assert prob > 0.70
