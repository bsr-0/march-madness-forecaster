import importlib.util
import json
import types
from pathlib import Path

import pandas as pd
import pytest

from src.data.team_name_resolver import TeamNameResolver
from src.exports.kaggle import (
    build_team_id_map,
    build_team_id_map_with_spellings,
    generate_predictions,
    is_womens_team,
    load_kaggle_spellings,
    parse_kaggle_id,
)
from src.prediction.torvik_kaggle import EnsembleKagglePredictor, TarvikKagglePredictor

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_kaggle_submission_module():
    script_path = REPO_ROOT / "scripts" / "kaggle_torvik_submission.py"
    spec = importlib.util.spec_from_file_location("kaggle_torvik_submission_test", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_kaggle_id_basic():
    season, team1, team2 = parse_kaggle_id("2026_1104_1234")
    assert season == 2026
    assert team1 == 1104
    assert team2 == 1234


def test_build_team_id_map_resolves_duke():
    resolver = TeamNameResolver()
    mapping = build_team_id_map({1234: "Duke"}, resolver)
    assert mapping[1234] == "duke"


def test_build_team_id_map_with_spellings_recovers_abbreviated_team_names():
    resolver = TeamNameResolver()
    team_map = {1170: "CS Sacramento"}
    spellings = load_kaggle_spellings(str(REPO_ROOT / "data/kaggle/MTeamSpellings.csv"))
    mapping = build_team_id_map_with_spellings(team_map, resolver, spellings)
    assert mapping[1170] == "sacramento_state"


def test_generate_predictions_unmapped_defaults_to_half():
    df = pd.DataFrame({"ID": ["2026_9999_8888"], "Pred": [0.0]})

    def predict_fn(_t1, _t2):
        raise AssertionError("predict_fn should not be called for unmapped IDs")

    out = generate_predictions(df, id_map={}, predict_fn=predict_fn, season_filter=2026)
    assert out["Pred"].iloc[0] == 0.5
    stats = out.attrs.get("kaggle_export_stats", {})
    assert stats.get("unmapped_rows") == 1


def test_generate_predictions_mapped_uses_predict_fn():
    df = pd.DataFrame({"ID": ["2026_1104_1234"], "Pred": [0.0]})

    def predict_fn(_t1, _t2):
        return 0.73

    out = generate_predictions(
        df,
        id_map={1104: "duke", 1234: "kentucky"},
        predict_fn=predict_fn,
        season_filter=2026,
    )
    assert out["Pred"].iloc[0] == 0.73
    stats = out.attrs.get("kaggle_export_stats", {})
    assert stats.get("mapped_rows") == 1


def test_script_load_kaggle_id_mapping_uses_real_kaggle_team_ids():
    module = _load_kaggle_submission_module()
    mapping = module.load_kaggle_id_mapping()

    assert mapping[1101] == "abilene_christian"
    assert mapping[1103] == "akron"
    assert mapping[1170] == "sacramento_state"
    assert mapping[1192] == "fairleigh_dickinson"
    assert mapping[1394] == "texas_a_m_corpus_christi"


def test_script_kaggle_mapping_covers_all_mens_ids_in_sample_submission():
    module = _load_kaggle_submission_module()
    mapping = module.load_kaggle_id_mapping()
    sample_df = pd.read_csv(REPO_ROOT / "data/kaggle/SampleSubmissionStage2.csv")

    missing = set()
    for raw_id in sample_df["ID"].astype(str):
        _season, team1, team2 = parse_kaggle_id(raw_id)
        if not is_womens_team(team1):
            if team1 not in mapping:
                missing.add(team1)
            if team2 not in mapping:
                missing.add(team2)

    assert not missing


def test_womens_id_mapping_covers_all_womens_ids_in_sample_submission():
    """Every women's TeamID in the sample submission must be mappable."""
    women_teams_path = REPO_ROOT / "data/kaggle/WTeams.csv"
    women_spellings_path = REPO_ROOT / "data/kaggle/WTeamSpellings.csv"
    if not women_teams_path.exists():
        pytest.skip("WTeams.csv not found")

    from src.exports.kaggle import load_kaggle_womens_teams

    women_team_map = load_kaggle_womens_teams(str(women_teams_path))
    women_spellings = load_kaggle_spellings(str(women_spellings_path))
    resolver = TeamNameResolver()
    women_id_map = build_team_id_map_with_spellings(women_team_map, resolver, women_spellings)

    sample_df = pd.read_csv(REPO_ROOT / "data/kaggle/SampleSubmissionStage2.csv")
    missing = set()
    for raw_id in sample_df["ID"].astype(str):
        _season, team1, team2 = parse_kaggle_id(raw_id)
        if is_womens_team(team1):
            if team1 not in women_id_map:
                missing.add(team1)
            if team2 not in women_id_map:
                missing.add(team2)

    assert not missing, f"Unmapped women's TeamIDs: {missing}"


def test_womens_kaggle_end_to_end_seed_only_predictions():
    """Full pipeline: build women's ID map, backfill seeds, verify predictions vary by seed."""
    women_teams_path = REPO_ROOT / "data/kaggle/WTeams.csv"
    women_spellings_path = REPO_ROOT / "data/kaggle/WTeamSpellings.csv"
    if not women_teams_path.exists():
        pytest.skip("WTeams.csv not found")

    from src.exports.kaggle import load_kaggle_womens_teams
    from src.pipeline.womens import WomensPipeline, WomensPipelineConfig

    # Build women's ID map from real Kaggle data
    women_team_map = load_kaggle_womens_teams(str(women_teams_path))
    women_spellings = load_kaggle_spellings(str(women_spellings_path))
    resolver = TeamNameResolver()
    women_id_map = build_team_id_map_with_spellings(women_team_map, resolver, women_spellings)

    # At least ~350 women's teams should be mapped
    assert len(women_id_map) >= 350, f"Only {len(women_id_map)} women's teams mapped"

    # Run pipeline in seed-only mode (no cached stats)
    pipeline = WomensPipeline(WomensPipelineConfig(seed_only_mode=True))
    pipeline.run()

    # Use 2025 seeded teams (known seed data) to build test matchups
    # W01=3376 (1-seed), W16=3399 (16-seed), W08=3428 (8-seed)
    seed_1_id = women_id_map.get(3376, "w_team_3376")
    seed_16_id = women_id_map.get(3399, "w_team_3399")
    seed_8_id = women_id_map.get(3428, "w_team_3428")

    pipeline.set_team_seeds({seed_1_id: 1, seed_16_id: 16, seed_8_id: 8})

    # Construct sample rows with these teams
    sample_df = pd.DataFrame(
        {
            "ID": ["2025_3376_3399", "2025_3376_3428", "2025_3399_3428"],
            "Pred": [0.0, 0.0, 0.0],
        }
    )

    out_df = generate_predictions(
        sample_df,
        id_map={},
        predict_fn=lambda t1, t2: 0.5,
        season_filter=2025,
        womens_id_map=women_id_map,
        womens_predict_fn=pipeline.predict_probability,
    )

    stats = out_df.attrs.get("kaggle_export_stats", {})
    preds = out_df["Pred"].tolist()

    assert stats.get("womens_mapped", 0) == 3, f"Expected 3 mapped, got {stats.get('womens_mapped')}"
    assert stats.get("predict_failures", 0) == 0

    # 1-seed vs 16-seed should heavily favor the 1-seed
    assert preds[0] > 0.80, f"1v16 prediction {preds[0]:.3f} should be > 0.80"
    # 1-seed vs 8-seed should still favor the 1-seed
    assert preds[1] > 0.55, f"1v8 prediction {preds[1]:.3f} should be > 0.55"
    # All bounded
    for p in preds:
        assert 0.005 <= p <= 0.995


def test_womens_submission_support_prefers_raw_kaggle_ids_when_pipeline_has_stats():
    module = _load_kaggle_submission_module()
    sample_df = pd.DataFrame({"ID": ["2025_3376_3399", "2025_3428_3435"]})

    women_id_map, women_predict_fn, women_info = module._build_womens_submission_support(sample_df, 2025)

    assert women_predict_fn is not None
    assert women_id_map is not None
    assert women_info is not None
    assert women_id_map[3376] == "3376"
    assert women_id_map[3399] == "3399"


def test_run_submission_honors_requested_mode(monkeypatch, tmp_path):
    module = _load_kaggle_submission_module()

    sample_path = tmp_path / "sample.csv"
    output_path = tmp_path / "submission.csv"
    pd.DataFrame({"ID": ["2026_1104_1234"]}).to_csv(sample_path, index=False)

    called = {}

    def fake_load_kaggle_id_mapping():
        return {1104: "duke", 1234: "kentucky"}

    def fake_build_predict_fn(
        year,
        mode,
        clip_lo,
        clip_hi,
        pp,
        ensemble_recent_year_start=2021,
        ensemble_recent_year_weight=2.0,
        ensemble_recent_year_count=None,
        ensemble_recent_total_ratio=1.0,
        torvik_correction_ridge=5.0,
        torvik_correction_max_correction=0.10,
    ):
        called["args"] = (
            year,
            mode,
            clip_lo,
            clip_hi,
            pp,
            ensemble_recent_year_start,
            ensemble_recent_year_weight,
            ensemble_recent_year_count,
            ensemble_recent_total_ratio,
            torvik_correction_ridge,
            torvik_correction_max_correction,
        )
        return (lambda _t1, _t2: 0.61), {"mode": mode, "alpha": 0.55, "torvik_stats": {"n_teams": 68}}

    def fake_womens_support(
        sample_df,
        year,
        womens_model_weight=None,
        womens_seed_weight=None,
        womens_weight_step=0.1,
        refresh_womens_weight_cache=False,
    ):
        called["womens_args"] = (
            year,
            womens_model_weight,
            womens_seed_weight,
            womens_weight_step,
            refresh_womens_weight_cache,
        )
        return None, None, None

    monkeypatch.setattr(module, "load_kaggle_id_mapping", fake_load_kaggle_id_mapping)
    monkeypatch.setattr(module, "_build_submission_predict_fn", fake_build_predict_fn)
    monkeypatch.setattr(module, "_build_womens_submission_support", fake_womens_support)

    module.run_submission(2026, str(sample_path), str(output_path), "ensemble", 0.01, 0.99, pp=None)

    out = pd.read_csv(output_path)
    assert out["Pred"].iloc[0] == 0.61
    assert called["args"][1] == "ensemble"


def test_walk_forward_alpha_upweights_recent_years(tmp_path):
    def make_records(team_prefix: str, torvik: float, outcome: float, count: int):
        return [
            {
                "team1": f"{team_prefix}_a_{idx}",
                "team2": f"{team_prefix}_b_{idx}",
                "torvik": torvik,
                "seed": 0.5,
                "outcome": outcome,
            }
            for idx in range(count)
        ]

    def make_pipeline_records(team_prefix: str, pipeline: float, count: int):
        return [
            {"team1": f"{team_prefix}_a_{idx}", "team2": f"{team_prefix}_b_{idx}", "pipeline": pipeline}
            for idx in range(count)
        ]

    pergame = {
        "2017": make_records("old1", torvik=0.2, outcome=1.0, count=20),
        "2018": make_records("old2", torvik=0.2, outcome=1.0, count=20),
        "2019": make_records("old3", torvik=0.2, outcome=1.0, count=20),
        "2021": make_records("recent", torvik=0.9, outcome=1.0, count=20),
    }
    backtest = {
        "per_year_games": {
            "2017": make_pipeline_records("old1", pipeline=0.8, count=20),
            "2018": make_pipeline_records("old2", pipeline=0.8, count=20),
            "2019": make_pipeline_records("old3", pipeline=0.8, count=20),
            "2021": make_pipeline_records("recent", pipeline=0.1, count=20),
        }
    }

    pergame_path = tmp_path / "pergame.json"
    backtest_path = tmp_path / "backtest.json"
    pergame_path.write_text(json.dumps(pergame), encoding="utf-8")
    backtest_path.write_text(json.dumps(backtest), encoding="utf-8")

    torvik = TarvikKagglePredictor({}, 2025)

    def pipeline_predict_fn(_t1, _t2):
        return 0.5

    unweighted = EnsembleKagglePredictor.from_walk_forward(
        torvik,
        pipeline_predict_fn,
        2025,
        pergame_path=str(pergame_path),
        backtest_path=str(backtest_path),
        recent_year_start=None,
        recent_year_weight=1.0,
    )
    weighted = EnsembleKagglePredictor.from_walk_forward(
        torvik,
        pipeline_predict_fn,
        2025,
        pergame_path=str(pergame_path),
        backtest_path=str(backtest_path),
        recent_year_start=2021,
        recent_year_weight=4.0,
    )

    assert unweighted.alpha == pytest.approx(0.2)
    assert weighted.alpha == pytest.approx(0.7)
    assert weighted.alpha > unweighted.alpha


def test_walk_forward_alpha_supports_recent_block_policy(tmp_path):
    def make_records(team_prefix: str, torvik: float, outcome: float, count: int):
        return [
            {
                "team1": f"{team_prefix}_a_{idx}",
                "team2": f"{team_prefix}_b_{idx}",
                "torvik": torvik,
                "seed": 0.5,
                "outcome": outcome,
            }
            for idx in range(count)
        ]

    def make_pipeline_records(team_prefix: str, pipeline: float, count: int):
        return [
            {"team1": f"{team_prefix}_a_{idx}", "team2": f"{team_prefix}_b_{idx}", "pipeline": pipeline}
            for idx in range(count)
        ]

    pergame = {
        "2017": make_records("old1", torvik=0.2, outcome=1.0, count=20),
        "2018": make_records("old2", torvik=0.2, outcome=1.0, count=20),
        "2019": make_records("old3", torvik=0.2, outcome=1.0, count=20),
        "2021": make_records("recent", torvik=0.9, outcome=1.0, count=20),
    }
    backtest = {
        "per_year_games": {
            "2017": make_pipeline_records("old1", pipeline=0.8, count=20),
            "2018": make_pipeline_records("old2", pipeline=0.8, count=20),
            "2019": make_pipeline_records("old3", pipeline=0.8, count=20),
            "2021": make_pipeline_records("recent", pipeline=0.1, count=20),
        }
    }

    pergame_path = tmp_path / "pergame.json"
    backtest_path = tmp_path / "backtest.json"
    pergame_path.write_text(json.dumps(pergame), encoding="utf-8")
    backtest_path.write_text(json.dumps(backtest), encoding="utf-8")

    torvik = TarvikKagglePredictor({}, 2025)

    policy = EnsembleKagglePredictor.from_walk_forward(
        torvik,
        lambda _t1, _t2: 0.5,
        2025,
        pergame_path=str(pergame_path),
        backtest_path=str(backtest_path),
        recent_year_count=1,
        recent_total_ratio=1.0,
    )
    explicit = EnsembleKagglePredictor.from_walk_forward(
        torvik,
        lambda _t1, _t2: 0.5,
        2025,
        pergame_path=str(pergame_path),
        backtest_path=str(backtest_path),
        recent_year_start=2021,
        recent_year_weight=3.0,
    )

    assert policy.alpha == explicit.alpha
    assert policy.alpha_training_info["weighting_mode"] == "recent_block"
    assert policy.alpha_training_info["recent_year_weight"] == pytest.approx(3.0)


def test_build_submission_predict_fn_forwards_recency_weighting(monkeypatch):
    module = _load_kaggle_submission_module()
    captured = {}

    class DummyEnsemble:
        def __init__(self):
            self.alpha = 0.65
            self.alpha_training_info = {"recent_year_start": 2021, "recent_year_weight": 3.0}

        def predict(self, _t1, _t2):
            return 0.64

    def fake_load_tournament_seeds(_year):
        return {"duke": 1, "kentucky": 2}

    def fake_from_year(year, seeds=None, clip_lo=0.01, clip_hi=0.99):
        return TarvikKagglePredictor({"duke": 0.9, "kentucky": 0.8}, year, clip_lo=clip_lo, clip_hi=clip_hi)

    def fake_load_pipeline_lookup(_year):
        return {"duke_kentucky": 0.55}

    def fake_from_walk_forward(torvik, pipeline_predict_fn, year, **kwargs):
        captured["year"] = year
        captured["kwargs"] = kwargs
        assert pipeline_predict_fn("duke", "kentucky") == 0.55
        return DummyEnsemble()

    monkeypatch.setattr(module, "load_tournament_seeds", fake_load_tournament_seeds)
    monkeypatch.setattr(module.TarvikKagglePredictor, "from_year", fake_from_year)
    monkeypatch.setattr(module, "_load_pipeline_lookup", fake_load_pipeline_lookup)
    monkeypatch.setattr(module.EnsembleKagglePredictor, "from_walk_forward", fake_from_walk_forward)

    predict_fn, info = module._build_submission_predict_fn(
        2026,
        "ensemble",
        0.01,
        0.99,
        pp=None,
        ensemble_recent_year_start=2021,
        ensemble_recent_year_weight=3.0,
    )

    assert captured["year"] == 2026
    assert captured["kwargs"]["recent_year_start"] == 2021
    assert captured["kwargs"]["recent_year_weight"] == 3.0
    assert predict_fn("duke", "kentucky") == 0.64
    assert info["alpha"] == 0.65


def test_build_submission_predict_fn_supports_recent_block_weighting(monkeypatch):
    module = _load_kaggle_submission_module()
    captured = {}

    class DummyEnsemble:
        def __init__(self):
            self.alpha = 0.61
            self.alpha_training_info = {
                "weighting_mode": "recent_block",
                "recent_year_count": 5,
                "recent_total_ratio": 1.0,
                "recent_year_weight": 2.4,
            }

        def predict(self, _t1, _t2):
            return 0.62

    monkeypatch.setattr(module, "load_tournament_seeds", lambda _year: {"duke": 1, "kentucky": 2})
    monkeypatch.setattr(
        module.TarvikKagglePredictor, "from_year", lambda *args, **kwargs: TarvikKagglePredictor({}, 2026)
    )
    monkeypatch.setattr(module, "_load_pipeline_lookup", lambda _year: {"duke_kentucky": 0.55})

    def fake_from_walk_forward(_torvik, _pipe_predict, _year, **kwargs):
        captured["kwargs"] = kwargs
        return DummyEnsemble()

    monkeypatch.setattr(module.EnsembleKagglePredictor, "from_walk_forward", fake_from_walk_forward)

    predict_fn, info = module._build_submission_predict_fn(
        2026,
        "ensemble",
        0.01,
        0.99,
        ensemble_recent_year_count=5,
        ensemble_recent_total_ratio=1.0,
    )

    assert captured["kwargs"]["recent_year_count"] == 5
    assert captured["kwargs"]["recent_total_ratio"] == 1.0
    assert predict_fn("duke", "kentucky") == 0.62
    assert info["alpha_training_info"]["weighting_mode"] == "recent_block"


def test_build_submission_predict_fn_supports_torvik_corrected(monkeypatch):
    module = _load_kaggle_submission_module()

    class DummyCorrection:
        coef_ = [0.1, -0.2, 0.3, -0.4]

        def predict_one(self, base_prob, seed1, seed2, market_prob=None, elo_prob=None, round_num=0.0):
            assert base_prob == 0.6
            assert seed1 == 1
            assert seed2 == 2
            return 0.66

    def fake_load_tournament_seeds(_year):
        return {"duke": 1, "kentucky": 2}

    class DummyTorvik:
        def predict(self, _t1, _t2):
            return 0.6

        def stats(self):
            return {"n_teams": 68}

    def fake_from_year(year, seeds=None, clip_lo=0.01, clip_hi=0.99):
        return DummyTorvik()

    monkeypatch.setattr(module, "load_tournament_seeds", fake_load_tournament_seeds)
    monkeypatch.setattr(module.TarvikKagglePredictor, "from_year", fake_from_year)
    monkeypatch.setattr(module, "_build_torvik_correction_model", lambda *args, **kwargs: DummyCorrection())
    monkeypatch.setattr(module, "load_market_ratings", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "load_elo_barthag", lambda *args, **kwargs: None)

    predict_fn, info = module._build_submission_predict_fn(
        2026,
        "torvik_corrected",
        0.01,
        0.99,
        pp=None,
        ensemble_recent_year_start=2021,
        ensemble_recent_year_weight=3.0,
    )

    assert predict_fn("duke", "kentucky") == 0.66
    assert info["mode"] == "torvik_corrected"
    assert info["correction_training_info"]["recent_year_start"] == 2021
    assert info["correction_training_info"]["recent_year_weight"] == 3.0


def test_build_submission_predict_fn_supports_recent5_conservative_preset(monkeypatch):
    module = _load_kaggle_submission_module()
    captured = {}

    class DummyCorrection:
        coef_ = [0.1, -0.2, 0.3, -0.4]
        training_info_ = {
            "weighting_mode": "recent_block",
            "recent_year_count": 5,
            "recent_total_ratio": 1.0,
            "recent_year_weight": 2.4,
        }

        def predict_one(self, base_prob, seed1, seed2, market_prob=None, elo_prob=None, round_num=0.0):
            assert base_prob == 0.6
            assert seed1 == 1
            assert seed2 == 2
            return 0.64

    monkeypatch.setattr(module, "load_tournament_seeds", lambda _year: {"duke": 1, "kentucky": 2})

    class DummyTorvik:
        def predict(self, _t1, _t2):
            return 0.6

        def stats(self):
            return {"n_teams": 68}

    monkeypatch.setattr(module.TarvikKagglePredictor, "from_year", lambda *args, **kwargs: DummyTorvik())

    def fake_build_torvik_correction_model(year, **kwargs):
        captured["year"] = year
        captured["kwargs"] = kwargs
        return DummyCorrection()

    monkeypatch.setattr(module, "_build_torvik_correction_model", fake_build_torvik_correction_model)
    monkeypatch.setattr(module, "load_market_ratings", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "load_elo_barthag", lambda *args, **kwargs: None)

    predict_fn, info = module._build_submission_predict_fn(
        2026,
        module.TORVIK_CORRECTED_RECENT5_CONSERVATIVE,
        0.01,
        0.99,
    )

    assert captured["year"] == 2026
    assert captured["kwargs"]["recent_year_count"] == 5
    assert captured["kwargs"]["recent_total_ratio"] == 1.0
    assert captured["kwargs"]["correction_ridge"] == 5.0
    assert captured["kwargs"]["max_correction"] == 0.06
    assert predict_fn("duke", "kentucky") == 0.64
    assert info["mode"] == module.TORVIK_CORRECTED_RECENT5_CONSERVATIVE
    assert info["base_mode"] == "torvik_corrected"


def test_womens_submission_support_uses_walk_forward_weights_by_default(monkeypatch):
    module = _load_kaggle_submission_module()
    sample_df = pd.DataFrame({"ID": ["2026_3376_3399"]})

    import src.pipeline.womens as womens_module

    captured = {}

    class FakePipeline:
        def __init__(self, config):
            captured["config"] = config
            self.feature_engineer = types.SimpleNamespace(team_features={"3376": object(), "3399": object()})
            self.team_stats = {"3376": object(), "3399": object()}

        def run(self):
            return {"status": "ready"}

        def set_team_seeds(self, seed_map):
            captured["seed_map"] = dict(seed_map)

        def predict_probability(self, _team1, _team2):
            return 0.5

    monkeypatch.setattr(module, "_get_cached_womens_weights_for_year", lambda year, step, **kwargs: (1.0, 0.0, 0.12))
    monkeypatch.setattr(womens_module, "WomensPipeline", FakePipeline)

    women_id_map, women_predict_fn, info = module._build_womens_submission_support(sample_df, 2026)

    assert women_predict_fn is not None
    assert women_id_map[3376] == "3376"
    assert captured["config"].model_weight == 1.0
    assert captured["config"].seed_weight == 0.0
    assert info["model_weight"] == 1.0
    assert info["seed_weight"] == 0.0


def test_womens_submission_support_returns_full_tuple_for_mens_only_sample(monkeypatch):
    module = _load_kaggle_submission_module()
    sample_df = pd.DataFrame({"ID": ["2026_1104_1234"]})

    monkeypatch.setattr(module, "REPO_ROOT", Path("/tmp/nonexistent-kaggle-root"))

    women_id_map, women_predict_fn, info = module._build_womens_submission_support(sample_df, 2026)

    assert women_id_map is None
    assert women_predict_fn is None
    assert info is None


def test_womens_submission_support_fails_loudly_when_womens_rows_need_missing_files(monkeypatch):
    module = _load_kaggle_submission_module()
    sample_df = pd.DataFrame({"ID": ["2026_3376_3399"]})

    monkeypatch.setattr(module, "REPO_ROOT", Path("/tmp/nonexistent-kaggle-root"))

    with pytest.raises(RuntimeError, match="Women's rows were detected"):
        module._build_womens_submission_support(sample_df, 2026)


def test_get_cached_womens_weights_force_refresh_rebuilds_cache(monkeypatch):
    module = _load_kaggle_submission_module()

    monkeypatch.setattr(
        module, "_load_womens_weight_cache", lambda _step: {"2026": {"model_weight": 0.2, "seed_weight": 0.8}}
    )

    called = {}

    def fake_build_cache(years, step, refresh=False):
        called["args"] = (list(years), step, refresh)
        return {"2026": {"model_weight": 0.6, "seed_weight": 0.4, "train_brier": 0.12}}

    monkeypatch.setattr(module, "build_womens_weight_cache", fake_build_cache)

    model_weight, seed_weight, train_brier = module._get_cached_womens_weights_for_year(2026, 0.1, force_refresh=True)

    assert called["args"] == ([2026], 0.1, True)
    assert model_weight == 0.6
    assert seed_weight == 0.4
    assert train_brier == 0.12
