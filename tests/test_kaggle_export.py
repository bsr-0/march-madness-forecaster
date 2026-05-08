import importlib.util
from pathlib import Path

import pandas as pd

from src.data.team_name_resolver import TeamNameResolver
from src.exports.kaggle import (
    build_team_id_map,
    build_team_id_map_with_spellings,
    generate_predictions,
    is_womens_team,
    load_kaggle_spellings,
    parse_kaggle_id,
)

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


def test_run_submission_honors_requested_mode(monkeypatch, tmp_path):
    module = _load_kaggle_submission_module()

    sample_path = tmp_path / "sample.csv"
    output_path = tmp_path / "submission.csv"
    pd.DataFrame({"ID": ["2026_1104_1234"]}).to_csv(sample_path, index=False)

    called = {}

    def fake_load_kaggle_id_mapping():
        return {1104: "duke", 1234: "kentucky"}

    def fake_build_predict_fn(year, mode, clip_lo, clip_hi, pp):
        called["args"] = (year, mode, clip_lo, clip_hi, pp)
        return (lambda _t1, _t2: 0.61), {"mode": mode, "alpha": 0.55, "torvik_stats": {"n_teams": 68}}

    monkeypatch.setattr(module, "load_kaggle_id_mapping", fake_load_kaggle_id_mapping)
    monkeypatch.setattr(module, "_build_submission_predict_fn", fake_build_predict_fn)

    module.run_submission(2026, str(sample_path), str(output_path), "ensemble", 0.01, 0.99, pp=None)

    out = pd.read_csv(output_path)
    assert out["Pred"].iloc[0] == 0.61
    assert called["args"][1] == "ensemble"
