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
