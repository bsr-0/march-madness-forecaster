import pandas as pd

from src.data.team_name_resolver import TeamNameResolver
from src.exports.kaggle import (
    build_team_id_map,
    generate_predictions,
    parse_kaggle_id,
)


def test_parse_kaggle_id_basic():
    season, team1, team2 = parse_kaggle_id("2026_1104_1234")
    assert season == 2026
    assert team1 == 1104
    assert team2 == 1234


def test_build_team_id_map_resolves_duke():
    resolver = TeamNameResolver()
    mapping = build_team_id_map({1234: "Duke"}, resolver)
    assert mapping[1234] == "duke"


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
