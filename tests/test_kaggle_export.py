"""Kaggle submission format: ID parsing, gender routing, defaults, validation."""

import logging

import pandas as pd
import pytest

from src.exports.kaggle import (
    DEFAULT_PRED,
    generate_predictions,
    is_womens_team,
    load_kaggle_teams,
    parse_kaggle_id,
    validate_submission,
)


def test_parse_kaggle_id():
    assert parse_kaggle_id("2026_1101_1102") == (2026, 1101, 1102)
    assert parse_kaggle_id(" 2026_3101_3102 ") == (2026, 3101, 3102)


@pytest.mark.parametrize("bad", [None, "", "2026_1101", "2026-1101-1102", "abcd_1101_1102", "2026_1101_1102_1"])
def test_parse_kaggle_id_rejects(bad):
    with pytest.raises(ValueError):
        parse_kaggle_id(bad)


def test_is_womens_team_boundary():
    assert not is_womens_team(2999)
    assert is_womens_team(3000)
    assert is_womens_team(3481)


def _sample(*ids):
    return pd.DataFrame({"ID": list(ids), "Pred": [0.5] * len(ids)})


def test_routes_by_team_id_range_and_defaults_unknown_pairs():
    sample = _sample("2026_1101_1102", "2026_1101_1103", "2026_3101_3102", "2026_3101_3103")
    mens = {(1101, 1102): 0.7}
    womens = {(3101, 3102): 0.2}
    out = generate_predictions(sample, lambda a, b: mens.get((a, b)), lambda a, b: womens.get((a, b)))
    assert out["Pred"].tolist() == [0.7, DEFAULT_PRED, 0.2, DEFAULT_PRED]
    stats = out.attrs["kaggle_export_stats"]
    assert stats["mens_rows"] == 2 and stats["mens_predicted"] == 1
    assert stats["womens_rows"] == 2 and stats["womens_predicted"] == 1
    assert stats["predicted_rows"] == 2 and stats["defaulted_rows"] == 2


def test_missing_callback_leaves_that_gender_at_default():
    sample = _sample("2026_1101_1102", "2026_3101_3102")
    out = generate_predictions(sample, lambda a, b: 0.9, None)
    assert out["Pred"].tolist() == [0.9, DEFAULT_PRED]


def test_season_filter_defaults_other_seasons():
    sample = _sample("2025_1101_1102", "2026_1101_1102")
    out = generate_predictions(sample, lambda a, b: 0.9, None, season_filter=2026)
    assert out["Pred"].tolist() == [DEFAULT_PRED, 0.9]
    assert out.attrs["kaggle_export_stats"]["season_mismatch"] == 1


def test_bad_ids_and_failing_callbacks_are_counted_not_raised(caplog):
    def boom(a, b):
        raise RuntimeError("no")

    sample = _sample("garbage", "2026_1101_1102")
    with caplog.at_level(logging.WARNING):
        out = generate_predictions(sample, boom, None)
    assert out["Pred"].tolist() == [DEFAULT_PRED, DEFAULT_PRED]
    stats = out.attrs["kaggle_export_stats"]
    assert stats["bad_id_rows"] == 1 and stats["predict_failures"] == 1


def test_predictions_are_clipped_to_unit_interval():
    sample = _sample("2026_1101_1102", "2026_1101_1103")
    out = generate_predictions(sample, lambda a, b: 1.7 if b == 1102 else -0.2, None)
    assert out["Pred"].tolist() == [1.0, 0.0]


def test_requires_id_column():
    with pytest.raises(ValueError):
        generate_predictions(pd.DataFrame({"Pred": [0.5]}), None, None)


def test_input_frame_is_not_mutated():
    sample = _sample("2026_1101_1102")
    generate_predictions(sample, lambda a, b: 0.9, None)
    assert sample["Pred"].tolist() == [0.5]


def test_validate_submission_accepts_matching_frame():
    sample = _sample("2026_1101_1102", "2026_1101_1103")
    out = generate_predictions(sample, lambda a, b: 0.6, None)
    validate_submission(out[["ID", "Pred"]], sample)


@pytest.mark.parametrize(
    "mutate, msg",
    [
        (lambda df: df.rename(columns={"Pred": "pred"}), "columns"),
        (lambda df: df.iloc[:1], "rows"),
        (lambda df: df.iloc[::-1].reset_index(drop=True), "order"),
        (lambda df: df.assign(Pred=[1.2, 0.5]), "probability"),
        (lambda df: df.assign(Pred=[float("nan"), 0.5]), "probability"),
    ],
)
def test_validate_submission_rejects(mutate, msg):
    sample = _sample("2026_1101_1102", "2026_1101_1103")
    out = generate_predictions(sample, lambda a, b: 0.6, None)[["ID", "Pred"]]
    with pytest.raises(ValueError, match=msg):
        validate_submission(mutate(out), sample)


def test_load_kaggle_teams(tmp_path):
    p = tmp_path / "WTeams.csv"
    p.write_text("TeamID,TeamName\n3101,Abilene Chr\n3163,Connecticut\nbad,Nope\n")
    assert load_kaggle_teams(p) == {3101: "Abilene Chr", 3163: "Connecticut"}
    with pytest.raises(ValueError):
        bad = tmp_path / "bad.csv"
        bad.write_text("Id,Name\n1,x\n")
        load_kaggle_teams(bad)


# ------------------------------------------------------------------ slot 2
def test_strongest_team_ranks_by_mean_win_probability():
    from src.exports.kaggle import strongest_team

    pw = {}
    strength = {"a": 0.9, "b": 0.6, "c": 0.3}
    for x in strength:
        for y in strength:
            if x != y:
                pw[(x, y)] = strength[x] / (strength[x] + strength[y])
    ranked = strongest_team(pw, ["c", "b", "a"], top=2)
    assert [t for t, _ in ranked] == ["a", "b"]
    assert ranked[0][1] == pytest.approx((pw[("a", "b")] + pw[("a", "c")]) / 2)


def test_champion_boost_pushes_both_orientations_and_leaves_others():
    from src.exports.kaggle import apply_champion_boost

    df = pd.DataFrame({"ID": ["2026_1101_1181", "2026_1181_1400", "2026_1101_1400"], "Pred": [0.3, 0.8, 0.5]})
    out, n = apply_champion_boost(df, 1181)
    assert n == 2
    assert out["Pred"].tolist() == [0.0, 1.0, 0.5]  # champion as team2 -> 0, as team1 -> 1, untouched
    assert df["Pred"].tolist() == [0.3, 0.8, 0.5]  # input not mutated

    partial, _ = apply_champion_boost(df, 1181, strength=0.5)
    assert partial["Pred"].tolist() == pytest.approx([0.15, 0.9, 0.5])


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_champion_boost_rejects_bad_strength(bad):
    from src.exports.kaggle import apply_champion_boost

    with pytest.raises(ValueError):
        apply_champion_boost(pd.DataFrame({"ID": ["2026_1101_1181"], "Pred": [0.5]}), 1181, strength=bad)


def test_champion_boost_output_still_validates():
    from src.exports.kaggle import apply_champion_boost

    sample = _sample("2026_1101_1181", "2026_1181_1400")
    out = generate_predictions(sample, lambda a, b: 0.6, None)[["ID", "Pred"]]
    boosted, _ = apply_champion_boost(out, 1181)
    validate_submission(boosted, sample)
