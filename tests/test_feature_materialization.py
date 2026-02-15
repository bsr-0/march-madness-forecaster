"""Tests for leakage-safe historical feature materialization."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.features.materialization import HistoricalFeatureMaterializer, MaterializationConfig


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)


def _build_historical_fixture(root: Path):
    historical_dir = root / "historical"
    raw_dir = root / "raw"

    games_2022 = {
        "season": 2022,
        "games": [],
        "team_games": [
            {
                "game_id": "g1",
                "season": 2022,
                "date": "2021-11-10",
                "team_id": "a",
                "team_name": "A",
                "opponent_id": "b",
                "opponent_name": "B",
                "team_score": 80,
                "opponent_score": 70,
                "possessions": 70,
                "fgm": 28,
                "fga": 58,
                "fg3m": 8,
                "fg3a": 22,
                "fta": 16,
                "turnovers": 9,
                "orb": 10,
                "drb": 22,
            },
            {
                "game_id": "g1",
                "season": 2022,
                "date": "2021-11-10",
                "team_id": "b",
                "team_name": "B",
                "opponent_id": "a",
                "opponent_name": "A",
                "team_score": 70,
                "opponent_score": 80,
                "possessions": 70,
                "fgm": 25,
                "fga": 57,
                "fg3m": 6,
                "fg3a": 20,
                "fta": 14,
                "turnovers": 12,
                "orb": 8,
                "drb": 19,
            },
            {
                "game_id": "g2",
                "season": 2022,
                "date": "2021-11-20",
                "team_id": "a",
                "team_name": "A",
                "opponent_id": "c",
                "opponent_name": "C",
                "team_score": 60,
                "opponent_score": 65,
                "possessions": 68,
                "fgm": 22,
                "fga": 55,
                "fg3m": 5,
                "fg3a": 19,
                "fta": 11,
                "turnovers": 11,
                "orb": 9,
                "drb": 20,
            },
            {
                "game_id": "g2",
                "season": 2022,
                "date": "2021-11-20",
                "team_id": "c",
                "team_name": "C",
                "opponent_id": "a",
                "opponent_name": "A",
                "team_score": 65,
                "opponent_score": 60,
                "possessions": 68,
                "fgm": 24,
                "fga": 54,
                "fg3m": 4,
                "fg3a": 16,
                "fta": 13,
                "turnovers": 10,
                "orb": 8,
                "drb": 23,
            },
        ],
    }
    games_2023 = {
        "season": 2023,
        "games": [],
        "team_games": [
            {
                "game_id": "g3",
                "season": 2023,
                "date": "2022-11-11",
                "team_id": "a",
                "team_name": "A",
                "opponent_id": "b",
                "opponent_name": "B",
                "team_score": 75,
                "opponent_score": 68,
                "possessions": 69,
                "fgm": 26,
                "fga": 57,
                "fg3m": 7,
                "fg3a": 20,
                "fta": 15,
                "turnovers": 8,
                "orb": 9,
                "drb": 22,
            },
            {
                "game_id": "g3",
                "season": 2023,
                "date": "2022-11-11",
                "team_id": "b",
                "team_name": "B",
                "opponent_id": "a",
                "opponent_name": "A",
                "team_score": 68,
                "opponent_score": 75,
                "possessions": 69,
                "fgm": 23,
                "fga": 56,
                "fg3m": 6,
                "fg3a": 19,
                "fta": 12,
                "turnovers": 10,
                "orb": 7,
                "drb": 21,
            },
        ],
    }

    tm_2022 = {
        "season": 2022,
        "teams": [
            {"team_id": "a", "team_name": "A", "off_rtg": 115, "def_rtg": 95, "pace": 70, "srs": 10, "sos": 5, "wins": 26, "losses": 8},
            {"team_id": "b", "team_name": "B", "off_rtg": 109, "def_rtg": 98, "pace": 68, "srs": 6, "sos": 3, "wins": 23, "losses": 11},
            {"team_id": "c", "team_name": "C", "off_rtg": 101, "def_rtg": 100, "pace": 67, "srs": 2, "sos": 1, "wins": 19, "losses": 14},
        ],
    }
    tm_2023 = {
        "season": 2023,
        "teams": [
            {"team_id": "a", "team_name": "A", "off_rtg": 117, "def_rtg": 93, "pace": 69, "srs": 11, "sos": 6, "wins": 27, "losses": 7},
            {"team_id": "b", "team_name": "B", "off_rtg": 108, "def_rtg": 99, "pace": 68, "srs": 5, "sos": 2, "wins": 22, "losses": 12},
        ],
    }
    seeds_2023 = {
        "season": 2023,
        "teams": [
            {"team_id": "a", "team_name": "A", "seed": 1, "region": "East", "school_slug": "a"},
            {"team_id": "b", "team_name": "B", "seed": 8, "region": "East", "school_slug": "b"},
        ],
    }
    rosters_2022 = {
        "season": 2022,
        "teams": [
            {
                "team_id": "a",
                "team_name": "A",
                "players": [
                    {"player_id": "a1", "rapm_total": 1.5, "warp": 0.3, "minutes_per_game": 30, "is_transfer": False, "class_year": "SR"},
                    {"player_id": "a2", "rapm_total": 0.8, "warp": 0.2, "minutes_per_game": 20, "is_transfer": True, "class_year": "SO"},
                ],
            }
        ],
    }

    _write_json(historical_dir / "historical_games_2022.json", games_2022)
    _write_json(historical_dir / "historical_games_2023.json", games_2023)
    _write_json(historical_dir / "team_metrics_2022.json", tm_2022)
    _write_json(historical_dir / "team_metrics_2023.json", tm_2023)
    _write_json(historical_dir / "tournament_seeds_2023.json", seeds_2023)
    _write_json(raw_dir / "rosters_2022.json", rosters_2022)
    return historical_dir, raw_dir


def _read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def test_materializer_is_leakage_safe_and_uses_prior_season_metrics(tmp_path):
    historical_dir, raw_dir = _build_historical_fixture(tmp_path)
    output_dir = tmp_path / "processed"

    config = MaterializationConfig(
        start_season=2022,
        end_season=2023,
        historical_dir=str(historical_dir),
        raw_dir=str(raw_dir),
        output_dir=str(output_dir),
        strict_validation=True,
    )
    manifest = HistoricalFeatureMaterializer(config).run()

    assert manifest["leakage_checks"]["passed"] is True

    team_df = _read_table(manifest["artifacts"]["team_game_features_path"])
    first_a = team_df[(team_df["game_id"] == "g1") & (team_df["team_id"] == "a")].iloc[0]
    second_a = team_df[(team_df["game_id"] == "g2") & (team_df["team_id"] == "a")].iloc[0]
    third_a = team_df[(team_df["game_id"] == "g3") & (team_df["team_id"] == "a")].iloc[0]

    assert float(first_a["games_played_prior"]) == 0.0
    assert pd.isna(first_a["off_eff_prior"])
    assert float(second_a["games_played_prior"]) == 1.0
    assert abs(float(second_a["off_eff_prior"]) - float(first_a["off_eff_game"])) < 1e-9

    # Season 2023 uses season-2022 team metrics as priors (no in-season leakage).
    assert abs(float(third_a["prior_season_off_rtg"]) - 115.0) < 1e-9
    assert abs(float(third_a["prior_season_def_rtg"]) - 95.0) < 1e-9
    assert "orb_rate_prior" in team_df.columns
    assert pd.notna(second_a["orb_rate_prior"])
    assert "prior_roster_minutes_returning_share" in team_df.columns
    assert abs(float(third_a["prior_roster_minutes_returning_share"]) - 0.6) < 1e-9


def test_materializer_creates_matchup_rows(tmp_path):
    historical_dir, raw_dir = _build_historical_fixture(tmp_path)
    output_dir = tmp_path / "processed"

    manifest = HistoricalFeatureMaterializer(
        MaterializationConfig(
            start_season=2022,
            end_season=2023,
            historical_dir=str(historical_dir),
            raw_dir=str(raw_dir),
            output_dir=str(output_dir),
            strict_validation=True,
        )
    ).run()

    matchup_df = _read_table(manifest["artifacts"]["matchup_features_path"])
    assert len(matchup_df) == 3
    assert {"team1_win", "margin", "diff_off_eff_prior", "avg_off_eff_prior"}.issubset(set(matchup_df.columns))

    tournament_df = _read_table(manifest["artifacts"]["tournament_matchup_features_path"])
    assert {"team1_seed", "team2_seed", "seed_diff", "is_seed_upset"}.issubset(set(tournament_df.columns))


def test_team_identity_alignment_maps_mascot_names():
    m = HistoricalFeatureMaterializer(MaterializationConfig())
    canonical = pd.DataFrame(
        [
            {"season": 2025, "team_id": "duke_blue_devils", "team_name": "Duke Blue Devils"},
            {"season": 2025, "team_id": "gonzaga_bulldogs", "team_name": "Gonzaga Bulldogs"},
        ]
    )
    source = pd.DataFrame(
        [
            {"season": 2025, "team_id": "duke", "team_name": "Duke", "off_rtg": 120.0},
            {"season": 2025, "team_id": "gonzaga", "team_name": "Gonzaga", "off_rtg": 118.0},
        ]
    )
    aligned = m._align_source_team_ids(source, canonical, source_name_col="team_name")

    assert set(aligned["team_id"]) == {"duke_blue_devils", "gonzaga_bulldogs"}
    assert (aligned["team_match_score"] >= 0.84).all()


def test_materializer_requires_all_requested_seasons_by_default(tmp_path):
    historical_dir, raw_dir = _build_historical_fixture(tmp_path)
    output_dir = tmp_path / "processed"

    with pytest.raises(ValueError, match="Missing requested seasons"):
        HistoricalFeatureMaterializer(
            MaterializationConfig(
                start_season=2022,
                end_season=2024,
                historical_dir=str(historical_dir),
                raw_dir=str(raw_dir),
                output_dir=str(output_dir),
                strict_validation=True,
            )
        ).run()

    manifest = HistoricalFeatureMaterializer(
        MaterializationConfig(
            start_season=2022,
            end_season=2024,
            historical_dir=str(historical_dir),
            raw_dir=str(raw_dir),
            output_dir=str(output_dir),
            strict_validation=True,
            require_all_seasons=False,
        )
    ).run()

    assert manifest["quality_report"]["season_coverage"]["missing_seasons"] == [2024]
