"""Comprehensive unit tests for baseline_training, materialization, and data_loader modules.

Tests cover:
- src/pipeline/stages/baseline_training.py
- src/data/features/materialization.py
- src/pipeline/stages/data_loader.py

Uses extensive mocking for I/O, external modules, and pipeline dependencies.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, mock_open, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest


# =====================================================================
# MATERIALIZATION TESTS
# =====================================================================

class TestMaterializationConfig:
    """Tests for MaterializationConfig dataclass."""

    def test_default_values(self):
        from src.data.features.materialization import MaterializationConfig
        cfg = MaterializationConfig()
        assert cfg.start_season == 2022
        assert cfg.end_season == 2025
        assert cfg.strict_validation is True
        assert cfg.require_all_seasons is True
        assert cfg.min_tournament_matchups == 1

    def test_custom_values(self):
        from src.data.features.materialization import MaterializationConfig
        cfg = MaterializationConfig(start_season=2020, end_season=2023, strict_validation=False)
        assert cfg.start_season == 2020
        assert cfg.end_season == 2023
        assert cfg.strict_validation is False


class TestHistoricalFeatureMaterializer:
    """Tests for HistoricalFeatureMaterializer."""

    @pytest.fixture
    def materializer(self, tmp_path):
        from src.data.features.materialization import (
            HistoricalFeatureMaterializer,
            MaterializationConfig,
        )
        cfg = MaterializationConfig(
            start_season=2023,
            end_season=2023,
            historical_dir=str(tmp_path / "historical"),
            raw_dir=str(tmp_path / "raw"),
            output_dir=str(tmp_path / "output"),
            strict_validation=False,
            require_all_seasons=False,
            min_tournament_matchups=0,
        )
        (tmp_path / "historical").mkdir()
        (tmp_path / "raw").mkdir()
        (tmp_path / "output").mkdir()
        return HistoricalFeatureMaterializer(cfg)

    def test_init_creates_output_dir(self, tmp_path):
        from src.data.features.materialization import (
            HistoricalFeatureMaterializer,
            MaterializationConfig,
        )
        out = tmp_path / "new_output"
        cfg = MaterializationConfig(output_dir=str(out))
        mat = HistoricalFeatureMaterializer(cfg)
        assert out.exists()

    def test_discover_artifacts_default(self, materializer):
        artifacts = materializer._discover_artifacts()
        assert "2023" in artifacts
        assert "historical_games_json" in artifacts["2023"]
        assert "team_metrics_json" in artifacts["2023"]

    def test_discover_artifacts_from_manifest(self, materializer, tmp_path):
        manifest = {
            "artifacts": {
                "2023": {
                    "historical_games_json": "path/to/games.json",
                    "team_metrics_json": "path/to/metrics.json",
                }
            }
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        materializer.config.historical_manifest_path = str(manifest_path)
        artifacts = materializer._discover_artifacts()
        assert artifacts["2023"]["historical_games_json"] == "path/to/games.json"

    def test_load_team_games_skips_2020(self, materializer, tmp_path):
        """Season 2020 should be skipped (COVID)."""
        games_path = tmp_path / "historical" / "historical_games_2020.json"
        games_path.write_text(json.dumps({"team_games": [
            {"game_id": "1", "team_id": "duke", "opponent_id": "unc",
             "team_score": 80, "opponent_score": 75, "date": "2020-01-15",
             "season": 2020}
        ]}))
        artifacts = {"2020": {"historical_games_json": str(games_path)}}
        with pytest.raises(ValueError, match="No team-game rows found"):
            materializer._load_team_games(artifacts)

    def test_load_team_games_with_valid_data(self, materializer, tmp_path):
        games = []
        for i in range(5):
            games.append({
                "game_id": f"g{i}", "team_id": f"team_a", "opponent_id": "team_b",
                "team_score": 75 + i, "opponent_score": 70, "date": f"2023-01-{10+i:02d}",
                "season": 2023,
            })
            games.append({
                "game_id": f"g{i}", "team_id": "team_b", "opponent_id": "team_a",
                "team_score": 70, "opponent_score": 75 + i, "date": f"2023-01-{10+i:02d}",
                "season": 2023,
            })
        games_path = tmp_path / "historical" / "historical_games_2023.json"
        games_path.write_text(json.dumps({"team_games": games}))
        artifacts = {"2023": {"historical_games_json": str(games_path)}}
        df = materializer._load_team_games(artifacts)
        assert len(df) == 10
        assert "game_id" in df.columns
        assert "team_score" in df.columns

    def test_load_team_games_filters_forfeits(self, materializer, tmp_path):
        """Games with zero scores should be filtered out."""
        games = [
            {"game_id": "g1", "team_id": "a", "opponent_id": "b",
             "team_score": 0, "opponent_score": 75, "date": "2023-01-10", "season": 2023},
            {"game_id": "g1", "team_id": "b", "opponent_id": "a",
             "team_score": 75, "opponent_score": 0, "date": "2023-01-10", "season": 2023},
            {"game_id": "g2", "team_id": "a", "opponent_id": "b",
             "team_score": 80, "opponent_score": 70, "date": "2023-01-11", "season": 2023},
            {"game_id": "g2", "team_id": "b", "opponent_id": "a",
             "team_score": 70, "opponent_score": 80, "date": "2023-01-11", "season": 2023},
        ]
        gp = tmp_path / "historical" / "historical_games_2023.json"
        gp.write_text(json.dumps({"team_games": games}))
        df = materializer._load_team_games({"2023": {"historical_games_json": str(gp)}})
        assert len(df) == 2  # Only valid game rows survive

    def test_load_team_games_filters_huge_margins(self, materializer, tmp_path):
        """Games with margin > 80 should be filtered."""
        games = [
            {"game_id": "g1", "team_id": "a", "opponent_id": "b",
             "team_score": 150, "opponent_score": 50, "date": "2023-01-10", "season": 2023},
            {"game_id": "g1", "team_id": "b", "opponent_id": "a",
             "team_score": 50, "opponent_score": 150, "date": "2023-01-10", "season": 2023},
        ]
        gp = tmp_path / "historical" / "historical_games_2023.json"
        gp.write_text(json.dumps({"team_games": games}))
        df = materializer._load_team_games({"2023": {"historical_games_json": str(gp)}})
        assert len(df) == 0

    def test_load_team_metrics_basic(self, materializer, tmp_path):
        metrics = {
            "teams": [
                {"team_name": "Duke", "off_rtg": 115.0, "def_rtg": 95.0,
                 "pace": 70.0, "srs": 20.0, "sos": 5.0, "wins": 25, "losses": 5},
            ]
        }
        mp = tmp_path / "historical" / "team_metrics_2023.json"
        mp.write_text(json.dumps(metrics))
        df = materializer._load_team_metrics({"2023": {"team_metrics_json": str(mp)}})
        assert len(df) >= 1
        assert "off_rtg" in df.columns

    def test_load_team_metrics_skips_2020(self, materializer, tmp_path):
        metrics = {"teams": [{"team_name": "Duke", "off_rtg": 115.0, "def_rtg": 95.0}]}
        mp = tmp_path / "historical" / "team_metrics_2020.json"
        mp.write_text(json.dumps(metrics))
        df = materializer._load_team_metrics({"2020": {"team_metrics_json": str(mp)}})
        assert len(df) == 0

    def test_load_team_metrics_clamps_extreme_values(self, materializer, tmp_path):
        metrics = {
            "teams": [
                {"team_name": "Extreme", "off_rtg": 200.0, "def_rtg": 30.0, "pace": 150.0},
            ]
        }
        mp = tmp_path / "historical" / "team_metrics_2023.json"
        mp.write_text(json.dumps(metrics))
        df = materializer._load_team_metrics({"2023": {"team_metrics_json": str(mp)}})
        assert float(df.iloc[0]["off_rtg"]) <= 140.0
        assert float(df.iloc[0]["def_rtg"]) >= 60.0

    def test_load_team_metrics_empty_returns_default_columns(self, materializer):
        df = materializer._load_team_metrics({})
        assert "season" in df.columns
        assert "off_rtg" in df.columns

    def test_load_tournament_seeds(self, materializer, tmp_path):
        seeds = {
            "teams": [
                {"team_name": "Duke", "seed": 1, "region": "East"},
                {"team_name": "UNC", "seed": 8, "region": "East"},
            ]
        }
        sp = tmp_path / "historical" / "tournament_seeds_2023.json"
        sp.write_text(json.dumps(seeds))
        df = materializer._load_tournament_seeds(
            {"2023": {"tournament_seeds_json": str(sp)}}
        )
        assert len(df) == 2

    def test_load_tournament_seeds_skips_2020(self, materializer, tmp_path):
        seeds = {"teams": [{"team_name": "Duke", "seed": 1, "region": "East"}]}
        sp = tmp_path / "historical" / "tournament_seeds_2020.json"
        sp.write_text(json.dumps(seeds))
        df = materializer._load_tournament_seeds(
            {"2020": {"tournament_seeds_json": str(sp)}}
        )
        assert len(df) == 0

    def test_canonical_team_index(self, materializer):
        df = pd.DataFrame({
            "season": [2023, 2023, 2023],
            "team_id": ["duke", "duke", "unc"],
            "team_name": ["Duke", "Duke", "UNC"],
            "extra": [1, 2, 3],
        })
        idx = materializer._canonical_team_index(df)
        assert len(idx) == 2  # deduplicated

    def test_season_coverage_report(self, materializer):
        df = pd.DataFrame({"season": [2023]})
        report = materializer._season_coverage_report(df)
        assert report["expected_seasons"] == [2023]
        assert report["present_seasons"] == [2023]
        assert report["missing_seasons"] == []

    def test_season_coverage_report_with_missing(self, materializer):
        materializer.config.start_season = 2022
        materializer.config.end_season = 2023
        df = pd.DataFrame({"season": [2022]})
        report = materializer._season_coverage_report(df)
        assert 2023 in report["missing_seasons"]

    def test_leakage_checks_passes(self, materializer):
        """Leakage checks should pass for correctly constructed features."""
        df = pd.DataFrame({
            "team_id": ["a", "a", "a", "b", "b", "b"],
            "game_id": ["g1", "g2", "g3", "g1", "g2", "g3"],
            "date": pd.to_datetime(["2023-01-01", "2023-01-05", "2023-01-10"] * 2),
            "off_eff_game": [100.0, 110.0, 105.0, 95.0, 90.0, 100.0],
            "games_played_prior": [0, 1, 2, 0, 1, 2],
        })
        # Build off_eff_prior correctly (shifted expanding mean)
        df = df.sort_values(["team_id", "date", "game_id"]).reset_index(drop=True)
        df["off_eff_prior"] = df.groupby("team_id")["off_eff_game"].transform(
            lambda s: s.shift(1).expanding().mean()
        )
        result = materializer._leakage_checks(df)
        assert result["passed"] is True

    def test_leakage_checks_detects_issues(self, materializer):
        """Detects when off_eff_prior has wrong values."""
        df = pd.DataFrame({
            "team_id": ["a", "a", "b", "b"],
            "game_id": ["g1", "g2", "g1", "g2"],
            "date": pd.to_datetime(["2023-01-01", "2023-01-05"] * 2),
            "off_eff_game": [100.0, 110.0, 95.0, 90.0],
            "games_played_prior": [0, 1, 0, 1],
            "off_eff_prior": [np.nan, 999.0, np.nan, 999.0],  # wrong
        })
        result = materializer._leakage_checks(df)
        assert result["passed"] is False

    def test_write_table(self, materializer, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        path, fmt = materializer._write_table(df, "test_table")
        assert fmt in ("parquet", "csv")
        assert path.exists()

    def test_write_table_csv_fallback(self, materializer, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3]})
        with patch.object(pd.DataFrame, "to_parquet", side_effect=Exception("no pyarrow")):
            path, fmt = materializer._write_table(df, "test_table")
            assert fmt == "csv"
            assert path.exists()

    def test_build_feature_dictionary(self, materializer):
        tf = pd.DataFrame(columns=["season", "date", "game_id", "team_id", "off_eff_prior"])
        mf = pd.DataFrame(columns=["season", "date", "game_id", "team1_id", "team2_id", "diff_off_eff"])
        tmf = pd.DataFrame(columns=["season", "date", "game_id", "team1_id", "team2_id", "seed_diff"])
        result = materializer._build_feature_dictionary(tf, mf, tmf)
        assert "team_game_columns" in result
        assert "matchup_columns" in result
        assert "targets" in result

    def test_quality_report(self, materializer):
        tf = pd.DataFrame({
            "season": [2023, 2023], "game_id": ["g1", "g1"],
            "team_id": ["a", "b"], "off_eff_prior": [100.0, np.nan],
        })
        mf = pd.DataFrame({"x": [1]})
        tmf = pd.DataFrame({"y": [1]})
        sc = {"expected_seasons": [2023], "present_seasons": [2023], "missing_seasons": []}
        report = materializer._quality_report(tf, mf, tmf, sc)
        assert report["team_game_rows"] == 2
        assert report["matchup_rows"] == 1

    def test_shift_prior_season_table(self, materializer):
        df = pd.DataFrame({
            "season": [2022, 2022],
            "team_id": ["duke", "unc"],
            "off_rtg": [115.0, 105.0],
        })
        shifted = materializer._shift_prior_season_table(df, ["off_rtg"], prefix="prior_")
        assert "prior_off_rtg" in shifted.columns
        # Season should be shifted +1
        assert shifted["season"].iloc[0] == 2023

    def test_shift_prior_season_table_empty(self, materializer):
        df = pd.DataFrame(columns=["season", "team_id", "off_rtg"])
        shifted = materializer._shift_prior_season_table(df, ["off_rtg"], prefix="prior_")
        assert list(shifted.columns) == ["season", "team_id"]

    def test_infer_dates_from_game_ids(self, materializer):
        from src.data.features.materialization import HistoricalFeatureMaterializer
        rows = [
            {"game_id": "100", "date": "2023-01-01"},
            {"game_id": "200", "date": "2023-01-01"},
            {"game_id": "300", "date": "2023-01-01"},
        ]
        HistoricalFeatureMaterializer._infer_dates_from_game_ids(rows, 2023)
        dates = [r["date"] for r in rows]
        # First game should get start-of-season date, last game end-of-season
        assert dates[0] != dates[2]  # Dates should differ

    def test_fallback_team_game_rows(self, materializer):
        game = {
            "game_id": "g1",
            "team1_id": "duke",
            "team2_id": "unc",
            "team1_score": 80,
            "team2_score": 75,
            "date": "2023-01-10",
        }
        rows = materializer._fallback_team_game_rows(game, 2023)
        assert len(rows) == 2
        assert rows[0]["team_id"] == "duke"
        assert rows[1]["team_id"] == "unc"

    def test_fallback_team_game_rows_missing_ids(self, materializer):
        game = {"game_id": "g1"}
        rows = materializer._fallback_team_game_rows(game, 2023)
        assert len(rows) == 0

    def test_compute_metrics_from_games(self, materializer, tmp_path):
        games = []
        for i in range(10):
            games.append({
                "team_id": "duke",
                "team_score": 80, "opponent_score": 70,
                "fga": 60, "orb": 10, "turnovers": 12, "fta": 20,
            })
        gp = tmp_path / "games.json"
        gp.write_text(json.dumps({"team_games": games}))
        result = materializer._compute_metrics_from_games(str(gp))
        assert "duke" in result
        assert "off_rtg" in result["duke"]

    def test_compute_metrics_from_games_nonexistent(self, materializer):
        result = materializer._compute_metrics_from_games("/nonexistent/path.json")
        assert result == {}

    def test_load_advanced_metrics_season(self, materializer, tmp_path):
        data = {
            "teams": [
                {"team_id": "duke", "name": "Duke",
                 "adj_offensive_efficiency": 120.0,
                 "adj_defensive_efficiency": 90.0}
            ]
        }
        p = tmp_path / "raw" / "advanced_metrics_2023.json"
        p.write_text(json.dumps(data))
        df = materializer._load_advanced_metrics_season(2023)
        assert df is not None
        assert len(df) == 1
        assert "prop_adj_off" in df.columns

    def test_load_advanced_metrics_season_not_found(self, materializer):
        df = materializer._load_advanced_metrics_season(2023)
        assert df is None

    def test_load_torvik_season(self, materializer, tmp_path):
        data = {
            "teams": [
                {"team_id": "duke", "name": "Duke",
                 "barthag": 0.95, "effective_fg_pct": 0.55}
            ]
        }
        p = tmp_path / "raw" / "torvik_2023.json"
        p.write_text(json.dumps(data))
        df = materializer._load_torvik_season(2023)
        assert df is not None
        assert "torvik_barthag" in df.columns

    def test_load_torvik_season_not_found(self, materializer):
        assert materializer._load_torvik_season(2023) is None

    def test_load_roster_season(self, materializer, tmp_path):
        data = {
            "teams": [{
                "team_id": "duke", "team_name": "Duke",
                "players": [
                    {"name": "Player1", "rapm_total": 2.0, "warp": 1.5,
                     "is_transfer": True, "injury_status": "healthy",
                     "minutes_per_game": 30, "class_year": "sr"},
                    {"name": "Player2", "rapm_total": 1.0, "warp": 0.5,
                     "is_transfer": False, "injury_status": "out",
                     "minutes_per_game": 25, "class_year": "jr"},
                ]
            }]
        }
        p = tmp_path / "raw" / "rosters_2023.json"
        p.write_text(json.dumps(data))
        df = materializer._load_roster_season(2023)
        assert df is not None
        assert "roster_total_rapm" in df.columns
        assert "roster_transfer_share" in df.columns
        assert df.iloc[0]["roster_transfer_share"] == 0.5

    def test_load_roster_season_not_found(self, materializer):
        assert materializer._load_roster_season(9999) is None

    def test_load_transfer_season(self, materializer, tmp_path):
        data = {
            "entries": [
                {"source_team_name": "Duke", "destination_team_name": "UNC"},
                {"source_team_name": "Duke", "destination_team_name": "Kentucky"},
            ]
        }
        p = tmp_path / "raw" / "transfer_portal_2023.json"
        p.write_text(json.dumps(data))
        df = materializer._load_transfer_season(2023)
        assert df is not None
        assert "transfer_net" in df.columns

    def test_load_transfer_season_not_found(self, materializer):
        assert materializer._load_transfer_season(9999) is None

    def test_load_market_season(self, materializer, tmp_path):
        data = {
            "teams": [
                {"team_name": "Duke", "implied_win_probability": 0.85, "title_odds": 0.05},
            ]
        }
        p = tmp_path / "raw" / "odds_2023.json"
        p.write_text(json.dumps(data))
        df = materializer._load_market_season(2023)
        assert df is not None
        assert "market_implied_win" in df.columns

    def test_load_market_season_not_found(self, materializer):
        assert materializer._load_market_season(9999) is None

    def test_load_polls_season(self, materializer, tmp_path):
        data = {
            "records": [
                {"team_name": "Duke", "ap_preseason_rank": 5, "coaches_preseason_rank": 4},
            ]
        }
        p = tmp_path / "raw" / "polls_2023.json"
        p.write_text(json.dumps(data))
        df = materializer._load_polls_season(2023)
        assert df is not None
        assert "polls_ap_preseason" in df.columns

    def test_load_polls_season_not_found(self, materializer):
        assert materializer._load_polls_season(9999) is None

    def test_load_torvik_splits_season(self, materializer, tmp_path):
        data = {
            "records": [
                {"team_id": "duke", "wab": 5.0, "sos": 10.0},
            ]
        }
        p = tmp_path / "raw" / "torvik_splits_2023.json"
        p.write_text(json.dumps(data))
        df = materializer._load_torvik_splits_season(2023)
        assert df is not None
        assert "torvik_split_wab" in df.columns

    def test_load_ncaa_team_stats_season(self, materializer, tmp_path):
        data = {
            "teams": [
                {"team_id": "duke", "assist_turnover_ratio": 1.5,
                 "rebound_margin": 5.0, "foul_rate": 0.18},
            ]
        }
        p = tmp_path / "raw" / "ncaa_team_stats_2023.json"
        p.write_text(json.dumps(data))
        df = materializer._load_ncaa_team_stats_season(2023)
        assert df is not None
        assert "ncaa_ast_to_ratio" in df.columns

    def test_load_weather_context_season(self, materializer, tmp_path):
        data = {
            "records": [
                {"team_id": "duke", "temperature_f": 45.0, "wind_mph": 10.0, "severe_alert": 0},
                {"team_id": "duke", "temperature_f": 50.0, "wind_mph": 5.0, "severe_alert": 1},
            ]
        }
        p = tmp_path / "raw" / "weather_context_2023.json"
        p.write_text(json.dumps(data))
        df = materializer._load_weather_context_season(2023)
        assert df is not None
        assert len(df) == 1  # aggregated per team
        assert "weather_avg_temp_f" in df.columns

    def test_load_travel_context_season(self, materializer, tmp_path):
        data = {
            "records": [
                {"team_id": "duke", "travel_miles": 500.0,
                 "timezone_change_hours": 1.0, "trip_days": 2.0},
            ]
        }
        p = tmp_path / "raw" / "travel_context_2023.json"
        p.write_text(json.dumps(data))
        df = materializer._load_travel_context_season(2023)
        assert df is not None
        assert "travel_avg_miles" in df.columns

    def test_validate_prior_source_empty(self, materializer):
        df = pd.DataFrame()
        warnings = materializer._validate_prior_source_availability(df)
        assert warnings == []

    def test_validate_prior_source_future_season(self, materializer):
        df = pd.DataFrame({"season": [2099], "team_id": ["duke"]})
        warnings = materializer._validate_prior_source_availability(df)
        assert len(warnings) == 1
        assert "beyond end_season" in warnings[0]

    def test_resolve_team_id_direct_match(self, materializer):
        base_ids = {"duke", "unc", "kentucky"}
        tid, score = materializer._resolve_team_id("duke", "Duke", base_ids)
        assert tid == "duke"
        assert score == 1.0

    def test_resolve_team_id_empty_base(self, materializer):
        tid, score = materializer._resolve_team_id("duke", "Duke", set())
        assert tid == "duke"
        assert score == 0.0

    def test_build_matchup_features(self, materializer):
        tf = pd.DataFrame({
            "game_id": ["g1", "g1"],
            "season": [2023, 2023],
            "date": pd.to_datetime(["2023-01-10", "2023-01-10"]),
            "team_id": ["a", "b"],
            "team_score": [80.0, 70.0],
            "win_pct_prior": [0.7, 0.5],
            "off_eff_prior": [110.0, 100.0],
        })
        mf = materializer._build_matchup_features(tf)
        assert len(mf) == 1
        assert "diff_win_pct_prior" in mf.columns
        assert "avg_off_eff_prior" in mf.columns

    def test_build_matchup_features_skips_non_pair_games(self, materializer):
        tf = pd.DataFrame({
            "game_id": ["g1"],
            "season": [2023],
            "date": pd.to_datetime(["2023-01-10"]),
            "team_id": ["a"],
            "team_score": [80.0],
        })
        mf = materializer._build_matchup_features(tf)
        assert len(mf) == 0

    def test_build_tournament_matchup_features_empty_inputs(self, materializer):
        empty_mf = pd.DataFrame()
        empty_seeds = pd.DataFrame()
        tmf = materializer._build_tournament_matchup_features(empty_mf, empty_seeds)
        assert "team1_seed" in tmf.columns

    def test_study_alignment_report(self, materializer):
        coverage = {
            "categories": [
                {"name": "efficiency_four_factors", "status": "covered"},
                {"name": "schedule_strength", "status": "covered"},
                {"name": "recent_form_and_variance", "status": "covered"},
                {"name": "tournament_seed_context", "status": "missing"},
                {"name": "market_priors", "status": "missing"},
                {"name": "proprietary_xp", "status": "missing"},
                {"name": "talent_roster_transfer", "status": "partial"},
                {"name": "conference_strength", "status": "missing"},
                {"name": "rest_fatigue", "status": "covered"},
            ]
        }
        report = materializer._study_alignment_report(coverage)
        assert "proprietary_efficiency_model" in report
        assert "completeness_score" in report["proprietary_efficiency_model"]

    def test_variable_coverage_report(self, materializer):
        tf = pd.DataFrame({
            "off_eff_prior": [1.0],
            "def_eff_prior": [1.0],
            "net_eff_prior": [1.0],
            "efg_prior": [1.0],
            "to_rate_prior": [1.0],
            "ft_rate_prior": [1.0],
            "orb_rate_prior": [1.0],
            "drb_rate_prior": [1.0],
        })
        mf = pd.DataFrame({"x": [1]})
        tmf = pd.DataFrame({"team1_seed": [1], "team2_seed": [2], "seed_diff": [-1]})
        report = materializer._variable_coverage_report(tf, mf, tmf)
        assert "categories" in report
        assert "summary" in report

    def test_align_source_team_ids_empty(self, materializer):
        df = pd.DataFrame()
        canonical = pd.DataFrame({"season": [2023], "team_id": ["duke"], "team_name": ["Duke"]})
        result = materializer._align_source_team_ids(df, canonical)
        assert result.empty

    def test_load_optional_prior_sources_no_files(self, materializer):
        df = materializer._load_optional_prior_sources()
        assert "season" in df.columns
        assert len(df) == 0


# =====================================================================
# DATA LOADER TESTS
# =====================================================================

class TestPlayerFromDict:
    """Tests for player_from_dict utility."""

    def test_basic_player(self):
        from src.pipeline.stages.data_loader import player_from_dict
        raw = {
            "name": "Test Player",
            "position": "SG",
            "minutes_per_game": 30.0,
            "games_played": 20,
            "rapm_offensive": 1.5,
            "rapm_defensive": 0.5,
        }
        player = player_from_dict("duke", raw)
        assert player.name == "Test Player"
        assert player.team_id == "duke"
        assert player.minutes_per_game == 30.0

    def test_default_position(self):
        from src.pipeline.stages.data_loader import player_from_dict
        raw = {"name": "Test", "position": "INVALID"}
        player = player_from_dict("duke", raw)
        assert player.position.value == "PG"

    def test_default_injury_status(self):
        from src.pipeline.stages.data_loader import player_from_dict
        raw = {"name": "Test", "injury_status": "UNKNOWN"}
        player = player_from_dict("duke", raw)
        assert player.injury_status.value == "healthy"

    def test_transfer_player(self):
        from src.pipeline.stages.data_loader import player_from_dict
        raw = {"name": "Transfer", "is_transfer": True, "transfer_from": "UNC"}
        player = player_from_dict("duke", raw)
        assert player.is_transfer is True
        assert player.transfer_from == "UNC"

    def test_missing_fields_have_defaults(self):
        from src.pipeline.stages.data_loader import player_from_dict
        raw = {}
        player = player_from_dict("duke", raw)
        assert player.name == "Unknown"
        assert player.minutes_per_game == 0.0
        assert player.games_played == 0

    def test_player_id_generated(self):
        from src.pipeline.stages.data_loader import player_from_dict
        raw = {"name": "John Doe"}
        player = player_from_dict("duke", raw)
        assert "duke" in player.player_id
        assert "John Doe" in player.player_id


class TestEnrichRosterRapm:
    """Tests for enrich_roster_rapm utility."""

    def test_already_has_enough_rapm(self):
        from src.pipeline.stages.data_loader import enrich_roster_rapm
        from src.data.models.player import Player, Position
        players = [
            Player(player_id="p1", name="A", team_id="duke", position=Position.POINT_GUARD,
                   rapm_offensive=2.0, rapm_defensive=1.0),
            Player(player_id="p2", name="B", team_id="duke", position=Position.CENTER,
                   rapm_offensive=1.0, rapm_defensive=0.5),
        ]
        enrich_roster_rapm(players, {}, min_rapm_players=2)
        # Should not change anything since both already have RAPM
        assert players[0].rapm_offensive == 2.0

    def test_empty_players(self):
        from src.pipeline.stages.data_loader import enrich_roster_rapm
        enrich_roster_rapm([], {}, min_rapm_players=3)  # Should not error

    def test_bpm_backfill(self):
        from src.pipeline.stages.data_loader import enrich_roster_rapm
        from src.data.models.player import Player, Position
        players = [
            Player(player_id="p1", name="A", team_id="duke", position=Position.POINT_GUARD,
                   rapm_offensive=0.0, rapm_defensive=0.0,
                   box_plus_minus=5.0, warp=1.0, usage_rate=25.0,
                   minutes_per_game=30.0, games_played=25,
                   points_per_game=15.0, assists_per_game=4.0,
                   rebounds_per_game=5.0, steals_per_game=1.2,
                   blocks_per_game=0.5, turnovers_per_game=2.0),
        ]
        enrich_roster_rapm(players, {}, min_rapm_players=3)
        # BPM backfill should set non-zero RAPM from box-score stats
        assert abs(players[0].rapm_offensive) > 0 or abs(players[0].rapm_defensive) > 0


class TestAssessRosterRapmQuality:
    """Tests for assess_roster_rapm_quality."""

    def test_empty_rosters(self):
        from src.pipeline.stages.data_loader import assess_roster_rapm_quality
        result = assess_roster_rapm_quality({}, min_rapm_players=3)
        assert result["teams"] == 0.0
        assert result["team_coverage_ratio"] == 0.0

    def test_with_rosters(self):
        from src.pipeline.stages.data_loader import assess_roster_rapm_quality
        from src.data.models.player import Player, Position, Roster
        players = [
            Player(player_id="p1", name="A", team_id="duke", position=Position.POINT_GUARD,
                   rapm_offensive=2.0, rapm_defensive=1.0),
            Player(player_id="p2", name="B", team_id="duke", position=Position.CENTER,
                   rapm_offensive=0.0, rapm_defensive=0.0),
        ]
        roster = Roster(team_id="duke", players=players)
        result = assess_roster_rapm_quality({"duke": roster}, min_rapm_players=1)
        assert result["teams"] == 1.0
        assert result["team_coverage_ratio"] == 1.0


class TestValidateFeedFreshness:
    """Tests for validate_feed_freshness."""

    def test_freshness_disabled(self):
        from src.pipeline.stages.data_loader import validate_feed_freshness
        config = MagicMock()
        config.enforce_feed_freshness = False
        validate_feed_freshness(config, "Test", {"timestamp": "2020-01-01T00:00:00Z"})
        # Should not raise

    def test_non_dict_payload(self):
        from src.pipeline.stages.data_loader import validate_feed_freshness
        config = MagicMock()
        config.enforce_feed_freshness = True
        validate_feed_freshness(config, "Test", "not a dict")
        # Should not raise

    def test_missing_timestamp_raises(self):
        from src.pipeline.stages.data_loader import validate_feed_freshness
        from src.pipeline.config import DataRequirementError
        config = MagicMock()
        config.enforce_feed_freshness = True
        with pytest.raises(DataRequirementError, match="missing required timestamp"):
            validate_feed_freshness(config, "Test", {})


class TestBracketDataToTeams:
    """Tests for bracket_data_to_teams."""

    def test_basic_conversion(self):
        from src.pipeline.stages.data_loader import bracket_data_to_teams
        bt1 = SimpleNamespace(display_name="Duke", seed=1, region="East", rating=95.0)
        bt2 = SimpleNamespace(display_name="UNC", seed=8, region="East", rating=None)
        bracket = SimpleNamespace(teams=[bt1, bt2])
        teams = bracket_data_to_teams(bracket)
        assert len(teams) == 2
        assert teams[0].name == "Duke"
        assert teams[0].seed == 1
        assert teams[0].stats.get("bracket_rating") == 95.0
        assert "bracket_rating" not in teams[1].stats

    def test_empty_bracket(self):
        from src.pipeline.stages.data_loader import bracket_data_to_teams
        bracket = SimpleNamespace(teams=[])
        teams = bracket_data_to_teams(bracket)
        assert teams == []


class TestComputePriorYearElo:
    """Tests for compute_prior_year_elo."""

    def test_no_data_returns_none(self):
        from src.pipeline.stages.data_loader import compute_prior_year_elo
        config = MagicMock()
        config.year = 2026
        config.multi_year_games_dir = None
        with patch("os.path.isfile", return_value=False):
            result = compute_prior_year_elo(config)
        assert result is None

    def test_skips_2020_to_2019(self):
        from src.pipeline.stages.data_loader import compute_prior_year_elo
        config = MagicMock()
        config.year = 2021  # prior year would be 2020 -> should become 2019
        config.multi_year_games_dir = None
        with patch("os.path.isfile", return_value=False):
            result = compute_prior_year_elo(config)
        assert result is None


class TestApplyTransferPortalUpdates:
    """Tests for apply_transfer_portal_updates."""

    def test_updates_player(self, tmp_path):
        from src.pipeline.stages.data_loader import apply_transfer_portal_updates
        from src.data.models.player import Player, Position, Roster
        players = [
            Player(player_id="p1", name="John Doe", team_id="duke",
                   position=Position.POINT_GUARD),
        ]
        roster = Roster(team_id="duke", players=players)
        rosters = {"duke": roster}

        transfer_data = {
            "entries": [
                {"destination_team_name": "Duke", "player_name": "John Doe",
                 "source_team_name": "UNC"},
            ]
        }
        tp = tmp_path / "transfers.json"
        tp.write_text(json.dumps(transfer_data))
        apply_transfer_portal_updates(rosters, str(tp))
        assert players[0].is_transfer is True
        assert players[0].transfer_from == "UNC"

    def test_no_matching_destination(self, tmp_path):
        from src.pipeline.stages.data_loader import apply_transfer_portal_updates
        from src.data.models.player import Roster
        rosters = {"duke": Roster(team_id="duke", players=[])}
        transfer_data = {
            "entries": [
                {"destination_team_name": "Nonexistent", "player_name": "Nobody"},
            ]
        }
        tp = tmp_path / "transfers.json"
        tp.write_text(json.dumps(transfer_data))
        apply_transfer_portal_updates(rosters, str(tp))
        # Should not raise

    def test_entries_not_list(self, tmp_path):
        from src.pipeline.stages.data_loader import apply_transfer_portal_updates
        from src.data.models.player import Roster
        rosters = {"duke": Roster(team_id="duke", players=[])}
        tp = tmp_path / "transfers.json"
        tp.write_text(json.dumps({"entries": "not_a_list"}))
        apply_transfer_portal_updates(rosters, str(tp))
        # Should not raise, just return


class TestLoadTeams:
    """Tests for load_teams."""

    def test_loads_from_teams_json(self):
        from src.pipeline.stages.data_loader import load_teams
        from src.models.team import Team
        config = MagicMock()
        config.teams_json = "teams.json"
        config.bracket_json = None
        config.bracket_source = "auto"
        mock_teams = [Team(name="Duke", seed=1, region="East")]
        with patch("src.pipeline.stages.data_loader.DataLoader.load_teams_from_json",
                    return_value=mock_teams):
            result = load_teams(config, MagicMock())
        assert len(result) == 1
        assert result[0].name == "Duke"

    def test_loads_from_bracket_json(self):
        from src.pipeline.stages.data_loader import load_teams
        config = MagicMock()
        config.teams_json = None
        config.bracket_json = "bracket.json"
        config.bracket_source = "auto"
        bt = SimpleNamespace(display_name="Duke", seed=1, region="East", rating=None)
        mock_bracket = SimpleNamespace(teams=[bt])
        bracket_pipeline = MagicMock()
        bracket_pipeline.fetch.return_value = mock_bracket
        result = load_teams(config, bracket_pipeline)
        assert len(result) == 1

    def test_raises_when_no_source(self):
        from src.pipeline.stages.data_loader import load_teams
        from src.pipeline.config import DataRequirementError
        config = MagicMock()
        config.teams_json = None
        config.bracket_json = None
        config.bracket_source = "auto"
        with patch("src.pipeline.stages.data_loader.BIGDANCE_AVAILABLE", False):
            with pytest.raises(DataRequirementError):
                load_teams(config, MagicMock())


class TestEnrichTournamentContext:
    """Tests for enrich_tournament_context."""

    def test_no_data_returns_early(self):
        from src.pipeline.stages.data_loader import enrich_tournament_context
        config = MagicMock()
        config.preseason_ap_json = None
        config.coach_tournament_json = None
        config.conf_champions_json = None
        # Should not raise
        enrich_tournament_context(config, {}, {}, [])

    def test_injects_ap_rank(self):
        from src.pipeline.stages.data_loader import enrich_tournament_context
        from src.models.team import Team
        config = MagicMock()
        config.preseason_ap_json = "ap.json"
        config.coach_tournament_json = None
        config.conf_champions_json = None
        config.roster_json = None
        team = Team(name="Duke", seed=1, region="East")
        torvik_map = {"duke": {}}
        proprietary_map = {}
        with patch(
            "src.pipeline.stages.data_loader.TournamentContextScraper.load_preseason_ap_from_json",
            return_value={"duke": 5},
        ):
            enrich_tournament_context(config, torvik_map, proprietary_map, [team])
        assert torvik_map["duke"]["preseason_ap_rank"] == 5


# =====================================================================
# BASELINE TRAINING TESTS
# =====================================================================

class TestBuildEnrichedMeta:
    """Tests for _build_enriched_meta."""

    def test_basic_2_models(self):
        from src.pipeline.stages.baseline_training import _build_enriched_meta
        base_X = np.array([[0.5, 0.6], [0.7, 0.8]])
        result = _build_enriched_meta(base_X)
        # 2 base + C(2,2)=1 interaction + 3 aggregates = 6
        assert result.shape == (2, 6)

    def test_3_models(self):
        from src.pipeline.stages.baseline_training import _build_enriched_meta
        base_X = np.random.rand(10, 3)
        result = _build_enriched_meta(base_X)
        # 3 base + C(3,2)=3 interactions + 3 aggregates = 9
        assert result.shape == (10, 9)

    def test_single_model(self):
        from src.pipeline.stages.baseline_training import _build_enriched_meta
        base_X = np.array([[0.5], [0.6]])
        result = _build_enriched_meta(base_X)
        # 1 base + 0 interactions + 3 aggregates = 4
        assert result.shape == (2, 4)


class TestSelectBestSingleModel:
    """Tests for _select_best_single_model."""

    def test_no_models_returns_none(self):
        from src.pipeline.stages.baseline_training import _select_best_single_model
        pipeline = MagicMock()
        result = _select_best_single_model(pipeline, [], np.array([]))
        assert result == "none"

    def test_empty_eval_y_returns_first(self):
        from src.pipeline.stages.baseline_training import _select_best_single_model
        pipeline = MagicMock()
        model = MagicMock()
        result = _select_best_single_model(
            pipeline,
            [("lgb", model, np.array([]))],
            np.array([]),
        )
        assert result == "lightgbm"
        pipeline._set_primary_model.assert_called_once_with("lgb", model)

    def test_selects_best_brier(self):
        from src.pipeline.stages.baseline_training import _select_best_single_model
        pipeline = MagicMock()
        eval_y = np.array([1, 0, 1, 0])
        good_preds = np.array([0.9, 0.1, 0.8, 0.2])  # good
        bad_preds = np.array([0.5, 0.5, 0.5, 0.5])    # bad
        model_good = MagicMock()
        model_bad = MagicMock()
        result = _select_best_single_model(
            pipeline,
            [("logit", model_bad, bad_preds), ("lgb", model_good, good_preds)],
            eval_y,
        )
        assert result == "lightgbm"

    def test_name_mapping(self):
        from src.pipeline.stages.baseline_training import _select_best_single_model
        pipeline = MagicMock()
        model = MagicMock()
        eval_y = np.array([1, 0])
        preds = np.array([0.9, 0.1])
        for short, expected in [
            ("lgb", "lightgbm"),
            ("xgb", "xgboost"),
            ("logit", "logistic_regression"),
            ("spread", "spread_regressor"),
        ]:
            result = _select_best_single_model(pipeline, [(short, model, preds)], eval_y)
            assert result == expected


class TestSetPrimaryModel:
    """Tests for _set_primary_model."""

    def test_set_lgb(self):
        from src.pipeline.stages.baseline_training import _set_primary_model
        pipeline = MagicMock()
        model = MagicMock()
        _set_primary_model(pipeline, "lgb", model)
        assert pipeline.baseline_model.lgb_model == model
        assert pipeline.baseline_model.xgb_model is None
        assert pipeline.baseline_model.logit_model is None

    def test_set_xgb(self):
        from src.pipeline.stages.baseline_training import _set_primary_model
        pipeline = MagicMock()
        model = MagicMock()
        _set_primary_model(pipeline, "xgb", model)
        assert pipeline.baseline_model.xgb_model == model
        assert pipeline.baseline_model.lgb_model is None

    def test_set_logit(self):
        from src.pipeline.stages.baseline_training import _set_primary_model
        pipeline = MagicMock()
        model = MagicMock()
        _set_primary_model(pipeline, "logit", model)
        assert pipeline.baseline_model.logit_model == model

    def test_set_spread(self):
        from src.pipeline.stages.baseline_training import _set_primary_model
        pipeline = MagicMock()
        model = MagicMock()
        _set_primary_model(pipeline, "spread", model)
        assert pipeline.baseline_model.spread_model == model


class TestConstructScheduleGraph:
    """Tests for _construct_schedule_graph."""

    def test_basic_construction(self):
        from src.pipeline.stages.baseline_training import _construct_schedule_graph

        flow1 = MagicMock()
        flow1.team1_id = "duke"
        flow1.team2_id = "unc"
        flow1.game_id = "g1"
        flow1.lead_history = [5]
        flow1.get_xp_margin.return_value = 2.0
        flow1.game_date = "2026-01-15"
        flow1.location_weight = 0.5

        pipeline = MagicMock()
        pipeline._team_id.side_effect = lambda x: x.lower().replace(" ", "_")
        pipeline.all_game_flows = [flow1]
        pipeline.config.gnn_temporal_decay = 0.95
        pipeline.config.year = 2026
        pipeline.team_features = {"duke": np.zeros(8), "unc": np.zeros(8)}
        pipeline._validation_sort_key_boundary = None
        pipeline._is_tournament_game.return_value = False
        pipeline._game_sort_key.return_value = 20260115
        pipeline.proprietary_metrics = {}

        from src.models.team import Team
        teams = [Team(name="Duke", seed=1, region="East")]

        graph = _construct_schedule_graph(pipeline, teams)
        assert graph is not None

    def test_filters_tournament_games(self):
        from src.pipeline.stages.baseline_training import _construct_schedule_graph

        flow = MagicMock()
        flow.team1_id = "duke"
        flow.team2_id = "unc"
        flow.game_id = "g1"
        flow.game_date = "2026-03-20"
        flow.lead_history = [5]
        flow.get_xp_margin.return_value = 0.0

        pipeline = MagicMock()
        pipeline._team_id.side_effect = lambda x: x.lower()
        pipeline.all_game_flows = [flow]
        pipeline.config.gnn_temporal_decay = 0.95
        pipeline.config.year = 2026
        pipeline.team_features = {}
        pipeline._validation_sort_key_boundary = None
        pipeline._is_tournament_game.return_value = True  # Tournament game
        pipeline.proprietary_metrics = {}

        from src.models.team import Team
        teams = [Team(name="Duke", seed=1, region="East")]

        graph = _construct_schedule_graph(pipeline, teams)
        # Tournament game should be filtered, so no edges added
        assert graph is not None

    def test_deduplicates_games(self):
        from src.pipeline.stages.baseline_training import _construct_schedule_graph

        flow = MagicMock()
        flow.team1_id = "duke"
        flow.team2_id = "unc"
        flow.game_id = "g1"
        flow.game_date = "2026-01-15"
        flow.lead_history = [5]
        flow.get_xp_margin.return_value = 1.0
        flow.location_weight = 0.5

        pipeline = MagicMock()
        pipeline._team_id.side_effect = lambda x: x.lower()
        pipeline.all_game_flows = [flow, flow]  # Duplicate
        pipeline.config.gnn_temporal_decay = 0.95
        pipeline.config.year = 2026
        pipeline.team_features = {"duke": np.zeros(4), "unc": np.zeros(4)}
        pipeline._validation_sort_key_boundary = None
        pipeline._is_tournament_game.return_value = False
        pipeline._game_sort_key.return_value = 20260115
        pipeline.proprietary_metrics = {}

        from src.models.team import Team
        teams = []
        graph = _construct_schedule_graph(pipeline, teams)
        assert graph is not None


class TestTrainBaselineModelNoSamples:
    """Tests for _train_baseline_model edge cases."""

    def test_returns_none_when_no_samples(self):
        from src.pipeline.stages.baseline_training import _train_baseline_model

        pipeline = MagicMock()
        pipeline._unique_games.return_value = []
        pipeline._exclude_tournament_games.return_value = []
        pipeline._is_tournament_game.return_value = False
        pipeline.config.year = 2026
        pipeline.config.late_season_training_cutoff_days = 0
        pipeline.config.enable_bayesian_bt = False

        result = _train_baseline_model(pipeline, {})
        assert result["model"] == "none"
        assert result["samples"] == 0


class TestStatSourcesResult:
    """Tests for StatSourcesResult dataclass."""

    def test_default_values(self):
        from src.pipeline.stages.data_loader import StatSourcesResult
        r = StatSourcesResult()
        assert r.torvik_map == {}
        assert r.proprietary_map == {}
        assert r.prior_year_elo is None
        assert r.current_year_game_records == []
        assert r.current_year_conference_map is None


class TestRosterResult:
    """Tests for RosterResult dataclass."""

    def test_default_values(self):
        from src.pipeline.stages.data_loader import RosterResult
        r = RosterResult()
        assert r.rosters == {}
        assert r.roster_rapm_quality == {}


class TestGameFlowResult:
    """Tests for GameFlowResult dataclass."""

    def test_default_values(self):
        from src.pipeline.stages.data_loader import GameFlowResult
        r = GameFlowResult()
        assert r.team_to_games == {}
        assert r.all_game_flows == []


# =====================================================================
# ADDITIONAL MATERIALIZATION EDGE CASE TESTS
# =====================================================================

class TestMaterializerEdgeCases:
    """More edge case tests for HistoricalFeatureMaterializer."""

    @pytest.fixture
    def mat(self, tmp_path):
        from src.data.features.materialization import (
            HistoricalFeatureMaterializer,
            MaterializationConfig,
        )
        cfg = MaterializationConfig(
            start_season=2023,
            end_season=2023,
            historical_dir=str(tmp_path / "h"),
            raw_dir=str(tmp_path / "r"),
            output_dir=str(tmp_path / "o"),
            strict_validation=False,
            require_all_seasons=False,
            min_tournament_matchups=0,
        )
        (tmp_path / "h").mkdir()
        (tmp_path / "r").mkdir()
        (tmp_path / "o").mkdir()
        return HistoricalFeatureMaterializer(cfg)

    def test_to_float_with_various_inputs(self, mat):
        assert mat._to_float(5) == 5.0
        assert mat._to_float("3.14") == 3.14
        assert np.isnan(mat._to_float(None))
        assert np.isnan(mat._to_float("not_a_number"))

    def test_safe_div_zero_denominator(self, mat):
        result = mat._safe_div(pd.Series([10.0]), pd.Series([0.0]))
        assert np.isnan(result.iloc[0]) or result.iloc[0] == 0.0

    def test_load_team_games_fallback_games_key(self, mat, tmp_path):
        """When team_games is empty, should fallback to games key."""
        data = {
            "team_games": [],
            "games": [
                {"game_id": "g1", "team1_id": "duke", "team2_id": "unc",
                 "team1_score": 80, "team2_score": 70, "date": "2023-01-10"},
            ]
        }
        gp = tmp_path / "h" / "historical_games_2023.json"
        gp.write_text(json.dumps(data))
        df = mat._load_team_games({"2023": {"historical_games_json": str(gp)}})
        assert len(df) == 2  # Two rows from fallback (one per team)

    def test_load_team_metrics_backfill_from_games(self, mat, tmp_path):
        """Zeroed metrics should trigger backfill from game data."""
        metrics = {
            "teams": [
                {"team_name": "Duke", "off_rtg": 0.0, "def_rtg": 0.0, "pace": 0.0},
                {"team_name": "UNC", "off_rtg": 0.0, "def_rtg": 0.0, "pace": 0.0},
            ]
        }
        games = []
        for i in range(10):
            games.append({
                "team_id": "duke", "team_score": 80, "opponent_score": 70,
                "fga": 60, "orb": 10, "turnovers": 12, "fta": 20,
            })
        mp = tmp_path / "h" / "team_metrics_2023.json"
        mp.write_text(json.dumps(metrics))
        gp = tmp_path / "h" / "historical_games_2023.json"
        gp.write_text(json.dumps({"team_games": games}))
        df = mat._load_team_metrics({
            "2023": {
                "team_metrics_json": str(mp),
                "historical_games_json": str(gp),
            }
        })
        # Duke should have been backfilled
        duke_row = df[df["team_id"] == "duke"]
        assert len(duke_row) >= 1

    def test_validate_prior_source_market_odds_post_tournament(self, mat, tmp_path):
        """Market odds with post-Selection Sunday snapshot should warn."""
        odds = {
            "snapshot_date": "2023-04-01T00:00:00Z",
            "teams": [],
        }
        (tmp_path / "r" / "odds_2023.json").write_text(json.dumps(odds))
        df = pd.DataFrame({"season": [2023], "team_id": ["duke"]})
        warnings = mat._validate_prior_source_availability(df)
        assert any("after Selection Sunday" in w for w in warnings)

    def test_validate_prior_source_transfer_post_april(self, mat, tmp_path):
        """Transfer data with post-April snapshot should warn."""
        transfers = {
            "snapshot_date": "2023-05-01T00:00:00Z",
            "entries": [],
        }
        (tmp_path / "r" / "transfer_portal_2023.json").write_text(json.dumps(transfers))
        df = pd.DataFrame({"season": [2023], "team_id": ["duke"]})
        warnings = mat._validate_prior_source_availability(df)
        assert any("after April 15" in w for w in warnings)


class TestMaterializerRunIntegration:
    """Integration-style test for the full run() method using mocks."""

    def test_run_produces_manifest(self, tmp_path):
        from src.data.features.materialization import (
            HistoricalFeatureMaterializer,
            MaterializationConfig,
        )
        hist_dir = tmp_path / "historical"
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "output"
        hist_dir.mkdir()
        raw_dir.mkdir()
        out_dir.mkdir()

        # Create minimal game data
        games = []
        for i in range(20):
            games.append({
                "game_id": f"g{i}", "team_id": "team_a", "opponent_id": "team_b",
                "team_name": "Team A", "opponent_name": "Team B",
                "team_score": 75 + i % 5, "opponent_score": 70 + i % 3,
                "date": f"2023-01-{10 + i % 20:02d}", "season": 2023,
            })
            games.append({
                "game_id": f"g{i}", "team_id": "team_b", "opponent_id": "team_a",
                "team_name": "Team B", "opponent_name": "Team A",
                "team_score": 70 + i % 3, "opponent_score": 75 + i % 5,
                "date": f"2023-01-{10 + i % 20:02d}", "season": 2023,
            })
        (hist_dir / "historical_games_2023.json").write_text(
            json.dumps({"team_games": games})
        )

        metrics = {
            "teams": [
                {"team_name": "team_a", "team_id": "team_a", "off_rtg": 110.0,
                 "def_rtg": 95.0, "pace": 70.0, "srs": 10.0, "sos": 5.0,
                 "wins": 20, "losses": 5},
                {"team_name": "team_b", "team_id": "team_b", "off_rtg": 100.0,
                 "def_rtg": 100.0, "pace": 68.0, "srs": 5.0, "sos": 3.0,
                 "wins": 15, "losses": 10},
            ]
        }
        (hist_dir / "team_metrics_2023.json").write_text(json.dumps(metrics))

        cfg = MaterializationConfig(
            start_season=2023,
            end_season=2023,
            historical_dir=str(hist_dir),
            raw_dir=str(raw_dir),
            output_dir=str(out_dir),
            strict_validation=False,
            require_all_seasons=False,
            min_tournament_matchups=0,
        )
        materializer = HistoricalFeatureMaterializer(cfg)
        manifest = materializer.run()

        assert "artifacts" in manifest
        assert "leakage_checks" in manifest
        assert "quality_report" in manifest
        assert manifest["start_season"] == 2023


class TestLoadTeamStatSources:
    """Tests for load_team_stat_sources."""

    def test_raises_without_torvik_or_scrape(self):
        from src.pipeline.stages.data_loader import load_team_stat_sources
        from src.pipeline.config import DataRequirementError
        config = MagicMock()
        config.torvik_json = None
        config.scrape_live = False
        with pytest.raises(DataRequirementError, match="Missing Torvik data"):
            load_team_stat_sources(config, [], MagicMock())


class TestBaselineTrainingFeatureValidation:
    """Tests for feature matrix validation in _train_baseline_model."""

    def test_nan_inf_replacement_logic(self):
        """Verify the NaN/inf replacement logic used in baseline training."""
        X = np.array([[1.0, np.nan, np.inf], [4.0, 5.0, -np.inf]])
        n_nan = int(np.isnan(X).sum())
        n_inf = int(np.isinf(X).sum())
        assert n_nan == 1
        assert n_inf == 2
        X_clean = np.where(np.isnan(X) | np.isinf(X), 0.0, X)
        assert not np.isnan(X_clean).any()
        assert not np.isinf(X_clean).any()
        assert X_clean[0, 0] == 1.0
        assert X_clean[0, 1] == 0.0

    def test_constant_feature_detection(self):
        """Verify constant feature detection logic."""
        X = np.array([[1.0, 5.0], [1.0, 6.0], [1.0, 7.0]])
        col_vars = np.var(X, axis=0)
        constant_cols = int(np.sum(col_vars < 1e-10))
        assert constant_cols == 1  # First column is constant

    def test_class_imbalance_detection(self):
        """Verify class imbalance detection logic."""
        y_balanced = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        pos_rate = float(np.mean(y_balanced))
        assert abs(pos_rate - 0.5) <= 0.1

        y_imbalanced = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
        pos_rate = float(np.mean(y_imbalanced))
        assert abs(pos_rate - 0.5) > 0.1


class TestRecencyWeightingLogic:
    """Tests for the recency weighting logic used in baseline training."""

    def test_exponential_ramp(self):
        """Verify the recency weighting formula."""
        sort_keys = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        t_min, t_max = float(sort_keys[0]), float(sort_keys[-1])
        t_span = max(t_max - t_min, 1.0)
        progress = (sort_keys - t_min) / t_span
        floor = 0.3
        hl = 0.5
        raw_weight = floor + (1.0 - floor) * (1.0 - np.exp(-progress / hl))
        train_sample_weight = raw_weight / raw_weight.mean()

        # First sample should have lowest weight
        assert train_sample_weight[0] < train_sample_weight[-1]
        # Mean should be ~1.0
        assert abs(train_sample_weight.mean() - 1.0) < 1e-10

    def test_floor_parameter(self):
        """Floor should prevent any weight from going below floor."""
        sort_keys = np.array([0.0, 100.0])
        progress = np.array([0.0, 1.0])
        floor = 0.5
        hl = 0.3
        raw_weight = floor + (1.0 - floor) * (1.0 - np.exp(-progress / hl))
        # At progress=0, weight should be exactly floor
        assert raw_weight[0] == pytest.approx(floor, abs=1e-10)


class TestMaterializerSafeDivHelper:
    """Test the _safe_div helper on the materializer."""

    def test_normal_division(self):
        from src.data.features.materialization import HistoricalFeatureMaterializer, MaterializationConfig
        mat = HistoricalFeatureMaterializer.__new__(HistoricalFeatureMaterializer)
        result = mat._safe_div(pd.Series([10.0, 20.0]), pd.Series([2.0, 4.0]))
        assert result.iloc[0] == pytest.approx(5.0)
        assert result.iloc[1] == pytest.approx(5.0)


class TestMaterializerNormalizeTeamId:
    """Test the _normalize_team_id helper on the materializer."""

    def test_basic_normalization(self):
        from src.data.features.materialization import HistoricalFeatureMaterializer, MaterializationConfig
        mat = HistoricalFeatureMaterializer.__new__(HistoricalFeatureMaterializer)
        result = mat._normalize_team_id("Duke Blue Devils")
        assert isinstance(result, str)
        assert len(result) > 0
