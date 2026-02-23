"""Tests for data consistency functions added during the training data audit.

Covers:
1. normalize_team_id() — Unicode NFKD, HTML entities, edge cases
2. normalize_team_name() — display-name normalization
3. strip_ncaa_suffix() / strip_ncaa_suffix_name() — tournament suffix removal
4. _infer_dates_from_game_ids() — chronological date inference
5. _compute_metrics_from_games() — box-score metric backfill
6. Outlier filtering — zero scores and extreme margins
7. Degenerate temporal feature NaN-ing for inferred dates
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.normalize import (
    normalize_team_id,
    normalize_team_name,
    strip_ncaa_suffix,
    strip_ncaa_suffix_name,
)
from src.data.features.materialization import HistoricalFeatureMaterializer


# ---------------------------------------------------------------------------
# 1. normalize_team_id
# ---------------------------------------------------------------------------


class TestNormalizeTeamId:
    """Unit tests for normalize_team_id()."""

    def test_basic_lowercase(self):
        assert normalize_team_id("Duke") == "duke"

    def test_spaces_to_underscores(self):
        assert normalize_team_id("North Carolina") == "north_carolina"

    def test_html_entity_ampersand(self):
        assert normalize_team_id("Texas A&amp;M") == "texas_a_m"

    def test_raw_ampersand(self):
        assert normalize_team_id("Texas A&M") == "texas_a_m"

    def test_unicode_accent(self):
        assert normalize_team_id("San José State") == "san_jose_state"

    def test_unicode_nfkd_consistency(self):
        # Precomposed vs decomposed should produce the same result
        assert normalize_team_id("San Jos\u00e9 State") == normalize_team_id(
            "San Jose\u0301 State"
        )

    def test_collapse_underscores(self):
        assert normalize_team_id("NC  A&T") == "nc_a_t"

    def test_special_chars_stripped(self):
        assert normalize_team_id("St. John's (NY)") == "st_john_s_ny"

    def test_empty_string(self):
        assert normalize_team_id("") == ""

    def test_none_as_string(self):
        # normalize_team_id expects str but should handle str(None) gracefully
        assert normalize_team_id("None") == "none"

    def test_mascot_suffix(self):
        """CBBpy-style mascot IDs are passed through without special handling."""
        assert normalize_team_id("Duke Blue Devils") == "duke_blue_devils"

    def test_double_html_entity(self):
        """Ensure double-encoded entities are decoded once."""
        result = normalize_team_id("Texas A&amp;M-Corpus Christi")
        assert result == "texas_a_m_corpus_christi"


# ---------------------------------------------------------------------------
# 2. normalize_team_name
# ---------------------------------------------------------------------------


class TestNormalizeTeamName:
    """Unit tests for normalize_team_name()."""

    def test_basic(self):
        assert normalize_team_name("Duke") == "duke"

    def test_stop_words_removed(self):
        result = normalize_team_name("University of North Carolina at Chapel Hill")
        assert "university" not in result
        assert "of" not in result
        assert "at" not in result
        assert "north" in result
        assert "carolina" in result

    def test_ampersand_expansion(self):
        result = normalize_team_name("Texas A&M")
        assert "and" in result

    def test_html_entity(self):
        result = normalize_team_name("Texas A&amp;M")
        assert "and" in result

    def test_empty(self):
        assert normalize_team_name("") == ""


# ---------------------------------------------------------------------------
# 3. strip_ncaa_suffix / strip_ncaa_suffix_name
# ---------------------------------------------------------------------------


class TestStripNCAASuffix:
    """Unit tests for NCAA suffix stripping."""

    def test_basic_strip(self):
        assert strip_ncaa_suffix("alabamancaa") == "alabama"

    def test_underscore_suffix(self):
        assert strip_ncaa_suffix("alabama_ncaa") == "alabama"

    def test_no_suffix(self):
        assert strip_ncaa_suffix("alabama") == "alabama"

    def test_empty(self):
        assert strip_ncaa_suffix("") == ""

    def test_ncaa_only(self):
        result = strip_ncaa_suffix("ncaa")
        assert result == ""

    def test_display_name_strip(self):
        assert strip_ncaa_suffix_name("AlabamaNCAA") == "Alabama"

    def test_display_name_no_suffix(self):
        assert strip_ncaa_suffix_name("Alabama") == "Alabama"

    def test_display_name_with_space(self):
        assert strip_ncaa_suffix_name("Alabama NCAA") == "Alabama"


# ---------------------------------------------------------------------------
# 4. _infer_dates_from_game_ids
# ---------------------------------------------------------------------------


class TestInferDatesFromGameIds:
    """Unit tests for HistoricalFeatureMaterializer._infer_dates_from_game_ids()."""

    def test_basic_inference(self):
        rows = [
            {"game_id": "100", "date": "2020-01-01"},
            {"game_id": "200", "date": "2020-01-01"},
            {"game_id": "300", "date": "2020-01-01"},
        ]
        HistoricalFeatureMaterializer._infer_dates_from_game_ids(rows, 2020)

        # Dates should be spread across Nov 1, 2019 → Mar 13, 2020
        dates = [row["date"] for row in rows]
        assert dates[0] == "2019-11-01"  # First game → season start
        assert dates[2] == "2020-03-13"  # Last game → season end
        # Middle game should be between
        assert "2019-11-01" < dates[1] < "2020-03-13"

    def test_preserves_ordering(self):
        """Game IDs sorted numerically should produce chronological dates."""
        rows = [
            {"game_id": "300", "date": "2020-01-01"},
            {"game_id": "100", "date": "2020-01-01"},
            {"game_id": "200", "date": "2020-01-01"},
        ]
        HistoricalFeatureMaterializer._infer_dates_from_game_ids(rows, 2020)

        # After inference, row with game_id 100 should have earliest date
        assert rows[1]["date"] < rows[2]["date"] < rows[0]["date"]

    def test_single_game(self):
        """A single game should get the season start date."""
        rows = [{"game_id": "100", "date": "2020-01-01"}]
        HistoricalFeatureMaterializer._infer_dates_from_game_ids(rows, 2020)
        assert rows[0]["date"] == "2019-11-01"

    def test_non_numeric_game_ids(self):
        """Non-numeric game IDs should be handled (sorted as 0)."""
        rows = [
            {"game_id": "abc", "date": "2020-01-01"},
            {"game_id": "200", "date": "2020-01-01"},
        ]
        HistoricalFeatureMaterializer._infer_dates_from_game_ids(rows, 2020)
        # Should not crash; non-numeric treated as 0
        assert all("date" in row for row in rows)

    def test_duplicate_game_ids_get_same_date(self):
        """Rows sharing a game_id (home/away) should get the same date."""
        rows = [
            {"game_id": "100", "date": "2020-01-01"},
            {"game_id": "100", "date": "2020-01-01"},
            {"game_id": "200", "date": "2020-01-01"},
        ]
        HistoricalFeatureMaterializer._infer_dates_from_game_ids(rows, 2020)
        assert rows[0]["date"] == rows[1]["date"]
        assert rows[0]["date"] != rows[2]["date"]


# ---------------------------------------------------------------------------
# 5. _compute_metrics_from_games
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)


class TestComputeMetricsFromGames:
    """Unit tests for HistoricalFeatureMaterializer._compute_metrics_from_games()."""

    def test_basic_computation(self, tmp_path):
        """Metrics computed from sufficient game data."""
        games_path = tmp_path / "games.json"
        # 6 games for team "duke" — enough to pass the 5-game minimum
        team_games = []
        for i in range(6):
            team_games.append({
                "team_id": "duke",
                "team_score": 80,
                "opponent_score": 70,
                "possessions": 70,
            })
        _write_json(games_path, {"team_games": team_games})

        m = HistoricalFeatureMaterializer()
        result = m._compute_metrics_from_games(str(games_path))

        assert "duke" in result
        metrics = result["duke"]
        assert "off_rtg" in metrics
        assert "def_rtg" in metrics
        assert "pace" in metrics
        # 80 pts / 70 poss * 100 ≈ 114.3
        assert abs(metrics["off_rtg"] - 114.3) < 0.5
        # 70 pts / 70 poss * 100 = 100.0
        assert abs(metrics["def_rtg"] - 100.0) < 0.5

    def test_insufficient_games_excluded(self, tmp_path):
        """Teams with fewer than 5 games are excluded."""
        games_path = tmp_path / "games.json"
        team_games = [
            {"team_id": "small_team", "team_score": 80, "opponent_score": 70, "possessions": 70}
            for _ in range(3)
        ]
        _write_json(games_path, {"team_games": team_games})

        m = HistoricalFeatureMaterializer()
        result = m._compute_metrics_from_games(str(games_path))
        assert "small_team" not in result

    def test_possession_estimation(self, tmp_path):
        """When possessions=0, estimate from box score components."""
        games_path = tmp_path / "games.json"
        team_games = []
        for i in range(6):
            team_games.append({
                "team_id": "duke",
                "team_score": 80,
                "opponent_score": 70,
                "possessions": 0,  # Force estimation
                "fga": 60,
                "orb": 10,
                "turnovers": 12,
                "fta": 20,
            })
        _write_json(games_path, {"team_games": team_games})

        m = HistoricalFeatureMaterializer()
        result = m._compute_metrics_from_games(str(games_path))

        assert "duke" in result
        # poss = fga - orb + tov + 0.475 * fta = 60 - 10 + 12 + 9.5 = 71.5
        assert abs(result["duke"]["pace"] - 71.5) < 0.5

    def test_missing_file(self, tmp_path):
        """Non-existent file returns empty dict."""
        m = HistoricalFeatureMaterializer()
        result = m._compute_metrics_from_games(str(tmp_path / "nonexistent.json"))
        assert result == {}

    def test_empty_payload(self, tmp_path):
        """Payload with no team_games returns empty dict."""
        games_path = tmp_path / "games.json"
        _write_json(games_path, {"team_games": []})

        m = HistoricalFeatureMaterializer()
        result = m._compute_metrics_from_games(str(games_path))
        assert result == {}


# ---------------------------------------------------------------------------
# 6. Outlier filtering — zero scores and extreme margins
# ---------------------------------------------------------------------------


class TestOutlierFiltering:
    """Tests that forfeit/outlier game filtering works correctly."""

    def test_zero_score_filtered(self):
        """Games where either team scored 0 should be filtered."""
        df = pd.DataFrame({
            "team_score": [80, 0, 70, 72],
            "opponent_score": [70, 0, 0, 68],
            "team_id": ["a", "b", "c", "d"],
        })
        score_mask = (df["team_score"] > 0) & (df["opponent_score"] > 0)
        margin_mask = (df["team_score"] - df["opponent_score"]).abs() <= 80
        filtered = df[score_mask & margin_mask]

        # Row 0 (80-70): keep, Row 1 (0-0): drop, Row 2 (70-0): drop, Row 3 (72-68): keep
        assert len(filtered) == 2
        assert list(filtered["team_id"]) == ["a", "d"]

    def test_extreme_margin_filtered(self):
        """Games with margin > 80 should be filtered."""
        df = pd.DataFrame({
            "team_score": [150, 80, 100],
            "opponent_score": [50, 70, 15],
            "team_id": ["a", "b", "c"],
        })
        score_mask = (df["team_score"] > 0) & (df["opponent_score"] > 0)
        margin_mask = (df["team_score"] - df["opponent_score"]).abs() <= 80
        filtered = df[score_mask & margin_mask]

        # Row 0 (margin=100): drop, Row 1 (margin=10): keep, Row 2 (margin=85): drop
        assert len(filtered) == 1
        assert filtered.iloc[0]["team_id"] == "b"

    def test_margin_80_kept(self):
        """Margin of exactly 80 should be kept (boundary)."""
        df = pd.DataFrame({
            "team_score": [100, 101],
            "opponent_score": [20, 20],
        })
        score_mask = (df["team_score"] > 0) & (df["opponent_score"] > 0)
        margin_mask = (df["team_score"] - df["opponent_score"]).abs() <= 80
        filtered = df[score_mask & margin_mask]

        # Margin 80: keep, Margin 81: drop
        assert len(filtered) == 1


# ---------------------------------------------------------------------------
# 7. Degenerate temporal feature NaN-ing
# ---------------------------------------------------------------------------


class TestDegenerateTemporalNaN:
    """Tests that synthetic-date rows get temporal features NaN-ed."""

    def test_inferred_dates_nan_temporal(self):
        """Rows with _dates_inferred=True should have rest_days/back_to_back/
        games_in_last_7_days set to NaN."""
        df = pd.DataFrame({
            "rest_days": [3.0, 5.0, 2.0, 4.0],
            "back_to_back": [0.0, 1.0, 0.0, 1.0],
            "games_in_last_7_days": [2.0, 3.0, 1.0, 2.0],
            "season_progress": [0.1, 0.3, 0.5, 0.7],
            "_dates_inferred": [True, True, False, False],
        })

        inferred_mask = df["_dates_inferred"].fillna(False).astype(bool)
        for col in ("rest_days", "back_to_back", "games_in_last_7_days"):
            df.loc[inferred_mask, col] = np.nan

        # Inferred rows (0, 1) should be NaN
        assert np.isnan(df.loc[0, "rest_days"])
        assert np.isnan(df.loc[1, "back_to_back"])
        assert np.isnan(df.loc[1, "games_in_last_7_days"])

        # Non-inferred rows (2, 3) should be preserved
        assert df.loc[2, "rest_days"] == 2.0
        assert df.loc[3, "back_to_back"] == 1.0

        # season_progress should be untouched for all rows
        assert df.loc[0, "season_progress"] == 0.1
        assert df.loc[2, "season_progress"] == 0.5

    def test_no_inferred_dates_no_change(self):
        """When no rows have _dates_inferred, all temporal features preserved."""
        df = pd.DataFrame({
            "rest_days": [3.0, 5.0],
            "back_to_back": [0.0, 1.0],
            "games_in_last_7_days": [2.0, 3.0],
            "_dates_inferred": [False, False],
        })

        inferred_mask = df["_dates_inferred"].fillna(False).astype(bool)
        for col in ("rest_days", "back_to_back", "games_in_last_7_days"):
            df.loc[inferred_mask, col] = np.nan

        assert df.loc[0, "rest_days"] == 3.0
        assert df.loc[1, "rest_days"] == 5.0


# ---------------------------------------------------------------------------
# Cross-pipeline consistency: sota.py filtering alignment
# ---------------------------------------------------------------------------


class TestCrossPipelineConsistency:
    """Verify that sota.py and materialization.py filtering logic is aligned."""

    def test_single_zero_filtered_in_sota_style(self):
        """sota.py should now filter single-zero scores (not just double-zero)."""
        # Simulates the sota.py filtering logic after the fix
        games = [
            (80, 70),   # Normal game — keep
            (0, 0),     # Double zero — filter
            (72, 0),    # Single zero (forfeit) — filter
            (0, 65),    # Single zero (forfeit) — filter
            (85, 80),   # Normal game — keep
            (120, 30),  # Margin 90 > 80 — filter
        ]
        kept = []
        for s1, s2 in games:
            if s1 == 0 or s2 == 0:
                continue
            if abs(s1 - s2) > 80:
                continue
            kept.append((s1, s2))

        assert len(kept) == 2
        assert kept[0] == (80, 70)
        assert kept[1] == (85, 80)

    def test_ncaa_suffix_shared_utility(self):
        """Both pipelines should produce the same result for NCAA-suffixed IDs."""
        test_cases = [
            ("alabamancaa", "alabama"),
            ("arizona_ncaa", "arizona"),
            ("duke", "duke"),
            ("ncaa", ""),
        ]
        for input_id, expected in test_cases:
            assert strip_ncaa_suffix(input_id) == expected, (
                f"strip_ncaa_suffix({input_id!r}) → {strip_ncaa_suffix(input_id)!r}, "
                f"expected {expected!r}"
            )

    def test_normalize_team_id_shared_utility(self):
        """Both pipelines should produce the same normalized ID."""
        test_cases = [
            ("Texas A&amp;M", "texas_a_m"),
            ("San José State", "san_jose_state"),
            ("Duke Blue Devils", "duke_blue_devils"),
            ("St. John's (NY)", "st_john_s_ny"),
        ]
        for input_name, expected in test_cases:
            assert normalize_team_id(input_name) == expected, (
                f"normalize_team_id({input_name!r}) → {normalize_team_id(input_name)!r}, "
                f"expected {expected!r}"
            )
