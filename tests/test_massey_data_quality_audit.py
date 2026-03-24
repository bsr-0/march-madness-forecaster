"""Massey Ordinals data quality audit tests.

These tests validate fixes for 6 data quality issues identified during audit:

1. Duplicate team entries within a system-day silently overwritten (now warns)
2. Out-of-range rank values accepted without validation (now filtered/logged)
3. Composite cache included corrupted systems (now applies corruption check)
4. Post-Selection-Sunday data used when no safe days exist (now skips system)
5. Low-coverage systems skewed normalization in multi-system path (now filtered)
6. Normalization consistency between extraction paths
"""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path
from typing import Dict, List

import pytest

from src.data.features.massey_systems import (
    MASSEY_TOP_SYSTEMS,
    extract_all_teams,
    extract_multi_system_features,
    features_to_vector,
)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, headers: list, rows: list):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def _make_kaggle_dir(tmp_path: Path, *, n_teams: int = 80, systems: list | None = None,
                     day: int = 128, season: int = 2025,
                     extra_ordinal_rows: list | None = None) -> Path:
    """Create a Kaggle dir with customizable ordinals."""
    kaggle_dir = tmp_path / "kaggle"
    kaggle_dir.mkdir(exist_ok=True)

    # MTeams
    team_rows = [[str(1100 + i), f"Team{i}"] for i in range(n_teams)]
    _write_csv(kaggle_dir / "MTeams.csv", ["TeamID", "TeamName"], team_rows)

    # MSeasons (for DayZero computation)
    _write_csv(kaggle_dir / "MSeasons.csv",
               ["Season", "DayZero", "RegionW", "RegionX", "RegionY", "RegionZ"],
               [[str(season), "2024-10-14", "W", "X", "Y", "Z"]])

    # Ordinals
    if systems is None:
        systems = ["POM", "SAG", "MOR"]
    ordinal_rows = []
    for system in systems:
        for i in range(n_teams):
            ordinal_rows.append([str(season), str(day), system, str(1100 + i), str(i + 1)])
    if extra_ordinal_rows:
        ordinal_rows.extend(extra_ordinal_rows)
    _write_csv(
        kaggle_dir / "MMasseyOrdinals.csv",
        ["Season", "RankingDayNum", "SystemName", "TeamID", "OrdinalRank"],
        ordinal_rows,
    )

    return kaggle_dir


# ===========================================================================
# Issue 1: Duplicate team entries
# ===========================================================================

class TestDuplicateTeamEntries:
    """Verify duplicate team entries within a system-day are detected."""

    def test_duplicate_team_logs_warning(self, tmp_path, caplog):
        """Duplicate team entries should produce a warning."""
        from src.data.kaggle_loader import KaggleDataLoader

        # Create ordinals with duplicate entries for team 1100 in POM
        extra = [
            ["2025", "128", "POM", "1100", "5"],  # duplicate with different rank
        ]
        kaggle_dir = _make_kaggle_dir(tmp_path, n_teams=80, extra_ordinal_rows=extra)
        loader = KaggleDataLoader(str(kaggle_dir))

        with caplog.at_level(logging.WARNING):
            result = loader.load_massey_ordinals(2025, max_day=200)

        # Should still load (last entry wins) but with warning
        assert "POM" in result
        assert any("duplicate team" in msg.lower() for msg in caplog.messages)

    def test_no_duplicates_no_warning(self, tmp_path, caplog):
        """Clean data should produce no duplicate warning."""
        from src.data.kaggle_loader import KaggleDataLoader

        kaggle_dir = _make_kaggle_dir(tmp_path, n_teams=80)
        loader = KaggleDataLoader(str(kaggle_dir))

        with caplog.at_level(logging.WARNING):
            loader.load_massey_ordinals(2025, max_day=200)

        assert not any("duplicate team" in msg.lower() for msg in caplog.messages)


# ===========================================================================
# Issue 2: Out-of-range rank validation
# ===========================================================================

class TestRankRangeValidation:
    """Validate that invalid ranks are handled correctly."""

    def test_zero_rank_filtered(self, tmp_path):
        """Rank <= 0 should be filtered out during loading."""
        from src.data.kaggle_loader import KaggleDataLoader

        extra = [["2025", "128", "DOL", "1100", "0"]]
        kaggle_dir = _make_kaggle_dir(tmp_path, n_teams=80, extra_ordinal_rows=extra)
        loader = KaggleDataLoader(str(kaggle_dir))
        result = loader.load_massey_ordinals(2025, max_day=200)

        # DOL should not have an entry for team 1100 (rank 0 filtered)
        if "DOL" in result:
            team_entry = result["DOL"].get(loader._canonical_id(1100))
            if team_entry is not None:
                assert team_entry.ordinal_rank > 0

    def test_negative_rank_filtered(self, tmp_path):
        """Negative ranks should be filtered out."""
        from src.data.kaggle_loader import KaggleDataLoader

        extra = [["2025", "128", "DOL", "1100", "-5"]]
        kaggle_dir = _make_kaggle_dir(tmp_path, n_teams=80, extra_ordinal_rows=extra)
        loader = KaggleDataLoader(str(kaggle_dir))
        result = loader.load_massey_ordinals(2025, max_day=200)

        if "DOL" in result:
            for entry in result["DOL"].values():
                assert entry.ordinal_rank > 0


# ===========================================================================
# Issue 3: Composite corruption check
# ===========================================================================

class TestCompositeCorruptionFilter:
    """Verify corrupted systems are excluded from composite."""

    def test_corrupted_system_excluded_from_composite(self, tmp_path):
        """A system where >80% entries share the same rank should be excluded."""
        from src.data.scrapers.external_ratings import ExternalRatingsLoader

        # Build a "corrupted" system where all teams have rank 1
        n_teams = 100
        corrupted_entries = [
            {"team_name": f"Team{i}", "team_id": f"team_{i}",
             "rating": 100.0, "ranking": 1, "normalized": 1.0}
            for i in range(n_teams)
        ]
        # Build a clean system
        clean_entries = [
            {"team_name": f"Team{i}", "team_id": f"team_{i}",
             "rating": float(n_teams - i), "ranking": i + 1,
             "normalized": round(1.0 - i / max(n_teams - 1, 1), 6)}
            for i in range(n_teams)
        ]

        all_systems = {
            "CORRUPT_SYS": corrupted_entries,
            "CLEAN_SYS": clean_entries,
        }

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        loader = ExternalRatingsLoader(cache_dir=str(cache_dir))

        # Directly test _build_massey_composite_cache
        loader._build_massey_composite_cache(2025, all_systems)

        # Load the composite and verify it doesn't contain corrupted data
        all_loaded = loader.load_all(2025)
        ratings = all_loaded.get("massey_composite", {})
        if ratings:
            # All teams should have normalized values from CLEAN_SYS only
            # (not inflated by the corrupted system giving everyone 1.0)
            normalized_vals = [r.normalized for r in ratings.values()]
            # If corrupted system were included, mean would be ~0.75
            # With only clean system, mean should be ~0.5
            mean_norm = sum(normalized_vals) / len(normalized_vals)
            assert mean_norm < 0.6, (
                f"Composite mean {mean_norm:.3f} too high — "
                "corrupted system may have leaked into composite"
            )

    def test_malformed_entries_missing_ranking_key(self, tmp_path):
        """Entries without a 'ranking' key should not falsely trigger corruption."""
        from src.data.scrapers.external_ratings import ExternalRatingsLoader

        n_teams = 100
        # Entries missing "ranking" entirely — should NOT be treated as corrupted
        malformed_entries = [
            {"team_name": f"Team{i}", "team_id": f"team_{i}",
             "rating": float(n_teams - i), "normalized": round(1.0 - i / max(n_teams - 1, 1), 6)}
            for i in range(n_teams)
        ]

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        loader = ExternalRatingsLoader(cache_dir=str(cache_dir))

        # Should not crash and should include the system in composite
        # (missing ranking means we can't detect corruption, so we allow it)
        all_systems = {"MALFORMED_SYS": malformed_entries}
        loader._build_massey_composite_cache(2025, all_systems)

        all_loaded = loader.load_all(2025)
        ratings = all_loaded.get("massey_composite", {})
        assert len(ratings) == n_teams, (
            "System with missing ranking keys should still contribute to composite"
        )

    def test_mixed_corrupted_and_clean_composite_values(self, tmp_path):
        """Composite with 2 clean + 1 corrupted: only clean systems contribute."""
        from src.data.scrapers.external_ratings import ExternalRatingsLoader

        n_teams = 100
        # Two clean systems with different orderings
        clean_a = [
            {"team_name": f"Team{i}", "team_id": f"team_{i}",
             "rating": float(n_teams - i), "ranking": i + 1,
             "normalized": round(1.0 - i / max(n_teams - 1, 1), 6)}
            for i in range(n_teams)
        ]
        clean_b = [
            {"team_name": f"Team{i}", "team_id": f"team_{i}",
             "rating": float(n_teams - i), "ranking": i + 1,
             "normalized": round(1.0 - i / max(n_teams - 1, 1), 6)}
            for i in range(n_teams)
        ]
        # Corrupted: all rank 1
        corrupted = [
            {"team_name": f"Team{i}", "team_id": f"team_{i}",
             "rating": 100.0, "ranking": 1, "normalized": 1.0}
            for i in range(n_teams)
        ]

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        loader = ExternalRatingsLoader(cache_dir=str(cache_dir))

        all_systems = {"CLEAN_A": clean_a, "CLEAN_B": clean_b, "CORRUPT": corrupted}
        loader._build_massey_composite_cache(2025, all_systems)

        all_loaded = loader.load_all(2025)
        ratings = all_loaded.get("massey_composite", {})
        assert len(ratings) == n_teams

        # Verify top-ranked team has highest composite, last has lowest
        top = ratings.get("team_0")
        bottom = ratings.get(f"team_{n_teams - 1}")
        assert top is not None and bottom is not None
        assert top.normalized > bottom.normalized

        # Mean should be ~0.5 (two clean systems averaging out)
        mean_norm = sum(r.normalized for r in ratings.values()) / len(ratings)
        assert 0.45 < mean_norm < 0.55, (
            f"Composite mean {mean_norm:.3f} should be ~0.5 from clean systems only"
        )


# ===========================================================================
# Issue 4: Leakage hardening
# ===========================================================================

class TestLeakageHardening:
    """Verify post-Selection-Sunday data is refused, not just warned."""

    def test_system_skipped_when_only_post_ceiling_days(self, tmp_path):
        """System should be skipped entirely when all its data is post-max_day."""
        from src.data.kaggle_loader import KaggleDataLoader

        # Create system with data only on day 200 (way past any Selection Sunday)
        kaggle_dir = _make_kaggle_dir(tmp_path, n_teams=80, day=200)
        loader = KaggleDataLoader(str(kaggle_dir))

        # Set max_day to 150 — all data is post-ceiling
        result = loader.load_massey_ordinals(2025, max_day=150)

        # Systems should be empty (skipped) rather than using unsafe data
        assert len(result) == 0, (
            f"Expected 0 systems when all data is post-ceiling, got {len(result)}"
        )

    def test_leakage_protection_through_external_ratings_wrapper(self, tmp_path):
        """load_massey_ordinals_as_external_ratings with max_day filters properly."""
        from src.data.kaggle_loader import KaggleDataLoader

        # All data on day 200, max_day=150 — should return empty
        kaggle_dir = _make_kaggle_dir(tmp_path, n_teams=80, day=200)
        loader = KaggleDataLoader(str(kaggle_dir))

        result = loader.load_massey_ordinals_as_external_ratings(2025, max_day=150)
        assert result == {}, (
            "External ratings wrapper should return empty when all data is post-ceiling"
        )

    def test_multiple_systems_skipped_all_logged(self, tmp_path, caplog):
        """Each system skipped for leakage should generate its own warning."""
        from src.data.kaggle_loader import KaggleDataLoader

        # Create 3 systems all with data only on day 200
        kaggle_dir = _make_kaggle_dir(
            tmp_path, n_teams=80, day=200,
            systems=["POM", "SAG", "MOR"],
        )
        loader = KaggleDataLoader(str(kaggle_dir))

        with caplog.at_level(logging.WARNING):
            result = loader.load_massey_ordinals(2025, max_day=150)

        assert len(result) == 0
        # Each system should have its own skip warning
        skip_warnings = [
            msg for msg in caplog.messages
            if "skipping system to prevent leakage" in msg.lower()
        ]
        assert len(skip_warnings) == 3, (
            f"Expected 3 skip warnings (one per system), got {len(skip_warnings)}: "
            f"{skip_warnings}"
        )


# ===========================================================================
# Issue 5: Low-coverage system filtering in multi-system path
# ===========================================================================

class TestLowCoverageFiltering:
    """Verify low-coverage systems are filtered in extract_multi_system_features."""

    def test_system_with_few_teams_skipped(self):
        """Systems with <50 teams should be skipped during feature extraction."""

        class _MockEntry:
            def __init__(self, ordinal_rank):
                self.ordinal_rank = ordinal_rank

        # POM has 300 teams (good coverage)
        pom_data = {f"team_{i}": _MockEntry(i + 1) for i in range(300)}
        # AP has only 25 teams (too few)
        ap_data = {f"team_{i}": _MockEntry(i + 1) for i in range(25)}

        ordinals = {"POM": pom_data, "AP": ap_data}
        features = extract_multi_system_features(ordinals, "team_0", n_teams_approx=300)

        assert "POM" in features.system_ratings
        assert "AP" not in features.system_ratings, (
            "AP with only 25 teams should be excluded from multi-system features"
        )

    def test_system_at_threshold_included(self):
        """Systems with exactly 50 teams should be included."""

        class _MockEntry:
            def __init__(self, ordinal_rank):
                self.ordinal_rank = ordinal_rank

        pom_data = {f"team_{i}": _MockEntry(i + 1) for i in range(50)}
        ordinals = {"POM": pom_data}
        features = extract_multi_system_features(ordinals, "team_0", n_teams_approx=50)

        assert "POM" in features.system_ratings


# ===========================================================================
# Issue 6: Normalization consistency
# ===========================================================================

class TestNormalizationConsistency:
    """Verify normalization is consistent across code paths."""

    def test_normalization_bounded_0_1(self):
        """All normalized values should be in [0.0, 1.0]."""

        class _MockEntry:
            def __init__(self, ordinal_rank):
                self.ordinal_rank = ordinal_rank

        n = 363
        ordinals = {"POM": {f"team_{i}": _MockEntry(i + 1) for i in range(n)}}

        for team_id in [f"team_{i}" for i in range(n)]:
            features = extract_multi_system_features(ordinals, team_id, n_teams_approx=n)
            if "POM" in features.system_ratings:
                val = features.system_ratings["POM"]
                assert 0.0 <= val <= 1.0, f"Normalized value {val} out of bounds"

    def test_rank_1_normalizes_to_1(self):
        """Rank 1 should always normalize to 1.0."""

        class _MockEntry:
            def __init__(self, ordinal_rank):
                self.ordinal_rank = ordinal_rank

        ordinals = {"POM": {f"team_{i}": _MockEntry(i + 1) for i in range(100)}}
        features = extract_multi_system_features(ordinals, "team_0", n_teams_approx=100)
        assert features.system_ratings["POM"] == 1.0

    def test_last_rank_normalizes_near_0(self):
        """Last-ranked team should normalize to ~0.0."""

        class _MockEntry:
            def __init__(self, ordinal_rank):
                self.ordinal_rank = ordinal_rank

        n = 363
        ordinals = {"POM": {f"team_{i}": _MockEntry(i + 1) for i in range(n)}}
        features = extract_multi_system_features(ordinals, f"team_{n - 1}", n_teams_approx=n)
        assert features.system_ratings["POM"] < 0.01

    def test_features_to_vector_nan_for_missing(self):
        """Missing systems should produce NaN, not 0.0."""
        from src.data.features.massey_systems import MasseyMultiSystemFeatures

        features = MasseyMultiSystemFeatures(
            system_ratings={"POM": 0.9},
            rank_mean=0.9,
            rank_std=0.0,
            n_systems=1,
        )
        vec = features_to_vector(features)
        # SAG (index 1) should be NaN, not 0.0
        assert math.isnan(vec[1]), "Missing system should produce NaN, not 0.0"
        # POM (index 0) should be the actual value
        assert abs(vec[0] - 0.9) < 1e-6


# ===========================================================================
# End-to-end: full pipeline data quality
# ===========================================================================

class TestEndToEndDataQuality:
    """End-to-end tests for Massey data quality through the pipeline."""

    def test_extract_all_teams_returns_valid_features(self):
        """extract_all_teams should produce features with valid statistics."""

        class _MockEntry:
            def __init__(self, ordinal_rank):
                self.ordinal_rank = ordinal_rank

        n = 100
        ordinals = {
            system: {f"team_{i}": _MockEntry(i + 1) for i in range(n)}
            for system in ["POM", "SAG", "MOR", "DOL", "COL", "WOL", "RTH"]
        }
        result = extract_all_teams(ordinals)

        assert len(result) == n
        for team_id, features in result.items():
            assert features.n_systems == 7
            assert not math.isnan(features.rank_mean)
            assert 0.0 <= features.rank_mean <= 1.0
            # All systems agree on relative order → low but non-zero std
            # (normalization varies slightly between systems of different sizes)
            assert features.rank_std >= 0.0

    def test_vector_dimension_matches_expected(self):
        """Feature vector should always have exactly 12 elements."""
        from src.data.features.massey_systems import MASSEY_MULTI_SYSTEM_DIM, MasseyMultiSystemFeatures

        for n_systems in [0, 1, 5, 10]:
            ratings = {MASSEY_TOP_SYSTEMS[i]: 0.5 for i in range(n_systems)}
            features = MasseyMultiSystemFeatures(
                system_ratings=ratings,
                rank_mean=0.5 if n_systems > 0 else float("nan"),
                rank_std=0.0,
                n_systems=n_systems,
            )
            vec = features_to_vector(features)
            assert len(vec) == MASSEY_MULTI_SYSTEM_DIM == 12
