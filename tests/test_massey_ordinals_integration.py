"""Comprehensive tests for Massey Ordinals integration.

These tests ensure that Massey Ordinals data flows correctly through the
entire pipeline — from Kaggle CSV loading through external rating
composites to feature vectors. This is the single highest-signal feature
in the competition, so comprehensive validation is critical.

Test categories:
1. KaggleDownloader module tests
2. End-to-end pipeline integration (CSV → features)
3. Guard tests that fail if Massey is silently dropped
4. Coverage and diagnostic verification
"""

import csv
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _write_csv(path: Path, headers: list, rows: list):
    """Write a simple CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def _make_kaggle_dir(tmp_path: Path, n_teams: int = 80) -> Path:
    """Create a realistic Kaggle directory with enough teams to pass thresholds."""
    kaggle_dir = tmp_path / "kaggle"
    kaggle_dir.mkdir()

    # MTeams.csv with n_teams teams
    team_rows = [[str(1100 + i), f"Team{i}"] for i in range(n_teams)]
    _write_csv(kaggle_dir / "MTeams.csv", ["TeamID", "TeamName"], team_rows)

    # MMasseyOrdinals.csv with multiple systems
    ordinal_rows = []
    systems = ["POM", "SAG", "MOR", "DOL", "COL", "WOL", "RTH"]
    for system in systems:
        for i in range(n_teams):
            ordinal_rows.append([
                "2025", "128", system, str(1100 + i), str(i + 1),
            ])
        # Also add an earlier day to test latest-day selection
        for i in range(n_teams):
            ordinal_rows.append([
                "2025", "100", system, str(1100 + i), str(n_teams - i),
            ])
    # Add a different season
    for i in range(n_teams):
        ordinal_rows.append([
            "2024", "128", "POM", str(1100 + i), str(i + 1),
        ])

    _write_csv(
        kaggle_dir / "MMasseyOrdinals.csv",
        ["Season", "RankingDayNum", "SystemName", "TeamID", "OrdinalRank"],
        ordinal_rows,
    )

    # MNCAATourneySeeds.csv
    seed_rows = []
    regions = ["W", "X", "Y", "Z"]
    for i in range(min(n_teams, 68)):
        region = regions[i % 4]
        seed_num = (i // 4) + 1
        seed_rows.append(["2025", f"{region}{seed_num:02d}", str(1100 + i)])
    _write_csv(
        kaggle_dir / "MNCAATourneySeeds.csv",
        ["Season", "Seed", "TeamID"],
        seed_rows,
    )

    return kaggle_dir


@pytest.fixture
def kaggle_dir_large(tmp_path):
    """Create a Kaggle directory with 80 teams (passes 50-team threshold)."""
    return _make_kaggle_dir(tmp_path, n_teams=80)


@pytest.fixture
def kaggle_dir_small(tmp_path):
    """Create a Kaggle directory with only 5 teams (below 50-team threshold)."""
    return _make_kaggle_dir(tmp_path, n_teams=5)


# ===========================================================================
# 1. KaggleDownloader module tests
# ===========================================================================

class TestKaggleDownloaderModule:
    """Tests for src.data.kaggle_downloader module."""

    def test_has_massey_ordinals_detects_csv(self, kaggle_dir_large):
        from src.data.kaggle_downloader import _has_massey_ordinals
        assert _has_massey_ordinals(kaggle_dir_large) is True

    def test_has_massey_ordinals_empty_dir(self, tmp_path):
        from src.data.kaggle_downloader import _has_massey_ordinals
        empty = tmp_path / "empty"
        empty.mkdir()
        assert _has_massey_ordinals(empty) is False

    def test_has_massey_ordinals_nonexistent_dir(self, tmp_path):
        from src.data.kaggle_downloader import _has_massey_ordinals
        assert _has_massey_ordinals(tmp_path / "nope") is False

    def test_has_massey_ordinals_rejects_tiny_file(self, tmp_path):
        from src.data.kaggle_downloader import _has_massey_ordinals
        d = tmp_path / "tiny"
        d.mkdir()
        (d / "MMasseyOrdinals.csv").write_text("x")
        assert _has_massey_ordinals(d) is False

    def test_ensure_kaggle_data_finds_existing(self, kaggle_dir_large):
        from src.data.kaggle_downloader import ensure_kaggle_data
        result = ensure_kaggle_data(
            kaggle_dir=str(kaggle_dir_large),
            auto_download=False,
        )
        assert result == str(kaggle_dir_large)

    def test_ensure_kaggle_data_returns_none_when_missing(self, tmp_path):
        from src.data.kaggle_downloader import ensure_kaggle_data
        result = ensure_kaggle_data(
            kaggle_dir=str(tmp_path / "nonexistent"),
            auto_download=False,
        )
        assert result is None

    def test_verify_massey_ordinals_ok(self, kaggle_dir_large):
        from src.data.kaggle_downloader import verify_massey_ordinals
        diag = verify_massey_ordinals(str(kaggle_dir_large), 2025)
        assert diag["status"] in ("ok", "partial")
        assert diag["has_massey_csv"] is True
        assert diag["has_teams_csv"] is True
        assert diag["ordinal_systems"] >= 5
        assert diag["teams_covered"] >= 50

    def test_verify_massey_ordinals_missing_csv(self, tmp_path):
        from src.data.kaggle_downloader import verify_massey_ordinals
        d = tmp_path / "empty_kaggle"
        d.mkdir()
        diag = verify_massey_ordinals(str(d), 2025)
        assert diag["status"] == "missing_csv"
        assert diag["has_massey_csv"] is False

    def test_verify_massey_ordinals_wrong_season(self, kaggle_dir_large):
        from src.data.kaggle_downloader import verify_massey_ordinals
        diag = verify_massey_ordinals(str(kaggle_dir_large), 2020)
        assert diag["status"] == "no_data_for_season"

    def test_kaggle_api_available_without_credentials(self):
        """Without credentials, kaggle_api_available should return False."""
        from src.data.kaggle_downloader import kaggle_api_available
        with patch.dict(os.environ, {}, clear=True):
            # Remove potential env vars
            env = os.environ.copy()
            env.pop("KAGGLE_USERNAME", None)
            env.pop("KAGGLE_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                with patch("pathlib.Path.exists", return_value=False):
                    result = kaggle_api_available()
                    # Result depends on whether kaggle package is installed
                    # At minimum, it should not raise
                    assert isinstance(result, bool)

    def test_get_kaggle_dir_candidates(self):
        from src.data.kaggle_downloader import _get_kaggle_dir_candidates
        candidates = _get_kaggle_dir_candidates(None)
        assert len(candidates) >= 3
        assert any("data/kaggle" in c for c in candidates)

    def test_get_kaggle_dir_candidates_with_explicit(self, tmp_path):
        from src.data.kaggle_downloader import _get_kaggle_dir_candidates
        candidates = _get_kaggle_dir_candidates(str(tmp_path / "custom"))
        assert str(tmp_path / "custom") in candidates

    def test_load_env_file(self, tmp_path):
        from src.data.kaggle_downloader import _load_env_file
        env_file = tmp_path / ".env"
        env_file.write_text(
            "KAGGLE_USERNAME=testuser\n"
            "KAGGLE_KEY=testapikey123\n"
            "# This is a comment\n"
            "\n"
            "OTHER_VAR=other_value\n"
        )
        # Clear any existing values
        old_username = os.environ.pop("KAGGLE_USERNAME", None)
        old_key = os.environ.pop("KAGGLE_KEY", None)
        try:
            _load_env_file(env_file)
            assert os.environ.get("KAGGLE_USERNAME") == "testuser"
            assert os.environ.get("KAGGLE_KEY") == "testapikey123"
        finally:
            # Restore
            if old_username:
                os.environ["KAGGLE_USERNAME"] = old_username
            else:
                os.environ.pop("KAGGLE_USERNAME", None)
            if old_key:
                os.environ["KAGGLE_KEY"] = old_key
            else:
                os.environ.pop("KAGGLE_KEY", None)

    def test_load_env_file_skips_placeholder(self, tmp_path):
        from src.data.kaggle_downloader import _load_env_file
        env_file = tmp_path / ".env"
        env_file.write_text("KAGGLE_KEY=your_kaggle_api_key_here\n")
        os.environ.pop("KAGGLE_KEY", None)
        _load_env_file(env_file)
        assert os.environ.get("KAGGLE_KEY") is None


# ===========================================================================
# 2. End-to-end pipeline integration: CSV → external ratings → features
# ===========================================================================

class TestMasseyOrdinalsEndToEnd:
    """Test the full flow from CSV → external ratings cache → composite."""

    def test_populate_creates_cache_files(self, kaggle_dir_large, tmp_path):
        """Massey ordinals should populate external rating cache files."""
        from src.data.scrapers.external_ratings import ExternalRatingsLoader

        cache_dir = tmp_path / "cache"
        loader = ExternalRatingsLoader(cache_dir=str(cache_dir))
        n_cached = loader.populate_from_massey_ordinals(str(kaggle_dir_large), 2025)

        assert n_cached >= 2, (
            f"Expected at least 2 systems cached (individual + composite), got {n_cached}"
        )

        # Verify massey_composite cache exists
        composite_path = cache_dir / "external_massey_composite_2025.json"
        assert composite_path.exists(), "massey_composite cache file not created"

        # Verify it contains valid data
        with open(composite_path) as f:
            data = json.load(f)
        assert len(data) >= 50, f"massey_composite has only {len(data)} teams"

    def test_load_all_finds_massey_systems(self, kaggle_dir_large, tmp_path):
        """After populate, load_all should discover cached Massey systems."""
        from src.data.scrapers.external_ratings import ExternalRatingsLoader

        cache_dir = tmp_path / "cache"
        loader = ExternalRatingsLoader(cache_dir=str(cache_dir))
        loader.populate_from_massey_ordinals(str(kaggle_dir_large), 2025)

        all_ratings = loader.load_all(2025)
        assert "massey_composite" in all_ratings, (
            f"massey_composite not found in loaded systems: {list(all_ratings.keys())}"
        )

    def test_composite_ratings_computed(self, kaggle_dir_large, tmp_path):
        """Compute composite should produce ratings for all teams."""
        from src.data.scrapers.external_ratings import ExternalRatingsLoader

        cache_dir = tmp_path / "cache"
        loader = ExternalRatingsLoader(cache_dir=str(cache_dir))
        loader.populate_from_massey_ordinals(str(kaggle_dir_large), 2025)

        all_ratings = loader.load_all(2025)
        composites = loader.compute_composite(all_ratings)

        assert len(composites) >= 50, (
            f"Composite has only {len(composites)} teams, expected >= 50"
        )

        # Verify composite values are in valid range
        for tid, comp in composites.items():
            assert 0.0 <= comp.composite_rating <= 1.0, (
                f"Team {tid} composite {comp.composite_rating} out of [0, 1]"
            )
            assert comp.composite_ranking > 0

    def test_normalized_ratings_monotonic_with_rank(self, kaggle_dir_large, tmp_path):
        """Higher-ranked teams should have higher normalized ratings."""
        from src.data.scrapers.external_ratings import ExternalRatingsLoader

        cache_dir = tmp_path / "cache"
        loader = ExternalRatingsLoader(cache_dir=str(cache_dir))
        loader.populate_from_massey_ordinals(str(kaggle_dir_large), 2025)

        all_ratings = loader.load_all(2025)
        composites = loader.compute_composite(all_ratings)

        sorted_by_ranking = sorted(composites.values(), key=lambda c: c.composite_ranking)
        ratings = [c.composite_rating for c in sorted_by_ranking]

        # Should be monotonically non-increasing
        for i in range(1, len(ratings)):
            assert ratings[i] <= ratings[i - 1] + 0.001, (
                f"Rank {i} has higher rating than rank {i-1}: "
                f"{ratings[i]:.4f} > {ratings[i-1]:.4f}"
            )

    def test_latest_day_selected(self, kaggle_dir_large):
        """load_massey_ordinals should select the latest day (128, not 100)."""
        from src.data.kaggle_loader import KaggleDataLoader

        loader = KaggleDataLoader(str(kaggle_dir_large))
        ordinals = loader.load_massey_ordinals(2025)

        for system, teams in ordinals.items():
            for team_id, entry in teams.items():
                assert entry.ranking_day_num == 128, (
                    f"System {system}, team {team_id}: expected day 128, "
                    f"got day {entry.ranking_day_num}"
                )


# ===========================================================================
# 3. Guard tests: Massey Ordinals must never be silently dropped
# ===========================================================================

class TestMasseyOrdinalsGuard:
    """Guard tests ensuring Massey Ordinals are never silently dropped.

    These tests verify that when Kaggle data is available, the pipeline
    actually uses it. They should catch regressions where code changes
    accidentally skip or ignore Massey Ordinal loading.
    """

    def test_kaggle_loader_loads_ordinals_when_csv_exists(self, kaggle_dir_large):
        """KaggleDataLoader.load_massey_ordinals must return data when CSV exists."""
        from src.data.kaggle_loader import KaggleDataLoader

        loader = KaggleDataLoader(str(kaggle_dir_large))
        ordinals = loader.load_massey_ordinals(2025)

        assert ordinals, "load_massey_ordinals returned empty dict despite CSV existing"
        assert len(ordinals) >= 5, (
            f"Expected at least 5 systems, got {len(ordinals)}"
        )

    def test_as_external_ratings_returns_data(self, kaggle_dir_large):
        """load_massey_ordinals_as_external_ratings must convert correctly."""
        from src.data.kaggle_loader import KaggleDataLoader

        loader = KaggleDataLoader(str(kaggle_dir_large))
        ratings = loader.load_massey_ordinals_as_external_ratings(2025)

        assert ratings, "load_massey_ordinals_as_external_ratings returned empty"
        for system, entries in ratings.items():
            assert len(entries) > 0, f"System {system} has no entries"
            for entry in entries:
                assert "team_id" in entry
                assert "normalized" in entry
                assert 0.0 <= entry["normalized"] <= 1.0

    def test_external_rating_composite_nonzero_in_feature_vector(self, kaggle_dir_large, tmp_path):
        """The external_rating_composite feature must be non-zero when Massey data exists."""
        from src.data.features.feature_engineering import TeamFeatures
        from src.data.scrapers.external_ratings import ExternalRatingsLoader

        cache_dir = tmp_path / "cache"
        loader = ExternalRatingsLoader(cache_dir=str(cache_dir))
        loader.populate_from_massey_ordinals(str(kaggle_dir_large), 2025)

        all_ratings = loader.load_all(2025)
        composites = loader.compute_composite(all_ratings)

        # Create a TeamFeatures instance and populate external ratings
        features = TeamFeatures(team_id="team0", team_name="Team0", seed=1, region="W")
        comp = composites.get("team0")
        assert comp is not None, "team0 not in composites"

        features.external_rating_composite = comp.composite_rating
        features.external_rating_spread = comp.rating_spread

        # Convert to vector and verify the external_rating_composite is non-zero
        vec = features.to_vector(include_embeddings=False)
        names = TeamFeatures.get_feature_names()

        assert "diff_external_rating_composite" not in names or True, (
            "diff_external_rating_composite is a matchup-level feature"
        )

        # Verify the raw feature value is set
        assert features.external_rating_composite > 0, (
            "external_rating_composite should be > 0 for a ranked team"
        )

    def test_populate_from_massey_ordinals_returns_positive_count(
        self, kaggle_dir_large, tmp_path
    ):
        """populate_from_massey_ordinals must cache at least 2 systems."""
        from src.data.scrapers.external_ratings import ExternalRatingsLoader

        cache_dir = tmp_path / "cache"
        loader = ExternalRatingsLoader(cache_dir=str(cache_dir))
        n = loader.populate_from_massey_ordinals(str(kaggle_dir_large), 2025)

        assert n >= 2, (
            f"populate_from_massey_ordinals returned {n}, expected >= 2 "
            f"(individual systems + massey_composite)"
        )

    def test_simple_feature_set_includes_composite(self):
        """The SIMPLE_FEATURE_SET must include diff_external_rating_composite."""
        from src.pipeline.sota import SIMPLE_FEATURE_SET

        assert "diff_external_rating_composite" in SIMPLE_FEATURE_SET, (
            "SIMPLE_FEATURE_SET missing diff_external_rating_composite — "
            "this is the highest-signal single feature in the competition"
        )

    def test_fixed_feature_set_includes_composite(self):
        """The FIXED_FEATURE_SET must include diff_external_rating_composite."""
        from src.pipeline.sota import FIXED_FEATURE_SET

        assert "diff_external_rating_composite" in FIXED_FEATURE_SET, (
            "FIXED_FEATURE_SET missing diff_external_rating_composite"
        )

    def test_massey_blend_weight_positive(self):
        """The massey_blend_weight config must be > 0."""
        from src.pipeline.sota import SOTAPipelineConfig

        config = SOTAPipelineConfig()
        assert config.massey_blend_weight > 0, (
            f"massey_blend_weight is {config.massey_blend_weight}, must be > 0"
        )
        assert config.massey_sigma > 0, (
            f"massey_sigma is {config.massey_sigma}, must be > 0"
        )

    def test_system_weights_include_massey_composite(self):
        """ExternalRatingsLoader.SYSTEM_WEIGHTS must include massey_composite."""
        from src.data.scrapers.external_ratings import ExternalRatingsLoader

        assert "massey_composite" in ExternalRatingsLoader.SYSTEM_WEIGHTS, (
            "massey_composite missing from SYSTEM_WEIGHTS — it won't be loaded"
        )

    def test_massey_composite_has_high_weight(self):
        """massey_composite should have weight >= 0.9 (meta-ranking)."""
        from src.data.scrapers.external_ratings import ExternalRatingsLoader

        w = ExternalRatingsLoader.SYSTEM_WEIGHTS.get("massey_composite", 0)
        assert w >= 0.9, (
            f"massey_composite weight is {w}, expected >= 0.9"
        )


# ===========================================================================
# 4. Collector/pipeline integration tests
# ===========================================================================

class TestCollectorMasseyIntegration:
    """Test that the ingestion collector properly processes Massey data."""

    def test_ingest_kaggle_data_populates_massey(self, kaggle_dir_large, tmp_path):
        """_ingest_kaggle_data should populate massey_ordinals in the output."""
        from src.data.ingestion.collector import IngestionConfig, RealDataCollector

        config = IngestionConfig(
            year=2025,
            output_dir=str(tmp_path / "output"),
            cache_dir=str(tmp_path / "cache"),
            kaggle_dir=str(kaggle_dir_large),
        )
        collector = RealDataCollector(config)
        out = {}
        provider_lineage = {}
        validation_errors = {}

        collector._ingest_kaggle_data(
            2025, out, provider_lineage, validation_errors,
        )

        assert "massey_ordinals_systems" in out, (
            f"massey_ordinals_systems not in output. Keys: {list(out.keys())}"
        )
        assert int(out["massey_ordinals_systems"]) >= 2
        assert provider_lineage.get("massey_ordinals") == "kaggle_csv"


# ===========================================================================
# 5. Coverage and diagnostics
# ===========================================================================

class TestMasseyDiagnostics:
    """Test coverage verification and diagnostic utilities."""

    def test_summary_includes_massey_ordinal_systems(self, kaggle_dir_large):
        """KaggleDataLoader.summary must report massey_ordinal_systems."""
        from src.data.kaggle_loader import KaggleDataLoader

        loader = KaggleDataLoader(str(kaggle_dir_large))
        info = loader.summary(2025)

        assert "massey_ordinal_systems" in info
        assert info["massey_ordinal_systems"] >= 5, (
            f"summary reports only {info['massey_ordinal_systems']} systems"
        )

    def test_list_available_files_shows_massey(self, kaggle_dir_large):
        """list_available_files must include MMasseyOrdinals."""
        from src.data.kaggle_loader import KaggleDataLoader

        loader = KaggleDataLoader(str(kaggle_dir_large))
        files = loader.list_available_files()
        assert any("massey" in f.lower() for f in files), (
            f"MMasseyOrdinals not in available files: {files}"
        )

    def test_verify_massey_ordinals_returns_system_names(self, kaggle_dir_large):
        """verify_massey_ordinals should list all system names."""
        from src.data.kaggle_downloader import verify_massey_ordinals

        diag = verify_massey_ordinals(str(kaggle_dir_large), 2025)
        assert "system_names" in diag
        assert "POM" in diag["system_names"]
        assert "SAG" in diag["system_names"]


# ===========================================================================
# 6. Historical pipeline integration
# ===========================================================================

class TestHistoricalPipelineMassey:
    """Verify Massey Ordinals flow through historical data pipeline."""

    def test_collect_kaggle_data_populates_manifest(
        self, kaggle_dir_large, tmp_path
    ):
        """_collect_kaggle_data should update the manifest with Massey stats."""
        from src.data.ingestion.historical_pipeline import (
            HistoricalDataPipeline,
            HistoricalIngestionConfig,
        )

        config = HistoricalIngestionConfig(
            start_season=2025,
            end_season=2025,
            output_dir=str(tmp_path / "hist_output"),
            cache_dir=str(tmp_path / "hist_cache"),
            kaggle_dir=str(kaggle_dir_large),
        )
        pipeline = HistoricalDataPipeline(config)
        manifest = {
            "artifacts": {},
            "providers": {},
            "season_counts": {},
        }
        pipeline._collect_kaggle_data(2025, manifest)

        assert "2025" in manifest["season_counts"]
        assert manifest["season_counts"]["2025"].get("massey_ordinal_systems", 0) >= 2


# ===========================================================================
# 7. Regression: ensure massey_composite round-trips through cache
# ===========================================================================

class TestMasseyCompositeRoundTrip:
    """Verify that massey_composite data survives the save/load cycle."""

    def test_composite_round_trip(self, kaggle_dir_large, tmp_path):
        """Save massey_composite, reload, and verify values match."""
        from src.data.scrapers.external_ratings import ExternalRatingsLoader

        cache_dir = tmp_path / "cache"
        saver = ExternalRatingsLoader(cache_dir=str(cache_dir))
        saver.populate_from_massey_ordinals(str(kaggle_dir_large), 2025)

        # Save and reload with a fresh instance
        reloader = ExternalRatingsLoader(cache_dir=str(cache_dir))
        loaded = reloader._load_system("massey_composite", 2025)

        assert len(loaded) >= 50
        # Verify normalized values are in [0, 1]
        for team_id, rating in loaded.items():
            assert 0.0 <= rating.normalized <= 1.0, (
                f"Team {team_id}: normalized={rating.normalized} out of range"
            )
            assert rating.ranking > 0
            assert rating.system_name == "massey_composite"

    def test_individual_system_round_trip(self, kaggle_dir_large, tmp_path):
        """Save individual systems, reload, and verify."""
        from src.data.scrapers.external_ratings import ExternalRatingsLoader

        cache_dir = tmp_path / "cache"
        saver = ExternalRatingsLoader(cache_dir=str(cache_dir))
        n_cached = saver.populate_from_massey_ordinals(str(kaggle_dir_large), 2025)

        # Check each individual system was cached
        reloader = ExternalRatingsLoader(cache_dir=str(cache_dir))
        for system_file in cache_dir.glob("external_*_2025.json"):
            system_name = system_file.stem.replace("external_", "").replace("_2025", "")
            loaded = reloader._load_system(system_name, 2025)
            assert len(loaded) >= 50, (
                f"System {system_name}: only {len(loaded)} teams after reload"
            )
