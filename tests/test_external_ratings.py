"""Tests for external rating system integration (WS3)."""

import json
import tempfile
from pathlib import Path

import pytest

from src.data.scrapers.external_ratings import (
    CompositeRating,
    ExternalRating,
    ExternalRatingsLoader,
)


class TestExternalRating:

    def test_dataclass_fields(self):
        r = ExternalRating(
            system_name="kenpom",
            team_name="Duke",
            team_id="duke",
            rating=25.5,
            ranking=3,
            normalized=0.95,
        )
        assert r.system_name == "kenpom"
        assert r.rating == 25.5
        assert r.ranking == 3


class TestCompositeRating:

    def test_defaults(self):
        cr = CompositeRating(team_id="duke", team_name="Duke")
        assert cr.composite_rating == 0.0
        assert cr.n_systems == 0
        assert cr.per_system == {}


class TestExternalRatingsLoader:

    def test_load_system_missing_file(self):
        loader = ExternalRatingsLoader(cache_dir="/nonexistent")
        result = loader._load_system("kenpom", 2025)
        assert result == {}

    def test_load_system_from_cache(self, tmp_path):
        """Load ratings from a cached JSON file."""
        data = [
            {"team_name": "Duke", "team_id": "duke", "rating": 25.0,
             "ranking": 1, "normalized": 0.98},
            {"team_name": "UNC", "team_id": "unc", "rating": 22.0,
             "ranking": 5, "normalized": 0.85},
        ]
        cache_file = tmp_path / "external_kenpom_2025.json"
        cache_file.write_text(json.dumps(data))

        loader = ExternalRatingsLoader(cache_dir=str(tmp_path))
        result = loader._load_system("kenpom", 2025)
        assert len(result) == 2
        assert "duke" in result
        assert result["duke"].rating == 25.0
        assert result["duke"].team_name == "Duke"

    def test_compute_composite_single_system(self, tmp_path):
        """Composite with one system should equal normalized ratings."""
        loader = ExternalRatingsLoader(cache_dir=str(tmp_path))
        all_ratings = {
            "kenpom": {
                "duke": ExternalRating("kenpom", "Duke", "duke", 25.0, 1, 0.98),
                "unc": ExternalRating("kenpom", "UNC", "unc", 15.0, 10, 0.5),
            }
        }
        composites = loader.compute_composite(all_ratings)
        assert len(composites) == 2
        assert composites["duke"].composite_rating > composites["unc"].composite_rating

    def test_compute_composite_multi_system(self, tmp_path):
        """Multi-system composite should weight by system accuracy."""
        loader = ExternalRatingsLoader(cache_dir=str(tmp_path))
        all_ratings = {
            "kenpom": {
                "duke": ExternalRating("kenpom", "Duke", "duke", 25.0, 1, 0.98),
                "unc": ExternalRating("kenpom", "UNC", "unc", 15.0, 10, 0.5),
            },
            "sagarin": {
                "duke": ExternalRating("sagarin", "Duke", "duke", 90.0, 2, 0.95),
                "unc": ExternalRating("sagarin", "UNC", "unc", 80.0, 8, 0.6),
            },
        }
        composites = loader.compute_composite(all_ratings)
        assert len(composites) == 2
        assert composites["duke"].n_systems == 2
        assert composites["duke"].rating_spread >= 0

    def test_compute_composite_assigns_rankings(self, tmp_path):
        """Rankings should be assigned by composite rating."""
        loader = ExternalRatingsLoader(cache_dir=str(tmp_path))
        all_ratings = {
            "kenpom": {
                "duke": ExternalRating("kenpom", "Duke", "duke", 30.0, 1, 0.98),
                "unc": ExternalRating("kenpom", "UNC", "unc", 20.0, 5, 0.7),
                "uk": ExternalRating("kenpom", "UK", "uk", 10.0, 10, 0.3),
            }
        }
        composites = loader.compute_composite(all_ratings)
        assert composites["duke"].composite_ranking == 1
        assert composites["uk"].composite_ranking == 3

    def test_compute_composite_empty(self, tmp_path):
        loader = ExternalRatingsLoader(cache_dir=str(tmp_path))
        assert loader.compute_composite({}) == {}

    def test_generate_from_seeds(self, tmp_path):
        """Seed-based fallback should produce valid ratings."""
        loader = ExternalRatingsLoader(cache_dir=str(tmp_path))
        seeds = {"duke": 1, "unc": 4, "msu": 8, "fairleigh": 16}
        composites = loader.generate_from_seeds(seeds)

        assert len(composites) == 4
        assert composites["duke"].composite_rating > composites["unc"].composite_rating
        assert composites["unc"].composite_rating > composites["msu"].composite_rating
        assert composites["msu"].composite_rating > composites["fairleigh"].composite_rating

    def test_generate_from_seeds_rankings(self, tmp_path):
        loader = ExternalRatingsLoader(cache_dir=str(tmp_path))
        seeds = {"a": 1, "b": 2, "c": 16}
        composites = loader.generate_from_seeds(seeds)
        assert composites["a"].composite_ranking == 1
        assert composites["c"].composite_ranking == 3

    def test_compute_composite_outlier_robust(self, tmp_path):
        """A single extreme outlier should not compress all other ratings."""
        loader = ExternalRatingsLoader(cache_dir=str(tmp_path))
        # 10 normal teams rated 10–19, plus one extreme outlier at 1000
        teams = {}
        for i in range(10):
            tid = f"team_{i}"
            teams[tid] = ExternalRating("sys", f"Team {i}", tid, 10.0 + i, i + 1, 0.0)
        teams["outlier"] = ExternalRating("sys", "Outlier", "outlier", 1000.0, 0, 0.0)

        composites = loader.compute_composite({"sys": teams})

        # Under naive min-max, team_9 (rating=19) would get (19-10)/(1000-10)=0.009
        # Percentile-rank normalization preserves full ordering regardless of outlier
        normal_ratings = [
            composites[f"team_{i}"].composite_rating for i in range(10)
        ]
        spread = max(normal_ratings) - min(normal_ratings)
        assert spread > 0.3, f"Normal team spread too compressed: {spread}"
        # Outlier gets highest rank, all normal teams retain distinct ranks
        assert composites["outlier"].composite_rating == 1.0
        # Every adjacent pair is distinguishable
        for i in range(9):
            assert composites[f"team_{i}"].composite_rating < composites[f"team_{i+1}"].composite_rating

    def test_compute_composite_no_outlier_still_works(self, tmp_path):
        """Normal distributions without outliers should preserve full ordering."""
        loader = ExternalRatingsLoader(cache_dir=str(tmp_path))
        all_ratings = {
            "kenpom": {
                "best": ExternalRating("kenpom", "Best", "best", 30.0, 1, 0.0),
                "mid": ExternalRating("kenpom", "Mid", "mid", 20.0, 2, 0.0),
                "worst": ExternalRating("kenpom", "Worst", "worst", 10.0, 3, 0.0),
            }
        }
        composites = loader.compute_composite(all_ratings)
        assert composites["best"].composite_rating > composites["mid"].composite_rating
        assert composites["mid"].composite_rating > composites["worst"].composite_rating
        # Percentile-rank: best=1.0, mid=0.5, worst=0.0 for 3 distinct teams
        assert composites["best"].composite_rating == 1.0
        assert composites["worst"].composite_rating == 0.0
        assert all(0.0 <= composites[t].composite_rating <= 1.0 for t in ["best", "mid", "worst"])

    def test_compute_composite_negative_ratings(self, tmp_path):
        """Sagarin-style negative ratings should normalize correctly."""
        loader = ExternalRatingsLoader(cache_dir=str(tmp_path))
        all_ratings = {
            "sagarin": {
                "best": ExternalRating("sagarin", "Best", "best", 25.0, 1, 0.0),
                "good": ExternalRating("sagarin", "Good", "good", 10.0, 2, 0.0),
                "avg": ExternalRating("sagarin", "Avg", "avg", 0.0, 3, 0.0),
                "bad": ExternalRating("sagarin", "Bad", "bad", -5.0, 4, 0.0),
                "worst": ExternalRating("sagarin", "Worst", "worst", -10.0, 5, 0.0),
            }
        }
        composites = loader.compute_composite(all_ratings)
        # Ordering must be preserved even with negative input ratings
        assert composites["best"].composite_rating > composites["good"].composite_rating
        assert composites["good"].composite_rating > composites["avg"].composite_rating
        assert composites["avg"].composite_rating > composites["bad"].composite_rating
        assert composites["bad"].composite_rating > composites["worst"].composite_rating
        # All normalized to [0, 1]
        for t in ["best", "good", "avg", "bad", "worst"]:
            assert 0.0 <= composites[t].composite_rating <= 1.0

    def test_compute_composite_single_team(self, tmp_path):
        """Single team should not crash and should get a valid rating."""
        loader = ExternalRatingsLoader(cache_dir=str(tmp_path))
        all_ratings = {
            "kenpom": {
                "only": ExternalRating("kenpom", "Only", "only", 20.0, 1, 0.0),
            }
        }
        composites = loader.compute_composite(all_ratings)
        assert len(composites) == 1
        assert 0.0 <= composites["only"].composite_rating <= 1.0

    def test_compute_composite_identical_ratings(self, tmp_path):
        """All teams with the same rating should get equal composites."""
        loader = ExternalRatingsLoader(cache_dir=str(tmp_path))
        all_ratings = {
            "kenpom": {
                "a": ExternalRating("kenpom", "A", "a", 20.0, 1, 0.0),
                "b": ExternalRating("kenpom", "B", "b", 20.0, 1, 0.0),
                "c": ExternalRating("kenpom", "C", "c", 20.0, 1, 0.0),
            }
        }
        composites = loader.compute_composite(all_ratings)
        ratings = [composites[t].composite_rating for t in ["a", "b", "c"]]
        # All should be identical
        assert ratings[0] == ratings[1] == ratings[2]

    def test_save_and_load_roundtrip(self, tmp_path):
        """Save ratings then load them back."""
        loader = ExternalRatingsLoader(cache_dir=str(tmp_path))
        ratings = {
            "duke": ExternalRating("kenpom", "Duke", "duke", 25.0, 1, 0.98),
            "unc": ExternalRating("kenpom", "UNC", "unc", 15.0, 5, 0.5),
        }
        loader.save_system("kenpom", 2025, ratings)
        loaded = loader._load_system("kenpom", 2025)
        assert len(loaded) == 2
        assert loaded["duke"].rating == 25.0
