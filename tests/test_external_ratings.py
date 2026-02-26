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
