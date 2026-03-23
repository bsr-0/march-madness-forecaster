"""Tests for Massey Ordinals multi-system feature extraction.

Validates the new module that extracts individual rating system features
from the Kaggle MMasseyOrdinals.csv data, expanding from 2 aggregate
features to 12 (10 individual systems + rank_mean + rank_std).
"""

import math

import numpy as np
import pytest

from src.data.features.massey_systems import (
    ALL_MASSEY_FEATURE_NAMES,
    MASSEY_MULTI_SYSTEM_DIM,
    MASSEY_TOP_SYSTEMS,
    MasseyMultiSystemFeatures,
    extract_all_teams,
    extract_multi_system_features,
    features_to_vector,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _MockEntry:
    """Minimal mock for MasseyOrdinalEntry."""
    def __init__(self, ordinal_rank):
        self.ordinal_rank = ordinal_rank


def _build_ordinals(systems_data):
    """Build a mock ordinals dict from {system: {team: rank}} mapping."""
    result = {}
    for system, team_ranks in systems_data.items():
        result[system] = {
            team_id: _MockEntry(rank)
            for team_id, rank in team_ranks.items()
        }
    return result


# ---------------------------------------------------------------------------
# Unit tests: extract_multi_system_features
# ---------------------------------------------------------------------------

class TestExtractMultiSystemFeatures:
    """Test individual team feature extraction from ordinals."""

    def test_basic_extraction(self):
        """Features extracted correctly for a team present in multiple systems."""
        ordinals = _build_ordinals({
            "POM": {"duke": 1, "unc": 5, "kansas": 10},
            "SAG": {"duke": 2, "unc": 3, "kansas": 8},
            "MOR": {"duke": 1, "unc": 6, "kansas": 12},
        })
        features = extract_multi_system_features(ordinals, "duke", n_teams_approx=3)
        assert features.n_systems == 3
        assert "POM" in features.system_ratings
        assert "SAG" in features.system_ratings
        assert "MOR" in features.system_ratings
        # Rank 1 out of 3 → normalized = 1.0
        assert features.system_ratings["POM"] == 1.0
        # rank_mean should be close to 1.0 (duke is top in all systems)
        assert features.rank_mean > 0.5

    def test_missing_team(self):
        """Team not in any system returns empty features with NaN."""
        ordinals = _build_ordinals({
            "POM": {"duke": 1},
        })
        features = extract_multi_system_features(ordinals, "unknown_team")
        assert features.n_systems == 0
        assert len(features.system_ratings) == 0
        assert math.isnan(features.rank_mean)
        assert math.isnan(features.rank_std)

    def test_missing_system(self):
        """Systems not in ordinals are skipped gracefully."""
        ordinals = _build_ordinals({
            "POM": {"duke": 1},
            # SAG, MOR, etc. not present
        })
        features = extract_multi_system_features(ordinals, "duke")
        assert features.n_systems == 1
        assert "POM" in features.system_ratings
        assert "SAG" not in features.system_ratings

    def test_normalization_rank_1(self):
        """Rank 1 out of N teams normalizes to ~1.0."""
        ordinals = _build_ordinals({
            "POM": {f"team_{i}": i + 1 for i in range(100)},
        })
        features = extract_multi_system_features(ordinals, "team_0", n_teams_approx=100)
        # team_0 has rank 1 → normalized = 1.0
        assert abs(features.system_ratings["POM"] - 1.0) < 1e-6

    def test_normalization_last_rank(self):
        """Last-ranked team normalizes to ~0.0."""
        ordinals = _build_ordinals({
            "POM": {f"team_{i}": i + 1 for i in range(100)},
        })
        features = extract_multi_system_features(ordinals, "team_99", n_teams_approx=100)
        assert abs(features.system_ratings["POM"] - 0.0) < 0.02

    def test_rank_std_agreement(self):
        """When all systems agree, rank_std is 0."""
        ordinals = _build_ordinals({
            "POM": {"duke": 1, "unc": 2},
            "SAG": {"duke": 1, "unc": 2},
            "MOR": {"duke": 1, "unc": 2},
        })
        features = extract_multi_system_features(ordinals, "duke", n_teams_approx=2)
        assert features.rank_std == 0.0

    def test_rank_std_disagreement(self):
        """When systems disagree, rank_std > 0."""
        ordinals = _build_ordinals({
            "POM": {"duke": 1, "unc": 100},
            "SAG": {"duke": 50, "unc": 100},
        })
        features = extract_multi_system_features(ordinals, "duke", n_teams_approx=100)
        assert features.rank_std > 0.0

    def test_dict_entries_supported(self):
        """Ordinal entries as dicts (not objects) are supported."""
        ordinals = {
            "POM": {"duke": {"ordinal_rank": 5}},
            "SAG": {"duke": {"ordinal_rank": 3}},
        }
        features = extract_multi_system_features(ordinals, "duke", n_teams_approx=50)
        assert features.n_systems == 2
        assert "POM" in features.system_ratings


# ---------------------------------------------------------------------------
# Unit tests: extract_all_teams
# ---------------------------------------------------------------------------

class TestExtractAllTeams:
    """Test bulk extraction across all teams."""

    def test_basic_all_teams(self):
        """All teams across all systems are extracted."""
        ordinals = _build_ordinals({
            "POM": {"duke": 1, "unc": 2, "kansas": 3},
            "SAG": {"duke": 1, "unc": 3, "kentucky": 5},
        })
        result = extract_all_teams(ordinals)
        # duke, unc, kansas, kentucky — 4 unique teams
        assert len(result) == 4
        assert "duke" in result
        assert "kentucky" in result

    def test_empty_ordinals(self):
        """Empty ordinals returns empty dict."""
        result = extract_all_teams({})
        assert result == {}


# ---------------------------------------------------------------------------
# Unit tests: features_to_vector
# ---------------------------------------------------------------------------

class TestFeaturesToVector:
    """Test conversion from MasseyMultiSystemFeatures to flat vector."""

    def test_vector_length(self):
        """Vector has exactly MASSEY_MULTI_SYSTEM_DIM elements."""
        features = MasseyMultiSystemFeatures(
            system_ratings={"POM": 0.95, "SAG": 0.88},
            rank_mean=0.9,
            rank_std=0.05,
            n_systems=2,
        )
        vec = features_to_vector(features)
        assert len(vec) == MASSEY_MULTI_SYSTEM_DIM
        assert len(vec) == 12

    def test_missing_systems_are_nan(self):
        """Systems not in system_ratings produce NaN in the vector."""
        features = MasseyMultiSystemFeatures(
            system_ratings={"POM": 0.95},
            rank_mean=0.95,
            rank_std=0.0,
            n_systems=1,
        )
        vec = features_to_vector(features)
        # POM is index 0, should be 0.95
        assert abs(vec[0] - 0.95) < 1e-6
        # SAG is index 1, should be NaN
        assert math.isnan(vec[1])
        # rank_mean is index 10
        assert abs(vec[10] - 0.95) < 1e-6

    def test_all_systems_present(self):
        """When all 10 systems present, no NaN in system slots."""
        ratings = {s: 0.5 for s in MASSEY_TOP_SYSTEMS}
        features = MasseyMultiSystemFeatures(
            system_ratings=ratings,
            rank_mean=0.5,
            rank_std=0.0,
            n_systems=10,
        )
        vec = features_to_vector(features)
        for i in range(10):
            assert not math.isnan(vec[i])
        assert abs(vec[10] - 0.5) < 1e-6
        assert abs(vec[11] - 0.0) < 1e-6


# ---------------------------------------------------------------------------
# Integration: TeamFeatures dimension
# ---------------------------------------------------------------------------

class TestTeamFeaturesDimension:
    """Verify TEAM_FEATURE_DIM includes the 12 new massey features."""

    def test_dimension_includes_massey(self):
        """TEAM_FEATURE_DIM accounts for the 12 new massey multi-system features."""
        from src.data.features.feature_engineering import TEAM_FEATURE_DIM
        # Was 74, now 86 (74 + 12)
        assert TEAM_FEATURE_DIM == 86

    def test_feature_names_include_massey(self):
        """get_feature_names() includes all massey feature names."""
        from src.data.features.feature_engineering import TeamFeatures
        names = TeamFeatures.get_feature_names(include_embeddings=False)
        for massey_name in ALL_MASSEY_FEATURE_NAMES:
            assert massey_name in names, f"Missing feature name: {massey_name}"

    def test_to_vector_matches_dim(self):
        """Default TeamFeatures.to_vector() matches TEAM_FEATURE_DIM."""
        from src.data.features.feature_engineering import TEAM_FEATURE_DIM, TeamFeatures
        tf = TeamFeatures(team_id="test", team_name="Test", seed=1, region="W")
        vec = tf.to_vector(include_embeddings=False)
        assert len(vec) == TEAM_FEATURE_DIM

    def test_feature_names_length_matches_dim(self):
        """get_feature_names() length matches TEAM_FEATURE_DIM."""
        from src.data.features.feature_engineering import TEAM_FEATURE_DIM, TeamFeatures
        names = TeamFeatures.get_feature_names(include_embeddings=False)
        assert len(names) == TEAM_FEATURE_DIM


# ---------------------------------------------------------------------------
# Era availability
# ---------------------------------------------------------------------------

class TestEraAvailability:
    """Verify massey features respect era availability constraints."""

    def test_massey_available_2003_onwards(self):
        """Massey multi-system features available from 2003 onwards."""
        from src.data.features.feature_engineering import era_available_features
        features_2005 = era_available_features(2005)
        assert "massey_pom" in features_2005
        assert "massey_rank_std" in features_2005

    def test_massey_unavailable_before_2003(self):
        """Massey multi-system features unavailable before 2003."""
        from src.data.features.feature_engineering import era_available_features
        features_2002 = era_available_features(2002)
        assert "massey_pom" not in features_2002
        assert "massey_rank_std" not in features_2002


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    """Verify module constants are consistent."""

    def test_top_systems_count(self):
        """10 top systems selected."""
        assert len(MASSEY_TOP_SYSTEMS) == 10

    def test_dim_matches_systems_plus_derived(self):
        """MASSEY_MULTI_SYSTEM_DIM = 10 systems + 2 derived."""
        assert MASSEY_MULTI_SYSTEM_DIM == 12

    def test_feature_names_count(self):
        """ALL_MASSEY_FEATURE_NAMES has exactly MASSEY_MULTI_SYSTEM_DIM entries."""
        assert len(ALL_MASSEY_FEATURE_NAMES) == MASSEY_MULTI_SYSTEM_DIM
