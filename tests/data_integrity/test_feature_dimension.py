"""Feature dimension contract tests.

Verifies that the TEAM_FEATURE_DIM constant matches the actual output
of TeamFeatures.to_vector(), ensuring no silent drift between the
documented feature count and the runtime feature vector length.
"""

import numpy as np
import pytest

from src.data.features.feature_engineering import TEAM_FEATURE_DIM, TeamFeatures


def _make_team_features(**kwargs):
    """Helper to create TeamFeatures with required positional args."""
    defaults = dict(team_id="test", team_name="Test", seed=1, region="East")
    defaults.update(kwargs)
    return TeamFeatures(**defaults)


class TestFeatureDimensionContract:
    """Verify TEAM_FEATURE_DIM matches to_vector() output length."""

    def test_default_team_features_match_dim(self):
        """A default-constructed TeamFeatures produces a vector of TEAM_FEATURE_DIM."""
        tf = _make_team_features()
        vec = tf.to_vector()
        assert len(vec) == TEAM_FEATURE_DIM, (
            f"to_vector() returned {len(vec)} features but TEAM_FEATURE_DIM={TEAM_FEATURE_DIM}"
        )

    def test_feature_vector_is_float64(self):
        """Feature vector dtype should be float64."""
        tf = _make_team_features()
        vec = tf.to_vector()
        assert vec.dtype == np.float64

    def test_feature_vector_nan_count_on_defaults(self):
        """Default-constructed features produce NaN only for massey multi-system.

        The 12 massey multi-system features default to NaN because they
        require Kaggle MMasseyOrdinals data.  Tree models (LightGBM/XGBoost)
        handle NaN natively as missing values.
        """
        tf = _make_team_features()
        vec = tf.to_vector()
        nan_count = int(np.isnan(vec).sum())
        # Exactly 12 NaN values from the massey multi-system features
        assert nan_count == 12, (
            f"Default TeamFeatures produced {nan_count} NaN values, expected 12 "
            f"(massey multi-system features)"
        )

    def test_team_feature_dim_is_positive(self):
        """TEAM_FEATURE_DIM should be a positive integer."""
        assert isinstance(TEAM_FEATURE_DIM, int)
        assert TEAM_FEATURE_DIM > 0

    def test_matchup_differential_dim(self):
        """Differential feature vector (team1 - team2) matches TEAM_FEATURE_DIM."""
        tf1 = _make_team_features(team_id="team_a", team_name="Team A")
        tf2 = _make_team_features(team_id="team_b", team_name="Team B")
        vec1 = tf1.to_vector()
        vec2 = tf2.to_vector()
        differential = vec1 - vec2
        # Matchup differential = team1_vec - team2_vec (same dim as single team)
        assert len(differential) == TEAM_FEATURE_DIM
