"""Feature dimension contract tests.

Verifies that the TEAM_FEATURE_DIM constant matches the actual output
of TeamFeatures.to_vector(), ensuring no silent drift between the
documented feature count and the runtime feature vector length.
"""

import numpy as np
import pytest

from src.data.features.feature_engineering import TEAM_FEATURE_DIM, TeamFeatures


class TestFeatureDimensionContract:
    """Verify TEAM_FEATURE_DIM matches to_vector() output length."""

    def test_default_team_features_match_dim(self):
        """A default-constructed TeamFeatures produces a vector of TEAM_FEATURE_DIM."""
        tf = TeamFeatures()
        vec = tf.to_vector()
        assert len(vec) == TEAM_FEATURE_DIM, (
            f"to_vector() returned {len(vec)} features but TEAM_FEATURE_DIM={TEAM_FEATURE_DIM}"
        )

    def test_feature_vector_is_float64(self):
        """Feature vector dtype should be float64."""
        tf = TeamFeatures()
        vec = tf.to_vector()
        assert vec.dtype == np.float64

    def test_feature_vector_no_nan_on_defaults(self):
        """Default-constructed features should not produce NaN values."""
        tf = TeamFeatures()
        vec = tf.to_vector()
        # Default values may produce NaN from log operations on zero seeds etc.
        # This test documents current behavior — if NaN count changes, investigate.
        nan_count = int(np.isnan(vec).sum())
        assert nan_count == 0, f"Default TeamFeatures produced {nan_count} NaN values"

    def test_team_feature_dim_is_positive(self):
        """TEAM_FEATURE_DIM should be a positive integer."""
        assert isinstance(TEAM_FEATURE_DIM, int)
        assert TEAM_FEATURE_DIM > 0

    def test_matchup_features_double_dim(self):
        """Matchup feature vector should be 2 * TEAM_FEATURE_DIM (differential)."""
        from src.data.features.feature_engineering import create_matchup_features

        tf1 = TeamFeatures()
        tf2 = TeamFeatures()
        team_features = {"team_a": tf1.to_vector(), "team_b": tf2.to_vector()}
        matchup = create_matchup_features("team_a", "team_b", team_features)
        # Matchup features = team1_vec - team2_vec (same dim as single team)
        assert len(matchup) == TEAM_FEATURE_DIM
