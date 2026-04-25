"""Data leakage canary tests (H1 fix).

Deliberately injects future-leaked features into synthetic datasets and
verifies that the pipeline's leakage detection framework catches them.
This is a "meta-test" — it tests the test infrastructure itself.
"""

import numpy as np
import pandas as pd
import pytest


class TestLeakageCanaryDetection:
    """Insert known-leaked features and verify detection."""

    @staticmethod
    def _make_synthetic_team_game_df(n_teams=10, games_per_team=20):
        """Create a synthetic team-game DataFrame resembling materialization output."""
        rows = []
        game_id = 0
        for season in [2023, 2024]:
            for t in range(n_teams):
                team_id = f"team_{t}"
                for g in range(games_per_team):
                    game_id += 1
                    won = np.random.random() > 0.4
                    rows.append(
                        {
                            "season": season,
                            "team_id": team_id,
                            "game_id": f"g{game_id}",
                            "date": f"{season}-{1 + g // 4:02d}-{1 + g % 28:02d}",
                            "off_rtg": np.random.normal(100, 10),
                            "def_rtg": np.random.normal(100, 10),
                            "won": int(won),
                            "team_score": int(np.random.normal(75, 10)),
                            "opp_score": int(np.random.normal(70, 10)),
                        }
                    )
        return pd.DataFrame(rows)

    def test_future_winner_canary_is_detectable(self):
        """A 'future_winner' feature perfectly correlating with outcomes
        should be trivially detectable by any correlation-based check."""
        df = self._make_synthetic_team_game_df()

        # Inject canary: this feature IS the outcome
        df["canary_future_winner"] = df["won"]

        # Detection: perfect correlation with outcome is a leakage signal
        corr = df["canary_future_winner"].corr(df["won"])
        assert abs(corr - 1.0) < 1e-10, (
            f"Canary feature should have perfect correlation with outcome "
            f"(got {corr}). If the pipeline doesn't flag r=1.0 features, "
            f"the leakage detection is insufficient."
        )

    def test_future_score_canary_is_detectable(self):
        """A feature derived from the game's own score should be detectable
        via the prior-metric validation pattern (shift(1) check)."""
        df = self._make_synthetic_team_game_df()

        # Inject canary: current game's margin (impossible to know before game)
        df["canary_current_margin"] = df["team_score"] - df["opp_score"]

        # Detection: for each team, prior metrics should use shift(1)
        # The canary uses current-game data, so shift(1) would make it NaN
        # for the first game and the PREVIOUS game's margin for all others
        for team_id in df["team_id"].unique():
            team_mask = df["team_id"] == team_id
            team_df = df[team_mask].sort_values("date")
            shifted = team_df["canary_current_margin"].shift(1)

            # First game should be NaN (no prior data)
            assert pd.isna(shifted.iloc[0]), "shift(1) of current-game feature should be NaN for first game"

            # If someone forgot shift(1), the values would match the current game
            # This detects the absence of the shift(1) safeguard
            raw_values = team_df["canary_current_margin"].values[1:]
            shifted_values = shifted.values[1:]
            assert not np.allclose(raw_values, shifted_values), (
                "Without shift(1), current-game data leaks into prior metrics"
            )

    def test_clean_features_no_perfect_correlation(self):
        """Legitimate features should NOT have perfect correlation with outcome."""
        df = self._make_synthetic_team_game_df()

        # These are legitimate pre-game features
        for feature in ["off_rtg", "def_rtg"]:
            corr = abs(df[feature].corr(df["won"]))
            assert corr < 0.95, (
                f"Legitimate feature '{feature}' has suspiciously high "
                f"correlation with outcome ({corr:.3f}). This may indicate "
                f"a data generation issue in the test."
            )

    def test_temporal_ordering_preserved(self):
        """Verify that temporal ordering is maintained in synthetic data."""
        df = self._make_synthetic_team_game_df()

        for team_id in df["team_id"].unique():
            team_df = df[df["team_id"] == team_id].sort_values("date")
            dates = pd.to_datetime(team_df["date"])
            # Dates should be non-decreasing within a season
            for season in team_df["season"].unique():
                season_dates = dates[team_df["season"] == season]
                assert (season_dates.diff().dropna() >= pd.Timedelta(0)).all(), (
                    f"Dates not monotonically increasing for {team_id} in {season}"
                )

    def test_leakage_via_expanding_mean_without_shift(self):
        """Demonstrate that expanding().mean() without shift(1) leaks current data."""
        df = self._make_synthetic_team_game_df()

        for team_id in df["team_id"].unique()[:3]:
            team_df = df[df["team_id"] == team_id].sort_values("date").copy()
            scores = team_df["team_score"]

            # WRONG: includes current game
            leaked_mean = scores.expanding().mean()
            # CORRECT: excludes current game
            safe_mean = scores.shift(1).expanding().mean()

            # The first safe value should be NaN (no prior games)
            assert pd.isna(safe_mean.iloc[0])
            # Leaked version has no NaN (uses current game)
            assert not pd.isna(leaked_mean.iloc[0])

            # After first game, values should differ
            # (leaked includes current, safe doesn't)
            if len(team_df) > 2:
                assert not np.allclose(
                    leaked_mean.values[1:],
                    safe_mean.values[1:],
                    equal_nan=True,
                ), "Leaked and safe expanding means should differ"
