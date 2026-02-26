"""Tests for Bayesian Bradley-Terry rating system."""

import math
import numpy as np
import pytest

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from src.ml.ensemble.bayesian_bt import BayesianBradleyTerry, BayesianRating


def _make_synthetic_games(n_teams=20, games_per_team=30, seed=42):
    """Generate games where team strength is proportional to team index."""
    rng = np.random.RandomState(seed)
    teams = [f"team_{i:02d}" for i in range(n_teams)]
    true_strength = np.linspace(-2.0, 2.0, n_teams)

    games = []
    for _ in range(n_teams * games_per_team // 2):
        i, j = rng.choice(n_teams, size=2, replace=False)
        diff = true_strength[i] - true_strength[j]
        p_win = 1.0 / (1.0 + np.exp(-diff))
        outcome = 1.0 if rng.random() < p_win else 0.0
        games.append((teams[i], teams[j], outcome))

    return games, teams, true_strength


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy not installed")
class TestBayesianBradleyTerry:

    def test_fit_basic(self):
        games, teams, _ = _make_synthetic_games()
        bt = BayesianBradleyTerry(prior_std=2.0)
        stats = bt.fit(games)
        assert stats["fitted"] is True
        assert stats["n_teams"] == 20
        assert stats["converged"] is True
        assert len(bt.ratings) == 20

    def test_strong_teams_rated_higher(self):
        """Teams with higher true strength should have higher posterior mean."""
        games, teams, true_strength = _make_synthetic_games(n_teams=20, games_per_team=60)
        bt = BayesianBradleyTerry(prior_std=2.0)
        bt.fit(games)

        # Compare top 5 vs bottom 5 teams
        top_teams = [teams[i] for i in np.argsort(true_strength)[-5:]]
        bot_teams = [teams[i] for i in np.argsort(true_strength)[:5]]

        top_mean = np.mean([bt.ratings[t].mean for t in top_teams])
        bot_mean = np.mean([bt.ratings[t].mean for t in bot_teams])
        assert top_mean > bot_mean, (
            f"Top teams (mean={top_mean:.3f}) should rate higher than "
            f"bottom teams (mean={bot_mean:.3f})"
        )

    def test_prediction_symmetry(self):
        """P(A beats B) + P(B beats A) should equal 1."""
        games, teams, _ = _make_synthetic_games()
        bt = BayesianBradleyTerry()
        bt.fit(games)

        for i in range(0, len(teams) - 1, 2):
            p_ab, _ = bt.predict_probability(teams[i], teams[i + 1])
            p_ba, _ = bt.predict_probability(teams[i + 1], teams[i])
            assert abs(p_ab + p_ba - 1.0) < 1e-10, (
                f"P({teams[i]}>{teams[i+1]})={p_ab:.6f} + "
                f"P({teams[i+1]}>{teams[i]})={p_ba:.6f} != 1.0"
            )

    def test_unseen_team_returns_prior(self):
        """Unseen teams should produce probability 0.5."""
        games, _, _ = _make_synthetic_games()
        bt = BayesianBradleyTerry()
        bt.fit(games)

        p, unc = bt.predict_probability("unknown_team_A", "unknown_team_B")
        assert abs(p - 0.5) < 1e-7, f"Unseen teams should give P=0.5, got {p}"
        assert unc > 0

    def test_one_unseen_team(self):
        """One known + one unseen team: probability pushed toward 0.5."""
        games, teams, true_strength = _make_synthetic_games()
        bt = BayesianBradleyTerry()
        bt.fit(games)

        best_team = teams[np.argmax(true_strength)]
        p_known, _ = bt.predict_probability(best_team, teams[0])
        p_unseen, _ = bt.predict_probability(best_team, "unseen_team")

        # Prediction against unseen team should be closer to 0.5 than
        # against a known weak team
        assert abs(p_unseen - 0.5) < abs(p_known - 0.5) + 0.1

    def test_uncertainty_higher_for_fewer_games(self):
        """Teams with fewer games should have higher posterior std."""
        rng = np.random.RandomState(42)
        # Create asymmetric game counts
        teams = ["frequent_A", "frequent_B", "rare_C", "rare_D"]
        games = []

        # Frequent teams play many games
        for _ in range(100):
            outcome = 1.0 if rng.random() > 0.5 else 0.0
            games.append(("frequent_A", "frequent_B", outcome))

        # Rare teams play few games
        for _ in range(5):
            outcome = 1.0 if rng.random() > 0.5 else 0.0
            games.append(("rare_C", "rare_D", outcome))

        # Cross-play so all are connected
        for _ in range(3):
            games.append(("frequent_A", "rare_C", 1.0))
            games.append(("frequent_B", "rare_D", 1.0))

        bt = BayesianBradleyTerry()
        bt.fit(games)

        freq_std = bt.ratings["frequent_A"].std
        rare_std = bt.ratings["rare_C"].std
        assert rare_std > freq_std, (
            f"Rare team std ({rare_std:.4f}) should exceed "
            f"frequent team std ({freq_std:.4f})"
        )

    def test_uncertainty_widens_prediction(self):
        """Probit approximation should push high-uncertainty predictions toward 0.5."""
        games, teams, true_strength = _make_synthetic_games(n_teams=20, games_per_team=60)
        bt = BayesianBradleyTerry()
        bt.fit(games)

        best = teams[np.argmax(true_strength)]
        worst = teams[np.argmin(true_strength)]

        p, unc = bt.predict_probability(best, worst)
        # With significant data, prediction should be confident (away from 0.5)
        assert p > 0.6

        # The uncertainty should be non-zero but modest
        assert 0 < unc < 0.5

    def test_unfitted_model(self):
        bt = BayesianBradleyTerry()
        p, unc = bt.predict_probability("A", "B")
        assert p == 0.5
        assert unc == 1.0
        assert bt.get_rating("A") is None

    def test_get_rating(self):
        games, teams, _ = _make_synthetic_games()
        bt = BayesianBradleyTerry()
        bt.fit(games)

        r = bt.get_rating(teams[0])
        assert isinstance(r, BayesianRating)
        assert isinstance(r.mean, float)
        assert isinstance(r.std, float)
        assert r.n_games > 0

    def test_get_rankings(self):
        games, teams, _ = _make_synthetic_games()
        bt = BayesianBradleyTerry()
        bt.fit(games)

        rankings = bt.get_rankings(top_n=5)
        assert len(rankings) == 5
        # Rankings should be sorted descending by mean
        for i in range(len(rankings) - 1):
            assert rankings[i][1].mean >= rankings[i + 1][1].mean

    def test_too_few_games(self):
        bt = BayesianBradleyTerry()
        stats = bt.fit([("A", "B", 1.0)])
        assert stats["fitted"] is False
        assert not bt._fitted

    def test_rating_correlation_with_truth(self):
        """Posterior means should correlate well with true strength."""
        games, teams, true_strength = _make_synthetic_games(
            n_teams=30, games_per_team=80, seed=123
        )
        bt = BayesianBradleyTerry(prior_std=3.0)
        bt.fit(games)

        estimated = np.array([bt.ratings[t].mean for t in teams])
        corr = np.corrcoef(estimated, true_strength)[0, 1]
        assert corr > 0.8, f"Rating-truth correlation too low: {corr:.3f}"

    def test_prior_std_effect(self):
        """Tighter prior should pull ratings closer to prior mean."""
        games, teams, _ = _make_synthetic_games(n_teams=10, games_per_team=20)

        bt_loose = BayesianBradleyTerry(prior_std=10.0)
        bt_loose.fit(games)

        bt_tight = BayesianBradleyTerry(prior_std=0.5)
        bt_tight.fit(games)

        loose_spread = np.std([bt_loose.ratings[t].mean for t in teams])
        tight_spread = np.std([bt_tight.ratings[t].mean for t in teams])
        assert tight_spread < loose_spread, (
            f"Tight prior (std={tight_spread:.3f}) should compress ratings "
            f"more than loose prior (std={loose_spread:.3f})"
        )
