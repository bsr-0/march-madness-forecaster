"""Bayesian Bradley-Terry rating system with uncertainty propagation.

Fits a Bradley-Terry model with Gaussian priors via MAP estimation:

    P(team_i beats team_j) = sigmoid(theta_i - theta_j)
    Prior: theta_i ~ N(prior_mean, prior_std^2)

Posterior approximation via Laplace method (MAP + inverse Hessian).
Prediction with uncertainty propagation uses the probit approximation:

    P(win) ≈ sigmoid((mu_i - mu_j) / sqrt(1 + pi/8 * (sig_i^2 + sig_j^2)))

This naturally shrinks predictions toward 0.5 when either team has high
uncertainty (few games), providing built-in regularization for rare matchups
like 16-seeds with only ~30 games.

Reference: Bradley & Terry (1952), Glickman (1999), Murphy (2012 §8.4.4).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import minimize
    from scipy.sparse import csc_matrix
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class BayesianRating:
    """Posterior rating for a single team."""
    mean: float       # Posterior mean (MAP estimate)
    std: float        # Posterior standard deviation (from Hessian)
    n_games: int      # Number of observed games


class BayesianBradleyTerry:
    """Bradley-Terry model with Gaussian priors and Laplace posterior.

    The log-posterior is strictly concave (logistic log-likelihood + Gaussian
    log-prior), so the MAP estimate is unique and found efficiently via
    L-BFGS-B.  The Hessian at the MAP provides a Gaussian approximation
    to the posterior, giving per-team uncertainty estimates.

    At prediction time, uncertainty is propagated via the probit
    approximation to the logistic function, naturally widening predictions
    toward 0.5 for under-sampled teams.
    """

    def __init__(
        self,
        prior_mean: float = 0.0,
        prior_std: float = 2.0,
    ):
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.ratings: Dict[str, BayesianRating] = {}
        self._team_to_idx: Dict[str, int] = {}
        self._idx_to_team: Dict[int, str] = {}
        self._theta: Optional[np.ndarray] = None
        self._theta_std: Optional[np.ndarray] = None
        self._fitted = False

    def fit(
        self,
        games: List[Tuple[str, str, float]],
        max_iter: int = 200,
    ) -> Dict:
        """Fit the model via MAP estimation.

        Args:
            games: List of (team1_id, team2_id, team1_won) triples.
                   team1_won should be 1.0 if team1 won, 0.0 if team2 won.
            max_iter: Maximum L-BFGS-B iterations.

        Returns:
            Dict with fitting statistics.
        """
        if not SCIPY_AVAILABLE:
            logger.warning("scipy not available; Bayesian BT model disabled.")
            return {"fitted": False, "reason": "scipy_unavailable"}

        if len(games) < 10:
            logger.warning("Too few games (%d) for Bradley-Terry.", len(games))
            return {"fitted": False, "reason": "insufficient_games"}

        # Build team index
        teams = set()
        for t1, t2, _ in games:
            teams.add(t1)
            teams.add(t2)

        teams_sorted = sorted(teams)
        self._team_to_idx = {t: i for i, t in enumerate(teams_sorted)}
        self._idx_to_team = {i: t for t, i in self._team_to_idx.items()}
        n_teams = len(teams_sorted)

        # Count games per team
        game_counts = np.zeros(n_teams, dtype=int)
        for t1, t2, _ in games:
            game_counts[self._team_to_idx[t1]] += 1
            game_counts[self._team_to_idx[t2]] += 1

        # Convert games to index arrays for vectorized computation
        idx1 = np.array([self._team_to_idx[t1] for t1, _, _ in games])
        idx2 = np.array([self._team_to_idx[t2] for _, t2, _ in games])
        outcomes = np.array([y for _, _, y in games])

        prior_var = self.prior_std ** 2

        def neg_log_posterior(theta):
            """Negative log-posterior (to minimize)."""
            diff = theta[idx1] - theta[idx2]
            # Log-likelihood: sum of log P(outcome | theta)
            # P(team1 wins) = sigmoid(theta1 - theta2)
            # log P = y * log(sigmoid(d)) + (1-y) * log(1 - sigmoid(d))
            #       = y * d - log(1 + exp(d))
            # Use numerically stable log-sigmoid
            log_lik = np.sum(outcomes * diff - _softplus(diff))

            # Log-prior: -0.5 * sum((theta - prior_mean)^2 / prior_var)
            log_prior = -0.5 * np.sum((theta - self.prior_mean) ** 2) / prior_var

            return -(log_lik + log_prior)

        def neg_log_posterior_grad(theta):
            """Gradient of negative log-posterior."""
            diff = theta[idx1] - theta[idx2]
            sigmoid_d = _sigmoid(diff)

            # Gradient of log-likelihood w.r.t. theta
            residuals = outcomes - sigmoid_d  # [n_games]
            grad = np.zeros(n_teams)
            np.add.at(grad, idx1, residuals)
            np.add.at(grad, idx2, -residuals)

            # Gradient of log-prior
            grad -= (theta - self.prior_mean) / prior_var

            return -grad

        # Initialize at prior mean
        theta0 = np.full(n_teams, self.prior_mean)

        result = minimize(
            neg_log_posterior,
            theta0,
            jac=neg_log_posterior_grad,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "ftol": 1e-8},
        )

        self._theta = result.x

        # Compute Hessian diagonal for posterior variance approximation
        # H_ii = sum_games_involving_i [sigmoid(d) * (1-sigmoid(d))] + 1/prior_var
        diff_at_map = self._theta[idx1] - self._theta[idx2]
        sig_d = _sigmoid(diff_at_map)
        hess_contrib = sig_d * (1.0 - sig_d)  # [n_games]

        hess_diag = np.full(n_teams, 1.0 / prior_var)  # Prior contribution
        np.add.at(hess_diag, idx1, hess_contrib)
        np.add.at(hess_diag, idx2, hess_contrib)

        # Posterior std ≈ 1/sqrt(H_ii) (diagonal Laplace approximation)
        self._theta_std = 1.0 / np.sqrt(np.maximum(hess_diag, 1e-10))

        # Build ratings dict
        for team, idx in self._team_to_idx.items():
            self.ratings[team] = BayesianRating(
                mean=float(self._theta[idx]),
                std=float(self._theta_std[idx]),
                n_games=int(game_counts[idx]),
            )

        self._fitted = True

        # Compute statistics
        rating_values = self._theta
        stats = {
            "fitted": True,
            "n_teams": n_teams,
            "n_games": len(games),
            "converged": result.success,
            "iterations": result.nit,
            "mean_rating": round(float(np.mean(rating_values)), 4),
            "std_rating": round(float(np.std(rating_values)), 4),
            "mean_posterior_std": round(float(np.mean(self._theta_std)), 4),
            "prior_std": self.prior_std,
        }

        logger.info(
            "BayesianBT: fitted %d teams from %d games, "
            "converged=%s, mean_posterior_std=%.3f",
            n_teams, len(games), result.success,
            float(np.mean(self._theta_std)),
        )
        return stats

    def predict_probability(
        self,
        team1_id: str,
        team2_id: str,
    ) -> Tuple[float, float]:
        """Predict win probability with uncertainty.

        Uses probit approximation to propagate posterior uncertainty:
            P(win) ≈ sigmoid((mu1 - mu2) / sqrt(1 + pi/8 * (sig1^2 + sig2^2)))

        Args:
            team1_id: First team identifier.
            team2_id: Second team identifier.

        Returns:
            (probability, uncertainty) where uncertainty is the posterior
            std of the probability estimate.
        """
        if not self._fitted:
            return (0.5, 1.0)

        r1 = self.ratings.get(team1_id)
        r2 = self.ratings.get(team2_id)

        if r1 is None and r2 is None:
            return (0.5, 1.0)

        # For unseen teams, use the prior
        mu1 = r1.mean if r1 else self.prior_mean
        mu2 = r2.mean if r2 else self.prior_mean
        sig1 = r1.std if r1 else self.prior_std
        sig2 = r2.std if r2 else self.prior_std

        # Probit approximation: scale factor accounts for uncertainty
        # pi/8 ≈ 0.3927
        scale = math.sqrt(1.0 + (math.pi / 8.0) * (sig1 ** 2 + sig2 ** 2))
        prob = _sigmoid_scalar((mu1 - mu2) / scale)

        # Uncertainty: approximate std of the probability via delta method
        # Var(P) ≈ (dP/d_delta)^2 * (sig1^2 + sig2^2) where dP/d_delta = P*(1-P)/scale
        dp_ddelta = prob * (1.0 - prob) / scale
        uncertainty = dp_ddelta * math.sqrt(sig1 ** 2 + sig2 ** 2)

        return (float(prob), float(uncertainty))

    def get_rating(self, team_id: str) -> Optional[BayesianRating]:
        """Get posterior rating for a team."""
        return self.ratings.get(team_id)

    def get_rankings(self, top_n: Optional[int] = None) -> List[Tuple[str, BayesianRating]]:
        """Get teams ranked by posterior mean rating."""
        ranked = sorted(
            self.ratings.items(),
            key=lambda x: x[1].mean,
            reverse=True,
        )
        if top_n is not None:
            ranked = ranked[:top_n]
        return ranked


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for arrays."""
    result = np.empty_like(x, dtype=float)
    pos = x >= 0
    neg = ~pos
    result[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    result[neg] = exp_x / (1.0 + exp_x)
    return result


def _sigmoid_scalar(x: float) -> float:
    """Numerically stable sigmoid for scalar."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def _softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable log(1 + exp(x))."""
    result = np.empty_like(x, dtype=float)
    large = x > 20
    small = x < -20
    mid = ~(large | small)
    result[large] = x[large]
    result[small] = np.exp(x[small])
    result[mid] = np.log1p(np.exp(x[mid]))
    return result
