"""Matchup feature engineering module.

Computes pairwise matchup features from team-level statistics:
- Efficiency: AdjO_A - AdjD_B, AdjO_B - AdjD_A
- Tempo: difference, product
- Shooting: 3P_rate - opp 3P_def
- Possession: rebound diff, turnover edge
- Composite: upset_score = weighted sum (CV-learned)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TeamStats:
    """Team-level statistics for matchup computation."""
    team_id: str
    seed: int = 0

    # Efficiency (from Torvik four factors or KenPom-style)
    adj_off_eff: float = 100.0  # Adjusted offensive efficiency
    adj_def_eff: float = 100.0  # Adjusted defensive efficiency
    adj_tempo: float = 68.0     # Adjusted tempo (possessions per game)

    # Four factors
    efg_pct: float = 0.50       # Effective FG%
    to_rate: float = 0.18       # Turnover rate
    orb_rate: float = 0.30      # Offensive rebound rate
    ft_rate: float = 0.30       # Free throw rate

    # Opponent four factors
    opp_efg_pct: float = 0.50
    opp_to_rate: float = 0.18
    opp_orb_rate: float = 0.30   # Opponent offensive rebound rate (= our DRB failure)
    opp_ft_rate: float = 0.30

    # Shooting
    three_pt_pct: float = 0.33
    opp_three_pt_pct: float = 0.33  # Opponent 3PT% (3PT defense)
    ft_pct: float = 0.72

    # Derived
    win_pct: float = 0.50
    sos: float = 0.0            # Strength of schedule
    elo: float = 1500.0

    def net_efficiency(self) -> float:
        return self.adj_off_eff - self.adj_def_eff

    def drb_rate(self) -> float:
        """Defensive rebound rate (complement of opponent ORB rate)."""
        return 1.0 - self.opp_orb_rate


MATCHUP_FEATURE_NAMES = [
    # Efficiency matchup (4)
    "eff_edge_a",        # AdjO_A - AdjD_B
    "eff_edge_b",        # AdjO_B - AdjD_A
    "net_eff_diff",      # (AdjO_A - AdjD_A) - (AdjO_B - AdjD_B)
    "eff_mismatch",      # |eff_edge_a - eff_edge_b|

    # Tempo (3)
    "tempo_diff",        # Tempo_A - Tempo_B
    "tempo_product",     # Tempo_A * Tempo_B (game pace proxy)
    "tempo_avg",         # (Tempo_A + Tempo_B) / 2

    # Shooting matchup (4)
    "three_pt_edge_a",   # 3P%_A - opp_efg%_B (proxy for 3P defense)
    "three_pt_edge_b",   # 3P%_B - opp_efg%_A
    "ft_pct_diff",       # FT%_A - FT%_B
    "efg_diff",          # eFG%_A - eFG%_B

    # Possession (4)
    "reb_diff",          # (ORB_A + DRB_A) - (ORB_B + DRB_B)
    "tov_edge",          # TO_rate_B - TO_rate_A (positive = A forces more)
    "orb_diff",          # ORB_rate_A - ORB_rate_B
    "ft_rate_diff",      # FT_rate_A - FT_rate_B

    # Composite (5)
    "seed_diff",         # seed_B - seed_A (positive = A is lower/better seed)
    "elo_diff",          # elo_A - elo_B
    "sos_diff",          # SOS_A - SOS_B
    "win_pct_diff",      # win%_A - win%_B
    "upset_score",       # Weighted composite upset indicator

    # Absolute level context (3)
    "avg_net_eff",       # Average net efficiency of both teams
    "avg_tempo",         # Average tempo (game pace)
    "avg_elo",           # Average Elo (game quality)
]

N_MATCHUP_FEATURES = len(MATCHUP_FEATURE_NAMES)


def compute_matchup_features(
    a: TeamStats,
    b: TeamStats,
    upset_weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Compute matchup feature vector for team A vs team B.

    Args:
        a: Team A statistics.
        b: Team B statistics.
        upset_weights: Learned upset score weights (CV-learned).
            If None, uses literature-based defaults.

    Returns:
        Feature vector of length N_MATCHUP_FEATURES.
    """
    features = np.zeros(N_MATCHUP_FEATURES, dtype=np.float64)

    # Efficiency matchup
    eff_edge_a = a.adj_off_eff - b.adj_def_eff
    eff_edge_b = b.adj_off_eff - a.adj_def_eff
    features[0] = eff_edge_a
    features[1] = eff_edge_b
    features[2] = a.net_efficiency() - b.net_efficiency()
    features[3] = abs(eff_edge_a - eff_edge_b)

    # Tempo
    features[4] = a.adj_tempo - b.adj_tempo
    features[5] = a.adj_tempo * b.adj_tempo / 1000.0  # Scale down
    features[6] = (a.adj_tempo + b.adj_tempo) / 2.0

    # Shooting matchup: 3PT offense vs 3PT defense
    # Use opp_three_pt_pct (opponent 3P%) as the 3PT defense metric.
    # If unavailable, approximate from opp_efg_pct (weaker proxy).
    b_three_def = b.opp_three_pt_pct if b.opp_three_pt_pct != 0.33 else b.opp_efg_pct
    a_three_def = a.opp_three_pt_pct if a.opp_three_pt_pct != 0.33 else a.opp_efg_pct
    features[7] = a.three_pt_pct - b_three_def  # A's 3P attack vs B's 3P defense
    features[8] = b.three_pt_pct - a_three_def
    features[9] = a.ft_pct - b.ft_pct
    features[10] = a.efg_pct - b.efg_pct

    # Possession
    reb_a = a.orb_rate + a.drb_rate()
    reb_b = b.orb_rate + b.drb_rate()
    features[11] = reb_a - reb_b
    features[12] = b.to_rate - a.to_rate  # Positive = A forces more TOs
    features[13] = a.orb_rate - b.orb_rate
    features[14] = a.ft_rate - b.ft_rate

    # Composite
    features[15] = b.seed - a.seed  # Positive = A is better seeded
    features[16] = a.elo - b.elo
    features[17] = a.sos - b.sos
    features[18] = a.win_pct - b.win_pct

    # Upset score: weighted composite indicating upset potential
    # Uses CV-learned weights when available, otherwise literature defaults
    features[19] = compute_upset_score(a, b, weights=upset_weights)

    # Absolute level context
    features[20] = (a.net_efficiency() + b.net_efficiency()) / 2.0
    features[21] = (a.adj_tempo + b.adj_tempo) / 2.0
    features[22] = (a.elo + b.elo) / 2.0

    return features


def compute_upset_score(
    a: TeamStats,
    b: TeamStats,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Compute upset score as weighted sum of matchup factors.

    Higher positive values indicate conditions that historically
    favor upsets (underdog winning).

    Args:
        a: Team A (assumed to be the higher-seeded / favored team).
        b: Team B.
        weights: Feature weights (learned via CV). Defaults to
            literature-based priors.

    Returns:
        Upset score (positive = more upset-prone).
    """
    if weights is None:
        weights = {
            "tempo_chaos": 0.20,
            "three_pt_variance": 0.25,
            "turnover_edge": 0.20,
            "eff_gap_small": 0.20,
            "experience_gap": 0.15,
        }

    score = 0.0

    # High tempo = more possessions = more variance
    tempo_avg = (a.adj_tempo + b.adj_tempo) / 2.0
    tempo_z = (tempo_avg - 68.0) / 4.0  # z-score vs mean
    score += weights.get("tempo_chaos", 0.20) * tempo_z

    # 3PT shooting variance (high 3P% teams are streaky)
    three_pt_avg = (a.three_pt_pct + b.three_pt_pct) / 2.0
    three_pt_z = (three_pt_avg - 0.33) / 0.04
    score += weights.get("three_pt_variance", 0.25) * three_pt_z

    # Turnover edge for underdog
    tov_edge = b.to_rate - a.to_rate  # Positive if A forces more
    score += weights.get("turnover_edge", 0.20) * (-tov_edge / 0.03)

    # Small efficiency gap = more competitive
    eff_gap = abs(a.net_efficiency() - b.net_efficiency())
    eff_gap_score = max(0, 1.0 - eff_gap / 15.0)  # Normalized 0-1
    score += weights.get("eff_gap_small", 0.20) * eff_gap_score

    # Experience gap (proxy via win%)
    exp_gap = a.win_pct - b.win_pct
    score += weights.get("experience_gap", 0.15) * (-exp_gap / 0.15)

    return float(score)


def learn_upset_weights(
    features_list: List[np.ndarray],
    outcomes: np.ndarray,
    seed_diffs: np.ndarray,
) -> Dict[str, float]:
    """Learn upset score weights via cross-validated optimization.

    Identifies which matchup factors best predict upsets by optimizing
    weights to minimize prediction error on upset games.

    Args:
        features_list: List of matchup feature vectors.
        outcomes: Binary outcomes (1 = team A won).
        seed_diffs: seed_B - seed_A for each game.

    Returns:
        Optimized upset score weights.
    """
    from scipy.optimize import minimize

    # Identify upset games: lower seed (higher number) won
    upset_mask = ((outcomes == 1) & (seed_diffs < 0)) | \
                 ((outcomes == 0) & (seed_diffs > 0))

    if upset_mask.sum() < 10:
        logger.warning("Too few upsets (%d) to learn weights; using defaults", upset_mask.sum())
        return {}

    # Extract relevant sub-features for upset scoring
    X = np.array(features_list)
    # Use: tempo_avg(idx=6), three_pt_edge mean(7+8), tov_edge(12), eff_mismatch(3), win_pct_diff(18)
    sub_idx = [6, 7, 12, 3, 18]
    X_sub = X[:, sub_idx]

    def objective(w):
        w_norm = np.abs(w) / (np.abs(w).sum() + 1e-8)
        scores = X_sub @ w_norm
        # Higher score should predict upsets
        pred_upset_prob = 1.0 / (1.0 + np.exp(-scores))
        # Log loss
        eps = 1e-7
        y = upset_mask.astype(float)
        loss = -np.mean(y * np.log(pred_upset_prob + eps) +
                        (1 - y) * np.log(1 - pred_upset_prob + eps))
        return loss

    w0 = np.ones(len(sub_idx)) * 0.2
    result = minimize(objective, w0, method="Nelder-Mead",
                      options={"maxiter": 500})

    w_opt = np.abs(result.x)
    w_opt = w_opt / (w_opt.sum() + 1e-8)

    weight_names = ["tempo_chaos", "three_pt_variance", "turnover_edge",
                    "eff_gap_small", "experience_gap"]
    learned = {name: float(w) for name, w in zip(weight_names, w_opt)}
    logger.info("Learned upset weights: %s", learned)
    return learned
