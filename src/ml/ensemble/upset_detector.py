"""Upset Detection Layer for March Madness bracket predictions.

Dedicated post-ensemble layer that identifies matchups where the lower-seeded
team (underdog) has an elevated probability of winning. Uses a Bayesian
framework combining:

1. **Historical seed-matchup priors** — base upset rates from 1985-2025
   (~2500+ first-round games) provide strong priors for canonical matchups.
2. **Model residual signal** — when the ensemble's P(upset) deviates from
   the historical prior, the deviation carries information about *this
   specific matchup* (e.g., a 12-seed with elite efficiency metrics).
3. **Upset amplifier features** — team-level signals that historically
   correlate with upsets: high 3PT variance, strong momentum, favorable
   experience gap, high pace-adjusted variance.

Output:
- Per-matchup ``UpsetSignal`` with upset probability, confidence, and
  human-readable risk tier (VERY_HIGH / HIGH / MODERATE / LOW / MINIMAL).
- Adjusted probabilities that nudge the ensemble toward historically
  calibrated upset rates when the model is overconfident on favorites.

Integration points:
- Bracket optimizers: use upset_score to diversify bracket portfolios.
- Kaggle submissions: use adjusted_prob for better-calibrated predictions.
- Simulation: feed adjusted probabilities into Monte Carlo engine.

Design principle: this layer *adjusts* ensemble probabilities rather than
replacing them. The ensemble remains the primary model; the upset detector
acts as a calibration correction focused on the upset tail.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =========================================================================
# Historical upset rates by seed matchup (1985-2025, men's NCAA tournament)
# P(higher_seed_wins) for canonical R64 pairings.
# Source: cross-referenced from tournament_features.py and monte_carlo.py
# =========================================================================
HISTORICAL_UPSET_RATES: Dict[Tuple[int, int], float] = {
    # (lower_seed, higher_seed) -> P(higher_seed_wins)
    (1, 16): 0.007,   # UMBC over Virginia (2018) — only instance
    (2, 15): 0.057,   # ~6% — St. Peter's, Oral Roberts, etc.
    (3, 14): 0.150,   # ~15% — Abilene Christian, Mercer, etc.
    (4, 13): 0.210,   # ~21% — frequent mid-major upsets
    (5, 12): 0.356,   # ~36% — the classic "upset special"
    (6, 11): 0.377,   # ~38% — First Four teams often dangerous
    (7, 10): 0.392,   # ~39% — near coin-flip territory
    (8, 9):  0.481,   # ~48% — essentially a toss-up
}

# Extended matchups for later rounds (less data, wider uncertainty)
HISTORICAL_UPSET_RATES_EXTENDED: Dict[Tuple[int, int], float] = {
    (1, 8): 0.204,   (1, 9): 0.173,
    (2, 7): 0.330,   (2, 10): 0.300,
    (3, 6): 0.420,   (3, 11): 0.385,
    (4, 5): 0.461,   (4, 12): 0.380,
    (1, 4): 0.370,   (1, 5): 0.315,
    (2, 3): 0.445,   (2, 6): 0.395,
    (1, 2): 0.465,   (1, 3): 0.390,
}

# Number of historical observations per matchup (approximate, for
# computing prior strength in Bayesian updates)
MATCHUP_SAMPLE_SIZES: Dict[Tuple[int, int], int] = {
    (1, 16): 156, (2, 15): 156, (3, 14): 156, (4, 13): 156,
    (5, 12): 156, (6, 11): 156, (7, 10): 156, (8, 9): 156,
    # Later rounds have fewer observations
    (1, 8): 80, (1, 9): 40, (2, 7): 80, (2, 10): 35,
    (3, 6): 70, (3, 11): 30, (4, 5): 75, (4, 12): 25,
    (1, 4): 60, (1, 5): 25, (2, 3): 55, (2, 6): 30,
    (1, 2): 40, (1, 3): 25,
}


class UpsetRiskTier(str, Enum):
    """Categorical upset risk classification."""
    VERY_HIGH = "VERY_HIGH"  # >= 40% upset probability
    HIGH = "HIGH"            # 25-40%
    MODERATE = "MODERATE"    # 15-25%
    LOW = "LOW"              # 5-15%
    MINIMAL = "MINIMAL"      # < 5%


@dataclass
class UpsetSignal:
    """Upset detection output for a single matchup."""

    team1_id: str
    team2_id: str
    seed1: int
    seed2: int

    # Which team is the underdog (higher seed number)?
    underdog_id: str
    favorite_id: str

    # Core outputs
    upset_prob: float           # P(underdog wins) — Bayesian posterior
    historical_prior: float     # P(underdog wins) from seed history alone
    model_upset_prob: float     # P(underdog wins) from ensemble model
    adjusted_prob: float        # Team1 win prob after upset adjustment

    # Risk assessment
    risk_tier: UpsetRiskTier
    upset_score: float          # Composite score [0, 1] for ranking upsets

    # Amplifier signals (normalized to [0, 1])
    volatility_signal: float = 0.0    # Underdog's 3PT + pace variance
    momentum_signal: float = 0.0      # Underdog's recent momentum advantage
    experience_signal: float = 0.0    # Underdog's experience/depth advantage
    efficiency_signal: float = 0.0    # Underdog overperforms seed expectation

    # Confidence in the upset signal
    confidence: float = 0.5     # Higher = more data supporting assessment

    def to_dict(self) -> Dict:
        """Serialize for reports and bracket optimizers."""
        return {
            "team1_id": self.team1_id,
            "team2_id": self.team2_id,
            "seed1": self.seed1,
            "seed2": self.seed2,
            "underdog_id": self.underdog_id,
            "favorite_id": self.favorite_id,
            "upset_prob": round(self.upset_prob, 4),
            "historical_prior": round(self.historical_prior, 4),
            "model_upset_prob": round(self.model_upset_prob, 4),
            "adjusted_prob": round(self.adjusted_prob, 4),
            "risk_tier": self.risk_tier.value,
            "upset_score": round(self.upset_score, 4),
            "volatility_signal": round(self.volatility_signal, 4),
            "momentum_signal": round(self.momentum_signal, 4),
            "experience_signal": round(self.experience_signal, 4),
            "efficiency_signal": round(self.efficiency_signal, 4),
            "confidence": round(self.confidence, 4),
        }


def _classify_risk_tier(upset_prob: float) -> UpsetRiskTier:
    """Map upset probability to categorical risk tier."""
    if upset_prob >= 0.40:
        return UpsetRiskTier.VERY_HIGH
    elif upset_prob >= 0.25:
        return UpsetRiskTier.HIGH
    elif upset_prob >= 0.15:
        return UpsetRiskTier.MODERATE
    elif upset_prob >= 0.05:
        return UpsetRiskTier.LOW
    else:
        return UpsetRiskTier.MINIMAL


def _get_historical_upset_rate(seed_lo: int, seed_hi: int) -> Optional[float]:
    """Look up historical upset rate for a seed matchup.

    Args:
        seed_lo: Lower seed number (stronger team, e.g. 1)
        seed_hi: Higher seed number (weaker team, e.g. 16)

    Returns:
        P(higher_seed wins) or None if no data.
    """
    key = (seed_lo, seed_hi)
    rate = HISTORICAL_UPSET_RATES.get(key)
    if rate is not None:
        return rate
    return HISTORICAL_UPSET_RATES_EXTENDED.get(key)


def _logistic_upset_fallback(seed_lo: int, seed_hi: int) -> float:
    """Logistic approximation for upset rate when no historical data exists.

    Fitted to historical NCAA data: P(upset) = 1 / (1 + exp(-0.175 * (hi - lo)))
    adjusted to return P(higher_seed_wins).
    """
    seed_gap = seed_hi - seed_lo
    # As gap shrinks, upset becomes more likely
    return 1.0 / (1.0 + math.exp(0.175 * seed_gap))


@dataclass
class UpsetDetector:
    """Post-ensemble upset detection and probability adjustment layer.

    Combines historical seed priors with model predictions and team-level
    upset amplifier signals to produce calibrated upset probabilities.

    The Bayesian update uses historical prior strength proportional to the
    number of historical observations for each seed matchup. Matchups with
    more data (R64 canonical pairings, ~156 games each) anchor more strongly
    to the prior, while rare later-round matchups defer more to the model.
    """

    # How much to weight the historical prior vs model (0 = pure model, 1 = pure prior)
    # This controls the overall prior strength scaling. The actual per-matchup
    # weight is further modulated by sample size.
    prior_strength: float = 0.20

    # Amplifier weights: how much each upset signal contributes to the
    # composite upset_score. Must sum to 1.0.
    w_volatility: float = 0.25
    w_momentum: float = 0.20
    w_experience: float = 0.15
    w_efficiency: float = 0.40

    # Adjustment strength: how much the upset detector can shift the
    # ensemble's probability. 0.0 = no adjustment, 1.0 = full override.
    adjustment_strength: float = 0.15

    # Minimum seed gap to consider for upset detection. Same-seed or
    # adjacent-seed matchups don't have meaningful "upsets".
    min_seed_gap: int = 1

    fitted: bool = False
    _fit_stats: Dict = field(default_factory=dict)

    def detect(
        self,
        team1_id: str,
        team2_id: str,
        seed1: int,
        seed2: int,
        model_prob_team1: float,
        team1_features: Optional[object] = None,
        team2_features: Optional[object] = None,
    ) -> UpsetSignal:
        """Detect upset potential for a single matchup.

        Args:
            team1_id: Team 1 identifier
            team2_id: Team 2 identifier
            seed1: Team 1 seed (1-16)
            seed2: Team 2 seed (1-16)
            model_prob_team1: Ensemble model's P(team1 wins)
            team1_features: Optional TeamFeatures for team 1
            team2_features: Optional TeamFeatures for team 2

        Returns:
            UpsetSignal with upset assessment and adjusted probability.
        """
        seed_gap = abs(seed1 - seed2)

        # Identify favorite and underdog
        if seed1 < seed2:
            # Team 1 is favorite (lower seed number = better team)
            favorite_id, underdog_id = team1_id, team2_id
            fav_seed, dog_seed = seed1, seed2
            model_upset_prob = 1.0 - model_prob_team1  # P(underdog wins)
            fav_features, dog_features = team1_features, team2_features
        elif seed2 < seed1:
            favorite_id, underdog_id = team2_id, team1_id
            fav_seed, dog_seed = seed2, seed1
            model_upset_prob = model_prob_team1  # Team 1 IS the underdog
            fav_features, dog_features = team2_features, team1_features
        else:
            # Same seed — no meaningful upset direction
            return self._same_seed_signal(
                team1_id, team2_id, seed1, seed2, model_prob_team1,
            )

        # Step 1: Historical prior
        hist_rate = _get_historical_upset_rate(fav_seed, dog_seed)
        if hist_rate is None:
            hist_rate = _logistic_upset_fallback(fav_seed, dog_seed)

        # Step 2: Compute amplifier signals from team features
        vol_signal, mom_signal, exp_signal, eff_signal = (
            self._compute_amplifier_signals(
                fav_features, dog_features, fav_seed, dog_seed,
            )
        )

        # Step 3: Composite upset amplifier
        amplifier = (
            self.w_volatility * vol_signal
            + self.w_momentum * mom_signal
            + self.w_experience * exp_signal
            + self.w_efficiency * eff_signal
        )

        # Step 4: Bayesian update — blend historical prior with model
        n_obs = MATCHUP_SAMPLE_SIZES.get((fav_seed, dog_seed), 20)
        # Effective prior strength scales with observation count
        # More data = stronger prior anchor
        alpha = self.prior_strength * math.log1p(n_obs) / math.log1p(156)
        alpha = min(alpha, 0.40)  # Cap prior influence

        # Posterior upset probability: weighted blend
        posterior_upset = (
            (1.0 - alpha) * model_upset_prob
            + alpha * hist_rate
        )

        # Step 5: Amplifier adjustment — shift toward upset when signals are strong
        # The amplifier nudges the posterior when team-specific signals suggest
        # the underdog is better than its seed implies
        amplifier_shift = self.adjustment_strength * (amplifier - 0.5) * 2.0
        # Scale shift by seed gap (bigger gaps = smaller absolute shifts)
        gap_damping = min(1.0, 4.0 / max(seed_gap, 1))
        adjusted_upset = posterior_upset + amplifier_shift * gap_damping

        # Clip to valid range
        adjusted_upset = float(np.clip(adjusted_upset, 0.003, 0.997))

        # Step 6: Convert back to team1 win probability
        if seed1 < seed2:
            adjusted_prob = 1.0 - adjusted_upset
        else:
            adjusted_prob = adjusted_upset
        adjusted_prob = float(np.clip(adjusted_prob, 0.003, 0.997))

        # Step 7: Confidence based on signal agreement
        confidence = self._compute_confidence(
            model_upset_prob, hist_rate, amplifier, n_obs,
        )

        # Step 8: Composite upset score for ranking
        upset_score = self._compute_upset_score(
            adjusted_upset, amplifier, confidence, seed_gap,
        )

        risk_tier = _classify_risk_tier(adjusted_upset)

        return UpsetSignal(
            team1_id=team1_id,
            team2_id=team2_id,
            seed1=seed1,
            seed2=seed2,
            underdog_id=underdog_id,
            favorite_id=favorite_id,
            upset_prob=adjusted_upset,
            historical_prior=hist_rate,
            model_upset_prob=model_upset_prob,
            adjusted_prob=adjusted_prob,
            risk_tier=risk_tier,
            upset_score=upset_score,
            volatility_signal=vol_signal,
            momentum_signal=mom_signal,
            experience_signal=exp_signal,
            efficiency_signal=eff_signal,
            confidence=confidence,
        )

    def detect_all(
        self,
        matchups: List[Dict],
        team_features: Optional[Dict[str, object]] = None,
    ) -> List[UpsetSignal]:
        """Detect upsets for all tournament matchups.

        Args:
            matchups: List of dicts with keys:
                team1_id, team2_id, seed1, seed2, model_prob_team1
            team_features: Optional dict of team_id -> TeamFeatures

        Returns:
            List of UpsetSignal sorted by upset_score descending.
        """
        features = team_features or {}
        signals = []
        for m in matchups:
            signal = self.detect(
                team1_id=m["team1_id"],
                team2_id=m["team2_id"],
                seed1=m["seed1"],
                seed2=m["seed2"],
                model_prob_team1=m["model_prob_team1"],
                team1_features=features.get(m["team1_id"]),
                team2_features=features.get(m["team2_id"]),
            )
            signals.append(signal)

        # Sort by upset score (most likely upsets first)
        signals.sort(key=lambda s: s.upset_score, reverse=True)
        return signals

    def fit(
        self,
        model_probs: np.ndarray,
        outcomes: np.ndarray,
        seeds1: np.ndarray,
        seeds2: np.ndarray,
    ) -> Dict:
        """Calibrate detector parameters on historical tournament data.

        Optimizes prior_strength and adjustment_strength to minimize
        Brier score on the upset subset (games where higher seed won).

        Args:
            model_probs: Model P(team1 wins) [N]
            outcomes: Actual outcomes (1 = team1 won) [N]
            seeds1: Team 1 seeds [N]
            seeds2: Team 2 seeds [N]

        Returns:
            Dict with fitting statistics.
        """
        model_probs = np.asarray(model_probs, dtype=np.float64)
        outcomes = np.asarray(outcomes, dtype=np.float64)
        seeds1 = np.asarray(seeds1, dtype=int)
        seeds2 = np.asarray(seeds2, dtype=int)

        n = len(model_probs)
        if n < 30:
            logger.warning(
                "UpsetDetector: too few samples (%d), using defaults", n,
            )
            self.fitted = True
            return {"fitted": True, "n_samples": n, "method": "default"}

        # Identify upset games (higher seed won)
        is_upset = np.array([
            (s1 < s2 and o < 0.5) or (s2 < s1 and o > 0.5)
            for s1, s2, o in zip(seeds1, seeds2, outcomes)
        ])
        n_upsets = int(np.sum(is_upset))

        # Grid search over prior_strength
        best_brier = float("inf")
        best_ps = self.prior_strength

        for ps in np.linspace(0.05, 0.40, 15):
            self.prior_strength = ps
            adjusted = np.array([
                self._quick_adjust(model_probs[i], seeds1[i], seeds2[i])
                for i in range(n)
            ])
            brier = float(np.mean((adjusted - outcomes) ** 2))
            if brier < best_brier:
                best_brier = brier
                best_ps = ps

        self.prior_strength = best_ps
        self.fitted = True

        # Compute stats
        baseline_brier = float(np.mean((model_probs - outcomes) ** 2))
        self._fit_stats = {
            "fitted": True,
            "n_samples": n,
            "n_upsets": n_upsets,
            "upset_rate": round(n_upsets / max(n, 1), 4),
            "prior_strength": round(best_ps, 4),
            "brier_before": round(baseline_brier, 5),
            "brier_after": round(best_brier, 5),
            "brier_delta": round(best_brier - baseline_brier, 5),
        }
        logger.info(
            "UpsetDetector fitted: prior_strength=%.3f, Brier %.5f -> %.5f "
            "(%d upsets in %d games)",
            best_ps, baseline_brier, best_brier, n_upsets, n,
        )
        return self._fit_stats

    def get_upset_summary(
        self, signals: List[UpsetSignal],
    ) -> Dict:
        """Generate summary statistics for bracket optimizer consumption.

        Args:
            signals: List of UpsetSignal from detect_all()

        Returns:
            Summary dict with upset counts by tier, top upset picks, etc.
        """
        by_tier = {}
        for tier in UpsetRiskTier:
            tier_signals = [s for s in signals if s.risk_tier == tier]
            by_tier[tier.value] = {
                "count": len(tier_signals),
                "matchups": [
                    {
                        "underdog": s.underdog_id,
                        "favorite": s.favorite_id,
                        "seeds": f"{s.seed1}v{s.seed2}",
                        "upset_prob": round(s.upset_prob, 3),
                        "upset_score": round(s.upset_score, 3),
                    }
                    for s in sorted(tier_signals, key=lambda x: x.upset_score, reverse=True)
                ],
            }

        top_upsets = [
            s for s in signals
            if s.risk_tier in (UpsetRiskTier.VERY_HIGH, UpsetRiskTier.HIGH)
        ]

        return {
            "total_matchups": len(signals),
            "by_tier": by_tier,
            "top_upset_picks": [s.to_dict() for s in top_upsets[:10]],
            "expected_upsets_r64": round(
                sum(s.upset_prob for s in signals if abs(s.seed1 - s.seed2) > 0),
                1,
            ),
        }

    # ----- Internal methods -----

    def _compute_amplifier_signals(
        self,
        fav_features: Optional[object],
        dog_features: Optional[object],
        fav_seed: int,
        dog_seed: int,
    ) -> Tuple[float, float, float, float]:
        """Extract upset amplifier signals from team features.

        Returns (volatility, momentum, experience, efficiency) each in [0, 1].
        """
        if fav_features is None or dog_features is None:
            return (0.5, 0.5, 0.5, 0.5)

        # Volatility signal: high favorite variance + low underdog variance = upset risk
        # A volatile favorite is more likely to have a "bad day"
        fav_vol = (
            getattr(fav_features, 'three_pt_variance', 0.095)
            + getattr(fav_features, 'pace_adjusted_variance', 0.0)
        )
        dog_vol = (
            getattr(dog_features, 'three_pt_variance', 0.095)
            + getattr(dog_features, 'pace_adjusted_variance', 0.0)
        )
        # Higher = more upset-prone (favorite is volatile, underdog is steady)
        vol_raw = fav_vol - dog_vol
        # Normalize using sigmoid: center at 0, scale 0.1
        vol_signal = 1.0 / (1.0 + math.exp(-vol_raw / 0.10))

        # Momentum signal: underdog trending up, favorite trending down
        dog_momentum = getattr(dog_features, 'momentum', 0.0)
        fav_momentum = getattr(fav_features, 'momentum', 0.0)
        mom_diff = dog_momentum - fav_momentum
        mom_signal = 1.0 / (1.0 + math.exp(-mom_diff / 3.0))

        # Experience signal: underdog has more experienced roster
        dog_exp = getattr(dog_features, 'avg_experience', 2.0)
        fav_exp = getattr(fav_features, 'avg_experience', 2.0)
        dog_depth = getattr(dog_features, 'bench_depth_score', 0.0)
        fav_depth = getattr(fav_features, 'bench_depth_score', 0.0)
        exp_diff = (dog_exp - fav_exp) + 0.3 * (dog_depth - fav_depth)
        exp_signal = 1.0 / (1.0 + math.exp(-exp_diff / 0.8))

        # Efficiency signal: underdog overperforms its seed expectation
        _SEED_EXPECTED_EM = {
            1: 28, 2: 21, 3: 16, 4: 12, 5: 9, 6: 6, 7: 4, 8: 2,
            9: 0, 10: -2, 11: -4, 12: -6, 13: -9, 14: -12, 15: -16, 16: -21,
        }
        dog_em = getattr(dog_features, 'adj_efficiency_margin', 0.0)
        fav_em = getattr(fav_features, 'adj_efficiency_margin', 0.0)
        dog_residual = dog_em - _SEED_EXPECTED_EM.get(dog_seed, 0)
        fav_residual = fav_em - _SEED_EXPECTED_EM.get(fav_seed, 0)
        # Positive = underdog is better than its seed implies relative to favorite
        eff_diff = dog_residual - fav_residual
        eff_signal = 1.0 / (1.0 + math.exp(-eff_diff / 5.0))

        return (
            float(np.clip(vol_signal, 0.0, 1.0)),
            float(np.clip(mom_signal, 0.0, 1.0)),
            float(np.clip(exp_signal, 0.0, 1.0)),
            float(np.clip(eff_signal, 0.0, 1.0)),
        )

    def _compute_confidence(
        self,
        model_upset_prob: float,
        hist_rate: float,
        amplifier: float,
        n_obs: int,
    ) -> float:
        """Compute confidence in the upset assessment.

        Higher confidence when:
        - Model and historical prior agree
        - Amplifier signals are decisive (far from 0.5)
        - More historical data available
        """
        # Agreement between model and prior (0 = disagree, 1 = agree)
        agreement = 1.0 - min(abs(model_upset_prob - hist_rate) * 2.0, 1.0)

        # Amplifier decisiveness (how far from neutral 0.5)
        decisiveness = abs(amplifier - 0.5) * 2.0

        # Data support (log-scaled)
        data_support = min(math.log1p(n_obs) / math.log1p(156), 1.0)

        confidence = 0.40 * agreement + 0.30 * decisiveness + 0.30 * data_support
        return float(np.clip(confidence, 0.1, 0.95))

    def _compute_upset_score(
        self,
        upset_prob: float,
        amplifier: float,
        confidence: float,
        seed_gap: int,
    ) -> float:
        """Composite upset score for ranking matchups by upset potential.

        Higher score = stronger upset candidate. Accounts for:
        - Raw upset probability (primary signal)
        - Amplifier strength (team-specific edge)
        - Confidence (signal reliability)
        - Seed gap (bigger upsets are more impactful for brackets)
        """
        # Impact multiplier: bigger seed gap upsets are worth more in brackets
        impact = min(seed_gap / 8.0, 1.0)

        score = (
            0.50 * upset_prob
            + 0.20 * amplifier
            + 0.15 * confidence
            + 0.15 * impact * upset_prob
        )
        return float(np.clip(score, 0.0, 1.0))

    def _same_seed_signal(
        self,
        team1_id: str,
        team2_id: str,
        seed1: int,
        seed2: int,
        model_prob: float,
    ) -> UpsetSignal:
        """Handle same-seed matchups (no meaningful upset direction)."""
        return UpsetSignal(
            team1_id=team1_id,
            team2_id=team2_id,
            seed1=seed1,
            seed2=seed2,
            underdog_id=team2_id if model_prob > 0.5 else team1_id,
            favorite_id=team1_id if model_prob > 0.5 else team2_id,
            upset_prob=min(model_prob, 1.0 - model_prob),
            historical_prior=0.5,
            model_upset_prob=min(model_prob, 1.0 - model_prob),
            adjusted_prob=model_prob,
            risk_tier=_classify_risk_tier(min(model_prob, 1.0 - model_prob)),
            upset_score=0.0,
            confidence=0.3,
        )

    def _quick_adjust(
        self, model_prob: float, seed1: int, seed2: int,
    ) -> float:
        """Fast probability adjustment for fitting (no feature signals)."""
        if seed1 == seed2:
            return model_prob

        if seed1 < seed2:
            fav_seed, dog_seed = seed1, seed2
            model_upset = 1.0 - model_prob
        else:
            fav_seed, dog_seed = seed2, seed1
            model_upset = model_prob

        hist_rate = _get_historical_upset_rate(fav_seed, dog_seed)
        if hist_rate is None:
            hist_rate = _logistic_upset_fallback(fav_seed, dog_seed)

        n_obs = MATCHUP_SAMPLE_SIZES.get((fav_seed, dog_seed), 20)
        alpha = self.prior_strength * math.log1p(n_obs) / math.log1p(156)
        alpha = min(alpha, 0.40)

        posterior_upset = (1.0 - alpha) * model_upset + alpha * hist_rate

        if seed1 < seed2:
            return float(np.clip(1.0 - posterior_upset, 0.003, 0.997))
        else:
            return float(np.clip(posterior_upset, 0.003, 0.997))
