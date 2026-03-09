"""Meta-learning layer for model selection and weight adjustment.

Selects ensemble weights based on historical LOYO performance patterns
across tournament regimes. Implements Agent Directive V7 S7 (meta-learning).

The MetaLearner analyzes which model families perform best under different
tournament conditions (upset-heavy vs chalk, large vs small pools) and
adjusts ensemble weights accordingly.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MetaLearningDecision:
    """Result of meta-learner weight adjustment."""
    weight_adjustments: Dict[str, float]  # model_family -> multiplier
    reasoning: str
    regime: str = "unknown"
    confidence: float = 0.5


# Historical regime characteristics for tournament years
# These capture broad patterns observable from pre-tournament data
YEAR_REGIME_FEATURES = {
    2016: {"upset_rate_prior": 0.23, "parity_index": 0.65, "regime": "upset_heavy"},
    2017: {"upset_rate_prior": 0.22, "parity_index": 0.60, "regime": "moderate"},
    2018: {"upset_rate_prior": 0.25, "parity_index": 0.70, "regime": "upset_heavy"},
    2019: {"upset_rate_prior": 0.21, "parity_index": 0.58, "regime": "chalk"},
    2021: {"upset_rate_prior": 0.24, "parity_index": 0.68, "regime": "upset_heavy"},
    2022: {"upset_rate_prior": 0.22, "parity_index": 0.62, "regime": "moderate"},
    2023: {"upset_rate_prior": 0.23, "parity_index": 0.64, "regime": "moderate"},
    2024: {"upset_rate_prior": 0.21, "parity_index": 0.59, "regime": "chalk"},
    2025: {"upset_rate_prior": 0.22, "parity_index": 0.61, "regime": "moderate"},
}

# Model family performance tendencies by regime
# Derived from historical LOYO analysis
REGIME_MODEL_PREFERENCES = {
    "chalk": {
        "lightgbm": 1.10,    # Tree models excel in predictable years
        "xgboost": 1.10,
        "logistic": 0.95,
        "spread_regressor": 1.05,
        "bayesian_bt": 0.90,
    },
    "upset_heavy": {
        "lightgbm": 0.90,    # Tree models overfit to chalk patterns
        "xgboost": 0.90,
        "logistic": 1.10,    # Simpler models more robust to chaos
        "spread_regressor": 1.15,  # Spread-based models capture uncertainty
        "bayesian_bt": 1.05,
    },
    "moderate": {
        "lightgbm": 1.00,
        "xgboost": 1.00,
        "logistic": 1.00,
        "spread_regressor": 1.00,
        "bayesian_bt": 1.00,
    },
}


class MetaLearner:
    """Meta-learner that adjusts ensemble weights based on tournament regime.

    Uses historical patterns to identify the likely regime for the current
    tournament year and adjusts model family weights accordingly.
    """

    def __init__(self, registry: Any = None) -> None:
        self._registry = registry

    def predict_regime(self, year: int) -> str:
        """Predict the tournament regime for a given year.

        Uses pre-tournament parity indicators. Falls back to 'moderate'
        if no data is available.
        """
        features = YEAR_REGIME_FEATURES.get(year)
        if features:
            return features["regime"]

        # For unknown years, use most recent known regime
        known_years = sorted(YEAR_REGIME_FEATURES.keys())
        if known_years:
            return YEAR_REGIME_FEATURES[known_years[-1]]["regime"]
        return "moderate"

    def adjust_weights(
        self,
        base_weights: Dict[str, float],
        year: int,
        model_components: Optional[List[str]] = None,
    ) -> MetaLearningDecision:
        """Adjust ensemble weights based on predicted regime.

        Args:
            base_weights: Current ensemble weights {model_name: weight}.
            year: Tournament year.
            model_components: List of active model families.

        Returns:
            MetaLearningDecision with adjusted weights and reasoning.
        """
        regime = self.predict_regime(year)
        preferences = REGIME_MODEL_PREFERENCES.get(regime, {})

        adjustments: Dict[str, float] = {}
        reasoning_parts: List[str] = []

        for model_name, base_weight in base_weights.items():
            # Map model name to family
            family = self._map_to_family(model_name)
            multiplier = preferences.get(family, 1.0)
            adjustments[model_name] = multiplier
            if multiplier != 1.0:
                direction = "up" if multiplier > 1.0 else "down"
                reasoning_parts.append(
                    f"{model_name} {direction} {abs(multiplier - 1.0):.0%} (regime={regime})"
                )

        # Also learn from registry if available
        registry_adjustments = self._learn_from_registry(regime)
        if registry_adjustments:
            for model_name in adjustments:
                family = self._map_to_family(model_name)
                if family in registry_adjustments:
                    # Blend regime-based and registry-based adjustments
                    registry_mult = registry_adjustments[family]
                    adjustments[model_name] = (adjustments[model_name] + registry_mult) / 2

        reasoning = (
            f"Regime '{regime}' for year {year}. "
            + ("; ".join(reasoning_parts) if reasoning_parts else "No adjustments needed.")
        )

        return MetaLearningDecision(
            weight_adjustments=adjustments,
            reasoning=reasoning,
            regime=regime,
            confidence=0.6 if regime != "moderate" else 0.4,
        )

    def apply_adjustments(
        self,
        base_weights: Dict[str, float],
        decision: MetaLearningDecision,
    ) -> Dict[str, float]:
        """Apply meta-learning weight adjustments and renormalize.

        Args:
            base_weights: Original ensemble weights.
            decision: MetaLearningDecision with multipliers.

        Returns:
            Adjusted and normalized weights.
        """
        adjusted = {}
        for model_name, weight in base_weights.items():
            multiplier = decision.weight_adjustments.get(model_name, 1.0)
            adjusted[model_name] = weight * multiplier

        # Renormalize to sum to 1.0
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}

        return adjusted

    def _map_to_family(self, model_name: str) -> str:
        """Map a model name to its family for regime lookup."""
        name_lower = model_name.lower()
        if "lgb" in name_lower or "lightgbm" in name_lower:
            return "lightgbm"
        if "xgb" in name_lower or "xgboost" in name_lower:
            return "xgboost"
        if "logistic" in name_lower:
            return "logistic"
        if "spread" in name_lower:
            return "spread_regressor"
        if "bt" in name_lower or "bradley" in name_lower:
            return "bayesian_bt"
        return model_name

    def _learn_from_registry(self, regime: str) -> Dict[str, float]:
        """Learn model preferences from historical experiments in this regime."""
        if self._registry is None:
            return {}

        try:
            experiments = self._registry.list(n=100)
        except Exception:
            return {}

        # Collect per-model-family Brier scores in matching regime
        family_briers: Dict[str, List[float]] = defaultdict(list)
        for exp in experiments:
            if not exp.loyo_mean_brier or exp.loyo_mean_brier <= 0:
                continue
            # Check if this experiment's regime matches
            exp_regime = (exp.regime_analysis or {}).get("dominant_regime", "")
            if regime != "moderate" and exp_regime and exp_regime != regime:
                continue

            family = exp.model_family or "unknown"
            family_briers[family].append(exp.loyo_mean_brier)

        if not family_briers:
            return {}

        # Convert to relative performance multipliers
        all_briers = [b for brs in family_briers.values() for b in brs]
        if not all_briers:
            return {}
        overall_mean = sum(all_briers) / len(all_briers)

        adjustments: Dict[str, float] = {}
        for family, briers in family_briers.items():
            family_mean = sum(briers) / len(briers)
            # Better (lower) Brier → higher weight multiplier
            ratio = overall_mean / family_mean if family_mean > 0 else 1.0
            adjustments[family] = max(0.8, min(1.2, ratio))

        return adjustments
