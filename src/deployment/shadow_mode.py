"""Shadow mode deployment: run two models side-by-side and compare."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ShadowResult:
    """Result of comparing primary and shadow model predictions."""

    primary_brier: float
    shadow_brier: float
    delta: float  # shadow - primary (negative = shadow is better)
    per_game_diffs: List[float]  # per-matchup probability differences
    max_divergence: float
    correlation: float
    recommendation: str  # "promote_shadow", "keep_primary", "needs_review"


class ShadowRunner:
    """Run primary and shadow models side-by-side and compare predictions."""

    PROMOTE_THRESHOLD = 0.002  # shadow must be this much better (lower Brier)
    MAX_DIVERGENCE_THRESHOLD = 0.15

    def __init__(self, primary_config: Any = None, shadow_config: Any = None) -> None:
        self.primary_config = primary_config
        self.shadow_config = shadow_config

    def run(
        self,
        predictions_primary: Dict[str, float],
        predictions_shadow: Dict[str, float],
        outcomes: Dict[str, float],
    ) -> ShadowResult:
        """Compare two sets of predictions against actual outcomes.

        Args:
            predictions_primary: Mapping of game_id -> predicted probability (primary).
            predictions_shadow: Mapping of game_id -> predicted probability (shadow).
            outcomes: Mapping of game_id -> actual outcome (0 or 1).

        Returns:
            ShadowResult with comparison metrics and recommendation.
        """
        common_keys = sorted(
            set(predictions_primary) & set(predictions_shadow) & set(outcomes)
        )
        if not common_keys:
            return ShadowResult(
                primary_brier=float("nan"),
                shadow_brier=float("nan"),
                delta=0.0,
                per_game_diffs=[],
                max_divergence=0.0,
                correlation=0.0,
                recommendation="needs_review",
            )

        primary_probs = np.array([predictions_primary[k] for k in common_keys])
        shadow_probs = np.array([predictions_shadow[k] for k in common_keys])
        actuals = np.array([outcomes[k] for k in common_keys])

        # Brier scores
        primary_brier = float(np.mean((primary_probs - actuals) ** 2))
        shadow_brier = float(np.mean((shadow_probs - actuals) ** 2))
        delta = shadow_brier - primary_brier  # negative means shadow is better

        # Per-game probability differences (shadow - primary)
        per_game_diffs = (shadow_probs - primary_probs).tolist()

        # Max divergence
        max_divergence = float(np.max(np.abs(shadow_probs - primary_probs)))

        # Correlation between the two sets of predictions
        if len(common_keys) > 1 and np.std(primary_probs) > 0 and np.std(shadow_probs) > 0:
            correlation = float(np.corrcoef(primary_probs, shadow_probs)[0, 1])
        else:
            correlation = 1.0

        # Recommendation logic
        if (
            delta < -self.PROMOTE_THRESHOLD
            and max_divergence < self.MAX_DIVERGENCE_THRESHOLD
        ):
            recommendation = "promote_shadow"
        elif delta >= -0.001:
            # Shadow is not meaningfully better (delta >= -0.001 means shadow Brier
            # is within 0.001 of primary or worse)
            recommendation = "keep_primary"
        else:
            recommendation = "needs_review"

        return ShadowResult(
            primary_brier=primary_brier,
            shadow_brier=shadow_brier,
            delta=delta,
            per_game_diffs=per_game_diffs,
            max_divergence=max_divergence,
            correlation=correlation,
            recommendation=recommendation,
        )

    def compare_configs(
        self, primary_config: Any, shadow_config: Any
    ) -> Dict[str, Any]:
        """Return a diff of config fields that differ between primary and shadow.

        Works with dataclasses and dicts. For other types, returns a simple comparison.
        """
        diffs: Dict[str, Any] = {}

        # Convert to dicts if dataclasses
        if hasattr(primary_config, "__dataclass_fields__"):
            primary_dict = {
                k: getattr(primary_config, k)
                for k in primary_config.__dataclass_fields__
            }
        elif isinstance(primary_config, dict):
            primary_dict = primary_config
        else:
            return {"primary": primary_config, "shadow": shadow_config}

        if hasattr(shadow_config, "__dataclass_fields__"):
            shadow_dict = {
                k: getattr(shadow_config, k)
                for k in shadow_config.__dataclass_fields__
            }
        elif isinstance(shadow_config, dict):
            shadow_dict = shadow_config
        else:
            return {"primary": primary_config, "shadow": shadow_config}

        all_keys = set(primary_dict) | set(shadow_dict)
        for key in sorted(all_keys):
            pval = primary_dict.get(key)
            sval = shadow_dict.get(key)
            if pval != sval:
                diffs[key] = {"primary": pval, "shadow": sval}

        return diffs
