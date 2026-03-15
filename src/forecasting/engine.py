"""ForecastEngine — the decoupled, honest game-level prediction engine.

Architectural contract:
    - Produces calibrated win probabilities P(team1 beats team2).
    - Output is strictly INVARIANT to competitor count, point systems,
      crowd behavior, prize distributions, or any other downstream
      optimization parameter.
    - No method accepts or references optimization-layer parameters.
    - Brier Score gate rejects mediocre models (BS > threshold)
      in production mode.

This module extracts the prediction core from SOTAPipeline, enforcing
a structural firewall between forecasting and optimization.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..exceptions import IntegrityError

logger = logging.getLogger(__name__)

# ── Forbidden parameter names ────────────────────────────────────────
# If any of these appear in ForecastEngineConfig, construction fails.
# This is the structural firewall that prevents strategy leakage.
_FORBIDDEN_PARAMS = frozenset({
    "pool_size",
    "ev_pool_size",
    "ev_scoring_system",
    "ev_contrarian_strength",
    "ev_target_percentile",
    "ev_payout_structure",
    "ev_pool_type",
    "ev_enable_archetypes",
    "ev_archetype_overrides",
    "ev_enable_search",
    "ev_auto_refresh",
    "public_picks_json",
    "scoring_rules_json",
    "payout_structure",
    "contrarian_strength",
    "contrarian_override",
})


@dataclass
class ForecastEngineConfig:
    """Configuration for the ForecastEngine — forecast-only parameters.

    Explicitly excludes all pool/strategy/EV parameters.  Attempting to
    set any forbidden parameter raises ValueError at construction time.
    """

    # --- Core ---
    year: int = 2026
    random_seed: int = 2026
    probability_profile: str = "production"  # "production" | "experimental"
    mode: str = "production"  # "production" | "experimental" (controls Brier gate behavior)

    # --- Calibration ---
    calibration_method: str = "temperature"
    enable_tournament_adaptation: bool = True
    tournament_shrinkage: float = 0.06
    pre_calibration_clip_lo: float = 0.001
    pre_calibration_clip_hi: float = 0.999

    # --- Model complexity ---
    model_complexity: str = "simple"
    enable_spread_model: bool = True
    spread_sigma_init: float = 11.0

    # --- Massey composite ---
    massey_blend_weight: float = 0.25
    massey_sigma: float = 4.5

    # --- Brier gate ---
    brier_threshold: float = 0.20
    strict_brier_gate: bool = True  # Hard gate in production

    # --- Experimental layers (must be False in production profile) ---
    enable_seed_overrides: bool = False
    enable_brier_sharpening: bool = False
    enable_goto_conversion: bool = False
    seed_prior_weight: float = 0.0
    consistency_bonus_max: float = 0.0

    def __post_init__(self):
        self._enforce_firewall()

    def _enforce_firewall(self) -> None:
        """Reject any forbidden pool/strategy parameters."""
        for attr_name in dir(self):
            if attr_name in _FORBIDDEN_PARAMS:
                raise ValueError(
                    f"ForecastEngineConfig contains forbidden pool/strategy "
                    f"parameter '{attr_name}'. Forecast configuration must be "
                    f"invariant to downstream optimization context."
                )


@dataclass
class CalibrationReport:
    """Result of Brier Score validation against historical data.

    BS = (1/N) * sum((f_t - o_t)^2)

    If BS > threshold, model_quality is 'rejected' regardless of
    bracket-level success.
    """

    brier_score: float
    n_samples: int
    threshold: float
    passed: bool
    model_quality: str  # "accepted" | "rejected"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "brier_score": round(self.brier_score, 6),
            "n_samples": self.n_samples,
            "threshold": self.threshold,
            "passed": self.passed,
            "model_quality": self.model_quality,
            "timestamp": self.timestamp,
        }


class ForecastEngine:
    """Isolated game-level prediction engine.

    Produces calibrated win probabilities that are strictly invariant
    to downstream optimization context (number of competitors, point
    systems, crowd behavior, or prize distributions).

    Typical lifecycle::

        engine = ForecastEngine(config)
        engine.set_pipeline(trained_pipeline)  # attach trained model state
        p = engine.predict_proba("duke", "unc")  # pure probability

        # Or with a custom predict function:
        engine = ForecastEngine(config)
        engine.set_predict_fn(my_predict_fn)
        p = engine.predict_proba("duke", "unc")

    The engine does NOT train models itself — it wraps a trained pipeline
    or a custom predict function, enforcing the architectural boundary.
    """

    def __init__(self, config: Optional[ForecastEngineConfig] = None):
        self._config = config or ForecastEngineConfig()
        self._pipeline = None  # Reference to trained SOTAPipeline
        self._predict_fn: Optional[Callable[[str, str], float]] = None
        self._calibration_report: Optional[CalibrationReport] = None

    @property
    def config(self) -> ForecastEngineConfig:
        return self._config

    def set_pipeline(self, pipeline: Any) -> None:
        """Attach a trained SOTAPipeline for prediction delegation.

        The engine extracts only the prediction capability; no
        optimization state is accessible through the engine's public API.

        Args:
            pipeline: A trained SOTAPipeline instance with a functional
                ``predict_probability()`` method.
        """
        if not hasattr(pipeline, "predict_probability"):
            raise TypeError(
                "Pipeline must have a predict_probability(team1_id, team2_id) method."
            )
        self._pipeline = pipeline

    def set_predict_fn(self, fn: Callable[[str, str], float]) -> None:
        """Set a custom prediction function.

        Args:
            fn: Callable taking (team1_id, team2_id) and returning P(team1 wins).
                Must return values in [0, 1].
        """
        self._predict_fn = fn

    def predict_proba(self, team1_id: str, team2_id: str) -> float:
        """Return calibrated P(team1 beats team2).

        This method's signature is intentionally restricted to two team IDs.
        No optional parameters for downstream optimization context
        (competitors, points, picks, prizes) are accepted.

        Args:
            team1_id: First team identifier.
            team2_id: Second team identifier.

        Returns:
            Win probability for team1, in [clip_lo, clip_hi].

        Raises:
            RuntimeError: If no pipeline or predict function is set.
        """
        if self._predict_fn is not None:
            raw = self._predict_fn(team1_id, team2_id)
            return float(np.clip(
                raw,
                self._config.pre_calibration_clip_lo,
                self._config.pre_calibration_clip_hi,
            ))

        if self._pipeline is not None:
            return self._pipeline.predict_probability(team1_id, team2_id)

        raise RuntimeError(
            "ForecastEngine has no prediction source. Call set_pipeline() "
            "or set_predict_fn() before predict_proba()."
        )

    def predict_all_matchups(
        self,
        team_ids: List[str],
    ) -> Dict[Tuple[str, str], float]:
        """Compute P(t1 > t2) for all ordered pairs.

        Returns a deep-copied dict so downstream consumers cannot mutate
        the engine's probability state.

        Args:
            team_ids: List of team identifiers.

        Returns:
            Dict mapping (team1_id, team2_id) -> P(team1 wins) for all
            ordered pairs where team1 != team2.
        """
        probs: Dict[Tuple[str, str], float] = {}
        for i, t1 in enumerate(team_ids):
            for j, t2 in enumerate(team_ids):
                if i < j:
                    p = self.predict_proba(t1, t2)
                    probs[(t1, t2)] = p
                    probs[(t2, t1)] = 1.0 - p
        return copy.deepcopy(probs)

    def validate_calibration(
        self,
        forecasts: List[float],
        outcomes: List[int],
    ) -> CalibrationReport:
        """Evaluate forecast quality via Brier Score.

        BS = (1/N) * sum((f_t - o_t)^2)

        If BS > threshold:
            - production mode: raises IntegrityError (hard gate).
            - experimental mode: logs CRITICAL_WARNING (soft flag).

        Args:
            forecasts: Predicted probabilities [0, 1].
            outcomes: Actual outcomes (0 or 1).

        Returns:
            CalibrationReport with pass/fail status.

        Raises:
            IntegrityError: If BS > threshold in production mode.
            ValueError: If inputs are empty or mismatched length.
        """
        if len(forecasts) != len(outcomes):
            raise ValueError(
                f"Forecast/outcome length mismatch: {len(forecasts)} vs {len(outcomes)}"
            )
        if len(forecasts) == 0:
            raise ValueError("Cannot validate calibration with zero samples.")

        f = np.array(forecasts, dtype=np.float64)
        o = np.array(outcomes, dtype=np.float64)
        bs = float(np.mean((f - o) ** 2))

        passed = bs <= self._config.brier_threshold
        quality = "accepted" if passed else "rejected"

        report = CalibrationReport(
            brier_score=bs,
            n_samples=len(forecasts),
            threshold=self._config.brier_threshold,
            passed=passed,
            model_quality=quality,
        )
        self._calibration_report = report

        if not passed:
            msg = (
                f"Model Calibration Failure: BS={bs:.4f} exceeds "
                f"threshold {self._config.brier_threshold}. "
                f"Model is rejected as 'mediocre' regardless of bracket-level success."
            )
            if self._config.mode == "production" and self._config.strict_brier_gate:
                raise IntegrityError(msg)
            else:
                logger.warning("CRITICAL_WARNING: %s", msg)

        return report

    @property
    def calibration_report(self) -> Optional[CalibrationReport]:
        """Most recent calibration report, or None if not yet validated."""
        return self._calibration_report
