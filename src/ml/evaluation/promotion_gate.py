"""Promotion gate ensuring new models beat the incumbent.

Implements Agent Directive V7 S14/S15: a candidate model must demonstrate
statistically meaningful improvement over the current best before being
promoted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class PromotionResult:
    """Result of a promotion gate check."""

    approved: bool
    candidate_brier: float
    incumbent_brier: Optional[float]
    delta: float  # candidate - incumbent (negative = improvement)
    reason: str


class PromotionGate:
    """Gate that blocks model promotion unless it beats the incumbent.

    Parameters
    ----------
    min_improvement : float
        Minimum Brier score improvement required (default 0.001).
        Candidate Brier must be at least ``min_improvement`` lower
        than the incumbent.
    """

    def __init__(self, min_improvement: float = 0.001) -> None:
        self.min_improvement = min_improvement

    def check(
        self,
        candidate_brier: float,
        registry: Any,
    ) -> PromotionResult:
        """Check whether a candidate model should be promoted.

        Parameters
        ----------
        candidate_brier : float
            LOYO mean Brier score of the candidate model.
        registry : ExperimentRegistry
            Experiment registry to query for the incumbent.

        Returns
        -------
        PromotionResult
            Approval decision with justification.
        """
        incumbent = registry.best(metric="loyo_mean_brier")

        if incumbent is None:
            return PromotionResult(
                approved=True,
                candidate_brier=candidate_brier,
                incumbent_brier=None,
                delta=0.0,
                reason="No incumbent model found — candidate auto-promoted.",
            )

        incumbent_brier = incumbent.loyo_mean_brier
        delta = candidate_brier - incumbent_brier

        if delta < -self.min_improvement:
            return PromotionResult(
                approved=True,
                candidate_brier=candidate_brier,
                incumbent_brier=incumbent_brier,
                delta=delta,
                reason=(
                    f"Candidate Brier {candidate_brier:.6f} beats incumbent "
                    f"{incumbent_brier:.6f} by {abs(delta):.6f} "
                    f"(threshold: {self.min_improvement:.4f})."
                ),
            )

        return PromotionResult(
            approved=False,
            candidate_brier=candidate_brier,
            incumbent_brier=incumbent_brier,
            delta=delta,
            reason=(
                f"Candidate Brier {candidate_brier:.6f} does not beat incumbent "
                f"{incumbent_brier:.6f} by required margin {self.min_improvement:.4f} "
                f"(actual delta: {delta:+.6f})."
            ),
        )
