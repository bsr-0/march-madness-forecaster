"""Experiment scheduler for automated config variant generation and execution.

Generates pipeline config variants by perturbing key parameters, queues them
for execution, and selects the best-performing variant based on LOYO Brier.

Implements Agent Directive V7 S14 (autonomous experiment scheduling).

Usage:
    from src.research.experiment_scheduler import ExperimentScheduler

    scheduler = ExperimentScheduler(registry)
    variants = scheduler.generate_variants(base_config, n=10)
    scheduler.queue_variants(variants)
    best = scheduler.select_best()
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_QUEUE_PATH = "data/research/experiment_queue.jsonl"


@dataclass
class ConfigVariant:
    """A candidate pipeline configuration variant."""

    variant_id: str = ""
    base_config_hash: str = ""
    timestamp: str = ""
    parameter_deltas: Dict[str, Any] = field(default_factory=dict)
    hypothesis: str = ""  # Why this variant might improve performance
    status: str = "queued"  # queued, running, completed, failed
    result_brier: Optional[float] = None
    result_experiment_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.variant_id:
            import uuid

            self.variant_id = str(uuid.uuid4())[:8]
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Parameter perturbation ranges for variant generation
DEFAULT_PARAM_RANGES: Dict[str, Dict[str, Any]] = {
    "lgb_weight": {"min": 0.05, "max": 0.40, "step": 0.05, "default": 0.15},
    "xgb_weight": {"min": 0.05, "max": 0.40, "step": 0.05, "default": 0.15},
    "spread_weight": {"min": 0.20, "max": 0.70, "step": 0.05, "default": 0.50},
    "logistic_weight": {"min": 0.05, "max": 0.35, "step": 0.05, "default": 0.20},
    "seed_prior_weight": {"min": 0.01, "max": 0.15, "step": 0.02, "default": 0.05},
    "temperature": {"min": 0.8, "max": 1.5, "step": 0.1, "default": 1.0},
    "mc_simulations": {"min": 10000, "max": 100000, "step": 10000, "default": 10000},
    "lgb_num_leaves": {"min": 8, "max": 64, "step": 8, "default": 31},
    "lgb_learning_rate": {"min": 0.01, "max": 0.3, "step": 0.02, "default": 0.1},
}


class ExperimentScheduler:
    """Generate and manage pipeline experiment variants.

    Analyzes past experiments to identify underexplored regions of the
    config space, generates targeted variants, and tracks their outcomes.
    """

    def __init__(
        self,
        registry: Any = None,  # ExperimentRegistry
        queue_path: str = DEFAULT_QUEUE_PATH,
        param_ranges: Optional[Dict[str, Dict[str, Any]]] = None,
        seed: int = 42,
    ) -> None:
        self._registry = registry
        self._queue_path = Path(queue_path)
        self._queue_path.parent.mkdir(parents=True, exist_ok=True)
        self._param_ranges = param_ranges or dict(DEFAULT_PARAM_RANGES)
        self._rng = random.Random(seed)

    def generate_variants(
        self,
        base_config: Dict[str, Any],
        n: int = 10,
        strategy: str = "perturbation",
    ) -> List[ConfigVariant]:
        """Generate N config variants from a base configuration.

        Args:
            base_config: Base pipeline configuration as a dict.
            n: Number of variants to generate.
            strategy: Generation strategy — 'perturbation' (random deltas),
                      'grid' (systematic sweep), or 'adaptive' (guided by history).

        Returns:
            List of ConfigVariant objects.
        """
        base_hash = hashlib.sha256(
            json.dumps(base_config, sort_keys=True).encode()
        ).hexdigest()[:12]

        if strategy == "grid":
            return self._generate_grid(base_config, base_hash, n)
        elif strategy == "adaptive":
            return self._generate_adaptive(base_config, base_hash, n)
        else:
            return self._generate_perturbation(base_config, base_hash, n)

    def _generate_perturbation(
        self,
        base_config: Dict[str, Any],
        base_hash: str,
        n: int,
    ) -> List[ConfigVariant]:
        """Generate variants by random perturbation of parameters."""
        variants: List[ConfigVariant] = []
        for _ in range(n):
            deltas: Dict[str, Any] = {}
            # Perturb 1-3 parameters per variant
            num_params = self._rng.randint(1, min(3, len(self._param_ranges)))
            params = self._rng.sample(list(self._param_ranges.keys()), num_params)

            for param in params:
                spec = self._param_ranges[param]
                step = spec["step"]
                direction = self._rng.choice([-1, 1])
                num_steps = self._rng.randint(1, 3)
                delta = direction * num_steps * step
                new_val = spec["default"] + delta
                new_val = max(spec["min"], min(spec["max"], new_val))

                # Round to avoid floating point noise
                if isinstance(step, float):
                    new_val = round(new_val, 4)
                else:
                    new_val = int(round(new_val))

                deltas[param] = new_val

            # Normalize ensemble weights if any were changed
            weight_params = ["lgb_weight", "xgb_weight", "spread_weight", "logistic_weight"]
            changed_weights = {k: v for k, v in deltas.items() if k in weight_params}
            if changed_weights:
                deltas = self._normalize_weights(deltas, weight_params)

            hypothesis = self._describe_hypothesis(deltas)
            variants.append(
                ConfigVariant(
                    base_config_hash=base_hash,
                    parameter_deltas=deltas,
                    hypothesis=hypothesis,
                )
            )

        return variants

    def _generate_grid(
        self,
        base_config: Dict[str, Any],
        base_hash: str,
        n: int,
    ) -> List[ConfigVariant]:
        """Generate variants via systematic parameter sweep."""
        variants: List[ConfigVariant] = []
        params = list(self._param_ranges.keys())

        for i in range(n):
            param = params[i % len(params)]
            spec = self._param_ranges[param]
            # Sweep from min to max
            num_steps = max(1, n // len(params))
            step_idx = i // len(params)
            val_range = spec["max"] - spec["min"]
            val = spec["min"] + (val_range * step_idx / max(1, num_steps - 1))

            if isinstance(spec["step"], float):
                val = round(val, 4)
            else:
                val = int(round(val))

            deltas = {param: val}
            hypothesis = f"Grid sweep: {param}={val}"
            variants.append(
                ConfigVariant(
                    base_config_hash=base_hash,
                    parameter_deltas=deltas,
                    hypothesis=hypothesis,
                )
            )

        return variants

    def _generate_adaptive(
        self,
        base_config: Dict[str, Any],
        base_hash: str,
        n: int,
    ) -> List[ConfigVariant]:
        """Generate variants guided by historical experiment performance.

        Identifies weak years/regimes and generates targeted variants.
        """
        weak_areas = self._identify_weak_areas()

        variants: List[ConfigVariant] = []
        for i in range(n):
            if weak_areas and i < len(weak_areas):
                area = weak_areas[i % len(weak_areas)]
                deltas = area.get("suggested_deltas", {})
                hypothesis = area.get("hypothesis", "Adaptive: targeting weak area")
            else:
                # Fall back to perturbation
                return variants + self._generate_perturbation(
                    base_config, base_hash, n - len(variants)
                )

            variants.append(
                ConfigVariant(
                    base_config_hash=base_hash,
                    parameter_deltas=deltas,
                    hypothesis=hypothesis,
                )
            )

        return variants

    def _identify_weak_areas(self) -> List[Dict[str, Any]]:
        """Analyze experiment history to identify areas for improvement."""
        if self._registry is None:
            return []

        best = self._registry.best()
        if best is None:
            return []

        weak_areas: List[Dict[str, Any]] = []

        # Check for weak years in LOYO
        if best.loyo_year_briers:
            mean_brier = sum(best.loyo_year_briers.values()) / len(
                best.loyo_year_briers
            )
            for year, brier in best.loyo_year_briers.items():
                if brier > mean_brier * 1.15:  # 15% worse than average
                    weak_areas.append(
                        {
                            "type": "weak_year",
                            "year": year,
                            "brier": brier,
                            "mean_brier": mean_brier,
                            "hypothesis": (
                                f"Year {year} underperforms (Brier={brier:.4f} vs "
                                f"mean={mean_brier:.4f}). Try adjusting seed prior "
                                f"or temperature."
                            ),
                            "suggested_deltas": {
                                "seed_prior_weight": round(
                                    self._rng.uniform(0.03, 0.12), 3
                                ),
                                "temperature": round(
                                    self._rng.uniform(0.9, 1.3), 2
                                ),
                            },
                        }
                    )

        # Check regime analysis
        if best.regime_analysis and "regimes" in best.regime_analysis:
            for regime_name, regime_data in best.regime_analysis.get(
                "regimes", {}
            ).items():
                if isinstance(regime_data, dict):
                    regime_brier = regime_data.get("mean_brier", 0)
                    if regime_brier > mean_brier * 1.1:
                        weak_areas.append(
                            {
                                "type": "weak_regime",
                                "regime": regime_name,
                                "brier": regime_brier,
                                "hypothesis": (
                                    f"Regime '{regime_name}' underperforms. "
                                    f"Try ensemble weight adjustment."
                                ),
                                "suggested_deltas": {
                                    "spread_weight": round(
                                        self._rng.uniform(0.30, 0.65), 3
                                    ),
                                },
                            }
                        )

        return weak_areas

    def queue_variants(self, variants: List[ConfigVariant]) -> int:
        """Add variants to the experiment queue.

        Returns number of variants queued (skips duplicates).
        """
        existing_ids = {v.variant_id for v in self._load_queue()}
        queued = 0

        with open(self._queue_path, "a") as f:
            for variant in variants:
                if variant.variant_id not in existing_ids:
                    f.write(json.dumps(variant.to_dict()) + "\n")
                    queued += 1

        logger.info("Queued %d experiment variants", queued)
        return queued

    def next_variant(self) -> Optional[ConfigVariant]:
        """Get the next queued variant for execution."""
        queue = self._load_queue()
        for variant in queue:
            if variant.status == "queued":
                return variant
        return None

    def mark_completed(
        self,
        variant_id: str,
        brier: float,
        experiment_id: str = "",
    ) -> None:
        """Mark a variant as completed with its result."""
        queue = self._load_queue()
        updated = False
        for variant in queue:
            if variant.variant_id == variant_id:
                variant.status = "completed"
                variant.result_brier = brier
                variant.result_experiment_id = experiment_id
                updated = True
                break

        if updated:
            self._save_queue(queue)

    def select_best(self) -> Optional[ConfigVariant]:
        """Select the best-performing completed variant."""
        queue = self._load_queue()
        completed = [
            v for v in queue if v.status == "completed" and v.result_brier is not None
        ]
        if not completed:
            return None
        return min(completed, key=lambda v: v.result_brier)  # type: ignore[arg-type]

    def summary(self) -> str:
        """Human-readable summary of the experiment queue."""
        queue = self._load_queue()
        if not queue:
            return "Experiment queue is empty."

        by_status: Dict[str, int] = {}
        for v in queue:
            by_status[v.status] = by_status.get(v.status, 0) + 1

        lines = [
            "Experiment Queue",
            "=" * 50,
            f"Total variants: {len(queue)}",
        ]
        for status, count in sorted(by_status.items()):
            lines.append(f"  {status}: {count}")

        best = self.select_best()
        if best:
            lines.append(
                f"\nBest result: Brier={best.result_brier:.4f} "
                f"(variant={best.variant_id})"
            )
            lines.append(f"  Hypothesis: {best.hypothesis}")

        return "\n".join(lines)

    def _normalize_weights(
        self, deltas: Dict[str, Any], weight_params: List[str]
    ) -> Dict[str, Any]:
        """Ensure ensemble weights sum to 1.0."""
        result = dict(deltas)
        weights = {}
        for p in weight_params:
            if p in result:
                weights[p] = result[p]
            else:
                weights[p] = self._param_ranges.get(p, {}).get("default", 0.25)

        total = sum(weights.values())
        if total > 0:
            for p in weights:
                result[p] = round(weights[p] / total, 4)
        return result

    @staticmethod
    def _describe_hypothesis(deltas: Dict[str, Any]) -> str:
        """Generate a human-readable hypothesis from parameter deltas."""
        parts = []
        for param, val in deltas.items():
            parts.append(f"{param}={val}")
        return f"Perturbation: {', '.join(parts)}"

    def _load_queue(self) -> List[ConfigVariant]:
        """Load the experiment queue from disk."""
        if not self._queue_path.exists():
            return []
        variants: List[ConfigVariant] = []
        with open(self._queue_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    variants.append(ConfigVariant(**data))
                except (json.JSONDecodeError, TypeError):
                    continue
        return variants

    def _save_queue(self, variants: List[ConfigVariant]) -> None:
        """Overwrite the queue file with updated variants."""
        with open(self._queue_path, "w") as f:
            for variant in variants:
                f.write(json.dumps(variant.to_dict()) + "\n")
