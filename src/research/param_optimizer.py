"""Experiment parameter optimization workflow.

Systematically searches over tunable production config parameters,
evaluates each via LOYO cross-validation, and produces a human-reviewable
config diff against the current production config.

Usage:
    from src.research.param_optimizer import ParamOptimizer

    optimizer = ParamOptimizer("configs/production_2026.json")
    report = optimizer.run()

CLI:
    python -m src.main optimize-params --dry-run
    python -m src.main optimize-params --params tournament_shrinkage massey_sigma
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Locked fields: must never be modified by optimization ──────────────────

LOCKED_FIELDS = frozenset({
    # Identity & mode
    "year", "probability_profile", "mode", "model_complexity",
    "pipeline_mode", "production_model_stack", "production_calibration",
    # Disabled modules
    "use_agent_orchestration", "enable_gnn", "enable_transformer",
    "enable_embedding_projections",
    # Year partitions
    "training_years", "dev_years", "holdout_years",
    # Leakage & governance
    "strict_leakage_mode", "require_freeze_file", "freeze_file",
    # Data paths
    "external_ratings_dir", "multi_year_games_dir", "kaggle_dir",
    "teams_json", "torvik_json", "historical_games_json",
    "roster_json", "public_picks_json", "scoring_rules_json",
    "mc_calibration_json", "sports_reference_json",
    "transfer_portal_json", "preseason_ap_json",
    "coach_tournament_json", "conf_champions_json",
    # Calibration method (locked in production)
    "calibration_method",
    # Seed
    "random_seed",
})

# ── Tunable parameter space ────────────────────────────────────────────────
# Union of ExperimentScheduler.DEFAULT_PARAM_RANGES and
# ConfigMutator.MUTABLE_PARAMS, filtered to SOTAPipelineConfig fields.

TUNABLE_PARAM_SPACE: Dict[str, Dict[str, Any]] = {
    # Pipeline-level parameters (from production config)
    "tournament_shrinkage": {
        "min": 0.02, "max": 0.15, "step": 0.01,
        "description": "Shrinkage toward 0.5 for tournament uncertainty",
    },
    "massey_blend_weight": {
        "min": 0.05, "max": 0.50, "step": 0.05,
        "description": "Weight for Massey rating blend",
    },
    "massey_sigma": {
        "min": 2.0, "max": 8.0, "step": 0.5,
        "description": "Spread parameter for Massey ratings",
    },
    "seed_prior_weight": {
        "min": 0.01, "max": 0.15, "step": 0.02,
        "description": "Seed-based prior regularization weight",
    },
    "consistency_bonus_max": {
        "min": 0.0, "max": 0.05, "step": 0.01,
        "description": "Max consistency bonus adjustment",
    },
    "mc_noise_std": {
        "min": 0.10, "max": 0.25, "step": 0.01,
        "description": "Monte Carlo simulation noise std",
    },
    "num_simulations": {
        "min": 10000, "max": 100000, "step": 10000,
        "description": "Number of Monte Carlo bracket simulations",
    },
    # Model training parameters (from ConfigMutator.MUTABLE_PARAMS)
    "training_year_decay": {
        "min": 0.5, "max": 1.0, "step": 0.05,
        "description": "Per-year weight decay for multi-year training",
    },
    "training_year_min_weight": {
        "min": 0.0, "max": 0.5, "step": 0.05,
        "description": "Floor weight for oldest training years",
    },
    "spread_sigma_init": {
        "min": 5.0, "max": 20.0, "step": 1.0,
        "description": "Initial spread sigma for point-spread model",
    },
}


# ── Dataclasses ────────────────────────────────────────────────────────────

@dataclass
class ParameterChange:
    """A recommended change to a single parameter."""

    name: str
    current_value: Any
    recommended_value: Any
    brier_impact: float  # Marginal Brier delta (negative = improvement)
    p_value: float
    confidence: str  # "high", "medium", "low"
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConfigDiffReport:
    """Structured diff between production and recommended config."""

    timestamp: str = ""
    production_config_path: str = ""
    production_config_hash: str = ""

    # Per-parameter changes
    parameter_changes: List[Dict[str, Any]] = field(default_factory=list)

    # Aggregate metrics
    baseline_loyo_brier: float = 0.0
    recommended_loyo_brier: float = 0.0
    brier_improvement: float = 0.0
    brier_improvement_pct: float = 0.0

    # Per-year breakdown
    per_year_briers: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Statistical validity
    p_value: float = 1.0
    cohens_d: float = 0.0
    folds_improved: str = ""

    # Audit trail
    n_configs_evaluated: int = 0
    total_wall_clock_seconds: float = 0.0
    experiment_ids: List[str] = field(default_factory=list)

    # Recommendation
    recommendation: str = "REJECT"  # "APPLY", "REVIEW", "REJECT"
    recommendation_rationale: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_json())
        logger.info("Config diff report saved to %s", path)


# ── ParamOptimizer ─────────────────────────────────────────────────────────

class ParamOptimizer:
    """Systematic parameter optimization with LOYO CV and config diff output.

    Workflow:
        1. Load production config as baseline
        2. Evaluate baseline via full pipeline LOYO
        3. Grid-sweep each tunable parameter independently
        4. Apply statistical gating per parameter
        5. Combine winning changes and re-evaluate
        6. Produce ConfigDiffReport with recommendation
    """

    def __init__(
        self,
        production_config_path: str = "configs/production_2026.json",
        output_dir: str = "data/optimization_reports",
        max_evaluations: int = 50,
        n_points: int = 5,
    ):
        self.production_config_path = production_config_path
        self.output_dir = output_dir
        self.max_evaluations = max_evaluations
        self.n_points = n_points
        self._eval_count = 0
        self._experiment_ids: List[str] = []

    def load_production_baseline(self) -> Any:
        """Load production config and build SOTAPipelineConfig."""
        from ..pipeline.config import SOTAPipelineConfig

        config_path = Path(self.production_config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Production config not found: {config_path}")

        with open(config_path) as f:
            raw = json.load(f)

        config = SOTAPipelineConfig()
        for key, val in raw.items():
            if hasattr(config, key):
                setattr(config, key, val)

        return config

    def production_config_hash(self) -> str:
        """SHA-256 hash of production config file."""
        config_path = Path(self.production_config_path)
        content = config_path.read_bytes()
        return hashlib.sha256(content).hexdigest()[:12]

    def build_search_space(
        self,
        params: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Return tunable parameters, filtered and validated.

        Args:
            params: If provided, restrict to these parameter names.

        Returns:
            Dict of param_name -> {min, max, step, description}.
        """
        from ..pipeline.config import SOTAPipelineConfig

        config = SOTAPipelineConfig()
        space = {}

        candidates = params or list(TUNABLE_PARAM_SPACE.keys())
        for name in candidates:
            if name in LOCKED_FIELDS:
                logger.warning("Skipping locked field: %s", name)
                continue
            if name not in TUNABLE_PARAM_SPACE:
                logger.warning("Unknown parameter: %s (not in TUNABLE_PARAM_SPACE)", name)
                continue
            if not hasattr(config, name):
                logger.warning("Parameter %s not on SOTAPipelineConfig, skipping", name)
                continue
            space[name] = TUNABLE_PARAM_SPACE[name]

        return space

    def evaluate_config(self, config: Any) -> float:
        """Run pipeline with config and return LOYO mean Brier.

        Returns float('inf') on failure.
        """
        if self._eval_count >= self.max_evaluations:
            logger.warning("Max evaluations (%d) reached", self.max_evaluations)
            return float("inf")

        self._eval_count += 1
        try:
            from ..pipeline.sota import SOTAPipeline

            pipeline = SOTAPipeline(config)
            result = pipeline.run()
            brier = result.get("loyo_mean_brier", 0)
            if brier <= 0:
                loyo = result.get("artifacts", {}).get("loyo_cv", {})
                brier = loyo.get("mean_brier", 0)
            return brier if brier > 0 else float("inf")
        except Exception as exc:
            logger.error("Pipeline evaluation failed: %s", exc)
            return float("inf")

    def evaluate_config_with_folds(
        self, config: Any,
    ) -> Tuple[float, Dict[int, float]]:
        """Run pipeline and return (mean_brier, {year: brier}).

        Returns (inf, {}) on failure.
        """
        if self._eval_count >= self.max_evaluations:
            logger.warning("Max evaluations (%d) reached", self.max_evaluations)
            return float("inf"), {}

        self._eval_count += 1
        try:
            from ..pipeline.sota import SOTAPipeline

            pipeline = SOTAPipeline(config)
            result = pipeline.run()

            loyo = result.get("artifacts", {}).get("loyo_cv", {})
            mean_brier = result.get("loyo_mean_brier", 0) or loyo.get("mean_brier", 0)
            year_briers = loyo.get("year_briers", {})

            if mean_brier <= 0 and year_briers:
                vals = [v for v in year_briers.values() if isinstance(v, (int, float)) and v > 0]
                mean_brier = sum(vals) / len(vals) if vals else 0

            if mean_brier <= 0:
                return float("inf"), {}

            return mean_brier, {int(k): float(v) for k, v in year_briers.items()}
        except Exception as exc:
            logger.error("Pipeline evaluation failed: %s", exc)
            return float("inf"), {}

    def _log_evaluation(
        self,
        config: Any,
        brier: float,
        tag: str,
        registry: Any,
    ) -> Optional[str]:
        """Log an evaluation to the experiment registry."""
        try:
            from ..ml.evaluation.experiment_registry import ExperimentRecord

            record = ExperimentRecord(
                config_hash=hashlib.sha256(
                    json.dumps(asdict(config) if hasattr(config, "__dataclass_fields__") else str(config),
                               sort_keys=True, default=str).encode()
                ).hexdigest()[:12],
                model_family="ensemble",
                validation_scheme="loyo_rolling_window",
                scoring_metric="brier",
                primary_metric_value=brier,
                loyo_mean_brier=brier,
                code_version="param_optimizer",
                dataset_version="2026",
                calibration_method="temperature",
                decision_policy="brier_minimization",
                tags=[tag, "param_optimization"],
            )
            exp_id = registry.log(record)
            self._experiment_ids.append(exp_id)
            return exp_id
        except Exception as exc:
            logger.warning("Failed to log experiment: %s", exc)
            return None

    def run_grid_phase(
        self,
        base_config: Any,
        search_space: Dict[str, Dict[str, Any]],
        baseline_brier: float,
        baseline_year_briers: Dict[int, float],
        registry: Any,
    ) -> List[ParameterChange]:
        """Sweep each parameter independently, return adopted changes."""
        import numpy as np

        adopted: List[ParameterChange] = []

        for param_name, spec in search_space.items():
            if self._eval_count >= self.max_evaluations:
                logger.info("Evaluation budget exhausted at param %s", param_name)
                break

            current_val = getattr(base_config, param_name, None)
            if current_val is None:
                continue

            lo, hi, step = spec["min"], spec["max"], spec["step"]
            values = np.arange(lo, hi + step / 2, step).tolist()
            values = [round(v, 6) for v in values]

            # Subsample if too many points
            if len(values) > self.n_points:
                indices = np.linspace(0, len(values) - 1, self.n_points, dtype=int)
                values = [values[i] for i in indices]

            # Ensure current value is included
            if current_val not in values:
                values.append(current_val)
                values.sort()

            logger.info("Sweeping %s: %d values in [%.4f, %.4f] (current=%.4f)",
                        param_name, len(values), lo, hi, current_val)

            best_brier = baseline_brier
            best_value = current_val
            best_year_briers: Dict[int, float] = {}

            for val in values:
                if self._eval_count >= self.max_evaluations:
                    break

                # Skip re-evaluating baseline value
                if val == current_val:
                    continue

                candidate_config = copy.deepcopy(base_config)
                setattr(candidate_config, param_name, val)

                brier, year_briers = self.evaluate_config_with_folds(candidate_config)
                self._log_evaluation(candidate_config, brier, f"grid_{param_name}", registry)

                if brier < best_brier:
                    best_brier = brier
                    best_value = val
                    best_year_briers = year_briers

            # Check if improvement is meaningful
            if best_value == current_val:
                logger.info("  %s: no improvement found", param_name)
                continue

            delta = best_brier - baseline_brier
            p_value = self._compute_fold_p_value(baseline_year_briers, best_year_briers)

            # Determine confidence level
            if p_value < 0.05 and abs(delta) >= 0.002:
                confidence = "high"
            elif p_value < 0.10 and abs(delta) >= 0.001:
                confidence = "medium"
            else:
                confidence = "low"

            change = ParameterChange(
                name=param_name,
                current_value=current_val,
                recommended_value=best_value,
                brier_impact=delta,
                p_value=p_value,
                confidence=confidence,
                rationale=(
                    f"Sweep found {param_name}={best_value} improves LOYO Brier "
                    f"by {abs(delta):.6f} (p={p_value:.4f})"
                ),
            )
            adopted.append(change)
            logger.info("  %s: %.4f -> %.4f (Brier delta=%.6f, p=%.4f, %s)",
                        param_name, current_val, best_value, delta, p_value, confidence)

        return adopted

    def _compute_fold_p_value(
        self,
        baseline_year_briers: Dict[int, float],
        candidate_year_briers: Dict[int, float],
    ) -> float:
        """Compute paired t-test p-value from per-year Brier scores."""
        import numpy as np

        common_years = sorted(set(baseline_year_briers) & set(candidate_year_briers))
        if len(common_years) < 3:
            return 1.0

        baseline_arr = np.array([baseline_year_briers[y] for y in common_years])
        candidate_arr = np.array([candidate_year_briers[y] for y in common_years])
        diffs = candidate_arr - baseline_arr  # negative = candidate better

        n = len(diffs)
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs, ddof=1))

        if std_diff < 1e-12:
            return 0.0 if abs(mean_diff) > 1e-12 else 1.0

        t_stat = mean_diff / (std_diff / np.sqrt(n))

        try:
            from scipy.stats import t as t_dist
            p_value = float(2 * t_dist.sf(abs(t_stat), df=n - 1))
        except ImportError:
            # Rough approximation without scipy
            p_value = min(1.0, 2.0 * np.exp(-0.5 * t_stat ** 2))

        return p_value

    def _compute_cohens_d(
        self,
        baseline_year_briers: Dict[int, float],
        candidate_year_briers: Dict[int, float],
    ) -> float:
        """Compute Cohen's d from per-year Brier scores."""
        import numpy as np

        common_years = sorted(set(baseline_year_briers) & set(candidate_year_briers))
        if len(common_years) < 2:
            return 0.0

        diffs = np.array([
            candidate_year_briers[y] - baseline_year_briers[y]
            for y in common_years
        ])
        std = float(np.std(diffs, ddof=1))
        if std < 1e-12:
            return 0.0
        return abs(float(np.mean(diffs))) / std

    def run_combined_evaluation(
        self,
        base_config: Any,
        changes: List[ParameterChange],
    ) -> Tuple[float, Dict[int, float]]:
        """Apply all adopted changes together and re-evaluate."""
        if not changes:
            return float("inf"), {}

        combined_config = copy.deepcopy(base_config)
        for change in changes:
            setattr(combined_config, change.name, change.recommended_value)

        return self.evaluate_config_with_folds(combined_config)

    def generate_config_diff(
        self,
        baseline_brier: float,
        baseline_year_briers: Dict[int, float],
        combined_brier: float,
        combined_year_briers: Dict[int, float],
        changes: List[ParameterChange],
        wall_clock_seconds: float,
    ) -> ConfigDiffReport:
        """Produce the final config diff report."""
        improvement = baseline_brier - combined_brier
        improvement_pct = (improvement / baseline_brier * 100) if baseline_brier > 0 else 0.0

        p_value = self._compute_fold_p_value(baseline_year_briers, combined_year_briers)
        cohens_d = self._compute_cohens_d(baseline_year_briers, combined_year_briers)

        # Count folds improved
        common_years = sorted(set(baseline_year_briers) & set(combined_year_briers))
        n_improved = sum(
            1 for y in common_years
            if combined_year_briers[y] < baseline_year_briers[y]
        )
        folds_improved = f"{n_improved}/{len(common_years)}" if common_years else "0/0"

        # Per-year breakdown
        per_year = {}
        for y in common_years:
            per_year[str(y)] = {
                "baseline": round(baseline_year_briers[y], 6),
                "recommended": round(combined_year_briers[y], 6),
                "delta": round(combined_year_briers[y] - baseline_year_briers[y], 6),
            }

        # Determine recommendation
        if p_value < 0.05 and cohens_d >= 0.3 and improvement >= 0.001:
            recommendation = "APPLY"
            rationale = (
                f"Combined changes improve LOYO Brier by {improvement:.4f} "
                f"({improvement_pct:.1f}%) with p={p_value:.4f} and "
                f"Cohen's d={cohens_d:.2f}. Improvement consistent across "
                f"{folds_improved} folds."
            )
        elif p_value < 0.10 and improvement > 0:
            recommendation = "REVIEW"
            rationale = (
                f"Marginal improvement of {improvement:.4f} ({improvement_pct:.1f}%) "
                f"with p={p_value:.4f}. Requires manual review before adoption."
            )
        else:
            recommendation = "REJECT"
            rationale = (
                f"Improvement {improvement:.4f} not statistically significant "
                f"(p={p_value:.4f}, Cohen's d={cohens_d:.2f})."
            )

        return ConfigDiffReport(
            production_config_path=self.production_config_path,
            production_config_hash=self.production_config_hash(),
            parameter_changes=[c.to_dict() for c in changes],
            baseline_loyo_brier=round(baseline_brier, 6),
            recommended_loyo_brier=round(combined_brier, 6),
            brier_improvement=round(improvement, 6),
            brier_improvement_pct=round(improvement_pct, 2),
            per_year_briers=per_year,
            p_value=round(p_value, 6),
            cohens_d=round(cohens_d, 4),
            folds_improved=folds_improved,
            n_configs_evaluated=self._eval_count,
            total_wall_clock_seconds=round(wall_clock_seconds, 1),
            experiment_ids=list(self._experiment_ids),
            recommendation=recommendation,
            recommendation_rationale=rationale,
        )

    def run(
        self,
        strategy: str = "full",
        params: Optional[List[str]] = None,
        dry_run: bool = False,
    ) -> ConfigDiffReport:
        """Run the full optimization workflow.

        Args:
            strategy: "grid-only" or "full" (grid + combined re-eval).
            params: Restrict optimization to these parameters.
            dry_run: Print search space and exit without evaluating.

        Returns:
            ConfigDiffReport with recommendation.
        """
        from ..ml.evaluation.experiment_registry import ExperimentRegistry
        from ..ml.evaluation.promotion_gate import PromotionGate

        start_time = time.time()

        # Step 1: Load baseline config
        base_config = self.load_production_baseline()
        search_space = self.build_search_space(params)

        if dry_run:
            return self._dry_run_report(base_config, search_space)

        logger.info("Starting parameter optimization: %d params, max %d evaluations",
                     len(search_space), self.max_evaluations)

        registry = ExperimentRegistry()

        # Step 2: Evaluate baseline
        logger.info("Evaluating baseline config...")
        baseline_brier, baseline_year_briers = self.evaluate_config_with_folds(base_config)
        self._log_evaluation(base_config, baseline_brier, "optimization_baseline", registry)
        logger.info("Baseline LOYO Brier: %.6f", baseline_brier)

        if baseline_brier == float("inf"):
            logger.error("Baseline evaluation failed, cannot proceed")
            return ConfigDiffReport(
                production_config_path=self.production_config_path,
                recommendation="REJECT",
                recommendation_rationale="Baseline evaluation failed.",
                total_wall_clock_seconds=time.time() - start_time,
            )

        # Step 3: Prioritize using KnowledgeStore if available
        try:
            from ..research.knowledge_store import KnowledgeStore
            knowledge = KnowledgeStore(registry=registry)
            unexplored = knowledge.unexplored_parameters()
            if unexplored:
                unexplored_names = {u["parameter"] for u in unexplored if "parameter" in u}
                # Move unexplored params to front
                ordered_space = {}
                for name in search_space:
                    if name in unexplored_names:
                        ordered_space[name] = search_space[name]
                for name in search_space:
                    if name not in unexplored_names:
                        ordered_space[name] = search_space[name]
                search_space = ordered_space
                logger.info("Prioritized %d unexplored parameters", len(unexplored_names & set(search_space)))
        except Exception as exc:
            logger.debug("KnowledgeStore unavailable: %s", exc)

        # Step 4: Grid sweep each parameter
        changes = self.run_grid_phase(
            base_config, search_space, baseline_brier,
            baseline_year_briers, registry,
        )

        if not changes:
            logger.info("No improvements found across any parameter")
            return ConfigDiffReport(
                production_config_path=self.production_config_path,
                production_config_hash=self.production_config_hash(),
                baseline_loyo_brier=round(baseline_brier, 6),
                recommended_loyo_brier=round(baseline_brier, 6),
                n_configs_evaluated=self._eval_count,
                total_wall_clock_seconds=time.time() - start_time,
                experiment_ids=list(self._experiment_ids),
                recommendation="REJECT",
                recommendation_rationale="No parameter changes improved LOYO Brier.",
            )

        # Filter to high/medium confidence changes
        strong_changes = [c for c in changes if c.confidence in ("high", "medium")]
        eval_changes = strong_changes if strong_changes else changes

        # Step 5: Combined re-evaluation
        if strategy == "full" and len(eval_changes) > 1:
            logger.info("Re-evaluating %d combined parameter changes...", len(eval_changes))
            combined_brier, combined_year_briers = self.run_combined_evaluation(
                base_config, eval_changes,
            )
            self._log_evaluation(
                self._apply_changes(base_config, eval_changes),
                combined_brier, "optimization_combined", registry,
            )

            # If combined is worse than baseline (interaction effects),
            # fall back to single best change
            if combined_brier >= baseline_brier:
                logger.warning("Combined changes interact negatively, falling back to best single change")
                eval_changes = sorted(changes, key=lambda c: c.brier_impact)[:1]
                combined_brier, combined_year_briers = self.run_combined_evaluation(
                    base_config, eval_changes,
                )
        elif eval_changes:
            combined_brier, combined_year_briers = self.run_combined_evaluation(
                base_config, eval_changes,
            )
        else:
            combined_brier = baseline_brier
            combined_year_briers = baseline_year_briers

        # Step 6: Promotion gate
        try:
            gate = PromotionGate()
            promotion = gate.check(combined_brier, registry)
            logger.info("Promotion gate: %s (%s)", "PASS" if promotion.approved else "FAIL", promotion.reason)
        except Exception as exc:
            logger.debug("Promotion gate check failed: %s", exc)

        # Step 7: Generate report
        report = self.generate_config_diff(
            baseline_brier, baseline_year_briers,
            combined_brier, combined_year_briers,
            eval_changes,
            wall_clock_seconds=time.time() - start_time,
        )

        # Save report
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f"optimization_report_{timestamp}.json")
        report.save(report_path)

        return report

    def _apply_changes(self, base_config: Any, changes: List[ParameterChange]) -> Any:
        """Apply changes to a config copy."""
        config = copy.deepcopy(base_config)
        for change in changes:
            setattr(config, change.name, change.recommended_value)
        return config

    def _dry_run_report(
        self,
        base_config: Any,
        search_space: Dict[str, Dict[str, Any]],
    ) -> ConfigDiffReport:
        """Print search space and return empty report."""
        print("\n=== Parameter Optimization Search Space (dry run) ===\n")
        print(f"Production config: {self.production_config_path}")
        print(f"Config hash: {self.production_config_hash()}")
        print(f"Max evaluations: {self.max_evaluations}")
        print(f"Grid points per param: {self.n_points}")
        print(f"\nTunable parameters ({len(search_space)}):\n")

        for name, spec in search_space.items():
            current = getattr(base_config, name, "N/A")
            print(f"  {name}:")
            print(f"    Current: {current}")
            print(f"    Range:   [{spec['min']}, {spec['max']}] step={spec['step']}")
            print(f"    Desc:    {spec['description']}")
            print()

        # Estimate evaluations
        total_evals = sum(
            min(
                int((spec["max"] - spec["min"]) / spec["step"]) + 1,
                self.n_points,
            )
            for spec in search_space.values()
        )
        print(f"Estimated evaluations: ~{total_evals} (grid) + 1 (baseline) + 1 (combined)")
        print(f"Budget: {self.max_evaluations} max evaluations\n")

        return ConfigDiffReport(
            production_config_path=self.production_config_path,
            production_config_hash=self.production_config_hash(),
            n_configs_evaluated=0,
            recommendation="REJECT",
            recommendation_rationale="Dry run — no evaluations performed.",
        )

    def print_summary(self, report: ConfigDiffReport) -> None:
        """Print human-readable summary of the optimization report."""
        print(f"\n{'=' * 60}")
        print("PARAMETER OPTIMIZATION REPORT")
        print(f"{'=' * 60}")
        print(f"Production config: {report.production_config_path}")
        print(f"Configs evaluated: {report.n_configs_evaluated}")
        print(f"Wall clock: {report.total_wall_clock_seconds:.0f}s")
        print()
        print(f"Baseline LOYO Brier:     {report.baseline_loyo_brier:.6f}")
        print(f"Recommended LOYO Brier:  {report.recommended_loyo_brier:.6f}")
        print(f"Improvement:             {report.brier_improvement:.6f} ({report.brier_improvement_pct:.1f}%)")
        print(f"p-value:                 {report.p_value:.4f}")
        print(f"Cohen's d:               {report.cohens_d:.3f}")
        print(f"Folds improved:          {report.folds_improved}")

        if report.parameter_changes:
            print(f"\nRecommended changes ({len(report.parameter_changes)}):")
            for change in report.parameter_changes:
                print(f"  {change['name']}: {change['current_value']} -> {change['recommended_value']}")
                print(f"    Brier impact: {change['brier_impact']:.6f}, p={change['p_value']:.4f} [{change['confidence']}]")

        if report.per_year_briers:
            print("\nPer-year breakdown:")
            for year, vals in sorted(report.per_year_briers.items()):
                marker = "+" if vals["delta"] > 0 else "-" if vals["delta"] < 0 else "="
                print(f"  {year}: {vals['baseline']:.4f} -> {vals['recommended']:.4f} ({marker}{abs(vals['delta']):.4f})")

        print(f"\nRecommendation: {report.recommendation}")
        print(f"  {report.recommendation_rationale}")
        print(f"{'=' * 60}\n")
