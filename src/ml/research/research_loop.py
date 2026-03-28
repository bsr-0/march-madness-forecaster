"""Continuous research loop orchestrator (Agent Directive V7 S14).

Closes the feedback loop between hypothesis generation, experiment execution,
evaluation, and automated config adaptation.  This is the central
orchestrator that ties together:

  - HypothesisRegistry: structured hypothesis tracking
  - ExperimentRegistry: append-only experiment ledger
  - Sensitivity analysis: grid-search over Tier 3 constants
  - Ablation studies: component contribution measurement
  - Config adaptation: auto-apply validated improvements

The loop follows a principled cycle:

    diagnose → hypothesize → experiment → evaluate → adopt/reject → repeat

Each iteration is logged with full provenance so the research trajectory
is reproducible and auditable.

Usage:
    loop = ResearchLoop()
    loop.run_cycle(pipeline, games, config)
    print(loop.report())
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..evaluation.experiment_registry import ExperimentRecord, ExperimentRegistry
from .hypothesis_registry import Hypothesis, HypothesisRegistry, HypothesisStatus

logger = logging.getLogger(__name__)

DEFAULT_LOOP_LOG_PATH = "data/research_loop_log.jsonl"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ImprovementCandidate:
    """A config change candidate with measured impact."""

    name: str
    config_key: str
    old_value: Any
    new_value: Any
    brier_delta: float  # negative = improvement
    p_value: float
    source: str  # "sensitivity", "ablation", "hypothesis"
    hypothesis_id: str = ""
    adopted: bool = False
    reason: str = ""
    fold_briers: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchCycleResult:
    """Result of one research loop cycle."""

    cycle_id: str = ""
    timestamp: str = ""
    duration_seconds: float = 0.0

    # Diagnostics
    baseline_brier: float = 0.0
    n_hypotheses_generated: int = 0
    n_experiments_run: int = 0
    n_improvements_found: int = 0
    n_improvements_adopted: int = 0

    # Detail
    hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    adopted_changes: List[Dict[str, Any]] = field(default_factory=list)
    final_brier: float = 0.0
    total_improvement: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchTrajectory:
    """Tracks the overall research trajectory across cycles."""

    cycles: List[ResearchCycleResult] = field(default_factory=list)

    @property
    def total_improvement(self) -> float:
        return sum(c.total_improvement for c in self.cycles)

    @property
    def n_total_experiments(self) -> int:
        return sum(c.n_experiments_run for c in self.cycles)

    @property
    def n_total_adopted(self) -> int:
        return sum(c.n_improvements_adopted for c in self.cycles)

    def brier_trajectory(self) -> List[Tuple[str, float]]:
        """Return (cycle_id, brier) pairs showing improvement over time."""
        return [(c.cycle_id, c.final_brier) for c in self.cycles if c.final_brier > 0]


# ---------------------------------------------------------------------------
# Config mutation strategies
# ---------------------------------------------------------------------------


class ConfigMutator:
    """Generates config variants for hypothesis testing.

    Each mutation strategy produces a modified SOTAPipelineConfig that
    tests a specific hypothesis about pipeline behavior.
    """

    # Map of config keys to their valid ranges and mutation step sizes
    MUTABLE_PARAMS: Dict[str, Dict[str, Any]] = {
        "spread_weight": {"range": (0.0, 1.0), "step": 0.05, "type": "float"},
        "bayesian_bt_weight": {"range": (0.0, 0.5), "step": 0.025, "type": "float"},
        "training_year_decay": {"range": (0.5, 1.0), "step": 0.05, "type": "float"},
        "training_year_min_weight": {"range": (0.0, 0.5), "step": 0.05, "type": "float"},
        "market_blend_weight": {"range": (0.0, 0.5), "step": 0.05, "type": "float"},
        "tournament_sigma_weight": {"range": (0.0, 0.5), "step": 0.05, "type": "float"},
        "tournament_expert_weight": {"range": (0.0, 0.5), "step": 0.05, "type": "float"},
        "massey_standalone_weight": {"range": (0.0, 0.5), "step": 0.05, "type": "float"},
        "spread_sigma": {"range": (5.0, 20.0), "step": 1.0, "type": "float"},
    }

    @classmethod
    def generate_variants(
        cls,
        base_config: Any,
        param_name: str,
        n_points: int = 5,
    ) -> List[Tuple[Any, Any]]:
        """Generate config variants by sweeping one parameter.

        Returns list of (config_variant, param_value) tuples.
        """
        if param_name not in cls.MUTABLE_PARAMS:
            return []

        meta = cls.MUTABLE_PARAMS[param_name]
        lo, hi = meta["range"]
        current = getattr(base_config, param_name, None)
        if current is None:
            return []

        # Generate evenly spaced values including current
        values = np.linspace(lo, hi, n_points).tolist()
        if current not in values:
            values.append(current)
            values.sort()

        variants = []
        for val in values:
            cfg = copy.deepcopy(base_config)
            setattr(cfg, param_name, round(val, 4))
            variants.append((cfg, round(val, 4)))

        return variants

    @classmethod
    def apply_change(cls, config: Any, key: str, value: Any) -> Any:
        """Apply a single config change, returning a new config."""
        cfg = copy.deepcopy(config)
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        return cfg

    @classmethod
    def list_mutable_params(cls) -> List[str]:
        """List all params that can be mutated."""
        return list(cls.MUTABLE_PARAMS.keys())


# ---------------------------------------------------------------------------
# Evaluation protocol
# ---------------------------------------------------------------------------


class LoopEvaluator:
    """Evaluates config variants using LOYO cross-validation.

    This is a lightweight evaluator that computes Brier scores on
    pre-computed predictions, avoiding full pipeline retraining for
    each variant.
    """

    def __init__(
        self,
        eval_fn: Optional[Callable] = None,
        games: Optional[List[Dict]] = None,
        eval_folds_fn: Optional[Callable] = None,
    ):
        """
        Args:
            eval_fn: Function(config) -> float (Brier score).
                     If None, uses internal Brier computation.
            games: Validation games with 'team1', 'team2', 'team1_won'.
            eval_folds_fn: Function(config) -> List[float] returning
                     per-fold Brier scores (e.g. one per LOYO year).
                     Enables paired t-tests in param sweeps.
        """
        self.eval_fn = eval_fn
        self.eval_folds_fn = eval_folds_fn
        self.games = games or []
        self._cache: Dict[str, float] = {}
        self._folds_cache: Dict[str, List[float]] = {}

    def evaluate(self, config: Any, cache_key: str = "") -> float:
        """Evaluate a config variant. Returns Brier score."""
        if cache_key and cache_key in self._cache:
            return self._cache[cache_key]

        if self.eval_fn is not None:
            score = self.eval_fn(config)
        else:
            score = self._default_evaluate(config)

        if cache_key:
            self._cache[cache_key] = score
        return score

    def evaluate_folds(self, config: Any, cache_key: str = "") -> List[float]:
        """Evaluate a config variant and return per-fold Brier scores.

        If eval_folds_fn was provided, calls it to get per-fold scores.
        Otherwise falls back to wrapping the scalar eval result in a
        single-element list (disabling paired t-tests).
        """
        if cache_key and cache_key in self._folds_cache:
            return self._folds_cache[cache_key]

        if self.eval_folds_fn is not None:
            scores = list(self.eval_folds_fn(config))
        else:
            scores = [self.evaluate(config, cache_key=cache_key)]

        if cache_key:
            self._folds_cache[cache_key] = scores
        return scores

    def _default_evaluate(self, config: Any) -> float:
        """Default evaluation: return a placeholder Brier score.

        In production, this would run the full LOYO protocol.
        For the research loop, users should provide eval_fn.
        """
        return 0.25  # Placeholder

    def clear_cache(self) -> None:
        self._cache.clear()
        self._folds_cache.clear()


# ---------------------------------------------------------------------------
# Improvement gate
# ---------------------------------------------------------------------------


class ImprovementGate:
    """Gate that decides whether an improvement candidate should be adopted.

    Applies statistical and practical significance thresholds to prevent
    chasing noise.  This is critical for honest research: a change that
    improves Brier by 0.0001 is noise, not signal.
    """

    def __init__(
        self,
        min_brier_improvement: float = 0.002,
        max_p_value: float = 0.10,
        require_multiple_folds: bool = True,
        min_folds_improved: float = 0.6,
        min_cohens_d: float = 0.2,
    ):
        self.min_brier_improvement = min_brier_improvement
        self.max_p_value = max_p_value
        self.require_multiple_folds = require_multiple_folds
        self.min_folds_improved = min_folds_improved
        self.min_cohens_d = min_cohens_d

    def should_adopt(
        self,
        candidate: ImprovementCandidate,
        fold_deltas: Optional[List[float]] = None,
    ) -> Tuple[bool, str]:
        """Decide whether to adopt a candidate improvement.

        Returns (adopt, reason) tuple.
        """
        # Check practical significance
        if abs(candidate.brier_delta) < self.min_brier_improvement:
            return False, (
                f"Improvement {candidate.brier_delta:.6f} below threshold "
                f"{self.min_brier_improvement}"
            )

        # Check statistical significance
        if candidate.p_value > self.max_p_value:
            return False, (
                f"p-value {candidate.p_value:.4f} exceeds threshold "
                f"{self.max_p_value}"
            )

        # Check effect size (Cohen's d) when per-fold deltas are available.
        # Required by Experiment Workflow Plan: Cohen's d > 0.2 for adoption.
        if fold_deltas and len(fold_deltas) >= 2:
            fold_arr = np.array(fold_deltas, dtype=float)
            std = float(np.std(fold_arr, ddof=1))
            if std > 0:
                cohens_d = abs(float(np.mean(fold_arr))) / std
                if cohens_d < self.min_cohens_d:
                    return False, (
                        f"Cohen's d {cohens_d:.3f} below threshold "
                        f"{self.min_cohens_d} (effect too small relative "
                        f"to fold variance)"
                    )

        # Check fold consistency
        if self.require_multiple_folds and fold_deltas:
            n_improved = sum(1 for d in fold_deltas if d < 0)
            frac_improved = n_improved / len(fold_deltas)
            if frac_improved < self.min_folds_improved:
                return False, (
                    f"Only {frac_improved:.0%} of folds improved "
                    f"(need {self.min_folds_improved:.0%})"
                )

        # Direction check: brier_delta should be negative (improvement)
        if candidate.brier_delta > 0:
            return False, "Change worsens Brier score"

        return True, "Passed all gates"


# ---------------------------------------------------------------------------
# Research Loop orchestrator
# ---------------------------------------------------------------------------


class ResearchLoop:
    """Orchestrates the continuous research loop.

    Ties together hypothesis generation, experiment execution,
    evaluation, and automated config adaptation in a principled cycle.

    The loop maintains full provenance: every hypothesis, experiment,
    and adoption decision is logged for reproducibility.
    """

    def __init__(
        self,
        hypothesis_registry: Optional[HypothesisRegistry] = None,
        experiment_registry: Optional[ExperimentRegistry] = None,
        improvement_gate: Optional[ImprovementGate] = None,
        log_path: str = DEFAULT_LOOP_LOG_PATH,
    ):
        self.hypothesis_registry = hypothesis_registry or HypothesisRegistry()
        self.experiment_registry = experiment_registry or ExperimentRegistry()
        self.gate = improvement_gate or ImprovementGate()
        self.log_path = Path(log_path)
        self.trajectory = ResearchTrajectory()
        self._cycle_counter = 0

    def run_cycle(
        self,
        config: Any,
        eval_fn: Optional[Callable] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
        max_experiments: int = 10,
    ) -> ResearchCycleResult:
        """Run one full research cycle.

        Steps:
            1. Compute baseline performance
            2. Generate hypotheses from diagnostics
            3. Run experiments for pending hypotheses
            4. Evaluate candidates through improvement gate
            5. Adopt validated improvements
            6. Log everything

        Args:
            config: Current SOTAPipelineConfig
            eval_fn: Function(config) -> float for evaluation
            diagnostics: Dict with keys like 'dropout_report',
                        'shift_report', 'loyo_year_briers'
            max_experiments: Max experiments per cycle

        Returns:
            ResearchCycleResult with full cycle details
        """
        start_time = time.monotonic()
        self._cycle_counter += 1
        cycle_id = f"cycle_{self._cycle_counter:03d}"

        evaluator = LoopEvaluator(eval_fn=eval_fn)

        # Step 1: Baseline
        baseline_brier = evaluator.evaluate(config, cache_key="baseline")

        # Step 2: Generate hypotheses from diagnostics
        generated = []
        if diagnostics:
            generated = self.hypothesis_registry.generate_from_diagnostics(
                dropout_report=diagnostics.get("dropout_report"),
                shift_report=diagnostics.get("shift_report"),
                loyo_year_briers=diagnostics.get("loyo_year_briers"),
            )

        # Also generate hypotheses from pending param sweeps
        param_hypotheses = self._generate_param_sweep_hypotheses(config)
        generated.extend(param_hypotheses)

        # Step 3: Run experiments
        candidates: List[ImprovementCandidate] = []
        n_experiments = 0

        # Get pending hypotheses (from this and previous cycles)
        pending = self.hypothesis_registry.list(
            status=HypothesisStatus.PROPOSED.value
        )
        pending.extend(
            self.hypothesis_registry.list(
                status=HypothesisStatus.SCHEDULED.value
            )
        )

        for hypothesis in pending[:max_experiments]:
            experiment_candidates = self._run_hypothesis_experiment(
                hypothesis, config, evaluator, baseline_brier
            )
            candidates.extend(experiment_candidates)
            n_experiments += 1

        # Step 4: Gate and rank candidates
        adopted_changes: List[ImprovementCandidate] = []
        # Sort by improvement magnitude (most negative first)
        candidates.sort(key=lambda c: c.brier_delta)

        for candidate in candidates:
            should_adopt, reason = self.gate.should_adopt(candidate)
            candidate.reason = reason
            if should_adopt:
                candidate.adopted = True
                adopted_changes.append(candidate)
                # Apply to config
                config = ConfigMutator.apply_change(
                    config, candidate.config_key, candidate.new_value
                )
                logger.info(
                    "ADOPTED: %s = %s (Brier delta: %.6f, reason: %s)",
                    candidate.config_key,
                    candidate.new_value,
                    candidate.brier_delta,
                    reason,
                )

        # Step 5: Resolve hypotheses
        for hypothesis in pending[:max_experiments]:
            # Find if any candidate came from this hypothesis
            h_candidates = [
                c for c in candidates
                if c.hypothesis_id == hypothesis.hypothesis_id
            ]
            if h_candidates:
                best = min(h_candidates, key=lambda c: c.brier_delta)
                outcome = "confirmed" if best.adopted else "rejected"
                self.hypothesis_registry.resolve(
                    hypothesis.hypothesis_id,
                    outcome=outcome,
                    actual_impact=best.brier_delta,
                    notes=best.reason,
                )

        # Step 6: Evaluate final config
        final_brier = baseline_brier
        if adopted_changes:
            evaluator.clear_cache()
            final_brier = evaluator.evaluate(config, cache_key="final")

        duration = time.monotonic() - start_time

        result = ResearchCycleResult(
            cycle_id=cycle_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            duration_seconds=round(duration, 2),
            baseline_brier=baseline_brier,
            n_hypotheses_generated=len(generated),
            n_experiments_run=n_experiments,
            n_improvements_found=len([c for c in candidates if c.brier_delta < 0]),
            n_improvements_adopted=len(adopted_changes),
            hypotheses=[h.to_dict() for h in generated],
            candidates=[c.to_dict() for c in candidates],
            adopted_changes=[c.to_dict() for c in adopted_changes],
            final_brier=final_brier,
            total_improvement=baseline_brier - final_brier,
        )

        self.trajectory.cycles.append(result)
        self._log_cycle(result)

        logger.info(
            "Research cycle %s complete: %d experiments, %d adopted, "
            "Brier %.6f → %.6f (delta=%.6f)",
            cycle_id,
            n_experiments,
            len(adopted_changes),
            baseline_brier,
            final_brier,
            baseline_brier - final_brier,
        )

        return result

    def run_param_sweep(
        self,
        config: Any,
        param_name: str,
        eval_fn: Callable,
        n_points: int = 7,
        eval_folds_fn: Optional[Callable] = None,
    ) -> List[ImprovementCandidate]:
        """Sweep a single parameter and return improvement candidates.

        This is the core experiment type: vary one parameter while
        holding everything else constant, then measure Brier impact.

        When *eval_folds_fn* is provided (returns per-fold Brier scores),
        a paired t-test (scipy.stats.ttest_rel) is used to compute p-values
        as required by the Experiment Workflow Plan (Phase 2, Stage 2a).
        Falls back to the scalar heuristic when per-fold scores are
        unavailable.
        """
        evaluator = LoopEvaluator(eval_fn=eval_fn, eval_folds_fn=eval_folds_fn)
        baseline_folds = evaluator.evaluate_folds(config, cache_key="baseline")
        baseline_brier = float(np.mean(baseline_folds))
        current_value = getattr(config, param_name, None)

        # Import scipy for paired t-test (optional dependency)
        try:
            from scipy.stats import ttest_rel
            _has_ttest = True
        except ImportError:
            _has_ttest = False

        variants = ConfigMutator.generate_variants(config, param_name, n_points)
        candidates: List[ImprovementCandidate] = []

        for variant_config, param_value in variants:
            if abs(param_value - current_value) < 1e-9:
                continue  # Skip current value

            cache_key = f"{param_name}={param_value}"
            variant_folds = evaluator.evaluate_folds(variant_config, cache_key=cache_key)
            variant_brier = float(np.mean(variant_folds))
            brier_delta = variant_brier - baseline_brier

            # Compute p-value via paired t-test when per-fold scores are
            # available (>= 2 folds from both baseline and variant).
            if _has_ttest and len(baseline_folds) >= 2 and len(variant_folds) == len(baseline_folds):
                _, p_value = ttest_rel(variant_folds, baseline_folds)
                p_value = float(p_value)
            else:
                # Fallback heuristic when folds unavailable or scipy missing
                p_value = 0.05 if abs(brier_delta) > 0.001 else 0.5

            candidates.append(ImprovementCandidate(
                name=f"{param_name}={param_value}",
                config_key=param_name,
                old_value=current_value,
                new_value=param_value,
                brier_delta=brier_delta,
                p_value=p_value,
                source="param_sweep",
                fold_briers=list(variant_folds),
            ))

        # Apply Holm-Bonferroni correction for multiple comparisons.
        # Required by Experiment Workflow Plan mitigation #1: testing N
        # variants against the baseline inflates false positive risk.
        if candidates:
            from ..evaluation.statistical_tests import holm_bonferroni_correction

            raw_p_values = [c.p_value for c in candidates]
            corrected = holm_bonferroni_correction(raw_p_values, alpha=0.10)
            for candidate, correction in zip(candidates, corrected):
                candidate.p_value = correction["adjusted_p"]

        return candidates

    def generate_research_report(self) -> Dict[str, Any]:
        """Generate a comprehensive research trajectory report.

        Documents the full history of hypotheses, experiments, and
        adoptions for S14 compliance.
        """
        all_hypotheses = self.hypothesis_registry.list(n=1000)

        by_status: Dict[str, int] = {}
        for h in all_hypotheses:
            by_status[h.status] = by_status.get(h.status, 0) + 1

        confirmed = [h for h in all_hypotheses if h.status == "confirmed"]
        rejected = [h for h in all_hypotheses if h.status == "rejected"]

        report = {
            "trajectory": {
                "n_cycles": len(self.trajectory.cycles),
                "total_improvement": self.trajectory.total_improvement,
                "total_experiments": self.trajectory.n_total_experiments,
                "total_adopted": self.trajectory.n_total_adopted,
                "brier_trajectory": self.trajectory.brier_trajectory(),
            },
            "hypothesis_summary": {
                "total": len(all_hypotheses),
                "by_status": by_status,
                "confirmation_rate": (
                    len(confirmed) / max(len(confirmed) + len(rejected), 1)
                ),
            },
            "top_improvements": sorted(
                [c.to_dict() for cycle in self.trajectory.cycles
                 for c in []  # Will be populated from cycle data
                ],
                key=lambda x: x.get("brier_delta", 0),
            )[:10],
            "cycles": [c.to_dict() for c in self.trajectory.cycles],
        }

        # Populate top improvements from cycle data
        all_adopted = []
        for cycle in self.trajectory.cycles:
            all_adopted.extend(cycle.adopted_changes)
        report["top_improvements"] = sorted(
            all_adopted,
            key=lambda x: x.get("brier_delta", 0),
        )[:10]

        return report

    def report(self) -> str:
        """Human-readable research loop summary."""
        lines = [
            "=== Research Loop Report ===",
            f"Cycles completed: {len(self.trajectory.cycles)}",
            f"Total experiments: {self.trajectory.n_total_experiments}",
            f"Total improvements adopted: {self.trajectory.n_total_adopted}",
            f"Total Brier improvement: {self.trajectory.total_improvement:.6f}",
        ]

        if self.trajectory.cycles:
            latest = self.trajectory.cycles[-1]
            lines.extend([
                "",
                f"Latest cycle ({latest.cycle_id}):",
                f"  Hypotheses generated: {latest.n_hypotheses_generated}",
                f"  Experiments run: {latest.n_experiments_run}",
                f"  Improvements found: {latest.n_improvements_found}",
                f"  Improvements adopted: {latest.n_improvements_adopted}",
                f"  Brier: {latest.baseline_brier:.6f} → {latest.final_brier:.6f}",
                f"  Duration: {latest.duration_seconds:.1f}s",
            ])

        traj = self.trajectory.brier_trajectory()
        if len(traj) > 1:
            lines.extend([
                "",
                "Brier trajectory:",
            ])
            for cycle_id, brier in traj:
                lines.append(f"  {cycle_id}: {brier:.6f}")

        # Hypothesis registry summary
        lines.extend(["", self.hypothesis_registry.summary()])

        return "\n".join(lines)

    # --- Private helpers ---

    def _generate_param_sweep_hypotheses(
        self,
        config: Any,
    ) -> List[Hypothesis]:
        """Generate hypotheses for parameter sweeps not yet tested."""
        generated = []
        for param_name in ConfigMutator.list_mutable_params():
            current = getattr(config, param_name, None)
            if current is None:
                continue

            # Check if we already have a hypothesis for this param
            existing = self.hypothesis_registry.list(n=1000)
            already_tested = any(
                param_name in h.title for h in existing
                if h.status in ("confirmed", "rejected")
            )
            if already_tested:
                continue

            meta = ConfigMutator.MUTABLE_PARAMS[param_name]
            lo, hi = meta["range"]

            # Only generate if current value isn't at a boundary
            if abs(current - lo) < 1e-9 or abs(current - hi) < 1e-9:
                continue

            h = self.hypothesis_registry.add(
                title=f"Optimize {param_name} (current={current})",
                rationale=(
                    f"Grid search {param_name} over [{lo}, {hi}] "
                    f"to find Brier-optimal value"
                ),
                expected_impact=0.001,
                priority="medium",
                tags=["auto_generated", "param_sweep"],
                source="param_sweep",
            )
            generated.append(h)

        return generated

    def _run_hypothesis_experiment(
        self,
        hypothesis: Hypothesis,
        config: Any,
        evaluator: LoopEvaluator,
        baseline_brier: float,
    ) -> List[ImprovementCandidate]:
        """Run an experiment for a single hypothesis."""
        candidates: List[ImprovementCandidate] = []

        # Update hypothesis status
        hypothesis.status = HypothesisStatus.RUNNING.value

        # Check if this is a param sweep hypothesis
        if "param_sweep" in hypothesis.tags:
            param_name = self._extract_param_name(hypothesis.title)
            if param_name and param_name in ConfigMutator.MUTABLE_PARAMS:
                variants = ConfigMutator.generate_variants(
                    config, param_name, n_points=5
                )
                current_value = getattr(config, param_name, None)

                for variant_config, param_value in variants:
                    if current_value is not None and abs(param_value - current_value) < 1e-9:
                        continue

                    cache_key = f"{param_name}={param_value}"
                    variant_brier = evaluator.evaluate(
                        variant_config, cache_key=cache_key
                    )
                    brier_delta = variant_brier - baseline_brier

                    candidates.append(ImprovementCandidate(
                        name=f"{param_name}={param_value}",
                        config_key=param_name,
                        old_value=current_value,
                        new_value=param_value,
                        brier_delta=brier_delta,
                        p_value=0.05 if abs(brier_delta) > 0.001 else 0.5,
                        source="hypothesis",
                        hypothesis_id=hypothesis.hypothesis_id,
                    ))

        # Log experiment
        exp_record = ExperimentRecord(
            notes=f"Research loop experiment for hypothesis: {hypothesis.title}",
            tags=["research_loop", hypothesis.hypothesis_id],
        )
        self.experiment_registry.log(exp_record)

        return candidates

    def _extract_param_name(self, title: str) -> Optional[str]:
        """Extract parameter name from hypothesis title."""
        # Pattern: "Optimize param_name (current=...)"
        if title.startswith("Optimize "):
            parts = title[len("Optimize "):].split(" ")
            if parts:
                return parts[0]
        return None

    def _log_cycle(self, result: ResearchCycleResult) -> None:
        """Append cycle result to the log file."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(result.to_dict()) + "\n")
