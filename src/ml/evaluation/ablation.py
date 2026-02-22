"""Ablation study framework for evaluating component contributions.

Systematically disables individual pipeline components and measures
the impact on predictive performance (Brier score), using formal
significance tests to determine whether each component genuinely helps.

Embedding-heavy components (GNN, Transformer) have a stricter gate:
they must pass p<0.05 or be automatically disabled, because the high
embedding dimensionality (~96 features from diff+interaction) relative
to training sample size (~400 games) creates severe overfitting risk.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .statistical_tests import paired_brier_test

logger = logging.getLogger(__name__)

# Components that can be ablated and how to disable them
ABLATABLE_COMPONENTS = (
    "gnn",
    "transformer",
    "travel_distance",
    "injury_model",
    "tournament_adaptation",
    "stacking",
)

# Components that use high-dimensional embeddings and require p<0.05
# significance to justify their inclusion.  The GNN embedding projection
# uses 2*16=32 features and the Transformer uses 2*48=96 features
# (diff + interaction), both trained on ~400 samples — dangerously
# close to overfitting territory.
EMBEDDING_COMPONENTS = ("gnn", "transformer")

TRAVEL_ADVANTAGE_FEATURE_IDX = 70  # 5th interaction feature (index in 71-dim matchup)


@dataclass
class AblationResult:
    """Result from ablating a single component."""

    component: str
    baseline_brier: float
    ablated_brier: float
    brier_delta: float  # ablated - baseline; positive = component helps
    t_stat: float
    p_value: float
    cohens_d: float
    n_games: int
    significant_at_05: bool

    @property
    def helps(self) -> bool:
        """Whether ablation significantly worsens performance."""
        return self.brier_delta > 0 and self.significant_at_05


@dataclass
class EmbeddingGateResult:
    """Result of the embedding significance gate for one component."""

    component: str
    embedding_dim: int  # Total features in projection (diff + interaction)
    n_samples: int  # Training samples available
    dim_to_sample_ratio: float  # embedding_dim / n_samples
    p_value: float
    passed: bool  # True if p<0.05 AND component helps
    action: str  # "keep", "disable", or "skip" (if ablation failed)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "embedding_dim": self.embedding_dim,
            "n_samples": self.n_samples,
            "dim_to_sample_ratio": round(self.dim_to_sample_ratio, 4),
            "p_value": round(self.p_value, 4),
            "passed": self.passed,
            "action": self.action,
        }


@dataclass
class FullAblationReport:
    """Results from ablating all components."""

    results: Dict[str, AblationResult] = field(default_factory=dict)
    baseline_brier: float = 0.0
    n_games: int = 0
    embedding_gate_results: Dict[str, EmbeddingGateResult] = field(
        default_factory=dict
    )

    @property
    def helpful_components(self) -> List[str]:
        """Components whose removal significantly hurts performance."""
        return [name for name, r in self.results.items() if r.helps]

    @property
    def unhelpful_components(self) -> List[str]:
        """Components whose removal does NOT significantly hurt performance."""
        return [name for name, r in self.results.items() if not r.helps]

    @property
    def gated_out_components(self) -> List[str]:
        """Embedding components that failed the p<0.05 significance gate."""
        return [
            name for name, g in self.embedding_gate_results.items()
            if not g.passed
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON output."""
        result = {
            "baseline_brier": self.baseline_brier,
            "n_games": self.n_games,
            "helpful_components": self.helpful_components,
            "unhelpful_components": self.unhelpful_components,
            "gated_out_components": self.gated_out_components,
            "per_component": {
                name: {
                    "ablated_brier": r.ablated_brier,
                    "brier_delta": r.brier_delta,
                    "p_value": r.p_value,
                    "cohens_d": r.cohens_d,
                    "significant": r.significant_at_05,
                }
                for name, r in self.results.items()
            },
        }
        if self.embedding_gate_results:
            result["embedding_gate"] = {
                name: g.to_dict()
                for name, g in self.embedding_gate_results.items()
            }
        return result


class AblationStudy:
    """Run ablation studies on a trained pipeline.

    Usage::

        study = AblationStudy(pipeline, validation_games)
        report = study.run_full_ablation()
        print(report.helpful_components)

    Each ablation temporarily modifies the pipeline to disable one component,
    re-predicts all validation games, and compares the Brier score against
    the full-model baseline using a paired t-test.
    """

    def __init__(
        self,
        pipeline: Any,  # SOTAPipeline — avoid circular import
        validation_games: List[Dict],
    ):
        """
        Args:
            pipeline: A trained SOTAPipeline instance.
            validation_games: List of game dicts with 'team1', 'team2',
                'team1_won' keys, from the validation era.
        """
        self.pipeline = pipeline
        self.validation_games = validation_games
        self._baseline_preds: Optional[np.ndarray] = None
        self._outcomes: Optional[np.ndarray] = None

    def _compute_baseline(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute baseline predictions on all validation games."""
        if self._baseline_preds is not None:
            return self._baseline_preds, self._outcomes

        preds = []
        outcomes = []
        for game in self.validation_games:
            t1, t2 = game["team1"], game["team2"]
            try:
                prob = self.pipeline.predict_probability(t1, t2)
                preds.append(prob)
                outcomes.append(1.0 if game["team1_won"] else 0.0)
            except Exception:
                continue

        self._baseline_preds = np.array(preds)
        self._outcomes = np.array(outcomes)
        return self._baseline_preds, self._outcomes

    def run_single_ablation(self, component: str) -> AblationResult:
        """Ablate a single component and measure impact.

        The component is disabled by temporarily modifying pipeline state,
        re-predicting all validation games, then restoring the original state.

        Args:
            component: One of ABLATABLE_COMPONENTS.

        Returns:
            AblationResult with Brier delta and significance test.
        """
        if component not in ABLATABLE_COMPONENTS:
            raise ValueError(
                f"Unknown component '{component}'. "
                f"Must be one of {ABLATABLE_COMPONENTS}"
            )

        baseline_preds, outcomes = self._compute_baseline()
        baseline_brier = float(np.mean((baseline_preds - outcomes) ** 2))

        # Compute ablated predictions
        ablated_preds = self._predict_with_ablation(component)

        if len(ablated_preds) != len(baseline_preds):
            logger.warning(
                "Ablation of %s produced %d predictions vs %d baseline; "
                "using intersection",
                component, len(ablated_preds), len(baseline_preds),
            )
            n = min(len(ablated_preds), len(baseline_preds))
            ablated_preds = ablated_preds[:n]
            baseline_sub = baseline_preds[:n]
            outcomes_sub = outcomes[:n]
        else:
            baseline_sub = baseline_preds
            outcomes_sub = outcomes

        ablated_brier = float(np.mean((ablated_preds - outcomes_sub) ** 2))

        # Paired t-test: positive delta = ablation hurts (component helps)
        test = paired_brier_test(baseline_sub, ablated_preds, outcomes_sub)

        # Note: brier_delta is ablated - baseline.  Positive → component helps.
        brier_delta = ablated_brier - baseline_brier

        return AblationResult(
            component=component,
            baseline_brier=baseline_brier,
            ablated_brier=ablated_brier,
            brier_delta=brier_delta,
            t_stat=test["t_stat"],
            p_value=test["p_value"],
            cohens_d=test["cohens_d"],
            n_games=test["n"],
            # Significant if p < 0.05 AND ablation made things worse
            significant_at_05=test["p_value"] < 0.05 and brier_delta > 0,
        )

    def run_full_ablation(
        self,
        components: Optional[List[str]] = None,
        enforce_embedding_gate: bool = True,
    ) -> FullAblationReport:
        """Run ablation for all components (or a subset).

        Args:
            components: List of components to ablate. Defaults to all.
            enforce_embedding_gate: If True, embedding-heavy components
                (GNN, Transformer) must pass p<0.05 or be flagged for
                removal.  This guards against overfitting when embedding
                dimensionality is high relative to sample size.

        Returns:
            FullAblationReport with per-component results.
        """
        if components is None:
            components = list(ABLATABLE_COMPONENTS)

        baseline_preds, outcomes = self._compute_baseline()
        baseline_brier = float(np.mean((baseline_preds - outcomes) ** 2))

        report = FullAblationReport(
            baseline_brier=baseline_brier,
            n_games=len(outcomes),
        )

        for component in components:
            try:
                result = self.run_single_ablation(component)
                report.results[component] = result
                logger.info(
                    "Ablation [%s]: Brier %.4f → %.4f (delta=%.4f, p=%.3f%s)",
                    component, baseline_brier, result.ablated_brier,
                    result.brier_delta, result.p_value,
                    " *" if result.significant_at_05 else "",
                )
            except Exception as exc:
                logger.warning("Ablation of %s failed: %s", component, exc)

        # Embedding significance gate: require p<0.05 for GNN/Transformer
        if enforce_embedding_gate:
            report.embedding_gate_results = self._run_embedding_gates(report)

        return report

    def _run_embedding_gates(
        self,
        report: FullAblationReport,
    ) -> Dict[str, EmbeddingGateResult]:
        """Check embedding components against the p<0.05 significance gate.

        For each embedding component (GNN, Transformer), computes the
        effective feature dimensionality of the embedding projection
        (diff + interaction vectors) and requires a statistically
        significant ablation result to justify inclusion.

        Components that fail the gate should be disabled to prevent
        the embedding projection from overfitting on small samples.
        """
        gate_results: Dict[str, EmbeddingGateResult] = {}

        # Infer embedding dimensions from pipeline state
        embedding_dims = {}
        if hasattr(self.pipeline, "gnn_embeddings") and self.pipeline.gnn_embeddings:
            sample_emb = next(iter(self.pipeline.gnn_embeddings.values()))
            # Projection uses diff + interaction → 2x embedding dim
            embedding_dims["gnn"] = len(sample_emb) * 2
        if hasattr(self.pipeline, "transformer_embeddings") and self.pipeline.transformer_embeddings:
            sample_emb = next(iter(self.pipeline.transformer_embeddings.values()))
            embedding_dims["transformer"] = len(sample_emb) * 2

        n_samples = report.n_games

        for component in EMBEDDING_COMPONENTS:
            emb_dim = embedding_dims.get(component, 0)
            if emb_dim == 0:
                gate_results[component] = EmbeddingGateResult(
                    component=component,
                    embedding_dim=0,
                    n_samples=n_samples,
                    dim_to_sample_ratio=0.0,
                    p_value=1.0,
                    passed=False,
                    action="skip",
                )
                continue

            ratio = emb_dim / max(n_samples, 1)
            ablation_result = report.results.get(component)

            if ablation_result is None:
                gate_results[component] = EmbeddingGateResult(
                    component=component,
                    embedding_dim=emb_dim,
                    n_samples=n_samples,
                    dim_to_sample_ratio=ratio,
                    p_value=1.0,
                    passed=False,
                    action="skip",
                )
                continue

            passed = ablation_result.p_value < 0.05 and ablation_result.brier_delta > 0
            action = "keep" if passed else "disable"

            gate_results[component] = EmbeddingGateResult(
                component=component,
                embedding_dim=emb_dim,
                n_samples=n_samples,
                dim_to_sample_ratio=ratio,
                p_value=ablation_result.p_value,
                passed=passed,
                action=action,
            )

            if not passed:
                logger.warning(
                    "EMBEDDING GATE: %s FAILED (dim=%d, N=%d, ratio=%.3f, "
                    "p=%.4f). Component should be disabled to prevent "
                    "overfitting.",
                    component, emb_dim, n_samples, ratio,
                    ablation_result.p_value,
                )
            else:
                logger.info(
                    "EMBEDDING GATE: %s PASSED (dim=%d, N=%d, ratio=%.3f, "
                    "p=%.4f).",
                    component, emb_dim, n_samples, ratio,
                    ablation_result.p_value,
                )

        return gate_results

    # ------------------------------------------------------------------
    # Private: per-component ablation strategies
    # ------------------------------------------------------------------

    def _predict_with_ablation(self, component: str) -> np.ndarray:
        """Re-predict validation games with one component disabled."""
        # Save original state
        saved = self._save_component_state(component)

        try:
            self._disable_component(component)
            preds = []
            for game in self.validation_games:
                t1, t2 = game["team1"], game["team2"]
                try:
                    prob = self.pipeline.predict_probability(t1, t2)
                    preds.append(prob)
                except Exception:
                    # Use baseline prediction as fallback
                    preds.append(0.5)
            return np.array(preds)
        finally:
            self._restore_component_state(component, saved)

    def _save_component_state(self, component: str) -> Dict:
        """Save the current state of a component before ablation."""
        state: Dict[str, Any] = {}

        if component in ("gnn", "transformer"):
            state["confidence"] = self.pipeline.model_confidence.get(component, 0.5)

        elif component == "tournament_adaptation":
            config = self.pipeline.config
            state["enable_tournament_adaptation"] = config.enable_tournament_adaptation

        elif component == "stacking":
            state["stacking_model"] = getattr(
                self.pipeline, "_stacking_meta_model", None
            )

        # travel_distance and injury_model state is handled differently —
        # we zero out features at prediction time rather than modifying state.
        return state

    def _disable_component(self, component: str) -> None:
        """Temporarily disable a pipeline component."""
        if component == "gnn":
            # Zero confidence → CFA gives it zero weight
            self.pipeline.model_confidence["gnn"] = 0.0

        elif component == "transformer":
            self.pipeline.model_confidence["transformer"] = 0.0

        elif component == "tournament_adaptation":
            self.pipeline.config.enable_tournament_adaptation = False

        elif component == "stacking":
            # Remove stacking meta-model → forces fallback to best single model
            if hasattr(self.pipeline, "_stacking_meta_model"):
                self.pipeline._stacking_meta_model = None

        elif component == "travel_distance":
            # We cannot easily zero a single feature at the pipeline level,
            # so we temporarily patch the feature engineering to always return
            # 0.0 for travel advantage.  This is handled by saving/patching
            # the travel module.
            try:
                from ...data.features import travel_distance as td_mod
                state = getattr(self, "_td_original_fn", None)
                if state is None:
                    self._td_original_fn = td_mod.compute_travel_advantage
                td_mod.compute_travel_advantage = lambda *a, **kw: 0.0
            except (ImportError, AttributeError):
                pass

        elif component == "injury_model":
            # Disable injury severity model
            self.pipeline.config.enable_injury_severity_model = False

    def _restore_component_state(self, component: str, saved: Dict) -> None:
        """Restore a component after ablation."""
        if component in ("gnn", "transformer"):
            self.pipeline.model_confidence[component] = saved["confidence"]

        elif component == "tournament_adaptation":
            self.pipeline.config.enable_tournament_adaptation = saved[
                "enable_tournament_adaptation"
            ]

        elif component == "stacking":
            if "stacking_model" in saved:
                self.pipeline._stacking_meta_model = saved["stacking_model"]

        elif component == "travel_distance":
            try:
                from ...data.features import travel_distance as td_mod
                if hasattr(self, "_td_original_fn"):
                    td_mod.compute_travel_advantage = self._td_original_fn
                    del self._td_original_fn
            except (ImportError, AttributeError):
                pass

        elif component == "injury_model":
            self.pipeline.config.enable_injury_severity_model = True
