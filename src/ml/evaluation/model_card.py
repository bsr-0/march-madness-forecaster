"""Model card generator for tournament prediction models.

Produces a structured model card following the Model Cards for Model
Reporting framework (Mitchell et al., 2019), adapted for Agent
Directive V7 S16 (final deliverables).

The model card includes:
  - Model details (architecture, training procedure, hyperparameters)
  - Intended use and limitations
  - Training data summary
  - Evaluation results (LOYO, calibration, robustness)
  - Ethical considerations and caveats
  - Quantitative analyses (per-round, per-seed-gap performance)

Usage:
    from src.ml.evaluation.model_card import ModelCardGenerator
    card = ModelCardGenerator.generate(pipeline_result)
    card.save("reports/model_card.json")
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelCard:
    """Structured model card for a tournament prediction model."""

    # --- Model Details ---
    model_name: str = "NCAA Tournament Prediction Ensemble"
    model_version: str = ""
    model_type: str = "Gradient-boosted ensemble with calibration"
    architecture: str = ""
    framework: str = "LightGBM + XGBoost + Logistic Regression"
    code_version: str = ""

    # --- Model Components ---
    components: List[Dict[str, Any]] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    calibration_method: str = ""

    # --- Training Data ---
    training_data: Dict[str, Any] = field(default_factory=dict)
    feature_count: int = 0
    feature_categories: Dict[str, int] = field(default_factory=dict)

    # --- Evaluation Results ---
    primary_metric: str = "brier_score"
    primary_metric_value: float = 0.0
    loyo_results: Dict[str, Any] = field(default_factory=dict)
    calibration_results: Dict[str, Any] = field(default_factory=dict)
    robustness_results: Dict[str, Any] = field(default_factory=dict)
    per_round_performance: Dict[str, float] = field(default_factory=dict)

    # --- Intended Use ---
    intended_use: str = (
        "Predict NCAA March Madness tournament game outcomes. "
        "Generate win probabilities for bracket construction and "
        "expected-value optimization in prediction contests."
    )
    out_of_scope: List[str] = field(default_factory=lambda: [
        "Real-money gambling decisions without additional risk management",
        "Predictions for non-Division I teams or non-tournament games",
        "Real-time in-game prediction",
    ])

    # --- Limitations ---
    limitations: List[str] = field(default_factory=lambda: [
        "Model is trained on historical data and may not capture current-season regime changes",
        "Injury/roster changes close to tournament may not be fully reflected",
        "Small sample size inherent in single-elimination tournament format",
        "Calibration is tuned for tournament games; regular-season calibration may differ",
    ])

    # --- Ethical Considerations ---
    ethical_considerations: List[str] = field(default_factory=lambda: [
        "Predictions should not be used as the sole basis for gambling decisions",
        "Model outputs are probabilistic estimates, not certainties",
        "Historical biases in team ratings may perpetuate systematic over/under-valuation",
    ])

    # --- Metadata ---
    created_at: str = ""
    dataset_version: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        """Save model card to JSON file."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Model card saved to %s", path)

    def summary(self) -> str:
        """Human-readable model card summary."""
        lines = [
            f"Model Card: {self.model_name}",
            f"  Version: {self.model_version or 'N/A'}",
            f"  Type: {self.model_type}",
            f"  Framework: {self.framework}",
            f"  Features: {self.feature_count}",
            f"  Primary metric: {self.primary_metric} = {self.primary_metric_value:.6f}",
            f"  Calibration: {self.calibration_method or 'N/A'}",
            f"  Code version: {self.code_version or 'N/A'}",
        ]
        if self.loyo_results:
            lines.append(
                f"  LOYO Brier: {self.loyo_results.get('mean_brier', 'N/A')}"
            )
        if self.robustness_results:
            gate = self.robustness_results.get("deployment_gate_passed", "N/A")
            lines.append(f"  Deployment gate: {'PASS' if gate else 'FAIL'}")
        return "\n".join(lines)


class ModelCardGenerator:
    """Generate model cards from pipeline results."""

    @staticmethod
    def generate(
        pipeline_result: Dict[str, Any],
        code_version: str = "",
        robustness_report: Optional[Dict[str, Any]] = None,
    ) -> ModelCard:
        """Generate a model card from pipeline output.

        Args:
            pipeline_result: Dictionary from SOTAPipeline.run().
            code_version: Git SHA or version tag.
            robustness_report: Optional robustness audit results.

        Returns:
            Populated ModelCard instance.
        """
        card = ModelCard(
            model_version=pipeline_result.get("year", ""),
            code_version=code_version,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        # Components from ensemble
        models = pipeline_result.get("model_components", [])
        if models:
            card.components = [{"name": m} for m in models]

        # Hyperparameters
        card.hyperparameters = pipeline_result.get("hyperparameters", {})
        card.calibration_method = pipeline_result.get("calibration_method", "")

        # Training data
        card.training_data = {
            "years": pipeline_result.get("dataset_version", ""),
            "n_games": pipeline_result.get("n_training_games", 0),
            "n_teams": pipeline_result.get("n_teams", 0),
        }
        card.feature_count = pipeline_result.get("n_features", 0)
        card.dataset_version = pipeline_result.get("dataset_version", "")

        # Evaluation
        loyo = pipeline_result.get("loyo_cv", {})
        if loyo:
            card.loyo_results = {
                "mean_brier": loyo.get("mean_brier", 0.0),
                "std_brier": loyo.get("std_brier", 0.0),
                "year_briers": loyo.get("year_briers", {}),
            }
            card.primary_metric_value = loyo.get("mean_brier", 0.0)

        # Calibration
        cal = pipeline_result.get("calibration", {})
        if cal:
            card.calibration_results = cal

        # Robustness
        if robustness_report:
            card.robustness_results = robustness_report

        # Per-round performance
        card.per_round_performance = pipeline_result.get(
            "per_round_brier", {}
        )

        # Architecture description
        card.architecture = (
            f"Ensemble of {len(card.components)} models "
            f"({', '.join(m.get('name', '?') for m in card.components[:5])}) "
            f"with {card.calibration_method or 'isotonic'} calibration"
        )

        return card


@dataclass
class DatasetCatalogEntry:
    """A single dataset entry in the catalog."""

    name: str
    description: str
    source: str
    path: str
    format: str = "json"
    refresh_frequency: str = ""
    fields: List[str] = field(default_factory=list)
    years_covered: str = ""
    record_count: int = 0
    sha256: str = ""


@dataclass
class DatasetCatalog:
    """Catalog of all datasets used in the pipeline.

    Implements Directive V7 S16 deliverable: dataset documentation.
    """

    entries: List[DatasetCatalogEntry] = field(default_factory=list)
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "created_at": self.created_at,
            "datasets": [asdict(e) for e in self.entries],
        }

    def save(self, path: str) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Dataset catalog saved to %s (%d entries)", path, len(self.entries))

    @classmethod
    def build_default(cls) -> DatasetCatalog:
        """Build catalog for the standard NCAA pipeline datasets."""
        catalog = cls(
            created_at=datetime.now(timezone.utc).isoformat(),
            entries=[
                DatasetCatalogEntry(
                    name="KenPom Advanced Metrics",
                    description="Adjusted efficiency, tempo, and four-factors data per team per season",
                    source="kenpom.com / barttorvik.com",
                    path="data/raw/advanced_metrics_{year}.json",
                    format="json",
                    refresh_frequency="Weekly during season",
                    fields=["adj_eff", "adj_tempo", "adj_off", "adj_def", "sos", "luck"],
                    years_covered="2008-2026",
                ),
                DatasetCatalogEntry(
                    name="Tournament Bracket",
                    description="Official NCAA tournament bracket with seeds, regions, and matchups",
                    source="NCAA / data entry",
                    path="data/raw/bracket_{year}.json",
                    format="json",
                    refresh_frequency="Once per tournament",
                    fields=["team", "seed", "region", "first_four"],
                    years_covered="2017-2026",
                ),
                DatasetCatalogEntry(
                    name="Historical Game Results",
                    description="Regular season and tournament game results with scores",
                    source="sports-reference.com",
                    path="data/raw/historical_games_{year}.json",
                    format="json",
                    refresh_frequency="Monthly",
                    fields=["date", "team", "opponent", "score", "opp_score", "location"],
                    years_covered="2008-2026",
                ),
                DatasetCatalogEntry(
                    name="Team Rosters",
                    description="Player rosters with height, class, and minutes data",
                    source="sports-reference.com",
                    path="data/raw/rosters_{year}.json",
                    format="json",
                    refresh_frequency="Monthly",
                    fields=["player", "height", "class", "minutes_pct"],
                    years_covered="2015-2026",
                ),
                DatasetCatalogEntry(
                    name="Massey Composite Ratings",
                    description="Composite rating from 100+ ranking systems",
                    source="masseyratings.com",
                    path="data/raw/massey_composite_{year}.json",
                    format="json",
                    refresh_frequency="Weekly during season",
                    fields=["team", "rank", "rating"],
                    years_covered="2016-2026",
                ),
                DatasetCatalogEntry(
                    name="Betting Market Lines",
                    description="Tournament game point spreads and moneylines",
                    source="Various sportsbooks",
                    path="data/raw/market_lines_{year}.json",
                    format="json",
                    refresh_frequency="Daily during tournament",
                    fields=["team_a", "team_b", "spread", "moneyline_a", "moneyline_b"],
                    years_covered="2019-2026",
                ),
            ],
        )
        return catalog


@dataclass
class FeatureCatalogEntry:
    """A single feature entry in the catalog."""

    name: str
    description: str
    category: str  # "efficiency", "tempo", "strength_of_schedule", "matchup", etc.
    source_dataset: str
    computation: str = ""  # How it's computed
    temporal_treatment: str = ""  # How PIT safety is maintained
    importance_rank: int = 0
    mean_importance: float = 0.0


@dataclass
class FeatureCatalog:
    """Catalog of all features used in the model.

    Implements Directive V7 S16 deliverable: feature documentation.
    """

    entries: List[FeatureCatalogEntry] = field(default_factory=list)
    created_at: str = ""
    total_features: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "created_at": self.created_at,
            "total_features": self.total_features,
            "features": [asdict(e) for e in self.entries],
            "by_category": self._by_category(),
        }

    def _by_category(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for e in self.entries:
            counts[e.category] = counts.get(e.category, 0) + 1
        return counts

    def save(self, path: str) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Feature catalog saved to %s (%d features)", path, len(self.entries))

    @classmethod
    def build_default(cls) -> FeatureCatalog:
        """Build catalog for the standard NCAA pipeline features."""
        entries = [
            FeatureCatalogEntry(
                name="adj_eff_margin",
                description="Adjusted efficiency margin (offense - defense)",
                category="efficiency",
                source_dataset="KenPom Advanced Metrics",
                computation="adj_off - adj_def",
                temporal_treatment="Season-to-date value as of game date",
            ),
            FeatureCatalogEntry(
                name="seed_diff",
                description="Seed number difference (team - opponent)",
                category="matchup",
                source_dataset="Tournament Bracket",
                computation="team_seed - opp_seed",
                temporal_treatment="Static per tournament",
            ),
            FeatureCatalogEntry(
                name="elo_rating",
                description="Elo rating based on chronological game results",
                category="rating",
                source_dataset="Historical Game Results",
                computation="Iterative Elo update with K=20, mean reversion",
                temporal_treatment="Point-in-time: only uses games before current date",
            ),
            FeatureCatalogEntry(
                name="massey_composite_rank",
                description="Composite ranking from 100+ systems",
                category="rating",
                source_dataset="Massey Composite Ratings",
                computation="Direct import",
                temporal_treatment="Latest pre-tournament snapshot",
            ),
            FeatureCatalogEntry(
                name="win_pct_last_10",
                description="Win percentage in last 10 games",
                category="momentum",
                source_dataset="Historical Game Results",
                computation="Wins / 10 for last 10 games before tournament",
                temporal_treatment="Point-in-time: expanding window up to game date",
            ),
            FeatureCatalogEntry(
                name="adj_tempo_diff",
                description="Adjusted tempo difference between teams",
                category="tempo",
                source_dataset="KenPom Advanced Metrics",
                computation="team_adj_tempo - opp_adj_tempo",
                temporal_treatment="Season-to-date value",
            ),
            FeatureCatalogEntry(
                name="strength_of_schedule",
                description="Opponent-adjusted strength of schedule metric",
                category="strength_of_schedule",
                source_dataset="Historical Game Results",
                computation="Average opponent rating weighted by recency",
                temporal_treatment="Point-in-time: uses games through current date",
            ),
            FeatureCatalogEntry(
                name="experience_score",
                description="Team experience based on roster class distribution",
                category="roster",
                source_dataset="Team Rosters",
                computation="Weighted sum of class years × minutes share",
                temporal_treatment="Pre-season snapshot, static during season",
            ),
        ]
        return cls(
            entries=entries,
            created_at=datetime.now(timezone.utc).isoformat(),
            total_features=len(entries),
        )
