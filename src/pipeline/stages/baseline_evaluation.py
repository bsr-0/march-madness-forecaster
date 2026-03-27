"""Phase 5 — Baseline Models: train simple baselines, evaluate, gate on EV.

Trains Logistic Regression and Elo baselines against validated features
from Phase 4.  Evaluates both on Brier Score and Bracket EV (the primary
metric).  Stops the pipeline if any baseline fails to meet the minimal
EV threshold — a signal that features or data are broken.

Usage::

    evaluator = BaselineEvaluator(config)
    results = evaluator.run(
        train_X, train_y, val_X, val_y,
        feature_names=names,
        team_matchups=matchups,   # for EV calculation
    )
    # results.passed is True only if all baselines clear thresholds.

Output is a ``BaselineResults`` dict stored in a consistent format for
downstream hyperparameter tuning and ensembling (Phase 6+).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.exceptions import IntegrityError
from src.ml.calibration.calibration import BrierScoreOptimizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default thresholds
# ---------------------------------------------------------------------------

# Brier score must be below this (coin-flip is 0.25, good models < 0.19).
DEFAULT_BRIER_THRESHOLD = 0.250

# Bracket EV: minimum expected points under ESPN scoring.
# A coin-flip bracket scores ~100-110 pts on average over 63 games with
# ESPN scoring (10/20/40/80/160/320).  A viable baseline should exceed
# a coin-flip by a meaningful margin.
DEFAULT_MIN_BRACKET_EV = 0.0  # Non-negative EV over coin-flip baseline

# ESPN-standard round scoring weights for EV calculation.
ROUND_SCORING_WEIGHTS: Dict[str, int] = {
    "R64": 10,
    "R32": 20,
    "S16": 40,
    "E8": 80,
    "F4": 160,
    "CHAMP": 320,
}


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass
class SingleModelResult:
    """Evaluation result for a single baseline model."""

    model_name: str
    brier_score: float
    log_loss: float
    accuracy: float
    bracket_ev: float  # Expected bracket points (primary metric)
    predictions: np.ndarray  # Raw predicted probabilities on validation set
    passed_brier: bool = True
    passed_ev: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.passed_brier and self.passed_ev

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "EV": round(self.bracket_ev, 4),
            "Brier": round(self.brier_score, 6),
            "log_loss": round(self.log_loss, 6),
            "accuracy": round(self.accuracy, 4),
            "passed": self.passed,
        }


@dataclass
class BaselineResults:
    """Aggregated output of Phase 5 baseline evaluation.

    This is the contract passed to Phase 6 (hyperparameter tuning /
    ensembling).  The ``models`` dict maps model_name → SingleModelResult,
    providing a consistent format for downstream consumers.
    """

    models: Dict[str, SingleModelResult] = field(default_factory=dict)
    passed: bool = True
    coin_flip_ev: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Baseline Results: {'PASSED' if self.passed else 'FAILED'}",
            f"  Coin-flip EV: {self.coin_flip_ev:.2f}",
        ]
        for name, r in self.models.items():
            status = "OK" if r.passed else "FAIL"
            lines.append(
                f"  {name}: Brier={r.brier_score:.4f}, "
                f"EV={r.bracket_ev:.2f}, [{status}]"
            )
        if self.errors:
            lines.append(f"  Errors: {self.errors}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Return {model_name: {EV, Brier}} dict for downstream consumers."""
        return {name: r.to_dict() for name, r in self.models.items()}


# ---------------------------------------------------------------------------
# EV computation helpers
# ---------------------------------------------------------------------------


def _safe_log_loss(predictions: np.ndarray, outcomes: np.ndarray) -> float:
    """Compute log-loss with clipping to avoid log(0)."""
    eps = 1e-7
    p = np.clip(predictions, eps, 1 - eps)
    return -float(np.mean(
        outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p)
    ))


def compute_bracket_ev(
    predictions: np.ndarray,
    round_labels: Optional[np.ndarray] = None,
    scoring_weights: Optional[Dict[str, int]] = None,
) -> float:
    """Compute expected bracket points from game-level predictions.

    For each game the model predicts P(higher_seed_wins).  The expected
    points for that game = P * scoring_weight_for_round.  Total bracket
    EV = sum across all games.

    If ``round_labels`` is not provided, assumes all games are R64
    (conservative fallback for validation sets without round info).

    Args:
        predictions: Array of predicted probabilities (higher = favourite wins).
        round_labels: Per-game round identifier ("R64", "R32", ...).
        scoring_weights: Round → point mapping. Defaults to ESPN scoring.

    Returns:
        Expected total bracket points.
    """
    if scoring_weights is None:
        scoring_weights = ROUND_SCORING_WEIGHTS

    if round_labels is None:
        # Conservative fallback: assume all games weighted equally at R64
        weight = scoring_weights.get("R64", 10)
        return float(np.sum(predictions * weight))

    ev = 0.0
    for pred, rnd in zip(predictions, round_labels):
        rnd_str = str(rnd)
        weight = scoring_weights.get(rnd_str, 10)
        ev += float(pred) * weight
    return ev


def compute_coin_flip_ev(
    n_games: int = 63,
    round_structure: Optional[Dict[str, int]] = None,
    scoring_weights: Optional[Dict[str, int]] = None,
) -> float:
    """Expected bracket points for a 50/50 coin-flip predictor.

    Args:
        n_games: Total tournament games (default 63 for a 64-team bracket).
        round_structure: Number of games per round.
            Defaults to standard NCAA bracket: 32/16/8/4/2/1.
        scoring_weights: Round → point mapping.

    Returns:
        Expected points.
    """
    if scoring_weights is None:
        scoring_weights = ROUND_SCORING_WEIGHTS

    if round_structure is None:
        round_structure = {
            "R64": 32,
            "R32": 16,
            "S16": 8,
            "E8": 4,
            "F4": 2,
            "CHAMP": 1,
        }

    ev = 0.0
    for rnd, n in round_structure.items():
        weight = scoring_weights.get(rnd, 10)
        ev += 0.5 * n * weight
    return ev


# ---------------------------------------------------------------------------
# Baseline model trainers
# ---------------------------------------------------------------------------


class _LogisticBaseline:
    """Minimal logistic regression baseline wrapping sklearn."""

    def __init__(self):
        self.model = None
        self.scaler = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(np.nan_to_num(X, nan=0.0))
        self.model = LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=2000,
            solver="lbfgs",
        )
        self.model.fit(X_scaled, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_clean = np.nan_to_num(X, nan=0.0)
        X_scaled = self.scaler.transform(X_clean)
        return self.model.predict_proba(X_scaled)[:, 1]


class _EloBaseline:
    """Simple Elo-style baseline derived from feature differentials.

    Uses seed strength and efficiency margin (if present) as a proxy
    for Elo ratings, converted to probabilities via a logistic function.
    This is intentionally a dumb model — it validates that the feature
    pipeline produces sensible signals.
    """

    def __init__(self, k_factor: float = 400.0):
        self.k_factor = k_factor
        self.feature_index: Optional[int] = None
        self.loc: float = 0.0
        self.scale: float = 1.0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Fit Elo baseline by finding the single best predictive feature.

        Tries seed_strength and efficiency features first, then falls back
        to the feature with the highest AUC.
        """
        preferred = ["seed_strength", "adj_off_eff", "adj_def_eff"]
        best_idx = 0
        best_auc = 0.0

        for idx in range(X.shape[1]):
            col = X[:, idx]
            if np.all(np.isnan(col)):
                continue
            col_clean = np.nan_to_num(col, nan=0.0)
            # Simple AUC proxy: correlation with outcome
            corr = abs(np.corrcoef(col_clean, y)[0, 1])
            if np.isnan(corr):
                continue

            # Check if this is a preferred feature
            is_preferred = False
            if feature_names and idx < len(feature_names):
                is_preferred = feature_names[idx] in preferred

            if corr > best_auc or (is_preferred and corr > 0.05):
                best_auc = corr
                best_idx = idx
                if is_preferred:
                    break  # Use first strong preferred feature

        self.feature_index = best_idx
        col = np.nan_to_num(X[:, best_idx], nan=0.0)
        self.loc = float(np.mean(col))
        self.scale = float(np.std(col)) if np.std(col) > 1e-10 else 1.0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict via logistic function on the selected feature."""
        col = np.nan_to_num(X[:, self.feature_index], nan=0.0)
        z = (col - self.loc) / self.scale
        # Logistic sigmoid
        probs = 1.0 / (1.0 + np.exp(-z))
        return np.clip(probs, 0.01, 0.99)


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------


class BaselineEvaluator:
    """Phase 5 orchestrator: train baselines, evaluate, gate on EV.

    Args:
        brier_threshold: Maximum acceptable Brier score.
        min_bracket_ev_above_coinflip: Minimum EV improvement over coin-flip.
        strict: If True, raise IntegrityError on failure.
        scoring_weights: Round → point mapping for EV.
    """

    def __init__(
        self,
        brier_threshold: float = DEFAULT_BRIER_THRESHOLD,
        min_bracket_ev_above_coinflip: float = DEFAULT_MIN_BRACKET_EV,
        strict: bool = True,
        scoring_weights: Optional[Dict[str, int]] = None,
    ):
        self.brier_threshold = brier_threshold
        self.min_bracket_ev_above_coinflip = min_bracket_ev_above_coinflip
        self.strict = strict
        self.scoring_weights = scoring_weights or ROUND_SCORING_WEIGHTS

    def run(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        val_X: np.ndarray,
        val_y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        round_labels: Optional[np.ndarray] = None,
    ) -> BaselineResults:
        """Train and evaluate all baselines.

        Args:
            train_X: Training features (n_train, n_features).
            train_y: Training labels (0/1).
            val_X: Validation features.
            val_y: Validation labels (0/1).
            feature_names: Feature names for Elo feature selection.
            round_labels: Per-validation-game round labels for EV.

        Returns:
            BaselineResults with per-model scores and overall pass/fail.

        Raises:
            IntegrityError: If strict=True and any baseline fails thresholds.
        """
        results = BaselineResults()
        results.coin_flip_ev = compute_coin_flip_ev(
            n_games=len(val_y),
            scoring_weights=self.scoring_weights,
        )

        # --- Logistic Regression baseline ---
        lr_result = self._train_and_evaluate(
            name="logistic_regression",
            model=_LogisticBaseline(),
            train_X=train_X,
            train_y=train_y,
            val_X=val_X,
            val_y=val_y,
            feature_names=feature_names,
            round_labels=round_labels,
            results=results,
        )
        if lr_result is not None:
            results.models["logistic_regression"] = lr_result

        # --- Elo baseline ---
        elo_result = self._train_and_evaluate(
            name="Elo",
            model=_EloBaseline(),
            train_X=train_X,
            train_y=train_y,
            val_X=val_X,
            val_y=val_y,
            feature_names=feature_names,
            round_labels=round_labels,
            results=results,
        )
        if elo_result is not None:
            results.models["Elo"] = elo_result

        # --- Gate check ---
        for name, r in results.models.items():
            if not r.passed:
                results.passed = False
                msg = (
                    f"Baseline '{name}' failed: "
                    f"Brier={r.brier_score:.4f} "
                    f"(threshold {self.brier_threshold}), "
                    f"EV={r.bracket_ev:.2f} "
                    f"(coin-flip={results.coin_flip_ev:.2f}, "
                    f"min_above={self.min_bracket_ev_above_coinflip})"
                )
                results.errors.append(msg)
                logger.error(msg)

        logger.info(results.summary())

        if not results.passed and self.strict:
            raise IntegrityError(
                f"Phase 5 baseline gate failed. "
                f"Fix Phase 4 or data preprocessing. "
                f"Errors: {results.errors}"
            )

        return results

    def _train_and_evaluate(
        self,
        name: str,
        model: Any,
        train_X: np.ndarray,
        train_y: np.ndarray,
        val_X: np.ndarray,
        val_y: np.ndarray,
        feature_names: Optional[List[str]],
        round_labels: Optional[np.ndarray],
        results: BaselineResults,
    ) -> Optional[SingleModelResult]:
        """Train a single baseline and evaluate it."""
        try:
            if hasattr(model, "fit") and "feature_names" in model.fit.__code__.co_varnames:
                model.fit(train_X, train_y, feature_names=feature_names)
            else:
                model.fit(train_X, train_y)

            preds = model.predict_proba(val_X)
            preds = np.clip(preds, 1e-7, 1 - 1e-7)

            brier = float(BrierScoreOptimizer.calculate(preds, val_y))
            logloss = _safe_log_loss(preds, val_y)
            accuracy = float(np.mean((preds > 0.5) == val_y))

            bracket_ev = compute_bracket_ev(
                preds, round_labels, self.scoring_weights
            )

            passed_brier = brier <= self.brier_threshold
            passed_ev = (bracket_ev - results.coin_flip_ev) >= self.min_bracket_ev_above_coinflip

            result = SingleModelResult(
                model_name=name,
                brier_score=brier,
                log_loss=logloss,
                accuracy=accuracy,
                bracket_ev=bracket_ev,
                predictions=preds,
                passed_brier=passed_brier,
                passed_ev=passed_ev,
            )

            logger.info(
                "Baseline %s: Brier=%.4f, LogLoss=%.4f, Acc=%.3f, EV=%.2f [%s]",
                name, brier, logloss, accuracy, bracket_ev,
                "PASS" if result.passed else "FAIL",
            )
            return result

        except Exception as e:
            msg = f"Baseline '{name}' training failed: {e}"
            results.errors.append(msg)
            results.passed = False
            logger.error(msg)
            return None
