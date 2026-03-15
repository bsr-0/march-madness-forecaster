"""Unified calibration model interface and implementations.

Provides a shared ``CalibrationModel`` ABC so that every calibration
technique exposes the same ``fit`` / ``predict`` contract.  Three of the
four implementations wrap existing code in ``src/ml/calibration/``;
BetaCalibration is new.

All models:
- Clamp outputs to [ε, 1-ε]
- Expose ``fitted``, ``name``, ``n_parameters``
- Are compatible with scikit-learn pipeline patterns
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Clamp epsilon
_EPS = 1e-7

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class CalibrationModel(ABC):
    """Base class for all calibration models.

    Subclasses must implement ``fit`` and ``predict``.
    """

    _fitted: bool = False

    @abstractmethod
    def fit(self, pred_probs: np.ndarray, outcomes: np.ndarray) -> None:
        """Fit the calibration model.

        Args:
            pred_probs: Raw predicted probabilities, shape (n,).
            outcomes: Binary outcomes (0 or 1), shape (n,).
        """

    @abstractmethod
    def predict(self, pred_probs: np.ndarray) -> np.ndarray:
        """Apply calibration to raw probabilities.

        Args:
            pred_probs: Raw predicted probabilities, shape (n,).

        Returns:
            Calibrated probabilities, shape (n,), in [ε, 1-ε].
        """

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this calibration method."""

    @property
    @abstractmethod
    def n_parameters(self) -> int:
        """Number of fitted parameters."""

    def get_params(self) -> Dict[str, Any]:
        """Return fitted parameters as a dict (for serialization)."""
        return {}


# ---------------------------------------------------------------------------
# Temperature Scaling — wraps ml.calibration.calibration.TemperatureScaling
# ---------------------------------------------------------------------------

class TemperatureScalingModel(CalibrationModel):
    """Temperature scaling (Guo et al., 2017).

    Single parameter T rescales log-odds: p_cal = sigmoid(logit(p) / T).
    Delegates to ``src.ml.calibration.calibration.TemperatureScaling``.
    """

    def __init__(self):
        from ..ml.calibration.calibration import TemperatureScaling
        self._inner = TemperatureScaling()
        self._fitted = False

    def fit(self, pred_probs: np.ndarray, outcomes: np.ndarray) -> None:
        pred_probs = np.asarray(pred_probs, dtype=float)
        outcomes = np.asarray(outcomes, dtype=float)
        self._inner.fit(pred_probs, outcomes)
        self._fitted = True

    def predict(self, pred_probs: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ValueError("TemperatureScalingModel is not fitted")
        pred_probs = np.asarray(pred_probs, dtype=float)
        result = self._inner.calibrate(pred_probs)
        return np.clip(result, _EPS, 1.0 - _EPS)

    @property
    def name(self) -> str:
        return "temperature_scaling"

    @property
    def n_parameters(self) -> int:
        return 1

    def get_params(self) -> Dict[str, Any]:
        return {"temperature": self._inner.temperature}


# ---------------------------------------------------------------------------
# Logistic / Platt Scaling — wraps ml.calibration.calibration.PlattScaling
# ---------------------------------------------------------------------------

class LogisticCalibrationModel(CalibrationModel):
    """Platt-style logistic recalibration (2 parameters: a, b).

    p_cal = sigmoid(a * logit(p) + b).
    Delegates to ``src.ml.calibration.calibration.PlattScaling``.
    """

    def __init__(self):
        from ..ml.calibration.calibration import PlattScaling
        self._inner = PlattScaling()
        self._fitted = False

    def fit(self, pred_probs: np.ndarray, outcomes: np.ndarray) -> None:
        pred_probs = np.asarray(pred_probs, dtype=float)
        outcomes = np.asarray(outcomes, dtype=float)
        self._inner.fit(pred_probs, outcomes)
        self._fitted = True

    def predict(self, pred_probs: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ValueError("LogisticCalibrationModel is not fitted")
        pred_probs = np.asarray(pred_probs, dtype=float)
        result = self._inner.calibrate(pred_probs)
        return np.clip(result, _EPS, 1.0 - _EPS)

    @property
    def name(self) -> str:
        return "logistic_calibration"

    @property
    def n_parameters(self) -> int:
        return 2

    def get_params(self) -> Dict[str, Any]:
        return {"a": self._inner.a, "b": self._inner.b}


# ---------------------------------------------------------------------------
# Isotonic Regression — wraps ml.calibration.calibration.IsotonicCalibrator
# ---------------------------------------------------------------------------

class IsotonicCalibrationModel(CalibrationModel):
    """Isotonic regression calibration (nonparametric).

    Fits a monotone piecewise-constant function.
    Delegates to ``src.ml.calibration.calibration.IsotonicCalibrator``.
    """

    def __init__(self):
        from ..ml.calibration.calibration import IsotonicCalibrator
        self._inner = IsotonicCalibrator()
        self._fitted = False

    def fit(self, pred_probs: np.ndarray, outcomes: np.ndarray) -> None:
        pred_probs = np.asarray(pred_probs, dtype=float)
        outcomes = np.asarray(outcomes, dtype=float)
        self._inner.fit(pred_probs, outcomes)
        self._fitted = True

    def predict(self, pred_probs: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ValueError("IsotonicCalibrationModel is not fitted")
        pred_probs = np.asarray(pred_probs, dtype=float)
        result = self._inner.calibrate(pred_probs)
        return np.clip(result, _EPS, 1.0 - _EPS)

    @property
    def name(self) -> str:
        return "isotonic_regression"

    @property
    def n_parameters(self) -> int:
        # Nonparametric — report number of breakpoints
        if hasattr(self._inner, "isotonic") and hasattr(self._inner.isotonic, "X_thresholds_"):
            return len(self._inner.isotonic.X_thresholds_)
        return 0

    def get_params(self) -> Dict[str, Any]:
        return {"type": "nonparametric"}


# ---------------------------------------------------------------------------
# Beta Calibration — new implementation
# ---------------------------------------------------------------------------

class BetaCalibrationModel(CalibrationModel):
    """Beta calibration (Kull et al., 2017).

    Fits: p_cal = sigmoid(a * log(p / (1-p)) + b * log(p) + c)

    More flexible than Platt scaling because it can model asymmetric
    miscalibration (e.g., overconfident favorites but well-calibrated
    underdogs).

    Falls back to Platt-style (b=0) if scipy is unavailable.
    """

    def __init__(self):
        self._a: float = 1.0
        self._b: float = 0.0
        self._c: float = 0.0
        self._fitted = False

    def fit(self, pred_probs: np.ndarray, outcomes: np.ndarray) -> None:
        pred_probs = np.clip(np.asarray(pred_probs, dtype=float), _EPS, 1.0 - _EPS)
        outcomes = np.asarray(outcomes, dtype=float)

        logits = np.log(pred_probs / (1.0 - pred_probs))
        log_p = np.log(pred_probs)

        def neg_log_likelihood(params):
            a, b, c = params
            z = a * logits + b * log_p + c
            z = np.clip(z, -30.0, 30.0)
            probs = 1.0 / (1.0 + np.exp(-z))
            probs = np.clip(probs, _EPS, 1.0 - _EPS)
            return -float(np.mean(
                outcomes * np.log(probs) + (1.0 - outcomes) * np.log(1.0 - probs)
            ))

        if SCIPY_AVAILABLE:
            result = minimize(
                neg_log_likelihood,
                x0=[1.0, 0.0, 0.0],
                method="Nelder-Mead",
                options={"maxiter": 1000, "xatol": 1e-6, "fatol": 1e-8},
            )
            self._a, self._b, self._c = result.x
        else:
            # Grid search fallback over a coarse grid
            best_nll = float("inf")
            for a in np.linspace(0.5, 2.0, 10):
                for c in np.linspace(-1.0, 1.0, 10):
                    nll = neg_log_likelihood([a, 0.0, c])
                    if nll < best_nll:
                        best_nll = nll
                        self._a, self._b, self._c = a, 0.0, c

        self._fitted = True
        logger.debug(
            "BetaCalibration fitted: a=%.4f, b=%.4f, c=%.4f",
            self._a, self._b, self._c,
        )

    def predict(self, pred_probs: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ValueError("BetaCalibrationModel is not fitted")
        pred_probs = np.clip(np.asarray(pred_probs, dtype=float), _EPS, 1.0 - _EPS)
        logits = np.log(pred_probs / (1.0 - pred_probs))
        log_p = np.log(pred_probs)
        z = self._a * logits + self._b * log_p + self._c
        z = np.clip(z, -30.0, 30.0)
        result = 1.0 / (1.0 + np.exp(-z))
        return np.clip(result, _EPS, 1.0 - _EPS)

    @property
    def name(self) -> str:
        return "beta_calibration"

    @property
    def n_parameters(self) -> int:
        return 3

    def get_params(self) -> Dict[str, Any]:
        return {"a": self._a, "b": self._b, "c": self._c}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CALIBRATION_REGISTRY: Dict[str, type] = {
    "temperature_scaling": TemperatureScalingModel,
    "logistic_calibration": LogisticCalibrationModel,
    "isotonic_regression": IsotonicCalibrationModel,
    "beta_calibration": BetaCalibrationModel,
}


def get_all_calibration_models() -> list[CalibrationModel]:
    """Instantiate one of each registered calibration model."""
    return [cls() for cls in CALIBRATION_REGISTRY.values()]


def get_calibration_model(name: str) -> CalibrationModel:
    """Instantiate a calibration model by name.

    Args:
        name: One of the keys in CALIBRATION_REGISTRY.

    Returns:
        An unfitted CalibrationModel instance.

    Raises:
        KeyError: If name is not registered.
    """
    if name not in CALIBRATION_REGISTRY:
        raise KeyError(
            f"Unknown calibration method '{name}'. "
            f"Available: {list(CALIBRATION_REGISTRY.keys())}"
        )
    return CALIBRATION_REGISTRY[name]()
