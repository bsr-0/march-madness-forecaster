"""Custom Brier-based training objective for LightGBM.

Replaces the standard binary log loss objective with a custom objective
that directly minimizes Brier score, aligning the training metric with
the Kaggle evaluation metric (Protocol Section 3.3).

The Brier score is ``mean((sigmoid(f) - y)^2)`` where ``f`` is the
raw logit output.  We derive gradient and hessian w.r.t. ``f``:

    Let s = sigmoid(f) = 1 / (1 + exp(-f))
    ds/df = s * (1 - s)

    L = (s - y)^2
    dL/df = 2 * (s - y) * ds/df = 2 * s * (1-s) * (s - y)
    d^2L/df^2 = 2 * s * (1-s) * ((1-2s)(s-y) + s*(1-s))

The hessian is clamped to 1e-6 minimum for numerical stability.

References:
    - Protocol v2 Section 3.3: Training-selection alignment
    - Kaggle March Machine Learning Mania (2023+): Brier scoring
"""

from __future__ import annotations

import numpy as np


def brier_objective(preds, train_data):
    """Custom Brier score objective for LightGBM.

    This function computes the gradient and hessian of the Brier score
    with respect to the raw logit predictions, allowing LightGBM to
    directly optimize for Brier score rather than log loss.

    Args:
        preds: Raw logit predictions from the model (before sigmoid).
        train_data: LightGBM Dataset with labels.

    Returns:
        Tuple of (gradient, hessian) arrays.
    """
    labels = train_data.get_label()
    s = 1.0 / (1.0 + np.exp(-np.clip(preds, -500, 500)))

    # Gradient: dBrier/df = 2 * s * (1-s) * (s - y)
    grad = 2.0 * s * (1.0 - s) * (s - labels)

    # Hessian: d^2Brier/df^2
    hess = 2.0 * s * (1.0 - s) * (
        (1.0 - 2.0 * s) * (s - labels) + s * (1.0 - s)
    )
    # Clamp for numerical stability (prevent negative curvature)
    hess = np.maximum(hess, 1e-6)

    return grad, hess


def brier_eval(preds, train_data):
    """Custom Brier score evaluation metric for LightGBM validation.

    Args:
        preds: Raw logit predictions (before sigmoid).
        train_data: LightGBM Dataset with labels.

    Returns:
        Tuple of (metric_name, metric_value, is_higher_better).
    """
    labels = train_data.get_label()
    s = 1.0 / (1.0 + np.exp(-np.clip(preds, -500, 500)))
    brier = float(np.mean((s - labels) ** 2))
    return "brier", brier, False  # Lower is better


try:
    from .cfa import LightGBMRanker

    class BrierLightGBMRanker(LightGBMRanker):
        """LightGBM ranker using custom Brier objective instead of log loss.

        Inherits all functionality from ``LightGBMRanker`` but replaces
        the ``objective`` parameter with the custom Brier gradient/hessian
        and adds the Brier evaluation metric.
        """

        def __init__(self, params=None):
            super().__init__(params)
            # Use objective="none" so LightGBM doesn't try to reset it
            # internally. The custom Brier objective is passed via fobj
            # in the Booster.update() call (see train() override below).
            self.params["objective"] = "none"
            self.params.pop("metric", None)  # Use custom feval instead
            self._brier_eval = brier_eval
            self._brier_objective = brier_objective

        def train(self, X, y, **kwargs):
            """Train with Brier objective and evaluation metric.

            Overrides parent to use the Booster API directly, because
            LightGBM 4.x crashes when a callable is passed in
            params["objective"] due to an internal reset_parameter bug.
            """
            import lightgbm as lgb

            if "feval" not in kwargs:
                kwargs["feval"] = self._brier_eval

            feature_names = kwargs.pop("feature_names", None) or self.feature_names
            num_rounds = kwargs.pop("num_rounds", 500)
            sample_weight = kwargs.pop("sample_weight", None)
            valid_set = kwargs.pop("valid_set", None)

            train_data = lgb.Dataset(
                X, label=y, feature_name=feature_names,
                weight=sample_weight, free_raw_data=False,
            )

            booster = lgb.Booster(self.params, train_data)
            for _ in range(num_rounds):
                booster.update(fobj=self._brier_objective)

            self.model = booster

except ImportError:
    # LightGBM not available; BrierLightGBMRanker cannot be created
    pass
