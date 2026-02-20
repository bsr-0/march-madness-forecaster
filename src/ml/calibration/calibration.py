"""
Probability calibration for tournament predictions.

Ensures predicted probabilities are well-calibrated:
- A prediction of 70% should win ~70% of the time
- Uses Brier Score optimization and Isotonic Regression

Reference: "Obtaining Calibrated Probabilities from Boosting" (Niculescu-Mizil & Caruana)
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy.optimize import minimize_scalar
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.calibration import calibration_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class CalibrationMetrics:
    """Metrics for evaluating probability calibration."""
    
    brier_score: float  # Mean squared error of probabilities
    log_loss: float  # Log loss (cross-entropy)
    expected_calibration_error: float  # ECE
    max_calibration_error: float  # MCE
    accuracy: float  # Classification accuracy
    
    # Calibration curve data
    prob_true: np.ndarray  # True positive rate per bin
    prob_pred: np.ndarray  # Mean predicted probability per bin
    
    def __str__(self) -> str:
        return (
            f"Calibration Metrics:\n"
            f"  Brier Score: {self.brier_score:.4f}\n"
            f"  Log Loss: {self.log_loss:.4f}\n"
            f"  ECE: {self.expected_calibration_error:.4f}\n"
            f"  MCE: {self.max_calibration_error:.4f}\n"
            f"  Accuracy: {self.accuracy:.4f}"
        )


class BrierScoreOptimizer:
    """
    Optimizes predictions to minimize Brier Score.
    
    Brier Score = (1/N) * Σ(predicted - actual)²
    
    Lower is better. Perfect = 0, Coin flip = 0.25
    """
    
    @staticmethod
    def calculate(predictions: np.ndarray, outcomes: np.ndarray) -> float:
        """
        Calculate Brier Score.
        
        Args:
            predictions: Predicted probabilities [N]
            outcomes: Actual outcomes (0 or 1) [N]
            
        Returns:
            Brier score
        """
        return np.mean((predictions - outcomes) ** 2)
    
    @staticmethod
    def decompose(
        predictions: np.ndarray, 
        outcomes: np.ndarray,
        n_bins: int = 10
    ) -> Tuple[float, float, float]:
        """
        Decompose Brier Score into reliability, resolution, and uncertainty.
        
        Brier = Reliability - Resolution + Uncertainty
        
        - Reliability (lower is better): How well calibrated
        - Resolution (higher is better): How much predictions deviate from base rate
        - Uncertainty: Inherent variance in outcomes
        
        Args:
            predictions: Predicted probabilities
            outcomes: Actual outcomes
            n_bins: Number of bins for calculation
            
        Returns:
            Tuple of (reliability, resolution, uncertainty)
        """
        # Base rate
        base_rate = np.mean(outcomes)
        uncertainty = base_rate * (1 - base_rate)
        
        # Bin predictions
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predictions, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        reliability = 0.0
        resolution = 0.0
        
        for i in range(n_bins):
            mask = bin_indices == i
            n_k = np.sum(mask)
            
            if n_k > 0:
                mean_pred = np.mean(predictions[mask])
                mean_outcome = np.mean(outcomes[mask])
                
                reliability += n_k * (mean_pred - mean_outcome) ** 2
                resolution += n_k * (mean_outcome - base_rate) ** 2
        
        n = len(predictions)
        reliability /= n
        resolution /= n
        
        return reliability, resolution, uncertainty
    
    @staticmethod
    def skill_score(predictions: np.ndarray, outcomes: np.ndarray) -> float:
        """
        Calculate Brier Skill Score (BSS).
        
        BSS = 1 - (BS / BS_ref)
        
        Where BS_ref is the Brier score of always predicting the base rate.
        BSS > 0 means better than just predicting base rate.
        
        Args:
            predictions: Predicted probabilities
            outcomes: Actual outcomes
            
        Returns:
            Brier Skill Score
        """
        bs = BrierScoreOptimizer.calculate(predictions, outcomes)
        
        # Reference: predicting base rate
        base_rate = np.mean(outcomes)
        bs_ref = base_rate * (1 - base_rate)
        
        if bs_ref == 0:
            return 0.0
        
        return 1 - (bs / bs_ref)


class IsotonicCalibrator:
    """
    Isotonic regression calibrator.
    
    Fits a monotonic function to map raw predictions to calibrated probabilities.
    Effective post-processing to improve calibration without hurting discrimination.
    """
    
    def __init__(self):
        """Initialize calibrator."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for isotonic calibration")
        
        self.isotonic = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds='clip'
        )
        self.fitted = False
    
    def fit(self, predictions: np.ndarray, outcomes: np.ndarray) -> None:
        """
        Fit calibrator on historical data.
        
        Args:
            predictions: Raw predicted probabilities
            outcomes: Actual outcomes (0 or 1)
        """
        self.isotonic.fit(predictions, outcomes)
        self.fitted = True
    
    def calibrate(self, predictions: np.ndarray) -> np.ndarray:
        """
        Calibrate predictions.
        
        Args:
            predictions: Raw predictions
            
        Returns:
            Calibrated predictions
        """
        if not self.fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        
        return self.isotonic.predict(predictions)
    
    def fit_calibrate(
        self, 
        predictions: np.ndarray, 
        outcomes: np.ndarray
    ) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            predictions: Raw predictions
            outcomes: Actual outcomes
            
        Returns:
            Calibrated predictions
        """
        self.fit(predictions, outcomes)
        return self.calibrate(predictions)


class PlattScaling:
    """
    Platt scaling (sigmoid calibration).
    
    Fits a logistic regression to map scores to probabilities.
    """
    
    def __init__(self):
        self.a = 1.0
        self.b = 0.0
        self.fitted = False
    
    def fit(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        max_iter: int = 200,
        lr: float = 0.1,
        tol: float = 1e-6,
        patience: int = 5,
    ) -> None:
        """
        Fit Platt scaling parameters using gradient descent with convergence checks.

        Uses best-tracking, adaptive learning rate (halves on NLL increase),
        early stopping (tolerance + patience), and gradient norm convergence.

        Args:
            predictions: Raw predictions (log-odds or probabilities)
            outcomes: Actual outcomes
            max_iter: Maximum iterations
            lr: Initial learning rate
            tol: Convergence tolerance on NLL change
            patience: Early-stop after this many non-improving iterations
        """
        # Convert to log-odds if predictions are probabilities
        if np.all(predictions >= 0) and np.all(predictions <= 1):
            predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
            scores = np.log(predictions / (1 - predictions))
        else:
            scores = predictions

        outcomes = np.asarray(outcomes, dtype=float)

        a, b = 1.0, 0.0
        best_a, best_b = 1.0, 0.0
        best_nll = float("inf")
        no_improve_count = 0
        prev_nll = float("inf")

        import math

        for _ in range(max_iter):
            # Forward pass
            scaled = np.clip(a * scores + b, -30.0, 30.0)
            probs = 1.0 / (1.0 + np.exp(-scaled))
            probs = np.clip(probs, 1e-7, 1 - 1e-7)

            # Negative log-likelihood
            nll = float(-np.mean(
                outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs)
            ))

            # Best-tracking
            if nll < best_nll:
                best_nll = nll
                best_a, best_b = a, b
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Early stopping: converged if NLL change < tol for patience iters
            if abs(prev_nll - nll) < tol and no_improve_count >= patience:
                break

            # Gradients
            grad_a = float(np.mean((probs - outcomes) * scores))
            grad_b = float(np.mean(probs - outcomes))

            # Gradient norm convergence check
            grad_norm = math.sqrt(grad_a ** 2 + grad_b ** 2)
            if grad_norm < 1e-8:
                break

            # Adaptive LR: halve if NLL increased
            if nll > prev_nll:
                lr *= 0.5

            prev_nll = nll

            a -= lr * grad_a
            b -= lr * grad_b

        self.a = best_a
        self.b = best_b
        self.fitted = True
    
    def calibrate(self, predictions: np.ndarray) -> np.ndarray:
        """Apply Platt scaling."""
        if not self.fitted:
            raise ValueError("Not fitted")
        
        # Convert to log-odds if needed
        if np.all(predictions >= 0) and np.all(predictions <= 1):
            predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
            scores = np.log(predictions / (1 - predictions))
        else:
            scores = predictions
        
        return 1.0 / (1.0 + np.exp(-(self.a * scores + self.b)))


class TemperatureScaling:
    """
    Temperature scaling calibration (Guo et al., 2017).

    Uses a single scalar parameter T to rescale log-odds:
        p_calibrated = sigmoid(logit(p_raw) / T)

    Advantages over Platt/Isotonic:
    - Only 1 parameter → far less overfitting risk with small datasets
    - Specifically targets the overconfidence problem
    - Preserves ranking (monotonic transformation)
    - Well-suited for March Madness where we have <1000 calibration samples

    Reference: "On Calibration of Modern Neural Networks" (Guo et al., 2017)
    """

    def __init__(self):
        self.temperature = 1.0
        self.fitted = False

    def fit(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        max_iter: int = 200,
        lr: float = 0.01,
        _skip_guard: bool = False,
    ) -> None:
        """
        Fit temperature parameter by minimizing negative log-likelihood (NLL).

        Uses scipy bounded scalar minimization (Brent's method) when available,
        falling back to gradient descent with best-tracking otherwise. Brent's
        method is superlinearly convergent on unimodal functions and respects
        bounds without gradient clipping artifacts.

        The NLL as a function of T is convex (it's a composition of the convex
        cross-entropy with the monotone logit rescaling), so Brent's method
        finds the global optimum within the bounded interval.

        Args:
            predictions: Raw predicted probabilities
            outcomes: Actual outcomes (0 or 1)
            max_iter: Maximum iterations (used for fallback GD only)
            lr: Learning rate (used for fallback GD only)
        """
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        logits = np.log(predictions / (1 - predictions))
        outcomes = outcomes.astype(float)

        # Small-sample guard: with fewer than 30 calibration samples, the NLL
        # landscape is noisy and Brent's method may converge to an extreme
        # boundary value (T=0.1 or T=10.0).  In this regime, verify via
        # bootstrap CI that T is statistically distinguishable from 1.0.
        # If not, keep T=1.0 (identity calibration) to avoid harmful distortion.
        # _skip_guard is True for bootstrap resamples to avoid infinite recursion.
        self._small_sample_guard = len(predictions) < 30 and not _skip_guard

        def nll_at_T(T: float) -> float:
            """Negative log-likelihood at temperature T."""
            scaled = logits / T
            # Numerically stable sigmoid: use np.clip on scaled logits to
            # prevent overflow in exp() for extreme logit / small T values.
            scaled = np.clip(scaled, -30.0, 30.0)
            probs = 1.0 / (1.0 + np.exp(-scaled))
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            return float(-np.mean(
                outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs)
            ))

        if SCIPY_AVAILABLE:
            # Brent's method: superlinearly convergent, respects bounds natively.
            # Bounds [0.1, 10.0] cover the practical range — T < 0.1 sharpens
            # probabilities toward 0/1 (extreme overconfidence), T > 10.0
            # flattens everything toward 0.5 (no discrimination).
            result = minimize_scalar(
                nll_at_T,
                bounds=(0.1, 10.0),
                method="bounded",
                options={"xatol": 1e-6, "maxiter": 500},
            )
            self.temperature = float(result.x)
        else:
            # Fallback: gradient descent with best-tracking (no clipping artifacts).
            T = 1.0
            best_T = 1.0
            best_nll = nll_at_T(1.0)

            for _ in range(max_iter):
                scaled_logits = np.clip(logits / T, -30.0, 30.0)
                probs = 1.0 / (1.0 + np.exp(-scaled_logits))
                probs = np.clip(probs, 1e-7, 1 - 1e-7)

                nll = float(-np.mean(
                    outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs)
                ))

                if nll < best_nll:
                    best_nll = nll
                    best_T = T

                # Gradient of NLL w.r.t. T
                grad = np.mean((probs - outcomes) * (-logits / (T ** 2)))
                T -= lr * grad
                T = max(0.1, min(T, 10.0))

            self.temperature = best_T

        # Small-sample guard: verify T is statistically justified.
        if self._small_sample_guard:
            try:
                ci_lo, ci_hi = self.bootstrap_ci(
                    predictions, outcomes,
                    n_bootstrap=200, ci_level=0.90,
                )
                if ci_lo <= 1.0 <= ci_hi:
                    # CI includes identity — not enough evidence to calibrate
                    self.temperature = 1.0
            except Exception as e:
                logger.warning(
                    "Temperature scaling bootstrap CI failed (%s); "
                    "falling back to T=1.0", e,
                )
                self.temperature = 1.0

        self.fitted = True

    def calibrate(self, predictions: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to predictions."""
        if not self.fitted:
            raise ValueError("Not fitted")

        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        logits = np.log(predictions / (1 - predictions))
        scaled_logits = logits / self.temperature
        return 1.0 / (1.0 + np.exp(-scaled_logits))

    def bootstrap_ci(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        n_bootstrap: int = 200,
        ci_level: float = 0.95,
        random_seed: int = 42,
    ) -> tuple:
        """Compute bootstrap confidence interval for the temperature parameter.

        Fits temperature scaling on ``n_bootstrap`` resamples (with replacement)
        and returns the (lower, upper) CI bounds for T.  If the CI includes 1.0,
        the calibration is not statistically justified (T=1 is the identity).

        Args:
            predictions: Raw predicted probabilities
            outcomes: Actual outcomes (0 or 1)
            n_bootstrap: Number of bootstrap resamples
            ci_level: Confidence level (default 0.95 for 95% CI)
            random_seed: Random seed for reproducibility

        Returns:
            Tuple of (T_lower, T_upper, T_values) where T_values is the array
            of fitted temperatures across bootstrap resamples.
        """
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        outcomes = outcomes.astype(float)
        n = len(predictions)
        rng = np.random.default_rng(random_seed)

        T_values = []
        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            boot_pred = predictions[idx]
            boot_out = outcomes[idx]

            ts = TemperatureScaling()
            ts.fit(boot_pred, boot_out, _skip_guard=True)
            T_values.append(ts.temperature)

        T_values = np.array(T_values)
        alpha = (1.0 - ci_level) / 2.0
        T_lower = float(np.percentile(T_values, 100 * alpha))
        T_upper = float(np.percentile(T_values, 100 * (1.0 - alpha)))

        return T_lower, T_upper, T_values


def calculate_calibration_metrics(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10
) -> CalibrationMetrics:
    """
    Calculate comprehensive calibration metrics.
    
    Args:
        predictions: Predicted probabilities
        outcomes: Actual outcomes (0 or 1)
        n_bins: Number of bins for calibration curve
        
    Returns:
        CalibrationMetrics object
    """
    predictions = np.array(predictions)
    outcomes = np.array(outcomes).astype(float)
    
    # Brier Score
    brier = BrierScoreOptimizer.calculate(predictions, outcomes)
    
    # Log Loss
    eps = 1e-7
    probs_clipped = np.clip(predictions, eps, 1 - eps)
    log_loss = -np.mean(
        outcomes * np.log(probs_clipped) + 
        (1 - outcomes) * np.log(1 - probs_clipped)
    )
    
    # Calibration curve
    if SKLEARN_AVAILABLE:
        prob_true, prob_pred = calibration_curve(
            outcomes, predictions, n_bins=n_bins, strategy='uniform'
        )
    else:
        # Manual calculation
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predictions, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        prob_true = []
        prob_pred = []
        
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                prob_true.append(np.mean(outcomes[mask]))
                prob_pred.append(np.mean(predictions[mask]))
        
        prob_true = np.array(prob_true)
        prob_pred = np.array(prob_pred)
    
    # ECE and MCE
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(predictions, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece = 0.0
    mce = 0.0
    
    for i in range(n_bins):
        mask = bin_indices == i
        n_k = np.sum(mask)
        
        if n_k > 0:
            mean_pred = np.mean(predictions[mask])
            mean_outcome = np.mean(outcomes[mask])
            gap = abs(mean_pred - mean_outcome)
            
            ece += (n_k / len(predictions)) * gap
            mce = max(mce, gap)
    
    # Accuracy
    predicted_class = (predictions >= 0.5).astype(int)
    accuracy = np.mean(predicted_class == outcomes)
    
    return CalibrationMetrics(
        brier_score=brier,
        log_loss=log_loss,
        expected_calibration_error=ece,
        max_calibration_error=mce,
        accuracy=accuracy,
        prob_true=prob_true,
        prob_pred=prob_pred,
    )


def plot_reliability_diagram(
    metrics: CalibrationMetrics,
    title: str = "Reliability Diagram",
    save_path: Optional[str] = None
) -> None:
    """
    Plot reliability diagram (calibration curve).
    
    Args:
        metrics: CalibrationMetrics with prob_true and prob_pred
        title: Plot title
        save_path: Path to save figure (optional)
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available for plotting")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Reliability diagram
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    ax1.plot(metrics.prob_pred, metrics.prob_true, 's-', label='Model')
    ax1.fill_between(
        metrics.prob_pred,
        metrics.prob_true,
        metrics.prob_pred,
        alpha=0.3,
        label='Gap'
    )
    
    ax1.set_xlabel('Mean Predicted Probability')
    ax1.set_ylabel('Fraction of Positives')
    ax1.set_title(title)
    ax1.legend()
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)
    
    # Add metrics text
    metrics_text = (
        f"Brier: {metrics.brier_score:.4f}\n"
        f"ECE: {metrics.expected_calibration_error:.4f}\n"
        f"MCE: {metrics.max_calibration_error:.4f}"
    )
    ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Histogram of predictions
    ax2.hist(metrics.prob_pred, bins=20, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Count')
    ax2.set_title('Prediction Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()


class CalibrationPipeline:
    """
    Complete calibration pipeline for tournament predictions.
    """
    
    def __init__(self, method: str = "isotonic"):
        """
        Initialize pipeline.

        Args:
            method: Calibration method ("isotonic", "platt", "temperature", or "none")
        """
        self.method = method

        if method == "isotonic":
            self.calibrator = IsotonicCalibrator()
        elif method == "platt":
            self.calibrator = PlattScaling()
        elif method == "temperature":
            self.calibrator = TemperatureScaling()
        else:
            self.calibrator = None
    
    def fit(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray
    ) -> CalibrationMetrics:
        """
        Fit calibrator and return metrics on training data.
        
        Args:
            predictions: Historical predictions
            outcomes: Historical outcomes
            
        Returns:
            Pre-calibration metrics
        """
        pre_metrics = calculate_calibration_metrics(predictions, outcomes)
        
        if self.calibrator:
            self.calibrator.fit(predictions, outcomes)
        
        return pre_metrics
    
    def calibrate(self, predictions: np.ndarray) -> np.ndarray:
        """
        Calibrate new predictions.
        
        Args:
            predictions: Raw predictions
            
        Returns:
            Calibrated predictions
        """
        if self.calibrator:
            return self.calibrator.calibrate(predictions)
        return predictions
    
    def evaluate(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray
    ) -> Tuple[CalibrationMetrics, CalibrationMetrics]:
        """
        Evaluate calibration improvement.
        
        Args:
            predictions: Raw predictions
            outcomes: Actual outcomes
            
        Returns:
            Tuple of (pre_calibration_metrics, post_calibration_metrics)
        """
        pre_metrics = calculate_calibration_metrics(predictions, outcomes)
        
        if self.calibrator:
            calibrated = self.calibrate(predictions)
            post_metrics = calculate_calibration_metrics(calibrated, outcomes)
        else:
            post_metrics = pre_metrics
        
        return pre_metrics, post_metrics
