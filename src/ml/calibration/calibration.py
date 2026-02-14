"""
Probability calibration for tournament predictions.

Ensures predicted probabilities are well-calibrated:
- A prediction of 70% should win ~70% of the time
- Uses Brier Score optimization and Isotonic Regression

Reference: "Obtaining Calibrated Probabilities from Boosting" (Niculescu-Mizil & Caruana)
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

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
        max_iter: int = 100
    ) -> None:
        """
        Fit Platt scaling parameters using gradient descent.
        
        Args:
            predictions: Raw predictions (log-odds or probabilities)
            outcomes: Actual outcomes
            max_iter: Maximum iterations
        """
        # Convert to log-odds if predictions are probabilities
        if np.all(predictions >= 0) and np.all(predictions <= 1):
            predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
            scores = np.log(predictions / (1 - predictions))
        else:
            scores = predictions
        
        # Fit using gradient descent
        a, b = 1.0, 0.0
        learning_rate = 0.1
        
        for _ in range(max_iter):
            probs = 1.0 / (1.0 + np.exp(-(a * scores + b)))
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            
            # Gradients
            grad_a = np.mean((probs - outcomes) * scores)
            grad_b = np.mean(probs - outcomes)
            
            a -= learning_rate * grad_a
            b -= learning_rate * grad_b
        
        self.a = a
        self.b = b
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
            method: Calibration method ("isotonic", "platt", or "none")
        """
        self.method = method
        
        if method == "isotonic":
            self.calibrator = IsotonicCalibrator()
        elif method == "platt":
            self.calibrator = PlattScaling()
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
