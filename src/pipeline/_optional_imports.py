"""Centralized optional dependency checks for the SOTA pipeline.

All try/except import guards live here to keep sota.py clean.
Each optional dependency exposes:
  - The imported symbol (or None if unavailable)
  - A boolean flag (e.g., TORCH_AVAILABLE)
"""

# --- PyTorch (GNN, Transformer) ---
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

# --- scikit-learn ---
try:
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    LogisticRegression = None  # type: ignore[assignment,misc]
    SKLEARN_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    SCALER_AVAILABLE = True
except ImportError:
    StandardScaler = None  # type: ignore[assignment,misc]
    SCALER_AVAILABLE = False

# --- Hyperparameter tuning (Optuna-based) ---
try:
    from ..ml.optimization.hyperparameter_tuning import (
        LightGBMTuner,
        BrierLightGBMTuner,
        XGBoostTuner,
        LogisticTuner,
        EnsembleWeightOptimizer,
        TemporalCrossValidator,
        LeaveOneYearOutCV,
        OPTUNA_AVAILABLE,
        XGBOOST_AVAILABLE as TUNER_XGBOOST_AVAILABLE,
    )
except ImportError:
    OPTUNA_AVAILABLE = False
    LightGBMTuner = None
    BrierLightGBMTuner = None
    XGBoostTuner = None
    LogisticTuner = None
    EnsembleWeightOptimizer = None
    TemporalCrossValidator = None
    LeaveOneYearOutCV = None
    TUNER_XGBOOST_AVAILABLE = False

# --- Statistical tests ---
try:
    from ..ml.evaluation.statistical_tests import model_significance_report
    SIGNIFICANCE_TESTING_AVAILABLE = True
except ImportError:
    model_significance_report = None
    SIGNIFICANCE_TESTING_AVAILABLE = False

# --- Ablation study ---
try:
    from ..ml.evaluation.ablation import AblationStudy
    ABLATION_AVAILABLE = True
except ImportError:
    AblationStudy = None
    ABLATION_AVAILABLE = False

# --- Spread model ---
try:
    from ..ml.ensemble.spread_model import SpreadRegressor
    SPREAD_MODEL_AVAILABLE = True
except ImportError:
    SpreadRegressor = None
    SPREAD_MODEL_AVAILABLE = False

# --- Tournament sigma calibrator ---
try:
    from ..ml.ensemble.tournament_sigma import (
        TournamentSigmaCalibrator,
        load_tournament_sigma_data,
        daynum_to_round,
    )
    TOURNAMENT_SIGMA_AVAILABLE = True
except ImportError:
    TournamentSigmaCalibrator = None
    TOURNAMENT_SIGMA_AVAILABLE = False

# --- Bayesian Bradley-Terry ---
try:
    from ..ml.ensemble.bayesian_bt import BayesianBradleyTerry
    BAYESIAN_BT_AVAILABLE = True
except ImportError:
    BayesianBradleyTerry = None
    BAYESIAN_BT_AVAILABLE = False
