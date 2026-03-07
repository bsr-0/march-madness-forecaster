"""Shared pytest fixtures for the march-madness-forecaster test suite.

Provides reusable fixtures for team data, feature vectors, predictions,
outcomes, and pipeline configurations used across multiple test modules.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Prediction / outcome fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_predictions():
    """8 sample predicted probabilities for calibration and metric tests."""
    return np.array([0.3, 0.5, 0.7, 0.9, 0.2, 0.6, 0.8, 0.4])


@pytest.fixture
def sample_outcomes():
    """8 sample binary outcomes matching sample_predictions."""
    return np.array([0, 1, 1, 1, 0, 0, 1, 0], dtype=float)


@pytest.fixture
def large_predictions(rng):
    """200 sample predictions for statistical tests."""
    return rng.uniform(0.1, 0.9, size=200)


@pytest.fixture
def large_outcomes(large_predictions, rng):
    """200 binary outcomes generated from large_predictions."""
    return (rng.random(len(large_predictions)) < large_predictions).astype(float)


# ---------------------------------------------------------------------------
# Random number generator
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    """Seeded numpy random generator for reproducible tests."""
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Team data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_team_ids():
    """List of 8 sample team IDs."""
    return [
        "duke", "north-carolina", "kansas", "kentucky",
        "gonzaga", "villanova", "michigan-st", "virginia",
    ]


@pytest.fixture
def sample_team_features(rng):
    """8 teams × 22 features matrix (standardized)."""
    return rng.standard_normal((8, 22))


@pytest.fixture
def sample_matchup_features(rng):
    """4 matchups × 22 feature differences."""
    return rng.standard_normal((4, 22))


# ---------------------------------------------------------------------------
# LOYO year Brier fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_loyo_briers():
    """Per-year Brier scores from a sample LOYO run."""
    return {
        2017: 0.210,
        2018: 0.195,
        2019: 0.225,
        2021: 0.180,
        2022: 0.205,
        2023: 0.190,
        2024: 0.215,
    }


# ---------------------------------------------------------------------------
# Pipeline config fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_pipeline_config():
    """Minimal SOTAPipelineConfig for unit tests (no file I/O)."""
    from src.pipeline.sota import SOTAPipelineConfig
    return SOTAPipelineConfig(
        year=2025,
        num_simulations=100,
        pool_size=10,
        random_seed=42,
        enable_multi_year_training=False,
        enable_hyperparameter_tuning=False,
        strict_leakage_mode=False,
    )


# ---------------------------------------------------------------------------
# Temporary directory fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporary data directory with raw/ subdirectory."""
    raw = tmp_path / "raw"
    raw.mkdir()
    return tmp_path
