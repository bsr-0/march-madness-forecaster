"""Shared pytest fixtures for the march-madness-forecaster test suite.

Provides reusable fixtures for team data, feature vectors, predictions,
outcomes, and pipeline configurations used across multiple test modules.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Phase 7.2: Auto-apply test gate markers based on file path / name
# ---------------------------------------------------------------------------

# Mapping from filename substrings to pytest markers.
# Tests can also opt-in to markers explicitly via @pytest.mark.<name>.
_AUTO_MARKER_RULES = [
    # data-contract tests
    ("data_integrity/test_leakage_rules", "leakage"),
    ("data_integrity/test_point_in_time_contracts", "data_contract"),
    ("data_integrity/test_training_row_assembly", "data_contract"),
    ("data_integrity/test_lineage_manifests", "data_contract"),
    ("data_integrity/test_provider_robustness", "data_contract"),
    # leakage tests
    ("test_leakage", "leakage"),
    ("test_date_integrity", "leakage"),
    ("test_elo_temporal", "leakage"),
    # calibration tests
    ("evaluation/test_calibration", "calibration"),
    ("evaluation/test_reliability", "calibration"),
    ("evaluation/test_selection_snapshot", "calibration"),
    ("evaluation/test_round_segmentation", "calibration"),
    # freeze / reproducibility tests
    ("test_production_2026_freeze", "freeze"),
    ("test_deterministic_replay", "freeze"),
    ("test_production_2026_freeze", "production"),
    # backtest regression tests
    ("test_sota_pipeline", "backtest_regression"),
    ("test_e2e_pipeline", "backtest_regression"),
    # live protocol
    ("test_live_protocol", "live_protocol"),
]


def pytest_collection_modifyitems(items):
    """Auto-apply gate markers to collected test items based on file path."""
    for item in items:
        rel_path = str(item.fspath)
        for pattern, marker_name in _AUTO_MARKER_RULES:
            if pattern in rel_path:
                item.add_marker(getattr(pytest.mark, marker_name))

        # Auto-mark anything under tests/ that doesn't already have a gate
        # marker as 'unit' (the default gate).
        gate_markers = {
            "unit", "data_contract", "leakage", "backtest_regression",
            "freeze", "production", "calibration", "live_protocol",
            "integration",
        }
        item_markers = {m.name for m in item.iter_markers()}
        if not item_markers & gate_markers:
            item.add_marker(pytest.mark.unit)


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
