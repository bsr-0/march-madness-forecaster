"""Forecasting module — isolated, honest game-level prediction engine.

This module contains the ForecastEngine and its configuration.  The engine
produces calibrated win probabilities that are strictly invariant to pool
size, scoring rules, public sentiment, or any other downstream optimization
parameter.
"""

from .engine import (
    CalibrationReport,
    ForecastEngine,
    ForecastEngineConfig,
)

__all__ = [
    "CalibrationReport",
    "ForecastEngine",
    "ForecastEngineConfig",
]
