"""ESPN bracket optimization subsystem.

Implements the end-to-end ESPN pathway described in Protocol v2:
- Leverage calculation from model probabilities vs public pick rates.
- Monte Carlo pool simulation with opponent bracket generation.
- Path-dependent bracket optimization for rank percentile and top-10% finishes.
"""

from .bracket_optimizer import (
    ESPNBracketOptimizer,
    ESPNOptimizationConfig,
    ESPNOptimizationResult,
)
from .leverage import LeverageSignal, compute_leverage_table
from .mc_simulator import ESPNMCSimConfig, ESPNMCSimResult, ESPNMonteCarloSimulator
from .public_pick_scraper import load_public_picks  # noqa: F401

__all__ = [
    "LeverageSignal",
    "compute_leverage_table",
    "load_public_picks",
    "ESPNMCSimConfig",
    "ESPNMCSimResult",
    "ESPNMonteCarloSimulator",
    "ESPNOptimizationConfig",
    "ESPNOptimizationResult",
    "ESPNBracketOptimizer",
]
