"""ESPN bracket optimization subsystem.

Implements the end-to-end ESPN pathway described in Protocol v2:
- Leverage calculation from model probabilities vs public pick rates.
- Monte Carlo pool simulation with opponent bracket generation.
- Path-dependent bracket optimization for rank percentile and top-10% finishes.
"""

from .public_pick_scraper import load_public_picks  # noqa: F401

__all__ = ["load_public_picks"]

# bracket_optimizer, leverage and mc_simulator were deleted in commit 44b048f
# (2026-04-21) and never restored; the eager imports made this package
# unimportable, including `load_public_picks`, which still exists. The two
# remaining call sites (pipeline orchestration and ev_analysis) already import
# bracket_optimizer inside try/except. Optional here for the same reason.
try:
    from .bracket_optimizer import (  # noqa: F401
        ESPNBracketOptimizer,
        ESPNOptimizationConfig,
        ESPNOptimizationResult,
    )

    __all__ += ["ESPNOptimizationConfig", "ESPNOptimizationResult", "ESPNBracketOptimizer"]
except ImportError:
    pass
try:
    from .leverage import LeverageSignal, compute_leverage_table  # noqa: F401

    __all__ += ["LeverageSignal", "compute_leverage_table"]
except ImportError:
    pass
try:
    from .mc_simulator import ESPNMCSimConfig, ESPNMCSimResult, ESPNMonteCarloSimulator  # noqa: F401

    __all__ += ["ESPNMCSimConfig", "ESPNMCSimResult", "ESPNMonteCarloSimulator"]
except ImportError:
    pass
