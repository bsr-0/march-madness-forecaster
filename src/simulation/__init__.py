"""Simulation and bracket generation."""

try:
    from .competition_simulation import (
        CompetitionSimulator,
        DualSubmissionGenerator,
        MatchupPrediction,
        SimulationResult,
    )

    __all__ = [
        "CompetitionSimulator",
        "DualSubmissionGenerator",
        "MatchupPrediction",
        "SimulationResult",
    ]
except ImportError:
    __all__ = []
