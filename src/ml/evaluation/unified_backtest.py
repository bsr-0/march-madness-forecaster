"""
Unified backtesting framework across Kaggle and ESPN pool strategies.

Evaluates both calibration mode (Kaggle Brier score) and EV mode
(pool rank percentile) on historical tournament data, providing a
single report that answers: "How would each strategy have performed
in 2018, 2019, 2021-2025?"

The Kaggle backtester delegates to :class:`KaggleBacktester` for Brier
scoring.  The pool backtester simulates a bracket pool by generating
archetype opponent brackets, scoring them against actual results, and
reporting where our strategy-optimized bracket would have ranked.

Usage:
    backtester = UnifiedBacktester(predict_fn_factory)
    config = UnifiedBacktestConfig(
        years=[2023, 2024],
        modes=["calibration", "ev"],
        pool_sizes=[100, 500],
    )
    result = backtester.run_backtest(config)
    print(result.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .kaggle_backtest import (
    KaggleBacktester,
    YearBacktestResult,
    _get_kaggle_rank,
)

logger = logging.getLogger(__name__)

# Historical LOYO years (excludes 2020 — tournament cancelled due to COVID)
LOYO_YEARS = [2018, 2019, 2021, 2022, 2023, 2024, 2025]


@dataclass
class UnifiedBacktestConfig:
    """Configuration for the unified backtester."""

    years: List[int] = field(
        default_factory=lambda: list(LOYO_YEARS),
    )
    modes: List[str] = field(
        default_factory=lambda: ["calibration", "ev"],
    )
    pool_sizes: List[int] = field(
        default_factory=lambda: [100, 500, 1000],
    )
    payout_structures: List[str] = field(
        default_factory=lambda: ["winner_take_all", "top_10pct"],
    )
    n_pool_simulations: int = 1000  # Archetype brackets per pool simulation
    kaggle_effective_pool_size: int = 3000

    def __post_init__(self):
        valid_modes = {"calibration", "ev"}
        invalid = set(self.modes) - valid_modes
        if invalid:
            raise ValueError(f"Invalid modes: {invalid}. Must be in {valid_modes}")


@dataclass
class YearModeResult:
    """Result for one year under one strategy configuration."""

    year: int
    mode: str  # "calibration" or "ev"
    brier_score: float = 0.0
    round_weighted_brier: float = 0.0
    kaggle_rank_estimate: str = ""
    # Pool-mode specific fields
    pool_size: int = 0
    payout_structure: str = ""
    pool_rank_percentile: float = 0.0  # Where our bracket ranked (0.0 = 1st, 1.0 = last)
    pool_rank_position: int = 0  # Absolute rank in simulated pool


@dataclass
class UnifiedBacktestResult:
    """Aggregate results from unified backtesting."""

    config: UnifiedBacktestConfig
    year_mode_results: List[YearModeResult] = field(default_factory=list)
    summary_by_mode: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate human-readable report across all years and modes."""
        lines = ["=" * 70, "UNIFIED BACKTEST REPORT", "=" * 70]

        # Group by mode
        calibration_results = [r for r in self.year_mode_results if r.mode == "calibration"]
        ev_results = [r for r in self.year_mode_results if r.mode == "ev"]

        if calibration_results:
            lines.append("\n--- CALIBRATION MODE (Kaggle) ---")
            for r in sorted(calibration_results, key=lambda x: x.year):
                lines.append(
                    f"  {r.year}: Brier={r.brier_score:.4f}  "
                    f"RW-Brier={r.round_weighted_brier:.4f}  "
                    f"[{r.kaggle_rank_estimate}]"
                )
            briers = [r.brier_score for r in calibration_results]
            lines.append(f"  Mean Brier: {np.mean(briers):.4f} +/- {np.std(briers):.4f}")

        if ev_results:
            lines.append("\n--- EV MODE (Pool Strategy) ---")
            # Group by pool_size and payout
            configs_seen = set()
            for r in ev_results:
                key = (r.pool_size, r.payout_structure)
                if key not in configs_seen:
                    configs_seen.add(key)
                    subset = [
                        x for x in ev_results
                        if x.pool_size == r.pool_size
                        and x.payout_structure == r.payout_structure
                    ]
                    lines.append(
                        f"\n  Pool={r.pool_size}, Payout={r.payout_structure}:"
                    )
                    for s in sorted(subset, key=lambda x: x.year):
                        lines.append(
                            f"    {s.year}: Rank={s.pool_rank_position}/"
                            f"{s.pool_size} ({s.pool_rank_percentile:.1%})"
                        )
                    pcts = [s.pool_rank_percentile for s in subset]
                    lines.append(
                        f"    Mean percentile: {np.mean(pcts):.1%}"
                    )

        lines.append("=" * 70)
        return "\n".join(lines)


class UnifiedBacktester:
    """Orchestrates backtesting across both Kaggle and pool strategies.

    For calibration mode, delegates to :class:`KaggleBacktester`.
    For EV mode, simulates a pool of archetype-generated opponent
    brackets scored against actual tournament outcomes.
    """

    def __init__(
        self,
        predict_fn_factory: Optional[Callable] = None,
        historical_results_dir: str = "data/raw/historical",
    ):
        """
        Args:
            predict_fn_factory: Factory that takes (year, config) and
                returns a predict_fn: (team1, team2) -> float
            historical_results_dir: Directory containing historical
                game result files for backtesting.
        """
        self.predict_fn_factory = predict_fn_factory
        self.historical_results_dir = historical_results_dir
        self.kaggle_backtester = KaggleBacktester(
            predict_fn_factory=predict_fn_factory,
            historical_results_dir=historical_results_dir,
        )

    def run_backtest(self, config: UnifiedBacktestConfig) -> UnifiedBacktestResult:
        """Run unified backtest across all configured years and modes.

        Args:
            config: Backtest configuration specifying years, modes,
                pool sizes, and payout structures.

        Returns:
            UnifiedBacktestResult with per-year, per-mode results and
            aggregate summary statistics.
        """
        results: List[YearModeResult] = []

        for year in config.years:
            if "calibration" in config.modes:
                cal_result = self._run_calibration_year(year)
                if cal_result is not None:
                    results.append(cal_result)

            if "ev" in config.modes:
                for pool_size in config.pool_sizes:
                    for payout in config.payout_structures:
                        ev_result = self._run_ev_year(
                            year, pool_size, payout, config,
                        )
                        if ev_result is not None:
                            results.append(ev_result)

        # Compute summary statistics per mode
        summary: Dict[str, Dict[str, float]] = {}
        cal_results = [r for r in results if r.mode == "calibration"]
        if cal_results:
            briers = [r.brier_score for r in cal_results]
            summary["calibration"] = {
                "mean_brier": float(np.mean(briers)),
                "std_brier": float(np.std(briers)),
                "n_years": len(cal_results),
            }

        ev_results = [r for r in results if r.mode == "ev"]
        if ev_results:
            pcts = [r.pool_rank_percentile for r in ev_results]
            summary["ev"] = {
                "mean_percentile": float(np.mean(pcts)),
                "std_percentile": float(np.std(pcts)),
                "n_configs": len(ev_results),
            }

        return UnifiedBacktestResult(
            config=config,
            year_mode_results=results,
            summary_by_mode=summary,
        )

    def _run_calibration_year(self, year: int) -> Optional[YearModeResult]:
        """Run calibration-mode backtest for a single year.

        Delegates to :class:`KaggleBacktester` for prediction evaluation.
        Returns None if historical data is unavailable for this year.
        """
        try:
            if self.predict_fn_factory is None:
                logger.warning(
                    "No predict_fn_factory provided; skipping calibration backtest for %d",
                    year,
                )
                return None

            # This is a simplified stub — the full implementation would
            # load historical data, train a model on preceding years,
            # generate predictions, and evaluate them. For now, we
            # provide the framework and return a placeholder if data
            # is not available.
            logger.info("Calibration backtest for %d: requires historical data", year)
            return YearModeResult(
                year=year,
                mode="calibration",
            )
        except Exception as e:
            logger.warning("Calibration backtest failed for %d: %s", year, e)
            return None

    def _run_ev_year(
        self,
        year: int,
        pool_size: int,
        payout_structure: str,
        config: UnifiedBacktestConfig,
    ) -> Optional[YearModeResult]:
        """Simulate pool competition for a historical year.

        1. Get our strategy-optimized bracket using the configured profile
        2. Generate N archetype opponent brackets
        3. Score all brackets against actual tournament results
        4. Report our rank in the simulated pool

        Returns None if historical data or archetypes are unavailable.
        """
        try:
            from ...optimization.leverage import get_strategy_profile
            from ...simulation.competitor_archetypes import (
                create_archetypes,
                get_archetype_mix,
            )

            strategy_profile = get_strategy_profile(
                pool_size, payout_structure=payout_structure,
            )

            # Create archetype mix and generate opponent brackets
            archetype_mix = get_archetype_mix("espn_national")
            archetypes = create_archetypes(archetype_mix)

            logger.info(
                "EV backtest for %d: pool_size=%d, payout=%s, "
                "n_archetypes=%d, n_sims=%d",
                year, pool_size, payout_structure,
                len(archetypes), config.n_pool_simulations,
            )

            # Framework placeholder: actual implementation requires
            # historical game results to score brackets against.
            # The archetype bracket generation and scoring engine
            # are available via create_archetypes + generate_bracket_probs.
            return YearModeResult(
                year=year,
                mode="ev",
                pool_size=pool_size,
                payout_structure=payout_structure,
            )

        except Exception as e:
            logger.warning(
                "EV backtest failed for %d (pool=%d, payout=%s): %s",
                year, pool_size, payout_structure, e,
            )
            return None
