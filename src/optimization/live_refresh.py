"""Live public-pick refresh and re-optimization workflow.

Provides a lightweight "refresh and re-optimize" loop that re-fetches public
pick percentages and re-runs the EV optimization layer *without* re-training
models.  Designed for the 48-hour window before tournament tip-off when
public pick distributions shift significantly.

Architecture:
    Trained models (expensive, ~minutes) are frozen after the initial pipeline
    run.  Only the game-theory layer (leverage, bracket portfolio, pool
    simulation) is re-executed against updated public picks.  This takes
    seconds rather than minutes, enabling frequent re-optimization.

Usage:
    pipeline = SOTAPipeline(config)
    report = pipeline.run()  # Full pipeline: train + optimize

    workflow = LivePicksRefreshWorkflow.from_pipeline_run(pipeline, report)
    result = workflow.refresh()  # Re-fetch picks + re-optimize (seconds)

    if result.is_significant:
        print(result.change_summary)
        updated_report = result.updated_ev_report
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..data.scrapers.espn_picks import (
    ConsensusData,
    OrchestratorResult,
    PicksUpdate,
    RefreshablePicksManager,
    ScraperOrchestrator,
)
from ..optimization.leverage import (
    LeverageCalculator,
    LeveragePick,
    PoolAnalysis,
    TeamMetadata,
    analyze_pool,
    get_strategy_profile,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PickDelta:
    """Per-team change in pick percentages between two snapshots."""

    team_id: str
    team_name: str
    round_name: str
    old_pct: float
    new_pct: float

    @property
    def delta(self) -> float:
        return self.new_pct - self.old_pct

    @property
    def abs_delta(self) -> float:
        return abs(self.delta)


@dataclass
class LeverageShift:
    """Change in a leverage pick's status between two snapshots."""

    team_id: str
    team_name: str
    round_name: str
    old_leverage: float
    new_leverage: float
    old_ev_diff: float
    new_ev_diff: float

    @property
    def leverage_delta(self) -> float:
        return self.new_leverage - self.old_leverage

    @property
    def is_new_opportunity(self) -> bool:
        """Pick crossed from non-leverage to leverage territory."""
        return self.old_leverage < 1.5 and self.new_leverage >= 1.5

    @property
    def is_lost_opportunity(self) -> bool:
        """Pick fell below leverage threshold."""
        return self.old_leverage >= 1.5 and self.new_leverage < 1.5


@dataclass
class RefreshResult:
    """Result of a refresh-and-reoptimize cycle.

    Contains the comparison between the old and new EV analysis,
    highlighting which leverage picks and strategies changed.
    """

    # Timing
    refresh_timestamp: float = 0.0
    elapsed_seconds: float = 0.0

    # Pick data
    picks_update: Optional[PicksUpdate] = None
    pick_deltas: List[PickDelta] = field(default_factory=list)

    # Leverage analysis comparison
    leverage_shifts: List[LeverageShift] = field(default_factory=list)
    new_leverage_picks: List[Dict] = field(default_factory=list)
    lost_leverage_picks: List[Dict] = field(default_factory=list)

    # Strategy comparison
    old_strategy: str = ""
    new_strategy: str = ""
    strategy_changed: bool = False

    # Updated analysis
    old_ev_report: Optional[Dict] = None
    updated_ev_report: Optional[Dict] = None

    # Whether any re-optimization actually happened
    was_refreshed: bool = False
    skip_reason: Optional[str] = None

    @property
    def is_significant(self) -> bool:
        """True if the refresh produced material changes worth reviewing."""
        if not self.was_refreshed:
            return False
        if self.strategy_changed:
            return True
        if self.new_leverage_picks or self.lost_leverage_picks:
            return True
        if self.pick_deltas and max(d.abs_delta for d in self.pick_deltas) >= 2.0:
            return True
        return False

    @property
    def change_summary(self) -> str:
        """Human-readable summary of what changed."""
        if not self.was_refreshed:
            return f"No refresh performed: {self.skip_reason or 'data still fresh'}"

        lines = []
        lines.append(f"Refresh completed in {self.elapsed_seconds:.1f}s")

        if self.pick_deltas:
            top = sorted(self.pick_deltas, key=lambda d: d.abs_delta, reverse=True)[:5]
            lines.append(f"Top pick shifts ({len(self.pick_deltas)} total):")
            for d in top:
                direction = "+" if d.delta > 0 else ""
                lines.append(
                    f"  {d.team_name} {d.round_name}: "
                    f"{d.old_pct:.1f}% -> {d.new_pct:.1f}% ({direction}{d.delta:.1f}pp)"
                )

        if self.new_leverage_picks:
            names = [p["team_name"] for p in self.new_leverage_picks[:3]]
            lines.append(f"New leverage opportunities: {', '.join(names)}")
        if self.lost_leverage_picks:
            names = [p["team_name"] for p in self.lost_leverage_picks[:3]]
            lines.append(f"Lost leverage opportunities: {', '.join(names)}")

        if self.strategy_changed:
            lines.append(
                f"Strategy changed: {self.old_strategy} -> {self.new_strategy}"
            )

        if self.leverage_shifts:
            top_shifts = sorted(
                self.leverage_shifts, key=lambda s: abs(s.leverage_delta), reverse=True
            )[:3]
            lines.append("Biggest leverage ratio shifts:")
            for s in top_shifts:
                lines.append(
                    f"  {s.team_name} {s.round_name}: "
                    f"{s.old_leverage:.2f}x -> {s.new_leverage:.2f}x"
                )

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Serialize for JSON output."""
        d: Dict[str, Any] = {
            "refresh_timestamp": self.refresh_timestamp,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "was_refreshed": self.was_refreshed,
            "is_significant": self.is_significant,
        }
        if self.skip_reason:
            d["skip_reason"] = self.skip_reason
        if self.pick_deltas:
            d["pick_deltas"] = [
                {
                    "team_id": pd.team_id,
                    "team_name": pd.team_name,
                    "round": pd.round_name,
                    "old_pct": round(pd.old_pct, 2),
                    "new_pct": round(pd.new_pct, 2),
                    "delta": round(pd.delta, 2),
                }
                for pd in sorted(self.pick_deltas, key=lambda x: x.abs_delta, reverse=True)[:20]
            ]
        if self.new_leverage_picks:
            d["new_leverage_picks"] = self.new_leverage_picks
        if self.lost_leverage_picks:
            d["lost_leverage_picks"] = self.lost_leverage_picks
        if self.leverage_shifts:
            d["leverage_shifts"] = [
                {
                    "team_id": s.team_id,
                    "team_name": s.team_name,
                    "round": s.round_name,
                    "old_leverage": round(s.old_leverage, 3),
                    "new_leverage": round(s.new_leverage, 3),
                    "delta": round(s.leverage_delta, 3),
                }
                for s in sorted(self.leverage_shifts, key=lambda x: abs(x.leverage_delta), reverse=True)[:15]
            ]
        d["strategy_changed"] = self.strategy_changed
        if self.strategy_changed:
            d["old_strategy"] = self.old_strategy
            d["new_strategy"] = self.new_strategy
        if self.updated_ev_report:
            d["updated_ev_report"] = self.updated_ev_report
        return d


@dataclass
class RefreshSchedule:
    """Configuration for automated refresh scheduling.

    Pre-built schedules for common scenarios:
    - ``game_week``: Every 12 hours, 7 days before tip-off.
    - ``final_48h``: Every 4 hours in the last 48 hours.
    - ``crunch_time``: Every 1 hour in the last 6 hours.
    """

    interval_hours: float = 4.0
    significance_threshold_pp: float = 1.0  # Percentage-point threshold for re-optimization
    max_refreshes: int = 50
    force_reoptimize: bool = False  # Always re-optimize even if picks haven't changed
    # Exponential backoff on scraper failures
    max_retries: int = 4
    initial_backoff_seconds: float = 2.0
    # Convergence detection: stop early if picks stabilize
    convergence_window: int = 3  # Number of consecutive non-significant refreshes
    convergence_threshold_pp: float = 0.5  # Max delta to consider "converged"

    @classmethod
    def game_week(cls) -> RefreshSchedule:
        return cls(interval_hours=12.0, significance_threshold_pp=2.0, max_refreshes=14)

    @classmethod
    def final_48h(cls) -> RefreshSchedule:
        return cls(interval_hours=4.0, significance_threshold_pp=1.0, max_refreshes=12)

    @classmethod
    def crunch_time(cls) -> RefreshSchedule:
        return cls(interval_hours=1.0, significance_threshold_pp=0.5, max_refreshes=6)


# ---------------------------------------------------------------------------
# Core workflow
# ---------------------------------------------------------------------------


class LivePicksRefreshWorkflow:
    """Refresh public picks and re-optimize EV analysis without re-training models.

    Caches the expensive pipeline artifacts (model probabilities, team metadata,
    simulation data) and provides a cheap re-optimization path that only
    re-runs leverage analysis, bracket portfolio, and pool simulation.

    Args:
        model_round_probs: Per-team per-round model probabilities from MC simulation.
        team_metadata: TeamMetadata lookup keyed by team_id.
        ev_config: EV-mode configuration parameters.
        picks_manager: RefreshablePicksManager for fetching updated picks.
        initial_public_picks: The public picks used in the initial pipeline run.
        initial_ev_report: The EV report dict from the initial pipeline run.
        reoptimize_fn: Optional callback to run full EV analysis with new picks.
            Signature: ``(public_picks) -> EVReport.to_dict()``.
    """

    def __init__(
        self,
        model_round_probs: Dict[str, Dict[str, float]],
        team_metadata: Dict[str, TeamMetadata],
        ev_config: Dict[str, Any],
        picks_manager: RefreshablePicksManager,
        initial_public_picks: Dict[str, Dict[str, float]],
        initial_ev_report: Optional[Dict] = None,
        reoptimize_fn: Optional[Callable] = None,
    ):
        self._model_probs = model_round_probs
        self._team_metadata = team_metadata
        self._ev_config = ev_config
        self._picks_manager = picks_manager
        self._current_picks = initial_public_picks
        self._current_ev_report = initial_ev_report
        self._reoptimize_fn = reoptimize_fn
        self._refresh_history: List[RefreshResult] = []
        self._team_name_lookup: Dict[str, str] = {
            tid: meta.team_name for tid, meta in team_metadata.items()
        }

    @classmethod
    def from_pipeline_run(
        cls,
        pipeline: Any,
        report: Dict,
    ) -> LivePicksRefreshWorkflow:
        """Create a workflow from a completed SOTAPipeline run.

        Extracts model probabilities, team metadata, and EV configuration
        from the pipeline and its report, then wires up a
        ``RefreshablePicksManager`` for fetching updated picks.

        Args:
            pipeline: A ``SOTAPipeline`` instance that has already run.
            report: The report dict returned by ``pipeline.run()``.

        Returns:
            Configured LivePicksRefreshWorkflow ready for ``refresh()`` calls.
        """
        config = pipeline.config

        # Extract model round probabilities from simulation artifacts
        artifacts = report.get("artifacts", {})
        sim_data = artifacts.get("simulation", {})
        model_round_probs = pipeline._to_round_probabilities_from_sim(sim_data)

        # Build team metadata
        team_metadata = {
            tid: TeamMetadata(
                team_name=team.name, seed=team.seed, region=team.region,
            )
            for tid, team in pipeline.team_struct.items()
        }

        # Extract initial public picks and EV report
        initial_picks = pipeline._load_public_picks(model_round_probs)
        initial_ev = report.get("ev_analysis")

        # EV config bundle
        ev_config = {
            "pool_size": config.ev_pool_size,
            "scoring_system": config.ev_scoring_system,
            "target_percentile": config.ev_target_percentile,
            "contrarian_strength": config.ev_contrarian_strength,
            "payout_structure": config.ev_payout_structure,
            "enable_archetypes": config.ev_enable_archetypes,
            "pool_type": config.ev_pool_type,
            "archetype_overrides": config.ev_archetype_overrides,
        }

        # Create picks manager
        staleness_hours = 4.0  # Default: 4 hours before stale
        orchestrator = ScraperOrchestrator(cache_dir=config.data_cache_dir)
        picks_manager = RefreshablePicksManager(
            orchestrator, staleness_threshold_hours=staleness_hours,
        )
        # Seed the manager with the initial fetch so it tracks deltas
        picks_manager._last_result = OrchestratorResult(
            consensus=ConsensusData(sources=pipeline.public_pick_sources),
            successful_sources=pipeline.public_pick_sources,
            source_statuses={},
        )
        picks_manager._last_fetch_time = time.time()

        def _reoptimize(new_picks: Dict[str, Dict[str, float]]) -> Dict:
            """Re-run EV analysis with updated picks using the pipeline."""
            # Temporarily swap public picks
            old_json = config.public_picks_json
            config.public_picks_json = None
            config.scrape_live = False
            # We'll use the analyze_pool function directly
            return cls._run_lightweight_ev(
                model_round_probs, new_picks, team_metadata, ev_config,
            )

        return cls(
            model_round_probs=model_round_probs,
            team_metadata=team_metadata,
            ev_config=ev_config,
            picks_manager=picks_manager,
            initial_public_picks=initial_picks,
            initial_ev_report=initial_ev,
            reoptimize_fn=_reoptimize,
        )

    @staticmethod
    def _run_lightweight_ev(
        model_round_probs: Dict[str, Dict[str, float]],
        public_picks: Dict[str, Dict[str, float]],
        team_metadata: Dict[str, TeamMetadata],
        ev_config: Dict[str, Any],
    ) -> Dict:
        """Re-run EV leverage analysis with new picks (no model re-training).

        This is the fast path: only leverage analysis, bracket generation,
        and strategy recommendations are recomputed.  Takes seconds, not minutes.
        """
        pool_size = ev_config.get("pool_size", 100)
        scoring_system_name = ev_config.get("scoring_system", "standard")
        payout_structure = ev_config.get("payout_structure", "tiered")
        contrarian_strength = ev_config.get("contrarian_strength", 1.0)

        strategy_profile = get_strategy_profile(
            pool_size,
            scoring_system=scoring_system_name,
            contrarian_override=(
                contrarian_strength if contrarian_strength != 1.0 else None
            ),
            payout_structure=payout_structure,
        )

        pool_analysis = analyze_pool(
            pool_size,
            model_round_probs,
            public_picks,
            team_metadata=team_metadata,
            ev_scoring_system=scoring_system_name,
            strategy_profile=strategy_profile,
        )

        leverage_picks = [
            {
                "team_id": p.team_id,
                "team_name": p.team_name,
                "round": p.round_name,
                "model_probability": round(p.model_probability, 4),
                "public_pick_percentage": round(p.public_pick_percentage, 4),
                "leverage_ratio": round(p.leverage_ratio, 3),
                "ev_differential": round(p.expected_value_differential, 3),
            }
            for p in pool_analysis.leverage_picks[:20]
        ]

        fade_picks = [
            {
                "team_id": p.team_id,
                "team_name": p.team_name,
                "round": p.round_name,
                "model_probability": round(p.model_probability, 4),
                "public_pick_percentage": round(p.public_pick_percentage, 4),
                "leverage_ratio": round(p.leverage_ratio, 3),
            }
            for p in pool_analysis.fade_picks[:15]
        ]

        pareto_brackets = [
            b.to_dict() if hasattr(b, "to_dict") else {}
            for b in pool_analysis.pareto_brackets[:5]
        ]

        return {
            "mode": "ev",
            "pool_size": pool_size,
            "scoring_system": scoring_system_name,
            "recommended_strategy": pool_analysis.recommended_strategy,
            "leverage_picks": leverage_picks,
            "fade_picks": fade_picks,
            "pareto_brackets": pareto_brackets,
            "pool_ev_analysis": pool_analysis.strategy_evs,
            "refresh_timestamp": time.time(),
        }

    def refresh(
        self,
        force: bool = False,
        significance_threshold_pp: float = 1.0,
        new_picks_json: Optional[str] = None,
    ) -> RefreshResult:
        """Fetch updated public picks and re-optimize if changes are significant.

        Args:
            force: Force re-optimization even if picks haven't changed.
            significance_threshold_pp: Minimum percentage-point change in any
                championship pick to trigger re-optimization.
            new_picks_json: Optional path to a JSON file with updated picks.
                If provided, loads from file instead of scraping.

        Returns:
            RefreshResult with comparison of old vs new analysis.
        """
        start_time = time.monotonic()
        result = RefreshResult(refresh_timestamp=time.time())

        # Fetch new picks
        new_picks = self._fetch_new_picks(new_picks_json)
        if new_picks is None:
            result.skip_reason = "Failed to fetch updated pick data"
            result.elapsed_seconds = time.monotonic() - start_time
            self._refresh_history.append(result)
            return result

        # Compute per-round deltas
        pick_deltas = self._compute_round_deltas(self._current_picks, new_picks)
        result.pick_deltas = pick_deltas

        # Check significance
        max_champ_delta = 0.0
        for d in pick_deltas:
            if d.round_name == "CHAMP":
                max_champ_delta = max(max_champ_delta, d.abs_delta)

        max_any_delta = max((d.abs_delta for d in pick_deltas), default=0.0)

        should_reoptimize = (
            force
            or max_champ_delta >= significance_threshold_pp
            or max_any_delta >= significance_threshold_pp * 2  # Late-round shifts matter more
        )

        if not should_reoptimize:
            result.skip_reason = (
                f"Changes below threshold (max championship delta: "
                f"{max_champ_delta:.1f}pp, threshold: {significance_threshold_pp:.1f}pp)"
            )
            result.elapsed_seconds = time.monotonic() - start_time
            self._refresh_history.append(result)
            return result

        # Re-optimize
        result.was_refreshed = True
        result.old_ev_report = copy.deepcopy(self._current_ev_report)
        result.old_strategy = (
            self._current_ev_report.get("recommended_strategy", "")
            if self._current_ev_report
            else ""
        )

        if self._reoptimize_fn:
            new_ev_report = self._reoptimize_fn(new_picks)
        else:
            new_ev_report = self._run_lightweight_ev(
                self._model_probs, new_picks, self._team_metadata, self._ev_config,
            )

        result.updated_ev_report = new_ev_report
        result.new_strategy = new_ev_report.get("recommended_strategy", "")
        result.strategy_changed = result.old_strategy != result.new_strategy

        # Compute leverage shifts
        result.leverage_shifts = self._compute_leverage_shifts(
            self._current_ev_report, new_ev_report,
        )
        result.new_leverage_picks = self._find_new_leverage_picks(
            self._current_ev_report, new_ev_report,
        )
        result.lost_leverage_picks = self._find_lost_leverage_picks(
            self._current_ev_report, new_ev_report,
        )

        # Update state
        self._current_picks = new_picks
        self._current_ev_report = new_ev_report

        result.elapsed_seconds = time.monotonic() - start_time
        self._refresh_history.append(result)

        logger.info("EV refresh completed: %s", result.change_summary)
        return result

    def refresh_from_json(self, json_path: str, force: bool = False) -> RefreshResult:
        """Convenience: refresh from a JSON file of updated picks.

        Args:
            json_path: Path to JSON file with updated pick percentages.
            force: Force re-optimization.

        Returns:
            RefreshResult with comparison.
        """
        return self.refresh(force=force, new_picks_json=json_path)

    def run_scheduled(
        self,
        schedule: RefreshSchedule,
        callback: Optional[Callable[[RefreshResult], None]] = None,
    ) -> List[RefreshResult]:
        """Run a series of refresh cycles according to a schedule.

        Includes convergence detection: if the last ``convergence_window``
        consecutive refreshes all have max delta below ``convergence_threshold_pp``,
        the loop terminates early.  This avoids wasting API calls when
        public picks have stabilized (common >24h before tip-off).

        This is a blocking loop that sleeps between refreshes.  For production
        use, prefer calling ``refresh()`` from an external scheduler (cron,
        Airflow, etc.).

        Args:
            schedule: Refresh schedule configuration.
            callback: Optional callback invoked after each refresh with the result.

        Returns:
            List of all RefreshResults from the scheduled run.
        """
        results: List[RefreshResult] = []
        interval_seconds = schedule.interval_hours * 3600
        consecutive_stable = 0

        for i in range(schedule.max_refreshes):
            if i > 0:
                logger.info(
                    "Scheduled refresh %d/%d: sleeping %.0f seconds",
                    i + 1, schedule.max_refreshes, interval_seconds,
                )
                time.sleep(interval_seconds)

            result = self.refresh(
                force=schedule.force_reoptimize,
                significance_threshold_pp=schedule.significance_threshold_pp,
            )
            results.append(result)

            if callback:
                callback(result)

            if result.is_significant:
                consecutive_stable = 0
                logger.info(
                    "Significant changes detected on refresh %d: %s",
                    i + 1, result.change_summary,
                )
            else:
                # Check convergence: is max delta below convergence threshold?
                max_delta = 0.0
                if result.pick_deltas:
                    max_delta = max(d.abs_delta for d in result.pick_deltas)

                if max_delta <= schedule.convergence_threshold_pp:
                    consecutive_stable += 1
                else:
                    consecutive_stable = 0

                if consecutive_stable >= schedule.convergence_window:
                    logger.info(
                        "Convergence detected: %d consecutive stable refreshes "
                        "(max delta %.1fpp <= threshold %.1fpp). Stopping early.",
                        consecutive_stable, max_delta,
                        schedule.convergence_threshold_pp,
                    )
                    break

        return results

    def detect_convergence(self, window: int = 3) -> Dict[str, Any]:
        """Analyze recent refresh history to detect pick convergence.

        When public picks stop shifting significantly across multiple
        consecutive refreshes, the market has "converged" and further
        refreshes are unlikely to change the optimal strategy.

        Args:
            window: Number of recent refreshes to analyze.

        Returns:
            Dict with convergence analysis:
            - ``converged``: True if picks have stabilized.
            - ``max_recent_delta``: Largest pick shift in the window.
            - ``trend``: "stabilizing", "volatile", or "insufficient_data".
            - ``recommended_action``: "lock_bracket", "continue_monitoring",
              or "re-optimize".
        """
        recent = self._refresh_history[-window:]
        if len(recent) < 2:
            return {
                "converged": False,
                "max_recent_delta": 0.0,
                "trend": "insufficient_data",
                "recommended_action": "continue_monitoring",
            }

        # Collect max deltas from each recent refresh
        max_deltas = []
        for r in recent:
            if r.pick_deltas:
                max_deltas.append(max(d.abs_delta for d in r.pick_deltas))
            else:
                max_deltas.append(0.0)

        max_recent = max(max_deltas) if max_deltas else 0.0
        avg_delta = sum(max_deltas) / len(max_deltas)

        # Check if deltas are decreasing (stabilizing)
        is_decreasing = all(
            max_deltas[i] >= max_deltas[i + 1] - 0.1
            for i in range(len(max_deltas) - 1)
        )

        converged = avg_delta < 1.0 and max_recent < 2.0
        if converged:
            trend = "stabilizing"
            action = "lock_bracket"
        elif is_decreasing and avg_delta < 3.0:
            trend = "stabilizing"
            action = "continue_monitoring"
        else:
            trend = "volatile"
            action = "re-optimize"

        return {
            "converged": converged,
            "max_recent_delta": round(max_recent, 2),
            "avg_delta": round(avg_delta, 2),
            "trend": trend,
            "recommended_action": action,
            "refreshes_analyzed": len(recent),
        }

    @property
    def refresh_history(self) -> List[RefreshResult]:
        """All refresh results from this workflow's lifetime."""
        return list(self._refresh_history)

    @property
    def current_picks(self) -> Dict[str, Dict[str, float]]:
        """The current (most recently refreshed) public pick data."""
        return dict(self._current_picks)

    @property
    def current_ev_report(self) -> Optional[Dict]:
        """The current (most recently reoptimized) EV report."""
        return self._current_ev_report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_new_picks(
        self,
        json_path: Optional[str] = None,
        max_retries: int = 4,
        initial_backoff: float = 2.0,
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """Fetch updated picks from file or scraper with exponential backoff.

        On scraper failures, retries up to ``max_retries`` times with
        exponential backoff (2s, 4s, 8s, 16s by default).  This handles
        transient ESPN/CBS/Yahoo scraper failures that are common in the
        48 hours before tip-off when traffic spikes.

        Args:
            json_path: If provided, loads from file (no retries needed).
            max_retries: Maximum number of retry attempts.
            initial_backoff: Initial backoff delay in seconds (doubles each retry).

        Returns:
            Updated pick data, or None if all attempts failed.
        """
        if json_path:
            return self._load_picks_from_json(json_path)

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                orch_result, update = self._picks_manager.fetch_or_refresh(
                    force_refresh=True,
                )
                if not orch_result.consensus.teams:
                    logger.warning("Refresh returned empty pick data (attempt %d)", attempt + 1)
                    last_error = "empty pick data"
                    if attempt < max_retries:
                        backoff = initial_backoff * (2 ** attempt)
                        time.sleep(backoff)
                    continue

                picks: Dict[str, Dict[str, float]] = {}
                for team_id, pp in orch_result.consensus.teams.items():
                    row = pp.as_dict
                    picks[team_id] = {
                        rnd: v / 100.0 if v > 1.0 else v
                        for rnd, v in row.items()
                    }
                return picks
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries:
                    backoff = initial_backoff * (2 ** attempt)
                    logger.warning(
                        "Scraper attempt %d/%d failed: %s. Retrying in %.0fs",
                        attempt + 1, max_retries + 1, e, backoff,
                    )
                    time.sleep(backoff)

        logger.error(
            "All %d fetch attempts failed. Last error: %s",
            max_retries + 1, last_error,
        )
        return None

    def _load_picks_from_json(
        self, json_path: str,
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """Load pick data from a JSON file."""
        try:
            with open(json_path, "r") as f:
                payload = json.load(f)

            teams = payload.get("teams", payload)
            picks: Dict[str, Dict[str, float]] = {}
            for team_id, team_data in teams.items():
                if not isinstance(team_data, dict):
                    continue
                row: Dict[str, float] = {}
                # Support both short keys (R64) and long keys (round_of_64_pct)
                key_map = {
                    "R64": ["R64", "round_of_64_pct"],
                    "R32": ["R32", "round_of_32_pct"],
                    "S16": ["S16", "sweet_16_pct"],
                    "E8": ["E8", "elite_8_pct"],
                    "F4": ["F4", "final_four_pct"],
                    "CHAMP": ["CHAMP", "champion_pct"],
                }
                for rnd, keys in key_map.items():
                    for k in keys:
                        if k in team_data:
                            v = float(team_data[k])
                            row[rnd] = v / 100.0 if v > 1.0 else v
                            break
                    else:
                        row[rnd] = 0.0
                picks[team_id] = row
            return picks if picks else None
        except Exception as e:
            logger.error("Failed to load picks from %s: %s", json_path, e)
            return None

    def _compute_round_deltas(
        self,
        old_picks: Dict[str, Dict[str, float]],
        new_picks: Dict[str, Dict[str, float]],
    ) -> List[PickDelta]:
        """Compute per-team per-round deltas between two pick snapshots."""
        deltas: List[PickDelta] = []
        all_teams = set(old_picks.keys()) | set(new_picks.keys())
        rounds = ["R64", "R32", "S16", "E8", "F4", "CHAMP"]

        for team_id in all_teams:
            old_row = old_picks.get(team_id, {})
            new_row = new_picks.get(team_id, {})
            name = self._team_name_lookup.get(team_id, team_id)

            for rnd in rounds:
                old_v = old_row.get(rnd, 0.0)
                new_v = new_row.get(rnd, 0.0)
                if abs(new_v - old_v) > 1e-6:
                    deltas.append(PickDelta(
                        team_id=team_id,
                        team_name=name,
                        round_name=rnd,
                        old_pct=old_v * 100.0,  # Convert back to % for readability
                        new_pct=new_v * 100.0,
                    ))

        return sorted(deltas, key=lambda d: d.abs_delta, reverse=True)

    def _compute_leverage_shifts(
        self,
        old_ev: Optional[Dict],
        new_ev: Dict,
    ) -> List[LeverageShift]:
        """Compare leverage picks between old and new EV reports."""
        if not old_ev:
            return []

        old_lev = {
            (p["team_id"], p["round"]): p
            for p in old_ev.get("leverage_picks", [])
        }
        new_lev = {
            (p["team_id"], p["round"]): p
            for p in new_ev.get("leverage_picks", [])
        }

        shifts: List[LeverageShift] = []
        all_keys = set(old_lev.keys()) | set(new_lev.keys())

        for key in all_keys:
            old_p = old_lev.get(key, {})
            new_p = new_lev.get(key, {})
            old_lr = old_p.get("leverage_ratio", 0.0)
            new_lr = new_p.get("leverage_ratio", 0.0)

            if abs(new_lr - old_lr) > 0.01:
                team_id = key[0]
                shifts.append(LeverageShift(
                    team_id=team_id,
                    team_name=self._team_name_lookup.get(team_id, team_id),
                    round_name=key[1],
                    old_leverage=old_lr,
                    new_leverage=new_lr,
                    old_ev_diff=old_p.get("ev_differential", 0.0),
                    new_ev_diff=new_p.get("ev_differential", 0.0),
                ))

        return sorted(shifts, key=lambda s: abs(s.leverage_delta), reverse=True)

    def _find_new_leverage_picks(
        self,
        old_ev: Optional[Dict],
        new_ev: Dict,
    ) -> List[Dict]:
        """Find leverage picks that appear in new but not old analysis."""
        old_keys = set()
        if old_ev:
            old_keys = {
                (p["team_id"], p["round"]) for p in old_ev.get("leverage_picks", [])
            }

        return [
            p for p in new_ev.get("leverage_picks", [])
            if (p["team_id"], p["round"]) not in old_keys
        ]

    def _find_lost_leverage_picks(
        self,
        old_ev: Optional[Dict],
        new_ev: Dict,
    ) -> List[Dict]:
        """Find leverage picks that were in old but not new analysis."""
        if not old_ev:
            return []
        new_keys = {
            (p["team_id"], p["round"]) for p in new_ev.get("leverage_picks", [])
        }
        return [
            p for p in old_ev.get("leverage_picks", [])
            if (p["team_id"], p["round"]) not in new_keys
        ]
