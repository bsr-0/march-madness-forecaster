"""PoolOptimizer — the downstream strategy layer.

Architectural contract:
    - Ingests READ-ONLY win probabilities P from ForecastEngine.
    - Never modifies game-level probabilities.
    - All adjustments (leverage, contrarian, portfolio) happen here.
    - Every output includes an AssumptionsManifest recording the
      environmental parameters that shaped the recommendation.
    - Sensitivity analysis flags High Strategy Uncertainty when a
      ±5% shift in public sentiment alters the optimal bracket.

This module is the sole consumer of pool_size, scoring_rules,
payout_structure, and public_pick_distribution.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PoolEnvironment:
    """The four mandatory environmental parameters for pool optimization.

    Every bracket recommendation is conditional on these variables.
    No recommendation is valid without recording them.
    """

    pool_size: int
    scoring_rules: Dict[str, int]  # Round -> points mapping, e.g. {"R64": 10, ...}
    payout_structure: str  # "winner_take_all", "top_3", "top_10pct", "top_25pct", "tiered"
    public_pick_distribution: Dict[str, Dict[str, float]]  # team_id -> {round: pick_pct}

    def validate(self) -> None:
        """Validate that all required fields are non-null and non-empty."""
        if self.pool_size <= 0:
            raise ValueError(f"pool_size must be positive, got {self.pool_size}")
        if not self.scoring_rules:
            raise ValueError("scoring_rules must be a non-empty dict")
        if not self.payout_structure:
            raise ValueError("payout_structure must be a non-empty string")
        if not self.public_pick_distribution:
            raise ValueError("public_pick_distribution must be a non-empty dict")


@dataclass
class AssumptionsManifest:
    """Mandatory metadata attached to every bracket recommendation.

    Records the environmental parameters and sensitivity analysis results
    so that no recommendation is presented without context.
    """

    pool_size: int
    expected_chalk: Dict[str, float]  # team_id -> public championship pick %
    payout_structure: str
    scoring_rules: Dict[str, int]
    sensitivity_flag: str = "NOT_EVALUATED"  # "STABLE", "HIGH_STRATEGY_UNCERTAINTY", "NOT_EVALUATED"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pool_size": self.pool_size,
            "expected_chalk": self.expected_chalk,
            "payout_structure": self.payout_structure,
            "scoring_rules": self.scoring_rules,
            "sensitivity_flag": self.sensitivity_flag,
            "timestamp": self.timestamp,
        }


@dataclass
class SensitivityReport:
    """Result of ±shift% stress test on public pick assumptions.

    If shifting public sentiment by ±shift_pct changes the optimal
    champion or ≥2 Final Four picks, the recommendation is flagged
    as HIGH_STRATEGY_UNCERTAINTY.
    """

    shift_pct: float
    baseline_champion: str
    shifted_up_champion: str
    shifted_down_champion: str
    baseline_final_four: List[str]
    shifted_up_final_four: List[str]
    shifted_down_final_four: List[str]
    is_stable: bool
    flag: str  # "STABLE" | "HIGH_STRATEGY_UNCERTAINTY"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shift_pct": self.shift_pct,
            "baseline_champion": self.baseline_champion,
            "shifted_up_champion": self.shifted_up_champion,
            "shifted_down_champion": self.shifted_down_champion,
            "baseline_final_four": self.baseline_final_four,
            "shifted_up_final_four": self.shifted_up_final_four,
            "shifted_down_final_four": self.shifted_down_final_four,
            "is_stable": self.is_stable,
            "flag": self.flag,
        }


@dataclass
class PoolResult:
    """Complete output from PoolOptimizer.optimize().

    Always includes the AssumptionsManifest — no recommendation is
    valid without it.
    """

    manifest: AssumptionsManifest
    recommended_strategy: str = "balanced"
    leverage_picks: List[Dict] = field(default_factory=list)
    fade_picks: List[Dict] = field(default_factory=list)
    pareto_brackets: List[Any] = field(default_factory=list)
    sensitivity_report: Optional[SensitivityReport] = None
    strategy_evs: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "assumptions_manifest": self.manifest.to_dict(),
            "recommended_strategy": self.recommended_strategy,
            "leverage_picks": self.leverage_picks,
            "fade_picks": self.fade_picks,
            "n_pareto_brackets": len(self.pareto_brackets),
            "strategy_evs": self.strategy_evs,
        }
        if self.sensitivity_report is not None:
            result["sensitivity_report"] = self.sensitivity_report.to_dict()
        return result


class PoolOptimizer:
    """Downstream optimizer that maximizes EV and tournament-long ROI.

    Ingests read-only probabilities P from the ForecastEngine and applies
    Monte Carlo simulations, leverage (contrarian logic), and portfolio
    generation conditional on environmental variables.

    Architectural invariant: the probability dict is deep-copied at
    construction time.  The optimizer NEVER modifies game-level
    probabilities — all adjustments happen in strategy space.

    Typical usage::

        probs = engine.predict_all_matchups(team_ids)
        mc_round_probs = to_round_probabilities(pipeline, sim_results)
        env = PoolEnvironment(pool_size=500, scoring_rules=..., ...)
        optimizer = PoolOptimizer(probs, env, model_round_probs=mc_round_probs)
        result = optimizer.optimize()
        print(result.manifest)  # Non-null AssumptionsManifest

    Args:
        probabilities: Deep-copied dict of (team1, team2) -> P(team1 wins).
        environment: Pool environmental parameters.
        model_round_probs: Pre-computed per-team per-round advancement
            probabilities from Monte Carlo simulation.  When provided,
            bypasses the heuristic fallback.
    """

    def __init__(
        self,
        probabilities: Dict[Tuple[str, str], float],
        environment: PoolEnvironment,
        model_round_probs: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        environment.validate()
        # Deep-copy: mutations of the original dict are invisible.
        self._probabilities: Dict[Tuple[str, str], float] = copy.deepcopy(probabilities)
        self._environment = environment
        self._frozen_prob_hash = self._hash_probabilities(self._probabilities)
        # When MC-derived round advancement probabilities are available,
        # use them directly instead of the heuristic approximation.
        self._mc_round_probs: Optional[Dict[str, Dict[str, float]]] = (
            copy.deepcopy(model_round_probs) if model_round_probs is not None else None
        )

    @property
    def probabilities(self) -> Dict[Tuple[str, str], float]:
        """Read-only access to the frozen probability dict."""
        return copy.deepcopy(self._probabilities)

    @property
    def environment(self) -> PoolEnvironment:
        return self._environment

    def optimize(self) -> PoolResult:
        """Run full pool optimization: leverage + Pareto + portfolio.

        Returns:
            PoolResult with AssumptionsManifest, leverage picks,
            Pareto brackets, and strategy recommendations.
        """
        # Verify probabilities haven't been tampered with
        current_hash = self._hash_probabilities(self._probabilities)
        if current_hash != self._frozen_prob_hash:
            raise RuntimeError(
                "INTEGRITY VIOLATION: Probability dict was mutated after "
                "PoolOptimizer construction. This violates the architectural "
                "firewall between forecasting and optimization."
            )

        env = self._environment

        # Build expected chalk (championship pick distribution)
        expected_chalk = self._extract_championship_picks(env.public_pick_distribution)

        # Use MC-derived round probabilities when available; fall back to heuristic.
        if self._mc_round_probs is not None:
            model_round_probs = self._mc_round_probs
        else:
            model_round_probs = self._build_round_probabilities()

        # Run leverage analysis via existing infrastructure
        leverage_picks, fade_picks, pareto_brackets, strategy_evs, recommended = self._run_leverage_analysis(
            model_round_probs, env
        )

        # Build the mandatory AssumptionsManifest
        manifest = AssumptionsManifest(
            pool_size=env.pool_size,
            expected_chalk=expected_chalk,
            payout_structure=env.payout_structure,
            scoring_rules=env.scoring_rules,
        )

        return PoolResult(
            manifest=manifest,
            recommended_strategy=recommended,
            leverage_picks=leverage_picks,
            fade_picks=fade_picks,
            pareto_brackets=pareto_brackets,
            strategy_evs=strategy_evs,
        )

    def sensitivity_analysis(
        self,
        pick_shift_pct: float = 0.05,
    ) -> SensitivityReport:
        """Stress-test public pick assumptions with ±shift%.

        If shifting public sentiment by ±pick_shift_pct changes the
        recommended champion or ≥2 Final Four picks, the result is
        flagged as HIGH_STRATEGY_UNCERTAINTY.

        Args:
            pick_shift_pct: Fractional shift to apply (default 0.05 = ±5%).

        Returns:
            SensitivityReport with stability flag.
        """
        env = self._environment

        # Baseline optimization
        baseline_result = self.optimize()
        baseline_champion = self._extract_champion(baseline_result)
        baseline_f4 = self._extract_final_four(baseline_result)

        # Shifted UP: public picks increase by shift_pct
        shifted_up_picks = self._shift_picks(env.public_pick_distribution, +pick_shift_pct)
        shifted_up_env = PoolEnvironment(
            pool_size=env.pool_size,
            scoring_rules=env.scoring_rules,
            payout_structure=env.payout_structure,
            public_pick_distribution=shifted_up_picks,
        )
        shifted_up_opt = PoolOptimizer(self._probabilities, shifted_up_env, self._mc_round_probs)
        shifted_up_result = shifted_up_opt.optimize()
        shifted_up_champion = self._extract_champion(shifted_up_result)
        shifted_up_f4 = self._extract_final_four(shifted_up_result)

        # Shifted DOWN: public picks decrease by shift_pct
        shifted_down_picks = self._shift_picks(env.public_pick_distribution, -pick_shift_pct)
        shifted_down_env = PoolEnvironment(
            pool_size=env.pool_size,
            scoring_rules=env.scoring_rules,
            payout_structure=env.payout_structure,
            public_pick_distribution=shifted_down_picks,
        )
        shifted_down_opt = PoolOptimizer(self._probabilities, shifted_down_env, self._mc_round_probs)
        shifted_down_result = shifted_down_opt.optimize()
        shifted_down_champion = self._extract_champion(shifted_down_result)
        shifted_down_f4 = self._extract_final_four(shifted_down_result)

        # Stability check: champion changed or ≥2 F4 picks differ
        champion_changed = shifted_up_champion != baseline_champion or shifted_down_champion != baseline_champion
        f4_diff_up = len(set(baseline_f4) - set(shifted_up_f4))
        f4_diff_down = len(set(baseline_f4) - set(shifted_down_f4))
        f4_unstable = f4_diff_up >= 2 or f4_diff_down >= 2

        is_stable = not (champion_changed or f4_unstable)
        flag = "STABLE" if is_stable else "HIGH_STRATEGY_UNCERTAINTY"

        if not is_stable:
            logger.warning(
                "SENSITIVITY: ±%.0f%% public pick shift produces unstable "
                "recommendations. Champion changed: %s. F4 diff: up=%d, down=%d. "
                "Flag: %s",
                pick_shift_pct * 100,
                champion_changed,
                f4_diff_up,
                f4_diff_down,
                flag,
            )

        return SensitivityReport(
            shift_pct=pick_shift_pct,
            baseline_champion=baseline_champion,
            shifted_up_champion=shifted_up_champion,
            shifted_down_champion=shifted_down_champion,
            baseline_final_four=baseline_f4,
            shifted_up_final_four=shifted_up_f4,
            shifted_down_final_four=shifted_down_f4,
            is_stable=is_stable,
            flag=flag,
        )

    # ── Private helpers ──────────────────────────────────────────────

    @staticmethod
    def _hash_probabilities(probs: Dict[Tuple[str, str], float]) -> int:
        """Deterministic hash of probability dict for integrity checking."""
        items = sorted(probs.items(), key=lambda kv: (kv[0][0], kv[0][1]))
        return hash(tuple((k, round(v, 10)) for k, v in items))

    @staticmethod
    def _extract_championship_picks(
        public_picks: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """Extract championship pick percentages from round-level distribution."""
        chalk: Dict[str, float] = {}
        for team_id, rounds in public_picks.items():
            champ_pct = rounds.get("CHAMP", rounds.get("champion", 0.0))
            if champ_pct > 0:
                chalk[team_id] = champ_pct
        return chalk

    def _build_round_probabilities(self) -> Dict[str, Dict[str, float]]:
        """DEPRECATED fallback: approximate round probs from pairwise matchup averages.

        This heuristic scales average pairwise win rate by hand-tuned
        per-round multipliers.  It ignores bracket structure, opponent
        paths, and correlation — producing round advancement probabilities
        that diverge significantly from Monte Carlo simulation.

        Prefer passing ``model_round_probs`` (from ``to_round_probabilities``
        on MC ``AggregatedResults``) to the PoolOptimizer constructor.
        """
        import warnings

        warnings.warn(
            "PoolOptimizer._build_round_probabilities is using a heuristic "
            "fallback. Pass model_round_probs from MC simulation to the "
            "constructor for accurate round advancement probabilities.",
            DeprecationWarning,
            stacklevel=2,
        )
        team_strengths: Dict[str, List[float]] = {}
        for (t1, _t2), p in self._probabilities.items():
            team_strengths.setdefault(t1, [])
            team_strengths[t1].append(p)

        model_probs: Dict[str, Dict[str, float]] = {}
        for team_id, probs_list in team_strengths.items():
            avg_win = float(np.mean(probs_list)) if probs_list else 0.5
            model_probs[team_id] = {
                "R64": min(0.99, avg_win * 1.1),
                "R32": min(0.95, avg_win * 0.95),
                "S16": min(0.85, avg_win * 0.80),
                "E8": min(0.70, avg_win * 0.65),
                "F4": min(0.50, avg_win * 0.50),
                "CHAMP": min(0.30, avg_win * 0.35),
            }
        return model_probs

    def _run_leverage_analysis(
        self,
        model_round_probs: Dict[str, Dict[str, float]],
        env: PoolEnvironment,
    ) -> Tuple[List[Dict], List[Dict], List, Dict[str, float], str]:
        """Run leverage analysis using existing optimization infrastructure."""
        try:
            from .leverage import (
                TeamMetadata,
                analyze_pool,
                get_strategy_profile,
            )

            strategy_profile = get_strategy_profile(
                env.pool_size,
                payout_structure=env.payout_structure,
                scoring_system="standard",
            )

            pool_analysis = analyze_pool(
                pool_size=env.pool_size,
                model_probs=model_round_probs,
                public_picks=env.public_pick_distribution,
                scoring_system=env.scoring_rules,
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

            strategy_evs = pool_analysis.strategy_evs if hasattr(pool_analysis, "strategy_evs") else {}

            return (
                leverage_picks,
                fade_picks,
                # Pass the full frontier through — ParetoOptimizer already
                # controls its own output size via num_brackets (default 5).
                # The champion-diversity augmentation in generate_pareto_brackets
                # may grow the frontier beyond num_brackets when a dominant
                # champion would otherwise occupy all slots, so double-
                # truncating here would drop the alternative-champion
                # brackets the augmentation exists to produce.
                pool_analysis.pareto_brackets,
                strategy_evs,
                pool_analysis.recommended_strategy,
            )

        except ImportError:
            logger.warning("Leverage analysis infrastructure not available.")
            return [], [], [], {}, "balanced"

    @staticmethod
    def _extract_champion(result: PoolResult) -> str:
        """Extract champion team ID from optimization result."""
        if result.pareto_brackets:
            bracket = result.pareto_brackets[0]
            if hasattr(bracket, "champion"):
                return bracket.champion
        if result.leverage_picks:
            # Fallback: highest leverage pick for CHAMP round
            champ_picks = [p for p in result.leverage_picks if p.get("round") == "CHAMP"]
            if champ_picks:
                return champ_picks[0]["team_id"]
        return ""

    @staticmethod
    def _extract_final_four(result: PoolResult) -> List[str]:
        """Extract Final Four team IDs from optimization result."""
        if result.pareto_brackets:
            bracket = result.pareto_brackets[0]
            if hasattr(bracket, "final_four"):
                return list(bracket.final_four)
        # Fallback: top 4 teams by F4 leverage
        f4_picks = [p for p in result.leverage_picks if p.get("round") in ("F4", "CHAMP")]
        return [p["team_id"] for p in f4_picks[:4]]

    @staticmethod
    def _shift_picks(
        picks: Dict[str, Dict[str, float]],
        shift: float,
    ) -> Dict[str, Dict[str, float]]:
        """Shift all public pick percentages by a fractional amount.

        Clamps to [0.001, 0.999] to avoid division by zero.
        """
        shifted: Dict[str, Dict[str, float]] = {}
        for team_id, rounds in picks.items():
            shifted[team_id] = {}
            for round_name, pct in rounds.items():
                new_pct = pct * (1.0 + shift)
                shifted[team_id][round_name] = max(0.001, min(0.999, new_pct))
        return shifted
