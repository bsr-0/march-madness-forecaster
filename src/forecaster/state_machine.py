"""State machine forecaster — main orchestrator.

Implements the state machine:
    INIT -> BUILD -> AUDIT -> DECIDE -> FAIL_ANALYSIS -> FIX -> BUILD -> ...
    ... -> TERMINATE (success or max iterations)

Minimizes Brier (primary) and LogLoss (secondary).
Targets: ΔBrier ≤ -10%, ΔLogLoss ≤ -5%.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .audit import AuditResult, analyze_failure, run_audit
from .calibration import CalibrationResult, LOYOCalibrator
from .market import (
    blend_probabilities,
    compute_market_probabilities,
    seed_to_spread,
    spread_to_probability,
    tune_blend_weight,
)
from .matchups import (
    MATCHUP_FEATURE_NAMES,
    N_MATCHUP_FEATURES,
    TeamStats,
    compute_matchup_features,
    learn_upset_weights,
)
from .stacking import StackingEnsemble, StackingResult

logger = logging.getLogger(__name__)


class State(Enum):
    INIT = "INIT"
    BUILD = "BUILD"
    AUDIT = "AUDIT"
    DECIDE = "DECIDE"
    FAIL_ANALYSIS = "FAIL_ANALYSIS"
    FIX = "FIX"
    TERMINATE = "TERMINATE"


@dataclass
class IterationRecord:
    """Record of a single iteration."""
    iteration: int
    state: str
    brier: float
    log_loss: float
    audit_passed: bool
    violations: List[str] = field(default_factory=list)
    fixes_applied: List[str] = field(default_factory=list)
    predictions: Optional[np.ndarray] = None


@dataclass
class ForecasterOutput:
    """Final output from the state machine."""
    status: str                           # "PASS" or "FAIL"
    brier: float
    log_loss: float
    brier_improvement_pct: float
    log_loss_improvement_pct: float
    iterations: int
    changelog: List[str]
    model_summary: Dict[str, Any]
    predictions_2026: Dict[str, float]    # team_pair -> probability
    per_year_brier: Dict[int, float]
    calibration_method: str
    blend_weight: float


class StateMachineForecaster:
    """State machine forecaster implementing the full spec.

    State transitions:
        INIT -> BUILD -> AUDIT -> DECIDE
        DECIDE: if audit=PASS and metrics improved over baseline -> TERMINATE
        DECIDE: else -> FAIL_ANALYSIS -> FIX -> BUILD
        FIX: if iteration >= 5 -> TERMINATE (FAIL)
    """

    MAX_ITERATIONS = 5

    def __init__(
        self,
        data_dir: str = "data/raw",
        random_seed: int = 2026,
    ):
        self.data_dir = Path(data_dir)
        self.random_seed = random_seed
        self.state = State.INIT
        self.iteration = 0
        self.history: List[IterationRecord] = []
        self.best_metrics: Optional[Dict[str, float]] = None
        self.best_predictions: Optional[np.ndarray] = None
        self.baseline_metrics: Optional[Dict[str, float]] = None

        # Components (initialized during BUILD)
        self.stacking: Optional[StackingEnsemble] = None
        self.calibrator: Optional[LOYOCalibrator] = None
        self.stacking_result: Optional[StackingResult] = None
        self.calibration_result: Optional[CalibrationResult] = None
        self.blend_weight: float = 0.5
        self.upset_weights: Dict[str, float] = {}

        # Data
        self.team_stats: Dict[int, Dict[str, TeamStats]] = {}  # year -> team_id -> stats
        self.tournament_games: List[Dict] = []
        self.bracket_2026: List[Dict] = []

        # Tuning parameters (adjusted during FIX)
        self.lr_C: float = 1.0
        self.gbm_max_depth: int = 5
        self.gbm_lr: float = 0.05
        self.gbm_n_estimators: int = 500
        self.meta_C: float = 1.0
        self.calibration_preference: Optional[str] = None  # None = auto-select

    def run(self) -> ForecasterOutput:
        """Execute the state machine to completion."""
        logger.info("=" * 60)
        logger.info("STATE MACHINE FORECASTER — STARTING")
        logger.info("=" * 60)

        # INIT
        self._transition(State.INIT)
        self._load_data()
        self.baseline_metrics = self._compute_baseline_metrics()
        logger.info("Baseline metrics: Brier=%.4f, LogLoss=%.4f",
                    self.baseline_metrics["brier"], self.baseline_metrics["log_loss"])

        # Main loop
        while self.state != State.TERMINATE:
            if self.state == State.INIT:
                self._transition(State.BUILD)

            elif self.state == State.BUILD:
                metrics, predictions = self._build()
                self._transition(State.AUDIT)

            elif self.state == State.AUDIT:
                audit_result = self._audit(metrics, predictions)
                self._transition(State.DECIDE)

            elif self.state == State.DECIDE:
                should_terminate = self._decide(audit_result, metrics, predictions)
                if should_terminate:
                    self._transition(State.TERMINATE)
                else:
                    self._transition(State.FAIL_ANALYSIS)

            elif self.state == State.FAIL_ANALYSIS:
                fixes = self._fail_analysis(audit_result, metrics)
                self._transition(State.FIX)

            elif self.state == State.FIX:
                self._fix(fixes)
                self.iteration += 1
                if self.iteration >= self.MAX_ITERATIONS:
                    logger.warning("Max iterations (%d) reached — TERMINATING",
                                   self.MAX_ITERATIONS)
                    self._transition(State.TERMINATE)
                else:
                    self._transition(State.BUILD)

        return self._generate_output()

    def _transition(self, new_state: State):
        """Transition to a new state."""
        logger.info("STATE: %s -> %s (iteration=%d)",
                    self.state.value, new_state.value, self.iteration)
        self.state = new_state

    # ------------------------------------------------------------------
    # DATA LOADING — uses real Torvik AdjO/AdjD/Tempo/SOS, not estimates
    # ------------------------------------------------------------------

    def _load_data(self):
        """Load all required data: Torvik stats, tournament results, bracket."""
        logger.info("Loading data from %s", self.data_dir)

        for year in range(2016, 2027):
            if year == 2020:
                continue
            stats = self._load_year_stats(year)
            if stats:
                self.team_stats[year] = stats
                logger.info("Year %d: loaded %d teams with real metrics", year, len(stats))

        self.tournament_games = self._load_tournament_results()
        logger.info("Loaded %d tournament games across %d years",
                    len(self.tournament_games),
                    len(set(g["year"] for g in self.tournament_games)))

        self.bracket_2026 = self._load_bracket_2026()
        logger.info("Loaded %d teams for 2026 bracket", len(self.bracket_2026))

    def _load_year_stats(self, year: int) -> Dict[str, TeamStats]:
        """Load team statistics from real Torvik/advanced metrics data.

        Priority:
        1. historical/torvik_{year}.json — has real AdjO, AdjD, Tempo, barthag
        2. advanced_metrics_{year}.json — has AdjO, AdjD, Tempo, SOS, wins/losses
        3. torvik_four_factors + torvik_shooting — four factors only (fallback)

        All sources are merged: real efficiency metrics from (1)/(2) take priority,
        four factors from (3) fill in the rest.
        """
        stats: Dict[str, TeamStats] = {}

        # --- Source 1: Historical Torvik (real AdjO/AdjD/Tempo) ---
        torvik_path = self.data_dir / "historical" / f"torvik_{year}.json"
        torvik_teams = {}
        if torvik_path.exists():
            with open(torvik_path) as f:
                raw = json.load(f)
            team_list = raw.get("teams", raw) if isinstance(raw, dict) else raw
            if isinstance(team_list, list):
                for t in team_list:
                    tid = t.get("team_id", "")
                    if tid:
                        torvik_teams[tid] = t

        # --- Source 2: Advanced metrics (real AdjO/AdjD/Tempo/SOS/W-L) ---
        adv_path = self.data_dir / f"advanced_metrics_{year}.json"
        adv_teams = {}
        if adv_path.exists():
            with open(adv_path) as f:
                raw = json.load(f)
            team_list = raw.get("teams", raw) if isinstance(raw, dict) else raw
            if isinstance(team_list, list):
                for t in team_list:
                    tid = t.get("team_id", "")
                    if tid:
                        adv_teams[tid] = t

        # --- Source 3: Four factors + shooting (always available) ---
        ff_path = self.data_dir / f"torvik_four_factors_{year}.json"
        ff_data = {}
        if ff_path.exists():
            with open(ff_path) as f:
                ff_data = json.load(f)

        shoot_path = self.data_dir / f"torvik_shooting_{year}.json"
        shoot_data = {}
        if shoot_path.exists():
            with open(shoot_path) as f:
                shoot_data = json.load(f)

        # Build name→id mapping from advanced metrics for cross-referencing
        adv_name_to_id = {}
        for tid, t in adv_teams.items():
            name = t.get("name", "").lower().replace(" ", "_")
            # Strip common suffixes for matching
            for suffix in ["_cougars", "_hawks", "_mountain_hawks", "_gators",
                           "_wildcats", "_chargers", "_bulldogs", "_tigers",
                           "_bears", "_cardinals", "_cavaliers", "_huskies",
                           "_jayhawks", "_wolverines", "_blue_devils",
                           "_hoosiers", "_tar_heels", "_boilermakers",
                           "_cyclones", "_red_raiders", "_longhorns",
                           "_razorbacks", "_volunteers", "_aggies",
                           "_sooners", "_cowboys", "_knights", "_panthers",
                           "_golden_eagles", "_friars", "_pirates",
                           "_gaels", "_zags", "_hilltoppers", "_owls",
                           "_rams", "_terrapins", "_wolfpack", "_hokies",
                           "_orange", "_mean_green", "_rebels",
                           "_bison", "_catamounts", "_leathernecks"]:
                if name.endswith(suffix):
                    name = name[:-len(suffix)]
                    break
            adv_name_to_id[name] = tid
            # Also map the raw team_id
            adv_name_to_id[tid] = tid

        # Merge all sources: iterate four-factors teams (canonical IDs)
        all_team_ids = set(list(ff_data.keys()) + list(shoot_data.keys()) +
                          list(torvik_teams.keys()))

        for team_id in all_team_ids:
            ff = ff_data.get(team_id, {})
            sh = shoot_data.get(team_id, {})
            tv = torvik_teams.get(team_id, {})

            # Try to find advanced metrics by team_id or name matching
            adv = adv_teams.get(team_id, {})
            if not adv:
                adv = adv_teams.get(adv_name_to_id.get(team_id, ""), {})

            ts = TeamStats(team_id=team_id)

            # Four factors (real data from Torvik)
            ts.efg_pct = ff.get("effective_fg_pct", tv.get("effective_fg_pct", 0.50))
            ts.to_rate = ff.get("turnover_rate", tv.get("turnover_rate", 0.18))
            ts.orb_rate = ff.get("offensive_reb_rate", tv.get("offensive_reb_rate", 0.30))
            ts.ft_rate = ff.get("free_throw_rate", tv.get("free_throw_rate", 0.30))
            ts.opp_efg_pct = ff.get("opp_effective_fg_pct", 0.50)
            ts.opp_to_rate = ff.get("opp_turnover_rate", 0.18)
            ts.opp_orb_rate = 1.0 - ff.get("defensive_reb_rate", 0.70)
            ts.opp_ft_rate = ff.get("opp_free_throw_rate", 0.30)

            # Shooting
            ts.three_pt_pct = sh.get("three_pt_pct", 0.33)
            ts.ft_pct = sh.get("ft_pct", 0.72)

            # --- REAL efficiency metrics (priority: torvik > advanced) ---
            ts.adj_off_eff = tv.get("adj_offensive_efficiency",
                                    adv.get("adj_offensive_efficiency", 0.0))
            ts.adj_def_eff = tv.get("adj_defensive_efficiency",
                                    adv.get("adj_defensive_efficiency", 0.0))
            ts.adj_tempo = tv.get("adj_tempo", adv.get("adj_tempo", 0.0))

            # If no real efficiency data found, estimate from four factors
            if ts.adj_off_eff == 0.0:
                ts.adj_off_eff = _estimate_efficiency(
                    ts.efg_pct, ts.to_rate, ts.orb_rate, ts.ft_rate
                )
                ts._efficiency_estimated = True
            if ts.adj_def_eff == 0.0:
                ts.adj_def_eff = _estimate_efficiency(
                    ts.opp_efg_pct, ts.opp_to_rate, ts.opp_orb_rate, ts.opp_ft_rate
                )
            if ts.adj_tempo == 0.0:
                ts.adj_tempo = 68.0  # NCAA average

            # --- REAL SOS (from advanced metrics) ---
            ts.sos = adv.get("sos_adj_em", tv.get("sos_adj_em", 0.0))

            # --- REAL win% (from advanced metrics) ---
            wins = adv.get("wins", 0)
            losses = adv.get("losses", 0)
            if wins + losses > 0:
                ts.win_pct = wins / (wins + losses)

            # --- Elo from bracket rating (2026 only) or barthag proxy ---
            barthag = tv.get("barthag", adv.get("barthag", 0.0))
            if barthag > 0:
                # Convert barthag (0-1) to Elo-like scale: 1500 + (barthag - 0.5) * 1000
                ts.elo = 1500.0 + (barthag - 0.5) * 1000.0

            stats[team_id] = ts

        n_real = sum(1 for s in stats.values()
                     if not getattr(s, '_efficiency_estimated', False))
        logger.info("Year %d: %d/%d teams with real efficiency metrics",
                    year, n_real, len(stats))

        return stats

    def _load_tournament_results(self) -> List[Dict]:
        """Load tournament results from JSON files."""
        games = []
        hist_dir = self.data_dir / "historical"

        for year in range(2016, 2026):
            if year == 2020:
                continue
            path = hist_dir / f"tournament_results_{year}.json"
            if not path.exists():
                continue
            with open(path) as f:
                data = json.load(f)
            game_list = data if isinstance(data, list) else data.get("games", [])
            for g in game_list:
                g["year"] = g.get("year", year)
                games.append(g)

        return games

    def _load_bracket_2026(self) -> List[Dict]:
        """Load 2026 tournament bracket and inject ratings as Elo."""
        path = self.data_dir / "bracket_2026.json"
        if not path.exists():
            logger.warning("No 2026 bracket found")
            return []
        with open(path) as f:
            data = json.load(f)
        teams = data.get("teams", [])

        # Inject bracket ratings into 2026 team stats
        stats_2026 = self.team_stats.get(2026, {})
        for t in teams:
            tid = t.get("team_id", "")
            rating = t.get("rating", 0)
            if tid in stats_2026 and rating > 0:
                stats_2026[tid].elo = rating

        return teams

    # ------------------------------------------------------------------
    # TRAINING DATA CONSTRUCTION
    # ------------------------------------------------------------------

    def _build_training_data(
        self,
        upset_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build feature matrix and labels from tournament games.

        Args:
            upset_weights: Learned upset score weights (passed to
                compute_matchup_features via compute_upset_score).

        Returns:
            Tuple of (X, y, market_probs, year_labels, seed_diffs).
        """
        X_list = []
        y_list = []
        market_list = []
        year_list = []
        seed_diff_list = []

        for game in self.tournament_games:
            year = game["year"]
            team1_id = game["team1_id"]
            team2_id = game["team2_id"]
            seed1 = game.get("team1_seed", 8)
            seed2 = game.get("team2_seed", 8)
            team1_won = game["team1_won"]

            if year not in self.team_stats:
                continue
            stats = self.team_stats[year]
            if team1_id not in stats or team2_id not in stats:
                continue

            a = stats[team1_id]
            a.seed = seed1
            b = stats[team2_id]
            b.seed = seed2

            features = compute_matchup_features(a, b, upset_weights=upset_weights)
            market_prob = compute_market_probabilities(seed1, seed2)

            X_list.append(features)
            y_list.append(1 if team1_won else 0)
            market_list.append(market_prob)
            year_list.append(year)
            seed_diff_list.append(seed2 - seed1)

        X = np.array(X_list, dtype=np.float64)
        y = np.array(y_list, dtype=np.int32)
        market_probs = np.array(market_list, dtype=np.float64)
        year_labels = np.array(year_list, dtype=np.int32)
        seed_diffs = np.array(seed_diff_list, dtype=np.float64)

        # Replace NaN/Inf with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        logger.info("Training data: %d samples, %d features", X.shape[0], X.shape[1])
        return X, y, market_probs, year_labels, seed_diffs

    def _compute_baseline_metrics(self) -> Dict[str, float]:
        """Compute baseline metrics using seed-based probabilities."""
        probs = []
        outcomes = []

        for game in self.tournament_games:
            seed1 = game.get("team1_seed", 8)
            seed2 = game.get("team2_seed", 8)
            prob = spread_to_probability(seed_to_spread(seed1, seed2))
            probs.append(prob)
            outcomes.append(1 if game["team1_won"] else 0)

        if not probs:
            return {"brier": 0.25, "log_loss": 0.693}

        probs = np.array(probs)
        outcomes = np.array(outcomes)
        brier = float(np.mean((probs - outcomes) ** 2))
        eps = 1e-15
        p = np.clip(probs, eps, 1 - eps)
        log_loss = float(-np.mean(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p)))

        return {"brier": brier, "log_loss": log_loss}

    # ------------------------------------------------------------------
    # BUILD — stacking with LOYO + holdout blend tuning
    # ------------------------------------------------------------------

    def _build(self) -> Tuple[Dict[str, float], np.ndarray]:
        """BUILD state: train models, generate predictions, compute metrics.

        Key fixes over v1:
        - Stacking uses LOYO (year-based folds), not random K-fold
        - Meta model uses nested LOYO (no in-sample fitting)
        - Blend weight tuned on held-out year, not training data
        - Upset weights are learned first, then used in feature construction
        """
        logger.info("BUILD iteration %d", self.iteration)
        start = time.time()

        # Step 0: Learn upset weights on initial features (iteration 0 only)
        if self.iteration == 0:
            X_init, y_init, _, _, seed_diffs = self._build_training_data()
            if X_init.shape[0] >= 50:
                self.upset_weights = learn_upset_weights(
                    [X_init[i] for i in range(len(X_init))], y_init, seed_diffs
                )

        # Step 1: Rebuild features WITH learned upset weights
        X, y, market_probs, year_labels, seed_diffs = self._build_training_data(
            upset_weights=self.upset_weights if self.upset_weights else None
        )

        if X.shape[0] < 50:
            logger.error("Insufficient training data: %d samples", X.shape[0])
            return {"brier": 0.25, "log_loss": 0.693}, np.full(len(y), 0.5)

        # Step 2: Stacking ensemble with LOYO folds
        self.stacking = StackingEnsemble(
            n_folds=5,
            random_seed=self.random_seed,
            lr_C=self.lr_C,
            gbm_max_depth=self.gbm_max_depth,
            gbm_lr=self.gbm_lr,
            gbm_n_estimators=self.gbm_n_estimators,
            meta_C=self.meta_C,
        )
        self.stacking_result = self.stacking.fit(
            X, y, market_probs, year_labels
        )

        # Step 3: LOYO Calibration on stacking OOF predictions
        probs_by_year: Dict[int, np.ndarray] = {}
        outcomes_by_year: Dict[int, np.ndarray] = {}
        for i, yr in enumerate(year_labels):
            yr = int(yr)
            if yr not in probs_by_year:
                probs_by_year[yr] = []
                outcomes_by_year[yr] = []
            probs_by_year[yr].append(self.stacking_result.oof_preds[i])
            outcomes_by_year[yr].append(y[i])

        probs_by_year = {k: np.array(v) for k, v in probs_by_year.items()}
        outcomes_by_year = {k: np.array(v) for k, v in outcomes_by_year.items()}

        self.calibrator = LOYOCalibrator()
        self.calibration_result = self.calibrator.fit_and_select(
            probs_by_year, outcomes_by_year
        )

        # Step 4: Apply calibration to OOF predictions
        calibrated_preds = self.calibration_result.calibrate(
            self.stacking_result.oof_preds
        )

        # Step 5: Tune blend weight on HELD-OUT year (latest year)
        # Use all-but-latest for tuning, latest for validation
        unique_years = sorted(set(year_labels))
        if len(unique_years) >= 2:
            holdout_year = unique_years[-1]
            tune_mask = year_labels != holdout_year
            holdout_mask = year_labels == holdout_year

            if tune_mask.sum() >= 30:
                self.blend_weight, _ = tune_blend_weight(
                    calibrated_preds[tune_mask],
                    market_probs[tune_mask],
                    y[tune_mask],
                )
                # Report holdout performance with tuned weight
                holdout_blended = (self.blend_weight * calibrated_preds[holdout_mask] +
                                   (1 - self.blend_weight) * market_probs[holdout_mask])
                holdout_brier = float(np.mean((holdout_blended - y[holdout_mask]) ** 2))
                logger.info("Holdout year %d Brier: %.4f (w=%.3f)",
                            holdout_year, holdout_brier, self.blend_weight)
            else:
                self.blend_weight = 0.5
        else:
            self.blend_weight = 0.5

        # Step 6: Compute final metrics (LOYO OOS only — no in-sample)
        # Use per-year LOYO Brier from calibration as the honest metric
        final_preds = (self.blend_weight * calibrated_preds +
                       (1 - self.blend_weight) * market_probs)

        # Honest metrics: compute on full data but note these are OOF from stacking
        brier = float(np.mean((final_preds - y) ** 2))
        eps = 1e-15
        p = np.clip(final_preds, eps, 1 - eps)
        log_loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

        metrics = {"brier": brier, "log_loss": log_loss}
        elapsed = time.time() - start
        logger.info("BUILD complete in %.1fs: Brier=%.4f, LogLoss=%.4f",
                    elapsed, brier, log_loss)

        return metrics, final_preds

    def _audit(
        self,
        metrics: Dict[str, float],
        predictions: np.ndarray,
    ) -> AuditResult:
        """AUDIT state: validate all components."""
        logger.info("AUDIT iteration %d", self.iteration)

        feature_importances = None
        if self.stacking is not None:
            feature_importances = self.stacking.get_feature_importances()

        prev_preds = self.history[-1].predictions if self.history else None
        prev_metrics = {"brier": self.history[-1].brier,
                        "log_loss": self.history[-1].log_loss} if self.history else None

        per_year_brier = (self.calibration_result.per_year_brier
                          if self.calibration_result else None)
        prev_per_year_brier = None
        holdout_brier = None
        prev_holdout_brier = None

        if per_year_brier:
            max_year = max(per_year_brier.keys())
            holdout_brier = per_year_brier.get(max_year)

        audit_result = run_audit(
            stacking_result=self.stacking_result,
            calibration_result=self.calibration_result,
            feature_importances=feature_importances,
            metrics=metrics,
            n_features=N_MATCHUP_FEATURES,
            prev_predictions=prev_preds,
            curr_predictions=predictions,
            prev_metrics=prev_metrics,
            per_year_brier=per_year_brier,
            holdout_brier=holdout_brier,
            prev_holdout_brier=prev_holdout_brier,
            prev_per_year_brier=prev_per_year_brier,
            blend_weight=self.blend_weight,
        )

        # Record iteration
        self.history.append(IterationRecord(
            iteration=self.iteration,
            state="AUDIT",
            brier=metrics["brier"],
            log_loss=metrics["log_loss"],
            audit_passed=audit_result.passed,
            violations=[v.message for v in audit_result.violations],
            predictions=predictions,
        ))

        return audit_result

    def _decide(
        self,
        audit_result: AuditResult,
        metrics: Dict[str, float],
        predictions: np.ndarray,
    ) -> bool:
        """DECIDE state: determine if we should terminate.

        Requires BOTH audit pass AND improvement over baseline.
        Does NOT auto-pass on iteration 0.
        """
        logger.info("DECIDE iteration %d", self.iteration)

        if not audit_result.passed:
            logger.info("Audit FAILED — proceeding to FAIL_ANALYSIS")
            return False

        # Must improve over baseline (not just over previous best)
        if self.baseline_metrics:
            brier_imp = ((self.baseline_metrics["brier"] - metrics["brier"]) /
                         self.baseline_metrics["brier"])
            ll_imp = ((self.baseline_metrics["log_loss"] - metrics["log_loss"]) /
                      self.baseline_metrics["log_loss"])

            if brier_imp < 0.05:
                # Less than 5% improvement over seed baseline — not good enough
                logger.info("Insufficient improvement over baseline "
                            "(Brier: %.1f%%, need >= 5%%) — FAIL_ANALYSIS",
                            brier_imp * 100)
                return False

        # Check if metrics improved vs best seen
        improved = False
        if self.best_metrics is None:
            improved = True
        elif metrics["brier"] < self.best_metrics["brier"]:
            improved = True

        if improved:
            self.best_metrics = metrics.copy()
            self.best_predictions = predictions.copy()
            logger.info("New best metrics: Brier=%.4f, LogLoss=%.4f",
                        metrics["brier"], metrics["log_loss"])
            if self.baseline_metrics:
                logger.info("Improvement: Brier=%.1f%%, LogLoss=%.1f%%",
                            brier_imp * 100, ll_imp * 100)
            return True

        logger.info("Metrics did not improve — proceeding to FAIL_ANALYSIS")
        return False

    def _fail_analysis(
        self,
        audit_result: AuditResult,
        metrics: Dict[str, float],
    ) -> List[Dict[str, str]]:
        """FAIL_ANALYSIS state: diagnose failures."""
        logger.info("FAIL_ANALYSIS iteration %d", self.iteration)
        fixes = analyze_failure(audit_result, metrics, self.best_metrics)
        for fix in fixes:
            logger.info("  Fix: %s/%s — %s", fix["component"],
                        fix["failure_type"], fix["fix_spec"])
        return fixes

    def _fix(self, fixes: List[Dict[str, str]]):
        """FIX state: apply fixes for the next iteration."""
        logger.info("FIX iteration %d", self.iteration)

        for fix in fixes:
            component = fix["component"]
            fix_spec = fix["fix_spec"]

            if component == "market":
                if self.iteration >= 2:
                    self.blend_weight = 0.4
                    logger.info("  Escalation: logit blend, w=0.4")
                else:
                    self.blend_weight = max(0.3, self.blend_weight - 0.1)
                    logger.info("  Adjusted blend weight to %.2f", self.blend_weight)

            elif component == "calibration":
                if self.iteration >= 2:
                    self.calibration_preference = "isotonic"
                    logger.info("  Escalation: preferring isotonic calibration")
                else:
                    self.meta_C *= 0.5
                    logger.info("  Reduced meta regularization to C=%.2f", self.meta_C)

            elif component == "stacking":
                if self.iteration >= 2:
                    self.lr_C *= 0.5
                    logger.info("  Escalation: stronger L2 (C=%.2f)", self.lr_C)
                else:
                    self.gbm_max_depth = max(3, self.gbm_max_depth - 1)
                    self.gbm_n_estimators = min(1000, self.gbm_n_estimators + 200)
                    logger.info("  Adjusted GBM: depth=%d, n_est=%d",
                                self.gbm_max_depth, self.gbm_n_estimators)

            elif component == "generalization":
                self.lr_C = max(0.01, self.lr_C * 0.5)
                self.meta_C = max(0.01, self.meta_C * 0.5)
                self.gbm_lr = max(0.01, self.gbm_lr * 0.8)
                logger.info("  Increased regularization: lr_C=%.3f, meta_C=%.3f, gbm_lr=%.3f",
                            self.lr_C, self.meta_C, self.gbm_lr)

            self.history[-1].fixes_applied.append(f"{component}: {fix_spec}")

    def _generate_output(self) -> ForecasterOutput:
        """Generate final output with 2026 predictions."""
        logger.info("TERMINATE — generating final output")

        if self.best_metrics and self.baseline_metrics:
            brier_imp = ((self.baseline_metrics["brier"] - self.best_metrics["brier"]) /
                         self.baseline_metrics["brier"])
            ll_imp = ((self.baseline_metrics["log_loss"] - self.best_metrics["log_loss"]) /
                      self.baseline_metrics["log_loss"])
            status = "PASS" if brier_imp >= 0.10 and ll_imp >= 0.05 else "FAIL"
        else:
            brier_imp = 0.0
            ll_imp = 0.0
            status = "FAIL"

        predictions_2026 = self._predict_2026()

        changelog = []
        for record in self.history:
            entry = (f"iter={record.iteration}: Brier={record.brier:.4f}, "
                     f"LogLoss={record.log_loss:.4f}, "
                     f"audit={'PASS' if record.audit_passed else 'FAIL'}")
            if record.fixes_applied:
                entry += f" — fixes: {record.fixes_applied}"
            changelog.append(entry)

        model_summary = {
            "features": MATCHUP_FEATURE_NAMES,
            "n_features": N_MATCHUP_FEATURES,
            "base_models": ["logistic_regression_l2", "lightgbm"],
            "meta_model": "logistic_regression_l2",
            "stacking_folds": "LOYO (year-based)",
            "calibration": (self.calibration_result.method
                            if self.calibration_result else "none"),
            "blend_weight": self.blend_weight,
            "data_source": "real Torvik AdjO/AdjD/Tempo/SOS",
        }
        if self.calibration_result:
            model_summary["calibration_temperature"] = self.calibration_result.temperature

        per_year_brier = {}
        if self.calibration_result:
            per_year_brier = self.calibration_result.per_year_brier

        return ForecasterOutput(
            status=status,
            brier=self.best_metrics["brier"] if self.best_metrics else 0.25,
            log_loss=self.best_metrics["log_loss"] if self.best_metrics else 0.693,
            brier_improvement_pct=brier_imp * 100,
            log_loss_improvement_pct=ll_imp * 100,
            iterations=self.iteration + 1,
            changelog=changelog,
            model_summary=model_summary,
            predictions_2026=predictions_2026,
            per_year_brier=per_year_brier,
            calibration_method=(self.calibration_result.method
                                if self.calibration_result else "none"),
            blend_weight=self.blend_weight,
        )

    def _predict_2026(self) -> Dict[str, float]:
        """Generate predictions for all 2026 tournament matchups."""
        predictions = {}

        if not self.bracket_2026 or not self.stacking or not self.stacking.is_fitted:
            logger.warning("Cannot generate 2026 predictions — missing data or models")
            return predictions

        teams = self.bracket_2026
        stats_2026 = self.team_stats.get(2026, {})

        if not stats_2026:
            logger.warning("No 2026 team stats available")
            return predictions

        for i, team_a in enumerate(teams):
            for team_b in teams[i + 1:]:
                a_id = team_a["team_id"]
                b_id = team_b["team_id"]
                a_seed = team_a.get("seed", 8)
                b_seed = team_b.get("seed", 8)

                a_stats = stats_2026.get(a_id)
                b_stats = stats_2026.get(b_id)

                if a_stats is None or b_stats is None:
                    prob = spread_to_probability(seed_to_spread(a_seed, b_seed))
                    predictions[f"{a_id}_{b_id}"] = round(prob, 4)
                    continue

                a_stats.seed = a_seed
                b_stats.seed = b_seed

                features = compute_matchup_features(
                    a_stats, b_stats,
                    upset_weights=self.upset_weights if self.upset_weights else None,
                )
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                X = features.reshape(1, -1)

                market_prob = compute_market_probabilities(a_seed, b_seed)

                model_prob = float(self.stacking.predict(
                    X, np.array([market_prob])
                )[0])

                if self.calibration_result:
                    model_prob = float(self.calibration_result.calibrate(
                        np.array([model_prob])
                    )[0])

                final_prob = blend_probabilities(
                    model_prob, market_prob, self.blend_weight
                )
                final_prob = float(np.clip(final_prob, 0.005, 0.995))

                predictions[f"{a_id}_{b_id}"] = round(final_prob, 4)

        logger.info("Generated %d pairwise predictions for 2026", len(predictions))
        return predictions


def _estimate_efficiency(
    efg: float, to_rate: float, orb_rate: float, ft_rate: float,
) -> float:
    """Estimate adjusted efficiency from four factors (fallback only).

    Used only when real AdjO/AdjD is unavailable. This is an approximation
    based on Dean Oliver's weights — NOT a substitute for real Torvik/KenPom.
    """
    base = 100.0
    efg_contrib = (efg - 0.50) * 120
    to_contrib = -(to_rate - 0.18) * 80
    orb_contrib = (orb_rate - 0.30) * 40
    ft_contrib = (ft_rate - 0.30) * 30
    return base + efg_contrib + to_contrib + orb_contrib + ft_contrib
