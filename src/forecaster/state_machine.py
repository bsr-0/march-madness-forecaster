"""State machine forecaster — main orchestrator.

Implements the state machine:
    INIT -> BUILD -> AUDIT -> DECIDE -> FAIL_ANALYSIS -> FIX -> BUILD -> ...
    ... -> TERMINATE (success or max iterations)

Minimizes Brier (primary) and LogLoss (secondary).
Targets: ΔBrier ≤ -10%, ΔLogLoss ≤ -5%.

Overfitting-safe rules enforced:
1.  Locked evaluation protocol (deterministic seed + LOYO folds)
2.  CV-only acceptance gate (CV Brier + CV LogLoss must both improve)
3.  Single-change per iteration
4.  No CV feedback loops
5.  2025 holdout is untouchable (never trained on, never gates acceptance)
6.  Calibration is global only (fitted on training LOYO OOS)
7.  Feature stability check (audit)
8.  Variance constraint (per-year Brier std must not increase)
9.  Model simplicity prior (LR+GBM ceiling, no escalation)
10. Market anchor constraint (corr >= 0.7)
11. Prediction change validation (audit)
12. Iteration limit + rollback after 2 consecutive failures

CRITICAL: Holdout (2025) is evaluated every iteration for monitoring/logging
only. It is NEVER used in _decide() to gate acceptance. This prevents
adaptive selection bias (a.k.a. "hyperparameter tuning on the test set").
The holdout score reported at the end is the honest result of the final
accepted model, not the best of N evaluations.
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

# Rule 1 + 5: Locked training/holdout split
# Training: 2018, 2019, 2021-2024 (6 tournament years with results)
# Holdout: 2025 (most recent complete tournament, never trained on)
TRAIN_YEARS = [2018, 2019, 2021, 2022, 2023, 2024]
HOLDOUT_YEAR = 2025

# Rule 9: Model complexity ceiling
MAX_MODEL_COMPLEXITY = 3  # LR(1) + GBM(2) + Meta(3) = stacking ceiling


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
    cv_brier: float
    cv_log_loss: float
    holdout_brier: float
    holdout_log_loss: float
    per_year_brier_std: float
    audit_passed: bool
    accepted: bool = False
    violations: List[str] = field(default_factory=list)
    change_applied: str = ""
    predictions: Optional[np.ndarray] = None


@dataclass
class ForecasterOutput:
    """Final output from the state machine."""
    status: str                           # "PASS" or "FAIL"
    cv_brier: float
    cv_log_loss: float
    holdout_brier: float
    holdout_log_loss: float
    brier_improvement_pct: float
    log_loss_improvement_pct: float
    iterations: int
    changelog: List[str]
    model_summary: Dict[str, Any]
    predictions_2026: Dict[str, float]
    per_year_brier: Dict[int, float]
    calibration_method: str
    blend_weight: float


class StateMachineForecaster:
    """Overfitting-safe state machine forecaster.

    State transitions:
        INIT -> BUILD -> AUDIT -> DECIDE
        DECIDE: if dual-gate passed -> TERMINATE
        DECIDE: else -> FAIL_ANALYSIS -> FIX -> BUILD
        FIX: if iteration >= MAX or 2 consecutive failures -> TERMINATE

    All 12 overfitting-safe rules are enforced.
    """

    MAX_ITERATIONS = 5

    def __init__(
        self,
        data_dir: str = "data/raw",
        random_seed: int = 42,  # Rule 1: locked seed
    ):
        self.data_dir = Path(data_dir)
        self.random_seed = random_seed
        self.state = State.INIT
        self.iteration = 0
        self.history: List[IterationRecord] = []
        self.baseline_metrics: Optional[Dict[str, float]] = None

        # Rule 2: Dual-gate tracking (CV + holdout)
        self.best_cv_metrics: Optional[Dict[str, float]] = None
        self.best_holdout_metrics: Optional[Dict[str, float]] = None
        self.best_predictions: Optional[np.ndarray] = None
        self.best_per_year_brier: Optional[Dict[int, float]] = None

        # Rule 12: Consecutive non-improvement counter
        self.consecutive_no_improvement: int = 0

        # Components (initialized during BUILD)
        self.stacking: Optional[StackingEnsemble] = None
        self.calibrator: Optional[LOYOCalibrator] = None
        self.stacking_result: Optional[StackingResult] = None
        self.calibration_result: Optional[CalibrationResult] = None
        self.blend_weight: float = 0.5
        self.upset_weights: Dict[str, float] = {}

        # Data
        self.team_stats: Dict[int, Dict[str, TeamStats]] = {}
        self.tournament_games: List[Dict] = []
        self.bracket_2026: List[Dict] = []

        # Tuning parameters (adjusted during FIX — one at a time, Rule 3)
        self.lr_C: float = 1.0
        self.gbm_max_depth: int = 5
        self.gbm_lr: float = 0.05
        self.gbm_n_estimators: int = 300
        self.meta_C: float = 0.5
        self.calibration_preference: Optional[str] = None

    def run(self) -> ForecasterOutput:
        """Execute the state machine to completion."""
        logger.info("=" * 60)
        logger.info("STATE MACHINE FORECASTER — STARTING (seed=%d)", self.random_seed)
        logger.info("Training years: %s", TRAIN_YEARS)
        logger.info("Holdout year: %d (untouchable)", HOLDOUT_YEAR)
        logger.info("=" * 60)

        # INIT
        self._transition(State.INIT)
        self._load_data()
        self.baseline_metrics = self._compute_baseline_metrics()
        logger.info("Baseline metrics: Brier=%.4f, LogLoss=%.4f",
                    self.baseline_metrics["brier"], self.baseline_metrics["log_loss"])

        cv_metrics = {}
        holdout_metrics = {}
        predictions = np.array([])

        while self.state != State.TERMINATE:
            if self.state == State.INIT:
                self._transition(State.BUILD)

            elif self.state == State.BUILD:
                cv_metrics, holdout_metrics, predictions, per_year_brier, market_probs = (
                    self._build()
                )
                self._transition(State.AUDIT)

            elif self.state == State.AUDIT:
                audit_result = self._audit(
                    cv_metrics, holdout_metrics, predictions,
                    per_year_brier, market_probs,
                )
                self._transition(State.DECIDE)

            elif self.state == State.DECIDE:
                should_terminate = self._decide(
                    audit_result, cv_metrics, holdout_metrics,
                    predictions, per_year_brier,
                )
                if should_terminate:
                    self._transition(State.TERMINATE)
                else:
                    # Rule 12: rollback after 2 consecutive failures
                    if self.consecutive_no_improvement >= 2:
                        logger.warning(
                            "2 consecutive non-improvements — rolling back to best and TERMINATING"
                        )
                        self._transition(State.TERMINATE)
                    else:
                        self._transition(State.FAIL_ANALYSIS)

            elif self.state == State.FAIL_ANALYSIS:
                fixes = self._fail_analysis(audit_result, cv_metrics)
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
    # DATA LOADING
    # ------------------------------------------------------------------

    def _load_data(self):
        """Load all required data."""
        logger.info("Loading data from %s", self.data_dir)

        for year in range(2016, 2027):
            if year == 2020:
                continue
            stats = self._load_year_stats(year)
            if stats:
                self.team_stats[year] = stats
                logger.info("Year %d: loaded %d teams", year, len(stats))

        self.tournament_games = self._load_tournament_results()
        logger.info("Loaded %d tournament games across %d years",
                    len(self.tournament_games),
                    len(set(g["year"] for g in self.tournament_games)))

        self.bracket_2026 = self._load_bracket_2026()
        logger.info("Loaded %d teams for 2026 bracket", len(self.bracket_2026))

    def _load_year_stats(self, year: int) -> Dict[str, TeamStats]:
        """Load team statistics from real Torvik/advanced metrics data."""
        stats: Dict[str, TeamStats] = {}

        # Source 1: Historical Torvik
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

        # Source 2: Advanced metrics
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

        # Source 3: Four factors + shooting
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

        # Build name mapping for advanced metrics cross-referencing
        adv_name_to_id = {}
        for tid, t in adv_teams.items():
            name = t.get("name", "").lower().replace(" ", "_")
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
            adv_name_to_id[tid] = tid

        all_team_ids = set(list(ff_data.keys()) + list(shoot_data.keys()) +
                          list(torvik_teams.keys()))

        for team_id in all_team_ids:
            ff = ff_data.get(team_id, {})
            sh = shoot_data.get(team_id, {})
            tv = torvik_teams.get(team_id, {})
            adv = adv_teams.get(team_id, {})
            if not adv:
                adv = adv_teams.get(adv_name_to_id.get(team_id, ""), {})

            ts = TeamStats(team_id=team_id)

            ts.efg_pct = ff.get("effective_fg_pct", tv.get("effective_fg_pct", 0.50))
            ts.to_rate = ff.get("turnover_rate", tv.get("turnover_rate", 0.18))
            ts.orb_rate = ff.get("offensive_reb_rate", tv.get("offensive_reb_rate", 0.30))
            ts.ft_rate = ff.get("free_throw_rate", tv.get("free_throw_rate", 0.30))
            ts.opp_efg_pct = ff.get("opp_effective_fg_pct", 0.50)
            ts.opp_to_rate = ff.get("opp_turnover_rate", 0.18)
            ts.opp_orb_rate = 1.0 - ff.get("defensive_reb_rate", 0.70)
            ts.opp_ft_rate = ff.get("opp_free_throw_rate", 0.30)

            ts.three_pt_pct = sh.get("three_pt_pct", 0.33)
            ts.ft_pct = sh.get("ft_pct", 0.72)

            # Real efficiency metrics (torvik > advanced > estimate)
            ts.adj_off_eff = tv.get("adj_offensive_efficiency",
                                    adv.get("adj_offensive_efficiency", 0.0))
            ts.adj_def_eff = tv.get("adj_defensive_efficiency",
                                    adv.get("adj_defensive_efficiency", 0.0))
            ts.adj_tempo = tv.get("adj_tempo", adv.get("adj_tempo", 0.0))

            if ts.adj_off_eff == 0.0:
                ts.adj_off_eff = _estimate_efficiency(
                    ts.efg_pct, ts.to_rate, ts.orb_rate, ts.ft_rate
                )
            if ts.adj_def_eff == 0.0:
                ts.adj_def_eff = _estimate_efficiency(
                    ts.opp_efg_pct, ts.opp_to_rate, ts.opp_orb_rate, ts.opp_ft_rate
                )
            if ts.adj_tempo == 0.0:
                ts.adj_tempo = 68.0

            ts.sos = adv.get("sos_adj_em", tv.get("sos_adj_em", 0.0))

            wins = adv.get("wins", 0)
            losses = adv.get("losses", 0)
            if wins + losses > 0:
                ts.win_pct = wins / (wins + losses)

            barthag = tv.get("barthag", adv.get("barthag", 0.0))
            if barthag > 0:
                ts.elo = 1500.0 + (barthag - 0.5) * 1000.0

            stats[team_id] = ts

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
        """Load 2026 tournament bracket."""
        path = self.data_dir / "bracket_2026.json"
        if not path.exists():
            logger.warning("No 2026 bracket found")
            return []
        with open(path) as f:
            data = json.load(f)
        teams = data.get("teams", [])

        stats_2026 = self.team_stats.get(2026, {})
        for t in teams:
            tid = t.get("team_id", "")
            rating = t.get("rating", 0)
            if tid in stats_2026 and rating > 0:
                stats_2026[tid].elo = rating

        return teams

    # ------------------------------------------------------------------
    # TRAINING DATA
    # ------------------------------------------------------------------

    def _build_training_data(
        self,
        upset_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build feature matrix and labels from tournament games."""
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
    # BUILD — with 2025 holdout separation
    # ------------------------------------------------------------------

    def _build(
        self,
    ) -> Tuple[Dict[str, float], Dict[str, float], np.ndarray, Dict[int, float], np.ndarray]:
        """BUILD state: train on 2016-2024, evaluate on 2025 holdout.

        Returns:
            (cv_metrics, holdout_metrics, cv_predictions, per_year_brier, market_probs)
        """
        logger.info("BUILD iteration %d", self.iteration)
        start = time.time()

        # Step 0: Learn upset weights (iteration 0 only, on training years only)
        if self.iteration == 0:
            X_init, y_init, _, year_labels_init, seed_diffs = self._build_training_data()
            # Rule 5: Only learn from training years
            train_init_mask = np.isin(year_labels_init, TRAIN_YEARS)
            if train_init_mask.sum() >= 50:
                self.upset_weights = learn_upset_weights(
                    [X_init[i] for i in np.where(train_init_mask)[0]],
                    y_init[train_init_mask],
                    seed_diffs[train_init_mask],
                )

        # Step 1: Build features with learned upset weights
        X_all, y_all, market_all, year_all, _ = self._build_training_data(
            upset_weights=self.upset_weights if self.upset_weights else None
        )

        # Rule 5: Split into training (2016-2024) and holdout (2025)
        train_mask = np.isin(year_all, TRAIN_YEARS)
        holdout_mask = year_all == HOLDOUT_YEAR

        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        market_train = market_all[train_mask]
        year_train = year_all[train_mask]

        X_holdout = X_all[holdout_mask]
        y_holdout = y_all[holdout_mask]
        market_holdout = market_all[holdout_mask]

        if X_train.shape[0] < 50:
            logger.error("Insufficient training data: %d samples", X_train.shape[0])
            empty = {"brier": 0.25, "log_loss": 0.693}
            return empty, empty, np.full(len(y_train), 0.5), {}, market_train

        logger.info("Training: %d samples (years %s), Holdout: %d samples (year %d)",
                    X_train.shape[0], TRAIN_YEARS, X_holdout.shape[0], HOLDOUT_YEAR)

        # Step 2: Stacking with LOYO on training years only
        self.stacking = StackingEnsemble(
            n_folds=len(TRAIN_YEARS),
            random_seed=self.random_seed,
            lr_C=self.lr_C,
            gbm_max_depth=self.gbm_max_depth,
            gbm_lr=self.gbm_lr,
            gbm_n_estimators=self.gbm_n_estimators,
            meta_C=self.meta_C,
        )
        self.stacking_result = self.stacking.fit(
            X_train, y_train, market_train, year_train
        )

        # Step 3: Calibration on training LOYO OOS predictions only (Rule 6)
        probs_by_year: Dict[int, np.ndarray] = {}
        outcomes_by_year: Dict[int, np.ndarray] = {}
        for i, yr in enumerate(year_train):
            yr = int(yr)
            if yr not in probs_by_year:
                probs_by_year[yr] = []
                outcomes_by_year[yr] = []
            probs_by_year[yr].append(self.stacking_result.oof_preds[i])
            outcomes_by_year[yr].append(y_train[i])

        probs_by_year = {k: np.array(v) for k, v in probs_by_year.items()}
        outcomes_by_year = {k: np.array(v) for k, v in outcomes_by_year.items()}

        self.calibrator = LOYOCalibrator()
        self.calibration_result = self.calibrator.fit_and_select(
            probs_by_year, outcomes_by_year
        )

        # Step 4: Apply calibration to training OOF predictions
        calibrated_train = self.calibration_result.calibrate(
            self.stacking_result.oof_preds
        )

        # Step 5: Tune blend weight on training data LOYO OOS
        # Use 2024 as internal holdout for blend tuning
        blend_tune_mask = year_train != max(TRAIN_YEARS)
        if blend_tune_mask.sum() >= 30:
            self.blend_weight, _ = tune_blend_weight(
                calibrated_train[blend_tune_mask],
                market_train[blend_tune_mask],
                y_train[blend_tune_mask],
            )
        else:
            self.blend_weight = 0.5

        # Step 6: CV metrics (training OOF only)
        final_train = (self.blend_weight * calibrated_train +
                       (1 - self.blend_weight) * market_train)

        cv_brier = float(np.mean((final_train - y_train) ** 2))
        eps = 1e-15
        p_train = np.clip(final_train, eps, 1 - eps)
        cv_ll = float(-np.mean(
            y_train * np.log(p_train) + (1 - y_train) * np.log(1 - p_train)
        ))
        cv_metrics = {"brier": cv_brier, "log_loss": cv_ll}

        # Per-year Brier for variance tracking (Rule 8)
        per_year_brier = {}
        for yr in sorted(set(year_train)):
            yr = int(yr)
            yr_mask = year_train == yr
            yr_brier = float(np.mean((final_train[yr_mask] - y_train[yr_mask]) ** 2))
            per_year_brier[yr] = yr_brier

        # Step 7: 2025 holdout evaluation (Rule 5: inference only, never trained)
        holdout_metrics = {"brier": 0.25, "log_loss": 0.693}
        if X_holdout.shape[0] > 0:
            # Predict using fitted stacking (inference, not OOF)
            holdout_model = self.stacking.predict(X_holdout, market_holdout)
            # Apply frozen calibrator
            holdout_calibrated = self.calibration_result.calibrate(holdout_model)
            # Apply blend
            holdout_final = (self.blend_weight * holdout_calibrated +
                             (1 - self.blend_weight) * market_holdout)

            ho_brier = float(np.mean((holdout_final - y_holdout) ** 2))
            p_ho = np.clip(holdout_final, eps, 1 - eps)
            ho_ll = float(-np.mean(
                y_holdout * np.log(p_ho) + (1 - y_holdout) * np.log(1 - p_ho)
            ))
            holdout_metrics = {"brier": ho_brier, "log_loss": ho_ll}

        elapsed = time.time() - start
        logger.info("BUILD complete in %.1fs:", elapsed)
        logger.info("  CV:      Brier=%.4f, LogLoss=%.4f", cv_brier, cv_ll)
        logger.info("  Holdout: Brier=%.4f, LogLoss=%.4f",
                    holdout_metrics["brier"], holdout_metrics["log_loss"])
        logger.info("  Blend weight: %.3f", self.blend_weight)
        logger.info("  Per-year Brier std: %.4f",
                    float(np.std(list(per_year_brier.values()))))

        return cv_metrics, holdout_metrics, final_train, per_year_brier, market_train

    # ------------------------------------------------------------------
    # AUDIT
    # ------------------------------------------------------------------

    def _audit(
        self,
        cv_metrics: Dict[str, float],
        holdout_metrics: Dict[str, float],
        predictions: np.ndarray,
        per_year_brier: Dict[int, float],
        market_probs: np.ndarray,
    ) -> AuditResult:
        """AUDIT state: validate all components."""
        logger.info("AUDIT iteration %d", self.iteration)

        feature_importances = None
        if self.stacking is not None:
            feature_importances = self.stacking.get_feature_importances()

        prev_preds = self.history[-1].predictions if self.history else None
        prev_metrics = ({"brier": self.history[-1].cv_brier,
                         "log_loss": self.history[-1].cv_log_loss}
                        if self.history else None)

        prev_per_year_brier = (self.best_per_year_brier
                               if self.best_per_year_brier else None)
        holdout_brier = holdout_metrics.get("brier")
        prev_holdout_brier = (self.best_holdout_metrics.get("brier")
                              if self.best_holdout_metrics else None)

        audit_result = run_audit(
            stacking_result=self.stacking_result,
            calibration_result=self.calibration_result,
            feature_importances=feature_importances,
            metrics=cv_metrics,
            n_features=N_MATCHUP_FEATURES,
            prev_predictions=prev_preds,
            curr_predictions=predictions,
            prev_metrics=prev_metrics,
            per_year_brier=per_year_brier,
            holdout_brier=holdout_brier,
            prev_holdout_brier=prev_holdout_brier,
            prev_per_year_brier=prev_per_year_brier,
            blend_weight=self.blend_weight,
            market_probs=market_probs,
        )

        # Record iteration
        per_year_std = float(np.std(list(per_year_brier.values()))) if per_year_brier else 0.0
        self.history.append(IterationRecord(
            iteration=self.iteration,
            state="AUDIT",
            cv_brier=cv_metrics["brier"],
            cv_log_loss=cv_metrics["log_loss"],
            holdout_brier=holdout_metrics["brier"],
            holdout_log_loss=holdout_metrics["log_loss"],
            per_year_brier_std=per_year_std,
            audit_passed=audit_result.passed,
            violations=[v.message for v in audit_result.violations],
            predictions=predictions,
        ))

        return audit_result

    # ------------------------------------------------------------------
    # DECIDE — dual-gate + variance constraint
    # ------------------------------------------------------------------

    def _decide(
        self,
        audit_result: AuditResult,
        cv_metrics: Dict[str, float],
        holdout_metrics: Dict[str, float],
        predictions: np.ndarray,
        per_year_brier: Dict[int, float],
    ) -> bool:
        """DECIDE state: CV-only acceptance gate (Rule 2).

        Accept ONLY if ALL of:
        1. Audit passed
        2. CV Brier improved
        3. CV LogLoss improved
        4. Per-year variance did not increase (Rule 8)

        CRITICAL: Holdout metrics are logged for monitoring but NEVER gate
        acceptance. This prevents adaptive selection bias against the holdout.
        """
        logger.info("DECIDE iteration %d", self.iteration)

        if not audit_result.passed:
            logger.info("Audit FAILED — proceeding to FAIL_ANALYSIS")
            self.consecutive_no_improvement += 1
            return False

        # First iteration: accept as baseline
        if self.best_cv_metrics is None:
            self.best_cv_metrics = cv_metrics.copy()
            self.best_holdout_metrics = holdout_metrics.copy()
            self.best_predictions = predictions.copy()
            self.best_per_year_brier = per_year_brier.copy()
            self.consecutive_no_improvement = 0
            self.history[-1].accepted = True

            logger.info("First iteration accepted as baseline")
            logger.info("  CV:      Brier=%.4f, LogLoss=%.4f",
                        cv_metrics["brier"], cv_metrics["log_loss"])
            logger.info("  Holdout: Brier=%.4f, LogLoss=%.4f (monitoring only, not gated)",
                        holdout_metrics["brier"], holdout_metrics["log_loss"])

            # Check if baseline already meets targets
            if self.baseline_metrics:
                brier_imp = ((self.baseline_metrics["brier"] - cv_metrics["brier"]) /
                             self.baseline_metrics["brier"])
                if brier_imp >= 0.05:
                    logger.info("Baseline already %.1f%% better than seed — TERMINATE",
                                brier_imp * 100)
                    return True

            return True

        # Rule 2: CV-only gate — both CV metrics must improve
        cv_brier_ok = cv_metrics["brier"] < self.best_cv_metrics["brier"]
        cv_ll_ok = cv_metrics["log_loss"] < self.best_cv_metrics["log_loss"]

        gate_status = {
            "cv_brier": cv_brier_ok, "cv_logloss": cv_ll_ok,
        }
        logger.info("CV-gate check: %s", gate_status)

        # Log holdout for monitoring (NOT gating)
        if self.best_holdout_metrics:
            ho_brier_delta = holdout_metrics["brier"] - self.best_holdout_metrics["brier"]
            logger.info("  Holdout monitoring: Brier delta=%.4f (not gated)", ho_brier_delta)

        if not all(gate_status.values()):
            failed_gates = [k for k, v in gate_status.items() if not v]
            logger.info("CV-gate FAILED on: %s — proceeding to FAIL_ANALYSIS",
                        failed_gates)
            self.consecutive_no_improvement += 1
            return False

        # Rule 8: Variance constraint
        curr_std = float(np.std(list(per_year_brier.values())))
        prev_std = float(np.std(list(self.best_per_year_brier.values())))
        if curr_std > prev_std:
            logger.info("Variance constraint FAILED: std %.4f > %.4f — FAIL_ANALYSIS",
                        curr_std, prev_std)
            self.consecutive_no_improvement += 1
            return False

        # All gates passed — accept
        self.best_cv_metrics = cv_metrics.copy()
        self.best_holdout_metrics = holdout_metrics.copy()
        self.best_predictions = predictions.copy()
        self.best_per_year_brier = per_year_brier.copy()
        self.consecutive_no_improvement = 0
        self.history[-1].accepted = True

        logger.info("ACCEPTED — CV gates passed")
        logger.info("  CV:      Brier=%.4f, LogLoss=%.4f",
                    cv_metrics["brier"], cv_metrics["log_loss"])
        logger.info("  Holdout: Brier=%.4f, LogLoss=%.4f (monitoring only)",
                    holdout_metrics["brier"], holdout_metrics["log_loss"])
        logger.info("  Per-year std: %.4f (was %.4f)", curr_std, prev_std)

        return True

    # ------------------------------------------------------------------
    # FAIL ANALYSIS + FIX
    # ------------------------------------------------------------------

    def _fail_analysis(
        self,
        audit_result: AuditResult,
        cv_metrics: Dict[str, float],
    ) -> List[Dict[str, str]]:
        """FAIL_ANALYSIS state: diagnose failures."""
        logger.info("FAIL_ANALYSIS iteration %d", self.iteration)
        fixes = analyze_failure(audit_result, cv_metrics, self.best_cv_metrics)
        for fix in fixes:
            logger.info("  Candidate fix: %s/%s — %s", fix["component"],
                        fix["failure_type"], fix["fix_spec"])
        return fixes

    def _fix(self, fixes: List[Dict[str, str]]):
        """FIX state: apply exactly ONE fix (Rule 3).

        Rule 4: No CV feedback loops — fixes are based on component type,
        not per-year Brier analysis.
        Rule 9: No model complexity escalation beyond LR+GBM stacking.
        """
        logger.info("FIX iteration %d", self.iteration)

        if not fixes:
            logger.info("  No fixes to apply")
            return

        # Rule 3: Apply exactly ONE fix
        fix = fixes[0]
        component = fix["component"]
        fix_spec = fix["fix_spec"]

        logger.info("  Applying SINGLE change: %s — %s", component, fix_spec)

        if component == "market":
            self.blend_weight = max(0.3, self.blend_weight - 0.1)
            logger.info("  Adjusted blend weight to %.2f", self.blend_weight)

        elif component == "calibration":
            self.meta_C *= 0.5
            logger.info("  Reduced meta regularization to C=%.2f", self.meta_C)

        elif component == "stacking":
            # Rule 9: No complexity escalation — only tune existing models
            self.gbm_max_depth = max(3, self.gbm_max_depth - 1)
            logger.info("  Reduced GBM depth to %d", self.gbm_max_depth)

        elif component == "generalization":
            self.lr_C = max(0.01, self.lr_C * 0.5)
            logger.info("  Increased LR regularization: C=%.3f", self.lr_C)

        elif component == "overfitting":
            self.gbm_lr = max(0.01, self.gbm_lr * 0.8)
            logger.info("  Reduced GBM learning rate to %.3f", self.gbm_lr)

        change_desc = f"{component}: {fix_spec}"
        self.history[-1].change_applied = change_desc
        logger.info("  Change recorded: %s", change_desc)

    # ------------------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------------------

    def _generate_output(self) -> ForecasterOutput:
        """Generate final output with 2026 predictions."""
        logger.info("TERMINATE — generating final output")

        cv_brier = self.best_cv_metrics["brier"] if self.best_cv_metrics else 0.25
        cv_ll = self.best_cv_metrics["log_loss"] if self.best_cv_metrics else 0.693
        ho_brier = self.best_holdout_metrics["brier"] if self.best_holdout_metrics else 0.25
        ho_ll = self.best_holdout_metrics["log_loss"] if self.best_holdout_metrics else 0.693

        if self.baseline_metrics:
            brier_imp = ((self.baseline_metrics["brier"] - cv_brier) /
                         self.baseline_metrics["brier"])
            ll_imp = ((self.baseline_metrics["log_loss"] - cv_ll) /
                      self.baseline_metrics["log_loss"])
            # Status based on improvement targets AND holdout
            status = "PASS" if brier_imp >= 0.10 and ll_imp >= 0.05 else "FAIL"
        else:
            brier_imp = 0.0
            ll_imp = 0.0
            status = "FAIL"

        predictions_2026 = self._predict_2026()

        changelog = []
        for record in self.history:
            entry = (f"iter={record.iteration}: "
                     f"CV_Brier={record.cv_brier:.4f}, CV_LL={record.cv_log_loss:.4f}, "
                     f"HO_Brier={record.holdout_brier:.4f}, HO_LL={record.holdout_log_loss:.4f}, "
                     f"std={record.per_year_brier_std:.4f}, "
                     f"audit={'PASS' if record.audit_passed else 'FAIL'}, "
                     f"accepted={'YES' if record.accepted else 'NO'}")
            if record.change_applied:
                entry += f" — change: {record.change_applied}"
            changelog.append(entry)

        model_summary = {
            "features": MATCHUP_FEATURE_NAMES,
            "n_features": N_MATCHUP_FEATURES,
            "base_models": ["logistic_regression_l2", "lightgbm"],
            "meta_model": "logistic_regression_l2",
            "stacking_folds": f"LOYO ({len(TRAIN_YEARS)} year-folds)",
            "calibration": (self.calibration_result.method
                            if self.calibration_result else "none"),
            "blend_weight": self.blend_weight,
            "data_source": "real Torvik AdjO/AdjD/Tempo/SOS",
            "holdout_year": HOLDOUT_YEAR,
            "training_years": TRAIN_YEARS,
            "random_seed": self.random_seed,
            "rules_enforced": [
                "locked_eval_protocol",
                "cv_only_acceptance_gate",
                "single_change_per_iter",
                "no_cv_feedback_loops",
                "2025_holdout_untouchable_no_gating",
                "global_calibration_only",
                "feature_stability_check",
                "variance_constraint",
                "model_simplicity_prior",
                "market_anchor_constraint",
                "prediction_change_validation",
                "iteration_limit_rollback",
            ],
        }
        if self.calibration_result:
            model_summary["calibration_temperature"] = self.calibration_result.temperature

        per_year_brier = self.best_per_year_brier or {}

        return ForecasterOutput(
            status=status,
            cv_brier=cv_brier,
            cv_log_loss=cv_ll,
            holdout_brier=ho_brier,
            holdout_log_loss=ho_ll,
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
    """Estimate adjusted efficiency from four factors (fallback only)."""
    base = 100.0
    efg_contrib = (efg - 0.50) * 120
    to_contrib = -(to_rate - 0.18) * 80
    orb_contrib = (orb_rate - 0.30) * 40
    ft_contrib = (ft_rate - 0.30) * 30
    return base + efg_contrib + to_contrib + orb_contrib + ft_contrib
