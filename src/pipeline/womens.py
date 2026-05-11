"""Women's tournament prediction pipeline.

This path is used for the women's half of the combined Kaggle submission.
When cached third-party metrics are unavailable, it falls back to offline
team-strength estimates built from Kaggle regular-season results rather than
collapsing to pure seed-based estimates.
"""

from __future__ import annotations

import copy
import csv
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ..data.features.womens_feature_engineering import WomensFeatureEngineer, compute_seed_win_probability
from ..data.scrapers.womens.herhoopstats import HerHoopStatsScraper, WomensTeamStats
from ..data.scrapers.womens.historical_results import WomensHistoricalResults
from ..data.scrapers.womens.ncaa_net import WomensNETScraper
from ..ml.calibration.post_processing import PostProcessingPipeline
from .probability_pipeline import apply_calibration, apply_final_clip, apply_tournament_shrinkage

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
KAGGLE_DATA_DIR = REPO_ROOT / "data" / "kaggle"

_TEAM_NAME_CACHE: Optional[Dict[str, str]] = None
_SEED_CACHE: Dict[int, Dict[str, int]] = {}
_SEASON_STATS_CACHE: Dict[int, Dict[str, WomensTeamStats]] = {}
_TOURNEY_ROWS_CACHE: Optional[Dict[int, list[tuple[str, str]]]] = None
_HISTORICAL_TRAINING_CACHE: Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency guard
    LogisticRegression = None
    StandardScaler = None
    SKLEARN_AVAILABLE = False


class _TemperatureCalibrator:
    """Small self-contained temperature scaler for women's probabilities."""

    def __init__(self) -> None:
        self.temperature = 1.0
        self.fitted = False

    def fit(self, probabilities: np.ndarray, outcomes: np.ndarray) -> None:
        probs = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
        y = np.asarray(outcomes, dtype=float)
        logits = np.log(probs / (1.0 - probs))

        best_t = 1.0
        best_brier = float("inf")
        for t in np.linspace(0.5, 2.5, 81):
            calibrated = 1.0 / (1.0 + np.exp(-logits / t))
            brier = float(np.mean((calibrated - y) ** 2))
            if brier < best_brier:
                best_brier = brier
                best_t = float(t)

        self.temperature = best_t
        self.fitted = True

    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        probs = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
        logits = np.log(probs / (1.0 - probs))
        return 1.0 / (1.0 + np.exp(-logits / max(self.temperature, 0.05)))


@dataclass
class _SeasonAggregate:
    games: int = 0
    wins: int = 0
    losses: int = 0
    points_for: float = 0.0
    points_against: float = 0.0
    possessions_for: float = 0.0
    possessions_against: float = 0.0
    fgm: float = 0.0
    fga: float = 0.0
    fgm3: float = 0.0
    fga3: float = 0.0
    ftm: float = 0.0
    fta: float = 0.0
    off_rebounds: float = 0.0
    def_rebounds: float = 0.0
    assists: float = 0.0
    turnovers: float = 0.0
    opp_fgm: float = 0.0
    opp_fga: float = 0.0
    opp_fgm3: float = 0.0
    opp_fta: float = 0.0
    opp_off_rebounds: float = 0.0
    opp_turnovers: float = 0.0
    off_reb_opportunities: float = 0.0
    def_reb_opportunities: float = 0.0
    opponents: list[str] = field(default_factory=list)


@dataclass
class WomensPipelineConfig:
    """Configuration for the women's tournament pipeline."""

    year: int = 2026
    cache_dir: str = "data/raw"
    calibration_method: str = "temperature"
    probability_profile: str = "production"
    clip_lo: float = 0.005
    clip_hi: float = 0.995
    seed_only_mode: bool = False
    enable_tournament_adaptation: bool = True
    tournament_shrinkage: float = 0.02
    enable_feature_scaling: bool = True
    model_weight: float = 0.55
    seed_weight: float = 0.45
    flb_threshold: float = 0.85
    flb_regression: float = 0.20

    def __post_init__(self) -> None:
        if self.probability_profile not in ("production", "experimental"):
            raise ValueError(
                f"Invalid probability_profile '{self.probability_profile}': must be 'production' or 'experimental'"
            )


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if abs(denominator) < 1e-12:
        return default
    return numerator / denominator


def _estimate_possessions(fga: int, offensive_rebounds: int, turnovers: int, fta: int) -> float:
    return float(fga - offensive_rebounds + turnovers + 0.475 * fta)


def _parse_seed(seed_code: str) -> int:
    digits = "".join(ch for ch in seed_code if ch.isdigit())
    return int(digits[:2]) if digits else 0


def _load_team_names() -> Dict[str, str]:
    global _TEAM_NAME_CACHE
    if _TEAM_NAME_CACHE is None:
        names: Dict[str, str] = {}
        with open(KAGGLE_DATA_DIR / "WTeams.csv", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                names[str(row["TeamID"])] = row["TeamName"]
        _TEAM_NAME_CACHE = names
    return _TEAM_NAME_CACHE


def _load_seed_map(year: int) -> Dict[str, int]:
    if year not in _SEED_CACHE:
        seed_map: Dict[str, int] = {}
        with open(KAGGLE_DATA_DIR / "WNCAATourneySeeds.csv", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if int(row["Season"]) != year:
                    continue
                seed_map[str(row["TeamID"])] = _parse_seed(row["Seed"])
        _SEED_CACHE[year] = seed_map
    return _SEED_CACHE[year]


def _load_tournament_rows() -> Dict[int, list[tuple[str, str]]]:
    global _TOURNEY_ROWS_CACHE
    if _TOURNEY_ROWS_CACHE is None:
        rows: Dict[int, list[tuple[str, str]]] = defaultdict(list)
        with open(KAGGLE_DATA_DIR / "WNCAATourneyDetailedResults.csv", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                year = int(row["Season"])
                rows[year].append((str(row["WTeamID"]), str(row["LTeamID"])))
        _TOURNEY_ROWS_CACHE = dict(rows)
    return _TOURNEY_ROWS_CACHE


class WomensPipeline:
    """Women-specific tournament prediction pipeline with robust fallbacks."""

    def __init__(self, config: Optional[WomensPipelineConfig] = None):
        self.config = config or WomensPipelineConfig()
        self.feature_engineer = WomensFeatureEngineer()
        self.team_stats: Dict[str, WomensTeamStats] = {}
        self.model = None
        self.scaler = None
        self.calibrator = None
        self.post_processor = PostProcessingPipeline(
            flb_threshold=self.config.flb_threshold,
            flb_regression=self.config.flb_regression,
            clip_lo=self.config.clip_lo,
            clip_hi=self.config.clip_hi,
        )
        self._trained = False

    def run(self) -> Dict[str, object]:
        """Load data, build features, and prepare the prediction path."""
        report: Dict[str, object] = {"status": "initialized"}

        scraper = HerHoopStatsScraper(cache_dir=self.config.cache_dir)
        self.team_stats = scraper.load_cached(self.config.year)
        report["cached_teams_loaded"] = len(self.team_stats)

        if len(self.team_stats) < 30:
            kaggle_stats = self._load_kaggle_team_stats(self.config.year)
            if kaggle_stats:
                self.team_stats = kaggle_stats
            report["kaggle_teams_loaded"] = len(kaggle_stats)

        report["teams_loaded"] = len(self.team_stats)

        net_scraper = WomensNETScraper(cache_dir=self.config.cache_dir)
        net_rankings = net_scraper.load_cached(self.config.year)
        report["net_rankings"] = len(net_rankings)

        self.feature_engineer.build_features(self.team_stats, net_rankings)
        report["features_built"] = len(self.feature_engineer.team_features)

        if not self.config.seed_only_mode and len(self.team_stats) >= 30:
            self._train_model()
        report["model_trained"] = self.model is not None

        self._fit_calibration()
        report["calibration_fitted"] = self.calibrator is not None

        self._trained = True
        report["status"] = "ready"
        return report

    def predict_probability(self, team1_id: str, team2_id: str) -> float:
        """Predict probability that team1 beats team2."""
        prob = self._raw_probability(team1_id, team2_id)
        prob = apply_calibration(prob, self.calibrator, self.config.clip_lo, self.config.clip_hi)
        if self.config.enable_tournament_adaptation:
            prob = apply_tournament_shrinkage(prob, self.config.tournament_shrinkage)

        if self.config.probability_profile == "experimental":
            seed1, seed2 = self._lookup_seeds(team1_id, team2_id)
            prob = self.post_processor.process(prob, seed1, seed2)

        return apply_final_clip(prob, self.config.clip_lo, self.config.clip_hi)

    def set_team_seeds(self, seed_map: Dict[str, int]) -> None:
        """Update or backfill teams from explicit seed information."""
        if not seed_map:
            return

        scraper = HerHoopStatsScraper(cache_dir=self.config.cache_dir)
        generated = scraper.generate_seed_based_estimates(seed_map)
        for team_id, seed in seed_map.items():
            if team_id in self.team_stats:
                self.team_stats[team_id].seed = seed
            else:
                self.team_stats[team_id] = generated[team_id]

        net_scraper = WomensNETScraper(cache_dir=self.config.cache_dir)
        net_rankings = net_scraper.estimate_from_seeds(seed_map)
        self.feature_engineer.build_features(self.team_stats, net_rankings)

    def _lookup_seeds(self, team1_id: str, team2_id: str) -> tuple[int, int]:
        f1 = self.feature_engineer.team_features.get(team1_id)
        f2 = self.feature_engineer.team_features.get(team2_id)
        team1 = self.team_stats.get(team1_id, WomensTeamStats(team_name=team1_id))
        team2 = self.team_stats.get(team2_id, WomensTeamStats(team_name=team2_id))
        seed1 = f1.seed if f1 is not None else team1.seed
        seed2 = f2.seed if f2 is not None else team2.seed
        return seed1, seed2

    def _raw_probability(self, team1_id: str, team2_id: str) -> float:
        """Blend matchup-model and seed prior into a raw pre-calibration probability."""
        seed1, seed2 = self._lookup_seeds(team1_id, team2_id)
        seed_prob = float(compute_seed_win_probability(seed1, seed2)) if seed1 > 0 and seed2 > 0 else 0.5

        if self.model is None:
            return seed_prob

        features_fwd = self.feature_engineer.get_matchup_features(team1_id, team2_id)
        features_rev = self.feature_engineer.get_matchup_features(team2_id, team1_id)
        if features_fwd is None or features_rev is None:
            return seed_prob

        model_fwd = self._predict_with_model(features_fwd)
        model_rev = self._predict_with_model(features_rev)
        model_prob = (model_fwd + (1.0 - model_rev)) / 2.0

        total = self.config.model_weight + self.config.seed_weight
        return (self.config.model_weight * model_prob + self.config.seed_weight * seed_prob) / total

    def _predict_with_model(self, features: np.ndarray) -> float:
        if self.model is None:
            return 0.5
        x = features.reshape(1, -1)
        if self.scaler is not None:
            x = self.scaler.transform(x)
        pred = float(self.model.predict_proba(x)[0][1])
        return float(np.clip(pred, 0.01, 0.99))

    def _fit_logistic_model(self, X: np.ndarray, y: np.ndarray) -> tuple[object, object]:
        scaler = None
        X_train = X
        if self.config.enable_feature_scaling:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)

        model = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs", random_state=42)
        model.fit(X_train, y)
        return model, scaler

    def _train_model(self) -> None:
        """Train a logistic model from historical women's tournament results."""
        if not SKLEARN_AVAILABLE:
            logger.info("Women's pipeline: sklearn unavailable, staying seed-only")
            return

        X, y, _, _ = self._historical_training_data()
        if len(X) < 100 or len(np.unique(y)) < 2:
            logger.info("Women's pipeline: insufficient historical data, staying seed-only")
            return

        self.model, self.scaler = self._fit_logistic_model(X, y)

    def _fit_calibration(self) -> None:
        """Fit temperature scaling from historical tournament outcomes."""
        if self.config.calibration_method == "none":
            return

        if self.model is not None:
            probabilities, outcomes = self._historical_oof_probabilities()
            if len(probabilities) >= 100:
                calibrator = _TemperatureCalibrator()
                calibrator.fit(probabilities, outcomes)
                self.calibrator = calibrator
                return

        history = WomensHistoricalResults(cache_dir=self.config.cache_dir)
        games = history.load_cached()
        if len(games) < 30:
            return

        probabilities, outcomes = history.get_calibration_data()
        calibrator = _TemperatureCalibrator()
        calibrator.fit(np.asarray(probabilities, dtype=float), np.asarray(outcomes, dtype=float))
        self.calibrator = calibrator

    def _historical_oof_probabilities(self) -> tuple[np.ndarray, np.ndarray]:
        X, y, seasons, seed_probs = self._historical_training_data()
        if len(X) < 100 or len(np.unique(y)) < 2 or not SKLEARN_AVAILABLE:
            return np.array([]), np.array([])

        probabilities = []
        outcomes = []
        total = self.config.model_weight + self.config.seed_weight

        for season in np.unique(seasons):
            train_mask = seasons != season
            val_mask = seasons == season
            if train_mask.sum() < 100 or val_mask.sum() == 0:
                continue

            model, scaler = self._fit_logistic_model(X[train_mask], y[train_mask])
            X_val = X[val_mask]
            if scaler is not None:
                X_val = scaler.transform(X_val)
            model_probs = np.asarray(model.predict_proba(X_val)[:, 1], dtype=float)
            blend = (self.config.model_weight * model_probs + self.config.seed_weight * seed_probs[val_mask]) / total
            probabilities.extend(blend.tolist())
            outcomes.extend(y[val_mask].tolist())

        return np.asarray(probabilities, dtype=float), np.asarray(outcomes, dtype=float)

    def _historical_training_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        global _HISTORICAL_TRAINING_CACHE
        if _HISTORICAL_TRAINING_CACHE is not None:
            return _HISTORICAL_TRAINING_CACHE

        tournament_rows = _load_tournament_rows()
        X_rows = []
        y_rows = []
        season_rows = []
        seed_prob_rows = []

        for season in sorted(tournament_rows):
            season_stats = self._load_kaggle_team_stats(season)
            if len(season_stats) < 30:
                continue

            feature_engineer = WomensFeatureEngineer()
            feature_engineer.build_features(season_stats)

            for winner_id, loser_id in tournament_rows[season]:
                if winner_id not in season_stats or loser_id not in season_stats:
                    continue

                seed_win = season_stats[winner_id].seed
                seed_lose = season_stats[loser_id].seed
                if seed_win <= 0 or seed_lose <= 0:
                    continue

                fwd = feature_engineer.get_matchup_features(winner_id, loser_id)
                rev = feature_engineer.get_matchup_features(loser_id, winner_id)
                if fwd is None or rev is None:
                    continue

                seed_prob = float(compute_seed_win_probability(seed_win, seed_lose))
                X_rows.append(fwd)
                y_rows.append(1.0)
                season_rows.append(season)
                seed_prob_rows.append(seed_prob)

                X_rows.append(rev)
                y_rows.append(0.0)
                season_rows.append(season)
                seed_prob_rows.append(1.0 - seed_prob)

        if not X_rows:
            empty = (np.empty((0, 0)), np.array([]), np.array([]), np.array([]))
            _HISTORICAL_TRAINING_CACHE = empty
            return empty

        dataset = (
            np.asarray(X_rows, dtype=float),
            np.asarray(y_rows, dtype=float),
            np.asarray(season_rows, dtype=int),
            np.asarray(seed_prob_rows, dtype=float),
        )
        _HISTORICAL_TRAINING_CACHE = dataset
        return dataset

    def _load_kaggle_team_stats(self, year: int) -> Dict[str, WomensTeamStats]:
        if year in _SEASON_STATS_CACHE:
            return copy.deepcopy(_SEASON_STATS_CACHE[year])

        path = KAGGLE_DATA_DIR / "WRegularSeasonDetailedResults.csv"
        if not path.exists():
            return {}

        names = _load_team_names()
        aggregates: Dict[str, _SeasonAggregate] = defaultdict(_SeasonAggregate)
        games_for_elo: list[tuple[int, str, str, str, int, int]] = []

        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if int(row["Season"]) != year:
                    continue

                wteam = str(row["WTeamID"])
                lteam = str(row["LTeamID"])
                games_for_elo.append(
                    (
                        int(row["DayNum"]),
                        wteam,
                        lteam,
                        row["WLoc"],
                        int(row["WScore"]),
                        int(row["LScore"]),
                    )
                )
                self._update_aggregate(aggregates[wteam], row, prefix="W", opp_prefix="L", won=True, opponent_id=lteam)
                self._update_aggregate(aggregates[lteam], row, prefix="L", opp_prefix="W", won=False, opponent_id=wteam)

        if not aggregates:
            _SEASON_STATS_CACHE[year] = {}
            return {}

        elo_ratings = self._compute_elo_ratings(games_for_elo)
        raw_net = {team_id: self._raw_net_rating(aggregate) for team_id, aggregate in aggregates.items()}
        ranking = sorted(aggregates.keys(), key=lambda tid: (raw_net[tid], elo_ratings[tid]), reverse=True)
        official_seeds = _load_seed_map(year)
        estimated_seeds = self._estimate_seed_map(ranking, official_seeds)

        team_stats: Dict[str, WomensTeamStats] = {}
        n_teams = len(ranking)
        rank_index = {team_id: idx for idx, team_id in enumerate(ranking)}

        for team_id, aggregate in aggregates.items():
            seed = official_seeds.get(team_id, estimated_seeds.get(team_id, 16))
            off_eff = 100.0 * _safe_div(aggregate.points_for, aggregate.possessions_for, 1.0)
            def_eff = 100.0 * _safe_div(aggregate.points_against, aggregate.possessions_against, 1.0)
            opp_strength = float(np.mean([raw_net.get(opp_id, 0.0) for opp_id in aggregate.opponents])) if aggregate.opponents else 0.0
            adjusted_off = off_eff + 0.12 * opp_strength
            adjusted_def = max(40.0, def_eff - 0.12 * opp_strength)
            adjusted_margin = adjusted_off - adjusted_def
            team_name = names.get(team_id, f"Team {team_id}")
            net_rating = 100.0 * (rank_index[team_id] + 0.5) / max(n_teams, 1)

            stats = WomensTeamStats(
                team_name=team_name,
                team_id=team_id,
                adj_offensive_efficiency=adjusted_off,
                adj_defensive_efficiency=adjusted_def,
                adj_efficiency_margin=adjusted_margin,
                adj_tempo=_safe_div(aggregate.possessions_for + aggregate.possessions_against, 2.0 * aggregate.games, 67.0),
                effective_fg_pct=_safe_div(aggregate.fgm + 0.5 * aggregate.fgm3, aggregate.fga, 0.44),
                turnover_rate=_safe_div(aggregate.turnovers, aggregate.possessions_for, 0.20),
                offensive_reb_rate=_safe_div(aggregate.off_rebounds, aggregate.off_reb_opportunities, 0.32),
                free_throw_rate=_safe_div(aggregate.fta, aggregate.fga, 0.30),
                opp_effective_fg_pct=_safe_div(aggregate.opp_fgm + 0.5 * aggregate.opp_fgm3, aggregate.opp_fga, 0.44),
                opp_turnover_rate=_safe_div(aggregate.opp_turnovers, aggregate.possessions_against, 0.20),
                defensive_reb_rate=_safe_div(aggregate.def_rebounds, aggregate.def_reb_opportunities, 0.68),
                opp_free_throw_rate=_safe_div(aggregate.opp_fta, aggregate.opp_fga, 0.30),
                sos_adj_em=opp_strength,
                wins=aggregate.wins,
                losses=aggregate.losses,
                win_pct=_safe_div(aggregate.wins, aggregate.games, 0.5),
                free_throw_pct=_safe_div(aggregate.ftm, aggregate.fta, 0.70),
                three_pt_pct=_safe_div(aggregate.fgm3, aggregate.fga3, 0.30),
                three_pt_variance=max(0.02, 0.12 / max(aggregate.games, 1)),
                seed=seed,
                elo_rating=elo_ratings[team_id],
            )
            stats.rebounding_margin = _safe_div(aggregate.off_rebounds - aggregate.opp_off_rebounds, aggregate.games, 0.0)
            stats.assist_to_turnover_ratio = _safe_div(aggregate.assists, aggregate.turnovers, 1.0)
            stats.net_rating = net_rating
            team_stats[team_id] = stats

        _SEASON_STATS_CACHE[year] = copy.deepcopy(team_stats)
        return copy.deepcopy(team_stats)

    @staticmethod
    def _estimate_seed_map(ranking: list[str], official_seeds: Dict[str, int]) -> Dict[str, int]:
        estimated = dict(official_seeds)
        for idx, team_id in enumerate(ranking):
            if team_id in estimated:
                continue
            if idx < 64:
                estimated[team_id] = idx // 4 + 1
            else:
                estimated[team_id] = 16
        return estimated

    @staticmethod
    def _raw_net_rating(aggregate: _SeasonAggregate) -> float:
        off_eff = 100.0 * _safe_div(aggregate.points_for, aggregate.possessions_for, 1.0)
        def_eff = 100.0 * _safe_div(aggregate.points_against, aggregate.possessions_against, 1.0)
        return off_eff - def_eff

    @staticmethod
    def _update_aggregate(
        aggregate: _SeasonAggregate,
        row: Dict[str, str],
        *,
        prefix: str,
        opp_prefix: str,
        won: bool,
        opponent_id: str,
    ) -> None:
        score = int(row[f"{prefix}Score"])
        opp_score = int(row[f"{opp_prefix}Score"])
        fga = int(row[f"{prefix}FGA"])
        fgm = int(row[f"{prefix}FGM"])
        fgm3 = int(row[f"{prefix}FGM3"])
        fga3 = int(row[f"{prefix}FGA3"])
        ftm = int(row[f"{prefix}FTM"])
        fta = int(row[f"{prefix}FTA"])
        offensive_rebounds = int(row[f"{prefix}OR"])
        defensive_rebounds = int(row[f"{prefix}DR"])
        assists = int(row[f"{prefix}Ast"])
        turnovers = int(row[f"{prefix}TO"])

        opp_fga = int(row[f"{opp_prefix}FGA"])
        opp_fgm = int(row[f"{opp_prefix}FGM"])
        opp_fgm3 = int(row[f"{opp_prefix}FGM3"])
        opp_fta = int(row[f"{opp_prefix}FTA"])
        opp_offensive_rebounds = int(row[f"{opp_prefix}OR"])
        opp_defensive_rebounds = int(row[f"{opp_prefix}DR"])
        opp_turnovers = int(row[f"{opp_prefix}TO"])

        aggregate.games += 1
        aggregate.wins += int(won)
        aggregate.losses += int(not won)
        aggregate.points_for += score
        aggregate.points_against += opp_score
        aggregate.possessions_for += _estimate_possessions(fga, offensive_rebounds, turnovers, fta)
        aggregate.possessions_against += _estimate_possessions(opp_fga, opp_offensive_rebounds, opp_turnovers, opp_fta)
        aggregate.fgm += fgm
        aggregate.fga += fga
        aggregate.fgm3 += fgm3
        aggregate.fga3 += fga3
        aggregate.ftm += ftm
        aggregate.fta += fta
        aggregate.off_rebounds += offensive_rebounds
        aggregate.def_rebounds += defensive_rebounds
        aggregate.assists += assists
        aggregate.turnovers += turnovers
        aggregate.opp_fgm += opp_fgm
        aggregate.opp_fga += opp_fga
        aggregate.opp_fgm3 += opp_fgm3
        aggregate.opp_fta += opp_fta
        aggregate.opp_off_rebounds += opp_offensive_rebounds
        aggregate.opp_turnovers += opp_turnovers
        aggregate.off_reb_opportunities += offensive_rebounds + opp_defensive_rebounds
        aggregate.def_reb_opportunities += defensive_rebounds + opp_offensive_rebounds
        aggregate.opponents.append(opponent_id)

    @staticmethod
    def _compute_elo_ratings(games: list[tuple[int, str, str, str, int, int]]) -> Dict[str, float]:
        ratings: Dict[str, float] = defaultdict(lambda: 1500.0)
        for _, winner_id, loser_id, location, winner_score, loser_score in sorted(games):
            winner_rating = ratings[winner_id]
            loser_rating = ratings[loser_id]
            home_edge = 65.0 if location == "H" else -65.0 if location == "A" else 0.0
            expected = 1.0 / (1.0 + 10 ** ((loser_rating - (winner_rating + home_edge)) / 400.0))
            margin = max(1, winner_score - loser_score)
            multiplier = 0.75 + 0.25 * math.log(margin + 1.0)
            delta = 18.0 * multiplier * (1.0 - expected)
            ratings[winner_id] = winner_rating + delta
            ratings[loser_id] = loser_rating - delta
        return dict(ratings)
