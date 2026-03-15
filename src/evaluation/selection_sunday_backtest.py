"""Selection Sunday Snapshot Backtest Framework.

The primary historical validation harness for this repository. Reconstructs
exactly what would have been known at the moment the bracket was released
for each tournament year, then evaluates model predictions under those
conditions.

Critical invariants:
1. Both training AND evaluation data use Selection Sunday snapshots.
2. Feature computation occurs AFTER timestamp filtering.
3. Cross-validation is strictly chronological expanding-window.
4. Play-in teams are modeled as probabilistic entrants.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from .bracket_snapshot import (
    BracketSnapshot,
    PlayInMatchup,
    load_bracket_snapshot,
    marginalise_play_in,
)
from .snapshot_integrity import (
    DataSourceRecord,
    SnapshotIntegrityValidator,
    SnapshotMetadata,
    compute_feature_hash,
    get_selection_sunday,
    selection_sunday_as_datetime,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Snapshot data structure
# ---------------------------------------------------------------------------

@dataclass
class SelectionSundaySnapshot:
    """A complete snapshot of information available on Selection Sunday.

    Contains the feature matrix, bracket structure, and metadata for a
    single tournament year. All data in this snapshot must predate the
    Selection Sunday cutoff.
    """

    year: int
    feature_matrix: Optional[np.ndarray] = None
    feature_names: List[str] = field(default_factory=list)
    team_ids: List[str] = field(default_factory=list)
    team_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    bracket: Optional[BracketSnapshot] = None
    metadata: Optional[SnapshotMetadata] = None

    # Training sample data (games before Selection Sunday)
    training_X: Optional[np.ndarray] = None
    training_y: Optional[np.ndarray] = None
    training_margins: Optional[np.ndarray] = None
    training_round_weights: Optional[np.ndarray] = None

    @property
    def feature_hash(self) -> str:
        if self.feature_matrix is not None:
            return compute_feature_hash(self.feature_matrix.tobytes())
        return ""

    @property
    def selection_sunday(self) -> date:
        return get_selection_sunday(self.year)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "year": self.year,
            "selection_sunday": self.selection_sunday.isoformat(),
            "feature_hash": self.feature_hash,
            "n_teams": len(self.team_ids),
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
        }
        if self.metadata:
            result["metadata"] = self.metadata.to_dict()
        if self.bracket:
            result["bracket_summary"] = {
                "n_teams": self.bracket.n_teams,
                "has_play_ins": self.bracket.has_play_ins,
                "n_play_in_matchups": len(self.bracket.play_in_matchups),
            }
        if self.training_X is not None:
            result["training_samples"] = int(self.training_X.shape[0])
        return result


# ---------------------------------------------------------------------------
# Year backtest result
# ---------------------------------------------------------------------------

@dataclass
class YearBacktestResult:
    """Results from backtesting a single tournament year."""

    year: int
    predictions: Dict[Tuple[str, str], float] = field(default_factory=dict)
    training_years: List[int] = field(default_factory=list)
    snapshot_metadata: Optional[SnapshotMetadata] = None
    feature_hash: str = ""
    selection_timestamp: str = ""
    model_artifact_ref: str = ""

    def to_dict(self) -> Dict[str, Any]:
        preds_serializable = {
            f"{k[0]}__vs__{k[1]}": v for k, v in self.predictions.items()
        }
        return {
            "year": self.year,
            "predictions": preds_serializable,
            "n_predictions": len(self.predictions),
            "training_years": self.training_years,
            "feature_hash": self.feature_hash,
            "selection_timestamp": self.selection_timestamp,
            "model_artifact_ref": self.model_artifact_ref,
        }


# ---------------------------------------------------------------------------
# Snapshot builder
# ---------------------------------------------------------------------------

class SnapshotBuilder:
    """Builds Selection Sunday snapshots from historical data.

    Enforces the critical invariant: raw data is filtered by timestamp
    BEFORE any feature computation occurs.
    """

    def __init__(
        self,
        historical_dir: str = "data/raw/historical",
        bracket_dir: str = "data/raw",
    ):
        self.historical_dir = Path(historical_dir)
        self.bracket_dir = Path(bracket_dir)

    def build_snapshot(self, year: int) -> SelectionSundaySnapshot:
        """Build a Selection Sunday snapshot for a single year.

        Pipeline:
        1. Determine Selection Sunday cutoff
        2. Load and filter raw game data by date
        3. Compute features from filtered data only (IncrementalMetricsEngine)
        4. Load bracket structure
        5. Validate integrity
        6. Return snapshot

        Args:
            year: Tournament year to build snapshot for.

        Returns:
            SelectionSundaySnapshot with all pre-Selection-Sunday data.
        """
        selection_date = get_selection_sunday(year)
        cutoff = selection_sunday_as_datetime(year)
        logger.info("Building Selection Sunday snapshot for %d (cutoff: %s)", year, selection_date)

        # Step 1: Load and filter games
        games, games_included, games_excluded = self._load_filtered_games(year, selection_date)

        # Step 2: Compute features from filtered games ONLY
        # This is where IncrementalMetricsEngine would be invoked.
        # The engine receives only pre-cutoff games, ensuring features
        # contain no future information.
        team_metrics, feature_matrix, feature_names, team_ids = (
            self._compute_features_from_filtered_games(year, games)
        )

        # Step 3: Build training samples from filtered games
        training_X, training_y, training_margins, training_rw = (
            self._build_training_samples(year, games, team_metrics)
        )

        # Step 4: Load bracket
        bracket = self._load_bracket(year)

        # Step 5: Build metadata
        data_sources = self._collect_data_sources(year, games_included, selection_date)
        metadata = SnapshotMetadata(
            year=year,
            selection_sunday=selection_date,
            cutoff_datetime=cutoff,
            data_sources=data_sources,
            games_included=games_included,
            games_excluded=games_excluded,
            teams_count=len(team_ids),
            feature_hash=compute_feature_hash(
                feature_matrix.tobytes() if feature_matrix is not None else b""
            ),
            features_computed_after_filter=True,
        )

        # Step 6: Validate
        validator = SnapshotIntegrityValidator(year)
        validator.validate_or_raise(metadata)

        snapshot = SelectionSundaySnapshot(
            year=year,
            feature_matrix=feature_matrix,
            feature_names=feature_names,
            team_ids=team_ids,
            team_metrics=team_metrics,
            bracket=bracket,
            metadata=metadata,
            training_X=training_X,
            training_y=training_y,
            training_margins=training_margins,
            training_round_weights=training_rw,
        )

        logger.info(
            "Snapshot for %d: %d games, %d teams, %d features, hash=%s",
            year, games_included, len(team_ids), len(feature_names),
            snapshot.feature_hash[:8],
        )
        return snapshot

    def _load_filtered_games(
        self,
        year: int,
        cutoff_date: date,
    ) -> Tuple[List[Dict[str, Any]], int, int]:
        """Load games and filter to only those before Selection Sunday.

        Returns:
            (filtered_games, included_count, excluded_count)
        """
        games_path = self.historical_dir / f"historical_games_{year}.json"
        if not games_path.exists():
            logger.warning("No games file for %d at %s", year, games_path)
            return [], 0, 0

        with open(games_path) as f:
            all_games = json.load(f)

        if isinstance(all_games, dict):
            all_games = all_games.get("games", [])

        filtered = []
        excluded = 0
        for game in all_games:
            game_date_str = game.get("date", game.get("game_date", ""))
            if not game_date_str:
                # No date field — include conservatively if no round info
                if game.get("round_name") or game.get("tournament_round"):
                    excluded += 1
                    continue
                filtered.append(game)
                continue

            try:
                if isinstance(game_date_str, str):
                    game_date = date.fromisoformat(game_date_str[:10])
                else:
                    game_date = game_date_str
            except (ValueError, TypeError):
                filtered.append(game)
                continue

            if game_date < cutoff_date:
                filtered.append(game)
            else:
                excluded += 1

        return filtered, len(filtered), excluded

    def _compute_features_from_filtered_games(
        self,
        year: int,
        filtered_games: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Dict[str, float]], Optional[np.ndarray], List[str], List[str]]:
        """Compute features using only pre-Selection-Sunday games.

        This is the critical step: features are derived from filtered data,
        not from pre-computed season aggregates.

        Returns:
            (team_metrics, feature_matrix, feature_names, team_ids)
        """
        if not filtered_games:
            return {}, None, [], []

        # Build team-level metrics from filtered games
        team_metrics: Dict[str, Dict[str, float]] = {}
        team_game_counts: Dict[str, int] = {}

        for game in filtered_games:
            for side in ("home", "away", "team1", "team2"):
                team_id = game.get(f"{side}_id", game.get(f"{side}_team_id", ""))
                if not team_id:
                    continue
                if team_id not in team_metrics:
                    team_metrics[team_id] = {
                        "games_played": 0,
                        "wins": 0,
                        "total_score": 0.0,
                        "total_opp_score": 0.0,
                    }
                    team_game_counts[team_id] = 0
                team_game_counts[team_id] += 1
                team_metrics[team_id]["games_played"] = team_game_counts[team_id]

        team_ids = sorted(team_metrics.keys())
        feature_names = ["games_played", "wins", "win_pct"]

        if not team_ids:
            return team_metrics, None, feature_names, team_ids

        # Build basic feature matrix
        n_teams = len(team_ids)
        n_features = len(feature_names)
        feature_matrix = np.zeros((n_teams, n_features))

        for i, tid in enumerate(team_ids):
            metrics = team_metrics[tid]
            gp = metrics.get("games_played", 0)
            wins = metrics.get("wins", 0)
            feature_matrix[i, 0] = gp
            feature_matrix[i, 1] = wins
            feature_matrix[i, 2] = wins / max(gp, 1)

        return team_metrics, feature_matrix, feature_names, team_ids

    def _build_training_samples(
        self,
        year: int,
        filtered_games: List[Dict[str, Any]],
        team_metrics: Dict[str, Dict[str, float]],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Build game-level training samples from filtered games.

        Each sample represents a matchup with features derived from
        team-level metrics computed from pre-Selection-Sunday data only.

        Returns:
            (X, y, margins, round_weights) or (None, None, None, None)
        """
        if not filtered_games or not team_metrics:
            return None, None, None, None

        samples_X = []
        samples_y = []
        samples_margins = []
        samples_weights = []

        for game in filtered_games:
            # Extract team IDs
            t1 = game.get("home_id", game.get("team1_id", ""))
            t2 = game.get("away_id", game.get("team2_id", ""))
            if not t1 or not t2 or t1 not in team_metrics or t2 not in team_metrics:
                continue

            # Game outcome
            t1_score = game.get("home_score", game.get("team1_score", 0))
            t2_score = game.get("away_score", game.get("team2_score", 0))
            if t1_score == 0 and t2_score == 0:
                continue

            t1_won = 1.0 if t1_score > t2_score else 0.0
            margin = t1_score - t2_score

            # Feature vector: diff of team metrics
            m1 = team_metrics[t1]
            m2 = team_metrics[t2]
            wp1 = m1.get("wins", 0) / max(m1.get("games_played", 1), 1)
            wp2 = m2.get("wins", 0) / max(m2.get("games_played", 1), 1)
            features = [wp1 - wp2]

            samples_X.append(features)
            samples_y.append(t1_won)
            samples_margins.append(margin)
            samples_weights.append(1.0)

        if not samples_X:
            return None, None, None, None

        return (
            np.array(samples_X),
            np.array(samples_y),
            np.array(samples_margins),
            np.array(samples_weights),
        )

    def _load_bracket(self, year: int) -> Optional[BracketSnapshot]:
        """Load bracket snapshot for a year."""
        try:
            return load_bracket_snapshot(
                year,
                data_dir=str(self.historical_dir),
                bracket_dir=str(self.bracket_dir),
            )
        except FileNotFoundError:
            logger.warning("No bracket data for %d", year)
            return None

    def _collect_data_sources(
        self,
        year: int,
        games_count: int,
        selection_date: date,
    ) -> List[DataSourceRecord]:
        """Collect metadata about data sources used in the snapshot."""
        sources = []
        cutoff = selection_sunday_as_datetime(year)

        games_path = self.historical_dir / f"historical_games_{year}.json"
        if games_path.exists():
            sources.append(DataSourceRecord(
                name=f"historical_games_{year}",
                source_type="games",
                timestamp=cutoff,
                record_count=games_count,
                file_path=str(games_path),
            ))

        metrics_path = self.historical_dir / f"team_metrics_{year}.json"
        if metrics_path.exists():
            sources.append(DataSourceRecord(
                name=f"team_metrics_{year}",
                source_type="metrics",
                timestamp=cutoff,
                record_count=0,
                file_path=str(metrics_path),
            ))

        return sources


# ---------------------------------------------------------------------------
# Selection Sunday Backtester
# ---------------------------------------------------------------------------

class SelectionSundayBacktester:
    """The primary historical validation harness.

    For each evaluation year:
    1. Build Selection Sunday snapshots for all training years
    2. Build Selection Sunday snapshot for the evaluation year
    3. Train model using only snapshot-based training data
    4. Generate predictions for the evaluation year bracket
    5. Store predictions and metadata

    Training data is ALWAYS snapshot-based: each training year uses its
    own Selection Sunday snapshot, not full-season data.

    Cross-validation is strictly chronological expanding-window:
    train on snapshots from years < eval_year, predict eval_year.
    """

    def __init__(
        self,
        predict_fn_factory: Optional[Callable] = None,
        snapshot_builder: Optional[SnapshotBuilder] = None,
        historical_dir: str = "data/raw/historical",
        bracket_dir: str = "data/raw",
    ):
        """
        Args:
            predict_fn_factory: Factory that takes (training_snapshots, eval_snapshot)
                and returns a predict_fn: (team1_id, team2_id) -> float.
                If None, a seed-based baseline is used.
            snapshot_builder: Custom snapshot builder. If None, default is created.
            historical_dir: Directory with historical data.
            bracket_dir: Directory with bracket data.
        """
        self.predict_fn_factory = predict_fn_factory
        self.snapshot_builder = snapshot_builder or SnapshotBuilder(
            historical_dir=historical_dir,
            bracket_dir=bracket_dir,
        )
        self._snapshot_cache: Dict[int, SelectionSundaySnapshot] = {}

    def build_snapshot(self, year: int, use_cache: bool = True) -> SelectionSundaySnapshot:
        """Build or retrieve a cached Selection Sunday snapshot."""
        if use_cache and year in self._snapshot_cache:
            return self._snapshot_cache[year]
        snapshot = self.snapshot_builder.build_snapshot(year)
        if use_cache:
            self._snapshot_cache[year] = snapshot
        return snapshot

    def run_backtest(
        self,
        evaluation_years: List[int],
        first_training_year: Optional[int] = None,
    ) -> List[YearBacktestResult]:
        """Run the full Selection Sunday backtest.

        For each evaluation year:
        1. Build snapshots for training years (< eval_year)
        2. Build snapshot for eval year
        3. Train model on training snapshots
        4. Generate predictions
        5. Return results

        Args:
            evaluation_years: Years to evaluate (predict).
            first_training_year: Earliest year to include in training.
                If None, uses the earliest available year.

        Returns:
            List of YearBacktestResult, one per evaluation year.
        """
        results: List[YearBacktestResult] = []

        for eval_year in sorted(evaluation_years):
            logger.info("=" * 60)
            logger.info("Backtesting year %d", eval_year)
            logger.info("=" * 60)

            # Determine training years (strictly < eval_year)
            training_years = [
                y for y in sorted(self._get_available_years())
                if y < eval_year and y != 2020  # Exclude COVID year
            ]
            if first_training_year is not None:
                training_years = [y for y in training_years if y >= first_training_year]

            if not training_years:
                logger.warning("No training years available for %d, skipping", eval_year)
                continue

            # Build snapshots — training years use their own Selection Sunday snapshots
            train_snapshots = []
            for ty in training_years:
                try:
                    snap = self.build_snapshot(ty)
                    train_snapshots.append(snap)
                except Exception as e:
                    logger.warning("Failed to build snapshot for training year %d: %s", ty, e)

            # Build evaluation snapshot
            try:
                eval_snapshot = self.build_snapshot(eval_year)
            except Exception as e:
                logger.error("Failed to build eval snapshot for %d: %s", eval_year, e)
                continue

            # Generate predictions
            predictions = self._generate_predictions(
                train_snapshots, eval_snapshot,
            )

            result = YearBacktestResult(
                year=eval_year,
                predictions=predictions,
                training_years=training_years,
                snapshot_metadata=eval_snapshot.metadata,
                feature_hash=eval_snapshot.feature_hash,
                selection_timestamp=selection_sunday_as_datetime(eval_year).isoformat(),
            )
            results.append(result)

            logger.info(
                "Year %d: %d predictions, trained on %d years (%s)",
                eval_year, len(predictions), len(training_years),
                f"{training_years[0]}-{training_years[-1]}" if training_years else "none",
            )

        return results

    def _generate_predictions(
        self,
        train_snapshots: List[SelectionSundaySnapshot],
        eval_snapshot: SelectionSundaySnapshot,
    ) -> Dict[Tuple[str, str], float]:
        """Generate predictions for all tournament matchups.

        If a predict_fn_factory is provided, it is used to create the
        prediction function. Otherwise, a seed-based baseline is used.

        Play-in matchups are marginalised over probabilistically.
        """
        if self.predict_fn_factory is not None:
            predict_fn = self.predict_fn_factory(train_snapshots, eval_snapshot)
        else:
            predict_fn = self._seed_baseline_predict_fn(eval_snapshot)

        predictions: Dict[Tuple[str, str], float] = {}

        if eval_snapshot.bracket is None:
            logger.warning("No bracket for %d, cannot generate predictions", eval_snapshot.year)
            return predictions

        bracket = eval_snapshot.bracket
        non_play_in_teams = [
            t.team_id for t in bracket.teams if not t.is_play_in
        ]
        play_in_teams = set(bracket.get_play_in_teams())

        # Generate predictions for all possible matchups
        all_team_ids = [t.team_id for t in bracket.teams]
        for i, t1 in enumerate(all_team_ids):
            for j, t2 in enumerate(all_team_ids):
                if i >= j:
                    continue

                t1_is_play_in = t1 in play_in_teams
                t2_is_play_in = t2 in play_in_teams

                if not t1_is_play_in and not t2_is_play_in:
                    # Normal matchup
                    p = predict_fn(t1, t2)
                elif t1_is_play_in and not t2_is_play_in:
                    # t1 is in a play-in; marginalise
                    pim = self._find_play_in_for_team(bracket, t1)
                    if pim:
                        p = 1.0 - marginalise_play_in(predict_fn, pim, t2)
                    else:
                        p = predict_fn(t1, t2)
                elif not t1_is_play_in and t2_is_play_in:
                    # t2 is in a play-in; marginalise
                    pim = self._find_play_in_for_team(bracket, t2)
                    if pim:
                        p = marginalise_play_in(predict_fn, pim, t1)
                    else:
                        p = predict_fn(t1, t2)
                else:
                    # Both play-in teams
                    p = predict_fn(t1, t2)

                predictions[(t1, t2)] = float(np.clip(p, 0.001, 0.999))
                predictions[(t2, t1)] = float(np.clip(1.0 - p, 0.001, 0.999))

        return predictions

    def _find_play_in_for_team(
        self,
        bracket: BracketSnapshot,
        team_id: str,
    ) -> Optional[PlayInMatchup]:
        """Find the play-in matchup for a given team."""
        for pim in bracket.play_in_matchups:
            if team_id in (pim.team_a, pim.team_b):
                return pim
        return None

    def _seed_baseline_predict_fn(
        self,
        eval_snapshot: SelectionSundaySnapshot,
    ) -> Callable[[str, str], float]:
        """Create a seed-based baseline prediction function.

        Uses historical upset rates by seed matchup.
        """
        from ..simulation.monte_carlo import HISTORICAL_UPSET_RATES

        seeds = eval_snapshot.bracket.seeds if eval_snapshot.bracket else {}

        def predict(team1: str, team2: str) -> float:
            s1 = seeds.get(team1, 8)
            s2 = seeds.get(team2, 8)

            if s1 == s2:
                return 0.5

            # Orient so that lower seed (better) is the favorite
            fav_seed = min(s1, s2)
            dog_seed = max(s1, s2)

            upset_rate = HISTORICAL_UPSET_RATES.get((fav_seed, dog_seed), 0.5)
            p_fav_wins = 1.0 - upset_rate

            # Return P(team1 wins)
            if s1 < s2:
                return p_fav_wins
            else:
                return 1.0 - p_fav_wins

        return predict

    def _get_available_years(self) -> List[int]:
        """Get years for which historical data exists."""
        from .snapshot_integrity import SELECTION_SUNDAY_DATES
        return sorted(SELECTION_SUNDAY_DATES.keys())


# ---------------------------------------------------------------------------
# Manifest for lineage tracking
# ---------------------------------------------------------------------------

@dataclass
class SelectionSundayManifest:
    """Manifest for a Selection Sunday backtest run, for lineage tracking."""

    run_timestamp: str = ""
    evaluation_years: List[int] = field(default_factory=list)
    results: List[Dict[str, Any]] = field(default_factory=list)
    random_seed: Optional[int] = None
    library_versions: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "selection_sunday_backtest",
            "run_timestamp": self.run_timestamp,
            "evaluation_years": self.evaluation_years,
            "results": self.results,
            "random_seed": self.random_seed,
            "library_versions": self.library_versions,
        }

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
