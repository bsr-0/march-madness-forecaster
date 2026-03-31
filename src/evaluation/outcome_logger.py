"""Live tournament outcome logging — prediction vs. actual result tracking.

Captures every production prediction alongside actual game outcomes for
post-tournament validation. Designed as a read-only observer of the
frozen production pipeline's outputs.

Usage:
    from src.evaluation.outcome_logger import (
        load_predictions_from_kaggle_csv,
        build_outcome_log,
        save_outcome_log,
    )

    predictions = load_predictions_from_kaggle_csv(
        "artifacts/kaggle_submission_2026.csv",
        "data/kaggle/MTeams.csv",
    )
    games = load_tournament_results(2026)
    log = build_outcome_log(predictions, games, year=2026)
    save_outcome_log(log, "artifacts/outcome_log_2026.json")
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..data.historical_tournament_results import TournamentGame, load_tournament_results

logger = logging.getLogger(__name__)

# Kaggle round-weighted Brier weights (matches kaggle_backtest.py)
_ROUND_WEIGHTS: Dict[str, float] = {
    "FF": 0.5, "First Four": 0.5, "Play-In": 0.5,
    "R64": 1.0, "Round of 64": 1.0, "First Round": 1.0,
    "R32": 2.0, "Round of 32": 2.0, "Second Round": 2.0,
    "S16": 4.0, "Sweet 16": 4.0, "Sweet Sixteen": 4.0,
    "E8": 8.0, "Elite 8": 8.0, "Elite Eight": 8.0,
    "F4": 16.0, "Final Four": 16.0, "Final 4": 16.0,
    "NCG": 32.0, "Championship": 32.0, "Finals": 32.0,
}


@dataclass
class PredictionOutcomeRecord:
    """A single game with its prediction and actual outcome."""

    round_name: str
    region: str
    team1_id: str
    team2_id: str
    team1_seed: int
    team2_seed: int
    predicted_prob_team1: float
    team1_score: int
    team2_score: int
    team1_won: bool

    @property
    def correct(self) -> bool:
        if self.team1_won:
            return self.predicted_prob_team1 >= 0.5
        return self.predicted_prob_team1 < 0.5

    @property
    def upset(self) -> bool:
        if self.team1_won:
            return self.team1_seed > self.team2_seed
        return self.team2_seed > self.team1_seed

    @property
    def brier_component(self) -> float:
        outcome = 1.0 if self.team1_won else 0.0
        return (self.predicted_prob_team1 - outcome) ** 2

    @property
    def log_loss_component(self) -> float:
        outcome = 1.0 if self.team1_won else 0.0
        p = np.clip(self.predicted_prob_team1, 1e-15, 1 - 1e-15)
        return -(outcome * np.log(p) + (1 - outcome) * np.log(1 - p))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["correct"] = self.correct
        d["upset"] = self.upset
        d["brier_component"] = round(self.brier_component, 6)
        d["log_loss_component"] = round(self.log_loss_component, 6)
        return d


@dataclass
class RoundMetrics:
    """Aggregate metrics for a single tournament round."""

    round_name: str
    n_games: int = 0
    brier: float = 0.0
    log_loss: float = 0.0
    accuracy: float = 0.0
    n_upsets: int = 0
    n_upsets_correct: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OutcomeLog:
    """Complete outcome log for a tournament year."""

    year: int
    generated_at: str = ""
    prediction_source: str = ""
    results_source: str = ""
    n_games_with_predictions: int = 0
    n_games_total: int = 0
    n_predictions_unmatched: int = 0
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)
    per_round_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    games: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_predictions_from_kaggle_csv(
    csv_path: str,
    mteams_path: str,
) -> Dict[Tuple[str, str], float]:
    """Load predictions from a Kaggle submission CSV.

    Parses the ``YYYY_TeamID1_TeamID2`` format and maps Kaggle TeamIDs
    to canonical internal team IDs using MTeams.csv and TeamNameResolver.

    Args:
        csv_path: Path to Kaggle submission CSV (ID, Pred columns).
        mteams_path: Path to Kaggle MTeams.csv for ID-to-name mapping.

    Returns:
        Dict mapping (team1_canonical_id, team2_canonical_id) to P(team1 wins).
    """
    import re

    import pandas as pd

    from ..data.team_name_resolver import TeamNameResolver
    from ..exports.kaggle import build_team_id_map, load_kaggle_teams, parse_kaggle_id

    df = pd.read_csv(csv_path)
    if "ID" not in df.columns or "Pred" not in df.columns:
        raise ValueError(f"CSV must have ID and Pred columns, got: {list(df.columns)}")

    team_id_to_name = load_kaggle_teams(mteams_path)
    resolver = TeamNameResolver()
    id_map = build_team_id_map(team_id_to_name, resolver)

    predictions: Dict[Tuple[str, str], float] = {}
    skipped = 0

    for _, row in df.iterrows():
        try:
            season, t1_kaggle, t2_kaggle = parse_kaggle_id(row["ID"])
        except ValueError:
            skipped += 1
            continue

        t1_canonical = id_map.get(t1_kaggle)
        t2_canonical = id_map.get(t2_kaggle)

        if t1_canonical is None or t2_canonical is None:
            skipped += 1
            continue

        predictions[(t1_canonical, t2_canonical)] = float(row["Pred"])

    if skipped:
        logger.warning("Skipped %d rows (bad ID or unmapped team)", skipped)
    logger.info(
        "Loaded %d predictions from %s", len(predictions), csv_path
    )
    return predictions


def _infer_team_ids(matchup_keys: List[str]) -> set[str]:
    """Infer the set of valid team IDs from ambiguous matchup keys.

    For each key like ``"duke_michigan_state"`` we try every split point
    and count how often each candidate ID appears.  IDs that appear many
    times across different matchups are almost certainly real team IDs.
    """
    from collections import Counter

    candidate_counts: Counter[str] = Counter()
    for key in matchup_keys:
        parts = key.split("_")
        for split_idx in range(1, len(parts)):
            left = "_".join(parts[:split_idx])
            right = "_".join(parts[split_idx:])
            candidate_counts[left] += 1
            candidate_counts[right] += 1

    # A real team ID will appear in many matchups (paired with many
    # different opponents).  Use a threshold: at least 3 appearances.
    threshold = max(3, len(matchup_keys) // 100)
    return {cid for cid, count in candidate_counts.items() if count >= threshold}


def _split_matchup_key(
    key: str, team_ids: set[str],
) -> Optional[Tuple[str, str]]:
    """Split ``"teamA_teamB"`` into ``(teamA, teamB)`` using known IDs.

    Tries every possible split point and picks the one where both sides
    are recognized team IDs.  Falls back to the first valid split if
    only one side matches, or a simple 2-part split as a last resort.
    """
    parts = key.split("_")
    if len(parts) == 2:
        return (parts[0], parts[1])

    # Try all split points; prefer splits where both sides are known
    best: Optional[Tuple[str, str]] = None
    for split_idx in range(1, len(parts)):
        left = "_".join(parts[:split_idx])
        right = "_".join(parts[split_idx:])
        left_known = left in team_ids
        right_known = right in team_ids
        if left_known and right_known:
            return (left, right)
        if (left_known or right_known) and best is None:
            best = (left, right)

    return best


def load_predictions_from_report(
    report_path: str,
    known_team_ids: Optional[List[str]] = None,
) -> Dict[Tuple[str, str], float]:
    """Load predictions from a production report JSON or forecaster output.

    Prediction keys use the format ``"{team1_id}_{team2_id}"`` where team
    IDs themselves may contain underscores (e.g. ``michigan_state``).  To
    parse these ambiguous keys we need the set of valid team IDs.  If
    *known_team_ids* is not supplied the function attempts to infer IDs
    from the prediction keys by testing every possible split point.

    Args:
        report_path: Path to JSON report file.
        known_team_ids: Optional list of canonical team IDs.  Dramatically
            improves parse accuracy for multi-word team names.

    Returns:
        Dict mapping (team1_id, team2_id) to P(team1 wins).
    """
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    # Try several known key patterns
    pred_dict: Optional[Dict[str, float]] = None
    for key in report:
        if key.startswith("predictions"):
            pred_dict = report[key]
            break

    if pred_dict is None:
        raise ValueError(
            f"No predictions_* key found in {report_path}. "
            f"Available keys: {list(report.keys())}"
        )

    # Build a set of known IDs for disambiguation
    team_set: set[str] = set(known_team_ids) if known_team_ids else set()
    if not team_set:
        # Infer team IDs: collect all keys, try all split points, and
        # keep IDs that appear on both sides of many matchups.
        team_set = _infer_team_ids(list(pred_dict.keys()))

    predictions: Dict[Tuple[str, str], float] = {}
    skipped = 0
    for matchup_key, prob in pred_dict.items():
        if "__vs__" in matchup_key:
            t1, t2 = matchup_key.split("__vs__", 1)
            predictions[(t1, t2)] = float(prob)
            continue

        parsed = _split_matchup_key(matchup_key, team_set)
        if parsed is not None:
            predictions[parsed] = float(prob)
        else:
            skipped += 1
            logger.debug("Skipping unparseable matchup key: %s", matchup_key)

    if skipped:
        logger.warning(
            "Skipped %d/%d unparseable matchup keys in %s",
            skipped, len(pred_dict), report_path,
        )

    logger.info("Loaded %d predictions from %s", len(predictions), report_path)
    return predictions


def _lookup_prediction(
    predictions: Dict[Tuple[str, str], float],
    team1_id: str,
    team2_id: str,
) -> Optional[float]:
    """Look up a prediction, trying both directions."""
    pred = predictions.get((team1_id, team2_id))
    if pred is not None:
        return pred
    pred = predictions.get((team2_id, team1_id))
    if pred is not None:
        return 1.0 - pred
    return None


def build_outcome_log(
    predictions: Dict[Tuple[str, str], float],
    actual_games: List[TournamentGame],
    year: int,
    prediction_source: str = "",
    results_source: str = "",
) -> OutcomeLog:
    """Join predictions with actual outcomes and compute metrics.

    Args:
        predictions: (team1_id, team2_id) -> P(team1 wins).
        actual_games: List of TournamentGame results.
        year: Tournament year.
        prediction_source: Description of where predictions came from.
        results_source: Description of where results came from.

    Returns:
        OutcomeLog with per-game records and aggregate metrics.
    """
    records: List[PredictionOutcomeRecord] = []
    unmatched = 0

    for game in actual_games:
        pred = _lookup_prediction(predictions, game.team1_id, game.team2_id)
        if pred is None:
            unmatched += 1
            logger.warning(
                "No prediction for %s vs %s (%s %s)",
                game.team1_id, game.team2_id, game.round_name, game.region,
            )
            continue

        records.append(PredictionOutcomeRecord(
            round_name=game.round_name,
            region=game.region,
            team1_id=game.team1_id,
            team2_id=game.team2_id,
            team1_seed=game.team1_seed,
            team2_seed=game.team2_seed,
            predicted_prob_team1=pred,
            team1_score=game.team1_score,
            team2_score=game.team2_score,
            team1_won=game.team1_won,
        ))

    # --- Aggregate metrics ---
    if not records:
        return OutcomeLog(
            year=year,
            generated_at=datetime.now(timezone.utc).isoformat(),
            prediction_source=prediction_source,
            results_source=results_source,
            n_games_with_predictions=0,
            n_games_total=len(actual_games),
            n_predictions_unmatched=unmatched,
        )

    brier_scores = [r.brier_component for r in records]
    log_losses = [r.log_loss_component for r in records]
    correct_count = sum(1 for r in records if r.correct)
    upsets = [r for r in records if r.upset]
    upsets_correct = sum(1 for r in records if r.upset and r.correct)

    # Round-weighted Brier (Kaggle 2023+ metric)
    weighted_se_sum = 0.0
    weight_sum = 0.0
    for r in records:
        w = _ROUND_WEIGHTS.get(r.round_name, 1.0)
        weighted_se_sum += w * r.brier_component
        weight_sum += w
    rw_brier = weighted_se_sum / weight_sum if weight_sum > 0 else 0.0

    aggregate = {
        "brier_score": round(float(np.mean(brier_scores)), 6),
        "log_loss": round(float(np.mean(log_losses)), 6),
        "accuracy": round(correct_count / len(records), 4),
        "round_weighted_brier": round(rw_brier, 6),
        "n_correct": correct_count,
        "n_upsets_total": len(upsets),
        "n_upsets_correct": upsets_correct,
        "upset_accuracy": (
            round(upsets_correct / len(upsets), 4) if upsets else 0.0
        ),
    }

    # --- Per-round metrics ---
    round_groups: Dict[str, List[PredictionOutcomeRecord]] = {}
    for r in records:
        round_groups.setdefault(r.round_name, []).append(r)

    per_round: Dict[str, Dict[str, Any]] = {}
    for rnd, group in sorted(round_groups.items()):
        rnd_brier = [r.brier_component for r in group]
        rnd_ll = [r.log_loss_component for r in group]
        rnd_correct = sum(1 for r in group if r.correct)
        rnd_upsets = sum(1 for r in group if r.upset)
        rnd_upsets_correct = sum(1 for r in group if r.upset and r.correct)
        per_round[rnd] = RoundMetrics(
            round_name=rnd,
            n_games=len(group),
            brier=round(float(np.mean(rnd_brier)), 6),
            log_loss=round(float(np.mean(rnd_ll)), 6),
            accuracy=round(rnd_correct / len(group), 4),
            n_upsets=rnd_upsets,
            n_upsets_correct=rnd_upsets_correct,
        ).to_dict()

    return OutcomeLog(
        year=year,
        generated_at=datetime.now(timezone.utc).isoformat(),
        prediction_source=prediction_source,
        results_source=results_source,
        n_games_with_predictions=len(records),
        n_games_total=len(actual_games),
        n_predictions_unmatched=unmatched,
        aggregate_metrics=aggregate,
        per_round_metrics=per_round,
        games=[r.to_dict() for r in records],
    )


def save_outcome_log(log: OutcomeLog, path: str) -> None:
    """Save outcome log to JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(log.to_dict(), f, indent=2)
    logger.info("Outcome log saved to %s (%d games)", path, log.n_games_with_predictions)


def load_outcome_log(path: str) -> OutcomeLog:
    """Load a previously saved outcome log."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return OutcomeLog(**{
        k: v for k, v in data.items()
        if k in OutcomeLog.__dataclass_fields__
    })


def format_summary(log: OutcomeLog) -> str:
    """Format a human-readable summary of the outcome log."""
    lines = [
        "=" * 60,
        f"OUTCOME LOG — {log.year} TOURNAMENT",
        "=" * 60,
        f"  Generated:    {log.generated_at}",
        f"  Predictions:  {log.prediction_source}",
        f"  Results:      {log.results_source}",
        f"  Games scored: {log.n_games_with_predictions} / {log.n_games_total}",
        "",
    ]

    m = log.aggregate_metrics
    if m:
        lines.extend([
            "  AGGREGATE METRICS",
            f"    Brier score:          {m.get('brier_score', 'N/A')}",
            f"    Log loss:             {m.get('log_loss', 'N/A')}",
            f"    Round-weighted Brier: {m.get('round_weighted_brier', 'N/A')}",
            f"    Accuracy:             {m.get('accuracy', 'N/A')}",
            f"    Upsets correct:       {m.get('n_upsets_correct', 0)}/{m.get('n_upsets_total', 0)}",
            "",
        ])

    if log.per_round_metrics:
        lines.append("  PER-ROUND BREAKDOWN")
        round_order = ["FF", "R64", "R32", "S16", "E8", "F4", "NCG"]
        for rnd in round_order:
            rm = log.per_round_metrics.get(rnd)
            if rm:
                lines.append(
                    f"    {rnd:>3s}: Brier={rm['brier']:.4f}  "
                    f"Acc={rm['accuracy']:.1%}  "
                    f"Upsets={rm['n_upsets_correct']}/{rm['n_upsets']}  "
                    f"(n={rm['n_games']})"
                )
        lines.append("")

    if log.n_predictions_unmatched:
        lines.append(
            f"  WARNING: {log.n_predictions_unmatched} games had no matching prediction"
        )

    lines.append("=" * 60)
    return "\n".join(lines)
