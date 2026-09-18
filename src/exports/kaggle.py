"""Kaggle March Machine Learning Mania submission format.

The competition scores one CSV -- ``ID,Pred`` with ``ID = SEASON_TEAM1_TEAM2``
and ``Pred = P(TEAM1 beats TEAM2)`` -- over every game played in BOTH the
men's and women's tournaments, plain Brier, no round weights. The sample
submission lists every pair of D1 teams (about 65k rows per gender for a
season); only the ~130 pairs that actually meet are scored, so an unmapped
or non-tournament pair filled with 0.5 costs nothing.

This module only knows the format. It takes prediction callbacks keyed by
Kaggle TeamID -- men's below 3000, women's at or above -- and leaves the
question of which model answers to scripts/kaggle_submission.py. Restored
from the 2026-05 Kaggle track (git 54a80b1:src/exports/kaggle.py) and cut
down: the old version resolved Kaggle names to canonical ids through the
fuzzy TeamNameResolver; the men's path now bridges through
src/prediction/kaggle_bridge (spellings table, exact) and the women's model
speaks Kaggle ids natively, so the resolver and its confidence thresholds
are gone.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

_KAGGLE_ID_RE = re.compile(r"^(\d{4})_(\d+)_(\d+)$")

# TeamID boundary between men's and women's in Kaggle data.
WOMENS_TEAM_ID_THRESHOLD = 3000

# Neutral fill for rows nothing can predict. Only unscored rows should get it.
DEFAULT_PRED = 0.5

# (team1_kaggle_id, team2_kaggle_id) -> P(team1 wins), or None for "no opinion".
PredictFn = Callable[[int, int], Optional[float]]


def is_womens_team(kaggle_team_id: int) -> bool:
    return int(kaggle_team_id) >= WOMENS_TEAM_ID_THRESHOLD


def parse_kaggle_id(id_str: str) -> Tuple[int, int, int]:
    """'YYYY_Team1_Team2' -> (season, team1, team2)."""
    if id_str is None:
        raise ValueError("Kaggle ID is None")
    match = _KAGGLE_ID_RE.match(str(id_str).strip())
    if not match:
        raise ValueError(f"Invalid Kaggle ID: {id_str}")
    season, team1, team2 = match.groups()
    return int(season), int(team1), int(team2)


def load_kaggle_teams(path: Path) -> Dict[int, str]:
    """TeamID -> TeamName from MTeams.csv / WTeams.csv (same layout)."""
    team_map: Dict[int, str] = {}
    with open(path, newline="", encoding="latin-1") as f:
        reader = csv.DictReader(f)
        if "TeamID" not in reader.fieldnames or "TeamName" not in reader.fieldnames:
            raise ValueError(f"{path} must contain TeamID and TeamName columns")
        for row in reader:
            try:
                team_map[int(row["TeamID"])] = row["TeamName"].strip()
            except (KeyError, TypeError, ValueError):
                logger.warning("Skipping invalid team row: %s", row)
    return team_map


@dataclass
class KaggleExportStats:
    total_rows: int = 0
    bad_id_rows: int = 0
    season_mismatch: int = 0
    mens_rows: int = 0
    womens_rows: int = 0
    mens_predicted: int = 0
    womens_predicted: int = 0
    predict_failures: int = 0

    @property
    def predicted_rows(self) -> int:
        return self.mens_predicted + self.womens_predicted

    @property
    def defaulted_rows(self) -> int:
        return self.total_rows - self.predicted_rows

    def to_dict(self) -> Dict[str, int]:
        return {
            "total_rows": self.total_rows,
            "predicted_rows": self.predicted_rows,
            "defaulted_rows": self.defaulted_rows,
            "bad_id_rows": self.bad_id_rows,
            "season_mismatch": self.season_mismatch,
            "mens_rows": self.mens_rows,
            "womens_rows": self.womens_rows,
            "mens_predicted": self.mens_predicted,
            "womens_predicted": self.womens_predicted,
            "predict_failures": self.predict_failures,
        }


def generate_predictions(
    sample_df: pd.DataFrame,
    mens_predict: Optional[PredictFn],
    womens_predict: Optional[PredictFn],
    season_filter: Optional[int] = None,
) -> pd.DataFrame:
    """Fill the sample submission's ``Pred`` column.

    Rows are routed by TeamID range. A callback returning ``None`` (or a
    missing callback) leaves the row at DEFAULT_PRED; a callback that raises
    is counted as a failure and also defaulted, so one bad pair cannot sink
    a 130k-row file. The stats land in ``out.attrs["kaggle_export_stats"]``
    so the caller can refuse to write a file that predicted nothing.
    """
    if "ID" not in sample_df.columns:
        raise ValueError("Sample submission must contain an 'ID' column")

    out = sample_df.copy()
    stats = KaggleExportStats(total_rows=len(out))
    preds = []
    for raw_id in out["ID"].astype(str).tolist():
        try:
            season, team1, team2 = parse_kaggle_id(raw_id)
        except ValueError:
            stats.bad_id_rows += 1
            preds.append(DEFAULT_PRED)
            continue
        if season_filter is not None and season != season_filter:
            stats.season_mismatch += 1
            preds.append(DEFAULT_PRED)
            continue

        womens = is_womens_team(team1) or is_womens_team(team2)
        if womens:
            stats.womens_rows += 1
            fn = womens_predict
        else:
            stats.mens_rows += 1
            fn = mens_predict
        if fn is None:
            preds.append(DEFAULT_PRED)
            continue

        try:
            pred = fn(team1, team2)
        except Exception:  # noqa: BLE001 - one bad pair must not sink the file
            logger.debug("prediction failed for %s", raw_id, exc_info=True)
            stats.predict_failures += 1
            preds.append(DEFAULT_PRED)
            continue
        if pred is None:
            preds.append(DEFAULT_PRED)
            continue

        if womens:
            stats.womens_predicted += 1
        else:
            stats.mens_predicted += 1
        preds.append(min(1.0, max(0.0, float(pred))))

    out["Pred"] = preds
    out.attrs["kaggle_export_stats"] = stats.to_dict()

    if stats.bad_id_rows:
        logger.warning("Kaggle export: %d rows with invalid IDs", stats.bad_id_rows)
    if stats.predict_failures:
        logger.warning(
            "Kaggle export: %d prediction failures (defaulted to %.1f)", stats.predict_failures, DEFAULT_PRED
        )
    if stats.predicted_rows == 0 and stats.total_rows:
        logger.error("Kaggle export: zero predictions out of %d rows", stats.total_rows)
    return out


def validate_submission(df: pd.DataFrame, sample_df: pd.DataFrame) -> None:
    """Raise if the frame is not something Kaggle will accept."""
    if list(df.columns) != ["ID", "Pred"]:
        raise ValueError(f"submission columns must be exactly ['ID', 'Pred'], got {list(df.columns)}")
    if len(df) != len(sample_df):
        raise ValueError(f"submission has {len(df)} rows; sample has {len(sample_df)}")
    if not df["ID"].equals(sample_df["ID"]):
        raise ValueError("submission IDs do not match the sample submission in order")
    if df["ID"].duplicated().any():
        raise ValueError("duplicate IDs in submission")
    p = df["Pred"].astype(float)
    if p.isna().any() or (p < 0).any() or (p > 1).any():
        raise ValueError("Pred must be a probability in [0, 1] on every row")


# ------------------------------------------------------------------ slot 2
def strongest_team(
    pairwise: Dict[Tuple[str, str], float], team_ids: List[str], top: int = 5
) -> List[Tuple[str, float]]:
    """Teams ranked by mean P(beats X) over the field, best first.

    This is the champion rule for the hedge. It is a strength ranking, not a
    simulated P(wins the tournament): the draw is ignored. For the top of a
    field the two orderings agree almost always, and the hedge's value comes
    from picking the strongest team, not from path arithmetic. Anything
    better needs the season's bracket topology and resolved play-ins, which
    a prospective Stage 2 run does not reliably have.
    """
    ranked = sorted(
        ((t, sum(pairwise[(t, u)] for u in team_ids if u != t) / (len(team_ids) - 1)) for t in team_ids),
        key=lambda kv: -kv[1],
    )
    return ranked[:top]


def apply_champion_boost(df: pd.DataFrame, champion: int, strength: float = 1.0) -> Tuple[pd.DataFrame, int]:
    """Push every game involving ``champion`` toward a win for them.

    The "0-1 trick" for a second Kaggle slot. With strength 1.0 every row
    where the champion appears becomes P(champion wins) = 1: near-zero Brier
    on each game they actually win, a full point on the game they lose. The
    slot's expected score is worse than the calibrated file's; its job is to
    be the better of the two when the pick is right. Only the games the
    champion plays are ever scored, so boosting every pair is exactly
    boosting their path.

    Returns the boosted frame and how many rows changed.
    """
    if not 0.0 < strength <= 1.0:
        raise ValueError(f"strength must be in (0, 1], got {strength}")
    out = df.copy()
    parsed = [parse_kaggle_id(i) for i in out["ID"].astype(str)]
    as_team1 = pd.Series([t1 == champion for _, t1, _ in parsed], index=out.index)
    as_team2 = pd.Series([t2 == champion for _, _, t2 in parsed], index=out.index)
    p = out["Pred"].astype(float)
    out.loc[as_team1, "Pred"] = p[as_team1] + strength * (1.0 - p[as_team1])
    out.loc[as_team2, "Pred"] = p[as_team2] - strength * p[as_team2]
    return out, int(as_team1.sum() + as_team2.sum())
