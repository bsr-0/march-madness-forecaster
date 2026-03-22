"""Kaggle export utilities for NCAA tournament prediction submissions.

Supports both men's (TeamIDs < 3000) and women's (TeamIDs >= 3000) tournaments.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from ..data.team_name_resolver import TeamNameResolver

logger = logging.getLogger(__name__)

_KAGGLE_ID_RE = re.compile(r"^(\d{4})_(\d+)_(\d+)$")

# TeamID boundary between men's and women's in Kaggle data
WOMENS_TEAM_ID_THRESHOLD = 3000


def is_womens_team(kaggle_team_id: int) -> bool:
    """Check if a Kaggle TeamID is a women's team."""
    return kaggle_team_id >= WOMENS_TEAM_ID_THRESHOLD


@dataclass
class KaggleExportStats:
    total_rows: int = 0
    mapped_rows: int = 0
    unmapped_rows: int = 0
    season_mismatch: int = 0
    bad_id_rows: int = 0
    predict_failures: int = 0
    mens_rows: int = 0
    womens_rows: int = 0
    mens_mapped: int = 0
    womens_mapped: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "total_rows": self.total_rows,
            "mapped_rows": self.mapped_rows,
            "unmapped_rows": self.unmapped_rows,
            "season_mismatch": self.season_mismatch,
            "bad_id_rows": self.bad_id_rows,
            "predict_failures": self.predict_failures,
            "mens_rows": self.mens_rows,
            "womens_rows": self.womens_rows,
            "mens_mapped": self.mens_mapped,
            "womens_mapped": self.womens_mapped,
        }


def parse_kaggle_id(id_str: str) -> Tuple[int, int, int]:
    """Parse Kaggle submission ID format: 'YYYY_Team1_Team2'."""
    if id_str is None:
        raise ValueError("Kaggle ID is None")
    text = str(id_str).strip()
    match = _KAGGLE_ID_RE.match(text)
    if not match:
        raise ValueError(f"Invalid Kaggle ID: {id_str}")
    season, team1, team2 = match.groups()
    return int(season), int(team1), int(team2)


def load_kaggle_teams(path: str) -> Dict[int, str]:
    """Load Kaggle MTeams.csv into TeamID -> TeamName mapping."""
    team_map: Dict[int, str] = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "TeamID" not in reader.fieldnames or "TeamName" not in reader.fieldnames:
            raise ValueError("MTeams.csv must contain TeamID and TeamName columns")
        for row in reader:
            try:
                team_id = int(row.get("TeamID", "").strip())
            except (TypeError, ValueError):
                logger.warning("Skipping invalid TeamID row: %s", row)
                continue
            name = (row.get("TeamName") or "").strip()
            if not name:
                continue
            team_map[team_id] = name
    return team_map


def build_team_id_map(
    team_id_to_name: Dict[int, str],
    resolver: TeamNameResolver,
) -> Dict[int, str]:
    """Map Kaggle TeamIDs to canonical internal team IDs."""
    mapping: Dict[int, str] = {}
    for team_id, name in team_id_to_name.items():
        result = resolver.resolve(name)
        if not result.canonical_id:
            continue
        if result.confidence < 0.80:
            continue
        mapping[int(team_id)] = result.canonical_id
    return mapping


def generate_predictions(
    sample_df: pd.DataFrame,
    id_map: Dict[int, str],
    predict_fn: Callable[[str, str], float],
    season_filter: Optional[int] = None,
    womens_id_map: Optional[Dict[int, str]] = None,
    womens_predict_fn: Optional[Callable[[str, str], float]] = None,
) -> pd.DataFrame:
    """Generate Kaggle submission predictions from a sample submission frame.

    Supports both men's and women's tournaments. Women's TeamIDs (>= 3000)
    are routed to the women's predict function and ID map if provided.

    Args:
        sample_df: Kaggle sample submission DataFrame with 'ID' column
        id_map: Men's Kaggle TeamID -> canonical team ID mapping
        predict_fn: Men's prediction function (team1_id, team2_id) -> probability
        season_filter: Only process rows matching this season year
        womens_id_map: Women's Kaggle TeamID -> canonical team ID mapping
        womens_predict_fn: Women's prediction function
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
            preds.append(0.5)
            continue

        if season_filter is not None and season != season_filter:
            stats.season_mismatch += 1
            preds.append(0.5)
            continue

        # Route to men's or women's pipeline based on TeamID range
        is_womens = is_womens_team(team1) or is_womens_team(team2)

        if is_womens:
            stats.womens_rows += 1
            active_map = womens_id_map if womens_id_map else id_map
            active_fn = womens_predict_fn if womens_predict_fn else predict_fn
        else:
            stats.mens_rows += 1
            active_map = id_map
            active_fn = predict_fn

        team1_id = active_map.get(team1)
        team2_id = active_map.get(team2)
        if not team1_id or not team2_id:
            stats.unmapped_rows += 1
            preds.append(0.5)
            continue

        try:
            pred = float(active_fn(team1_id, team2_id))
        except Exception:
            stats.predict_failures += 1
            preds.append(0.5)
            continue

        stats.mapped_rows += 1
        if is_womens:
            stats.womens_mapped += 1
        else:
            stats.mens_mapped += 1

        if pred < 0.0:
            pred = 0.0
        elif pred > 1.0:
            pred = 1.0
        preds.append(pred)

    out["Pred"] = preds
    out.attrs["kaggle_export_stats"] = stats.to_dict()

    # Post-export validation: log warnings for suspicious outputs
    if stats.bad_id_rows > 0:
        logger.warning("Kaggle export: %d rows with invalid IDs", stats.bad_id_rows)
    if stats.unmapped_rows > 0:
        logger.warning("Kaggle export: %d rows with unmapped teams (defaulting to 0.5)", stats.unmapped_rows)
    if stats.predict_failures > 0:
        logger.warning("Kaggle export: %d prediction failures", stats.predict_failures)
    if stats.mapped_rows == 0 and stats.total_rows > 0:
        logger.error("Kaggle export: zero successful predictions out of %d rows", stats.total_rows)

    return out


def load_kaggle_womens_teams(path: str) -> Dict[int, str]:
    """Load Kaggle WTeams.csv into TeamID -> TeamName mapping.

    Same format as MTeams.csv but for women's tournament teams.
    """
    return load_kaggle_teams(path)
