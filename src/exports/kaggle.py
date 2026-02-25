"""Kaggle export utilities for NCAA tournament prediction submissions."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import pandas as pd

from ..data.team_name_resolver import TeamNameResolver


_KAGGLE_ID_RE = re.compile(r"^(\d{4})_(\d+)_(\d+)$")


@dataclass
class KaggleExportStats:
    total_rows: int = 0
    mapped_rows: int = 0
    unmapped_rows: int = 0
    season_mismatch: int = 0
    bad_id_rows: int = 0
    predict_failures: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "total_rows": self.total_rows,
            "mapped_rows": self.mapped_rows,
            "unmapped_rows": self.unmapped_rows,
            "season_mismatch": self.season_mismatch,
            "bad_id_rows": self.bad_id_rows,
            "predict_failures": self.predict_failures,
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
) -> pd.DataFrame:
    """Generate Kaggle submission predictions from a sample submission frame."""
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

        team1_id = id_map.get(team1)
        team2_id = id_map.get(team2)
        if not team1_id or not team2_id:
            stats.unmapped_rows += 1
            preds.append(0.5)
            continue

        try:
            pred = float(predict_fn(team1_id, team2_id))
        except Exception:
            stats.predict_failures += 1
            preds.append(0.5)
            continue

        stats.mapped_rows += 1
        if pred < 0.0:
            pred = 0.0
        elif pred > 1.0:
            pred = 1.0
        preds.append(pred)

    out["Pred"] = preds
    out.attrs["kaggle_export_stats"] = stats.to_dict()
    return out
