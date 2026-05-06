"""Tournament-specific box score features (S2).

Provides per-team metrics for the meta-selector context:
- Four-factor free throw rate and turnover margin (from Torvik)
- Conference tournament champion indicator (from Kaggle)
- Ranking momentum via KenPom trajectory (from Kaggle Massey ordinals)
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

from src.prediction.kaggle_bridge import build_bridge, normalize_kaggle_spellings

_SNAPSHOT_META_KEYS = {"data_type", "cutoff_date", "tournament_start", "n_teams"}
_EARLY_DAY_TARGET = 56  # ~early February
_MOMENTUM_SYSTEM = "POM"  # KenPom


def load_four_factors(
    year: int,
    canonical_ids: Iterable[str],
    data_root: Path,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Load free throw rate and turnover margin from Torvik four-factors snapshot.

    Returns (ft_rate_dict, to_margin_dict) keyed by canonical team ID.
    """
    data_root = Path(data_root)
    primary = data_root / "raw" / f"torvik_four_factors_{year}.json"
    fallback = data_root / "raw" / "historical" / f"torvik_four_factors_{year}.json"

    if primary.exists():
        path = primary
    elif fallback.exists():
        path = fallback
    else:
        raise FileNotFoundError(f"No four-factors file for {year} at {primary} or {fallback}")

    with open(path) as f:
        data = json.load(f)

    canonical_set = set(canonical_ids)
    ft_rate: Dict[str, float] = {}
    to_margin: Dict[str, float] = {}

    for team_id, entry in data.items():
        if team_id in _SNAPSHOT_META_KEYS:
            continue
        if team_id not in canonical_set:
            continue
        ft_rate[team_id] = entry["free_throw_rate"]
        to_margin[team_id] = entry["opp_turnover_rate"] - entry["turnover_rate"]

    return ft_rate, to_margin


def load_conf_tourney_champ(
    year: int,
    canonical_ids: Iterable[str],
    data_root: Path,
) -> Dict[str, float]:
    """Indicator for conference tournament champions.

    Returns {team_id: 1.0} for champions, {team_id: 0.0} for others in the field.
    """
    data_root = Path(data_root)
    csv_path = data_root / "kaggle" / "MConferenceTourneyGames.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Conference tourney CSV not found: {csv_path}")

    canonical_set = set(canonical_ids)

    # Find the final-round winner per conference for this season.
    # Group by conference, pick the row with max DayNum — its WTeamID is the champ.
    conf_finals: Dict[str, Tuple[int, int]] = {}  # conf -> (max_day, winner_kaggle_id)
    year_found = False

    with open(csv_path, encoding="latin-1") as f:
        for row in csv.DictReader(f):
            if int(row["Season"]) != year:
                continue
            year_found = True
            conf = row["ConfAbbrev"]
            day = int(row["DayNum"])
            winner = int(row["WTeamID"])
            prev_day, _ = conf_finals.get(conf, (-1, -1))
            if day > prev_day:
                conf_finals[conf] = (day, winner)

    if not year_found:
        raise FileNotFoundError(f"No conference tourney data for season {year} in {csv_path}")

    # Bridge Kaggle IDs to canonical IDs
    spellings = normalize_kaggle_spellings(data_root)
    _, kag_to_canon = build_bridge(canonical_set, spellings)

    champ_ids: set[str] = set()
    for _, winner_kag in conf_finals.values():
        canon = kag_to_canon.get(winner_kag)
        if canon is not None and canon in canonical_set:
            champ_ids.add(canon)

    return {tid: (1.0 if tid in champ_ids else 0.0) for tid in canonical_set}


def load_ranking_momentum(
    year: int,
    canonical_ids: Iterable[str],
    data_root: Path,
) -> Dict[str, float]:
    """KenPom ranking momentum: early-season rank minus late-season rank.

    Positive values mean the team improved (rank number decreased).
    """
    data_root = Path(data_root)
    csv_path = data_root / "kaggle" / "MMasseyOrdinals.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Massey ordinals CSV not found: {csv_path}")

    canonical_set = set(canonical_ids)
    year_str = str(year)

    # Collect {TeamID: {DayNum: OrdinalRank}} for POM system in this season.
    team_ranks: Dict[int, Dict[int, int]] = defaultdict(dict)
    system_found = False

    with open(csv_path, encoding="latin-1") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Expected: Season, RankingDayNum, SystemName, TeamID, OrdinalRank
        for row in reader:
            if row[0] != year_str or row[2] != _MOMENTUM_SYSTEM:
                continue
            system_found = True
            team_id = int(row[3])
            day = int(row[1])
            rank = int(row[4])
            team_ranks[team_id][day] = rank

    if not system_found:
        raise FileNotFoundError(f"No {_MOMENTUM_SYSTEM} rankings for season {year} in {csv_path}")

    # Bridge Kaggle IDs to canonical IDs
    spellings = normalize_kaggle_spellings(data_root)
    _, kag_to_canon = build_bridge(canonical_set, spellings)

    result: Dict[str, float] = {}
    for kag_id, day_rank in team_ranks.items():
        canon = kag_to_canon.get(kag_id)
        if canon is None or canon not in canonical_set:
            continue

        days = sorted(day_rank.keys())
        late_day = days[-1]

        # Find the day closest to _EARLY_DAY_TARGET
        early_day = min(days, key=lambda d: abs(d - _EARLY_DAY_TARGET))

        early_rank = day_rank[early_day]
        late_rank = day_rank[late_day]
        result[canon] = float(early_rank - late_rank)

    return result
