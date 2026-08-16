"""Rule-based upset detector (pattern mining approach).

Combines 7 independent signals to identify tournament teams likely to
exceed their seed expectations. Not ML -- this is a rules engine.

Signals: Torvik gap, turnover margin, rebounding, conf tourney champion,
ranking momentum, Evan Miya rating, coach tournament track record.

Each signal scores 0.0--1.0; the aggregate upset_score is the weighted
sum of all signals.  Missing data for any signal silently falls back to
0.0 (neutral), so the module never crashes on partial data.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.prediction.kaggle_bridge import (
    build_bridge,
    normalize_kaggle_spellings,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Signal weights (must sum to 1.0)
# ---------------------------------------------------------------------------
_W_TORVIK_GAP = 0.25
_W_TURNOVER = 0.15
_W_REBOUNDING = 0.10
_W_CONF_CHAMP = 0.10
_W_MOMENTUM = 0.10
_W_EVAN_MIYA = 0.15
_W_COACH = 0.15

# Expected Torvik rank for each seed line.
_EXPECTED_RANK_FOR_SEED: Dict[int, int] = {
    1: 4,
    2: 8,
    3: 12,
    4: 16,
    5: 22,
    6: 28,
    7: 34,
    8: 40,
    9: 46,
    10: 55,
    11: 65,
    12: 75,
    13: 90,
    14: 120,
    15: 150,
    16: 200,
}

# ---------------------------------------------------------------------------
# Module-level cache -- keyed by (loader_name, year)
# ---------------------------------------------------------------------------
_cache: Dict[Tuple[str, int], object] = {}


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Data loaders (lazy, cached)
# ---------------------------------------------------------------------------


def _load_torvik_ranks(year: int, data_root: Path) -> Dict[str, int]:
    """Return {canonical_team_id: t_rank} from torvik_{year}.json."""
    key = ("torvik_ranks", year)
    if key in _cache:
        return _cache[key]  # type: ignore[return-value]

    path = data_root / "raw" / "historical" / f"torvik_{year}.json"
    result: Dict[str, int] = {}
    if not path.exists():
        _cache[key] = result
        return result

    with open(path) as f:
        blob = json.load(f)

    teams = blob.get("teams", [])
    for t in teams:
        tid = t.get("team_id")
        rank = t.get("t_rank")
        if tid and rank is not None:
            result[tid] = int(rank)

    _cache[key] = result
    return result


def _load_four_factors(year: int, data_root: Path) -> Dict[str, dict]:
    """Return {canonical_team_id: {factor_name: value}} from four-factors JSON.

    Tries the season-end file in data/raw/ first, then historical/.
    """
    key = ("four_factors", year)
    if key in _cache:
        return _cache[key]  # type: ignore[return-value]

    meta_keys = {"data_type", "cutoff_date", "tournament_start", "n_teams"}
    result: Dict[str, dict] = {}

    # Prefer the merged torvik_{year}.json's "four_factors" sub-dict.
    merged_candidates = [
        data_root / "raw" / f"torvik_{year}.json",
        data_root / "raw" / "historical" / f"torvik_{year}.json",
    ]
    blob = None
    for path in merged_candidates:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            if "four_factors" in data:
                blob = data["four_factors"]
            break

    if blob is None:
        candidates = [
            data_root / "raw" / f"torvik_four_factors_{year}.json",
            data_root / "raw" / "historical" / f"torvik_four_factors_{year}.json",
        ]
        for path in candidates:
            if path.exists():
                with open(path) as f:
                    blob = json.load(f)
                break

    if blob is not None:
        # Keys are metadata ('data_type', 'cutoff_date', ...) plus
        # canonical team IDs mapped to factor dicts.
        for tid, val in blob.items():
            if tid not in meta_keys and isinstance(val, dict):
                result[tid] = val

    _cache[key] = result
    return result


def _load_conf_tourney_champs(year: int, data_root: Path) -> Dict[int, int]:
    """Return {conf_index: winning_kaggle_team_id} -- one per conference.

    Actually returns a set of winning Kaggle TeamIDs for the year (the
    conference identity doesn't matter for our purposes).
    """
    key = ("conf_champs", year)
    if key in _cache:
        return _cache[key]  # type: ignore[return-value]

    path = data_root / "kaggle" / "MConferenceTourneyGames.csv"
    if not path.exists():
        _cache[key] = set()
        return set()  # type: ignore[return-value]

    # Group games by (season, conference), find the last game per conf.
    conf_games: Dict[str, list] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["Season"]) == year:
                conf_games[row["ConfAbbrev"]].append(row)

    champ_ids: set = set()
    for _conf, games in conf_games.items():
        last_game = max(games, key=lambda g: int(g["DayNum"]))
        champ_ids.add(int(last_game["WTeamID"]))

    _cache[key] = champ_ids
    return champ_ids  # type: ignore[return-value]


def _load_pom_rankings(year: int, data_root: Path) -> Dict[int, Tuple[int, int]]:
    """Return {kaggle_team_id: (early_rank, late_rank)} from POM system.

    Early = lowest DayNum available; late = highest DayNum available.
    """
    key = ("pom_rankings", year)
    if key in _cache:
        return _cache[key]  # type: ignore[return-value]

    path = data_root / "kaggle" / "MMasseyOrdinals.csv"
    if not path.exists():
        _cache[key] = {}
        return {}

    # Collect all POM rows for this year, track min/max day per team.
    early: Dict[int, Tuple[int, int]] = {}  # team_id -> (day, rank)
    late: Dict[int, Tuple[int, int]] = {}

    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["SystemName"] != "POM" or int(row["Season"]) != year:
                continue
            tid = int(row["TeamID"])
            day = int(row["RankingDayNum"])
            rank = int(row["OrdinalRank"])

            if tid not in early or day < early[tid][0]:
                early[tid] = (day, rank)
            if tid not in late or day > late[tid][0]:
                late[tid] = (day, rank)

    result: Dict[int, Tuple[int, int]] = {}
    for tid in early:
        if tid in late:
            result[tid] = (early[tid][1], late[tid][1])

    _cache[key] = result
    return result


def _load_evan_miya(year: int, data_root: Path) -> Dict[int, float]:
    """Return {kaggle_team_id: relative_rating} for the year."""
    key = ("evan_miya", year)
    if key in _cache:
        return _cache[key]  # type: ignore[return-value]

    path = data_root / "kaggle" / "evan_miya.json"
    if not path.exists():
        _cache[key] = {}
        return {}

    with open(path) as f:
        blob = json.load(f)

    cols = blob["columns"]
    year_idx = cols.index("year")
    team_no_idx = cols.index("team_no")
    rr_idx = cols.index("relative_rating")

    result: Dict[int, float] = {}
    for row in blob["data"]:
        if row[year_idx] == year:
            tid = row[team_no_idx]
            rr = row[rr_idx]
            if tid is not None and rr is not None:
                result[int(tid)] = float(rr)

    _cache[key] = result
    return result


def _load_coach_f4pct(data_root: Path) -> Dict[str, float]:
    """Return {coach_name_normalized: f4pct} from coach_results.json.

    Coach name is lowercased with spaces replaced by underscores to match
    the MTeamCoaches.csv format.
    """
    key = ("coach_f4pct", 0)
    if key in _cache:
        return _cache[key]  # type: ignore[return-value]

    path = data_root / "kaggle" / "coach_results.json"
    if not path.exists():
        _cache[key] = {}
        return {}

    with open(path) as f:
        blob = json.load(f)

    cols = blob["columns"]
    name_idx = cols.index("coach")
    f4pct_idx = cols.index("f4pct")

    result: Dict[str, float] = {}
    for row in blob["data"]:
        name = row[name_idx]
        pct = row[f4pct_idx]
        if name and pct is not None:
            normalized = re.sub(r"_+", "_", name.lower().replace(".", "_").replace(" ", "_")).strip("_")
            result[normalized] = float(pct)

    _cache[key] = result
    return result


def _load_season_coach_map(year: int, data_root: Path) -> Dict[int, str]:
    """Return {kaggle_team_id: coach_name} for the given year.

    When a team has multiple coaches (mid-season change), pick the one
    with the longest tenure.
    """
    key = ("season_coach", year)
    if key in _cache:
        return _cache[key]  # type: ignore[return-value]

    path = data_root / "kaggle" / "MTeamCoaches.csv"
    if not path.exists():
        _cache[key] = {}
        return {}

    best: Dict[int, Tuple[int, str]] = {}  # team_id -> (tenure, coach)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["Season"]) != year:
                continue
            tid = int(row["TeamID"])
            tenure = int(row["LastDayNum"]) - int(row["FirstDayNum"])
            coach = row["CoachName"]
            if tid not in best or tenure > best[tid][0]:
                best[tid] = (tenure, coach)

    result = {tid: coach for tid, (_, coach) in best.items()}
    _cache[key] = result
    return result


# ---------------------------------------------------------------------------
# Signal scorers
# ---------------------------------------------------------------------------


def _score_torvik_gap(
    canonical_id: str,
    seed: int,
    torvik_ranks: Dict[str, int],
) -> float:
    """Signal 1: seed-to-Torvik gap.  High when team is underseeded."""
    expected = _EXPECTED_RANK_FOR_SEED.get(seed)
    actual = torvik_ranks.get(canonical_id)
    if expected is None or actual is None:
        return 0.0
    gap = expected - actual
    return _clip(gap / 30.0)


def _score_turnover_margin(
    canonical_id: str,
    four_factors: Dict[str, dict],
) -> float:
    """Signal 2: positive turnover margin."""
    ff = four_factors.get(canonical_id)
    if ff is None:
        return 0.0
    opp_to = ff.get("opp_turnover_rate", 0.0)
    own_to = ff.get("turnover_rate", 0.0)
    if opp_to is None or own_to is None:
        return 0.0
    margin = opp_to - own_to
    return _clip(margin * 10.0)


def _score_rebounding(
    canonical_id: str,
    four_factors: Dict[str, dict],
) -> float:
    """Signal 3: elite offensive rebounding."""
    ff = four_factors.get(canonical_id)
    if ff is None:
        return 0.0
    or_rate = ff.get("offensive_reb_rate", 0.0)
    if or_rate is None:
        return 0.0
    return _clip((or_rate - 0.25) / 0.15)


def _score_conf_champ(
    kaggle_id: Optional[int],
    conf_champ_ids: set,
) -> float:
    """Signal 4: conference tournament champion."""
    if kaggle_id is None:
        return 0.0
    return 1.0 if kaggle_id in conf_champ_ids else 0.0


def _score_momentum(
    kaggle_id: Optional[int],
    pom_rankings: Dict[int, Tuple[int, int]],
) -> float:
    """Signal 5: ranking momentum (POM system).

    Positive when early_rank > late_rank (i.e., team improved).
    """
    if kaggle_id is None:
        return 0.0
    ranks = pom_rankings.get(kaggle_id)
    if ranks is None:
        return 0.0
    early_rank, late_rank = ranks
    improvement = early_rank - late_rank
    return _clip(improvement / 30.0)


def _score_evan_miya(
    kaggle_id: Optional[int],
    miya_ratings: Dict[int, float],
) -> float:
    """Signal 6: Evan Miya relative rating percentile."""
    if kaggle_id is None or not miya_ratings:
        return 0.0
    rating = miya_ratings.get(kaggle_id)
    if rating is None:
        return 0.0
    # Compute percentile rank within the year's field.
    all_ratings = sorted(miya_ratings.values())
    n = len(all_ratings)
    if n == 0:
        return 0.0
    # Count how many teams this one is better than.
    rank_below = sum(1 for r in all_ratings if r < rating)
    return _clip(rank_below / n)


def _score_coach(
    kaggle_id: Optional[int],
    season_coaches: Dict[int, str],
    coach_f4pct: Dict[str, float],
) -> float:
    """Signal 7: coach tournament track record (Final Four %)."""
    if kaggle_id is None:
        return 0.0
    coach_name = season_coaches.get(kaggle_id)
    if coach_name is None:
        return 0.0
    pct = coach_f4pct.get(coach_name)
    if pct is None:
        return 0.0
    return _clip(pct / 100.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_upset_scores(
    year: int,
    canonical_ids: Iterable[str],
    seeds: Dict[str, int],
    data_root: Path = Path("data"),
) -> Dict[str, float]:
    """Return {team_id: upset_score} for all tournament teams.

    Each team gets an aggregate score from 0.0 to 1.0 based on 7
    independent signals.  Missing data for any signal silently falls
    back to 0.0 (neutral).
    """
    canonical_ids = list(canonical_ids)

    # Load all data sources.
    torvik_ranks = _load_torvik_ranks(year, data_root)
    four_factors = _load_four_factors(year, data_root)
    conf_champ_ids = _load_conf_tourney_champs(year, data_root)
    pom_rankings = _load_pom_rankings(year, data_root)
    miya_ratings = _load_evan_miya(year, data_root)
    coach_f4pct = _load_coach_f4pct(data_root)
    season_coaches = _load_season_coach_map(year, data_root)

    # Build Kaggle bridge for ID mapping.
    spellings = normalize_kaggle_spellings(data_root)
    canon_to_kag, _kag_to_canon = build_bridge(canonical_ids, spellings)

    results: Dict[str, float] = {}
    for tid in canonical_ids:
        seed = seeds.get(tid)
        kag_id = canon_to_kag.get(tid)

        s1 = _score_torvik_gap(tid, seed, torvik_ranks) if seed else 0.0
        s2 = _score_turnover_margin(tid, four_factors)
        s3 = _score_rebounding(tid, four_factors)
        s4 = _score_conf_champ(kag_id, conf_champ_ids)
        s5 = _score_momentum(kag_id, pom_rankings)
        s6 = _score_evan_miya(kag_id, miya_ratings)
        s7 = _score_coach(kag_id, season_coaches, coach_f4pct)

        score = (
            _W_TORVIK_GAP * s1
            + _W_TURNOVER * s2
            + _W_REBOUNDING * s3
            + _W_CONF_CHAMP * s4
            + _W_MOMENTUM * s5
            + _W_EVAN_MIYA * s6
            + _W_COACH * s7
        )
        results[tid] = _clip(score)

    return results


def apply_upset_overrides(
    bracket: np.ndarray,
    first_round_matchups: List[str],
    seeds: Dict[str, int],
    upset_scores: Dict[str, float],
    override_threshold: float = 0.45,
    min_round: int = 2,
) -> np.ndarray:
    """Selectively override chalk picks where upset signals converge.

    Parameters
    ----------
    bracket : ndarray of bool, shape (63,)
        True = team1 wins (first of each pair in matchup list).
    first_round_matchups : list of str, length 64
        Flat list of 64 canonical team IDs.  Adjacent pairs form R64
        matchups: ``[t1, t2, t3, t4, ...]`` means game 0 is t1 vs t2,
        game 1 is t3 vs t4, etc.
    seeds : dict
        {canonical_team_id: seed_int}.
    upset_scores : dict
        {canonical_team_id: float} from ``compute_upset_scores``.
    override_threshold : float
        Minimum upset_score for the underdog to trigger an override.
    min_round : int
        Earliest round (0-indexed) to apply overrides.  0=R64, 1=R32,
        2=S16, 3=E8, 4=F4, 5=Championship.

    Returns
    -------
    ndarray of bool, shape (63,)
        Modified bracket with upset overrides applied and path
        consistency maintained.
    """
    bracket = bracket.copy()

    # 63 games: indices 0-31 = R64, 32-47 = R32, 48-55 = S16,
    # 56-59 = E8, 60-61 = F4, 62 = Championship.
    round_slices = [
        (0, 32),  # R64, round 0
        (32, 48),  # R32, round 1
        (48, 56),  # S16, round 2
        (56, 60),  # E8,  round 3
        (60, 62),  # F4,  round 4
        (62, 63),  # Championship, round 5
    ]

    # Resolve team assignments for every game.  Start with R64 matchups.
    n_games = 63
    game_teams: List[Optional[Tuple[str, str]]] = [None] * n_games
    game_winner: List[str] = [""] * n_games

    # Parse R64 matchups from flat 64-team list (adjacent pairs).
    for i in range(32):
        idx = 2 * i
        if idx + 1 >= len(first_round_matchups):
            break
        t1, t2 = first_round_matchups[idx], first_round_matchups[idx + 1]
        game_teams[i] = (t1, t2)
        game_winner[i] = t1 if bracket[i] else t2

    # Forward-propagate to fill later rounds.
    for rnd_idx, (start, end) in enumerate(round_slices):
        if rnd_idx == 0:
            continue
        prev_start = round_slices[rnd_idx - 1][0]
        for g in range(start, end):
            g_in_round = g - start
            prev_g1 = prev_start + 2 * g_in_round
            prev_g2 = prev_start + 2 * g_in_round + 1
            w1 = game_winner[prev_g1]
            w2 = game_winner[prev_g2]
            game_teams[g] = (w1, w2)
            game_winner[g] = w1 if bracket[g] else w2

    # Apply overrides round by round.  After each round with overrides,
    # re-propagate downstream so subsequent rounds see updated matchups.
    for rnd_idx, (start, end) in enumerate(round_slices):
        if rnd_idx < min_round:
            continue

        any_flipped = False
        for g in range(start, end):
            pair = game_teams[g]
            if pair is None:
                continue
            team_a, team_b = pair
            if not team_a or not team_b:
                continue

            seed_a = seeds.get(team_a, 16)
            seed_b = seeds.get(team_b, 16)

            # Identify chalk (lower seed number) and underdog.
            if seed_a < seed_b:
                chalk, dog = team_a, team_b
                chalk_wins = bracket[g]
            elif seed_b < seed_a:
                chalk, dog = team_b, team_a
                chalk_wins = not bracket[g]
            else:
                continue

            dog_score = upset_scores.get(dog, 0.0)
            if chalk_wins and dog_score >= override_threshold:
                # Flip the pick.
                if chalk == team_a:
                    bracket[g] = False
                else:
                    bracket[g] = True
                game_winner[g] = dog
                any_flipped = True
                logger.info(
                    "Upset override game %d (round %d): %s (seed %d, score %.3f) over %s (seed %d)",
                    g,
                    rnd_idx,
                    dog,
                    seeds.get(dog, 16),
                    dog_score,
                    chalk,
                    seeds.get(chalk, 16),
                )

        # Re-propagate all downstream rounds after any flip in this round.
        if any_flipped:
            for later_rnd in range(rnd_idx + 1, len(round_slices)):
                ls, le = round_slices[later_rnd]
                ps = round_slices[later_rnd - 1][0]
                for g2 in range(ls, le):
                    g_in = g2 - ls
                    f1 = ps + 2 * g_in
                    f2 = ps + 2 * g_in + 1
                    w1 = game_winner[f1]
                    w2 = game_winner[f2]
                    old_pair = game_teams[g2]
                    game_teams[g2] = (w1, w2)

                    if old_pair == (w1, w2):
                        # Matchup unchanged, keep existing pick.
                        game_winner[g2] = w1 if bracket[g2] else w2
                        continue

                    # Matchup changed -- the old winner may no longer be
                    # in the game.  If the old winner is still here, keep
                    # the pick; otherwise default to higher upset score.
                    old_winner = game_winner[g2]
                    if old_winner in (w1, w2):
                        bracket[g2] = old_winner == w1
                        game_winner[g2] = old_winner
                    else:
                        us1 = upset_scores.get(w1, 0.0)
                        us2 = upset_scores.get(w2, 0.0)
                        new_winner = w1 if us1 >= us2 else w2
                        bracket[g2] = new_winner == w1
                        game_winner[g2] = new_winner

    return bracket


def clear_cache() -> None:
    """Clear all cached data (useful between test runs)."""
    _cache.clear()
