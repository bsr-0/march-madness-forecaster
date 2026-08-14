"""Coach adjustment (catalog C1).

Boosts teams whose head coach has prior NCAA tournament experience by a
small one-sided multiplicative factor (0% to +3% on round_probs). Captures
the veteran-coach effect — coaches who have been to the dance many times
have a track record team-level efficiency metrics don't directly carry.

Asymmetric design (per catalog spec): zero-experience coaches get
factor=1.0 (no boost), more-experienced coaches get progressively larger
boosts up to a cap. Never penalizes — a new coach is informationally
neutral, not bad.

Walk-forward safety: appearances for test year Y are cumulated only over
seasons strictly < Y. Each test year is computed in isolation.

Implementation note (deliberate deviation from catalog spec): the spec
describes C1 as a barthag-level multiplicative tweak. Adjustments in
this architecture operate on round_probs before sampling, not on barthag
(same pattern as ``volatile`` / ``upset_tuned`` / ``roster_adj``). The
multiplicative factor is applied at the round_probs level and re-normalized
per round.

Data sources (all under ``data/kaggle/``):
- ``MTeamCoaches.csv``: per-(season, kaggle_team_id) head-coach assignments.
  When a team has multiple coaches in one season (mid-year firing), the
  one with the longest tenure (max ``LastDayNum - FirstDayNum``) wins.
- ``MNCAATourneySeeds.csv``: which (season, kaggle_team_id) pairs made
  the NCAA tournament. Joining MTeamCoaches against this set gives the
  coach-by-tournament-season tally.
- ``MTeamSpellings.csv``: canonical → Kaggle ``TeamID`` bridge. Uses a
  cascade of name normalizations + a small manual alias table to push
  coverage on the 2026 field from 61/68 to 67-68/68.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

from src.prediction.kaggle_bridge import build_bridge, normalize_kaggle_spellings

ROUND_NAMES = ("R64", "R32", "S16", "E8", "F4", "CHAMP")
# round_probs[team][rnd] = P(team WINS its rnd game) — see the identical
# fix + comment in src/prediction/roster_adj_probabilities.py, same bug.
_TEAMS_PER_ROUND = {"R64": 32, "R32": 16, "S16": 8, "E8": 4, "F4": 2, "CHAMP": 1}

# Catalog spec C1: barthag *= 1 + 0.01 * min(log(1 + apps), 3).
# log(1 + 19) ≈ 3, so the cap binds at ~19 prior tournament appearances.
_LOG_SCALE = 0.01
_LOG_CEILING = 3.0  # max bump = 3% per spec


def _load_season_coach_map(data_root: Path) -> Dict[Tuple[int, int], str]:
    """Load primary head coach per (season, kaggle_team_id) from MTeamCoaches.csv.

    When a team has multiple coaches in one season (mid-year firing),
    return the one with the longest tenure (max LastDayNum - FirstDayNum).
    Coach names are normalized to lowercase_underscore as stored.
    """
    path = Path(data_root) / "kaggle" / "MTeamCoaches.csv"
    if not path.exists():
        return {}

    # Track (season, team) → list of (tenure_days, coach_name).
    accum: Dict[Tuple[int, int], list] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            season = int(row["Season"])
            team_id = int(row["TeamID"])
            first = int(row["FirstDayNum"])
            last = int(row["LastDayNum"])
            coach = row["CoachName"].strip().lower()
            tenure = last - first
            accum.setdefault((season, team_id), []).append((tenure, coach))

    return {key: max(stints)[1] for key, stints in accum.items()}


def _load_tournament_season_teams(data_root: Path) -> Dict[int, set]:
    """Load {season: {kaggle_team_id, ...}} from MNCAATourneySeeds.csv.

    Each Kaggle team listed in a season's seed file made the tournament.
    """
    path = Path(data_root) / "kaggle" / "MNCAATourneySeeds.csv"
    if not path.exists():
        return {}

    out: Dict[int, set] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            season = int(row["Season"])
            team_id = int(row["TeamID"])
            out.setdefault(season, set()).add(team_id)
    return out


def load_coach_experience(
    year: int,
    canonical_ids: Iterable[str],
    data_root: Path,
) -> Dict[str, int]:
    """Per-team count of head coach's prior NCAA tournament appearances.

    For each team in ``canonical_ids``:
      1. Bridge canonical → Kaggle TeamID via MTeamSpellings + aliases.
      2. Look up the team's head coach in season ``year`` via MTeamCoaches
         (primary coach if mid-season swap).
      3. Count seasons strictly before ``year`` where that coach was the
         primary head coach of any team that made the NCAA tournament.

    Walk-forward by construction: only seasons < ``year`` contribute to
    the appearance count. The current-year coach lookup is taken from
    ``year`` itself, but no statistics from year ``year`` are used.

    Args:
        year: Test year (the tournament whose coach lineup we're labeling).
        canonical_ids: Tournament team IDs to compute experience for —
            usually the ``year``'s seed set.
        data_root: Repo ``data/`` root.

    Returns:
        Dict[canonical_id, prior_apps_count]. Always 0 or positive
        integer. Teams that don't bridge to Kaggle, or whose ``year``
        coach isn't in MTeamCoaches, get 0.
    """
    canonical_set = frozenset(canonical_ids)
    if not canonical_set:
        return {}

    spellings = normalize_kaggle_spellings(data_root)
    canon_to_kag, _ = build_bridge(canonical_set, spellings)
    coach_map = _load_season_coach_map(data_root)
    tourney_teams_by_season = _load_tournament_season_teams(data_root)

    if not coach_map or not tourney_teams_by_season:
        return {tid: 0 for tid in canonical_set}

    # Cumulative appearance count per coach across seasons strictly < year.
    coach_appearances: Dict[str, int] = {}
    for season, teams in tourney_teams_by_season.items():
        if season >= year:
            continue
        for kag_tid in teams:
            coach = coach_map.get((season, kag_tid))
            if coach:
                coach_appearances[coach] = coach_appearances.get(coach, 0) + 1

    # For each canonical team, look up its year coach and attach apps count.
    result: Dict[str, int] = {}
    for tid in canonical_set:
        kag = canon_to_kag.get(tid)
        if kag is None:
            result[tid] = 0
            continue
        coach = coach_map.get((year, kag))
        if not coach:
            result[tid] = 0
            continue
        result[tid] = coach_appearances.get(coach, 0)

    return result


def build_coach_adj_round_probs(
    model_rp: Dict[str, Dict[str, float]],
    coach_experience: Dict[str, int],
) -> Dict[str, Dict[str, float]]:
    """Apply a one-sided log-experience multiplicative boost to round_probs.

    Per team ``t`` with experience ``apps_t``:
        bump_t       = min(log(1 + apps_t), 3.0)
        factor_t     = 1.0 + 0.01 * bump_t        ∈ [1.00, 1.03]
        adjusted[r]  = base[r] * factor_t          for each round r

    Then re-normalized per round to bracket team-counts {64, 32, 16, 8, 4, 2}.
    Teams missing from ``coach_experience`` are treated as zero apps —
    factor=1.0, no boost — so missing coaches and brand-new coaches are
    both informationally neutral.

    The cap binds at ~19 prior appearances (log(20) ≈ 3.0). Even a
    50-appearance coach stays at +3% — no runaway boost from outliers.
    Asymmetric (factor ≥ 1.0) per catalog spec: experience never penalizes.

    Args:
        model_rp: Base round_probs to adjust.
        coach_experience: Per-team prior tournament appearance count
            (from ``load_coach_experience``).

    Returns:
        Adjusted, per-round-renormalized round_probs.
    """
    result: Dict[str, Dict[str, float]] = {}
    for tid, rounds in model_rp.items():
        apps = coach_experience.get(tid, 0)
        bump = min(math.log(1.0 + max(apps, 0)), _LOG_CEILING)
        factor = 1.0 + _LOG_SCALE * bump  # ∈ [1.00, 1.03]

        result[tid] = {}
        for rnd in ROUND_NAMES:
            base_p = rounds.get(rnd, 0.001)
            result[tid][rnd] = max(base_p * factor, 0.001)

    # Re-normalize per round (same pattern as roster_adj / volatile).
    for rnd in ROUND_NAMES:
        total = sum(result[tid].get(rnd, 0.0) for tid in result)
        target = _TEAMS_PER_ROUND.get(rnd, 32)
        if total > 0:
            scale = target / total
            for tid in result:
                if rnd in result[tid]:
                    result[tid][rnd] *= scale

    return result
