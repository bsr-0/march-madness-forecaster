"""Elo source (catalog A4).

Self-contained Elo computation from regular-season game results, bridged
to canonical tournament team IDs via ``src.data.normalize.resolve_cbbpy_bridge``
so the output slots into the same ``base_round_probs`` dispatch that
torvik / odds / spread_power already use.

Why self-contained instead of reusing ``IncrementalMetricsEngine``:
- The production engine builds ``GameRecord`` objects from a different
  pipeline and enforces hparam-tuning provenance that's orthogonal to
  strategy evaluation.
- We want Elo as a *probability source*, not a feature in the ML model.
  For that we only need the running per-team rating on a cutoff date,
  and a barthag-normalized version of it. Simpler is faster to audit.

Algorithm:
1. Load ``historical_games_{year}.json``, filter to games strictly before
   Torvik's ``tournament_start`` date (walk-forward safe).
2. Process chronologically. Every team starts at 1500. After each game:
     expected_t1 = 1 / (1 + 10 ** ((elo[t2] - elo[t1]) / 400))
     adj = K * (actual_t1 - expected_t1)
     elo[t1] += adj
     elo[t2] -= adj
   K=38 per catalog. No MOV multiplier (simpler, still captures the signal).
   No cross-season carryover (each season is independent — the annual
   turnover of rosters makes prior-year regression noisy at best).
3. Bridge each cbbpy team ID to a canonical ID. For canonical IDs with
   multiple bridged cbbpy IDs (shouldn't happen but handled), keep the
   one with the highest games-played count.
4. Convert final Elo to barthag: ``barthag = 1 / (1 + 10 ** ((1500 - elo) / 400))``
   i.e., the team's implied win probability vs a 1500-rated opponent.

Output shape matches ``_load_torvik_barthag``: ``Dict[canonical_id, float in (0,1)]``
with missing teams filled from a seed-based fallback so every tournament
team has a value.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional

from src.data.normalize import load_d1_team_ids, resolve_cbbpy_bridge

_ELO_START = 1500.0
_ELO_K = 38.0  # catalog A4 spec
_MIN_GAMES_FOR_ELO = 5  # below this, fall back to seed-based estimate


def _load_tournament_cutoff(year: int, data_root: Path) -> Optional[str]:
    """Return the Torvik-recorded tournament_start date (YYYY-MM-DD) for the year."""
    path = Path(data_root) / "raw" / "historical" / f"torvik_{year}.json"
    if not path.exists():
        return None
    raw = json.load(open(path))
    return raw.get("tournament_start")


def _seed_fallback_barthag(seed: Optional[int]) -> float:
    """Conservative seed-based barthag fallback when a team has no game data.

    Matches the logic in ``_load_torvik_barthag``'s fallback branch so the
    Elo source degrades gracefully for teams with no cbbpy game records.
    """
    if seed is None:
        return 0.50
    return max(0.10, 1.0 - seed * 0.04)


def compute_elo_ratings(
    year: int,
    data_root: Path,
) -> Dict[str, float]:
    """Per-cbbpy-team Elo rating after all pre-tournament games for the year.

    Returns a dict keyed by *raw cbbpy team ID* (as written in
    ``historical_games_{year}.json``). The caller bridges to canonical IDs.
    Missing files or empty data return an empty dict.
    """
    games_path = Path(data_root) / "raw" / "historical" / f"historical_games_{year}.json"
    if not games_path.exists():
        return {}

    cutoff = _load_tournament_cutoff(year, Path(data_root))
    raw = json.load(open(games_path))
    games = raw["games"] if isinstance(raw, dict) and "games" in raw else raw

    # Sort chronologically so Elo updates in the real order games played.
    dated_games = []
    for g in games:
        date = g.get("date") or ""
        if cutoff is not None and date >= cutoff:
            continue  # tournament games — skip for walk-forward safety
        if g.get("team1_score") is None or g.get("team2_score") is None:
            continue
        dated_games.append(g)
    dated_games.sort(key=lambda g: g.get("date") or "")

    elo: Dict[str, float] = defaultdict(lambda: _ELO_START)
    games_played: Dict[str, int] = defaultdict(int)
    for g in dated_games:
        t1, t2 = g["team1_id"], g["team2_id"]
        won1 = 1.0 if g["team1_score"] > g["team2_score"] else 0.0
        expected1 = 1.0 / (1.0 + 10.0 ** ((elo[t2] - elo[t1]) / 400.0))
        adj = _ELO_K * (won1 - expected1)
        elo[t1] += adj
        elo[t2] -= adj
        games_played[t1] += 1
        games_played[t2] += 1

    # Return only teams with enough games — low-N teams keep the fallback
    # behavior at the caller.
    return {cbbpy_id: float(rating) for cbbpy_id, rating in elo.items() if games_played[cbbpy_id] >= _MIN_GAMES_FOR_ELO}


def _elo_to_barthag(elo: float) -> float:
    """Elo → barthag: implied win probability vs a 1500-rated opponent."""
    return 1.0 / (1.0 + 10.0 ** ((_ELO_START - elo) / 400.0))


def load_elo_barthag(
    year: int,
    seeds: Dict[str, int],
    data_root: Optional[Path] = None,
) -> Optional[Dict[str, float]]:
    """Elo-derived barthag per tournament team for the given year.

    Args:
        year: Tournament year.
        seeds: Canonical tournament team ID → seed (1-16). Used both for
            the output keyset and for the seed-based fallback when a team
            has no Elo-estimated rating.
        data_root: Repo data root. Defaults to ``<project>/data``.

    Returns:
        ``Dict[canonical_id, barthag ∈ (0, 1)]`` for every team in ``seeds``.
        Teams without enough games get the seed fallback. Returns ``None``
        when no historical_games file exists for the year (so the caller
        can decide to skip this source for that year).
    """
    if data_root is None:
        data_root = Path(__file__).resolve().parent.parent.parent / "data"

    canonical_ids = frozenset(seeds.keys())
    elo_by_cbbpy = compute_elo_ratings(year, Path(data_root))
    if not elo_by_cbbpy:
        return None

    # Bridge every cbbpy ID at once rather than one at a time. Against the 68
    # tournament teams alone, the bridge's prefix fallback routes other D1
    # schools onto a seeded team ("alabama_state_hornets" -> "alabama"), so
    # this needs the full D1 field to disambiguate. Rating doubles as the
    # collision weight, preserving this function's original highest-Elo
    # tiebreak; verified a no-op across all 1020 team-years of 2011-2026.
    bridge = resolve_cbbpy_bridge(
        elo_by_cbbpy, canonical_ids, universe=load_d1_team_ids(year, data_root)
    )
    canonical_elo: Dict[str, float] = {
        canonical: elo_by_cbbpy[raw] for raw, canonical in bridge.items()
    }

    # Build the output, using the seed-based fallback for unresolved teams.
    result: Dict[str, float] = {}
    for canonical, seed in seeds.items():
        if canonical in canonical_elo:
            result[canonical] = _elo_to_barthag(canonical_elo[canonical])
        else:
            result[canonical] = _seed_fallback_barthag(seed)
    return result
