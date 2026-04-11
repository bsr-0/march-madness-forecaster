"""Torvik barthag + Log5 + Monte Carlo probabilities for pool optimization.

Produces pairwise and round-advancement probabilities using Torvik's barthag
rating (expected win % vs an average opponent) as the only input. No ML
ensemble, no calibration stage — Log5 for pairwise and a bracket-structure
Monte Carlo for round advancement.

This is the `torvik` mode referenced in POOL_STRATEGY_RECOMMENDATION.md. On the
13-year backtest (2011–2025 minus 2020 and 2012) it has the numerically best
BestRnk (31.5) and P(top5%) (5.06%) among the four statistically-tied base
modes (seed, noseed, blend, torvik), which is why it's the defensible
recommendation for a single-entry pool submission.

The function signatures mirror `src.prediction.seed_probabilities` so the
PoolOptimizer can consume torvik output with no changes.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .noseed_model import _validate_pretournament

HIST_DIR = Path("data/raw/historical")
DATA_DIR = Path("data/raw")

_ROUND_NAMES: Tuple[str, ...] = ("R64", "R32", "S16", "E8", "F4", "CHAMP")
_REGION_ORDER: Tuple[str, ...] = ("East", "West", "South", "Midwest")
_SEED_MATCHUP_ORDER: Tuple[Tuple[int, int], ...] = (
    (1, 16),
    (8, 9),
    (5, 12),
    (4, 13),
    (6, 11),
    (3, 14),
    (7, 10),
    (2, 15),
)
# 2011 used Southeast/Southwest labels; normalize so REGION_ORDER works uniformly.
_REGION_ALIASES: Dict[str, str] = {"Southeast": "South", "Southwest": "Midwest"}


def load_torvik_barthag(
    year: int,
    seeds: Dict[str, int],
    data_dir: Path | str | None = None,
) -> Dict[str, float]:
    """Load barthag ratings for tournament teams.

    Searches `data/raw/historical/torvik_{year}.json` first, then
    `{data_dir}/torvik_{year}.json` (defaults to `data/raw/`). The file is
    validated for pre-tournament provenance — a missing or non-pre_tournament
    `data_type` field raises `LeakageError` rather than silently ingesting
    potentially look-ahead-biased data.

    Teams that exist in `seeds` but are absent from the Torvik file fall back
    to a seed-based estimate: `max(0.10, 1 - seed * 0.04)`. This is a crude
    floor that keeps MC simulation well-defined when scraping misses a team.

    Returns: dict of team_id -> barthag in [0.0, 1.0].
    """
    barthag: Dict[str, float] = {}
    search_dirs: Iterable[Path] = [HIST_DIR, Path(data_dir) if data_dir else DATA_DIR]

    for prefix in search_dirs:
        path = prefix / f"torvik_{year}.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        _validate_pretournament(data, path)
        for t in data.get("teams", []):
            tid = t.get("team_id", "")
            b = t.get("barthag")
            if tid in seeds and b is not None:
                barthag[tid] = float(b)
        break

    for tid, seed in seeds.items():
        if tid not in barthag:
            barthag[tid] = max(0.10, 1.0 - seed * 0.04)

    return barthag


def log5(pa: float, pb: float) -> float:
    """Log5: P(A beats B) given each team's win rate vs an average opponent.

    Returns 0.5 when both rates are degenerate (pa == pb == 0 or 1).
    """
    num = pa * (1 - pb)
    denom = pa * (1 - pb) + pb * (1 - pa)
    if denom < 1e-12:
        return 0.5
    return num / denom


def build_torvik_probabilities(
    seeds: Dict[str, int],
    barthag: Dict[str, float],
) -> Dict[Tuple[str, str], float]:
    """Build pairwise win probabilities for every team pair via Log5.

    Return format matches `src.prediction.seed_probabilities.build_seed_probabilities`:
    both orientations of every pair are included, with `probs[(a, b)] + probs[(b, a)] == 1`.
    """
    probs: Dict[Tuple[str, str], float] = {}
    team_ids = list(seeds.keys())
    for t1, t2 in combinations(team_ids, 2):
        p = log5(barthag.get(t1, 0.5), barthag.get(t2, 0.5))
        probs[(t1, t2)] = p
        probs[(t2, t1)] = 1.0 - p
    return probs


def build_torvik_round_probabilities(
    seeds: Dict[str, int],
    regions: Dict[str, str],
    barthag: Dict[str, float],
    n_sims: int = 10000,
    rng_seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Build per-round advancement probabilities via bracket Monte Carlo.

    Simulates `n_sims` runs of the full 64-team bracket. Each matchup outcome
    is drawn from `log5(barthag_a, barthag_b)`. Counts how often each team
    advances to each round, then converts counts to probabilities with a
    1/n_sims floor (0.001 by default) so downstream consumers don't see
    absolute zeros.

    Return format matches
    `src.prediction.seed_probabilities.build_seed_round_probabilities`.

    Region aliases (Southeast → South, Southwest → Midwest) are normalized
    so pre-2012 seasons work without custom handling.

    First Four note: NCAA seed files include 68 teams, with 4 regions that
    contain duplicate seeds (typically 11 and 16). Teams are assigned to
    `region_teams[region][seed]` as a plain dict, so the second team at a
    duplicated seed silently overwrites the first — one of the two play-in
    teams is arbitrarily treated as the bracket slot winner. The displaced
    team ends up with floor-value round probs. This matches the behavior of
    `scripts/mc_pool_backtest.py` and is therefore consistent with the
    numbers reported in POOL_STRATEGY_RECOMMENDATION.md. Run after the First
    Four has been resolved (seed file regenerated to 64 teams) for a faithful
    bracket.
    """
    rng = np.random.default_rng(rng_seed)

    normalized_regions = {tid: _REGION_ALIASES.get(r, r) for tid, r in regions.items()}

    region_teams: Dict[str, Dict[int, str]] = {r: {} for r in _REGION_ORDER}
    for tid, seed in seeds.items():
        r = normalized_regions.get(tid, "")
        if r in region_teams:
            region_teams[r][seed] = tid

    bracket_order: List[str] = []
    for region in _REGION_ORDER:
        rt = region_teams[region]
        for high, low in _SEED_MATCHUP_ORDER:
            bracket_order.append(rt.get(high, f"unknown_{region}_{high}"))
            bracket_order.append(rt.get(low, f"unknown_{region}_{low}"))

    advance_counts = {tid: {rnd: 0 for rnd in _ROUND_NAMES} for tid in seeds}

    for _ in range(n_sims):
        current = list(bracket_order)
        for rnd in _ROUND_NAMES:
            next_round: List[str] = []
            for g in range(0, len(current), 2):
                t1, t2 = current[g], current[g + 1]
                p1 = log5(barthag.get(t1, 0.5), barthag.get(t2, 0.5))
                winner = t1 if rng.random() < p1 else t2
                if winner in advance_counts:
                    advance_counts[winner][rnd] += 1
                next_round.append(winner)
            current = next_round

    floor = 1.0 / n_sims
    return {tid: {rnd: max(floor, advance_counts[tid][rnd] / n_sims) for rnd in _ROUND_NAMES} for tid in seeds}
