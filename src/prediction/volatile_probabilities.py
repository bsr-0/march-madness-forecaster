"""Volatile adjustment (catalog D1).

Adjusts round_probs by injecting each team's regular-season game-to-game
inconsistency into the advancement distribution. High-volatility teams
(wildly inconsistent point margins over the season) get flattened toward
the per-round mean — giving both more upset wins (for weak volatile teams)
and more upset losses (for strong volatile teams). Low-volatility teams
(consistent performance) retain their baseline advancement rates.

Sign is the key property the catalog wants:
- A strong team with barthag 0.9 that plays a chaotic season ends up with
  round_probs *below* 0.9 for deep rounds — they might lose to anyone.
- A weak team with barthag 0.3 that plays wildly ends up with round_probs
  *above* 0.3 for early rounds — they might pull a Cinderella.
- Consistent teams (at either end) keep their baseline prediction.

Implementation note (deliberate deviation from catalog spec): the catalog
describes D1 as pairwise Log5 noise at MC sampling time. Adjustments in
this architecture operate on round_probs before sampling, not on pairwise
matchups inside the sampler. We apply the volatility at the round-probs
level by blending each team's base rp toward the per-round uniform rate
weighted by normalized volatility. This captures the same signal (volatile
teams' outcomes are harder to predict in either direction) while remaining
composable with every source and construction mode.

Walk-forward safety: per-team volatility is computed from regular-season
games only (strictly before ``tournament_start`` per the Torvik snapshot).
Historical tournament years' volatility tables are independent — no
cross-year contamination.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Dict, Iterable, Optional

from src.data.normalize import bridge_cbbpy_id

ROUND_NAMES = ("R64", "R32", "S16", "E8", "F4", "CHAMP")
# round_probs[team][rnd] = P(team WINS its rnd game) — see the identical
# fix + comment in src/prediction/roster_adj_probabilities.py, same bug.
_TEAMS_PER_ROUND = {"R64": 32, "R32": 16, "S16": 8, "E8": 4, "F4": 2, "CHAMP": 1}

# Minimum games required for a stable volatility estimate.
_MIN_GAMES_FOR_VOLATILITY = 5
# Teams without enough games get the neutral value 0.5 — no bias up or down.
_NEUTRAL_VOLATILITY = 0.5


def _load_tournament_cutoff(year: int, data_root: Path) -> Optional[str]:
    """Return the Torvik-recorded tournament_start date (YYYY-MM-DD) for the year.

    If the file or field is missing, return None — the caller will fall back
    to using all games in ``historical_games_{year}.json`` (safe in practice
    since cbbpy stops scraping at the cutoff).
    """
    path = Path(data_root) / "raw" / "historical" / f"torvik_{year}.json"
    if not path.exists():
        return None
    raw = json.load(open(path))
    return raw.get("tournament_start")


def load_team_volatility(
    year: int,
    canonical_ids: Iterable[str],
    data_root: Path,
    *,
    min_games: int = _MIN_GAMES_FOR_VOLATILITY,
) -> Dict[str, float]:
    """Per-team regular-season point-margin volatility, normalized to [0, 1].

    For each team in ``canonical_ids``, pull its game-by-game point margins
    from ``historical_games_{year}.json`` (bridging cbbpy IDs via
    ``src.data.normalize.bridge_cbbpy_id``), compute the standard deviation
    of margins, and normalize by the league max. Teams with fewer than
    ``min_games`` games get the neutral 0.5 fallback — no volatility signal
    either way.

    Args:
        year: Tournament year (used to locate both the games file and the
            Torvik cutoff date).
        canonical_ids: Tournament team IDs to compute volatility for —
            usually the seeds set.
        data_root: Repo ``data/`` root.
        min_games: Minimum games required for a volatility estimate.

    Returns:
        Dict[canonical_id, volatility ∈ [0, 1]]. Always contains every
        canonical_id in the input (fallback to 0.5 if data missing).
    """
    canonical_set = frozenset(canonical_ids)
    games_path = Path(data_root) / "raw" / "historical" / f"historical_games_{year}.json"
    if not games_path.exists():
        return {tid: _NEUTRAL_VOLATILITY for tid in canonical_set}

    cutoff = _load_tournament_cutoff(year, Path(data_root))

    raw = json.load(open(games_path))
    games = raw["games"] if isinstance(raw, dict) and "games" in raw else raw

    # Accumulate point margins per canonical team ID.
    margins: Dict[str, list] = {tid: [] for tid in canonical_set}
    for g in games:
        if cutoff is not None and g.get("date") and g["date"] >= cutoff:
            continue  # skip tournament games
        s1 = g.get("team1_score")
        s2 = g.get("team2_score")
        if s1 is None or s2 is None:
            continue
        t1 = bridge_cbbpy_id(g["team1_id"], canonical_set)
        t2 = bridge_cbbpy_id(g["team2_id"], canonical_set)
        if t1 is not None:
            margins[t1].append(s1 - s2)
        if t2 is not None:
            margins[t2].append(s2 - s1)

    # Raw stddev per team (or None if too few games).
    raw_vol: Dict[str, Optional[float]] = {}
    for tid in canonical_set:
        if len(margins[tid]) >= min_games:
            raw_vol[tid] = statistics.stdev(margins[tid])
        else:
            raw_vol[tid] = None

    # Normalize to [0, 1] by the league max. A team's volatility of 1.0
    # means "noisiest team this season"; 0.0 means "most consistent".
    observed = [v for v in raw_vol.values() if v is not None]
    if not observed:
        return {tid: _NEUTRAL_VOLATILITY for tid in canonical_set}

    league_max = max(observed)
    league_min = min(observed)
    span = league_max - league_min if league_max > league_min else 1.0

    normalized: Dict[str, float] = {}
    for tid, v in raw_vol.items():
        if v is None:
            normalized[tid] = _NEUTRAL_VOLATILITY
        else:
            normalized[tid] = (v - league_min) / span
    return normalized


def build_volatile_round_probs(
    model_rp: Dict[str, Dict[str, float]],
    volatility: Dict[str, float],
    *,
    strength: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """Flatten round_probs toward the per-round uniform rate by team volatility.

    Per team ``t``, round ``r``:
        uniform_r    = _TEAMS_PER_ROUND[r] / 64
        adjusted[r]  = (1 - v_t * strength) * base[r] + v_t * strength * uniform_r

    Where ``v_t`` ∈ [0, 1] is the team's normalized volatility. High-vol
    teams get regressed toward the round's mean advancement rate — strong
    volatile teams lose mass (more upset losses), weak volatile teams
    gain mass (more upset wins). Low-vol teams keep their baseline
    prediction. Re-normalized per round so the tournament tree still
    respects team-count constraints.

    ``strength`` controls the maximum flattening at ``v=1`` — default 0.5
    means "at most 50% regression to the mean for the noisiest team".
    """
    result: Dict[str, Dict[str, float]] = {}
    for tid, rounds in model_rp.items():
        result[tid] = {}
        v = volatility.get(tid, _NEUTRAL_VOLATILITY)
        blend = v * strength  # fraction of uniform to mix in
        for rnd in ROUND_NAMES:
            base_p = rounds.get(rnd, 0.001)
            uniform_r = _TEAMS_PER_ROUND.get(rnd, 32) / 64.0
            adjusted = (1.0 - blend) * base_p + blend * uniform_r
            result[tid][rnd] = max(adjusted, 0.001)

    # Re-normalize per round to the bracket's team-count targets (same
    # pattern as contrarian/upset_tuned).
    for rnd in ROUND_NAMES:
        total = sum(result[tid].get(rnd, 0.0) for tid in result)
        target = _TEAMS_PER_ROUND.get(rnd, 32)
        if total > 0:
            scale = target / total
            for tid in result:
                if rnd in result[tid]:
                    result[tid][rnd] *= scale

    return result
