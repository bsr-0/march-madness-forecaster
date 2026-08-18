"""Roster adjustment (catalog C2).

Boosts strong-roster teams and penalizes weak-roster teams by a small
multiplicative tweak (±4% cap) on round_probs. Captures star-player effect
that team-level efficiency metrics miss — a team's top-5 WARP measures
how much value the rotation produces beyond a replacement-level lineup.

Implementation note (deliberate deviation from catalog spec): the catalog
describes C2 as a barthag-level multiplicative tweak. Adjustments in this
architecture operate on round_probs before sampling, not on barthag, so
roster_adj applies the same multiplicative factor at the round_probs
level and re-normalizes per round. Captures the same signal (top-5 WARP
z-score → small advancement boost/penalty) while remaining composable
with every source and construction mode.

Walk-forward safety: per-team WARP is read from
``cbbpy_rosters_{year}.json`` only — no cross-year contamination. Each
test year's adjustment is computed in isolation.

cbbpy team IDs (e.g., ``abilene_christian_wildcats``) are bridged to the
canonical Torvik / seeds ID scheme via
``src.data.normalize.resolve_cbbpy_bridge`` — the same bridge used by the Elo
and volatile sources (catalog phase 1b7).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Dict, Iterable

from src.data.normalize import load_d1_team_ids, resolve_cbbpy_bridge

ROUND_NAMES = ("R64", "R32", "S16", "E8", "F4", "CHAMP")
# round_probs[team][rnd] = P(team WINS its rnd game), per
# build_torvik_round_probabilities — so each round's total across all 64
# teams sums to the number of WINNERS of that round (32/16/8/4/2/1), not
# the number of teams entering it. Verified directly against real
# round_probs output 2026-08-13 (sums were 32.0/16.0/8.0/4.0/2.0/1.0).
# This constant previously used the entrants convention {64,32,16,8,4,2}
# — exactly double at every round — which silently inflated every team's
# adjusted probability ~2x per round during renormalization.
_TEAMS_PER_ROUND = {"R64": 32, "R32": 16, "S16": 8, "E8": 4, "F4": 2, "CHAMP": 1}

# Catalog spec C2: barthag *= 1 + 0.02 * z, clipped to ±4%.
_Z_SCALE = 0.02
_MAX_ADJUSTMENT = 0.04
# Minimum games for a player's WARP to be included in the top-5 average.
# Filters out late-season call-ups and walk-ons whose WARP is unreliable.
_MIN_PLAYER_GAMES = 5
_TOP_N_PLAYERS = 5


def _team_top5_warp(players: list) -> float | None:
    """Return the mean WARP across a team's top-5 players (by WARP)."""
    valid_warps = []
    for p in players:
        warp = p.get("warp")
        games = p.get("games_played") or 0
        if warp is None or games < _MIN_PLAYER_GAMES:
            continue
        try:
            valid_warps.append(float(warp))
        except (TypeError, ValueError):
            continue
    if len(valid_warps) < _TOP_N_PLAYERS:
        return None
    valid_warps.sort(reverse=True)
    return sum(valid_warps[:_TOP_N_PLAYERS]) / _TOP_N_PLAYERS


def load_team_talent(
    year: int,
    canonical_ids: Iterable[str],
    data_root: Path,
) -> Dict[str, float]:
    """Per-team top-5 mean WARP from cbbpy rosters, bridged to canonical IDs.

    For each team in ``canonical_ids``, look up its cbbpy roster, take the
    top-5 players by WARP (filtered to those with >=5 games played), and
    return the mean. Teams missing from the roster file or with fewer than
    5 qualifying players are simply absent from the result — the caller's
    ``build_roster_adj_round_probs`` treats absence as the neutral z=0
    fallback.

    Args:
        year: Tournament year (locates ``cbbpy_rosters_{year}.json``).
        canonical_ids: Tournament team IDs to compute talent for — usually
            the seeds set.
        data_root: Repo ``data/`` root.

    Returns:
        Dict[canonical_id, top5_warp_mean]. May be smaller than
        ``canonical_ids`` if rosters are missing or sparse.
    """
    canonical_set = frozenset(canonical_ids)
    rosters_path = Path(data_root) / "raw" / "historical" / f"cbbpy_rosters_{year}.json"
    if not rosters_path.exists():
        return {}

    raw = json.load(open(rosters_path))
    teams = raw.get("teams") if isinstance(raw, dict) else None
    if not teams:
        return {}

    # Resolve the bridge across the whole roster file at once, against the full
    # D1 field and weighted by each roster's total player-games. Bridging one
    # ID at a time against the 68 tournament teams hands "alabama_state_hornets"
    # the canonical "alabama", and the loop below is last-write-wins, so
    # Alabama's WARP would simply be overwritten by Alabama State's.
    weights = {
        team["team_id"]: sum((p.get("games_played") or 0) for p in (team.get("players") or []))
        for team in teams
        if team.get("team_id")
    }
    bridge = resolve_cbbpy_bridge(
        weights, canonical_set, universe=load_d1_team_ids(year, data_root)
    )

    talent: Dict[str, float] = {}
    for team in teams:
        canonical = bridge.get(team.get("team_id"))
        if canonical is None:
            continue
        top5 = _team_top5_warp(team.get("players") or [])
        if top5 is not None:
            talent[canonical] = top5

    return talent


def build_roster_adj_round_probs(
    model_rp: Dict[str, Dict[str, float]],
    team_talent: Dict[str, float],
    *,
    z_scale: float = _Z_SCALE,
    max_adjustment: float = _MAX_ADJUSTMENT,
) -> Dict[str, Dict[str, float]]:
    """Apply a top-5-WARP-z-score multiplicative adjustment to round_probs.

    Per team ``t`` with talent ``w_t``:
        z_t          = (w_t - league_mean) / league_std   (0 if t missing)
        factor_t     = 1 + clip(z_scale * z_t, -max_adjustment, +max_adjustment)
        adjusted[r]  = base[r] * factor_t                  for each round r

    Then re-normalized per round so the bracket's team-count constraints
    {64, 32, 16, 8, 4, 2} hold. Teams missing from ``team_talent`` get the
    neutral z=0 fallback (no boost or penalty before renorm).

    The clip caps the per-team boost/penalty at ±4% of barthag (catalog
    spec C2). Even an extreme talent outlier (z=±100) gets the cap, not
    a pathological 100× swing.

    Args:
        model_rp: Base round_probs to adjust.
        team_talent: Per-team top-5 mean WARP (subset of teams in model_rp).
        z_scale: Multiplier on the z-score (default 0.02 per catalog).
        max_adjustment: Clip bound on the per-team factor (default 0.04).

    Returns:
        Adjusted, per-round-renormalized round_probs.
    """
    # League stats are computed only over teams that have talent data —
    # missing teams aren't allowed to drag the mean down to zero.
    talent_values = list(team_talent.values())
    if len(talent_values) >= 2:
        league_mean = statistics.fmean(talent_values)
        league_std = statistics.stdev(talent_values)
    else:
        league_mean = 0.0
        league_std = 0.0

    result: Dict[str, Dict[str, float]] = {}
    for tid, rounds in model_rp.items():
        w = team_talent.get(tid)
        if w is None or league_std <= 0:
            z = 0.0
        else:
            z = (w - league_mean) / league_std
        raw_factor = z_scale * z
        clipped = max(-max_adjustment, min(max_adjustment, raw_factor))
        factor = 1.0 + clipped

        result[tid] = {}
        for rnd in ROUND_NAMES:
            base_p = rounds.get(rnd, 0.001)
            result[tid][rnd] = max(base_p * factor, 0.001)

    # Re-normalize per round to the bracket's team-count targets (same
    # pattern as volatile / upset_tuned).
    for rnd in ROUND_NAMES:
        total = sum(result[tid].get(rnd, 0.0) for tid in result)
        target = _TEAMS_PER_ROUND.get(rnd, 32)
        if total > 0:
            scale = target / total
            for tid in result:
                if rnd in result[tid]:
                    result[tid][rnd] *= scale

    return result
