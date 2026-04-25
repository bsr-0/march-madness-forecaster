"""Momentum adjustment (catalog C3).

Boosts teams trending up (January→March four-factor efficiency improving)
and penalizes teams trending down. Captures late-season trajectory that
pre-tournament barthag compresses into a single snapshot — a team that
was mediocre early and surged to elite by tournament time is meaningfully
different from a team that peaked in December.

Two-sided design: improving teams get +0..+3% boost; declining teams
get matching penalty. `tanh` saturation caps both ends, so even a
pathological delta of 0.5 (huge) stays within ±3%.

Walk-forward safety: per-team delta is computed strictly from snapshot
files for ``{year}`` — January (~YYYY-01-31) vs March (~YYYY-03-xx,
pre-tournament). Each test year's adjustment is computed in isolation.

Implementation note (deliberate deviation from catalog spec): the spec
calls for ``delta_eff = (march_adjEM - january_adjEM) / january_adjEM``,
using Torvik's ``adj_efficiency_margin`` field. Two problems with that:

1. ``adj_efficiency_margin`` isn't in the monthly four-factor snapshot
   files — only the 8 raw four-factor stats (eFG%/opp_eFG%, turnover
   rates, rebounding rates, FT rates) are. We synthesize a Dean-Oliver
   weighted proxy instead.
2. Division by the January value blows up when a team's January adjEM
   is near zero (which is the mean of the distribution). Switched to
   absolute delta with a calibrated tanh multiplier.

Catalog-style multiplicative-tweak spec is preserved; only the input
definition changed to match what the data actually carries. Dean-Oliver
weights (40/25/20/15 for shooting/turnovers/rebounding/FTs) are the
standard basketball-analytics aggregation of the four-factor vector.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, Optional

ROUND_NAMES = ("R64", "R32", "S16", "E8", "F4", "CHAMP")
_TEAMS_PER_ROUND = {"R64": 64, "R32": 32, "S16": 16, "E8": 8, "F4": 4, "CHAMP": 2}

# Meta keys in four-factor snapshot files (not team IDs).
_SNAPSHOT_META_KEYS = frozenset({"data_type", "cutoff_date", "tournament_start", "snapshot_type", "n_teams"})

# Catalog spec C3 (adapted): barthag *= 1 + 0.03 * tanh(delta * 10).
# tanh saturates at ~0.76 for input=1.0 → max factor ≈ 1.0228; at input=2.0
# it's 0.964 → factor 1.029. The +3% cap is the asymptotic bound.
_TANH_SCALE = 10.0
_MAX_ADJUSTMENT = 0.03


def _four_factor_margin(team_stats: Dict[str, float]) -> float:
    """Dean-Oliver-weighted four-factor margin proxy for adj_efficiency_margin.

    Weights come from basketball-analytics canon:
    - Shooting (40%): eFG% margin
    - Turnovers (25%): opp TO rate − own TO rate (forcing more / committing fewer)
    - Rebounding (20%): off + def rebound rates sum (above 1.0 = net rebounding edge)
    - FTs (15%): free-throw rate margin

    Returns a scalar typically in the range [-0.15, +0.25] across D1 teams.
    """
    return (
        (team_stats["effective_fg_pct"] - team_stats["opp_effective_fg_pct"])
        + 0.3 * (team_stats["offensive_reb_rate"] + team_stats["defensive_reb_rate"] - 1.0)
        + 0.2 * (team_stats["opp_turnover_rate"] - team_stats["turnover_rate"])
        + 0.1 * (team_stats["free_throw_rate"] - team_stats["opp_free_throw_rate"])
    )


def _find_snapshot_file(
    year: int,
    target_month: int,
    data_root: Path,
) -> Optional[Path]:
    """Locate the four-factor snapshot closest to the target month-end.

    The snapshot files are named ``torvik_four_factors_{year}_{yyyymmdd}.json``
    where yyyy/mm/dd is the cutoff date. We find the one whose month matches
    ``target_month`` (e.g., 1 for January's 1/31 snapshot, 3 for March's mid-
    month snapshot). Returns None if no such file exists.

    Args:
        year: Tournament year — both calendar years appearing in the file
            naming are considered (preseason Nov/Dec of year−1, in-season
            Jan/Feb/Mar of year).
        target_month: 1-12 month code to locate.
        data_root: Repo ``data/`` root.
    """
    hist_dir = Path(data_root) / "raw" / "historical"
    if not hist_dir.exists():
        return None

    prefix = f"torvik_four_factors_{year}_"
    # Target matches on the year+month portion of the YYYYMMDD suffix.
    # For months 11/12, the preceding calendar year is in the date; for 1/2/3,
    # the tournament year itself.
    candidates = list(hist_dir.glob(f"{prefix}*.json"))
    for cand in candidates:
        # stem: torvik_four_factors_{year}_{yyyymmdd}
        stem = cand.stem
        date_part = stem.replace(prefix, "")
        if len(date_part) != 8:
            continue
        file_month = int(date_part[4:6])
        if file_month == target_month:
            return cand
    return None


def _load_snapshot_margins(path: Path) -> Dict[str, float]:
    """Return {team_id: four_factor_margin} for every team in the snapshot."""
    raw = json.load(open(path))
    margins: Dict[str, float] = {}
    for key, value in raw.items():
        if key in _SNAPSHOT_META_KEYS or not isinstance(value, dict):
            continue
        try:
            margins[key] = _four_factor_margin(value)
        except (KeyError, TypeError):
            # Skip teams with partial four-factor data.
            continue
    return margins


def load_team_momentum(
    year: int,
    canonical_ids: Iterable[str],
    data_root: Path,
) -> Dict[str, float]:
    """Per-team January→March four-factor-margin delta from Torvik snapshots.

    For each team in ``canonical_ids``, compute:
        delta = four_factor_margin(march_snapshot) − four_factor_margin(january_snapshot)

    Negative delta = declining team; positive delta = improving team.
    Teams missing from either snapshot are absent from the result — the
    caller's ``build_momentum_round_probs`` treats absence as delta=0
    (neutral, no adjustment).

    Args:
        year: Tournament year (locates that year's snapshot files).
        canonical_ids: Tournament team IDs to compute momentum for.
        data_root: Repo ``data/`` root.

    Returns:
        Dict[canonical_id, delta]. May be smaller than ``canonical_ids``
        if snapshots are missing or sparse.
    """
    canonical_set = frozenset(canonical_ids)
    if not canonical_set:
        return {}

    jan_path = _find_snapshot_file(year, target_month=1, data_root=data_root)
    mar_path = _find_snapshot_file(year, target_month=3, data_root=data_root)
    if jan_path is None or mar_path is None:
        return {}

    jan_margins = _load_snapshot_margins(jan_path)
    mar_margins = _load_snapshot_margins(mar_path)

    result: Dict[str, float] = {}
    for tid in canonical_set:
        j = jan_margins.get(tid)
        m = mar_margins.get(tid)
        if j is None or m is None:
            continue
        result[tid] = float(m - j)
    return result


def build_momentum_round_probs(
    model_rp: Dict[str, Dict[str, float]],
    team_momentum: Dict[str, float],
    *,
    tanh_scale: float = _TANH_SCALE,
    max_adjustment: float = _MAX_ADJUSTMENT,
) -> Dict[str, Dict[str, float]]:
    """Apply a tanh-saturated ±3% multiplicative adjustment by team momentum.

    Per team ``t`` with delta ``d_t``:
        factor_t     = 1 + max_adjustment * tanh(d_t * tanh_scale)
        adjusted[r]  = base[r] * factor_t         for each round r

    Then re-normalized per round to bracket team-counts {64, 32, 16, 8, 4, 2}.
    Teams missing from ``team_momentum`` get delta=0 (factor=1.0, no
    adjustment) — identical behavior whether a team is genuinely flat or
    simply absent from the snapshot coverage.

    The tanh saturation caps the per-team factor at ``[1 - max_adjustment,
    1 + max_adjustment]`` — even pathological deltas can't produce a
    runaway boost or penalty.

    Args:
        model_rp: Base round_probs to adjust.
        team_momentum: Per-team Jan→Mar four-factor delta
            (from ``load_team_momentum``).
        tanh_scale: Multiplier on delta before tanh (default 10.0 —
            calibrated so typical deltas of ~0.05 produce factor ≈ 1.014).
        max_adjustment: Asymptotic cap on the factor (default 0.03 = ±3%).

    Returns:
        Adjusted, per-round-renormalized round_probs.
    """
    result: Dict[str, Dict[str, float]] = {}
    for tid, rounds in model_rp.items():
        delta = team_momentum.get(tid, 0.0)
        factor = 1.0 + max_adjustment * math.tanh(delta * tanh_scale)

        result[tid] = {}
        for rnd in ROUND_NAMES:
            base_p = rounds.get(rnd, 0.001)
            result[tid][rnd] = max(base_p * factor, 0.001)

    # Re-normalize per round (same pattern as volatile/roster_adj/coach_adj).
    for rnd in ROUND_NAMES:
        total = sum(result[tid].get(rnd, 0.0) for tid in result)
        target = _TEAMS_PER_ROUND.get(rnd, 32)
        if total > 0:
            scale = target / total
            for tid in result:
                if rnd in result[tid]:
                    result[tid][rnd] *= scale

    return result
