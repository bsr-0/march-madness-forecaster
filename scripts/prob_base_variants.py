"""Shared helper: resolve a probability-base name to (rating, round_probs).

Used by every docs/data/bracket_2026*.json generator script so "torvik" /
"roster" / "coach" mean the same adjustment everywhere — one source of
truth for the per-team factor, instead of copies that could silently
drift apart (see the _TEAMS_PER_ROUND bug from 2026-08-13 for what
duplicating this kind of constant costs).

roster/coach are exploratory lenses, not validated strategies: tested
against the production meta_region_poolaware construction, both lost
(MEMORY.md D19 — 10.4%/11.1% P(1st) vs 11.2% baseline, 0/15 years ahead).
"""

import math
import statistics
from pathlib import Path
from typing import Dict, Tuple

from src.prediction.coach_adj_probabilities import (
    _LOG_CEILING,
    _LOG_SCALE,
    build_coach_adj_round_probs,
    load_coach_experience,
)
from src.prediction.roster_adj_probabilities import (
    _MAX_ADJUSTMENT,
    _Z_SCALE,
    build_roster_adj_round_probs,
    load_team_talent,
)

MODEL_LABELS = {
    "torvik": "Torvik Barthag",
    "roster": "Roster Talent-Adjusted Torvik",
    "coach": "Coach Experience-Adjusted Torvik",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"


def roster_factor(z: float) -> float:
    """Same per-team multiplicative factor as build_roster_adj_round_probs,
    applied to a scalar barthag instead of round_probs."""
    return 1.0 + max(-_MAX_ADJUSTMENT, min(_MAX_ADJUSTMENT, _Z_SCALE * z))


def coach_factor(prior_apps: int) -> float:
    """Same per-team multiplicative factor as build_coach_adj_round_probs,
    applied to a scalar barthag instead of round_probs."""
    return 1.0 + _LOG_SCALE * min(math.log(1 + prior_apps), _LOG_CEILING)


def load_prob_base(
    base: str,
    year: int,
    seeds: Dict[str, int],
    torvik_rp: Dict[str, Dict[str, float]],
    barthag: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Resolve "torvik" | "roster" | "coach" to (rating, round_probs).

    rating: scalar per-team barthag-equivalent, used for displayed Log5
        win_prob in the exported bracket JSON (see build_bracket_json).
    round_probs: what construct_bracket actually optimizes candidates over.
    """
    if base == "torvik":
        return barthag, torvik_rp

    if base == "roster":
        team_talent = load_team_talent(year, seeds.keys(), DATA_ROOT)
        vals = list(team_talent.values())
        mean = statistics.fmean(vals) if len(vals) >= 2 else 0.0
        std = statistics.stdev(vals) if len(vals) >= 2 else 0.0
        rating = {}
        for tid in seeds:
            w = team_talent.get(tid)
            z = (w - mean) / std if w is not None and std > 1e-9 else 0.0
            rating[tid] = barthag.get(tid, 0.0) * roster_factor(z)
        round_probs = build_roster_adj_round_probs(torvik_rp, team_talent)
        return rating, round_probs

    if base == "coach":
        coach_experience = load_coach_experience(year, seeds.keys(), DATA_ROOT)
        rating = {tid: barthag.get(tid, 0.0) * coach_factor(coach_experience.get(tid, 0)) for tid in seeds}
        round_probs = build_coach_adj_round_probs(torvik_rp, coach_experience)
        return rating, round_probs

    raise ValueError(f"Unknown prob base: {base!r}. Valid: torvik, roster, coach.")
