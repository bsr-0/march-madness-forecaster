"""Shared helper: resolve a probability-base name to (rating, round_probs).

Used by every docs/data/bracket_2026*.json generator script so "torvik" /
"elo" / "ap" / "upset" mean the same thing everywhere.

Superseded 2026-08-15: roster_adj/coach_adj (top-5 WARP, coach tournament
experience) were the original probability bases here, but investigation
showed they rarely move any pick for the 2026 field — they're small,
capped (+-4%/+3%), post-hoc rescales of Torvik's own round_probs, and
Torvik's structural gaps between top teams are usually bigger than what
the cap can close. Only 1-2 of 63 games ever differed from Torvik across
any construction. Replaced with elo and ap_strength: fully independent
rating systems (not derived from Torvik at all), each simulated fresh via
the same MC round_probs builder Torvik uses. Checked against the 2026
field before adopting: elo disagrees with Torvik on 20/68 first-round
favorites (and flips the Duke-Michigan championship favorite outright);
ap_strength disagrees on 14/68. Both are "preference lens" bases, not
separately backtested as standalone P(1st) contenders — see the honest
framing in docs/app.js's PROB_BASE_DEFS.

Added 2026-08-15 (same session): even elo/ap kept Duke/Michigan/Arizona
in the Final Four most of the time — reasonable rating systems converge
on the same genuinely elite teams. Real variation at the Final Four
level (not just who wins between the same two) needed a second axis:
risk_level, bracket_construction.py's own contrarian-weighting knob
(0 = pure probability, 1 = maximum differentiation-seeking). At
risk_level=1.0 on elo round_probs, region_top_n produces a Final Four
with ZERO 1-seeds (checked against the real 2026 field: St. John's,
Gonzaga, Saint Mary's, Miami OH — champion Miami OH). "upset" reuses
elo's round_probs but forces risk_level=1.0 in construction instead of
each approach's normal risk_level — see RISK_LEVEL below. Not run
through Pool Optimizer's own pool-simulation candidate selection (that
selection correctly rejects near-0%-real-win-chance candidates by
design, which is WHY it's the validated strategy — see
generate_poolaware_bracket.py for how "upset" bypasses that selection
for a direct single-shot construction instead).
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mc_pool_backtest import build_torvik_round_probabilities
from src.prediction.ap_probabilities import load_ap_strength_barthag
from src.prediction.elo_probabilities import load_elo_barthag

MODEL_LABELS = {
    "torvik": "Torvik Barthag",
    "elo": "Elo Rating",
    "ap": "AP Poll Strength",
    "upset": "Elo Rating, Max Contrarian",
}

# Construction risk_level per base — 0 = pure probability-driven picks,
# 1 = maximum weight on differentiation/contrarian value (bracket_
# construction.py's _make_ev_scorer blended_diff). Only "upset" deviates
# from each approach's normal default (0.5 for exhaustive/region; Pool
# Optimizer's own diversity sweep covers 0.1-0.9 already and is untouched
# by this — see generate_poolaware_bracket.py).
RISK_LEVEL = {
    "torvik": 0.5,
    "elo": 0.5,
    "ap": 0.5,
    "upset": 1.0,
}

# "upset" is a risk_level override on top of elo's round_probs, not a
# separate rating system — this maps it back to what load_prob_base()
# actually knows how to compute.
UNDERLYING_BASE = {"upset": "elo"}

DATA_ROOT = PROJECT_ROOT / "data"


def load_prob_base(
    base: str,
    year: int,
    seeds: Dict[str, int],
    regions: Dict[str, str],
    torvik_rp: Dict[str, Dict[str, float]],
    barthag: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Resolve "torvik" | "elo" | "ap" | "upset" to (rating, round_probs).

    rating: scalar per-team barthag-equivalent, used for displayed Log5
        win_prob in the exported bracket JSON (see build_bracket_json) and
        for Chalk's live client-side lens.
    round_probs: what construct_bracket actually optimizes candidates
        over. For elo/ap/upset this is a FRESH Monte Carlo simulation from
        that base's own barthag (via the same build_torvik_round_
        probabilities machinery Torvik uses) -- not a rescale of Torvik's
        round_probs -- so round_probs and the displayed win_prob can never
        structurally disagree the way a post-hoc rescale can.
    """
    base = UNDERLYING_BASE.get(base, base)

    if base == "torvik":
        return barthag, torvik_rp

    if base == "elo":
        elo_barthag = load_elo_barthag(year, seeds, DATA_ROOT)
        if elo_barthag is None:
            return barthag, torvik_rp
        return elo_barthag, build_torvik_round_probabilities(seeds, regions, elo_barthag)

    if base == "ap":
        ap_barthag = load_ap_strength_barthag(year, seeds, seeds.keys(), DATA_ROOT)
        if ap_barthag is None:
            return barthag, torvik_rp
        return ap_barthag, build_torvik_round_probabilities(seeds, regions, ap_barthag)

    raise ValueError(f"Unknown prob base: {base!r}. Valid: torvik, elo, ap, upset.")
