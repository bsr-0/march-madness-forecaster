"""Shared helper: resolve a probability-base name to (rating, round_probs).

Used by every docs/data/bracket_2026*.json generator script so "torvik" /
"elo" / "ap" mean the same thing everywhere.

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
}

DATA_ROOT = PROJECT_ROOT / "data"


def load_prob_base(
    base: str,
    year: int,
    seeds: Dict[str, int],
    regions: Dict[str, str],
    torvik_rp: Dict[str, Dict[str, float]],
    barthag: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Resolve "torvik" | "elo" | "ap" to (rating, round_probs).

    rating: scalar per-team barthag-equivalent, used for displayed Log5
        win_prob in the exported bracket JSON (see build_bracket_json) and
        for Chalk's live client-side lens.
    round_probs: what construct_bracket actually optimizes candidates
        over. For elo/ap this is a FRESH Monte Carlo simulation from that
        base's own barthag (via the same build_torvik_round_probabilities
        machinery Torvik uses) -- not a rescale of Torvik's round_probs --
        so round_probs and the displayed win_prob can never structurally
        disagree the way a post-hoc rescale can.
    """
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

    raise ValueError(f"Unknown prob base: {base!r}. Valid: torvik, elo, ap.")
