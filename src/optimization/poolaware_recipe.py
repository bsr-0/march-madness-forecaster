"""The one definition of the ``meta_region_poolaware`` candidate recipe.

`meta_region_poolaware` is the production pool strategy, and its candidate
set is generated in two places: `scripts/mc_pool_backtest.py` (which measures
it over 14 seasons and produces the headline P(1st) figure) and
`scripts/generate_poolaware_bracket.py` (which builds the bracket that
actually ships to `docs/data/`).

Those two copies drifted. The backtest swept five probability bases; the
shipped script swept two. The gap was not cosmetic — over the 15-season
reference run the backtest selected a base the shipped script cannot build
in 11 of 15 seasons (`blend` in 9, `tv_mass80` in 2), and for 2026 it
selected `tv_mass80_region_risk=0.7` specifically. So the strategy that was
measured and the strategy that shipped were different strategies, and the
measured number did not describe the shipped bracket. See
AUDIT_INDEPENDENT_EVALUATOR_2027.md finding C3.

This module holds the parts that differed, so both callers read one source.
Keep it free of I/O and RNG: it takes already-built probability bases and
returns the sweep, which is what makes it safe for both callers to share and
cheap to test.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

# Risk levels swept against every probability base for `region_top_n`.
POOLAWARE_RISK_LEVELS: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)

# The narrower risk sweep used for `exhaustive_champion` (which is far more
# expensive per candidate, so it gets 3 risk levels rather than 5).
POOLAWARE_EXHAUSTIVE_RISKS: Tuple[float, ...] = (0.3, 0.5, 0.7)

# Weight on torvik in the `tv_mass80` mixed base.
TV_MASS80_TORVIK_WEIGHT: float = 0.8


def build_tv_mass80(
    torvik_rp: Mapping[str, Mapping[str, float]],
    massey_avg_rp: Mapping[str, Mapping[str, float]],
    torvik_weight: float = TV_MASS80_TORVIK_WEIGHT,
) -> Dict[str, Dict[str, float]]:
    """Mix torvik and massey_avg round probabilities in marginal space.

    A team/round missing from ``massey_avg_rp`` falls back to the torvik
    value for that cell, so the mix degrades to plain torvik rather than
    dropping the team.
    """
    mixed: Dict[str, Dict[str, float]] = {}
    for tid in torvik_rp:
        mixed[tid] = {}
        for rn in torvik_rp[tid]:
            tv_val = torvik_rp[tid][rn]
            ma_val = massey_avg_rp.get(tid, {}).get(rn, tv_val)
            mixed[tid][rn] = torvik_weight * tv_val + (1.0 - torvik_weight) * ma_val
    return mixed


def build_poolaware_prob_bases(
    torvik_rp: Mapping[str, Mapping[str, float]],
    *,
    massey_avg: Optional[Mapping[str, Mapping[str, float]]] = None,
    massey_best: Optional[Mapping[str, Mapping[str, float]]] = None,
    blend: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> List[Tuple[str, Mapping[str, Mapping[str, float]]]]:
    """Return the ordered ``(name, round_probs)`` sweep the selector scores.

    ORDER IS LOAD-BEARING. The selector keeps the first candidate that
    achieves the best P(1st) (`p1 > best_p1`, strictly greater), so ties are
    broken by position. Reordering this list can change which bracket ships
    without changing any probability.

    Bases whose data is unavailable for a season are omitted rather than
    substituted — a season with no Massey data sweeps fewer candidates, which
    is what the backtest has always done.
    """
    bases: List[Tuple[str, Mapping[str, Mapping[str, float]]]] = [("tv", torvik_rp)]
    if massey_avg is not None:
        bases.append(("mass_avg", massey_avg))
    if massey_best is not None:
        bases.append(("mass_best", massey_best))
    if blend is not None:
        bases.append(("blend", blend))
    if massey_avg is not None:
        bases.append(("tv_mass80", build_tv_mass80(torvik_rp, massey_avg)))
    return bases


def poolaware_base_names(
    bases: Sequence[Tuple[str, Mapping[str, Mapping[str, float]]]],
) -> List[str]:
    """Names only — for logging and for parity assertions in tests."""
    return [name for name, _rp in bases]
