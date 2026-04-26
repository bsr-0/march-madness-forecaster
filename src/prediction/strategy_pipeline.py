"""Composable strategy pipeline: source → adjustments → round_probs.

Separates team strength estimation into three composable layers:
  1. SOURCES: Raw team ratings (odds, torvik, elo, massey, etc.)
  2. ADJUSTMENTS: Modifiers applied to ratings (coach, momentum, contrarian)
  3. BLENDS: Weighted combinations of multiple sources

Strategies are specified as pipeline strings:
  "odds"                    → single source
  "odds+contrarian"         → source + adjustment
  "odds+coach_adj+contrarian" → source + chained adjustments
  "0.7*torvik+0.3*odds"     → weighted blend of two sources
  "consensus+contrarian"    → blend + adjustment (consensus = equal-weight average)

The experiment loop auto-generates all valid permutations for testing.
"""

from __future__ import annotations

import re
from itertools import combinations, product
from typing import Dict, List, Optional, Sequence, Tuple


# Registry categories
SOURCES = (
    "seed",  # A1
    "torvik",  # A2
    "odds",  # A3
    "elo",  # A4
    "massey_avg",  # A5
    "massey_best",  # A6
    "spread_power",  # A7
    "ap_strength",  # A8
    "stacked",  # B5
    "pool_wisdom",  # B7
)

ADJUSTMENTS = (
    "contrarian",  # B6: ownership-gap adjustment
    "coach_adj",  # C1 (not yet implemented)
    "roster_adj",  # C2 (not yet implemented)
    "momentum",  # C3 (not yet implemented)
    "volatile",  # D1 (not yet implemented)
    "upset_tuned",  # D2 (not yet implemented)
)

CONSTRUCTIONS = (
    "forward",  # M1
    "champ_first",  # M2
    "f4_first",  # M3
    "e8_first",  # M4
    "confidence",  # M6
    "f4_chalk",  # M3a: f4_first restricted to seeds 1-3 anchors
    "f4_diverse",  # M3b: f4_first excluding 1-seeds from anchor pool
    "f4_top4",  # M3c: f4_first restricted to seeds 1-4 anchors
    "e8_chalk",  # M4a: e8_first restricted to top seeds (1-6) per quadrant
    "e8_diverse",  # M4b: e8_first excluding 1-seeds from S16 anchor pool
    # "backward",    # M5 (not yet implemented)
)

# Which sources and adjustments are actually implemented and available
IMPLEMENTED_SOURCES = {
    "seed",
    "torvik",
    "odds",
    "spread_power",
    "pool_wisdom",
    "elo",
    "massey_avg",
    "massey_best",
    "ap_strength",
    "stacked",
}
IMPLEMENTED_ADJUSTMENTS = {"contrarian", "upset_tuned", "volatile", "roster_adj", "coach_adj", "momentum"}
IMPLEMENTED_CONSTRUCTIONS = {
    "forward",
    "champ_first",
    "f4_first",
    "e8_first",
    "confidence",
    "f4_chalk",
    "f4_diverse",
    "f4_top4",
    "e8_chalk",
    "e8_diverse",
    "backward",
}


def parse_pipeline(spec: str) -> Tuple[List[Tuple[float, str]], List[str], str]:
    """Parse a pipeline specification string.

    Format: "[weight*]source[+source...][+adjustment...][_construction]"

    Examples:
        "torvik_forward" → sources=[torvik], adjustments=[], construction=forward
        "odds+contrarian_f4_first" → sources=[odds], adjustments=[contrarian], construction=f4_first
        "0.7*torvik+0.3*odds+contrarian_e8_first" → blend + adjustment + construction

    Returns:
        (sources_with_weights, adjustments, construction_mode)
        sources_with_weights: [(weight, source_name), ...]
        adjustments: [adj_name, ...]
        construction_mode: str
    """
    # Split off construction mode (last segment after final _)
    construction = "forward"  # default
    for cmode in sorted(CONSTRUCTIONS, key=len, reverse=True):
        suffix = f"_{cmode}"
        if spec.endswith(suffix):
            construction = cmode
            spec = spec[: -len(suffix)]
            break

    # Split remaining by + into components
    components = spec.split("+")

    sources = []
    adjustments = []

    for comp in components:
        comp = comp.strip()
        if not comp:
            continue

        # Check if it's a weighted source: "0.7*torvik"
        weight_match = re.match(r"^(\d+\.?\d*)\*(.+)$", comp)
        if weight_match:
            weight = float(weight_match.group(1))
            name = weight_match.group(2)
            sources.append((weight, name))
        elif comp in ADJUSTMENTS:
            adjustments.append(comp)
        elif comp in SOURCES:
            sources.append((1.0, comp))
        else:
            # Try as source anyway (may be a future source)
            sources.append((1.0, comp))

    # If no explicit weights, equal-weight all sources
    if sources and all(w == 1.0 for w, _ in sources) and len(sources) > 1:
        equal_w = 1.0 / len(sources)
        sources = [(equal_w, s) for _, s in sources]

    if not sources:
        sources = [(1.0, "seed")]

    return sources, adjustments, construction


def build_pipeline_name(
    sources: List[Tuple[float, str]],
    adjustments: List[str],
    construction: str,
) -> str:
    """Build a canonical pipeline name from components."""
    parts = []
    if len(sources) == 1 and sources[0][0] == 1.0:
        parts.append(sources[0][1])
    else:
        for w, s in sources:
            if w == 1.0:
                parts.append(s)
            else:
                parts.append(f"{w:.1f}*{s}")
    for adj in adjustments:
        parts.append(adj)
    name = "+".join(parts)
    if construction != "forward":
        name += f"_{construction}"
    return name


def generate_all_permutations(
    sources: Optional[Sequence[str]] = None,
    adjustments: Optional[Sequence[str]] = None,
    constructions: Optional[Sequence[str]] = None,
    max_blend_size: int = 2,
    max_adjustments: int = 2,
    implemented_only: bool = True,
) -> List[str]:
    """Generate all valid strategy permutations.

    Args:
        sources: Source bases to include (default: all implemented)
        adjustments: Adjustments to include (default: all implemented)
        constructions: Construction modes (default: all implemented)
        max_blend_size: Maximum number of sources to blend (1=no blending, 2=pairs, 3=triples)
        max_adjustments: Maximum adjustments to chain (0=none, 1=one, 2=two)
        implemented_only: Only include implemented components

    Returns:
        List of pipeline specification strings.
    """
    if sources is None:
        sources = list(IMPLEMENTED_SOURCES if implemented_only else SOURCES)
    if adjustments is None:
        adjustments = list(IMPLEMENTED_ADJUSTMENTS if implemented_only else ADJUSTMENTS)
    if constructions is None:
        constructions = list(IMPLEMENTED_CONSTRUCTIONS if implemented_only else CONSTRUCTIONS)

    strategies = []

    # Generate source combinations (single + blends)
    source_combos = []
    for size in range(1, min(max_blend_size, len(sources)) + 1):
        for combo in combinations(sources, size):
            if size == 1:
                source_combos.append([(1.0, combo[0])])
            else:
                # Equal weight blend
                w = 1.0 / size
                source_combos.append([(w, s) for s in combo])

    # Generate adjustment chains (none, single, pairs)
    adj_combos = [[]]  # empty = no adjustments
    for size in range(1, min(max_adjustments, len(adjustments)) + 1):
        for combo in combinations(adjustments, size):
            adj_combos.append(list(combo))

    # Cross-product: sources × adjustments × constructions
    for src_combo in source_combos:
        for adj_combo in adj_combos:
            for construction in constructions:
                name = build_pipeline_name(src_combo, adj_combo, construction)
                strategies.append(name)

    return strategies


def resolve_pipeline_round_probs(
    sources_with_weights: List[Tuple[float, str]],
    adjustments: List[str],
    base_round_probs: Dict[str, Dict[str, Dict[str, float]]],
    pick_dist: Dict[str, Dict[str, float]],
    *,
    seeds: Optional[Dict[str, int]] = None,
    historical_seed_reach_rates: Optional[Dict[int, Dict[str, float]]] = None,
    team_volatility: Optional[Dict[str, float]] = None,
    team_talent: Optional[Dict[str, float]] = None,
    coach_experience: Optional[Dict[str, int]] = None,
    team_momentum: Optional[Dict[str, float]] = None,
) -> Optional[Dict[str, Dict[str, float]]]:
    """Resolve a pipeline specification into round_probs.

    Args:
        sources_with_weights: [(weight, source_name), ...]
        adjustments: [adj_name, ...]
        base_round_probs: Pre-computed round_probs for each source
        pick_dist: Public pick distribution (for contrarian adjustment)
        seeds: Per-team seed map (required for upset_tuned adjustment)
        team_volatility: Per-team [0,1]-normalized volatility
            (required for volatile adjustment; produced by
            ``src.prediction.volatile_probabilities.load_team_volatility``)
        team_talent: Per-team top-5 mean WARP (required for roster_adj
            adjustment; produced by
            ``src.prediction.roster_adj_probabilities.load_team_talent``).
        coach_experience: Per-team prior NCAA tournament appearance count
            for the head coach (required for coach_adj adjustment;
            produced by
            ``src.prediction.coach_adj_probabilities.load_coach_experience``).
        team_momentum: Per-team Jan→Mar four-factor-margin delta
            (required for momentum adjustment; produced by
            ``src.prediction.momentum_probabilities.load_team_momentum``).
        historical_seed_reach_rates: Walk-forward seed-by-round empirical
            rates (required for upset_tuned; produced by
            ``src.prediction.upset_tuned_probabilities.load_upset_tuned_context``)

    Returns:
        Blended and adjusted round_probs, or None if sources unavailable or
        a required adjustment context is missing.
    """
    from src.prediction.contrarian_probabilities import build_contrarian_round_probs

    # Step 1: Blend sources
    available_sources = []
    for weight, source in sources_with_weights:
        if source in base_round_probs:
            available_sources.append((weight, base_round_probs[source]))
        else:
            return None  # Source not available for this year

    if not available_sources:
        return None

    # Weighted blend
    if len(available_sources) == 1:
        result = available_sources[0][1]
    else:
        # Re-normalize weights for available sources
        total_w = sum(w for w, _ in available_sources)
        result = {}
        for tid in available_sources[0][1]:
            result[tid] = {}
            for rnd in ("R64", "R32", "S16", "E8", "F4", "CHAMP"):
                blended = sum((w / total_w) * rp.get(tid, {}).get(rnd, 0.001) for w, rp in available_sources)
                result[tid][rnd] = blended

    # Step 2: Apply adjustments in order
    for adj in adjustments:
        if adj == "contrarian":
            result = build_contrarian_round_probs(result, pick_dist)
        elif adj == "upset_tuned":
            if seeds is None or historical_seed_reach_rates is None:
                return None  # Required context not supplied
            from src.prediction.upset_tuned_probabilities import build_upset_tuned_round_probs

            result = build_upset_tuned_round_probs(result, seeds, historical_seed_reach_rates)
        elif adj == "volatile":
            if team_volatility is None:
                return None  # Required context not supplied
            from src.prediction.volatile_probabilities import build_volatile_round_probs

            result = build_volatile_round_probs(result, team_volatility)
        elif adj == "roster_adj":
            if team_talent is None:
                return None  # Required context not supplied
            from src.prediction.roster_adj_probabilities import build_roster_adj_round_probs

            result = build_roster_adj_round_probs(result, team_talent)
        elif adj == "coach_adj":
            if coach_experience is None:
                return None  # Required context not supplied
            from src.prediction.coach_adj_probabilities import build_coach_adj_round_probs

            result = build_coach_adj_round_probs(result, coach_experience)
        elif adj == "momentum":
            if team_momentum is None:
                return None  # Required context not supplied
            from src.prediction.momentum_probabilities import build_momentum_round_probs

            result = build_momentum_round_probs(result, team_momentum)
        else:
            return None  # Adjustment not implemented

    return result
