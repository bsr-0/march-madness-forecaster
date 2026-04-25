"""Massey-best source (catalog A6).

Walks forward over the per-system ranking archive and picks the system
with lowest cumulative Brier on NCAA tournament games in all prior
years. Uses that system's current-year ratings to produce round_probs.

Distinct from massey_avg (A5):
- A5 uses the *pre-aggregated* Massey composite (ensemble of ~150 systems).
- A6 *selects* a single best-historical-Brier system per test year.

A5 is a fixed ensemble; A6 is data-driven ranker selection. The
orthogonality argument: every other source in the catalog uses one
fixed model, or statically ensembles many. A6 is the only source where
the data picks the ranker each year. Closes catalog phase 1d-2.

Relationship to MEMORY.md §2 D1 (BSS=0 ceiling): A6 does NOT claim to
improve prediction accuracy. It picks the best *available* system per
year. If every individual system has BSS≈0, the selected one still does.
The open question is whether seed-level calibration differs enough
between systems to shift P(1st). That's a catalog experiment question,
orthogonal to D1's BSS verdict.

Walk-forward contract: selection at test_year Y uses only tournament
outcomes from years strictly < Y. Each test year's selection is
computed in isolation.

Data:
- ``data/raw/historical/external_{SYSTEM}_{year}.json`` — 56+ systems,
  22 years of coverage (2005-2026). Each file is a list of
  ``{team_id, team_name, rating, ranking, normalized}`` dicts.
  ``normalized`` is a [0, 1] scalar suitable for direct use as a
  barthag-equivalent.
- ``data/raw/historical/tournament_results_{year}.json`` — per-year
  tournament games list, used for Brier scoring. Same file used by
  ``upset_tuned``.

Brier scoring: for each tournament game, compute P(team1 beats team2)
via Log5 on the system's ``normalized`` field and score against the
actual outcome. Per-game ``brier = (P - y_actual)²``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ROUND_NAMES = ("R64", "R32", "S16", "E8", "F4", "CHAMP")

# Minimum cumulative tournament games a system must have been scored on
# to be eligible for selection. Screens out systems with only a handful
# of years of coverage or sparse per-year rosters.
#
# 500 games ≈ 8 prior tournaments of coverage (63 games × 8 = 504). Below
# this, selection over-fits to whichever 1-2 tournaments a new system
# happened to be around for. At min_games=100, the 2026 selection is
# "DP" with only 113 games — a likely fluke. At min_games=500 the pool
# is 47 well-established rankers (TRK, BWE, TRP, EBP, LOG, ...) with
# genuine track records. The walk-forward Brier ranking within that
# filtered pool is then a data-defensible selection.
_DEFAULT_MIN_GAMES = 500

# 2020 excluded from Brier cumulation (COVID — no tournament).
_EXCLUDED_YEARS = {2020}

# Round names on tournament_results games that contribute to Brier.
# "FF" (First Four) games are pre-tournament play-ins; we skip them since
# most systems' ratings are derived from the First-Four-inclusive field
# and FF games aren't part of the 64-team bracket scoring.
_VALID_ROUND_NAMES = {"R64", "R32", "S16", "E8", "F4", "CHAMP"}


def _pairwise_win_prob(norm_t1: float, norm_t2: float) -> float:
    """Log5 pairwise win probability from two [0, 1] rating values.

    Standard formula:
        P(t1 wins) = (n1 - n1 * n2) / (n1 + n2 - 2 * n1 * n2)

    Guards against the 0/0 case (both ratings zero) by returning 0.5.
    """
    if norm_t1 <= 0.0 and norm_t2 <= 0.0:
        return 0.5
    denom = norm_t1 + norm_t2 - 2.0 * norm_t1 * norm_t2
    if denom <= 1e-12:
        return 0.5
    return (norm_t1 - norm_t1 * norm_t2) / denom


def _load_system_ratings(system: str, year: int, data_root: Path) -> Optional[Dict[str, float]]:
    """Load {team_id: normalized_rating} for one (system, year) file."""
    path = Path(data_root) / "raw" / "historical" / f"external_{system}_{year}.json"
    if not path.exists():
        return None
    raw = json.load(open(path))
    if not isinstance(raw, list):
        return None
    result: Dict[str, float] = {}
    for entry in raw:
        tid = entry.get("team_id")
        norm = entry.get("normalized")
        if tid and isinstance(norm, (int, float)):
            result[tid] = float(norm)
    return result


def _load_tournament_games(year: int, data_root: Path) -> Optional[List[dict]]:
    """Load the games list for one tournament year, or None if missing."""
    path = Path(data_root) / "raw" / "historical" / f"tournament_results_{year}.json"
    if not path.exists():
        return None
    raw = json.load(open(path))
    if isinstance(raw, dict) and "games" in raw:
        return raw["games"]
    if isinstance(raw, list):
        return raw
    return None


def _score_system_on_year(
    system: str,
    year: int,
    data_root: Path,
) -> Optional[Tuple[float, int]]:
    """Brier sum + game count for (system, year) over valid tournament games.

    Returns None if the system's ratings file or the year's tournament
    results file is missing.

    Games where either team is absent from the system's ratings are
    skipped — the system isn't penalized for not ranking a First Four
    winner it had no data for, but those games also don't contribute
    to its Brier on teams it does rank. Same missing-teams treatment
    applies uniformly across systems, so selection remains fair.
    """
    ratings = _load_system_ratings(system, year, data_root)
    games = _load_tournament_games(year, data_root)
    if ratings is None or games is None:
        return None

    brier_sum = 0.0
    n_games = 0
    for g in games:
        rnd = g.get("round_name")
        if rnd not in _VALID_ROUND_NAMES:
            continue
        t1 = g.get("team1_id")
        t2 = g.get("team2_id")
        if not t1 or not t2:
            continue
        r1 = ratings.get(t1)
        r2 = ratings.get(t2)
        if r1 is None or r2 is None:
            continue
        p1 = _pairwise_win_prob(r1, r2)
        actual = 1.0 if g.get("team1_won") else 0.0
        brier_sum += (p1 - actual) ** 2
        n_games += 1

    return (brier_sum, n_games)


def _list_available_systems_for_year(year: int, data_root: Path) -> List[str]:
    """Every SYSTEM for which ``external_{SYSTEM}_{year}.json`` exists."""
    hist_dir = Path(data_root) / "raw" / "historical"
    if not hist_dir.exists():
        return []
    prefix = "external_"
    suffix = f"_{year}.json"
    systems: List[str] = []
    pattern = re.compile(rf"^external_(.+?)_{year}\.json$")
    for p in hist_dir.glob(f"{prefix}*{suffix}"):
        m = pattern.match(p.name)
        if m:
            sys_name = m.group(1)
            # Skip the composite (used by massey_avg) — massey_best is about
            # single-system selection, not re-using the composite.
            if sys_name == "massey_composite":
                continue
            systems.append(sys_name)
    return systems


def _cumulative_brier_per_system(
    test_year: int,
    data_root: Path,
) -> Dict[str, Tuple[float, int]]:
    """{system: (brier_sum, n_games)} cumulated across years strictly < test_year.

    Walk-forward: only prior-year tournaments contribute. 2020 excluded
    (COVID — no tournament file exists). Systems not present in any
    prior year don't appear in the result.
    """
    # Gather the union of systems that appeared in any year before test_year.
    all_systems: set = set()
    hist_dir = Path(data_root) / "raw" / "historical"
    pattern = re.compile(r"^external_(.+?)_(\d{4})\.json$")
    if hist_dir.exists():
        for p in hist_dir.glob("external_*.json"):
            m = pattern.match(p.name)
            if not m:
                continue
            sys_name = m.group(1)
            if sys_name == "massey_composite":
                continue
            year = int(m.group(2))
            if year < test_year and year not in _EXCLUDED_YEARS:
                all_systems.add(sys_name)

    accum: Dict[str, Tuple[float, int]] = {}
    for sys_name in all_systems:
        brier_sum = 0.0
        n_games = 0
        # Only walk years strictly < test_year for which both the system's
        # ratings file and a tournament_results file exist.
        for year in range(2005, test_year):
            if year in _EXCLUDED_YEARS:
                continue
            result = _score_system_on_year(sys_name, year, data_root)
            if result is None:
                continue
            brier_sum += result[0]
            n_games += result[1]
        if n_games > 0:
            accum[sys_name] = (brier_sum, n_games)
    return accum


def select_best_system(
    test_year: int,
    data_root: Path,
    min_games: int = _DEFAULT_MIN_GAMES,
) -> Optional[str]:
    """Argmin-Brier system over years < test_year, respecting ``min_games``.

    Returns None if no system has at least ``min_games`` scored games
    over the walk-forward window, or if there are no prior years with
    both ratings and tournament results.

    The system must also have a ratings file for ``test_year`` itself —
    otherwise its selected ratings are useless for producing round_probs.
    """
    accum = _cumulative_brier_per_system(test_year, data_root)
    if not accum:
        return None

    # Only consider systems that (a) have >= min_games historical coverage
    # and (b) have a ratings file for test_year.
    available_for_test = set(_list_available_systems_for_year(test_year, data_root))

    best_sys: Optional[str] = None
    best_mean_brier = float("inf")
    for sys_name, (brier_sum, n_games) in accum.items():
        if n_games < min_games:
            continue
        if sys_name not in available_for_test:
            continue
        mean_brier = brier_sum / n_games
        if mean_brier < best_mean_brier:
            best_mean_brier = mean_brier
            best_sys = sys_name
    return best_sys


def load_massey_best_barthag(
    test_year: int,
    data_root: Path,
    min_games: int = _DEFAULT_MIN_GAMES,
) -> Optional[Dict[str, float]]:
    """Load the selected system's test-year normalized ratings as barthag.

    Returns None if no system qualifies for selection. Teams absent from
    the selected system's ratings will be absent from the returned dict —
    downstream round_probs builders apply their own seed-based fallback
    for missing teams.
    """
    best = select_best_system(test_year, data_root, min_games=min_games)
    if best is None:
        return None
    ratings = _load_system_ratings(best, test_year, data_root)
    if ratings is None:
        return None
    # Clip to [0.10, 0.99] per the catalog pattern (matches massey_avg).
    return {tid: max(0.10, min(0.99, norm)) for tid, norm in ratings.items()}


def build_massey_best_round_probabilities(
    seeds: Dict[str, int],
    regions: Dict[str, str],
    test_year: int,
    data_root: Path,
    n_sims: int = 10000,
    min_games: int = _DEFAULT_MIN_GAMES,
) -> Optional[Dict[str, Dict[str, float]]]:
    """End-to-end: select best system → load its ratings → MC round_probs.

    Uses the same MC construction as torvik (``build_torvik_round_probabilities``).
    Teams not in the selected system's ratings get the seed-based fallback
    ``max(0.10, 1.0 - seed * 0.04)``.

    Returns None if no system qualifies for selection — the caller should
    treat that as "massey_best source unavailable this year" (same pattern
    as missing torvik data).
    """
    barthag_from_system = load_massey_best_barthag(test_year, data_root, min_games=min_games)
    if barthag_from_system is None:
        return None

    # Apply seed-based fallback for any team missing from the system's ratings.
    barthag: Dict[str, float] = {}
    for tid in seeds:
        if tid in barthag_from_system:
            barthag[tid] = barthag_from_system[tid]
        else:
            s = seeds[tid]
            barthag[tid] = max(0.10, 1.0 - float(s) * 0.04)

    # Delegate to the torvik round-probs MC (same Log5-with-barthag mechanism).
    from scripts.mc_pool_backtest import build_torvik_round_probabilities

    return build_torvik_round_probabilities(seeds, regions, barthag, n_sims=n_sims)
