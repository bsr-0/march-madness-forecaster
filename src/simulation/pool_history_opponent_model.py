"""Pool-calibrated opponent model from historical pool bracket data.

Addresses FP2 (ESPN opponent model mis-calibration): the EV / leverage
calculation previously assumed opponents pick from the ESPN public pick
distribution (~20M entries), but the user's pool is ~30 entries with
materially different pick behavior.  Feeding pool-history data here lets
the optimizer value contrarian picks against *your pool* rather than the
ESPN field.

Input format (``pool_hist_results.json``)::

    {
      "pool": "pool0",
      "years": {
        "2026": {
          "year": 2026,
          "groupSize": 31,
          "brackets": [
            {"rank": 1, "pts": 560, "r64": ["DUKE", ...], "r32": [...],
             "s16": [...], "e8": [...], "f4": [...], "champ": "MICH"},
            ...
          ]
        },
        ...
      }
    }

Teams are named with ESPN abbreviations (``DUKE``, ``MICH``).  The
resolver maps abbrevs to the canonical ``team_id`` used by the rest of
the pipeline (``duke``, ``michigan``).  Unresolved abbrevs are logged
loudly and dropped — no silent fallback (CLAUDE.md §2).
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.data.normalize import normalize_team_id

logger = logging.getLogger(__name__)

# Round ordering used everywhere else in the pipeline.
ROUNDS = ["R64", "R32", "S16", "E8", "F4", "CHAMP"]

# Pool-history key -> pipeline round key.
_POOL_ROUND_MAP = {
    "r64": "R64",
    "r32": "R32",
    "s16": "S16",
    "e8": "E8",
    "f4": "F4",
    "champ": "CHAMP",
}

# ESPN abbreviation -> canonical team_id.  This is the superset of
# abbreviations observed in the available pool-history data (2023-2026)
# plus common NCAA tournament programs not yet seen there.  New abbrevs
# that appear in future years will log a WARNING and be dropped; add
# them here to restore coverage.
#
# Invariant: every value must match the ``team_id`` form produced by
# ``src.data.normalize.normalize_team_id`` for that program so lookups
# against the year's seeds dict succeed.
ABBREV_TO_TEAM_ID: Dict[str, str] = {
    # A
    "AKR": "akron",
    "ALA": "alabama",
    "APPS": "appalachian_state",
    "ARIZ": "arizona",
    "ARK": "arkansas",
    "ASU": "arizona_state",
    "AUB": "auburn",
    # B
    "BAY": "baylor",
    "BOIS": "boise_state",
    "BRAD": "bradley",
    "BRWN": "brown",
    "BUCK": "bucknell",
    "BYU": "brigham_young",
    # C
    "CAL": "california",
    "CLEM": "clemson",
    "COFC": "college_of_charleston",
    "COLG": "colgate",
    "COLO": "colorado",
    "CONN": "connecticut",
    "CREI": "creighton",
    "CSU": "colorado_state",
    # D
    "DAV": "davidson",
    "DAY": "dayton",
    "DEPL": "depaul",
    "DRKE": "drake",
    "DUKE": "duke",
    "DUQ": "duquesne",
    # E
    "EMU": "eastern_michigan",
    "ETSU": "east_tennessee_state",
    # F
    "FAU": "florida_atlantic",
    "FLA": "florida",
    "FSU": "florida_state",
    "FUR": "furman",
    # G
    "GA": "georgia",
    "GCU": "grand_canyon",
    "GONZ": "gonzaga",
    "GT": "georgia_tech",
    "UGA": "georgia",
    # H
    "HALL": "seton_hall",
    "HAW": "hawaii",
    "HOF": "hofstra",
    "HOU": "houston",
    "HPU": "high_point",
    # I
    "ILL": "illinois",
    "IND": "indiana",
    "IONA": "iona",
    "ISI": "illinois_state",  # Illinois State appears as ISI in some ESPN feeds
    "ISU": "iowa_state",
    "IU": "indiana",
    "IOWA": "iowa",
    # J
    "JAX": "jacksonville_state",
    "JMU": "james_madison",
    # K
    "KENT": "kent_state",
    "KSU": "kansas_state",
    "KU": "kansas",
    # L
    "LBSU": "long_beach_state",
    "LIB": "liberty",
    "LIP": "lipscomb",
    "LIU": "long_island_university",
    "LOU": "louisville",
    "LSU": "louisiana_state",
    # M
    "M-OH": "miami__oh",
    "MARQ": "marquette",
    "MCN": "mcneese_state",
    "MD": "maryland",
    "MEM": "memphis",
    "MIA": "miami__fl",
    "MICH": "michigan",
    "MISS": "mississippi",
    "MIZ": "missouri",
    "MONT": "montana",
    "MORE": "morehead_state",
    "MSST": "mississippi_state",
    "MSU": "michigan_state",
    "MTST": "montana_state",
    # N
    "NCST": "nc_state",
    "NCSU": "nc_state",
    "NDSU": "north_dakota_state",
    "NEB": "nebraska",
    "NEV": "nevada",
    "NIU": "northern_iowa",
    "NU": "northwestern",
    # O
    "OKST": "oklahoma_state",
    "OMA": "omaha",
    "ORE": "oregon",
    "ORU": "oral_roberts",
    "OSU": "ohio_state",
    "OU": "oklahoma",
    # P
    "PITT": "pittsburgh",
    "PROV": "providence",
    "PSU": "penn_state",
    "PUR": "purdue",
    # Q-S
    "QNS": "queens__nc",
    "SAM": "samford",
    "SC": "south_carolina",
    "SCU": "santa_clara",
    "SDSD": "south_dakota_state",
    "SDSU": "san_diego_state",
    "SIEN": "siena",
    "SJU": "st__john_s__ny",
    "SLU": "saint_louis",
    "SMC": "saint_mary_s__ca",
    "SMU": "southern_methodist",
    "SPU": "saint_peter_s",
    "STAN": "stanford",
    "SYR": "syracuse",
    # T
    "TA&M": "texas_a_m",
    "TAM": "texas_a_m",
    "TCU": "tcu",
    "TENN": "tennessee",
    "TEX": "texas",
    "TROY": "troy",
    "TTU": "texas_tech",
    # U
    "UAB": "alabama_birmingham",
    "UCF": "ucf",
    "UCLA": "ucla",
    "UCSD": "uc_san_diego",
    "UK": "kentucky",
    "UL": "louisiana",
    "ULM": "louisiana_monroe",
    "UMASS": "massachusetts",
    "UMBC": "maryland_baltimore_county",
    "UNC": "north_carolina",
    "UNCW": "unc_wilmington",
    "UND": "north_dakota",
    "UNM": "new_mexico",
    "USC": "southern_california",
    "USF": "south_florida",
    "USU": "utah_state",
    "UT": "utah",
    "UVA": "virginia",
    "UVM": "vermont",
    # V
    "VAN": "vanderbilt",
    "VCU": "virginia_commonwealth",
    "VILL": "villanova",
    # W
    "WAKE": "wake_forest",
    "WICH": "wichita_state",
    "WIS": "wisconsin",
    "WSU": "washington_state",
    "WVU": "west_virginia",
    # X-Z
    "XAV": "xavier",
    "YALE": "yale",
}


class PoolHistoryResolutionError(Exception):
    """Raised when pool-history loading has no resolvable teams for the year."""


def resolve_abbrev(
    abbrev: str,
    seeds: Mapping[str, int],
) -> Optional[str]:
    """Map an ESPN abbreviation to a canonical team_id in ``seeds``.

    Resolution order (no silent fallback — returns None on failure):
        1. Static ABBREV_TO_TEAM_ID lookup whose target is in ``seeds``.
        2. ``normalize_team_id(abbrev)`` if that result is in ``seeds``.

    Args:
        abbrev: Raw abbreviation as it appears in pool_hist_results.json.
        seeds: team_id -> seed mapping for the target tournament year.

    Returns:
        Canonical team_id from ``seeds``, or None if nothing matches.
    """
    if abbrev in ABBREV_TO_TEAM_ID:
        candidate = ABBREV_TO_TEAM_ID[abbrev]
        if candidate in seeds:
            return candidate
    # Fallback: normalize and check against seeds directly.
    normalized = normalize_team_id(abbrev)
    if normalized in seeds:
        return normalized
    return None


def load_pool_brackets(
    path: str | Path,
    year: int,
) -> Tuple[List[dict], int]:
    """Load bracket entries for ``year`` from a pool_hist_results.json file.

    Returns:
        (brackets, group_size) — ``brackets`` is the list of per-entry
        dicts; ``group_size`` is the reported pool size (may exceed
        len(brackets) if some entries were missing).

    Raises:
        FileNotFoundError: File does not exist.
        KeyError: File has no entry for ``year``.
        ValueError: File format is not recognized.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pool history file not found: {path}")
    with open(path) as f:
        data = json.load(f)
    years = data.get("years")
    if not isinstance(years, dict):
        raise ValueError(f"Pool history file {path} has no 'years' mapping; got keys {list(data.keys())[:5]}")
    ystr = str(year)
    if ystr not in years:
        raise KeyError(f"Pool history file {path} has no entry for year {year}; available: {sorted(years.keys())}")
    entry = years[ystr]
    brackets = entry.get("brackets") or []
    group_size = int(entry.get("groupSize") or len(brackets))
    return list(brackets), group_size


def build_pool_pick_distribution(
    brackets: Iterable[dict],
    seeds: Mapping[str, int],
    floor_strategy: str = "laplace",
    laplace_alpha: float = 0.5,
    strict: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Empirical pick distribution from actual pool bracket entries.

    For each team in ``seeds``, returns the fraction of pool brackets
    that advanced that team to each round.  Smooths zero-count rounds
    with Laplace pseudocounts so the optimizer's log-EV calculations
    don't blow up on teams nobody picked.

    Args:
        brackets: Pool-history bracket entries (one dict per entry).
        seeds: team_id -> seed for the target year.  Teams listed here
            receive a full 6-round distribution; teams appearing in the
            pool data but not in ``seeds`` are dropped with a warning.
        floor_strategy: "laplace" (default) or "none".  With "laplace",
            each round prob is ``(count + alpha) / (n + alpha)`` so
            unseen teams get ``alpha / (n + alpha)`` instead of 0.
        laplace_alpha: Pseudocount.  Default 0.5 (Jeffreys prior) —
            for n=30 brackets, this puts an unseen team at ~1.6%.
        strict: If True, raise instead of logging when an abbrev can't
            be resolved.

    Returns:
        team_id -> {round_name: probability in [0, 1]}.  CHAMP row is
        normalized so champion probabilities sum to 1.0 across teams.

    Raises:
        PoolHistoryResolutionError: If ``strict`` and no brackets could
            be parsed, or any abbrev is unresolvable.
    """
    brackets = list(brackets)
    n = len(brackets)
    if n == 0:
        raise PoolHistoryResolutionError("No pool brackets provided; cannot build pick distribution.")

    counts: Dict[str, Dict[str, int]] = {tid: {r: 0 for r in ROUNDS} for tid in seeds}
    unresolved: Dict[str, int] = defaultdict(int)

    for bkt in brackets:
        for pool_key, round_name in _POOL_ROUND_MAP.items():
            picks = bkt.get(pool_key)
            if picks is None:
                continue
            if isinstance(picks, str):
                picks = [picks]
            for raw in picks:
                if not isinstance(raw, str):
                    continue
                tid = resolve_abbrev(raw, seeds)
                if tid is None:
                    unresolved[raw] += 1
                    continue
                counts[tid][round_name] += 1

    if unresolved:
        msg = "Pool history: %d abbreviation(s) could not be resolved and were DROPPED (no silent fallback): %s" % (
            len(unresolved),
            ", ".join(f"{a}×{c}" for a, c in sorted(unresolved.items(), key=lambda x: -x[1])[:15]),
        )
        if strict:
            raise PoolHistoryResolutionError(msg)
        logger.warning(msg)

    if floor_strategy not in {"laplace", "none"}:
        raise ValueError(f"Unknown floor_strategy {floor_strategy!r}; expected 'laplace' or 'none'.")

    alpha = float(laplace_alpha) if floor_strategy == "laplace" else 0.0
    denom = float(n) + alpha

    result: Dict[str, Dict[str, float]] = {}
    for tid, round_counts in counts.items():
        row = {r: (round_counts[r] + alpha) / denom for r in ROUNDS}
        result[tid] = row

    # Normalize CHAMP so champion probabilities sum to ~1.0 across teams.
    # Without normalization, Laplace-smoothed champs sum to > 1.0 (every
    # team gets the alpha floor even though only one champion exists per
    # bracket).
    champ_total = sum(result[t]["CHAMP"] for t in result)
    if champ_total > 0:
        for tid in result:
            result[tid]["CHAMP"] /= champ_total

    return result


def pool_entry_to_bracket_vector(
    entry: Mapping[str, object],
    first_round: Sequence[str],
    seeds: Mapping[str, int],
    strict: bool = False,
) -> np.ndarray:
    """Convert a pool_hist_results.json entry into a (63,) bool bracket vector.

    The output is positionally aligned with ``first_round`` (the 64-team
    list, [team1, team2, team3, team4, ...] in pair order) so the result
    is directly consumable by ``score_brackets_against_outcome``.

    Walks the bracket round-by-round.  At each round, for every current
    matchup ``(t1, t2)`` checks which team appears in the entry's pick
    set for that round.  Resolution rules:

    * Exactly one of ``{t1, t2}`` is in the round's picks → that team wins;
      ``vector[i] = (winner == t1)``.
    * Neither is in the round's picks (e.g., the bracket has an incomplete
      ``r64`` of 28-29 picks; observed for 2 known low-rank entries) →
      fall back to "higher-seeded team wins" (``team1`` per the
      ``first_round`` ordering convention) and log a debug record.  This
      mirrors ``build_actual_outcome``'s default in
      ``scripts/o25_g3_diversity_diagnostic.py:267``.
    * Both ``t1`` and ``t2`` are in the round's picks → bracket is
      internally inconsistent (a team can only advance once per round);
      raises ``PoolHistoryResolutionError``.

    Args:
        entry: Single bracket dict from ``pool_hist_results.json`` with
            ``r64``/``r32``/``s16``/``e8``/``f4``/``champ`` pick fields.
        first_round: Length-64 list of team_ids in matchup order; produced
            by ``scripts/o25_g3_diversity_diagnostic.build_first_round_matchups``.
            Pair ``(first_round[2i], first_round[2i+1])`` is R64 game ``i``.
        seeds: ``team_id -> seed`` mapping for the target year.  Used by
            ``resolve_abbrev`` and to break ties when picks are missing.
        strict: If True, raise on any unresolved abbrev.  If False
            (default), unresolved abbrevs are dropped from the round's
            pick set and logged at WARNING; the fallback (top seed wins)
            then applies.

    Returns:
        Boolean ``np.ndarray`` of shape ``(63,)``.  Index ordering:
        ``[0..31]`` = R64, ``[32..47]`` = R32, ``[48..55]`` = S16,
        ``[56..59]`` = E8, ``[60..61]`` = F4, ``[62]`` = CHAMP.

    Raises:
        PoolHistoryResolutionError: Bracket picks both teams in a single
            matchup, or ``strict=True`` and an abbrev is unresolvable.
        ValueError: ``first_round`` is not length 64.
    """
    if len(first_round) != 64:
        raise ValueError(f"first_round must be length 64, got {len(first_round)}")

    pick_sets: Dict[str, set[str]] = {}
    for pool_key, round_name in _POOL_ROUND_MAP.items():
        raw_picks = entry.get(pool_key)
        if raw_picks is None:
            pick_sets[round_name] = set()
            continue
        if isinstance(raw_picks, str):
            raw_picks = [raw_picks]
        resolved: set[str] = set()
        unresolved: List[str] = []
        for raw in raw_picks:
            if not isinstance(raw, str):
                continue
            tid = resolve_abbrev(raw, seeds)
            if tid is None:
                unresolved.append(raw)
                continue
            resolved.add(tid)
        if unresolved:
            msg = "pool_entry_to_bracket_vector(rank=%s): %d unresolved abbrev(s) in %s: %s" % (
                entry.get("rank"),
                len(unresolved),
                pool_key,
                ", ".join(unresolved),
            )
            if strict:
                raise PoolHistoryResolutionError(msg)
            logger.warning(msg)
        pick_sets[round_name] = resolved

    vector = np.zeros(63, dtype=bool)
    current = list(first_round)
    gi = 0
    for round_name in ROUNDS:
        picks = pick_sets[round_name]
        nxt: List[str] = []
        for g in range(0, len(current), 2):
            t1, t2 = current[g], current[g + 1]
            t1_in = t1 in picks
            t2_in = t2 in picks
            if t1_in and t2_in:
                raise PoolHistoryResolutionError(
                    f"pool_entry_to_bracket_vector(rank={entry.get('rank')}): "
                    f"both {t1} and {t2} are picked in {round_name}"
                )
            if t1_in:
                winner = t1
                vector[gi] = True
            elif t2_in:
                winner = t2
                vector[gi] = False
            else:
                # No pick for this matchup: default to higher seed (team1).
                # Mirrors build_actual_outcome's default.
                winner = t1
                vector[gi] = True
                logger.debug(
                    "pool_entry_to_bracket_vector(rank=%s): no %s pick for (%s, %s); defaulting to team1=%s",
                    entry.get("rank"),
                    round_name,
                    t1,
                    t2,
                    t1,
                )
            nxt.append(winner)
            gi += 1
        current = nxt

    return vector


def load_pool_bracket_vectors(
    path: str | Path,
    year: int,
    first_round: Sequence[str],
    seeds: Mapping[str, int],
    strict: bool = False,
) -> Tuple[np.ndarray, int]:
    """Load ``year``'s pool entries and return a ``(n_opp, 63)`` bool matrix.

    Convenience wrapper: combines ``load_pool_brackets`` +
    ``pool_entry_to_bracket_vector`` for callers that want a ready-to-score
    opponent matrix.

    Returns:
        ``(opponent_matrix, group_size)`` — ``opponent_matrix.shape ==
        (len(brackets), 63)``; ``group_size`` is the reported pool size
        (may exceed ``len(brackets)`` per the gap convention in
        ``tests/test_pool_hist_ev_validation.py:_GROUP_SIZE_GAP``).
    """
    brackets, group_size = load_pool_brackets(path, year)
    if not brackets:
        raise PoolHistoryResolutionError(f"No pool brackets for year {year} in {path}; cannot build opponent matrix.")
    matrix = np.stack([pool_entry_to_bracket_vector(b, first_round, seeds, strict=strict) for b in brackets])
    return matrix, group_size


def load_pool_history_picks(
    path: str | Path,
    year: int,
    seeds: Mapping[str, int],
    floor_strategy: str = "laplace",
    laplace_alpha: float = 0.5,
    strict: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Convenience wrapper: load brackets for ``year`` and build the distribution."""
    brackets, group_size = load_pool_brackets(path, year)
    logger.info(
        "Pool history %d: loaded %d brackets (groupSize=%d) from %s",
        year,
        len(brackets),
        group_size,
        path,
    )
    return build_pool_pick_distribution(
        brackets,
        seeds,
        floor_strategy=floor_strategy,
        laplace_alpha=laplace_alpha,
        strict=strict,
    )


# ---------------------------------------------------------------------------
# Cross-year pool behavioral model
# ---------------------------------------------------------------------------

# Seed matchups in R64 (higher seed listed first = "chalk" pick)
_R64_MATCHUPS = [(1, 16), (2, 15), (3, 14), (4, 13), (5, 12), (6, 11), (7, 10), (8, 9)]


def build_pool_behavioral_model(
    path: str | Path,
    seeds: Mapping[str, int],
    exclude_year: Optional[int] = None,
) -> Tuple[Dict[str, Dict[str, float]], float]:
    """Build a seed-based pick distribution from cross-year pool behavior.

    For years without direct pool bracket data, this learns the pool's
    behavioral patterns (per-seed round advancement rates, bracket
    correlation) from all available pool history years and maps them onto the
    target year's teams via their seeds.

    The model directly counts how many brackets pick teams of each seed to
    advance to each round, then divides by (n_brackets × 4) to get
    per-team advancement rates.  This produces the correct cumulative
    quantity — P(team picked to reach round R) — rather than per-game
    conditional win rates which don't compound properly.

    Args:
        path: Path to pool_hist_results.json.
        seeds: team_id -> seed for the TARGET year.
        exclude_year: Year to exclude (LOOY).  If None, uses all available.

    Returns:
        (pick_dist, chalk_noise_std):
            pick_dist: team_id -> {round: probability} for each team in seeds.
            chalk_noise_std: Estimated bracket-level correlation parameter.
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    years_data = data.get("years", {})

    # Count per-seed, per-round advancement picks across all brackets.
    # seed_round_counts[seed][round] = total teams of that seed picked
    # to advance to that round, summed across all brackets and years.
    seed_round_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    n_brackets = 0
    # Track per-bracket chalk fraction for correlation estimation
    bracket_chalk_fractions: List[float] = []

    for year_str, year_entry in years_data.items():
        yr = int(year_str)
        if exclude_year is not None and yr == exclude_year:
            continue
        brackets = year_entry.get("brackets", [])
        if not brackets:
            continue

        # Load seeds for this historical year to map abbreviations to seeds
        try:
            yr_seeds, _yr_regions = _load_year_seeds(yr)
        except (FileNotFoundError, KeyError):
            continue
        if not yr_seeds:
            continue

        for bkt in brackets:
            chalk_picks = 0
            total_picks = 0

            for round_name in ROUNDS:
                pool_key = {v: k for k, v in _POOL_ROUND_MAP.items()}[round_name]
                raw_picks = bkt.get(pool_key)
                if raw_picks is None:
                    continue
                if isinstance(raw_picks, str):
                    raw_picks = [raw_picks]

                for raw in raw_picks:
                    if not isinstance(raw, str):
                        continue
                    tid = resolve_abbrev(raw, yr_seeds)
                    if tid is None:
                        continue
                    team_seed = yr_seeds.get(tid)
                    if team_seed is None:
                        continue
                    seed_round_counts[team_seed][round_name] += 1
                    total_picks += 1
                    if team_seed <= 4:
                        chalk_picks += 1

            n_brackets += 1
            if total_picks > 0:
                bracket_chalk_fractions.append(chalk_picks / total_picks)

    # Build pick_dist for the target year's teams.
    # P(specific team of seed S advances to round R) =
    #   seed_round_counts[S][R] / (n_brackets * 4)
    # where 4 = number of teams of each seed (one per region).
    #
    # Use Bayesian smoothing with historical NCAA advancement rates as
    # prior (from _SEED_ADVANCEMENT_RATES). PRIOR_WEIGHT acts as a
    # pseudocount in equivalent-brackets.
    PRIOR_WEIGHT = 5  # pseudocount (in equivalent-bracket units)
    TEAMS_PER_SEED = 4
    pick_dist: Dict[str, Dict[str, float]] = {}

    for tid, seed in seeds.items():
        row: Dict[str, float] = {}
        prior_rates = _SEED_ADVANCEMENT_RATES.get(seed, _SEED_ADVANCEMENT_RATES[8])
        for round_name in ROUNDS:
            observed = seed_round_counts.get(seed, {}).get(round_name, 0)
            # observed is total picks across all brackets (NOT per-team).
            # Convert to per-team rate: observed / (n_brackets * TEAMS_PER_SEED)
            if n_brackets > 0:
                empirical = observed / (n_brackets * TEAMS_PER_SEED)
            else:
                empirical = 0.0
            prior_val = prior_rates.get(round_name, 0.01)
            # Bayesian blend in per-team-rate space
            n_eff = n_brackets  # effective observation count
            row[round_name] = (empirical * n_eff + prior_val * PRIOR_WEIGHT) / (n_eff + PRIOR_WEIGHT)
        pick_dist[tid] = row

    # Normalize CHAMP probabilities to sum to ~1.0
    champ_total = sum(pick_dist[tid].get("CHAMP", 0.0) for tid in pick_dist)
    if champ_total > 0:
        for tid in pick_dist:
            pick_dist[tid]["CHAMP"] = pick_dist[tid].get("CHAMP", 0.0) / champ_total

    # Estimate chalk_noise_std from bracket-level variance
    if len(bracket_chalk_fractions) >= 5:
        chalk_noise_std = float(np.std(bracket_chalk_fractions))
    else:
        chalk_noise_std = 0.15  # conservative default

    return pick_dist, chalk_noise_std


def _load_year_seeds(year: int) -> Tuple[Dict[str, int], Dict[str, str]]:
    """Load seeds and regions for a historical year (utility for behavioral model)."""
    from scripts.mc_pool_backtest import load_seeds_and_regions

    return load_seeds_and_regions(year)


# ---------------------------------------------------------------------------
# Blended pool-calibrated opponent model (Phase 1)
# ---------------------------------------------------------------------------

# Seed-based advancement rates used as fallback when ESPN picks are
# unavailable. Duplicated from mc_pool_backtest.SEED_PICK_RATES to avoid
# a circular import.  These are ESPN national pick rates aggregated by
# seed (2012-2024 average).
_SEED_ADVANCEMENT_RATES: Dict[int, Dict[str, float]] = {
    1: {"R64": 0.99, "R32": 0.85, "S16": 0.60, "E8": 0.37, "F4": 0.20, "CHAMP": 0.10},
    2: {"R64": 0.94, "R32": 0.67, "S16": 0.38, "E8": 0.20, "F4": 0.09, "CHAMP": 0.04},
    3: {"R64": 0.85, "R32": 0.52, "S16": 0.26, "E8": 0.12, "F4": 0.05, "CHAMP": 0.02},
    4: {"R64": 0.79, "R32": 0.43, "S16": 0.20, "E8": 0.09, "F4": 0.04, "CHAMP": 0.015},
    5: {"R64": 0.64, "R32": 0.30, "S16": 0.13, "E8": 0.05, "F4": 0.02, "CHAMP": 0.007},
    6: {"R64": 0.63, "R32": 0.28, "S16": 0.12, "E8": 0.05, "F4": 0.02, "CHAMP": 0.006},
    7: {"R64": 0.60, "R32": 0.23, "S16": 0.09, "E8": 0.03, "F4": 0.01, "CHAMP": 0.004},
    8: {"R64": 0.50, "R32": 0.17, "S16": 0.06, "E8": 0.02, "F4": 0.007, "CHAMP": 0.003},
    9: {"R64": 0.50, "R32": 0.17, "S16": 0.06, "E8": 0.02, "F4": 0.007, "CHAMP": 0.002},
    10: {"R64": 0.40, "R32": 0.13, "S16": 0.04, "E8": 0.015, "F4": 0.005, "CHAMP": 0.001},
    11: {"R64": 0.37, "R32": 0.11, "S16": 0.04, "E8": 0.01, "F4": 0.004, "CHAMP": 0.001},
    12: {"R64": 0.36, "R32": 0.10, "S16": 0.03, "E8": 0.008, "F4": 0.003, "CHAMP": 0.0005},
    13: {"R64": 0.21, "R32": 0.04, "S16": 0.01, "E8": 0.002, "F4": 0.0005, "CHAMP": 0.0001},
    14: {"R64": 0.15, "R32": 0.03, "S16": 0.005, "E8": 0.001, "F4": 0.0003, "CHAMP": 0.0001},
    15: {"R64": 0.06, "R32": 0.01, "S16": 0.002, "E8": 0.0005, "F4": 0.0001, "CHAMP": 0.00005},
    16: {"R64": 0.01, "R32": 0.002, "S16": 0.0003, "E8": 0.0001, "F4": 0.00002, "CHAMP": 0.00001},
}


def _seed_pick_distribution(
    seeds: Mapping[str, int],
) -> Dict[str, Dict[str, float]]:
    """Build opponent pick distribution from seed advancement rates."""
    return {tid: dict(_SEED_ADVANCEMENT_RATES.get(seed, _SEED_ADVANCEMENT_RATES[8])) for tid, seed in seeds.items()}


def build_blended_pool_opponent_model(
    pool_history_path: str | Path,
    seeds: Mapping[str, int],
    year: int,
    espn_pick_dist: Optional[Dict[str, Dict[str, float]]] = None,
    pool_weight: float = 0.7,
) -> Tuple[Dict[str, Dict[str, float]], float]:
    """Build a pool-calibrated opponent pick distribution blended with ESPN.

    For every backtest year Y, this produces an opponent model that combines:
      - **Pool behavioral component** (70% default): seed-level pick patterns
        learned from cross-year pool brackets (LOOY, excluding year Y).
      - **ESPN component** (30% default): year-specific national pick rates.
        Falls back to seed-based rates if ESPN data is unavailable.

    This mirrors the production scenario for 2027 where we have pool
    behavioral data from 2023-2026 and ESPN picks for 2027.

    Args:
        pool_history_path: Path to ``pool_hist_results.json``.
        seeds: Team-ID → seed mapping for the target year.
        year: Target year (passed as ``exclude_year`` to behavioral model).
        espn_pick_dist: Pre-loaded ESPN national pick distribution for the
            target year.  ``None`` triggers fallback to seed-based rates.
        pool_weight: Blend weight for pool behavioral component (0.0 = pure
            ESPN, 1.0 = pure pool behavioral).  Default 0.7.

    Returns:
        ``(pick_dist, chalk_noise_std)`` where ``pick_dist`` is
        ``Dict[team_id, Dict[round_name, probability]]`` and
        ``chalk_noise_std`` is the bracket-level chalk correlation
        estimated from the pool behavioral model.
    """
    # 1. Pool behavioral component (LOOY safe)
    behavioral_dist, chalk_noise_std = build_pool_behavioral_model(
        pool_history_path,
        seeds,
        exclude_year=year,
    )

    # 2. ESPN component (or seed-based fallback)
    if espn_pick_dist is not None:
        espn_dist = espn_pick_dist
    else:
        espn_dist = _seed_pick_distribution(seeds)

    # 3. Blend per team, per round
    blended: Dict[str, Dict[str, float]] = {}
    for tid in seeds:
        beh_row = behavioral_dist.get(tid, {})
        espn_row = espn_dist.get(tid, {})
        row: Dict[str, float] = {}
        for rnd in ROUNDS:
            beh_val = beh_row.get(rnd, 0.0)
            espn_val = espn_row.get(rnd, 0.0)
            row[rnd] = pool_weight * beh_val + (1.0 - pool_weight) * espn_val
        blended[tid] = row

    # 4. Normalize CHAMP probabilities to sum to ~1.0
    champ_total = sum(blended[tid].get("CHAMP", 0.0) for tid in blended)
    if champ_total > 0:
        for tid in blended:
            blended[tid]["CHAMP"] = blended[tid].get("CHAMP", 0.0) / champ_total

    return blended, chalk_noise_std


# ---------------------------------------------------------------------------
# Real bracket opponent matrix (LOYO-safe cross-year resampler)
# ---------------------------------------------------------------------------


def _bracket_to_seed_walk(
    entry: Mapping[str, object],
    yr_seeds: Mapping[str, int],
) -> Optional[Dict[str, set]]:
    """Convert a pool bracket entry to a {round_name: set_of_advancing_seeds} map.

    Returns None if the entry cannot be parsed (e.g., no picks at all).
    Seeds with multiple teams per region are tracked by (seed, region_index)
    but for cross-year transfer we only care about the seed number, so we
    collect seeds of advancing teams per round.  Duplicate seeds (two teams
    of the same seed in different regions) are handled by treating each as
    an independent pick.

    The result is a list of seed values that were picked to advance to each
    round.  This is the information we can transfer cross-year: "picked a
    12-seed to reach the S16" → in the test year, pick whichever 12-seed is
    in the matching slot.
    """
    round_seeds: Dict[str, List[int]] = {r: [] for r in ROUNDS}
    has_any = False

    for pool_key, round_name in _POOL_ROUND_MAP.items():
        raw = entry.get(pool_key)
        if raw is None:
            continue
        if isinstance(raw, str):
            raw = [raw]
        for abbrev in raw:
            if not isinstance(abbrev, str):
                continue
            tid = resolve_abbrev(abbrev, yr_seeds)
            if tid is None:
                continue
            seed = yr_seeds.get(tid)
            if seed is None:
                continue
            round_seeds[round_name].append(seed)
            has_any = True

    if not has_any:
        return None
    return {r: set(v) for r, v in round_seeds.items()}


def _seed_walk_to_bracket_vector(
    seed_walk: Dict[str, set],
    first_round: Sequence[str],
    seeds: Mapping[str, int],
) -> np.ndarray:
    """Convert a seed-walk dict to a (63,) bool vector for the test year.

    For each game slot in ``first_round``, determines the winner by checking
    which team's seed appears in the seed_walk for that round.  When both or
    neither seed appears (ambiguous due to multiple teams sharing the same
    seed), defaults to the lower seed number (chalk).

    Args:
        seed_walk: {round_name: set_of_advancing_seed_numbers} from a
            historical bracket.  Seed values are 1-16.
        first_round: 64 team_ids in matchup order for the test year.
        seeds: team_id -> seed for the test year.

    Returns:
        Boolean array (63,) compatible with score_brackets_team_identity.
    """
    if len(first_round) != 64:
        raise ValueError(f"first_round must be length 64, got {len(first_round)}")

    vector = np.zeros(63, dtype=bool)
    current = list(first_round)
    gi = 0

    for round_name in ROUNDS:
        round_seeds_set = seed_walk.get(round_name, set())
        nxt: List[str] = []
        for g in range(0, len(current), 2):
            t1, t2 = current[g], current[g + 1]
            s1 = seeds.get(t1, 8)
            s2 = seeds.get(t2, 8)
            t1_picked = s1 in round_seeds_set
            t2_picked = s2 in round_seeds_set
            # Prefer explicit pick; when ambiguous default to lower seed (chalk)
            if t1_picked and not t2_picked:
                winner = t1
                vector[gi] = True
            elif t2_picked and not t1_picked:
                winner = t2
                vector[gi] = False
            elif s1 <= s2:
                # chalk default (also covers both/neither)
                winner = t1
                vector[gi] = True
            else:
                winner = t2
                vector[gi] = False
            nxt.append(winner)
            gi += 1
        current = nxt

    return vector


def build_pool_history_opponent_matrix(
    path: str | Path,
    test_year: int,
    first_round: Sequence[str],
    seeds: Mapping[str, int],
    n_opponents: int,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """Build an opponent matrix by resampling real pool brackets (LOYO-safe).

    Loads all available pool brackets from years ≠ ``test_year``, converts
    each to the test year's coordinate space via seed-walk mapping, and
    resamples (with replacement) to produce ``n_opponents`` opponent brackets.

    The seed-walk conversion transfers the pool's behavioral tendencies
    (chalk bias, upset affinity per seed matchup) across years without
    requiring team identity to be shared.  For example, a 2023 bracket that
    picked the 12-seed to reach the S16 maps to "pick whichever 12-seed
    appears in the equivalent slot of the test year's bracket".

    LOYO contract: this function NEVER includes brackets from ``test_year``
    in the returned matrix.  The walk-forward firewall is enforced by the
    ``exclude_year`` parameter passed to ``load_pool_brackets``.

    Args:
        path: Path to pool_hist_results.json.
        test_year: Year being backtested.  Excluded from source data.
        first_round: 64 team_ids in matchup order for the test year.
        seeds: team_id -> seed for the test year.
        n_opponents: Number of opponent brackets to return.
        rng: NumPy random generator for resampling.

    Returns:
        Boolean array (n_opponents, 63) ready for score_brackets_team_identity,
        or None if no pool history data is available for years ≠ test_year.
    """
    path = Path(path)
    if not path.exists():
        return None

    with open(path) as f:
        data = json.load(f)
    years_data = data.get("years", {})

    # Collect seed-walk vectors from all years except test_year.
    source_vectors: List[np.ndarray] = []
    for year_str, year_entry in years_data.items():
        yr = int(year_str)
        if yr == test_year:
            continue  # LOYO firewall

        brackets = year_entry.get("brackets", [])
        if not brackets:
            continue

        # Load seeds for this historical year to map abbreviations to seeds.
        try:
            yr_seeds, _yr_regions = _load_year_seeds(yr)
        except Exception:
            logger.debug("build_pool_history_opponent_matrix: cannot load seeds for %d, skipping", yr)
            continue
        if not yr_seeds:
            continue

        for entry in brackets:
            seed_walk = _bracket_to_seed_walk(entry, yr_seeds)
            if seed_walk is None:
                continue
            vec = _seed_walk_to_bracket_vector(seed_walk, first_round, seeds)
            source_vectors.append(vec)

    if not source_vectors:
        return None

    # Stack into matrix and resample with replacement.
    source_matrix = np.stack(source_vectors)  # (n_source, 63)
    n_source = len(source_matrix)
    logger.info(
        "build_pool_history_opponent_matrix: %d source brackets from pool years excluding %d; resampling %d opponents",
        n_source,
        test_year,
        n_opponents,
    )
    indices = rng.integers(0, n_source, size=n_opponents)
    return source_matrix[indices]
