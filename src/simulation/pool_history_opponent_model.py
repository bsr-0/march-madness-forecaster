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
    "COFC": "charleston",
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
