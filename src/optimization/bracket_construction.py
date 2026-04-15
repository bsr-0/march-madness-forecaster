"""Bracket construction modes for pool optimization.

Four algorithms that differ on WHICH ROUND serves as the anchor — the round
that gets decided first and constrains everything else. All four modes share
the same scoring function, round probabilities, seed/region inputs, and
output type (a complete 63-game bracket as a picks dict). The only thing
that varies is construction order.

Modes:

  champ_first   — pick the globally optimal champion first, lock their
                  R64→CHAMP path, fill the rest of the bracket by forward
                  greedy _ev_score. Anchor round: CHAMP.

  f4_first      — pick the 4 regional champions first (one per region, each
                  by argmax _ev_score for F4), lock each one's R64→E8 path
                  within their region, fill the rest by forward greedy,
                  then pick F4 semifinal winners and champion by argmax
                  forward. Anchor round: F4 (regional champion).

  e8_first      — pick 8 S16 winners first (two per region, one per
                  quadrant, each by argmax _ev_score for S16), lock each
                  one's R64→S16 path within their quadrant, fill R64-S16
                  non-path games by forward greedy, then pick E8/F4/CHAMP
                  forward greedy from the constrained set. Anchor round:
                  S16 (quadrant winner).

  forward_greedy — no anchor. At every game from R64 to CHAMP, pick the
                  winner by argmax _ev_score for the current round. This
                  is the status quo baseline and matches the existing
                  ``_generate_full_bracket`` behavior in leverage.py. If
                  a forced_champion is provided, this mode falls back to
                  champ_first because forward_greedy has no natural way
                  to inject a forced champion.

All four modes are parameterized by a scoring function ``_ev_score`` (same
formula across modes, isolating the construction-order signal from the
scoring-function signal). The `risk_level` parameter controls how much
the scorer weights differentiation vs raw probability; low risk = chalk,
high risk = contrarian. This is identical to how leverage.py's ParetoOptimizer
currently works, so a risk sweep across [0, 1] produces a Pareto frontier
per construction mode.

Outputs mirror the existing ``_generate_full_bracket`` / ``_generate_summary_bracket``
tuple: ``(picks, champion, final_four, expected_points, variance)`` where
picks is a dict mapping 63 game keys to winner team_ids. Expected points
and variance are computed under the standard independent-pick approximation:

    E[pts] = sum over 63 games of P(picked team wins their round game) * round_pts
    Var[pts] = sum over 63 games of P * (1-P) * round_pts^2

This matches the formula used by ``_summary_bracket_full_score`` and avoids
the survival-probability double-counting bug in the legacy ``_path_ev_var``
helper.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ROUND_NAMES: Tuple[str, ...] = ("R64", "R32", "S16", "E8", "F4", "CHAMP")
_ROUND_INDEX: Dict[str, int] = {r: i for i, r in enumerate(_ROUND_NAMES)}
_REGION_ORDER: Tuple[str, ...] = ("East", "West", "South", "Midwest")
_SEED_MATCHUP_ORDER: Tuple[Tuple[int, int], ...] = (
    (1, 16),
    (8, 9),
    (5, 12),
    (4, 13),
    (6, 11),
    (3, 14),
    (7, 10),
    (2, 15),
)
_REGION_ALIASES: Dict[str, str] = {"Southeast": "South", "Southwest": "Midwest"}

# ESPN standard scoring, used as default when caller doesn't provide one.
_DEFAULT_SCORING: Dict[str, int] = {
    "R64": 10,
    "R32": 20,
    "S16": 40,
    "E8": 80,
    "F4": 160,
    "CHAMP": 320,
}

# Supported construction modes.
CONSTRUCTION_MODES: Tuple[str, ...] = (
    "champ_first",
    "f4_first",
    "e8_first",
    "forward_greedy",
    "champ_first_chalkfade",
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_seed_map(
    seeds: Dict[str, int],
    regions: Dict[str, str],
) -> Dict[str, Dict[int, str]]:
    """Normalize a (possibly 68-team) seeds/regions input to a 16-per-region map.

    Uses dict-overwrite for duplicate seeds (First Four play-ins): whichever
    team is iterated last for a given (region, seed) pair wins the slot and
    the other is silently dropped. This matches the existing dict-overwrite
    behavior in ``leverage.py::_build_seed_map`` and ``mc_pool_backtest.py``,
    so the construction modes produce brackets consistent with what the
    backtest script's stochastic samplers operate on.

    Returns: ``{region: {seed: team_id}}`` with seeds 1-16 represented in
    each of the 4 regions. Raises ValueError if any region is missing a seed
    1-16 after normalization.
    """
    by_region: Dict[str, Dict[int, str]] = {r: {} for r in _REGION_ORDER}
    for tid in seeds:
        raw_region = regions.get(tid, "")
        region = _REGION_ALIASES.get(raw_region, raw_region)
        if region not in by_region:
            continue
        seed = seeds[tid]
        if seed <= 0:
            continue
        by_region[region][seed] = tid

    # Verify every region has all seeds 1-16
    for region in _REGION_ORDER:
        for seed in range(1, 17):
            if seed not in by_region[region]:
                raise ValueError(
                    f"region {region} missing seed {seed} after normalization; cannot build full 64-team bracket"
                )
    return by_region


def _make_ev_scorer(
    round_probs: Dict[str, Dict[str, float]],
    public_picks: Dict[str, Dict[str, float]],
    risk_level: float,
    pool_size: int,
    scoring_system: Dict[str, int],
) -> Callable[[str, str], float]:
    """Build a closure that scores (team_id, round_name) via the _ev_score formula.

    The formula mirrors ``leverage.py::_ev_score`` in EV mode:

        score = model_prob * pts * blended_diff

    where ``blended_diff = (1 - risk_level) * 1.0 + risk_level * diff`` and
    ``diff = uniqueness * pool_factor`` with ``uniqueness = 1 - public_prob``
    and ``pool_factor = 1 / log2(max(pool_size * public_prob, 1))`` for pools
    larger than 50 (else pool_factor = 1.0).

    Kept as a closure so each construction mode can build one scorer with
    fixed risk_level/pool_size/scoring_system and then pass it around without
    threading 5 parameters everywhere.
    """

    def scorer(team_id: str, round_name: str) -> float:
        model_prob = float(round_probs.get(team_id, {}).get(round_name, 0.0))
        public_prob = float(public_picks.get(team_id, {}).get(round_name, 0.0))
        pts = float(scoring_system.get(round_name, 10))

        uniqueness = max(0.0, 1.0 - public_prob)
        pool_factor = 1.0
        if pool_size > 50:
            expected_dups = pool_size * public_prob
            pool_factor = 1.0 / max(1.0, math.log2(max(expected_dups, 1.0)))
        diff = uniqueness * pool_factor

        blended_diff = (1.0 - risk_level) * 1.0 + risk_level * diff
        return model_prob * pts * blended_diff

    return scorer


def _make_chalkfade_champ_scorer(
    base_scorer: Callable[[str, str], float],
    seeds: Dict[str, int],
    chalk_bias_table: Dict[int, Dict[str, float]],
) -> Callable[[str, str], float]:
    """Wrap base scorer to fade over-picked champions at CHAMP selection.

    Divides the CHAMP ev_score by the empirical chalk-bias ratio for the
    team's seed. Seeds the public over-picks at CHAMP (high ratio) get a
    lower chalkfade score; under-picked seeds (low ratio) get relatively
    boosted. Non-CHAMP rounds pass through unchanged so path-fill games
    use the unmodified base scorer.
    """

    def scorer(team_id: str, round_name: str) -> float:
        score = base_scorer(team_id, round_name)
        if round_name == "CHAMP":
            seed = seeds.get(team_id, 8)
            ratio = chalk_bias_table.get(seed, {}).get("CHAMP", 1.0) or 1.0
            return score / max(ratio, 0.01)
        return score

    return scorer


def _decide_winner(
    t1: str,
    t2: str,
    round_name: str,
    locked_teams: Set[str],
    lock_through_round: Optional[str],
    scorer: Callable[[str, str], float],
    forced_champion: Optional[str] = None,
) -> str:
    """Pick the winner of a single game.

    Priority order (highest to lowest):

    1. ``forced_champion``: if one of the two teams is the forced champion,
       they win regardless of round or locked_teams state. This is what
       guarantees the forced champion actually makes it to the CHAMP game
       even in modes like f4_first and e8_first where the primary lock
       only extends through E8 or S16.

    2. ``lock_through_round`` + ``locked_teams``: if the current round is
       at or before ``lock_through_round`` and one of the two teams is in
       ``locked_teams``, that locked team wins. This handles the per-mode
       anchor locking (champ_first locks through CHAMP, f4_first locks
       through E8, e8_first locks through S16).

    3. Argmax ``_ev_score`` with lexical tie-break for determinism.

    The caller guarantees no game will have both teams in ``locked_teams``
    simultaneously (see the per-mode invariants in ``construct_bracket``).
    Similarly, the forced champion is never paired against another locked
    team in a way that would require both to win the same game.
    """
    # Priority 1: forced_champion overrides everything else
    if forced_champion is not None:
        if t1 == forced_champion:
            return t1
        if t2 == forced_champion:
            return t2

    # Priority 2: locked_teams within lock_through_round
    if lock_through_round is not None:
        if _ROUND_INDEX[round_name] <= _ROUND_INDEX[lock_through_round]:
            if t1 in locked_teams:
                return t1
            if t2 in locked_teams:
                return t2

    # Priority 3: argmax _ev_score
    score1 = scorer(t1, round_name)
    score2 = scorer(t2, round_name)
    if abs(score1 - score2) > 1e-9:
        return t1 if score1 > score2 else t2
    # Deterministic tie-break by team_id
    return t1 if t1 < t2 else t2


def _walk_bracket(
    by_region: Dict[str, Dict[int, str]],
    locked_teams: Set[str],
    lock_through_round: Optional[str],
    scorer: Callable[[str, str], float],
    forced_champion: Optional[str] = None,
) -> Tuple[Dict[str, str], str, List[str]]:
    """Walk the bracket from R64 to CHAMP, deciding each game forward.

    This is the round-by-round driver shared by all 4 construction modes.
    Mode-specific logic is encoded entirely in ``locked_teams``,
    ``lock_through_round``, and ``forced_champion`` — the walker itself
    is mode-agnostic.

    ``forced_champion`` is passed through to every ``_decide_winner`` call
    so it wins any game it appears in, regardless of round. This is what
    makes forced-champion injection work for f4_first and e8_first, which
    would otherwise only lock through E8 or S16 respectively and allow the
    forced champion to lose at the F4 or CHAMP game.

    Returns ``(picks, champion, final_four)`` where picks has 63 game keys
    matching the existing ``_generate_full_bracket`` format.
    """
    picks: Dict[str, str] = {}

    # R64: 8 games per region in standard seed-pairing order
    r64_winners: Dict[str, List[str]] = {}
    for region in _REGION_ORDER:
        winners_in_region: List[str] = []
        for high_seed, low_seed in _SEED_MATCHUP_ORDER:
            t1 = by_region[region][high_seed]
            t2 = by_region[region][low_seed]
            winner = _decide_winner(
                t1,
                t2,
                "R64",
                locked_teams,
                lock_through_round,
                scorer,
                forced_champion,
            )
            picks[f"R64_{region}_{high_seed}v{low_seed}"] = winner
            winners_in_region.append(winner)
        r64_winners[region] = winners_in_region

    # R32: 4 games per region, pairing adjacent R64 winners (0-1, 2-3, 4-5, 6-7)
    r32_winners: Dict[str, List[str]] = {}
    for region in _REGION_ORDER:
        winners_in_region = []
        region_r64 = r64_winners[region]
        for idx in range(0, 8, 2):
            t1 = region_r64[idx]
            t2 = region_r64[idx + 1]
            winner = _decide_winner(
                t1,
                t2,
                "R32",
                locked_teams,
                lock_through_round,
                scorer,
                forced_champion,
            )
            picks[f"R32_{region}_{idx // 2 + 1}"] = winner
            winners_in_region.append(winner)
        r32_winners[region] = winners_in_region

    # S16: 2 games per region, pairing adjacent R32 winners
    s16_winners: Dict[str, List[str]] = {}
    for region in _REGION_ORDER:
        winners_in_region = []
        region_r32 = r32_winners[region]
        for idx in range(0, 4, 2):
            t1 = region_r32[idx]
            t2 = region_r32[idx + 1]
            winner = _decide_winner(
                t1,
                t2,
                "S16",
                locked_teams,
                lock_through_round,
                scorer,
                forced_champion,
            )
            picks[f"S16_{region}_{idx // 2 + 1}"] = winner
            winners_in_region.append(winner)
        s16_winners[region] = winners_in_region

    # E8: 1 game per region — the 2 S16 winners
    e8_winners: Dict[str, str] = {}
    for region in _REGION_ORDER:
        t1, t2 = s16_winners[region]
        winner = _decide_winner(
            t1,
            t2,
            "E8",
            locked_teams,
            lock_through_round,
            scorer,
            forced_champion,
        )
        picks[f"E8_{region}"] = winner
        e8_winners[region] = winner

    # F4: East vs West, South vs Midwest
    semi_east_west = _decide_winner(
        e8_winners["East"],
        e8_winners["West"],
        "F4",
        locked_teams,
        lock_through_round,
        scorer,
        forced_champion,
    )
    semi_south_midwest = _decide_winner(
        e8_winners["South"],
        e8_winners["Midwest"],
        "F4",
        locked_teams,
        lock_through_round,
        scorer,
        forced_champion,
    )
    picks["F4_East_West"] = semi_east_west
    picks["F4_South_Midwest"] = semi_south_midwest

    # CHAMP: the two F4 winners
    champion = _decide_winner(
        semi_east_west,
        semi_south_midwest,
        "CHAMP",
        locked_teams,
        lock_through_round,
        scorer,
        forced_champion,
    )
    picks["CHAMP"] = champion

    # F4 list matches the existing convention: the 4 regional champions
    final_four = [e8_winners[r] for r in _REGION_ORDER]

    return picks, champion, final_four


def _compute_expected_points(
    picks: Dict[str, str],
    round_probs: Dict[str, Dict[str, float]],
    scoring_system: Dict[str, int],
) -> Tuple[float, float]:
    """Compute expected points and variance for a complete 63-game bracket.

    Formula (independent-pick approximation):

        E[pts] = sum over 63 games of P(picked team wins their round game) * round_pts
        Var[pts] = sum over 63 games of P * (1 - P) * round_pts^2

    This does NOT model path-dependent covariance between picks on the same
    bracket path (where winning round N requires having won round N-1). A
    full covariance model would be more accurate but requires computing
    pairwise covariance for every (game_i, game_j) pair along each path —
    see ``leverage.py::_generate_full_bracket`` which does this in its
    ``all_branch_picks`` loop. For our purposes the independent-pick
    approximation is adequate and matches the formula used by
    ``_summary_bracket_full_score``.

    The key correctness property: ``round_probs[team][round]`` is expected
    to represent P(team wins their round-R game), i.e., the joint probability
    of reaching AND winning that round. Summing across all teams for round R
    equals the number of winners in R (e.g., 32 for R64, 1 for CHAMP).
    """
    total_ev = 0.0
    total_var = 0.0
    for game_key, winner in picks.items():
        round_name = game_key.split("_", 1)[0]
        p = float(round_probs.get(winner, {}).get(round_name, 0.0))
        pts = float(scoring_system.get(round_name, 0))
        total_ev += p * pts
        total_var += p * (1.0 - p) * (pts**2)
    return total_ev, total_var


# ---------------------------------------------------------------------------
# Per-mode anchor selection
# ---------------------------------------------------------------------------


def _pick_champion(
    scorer: Callable[[str, str], float],
    all_teams: List[str],
    forced_champion: Optional[str],
) -> str:
    """Pick the champion by argmax _ev_score, or return forced_champion.

    ``all_teams`` must be the normalized 64-team list (post-dict-overwrite).
    """
    if forced_champion is not None and forced_champion in all_teams:
        return forced_champion
    if not all_teams:
        raise ValueError("cannot pick champion from empty team list")
    best_team = all_teams[0]
    best_score = scorer(best_team, "CHAMP")
    for team in all_teams[1:]:
        s = scorer(team, "CHAMP")
        if s > best_score or (s == best_score and team < best_team):
            best_team = team
            best_score = s
    return best_team


def _pick_f4_teams(
    scorer: Callable[[str, str], float],
    by_region: Dict[str, Dict[int, str]],
    forced_champion: Optional[str],
    max_one_seeds: int = 2,
) -> Set[str]:
    """Pick 4 F4 teams, one per region, by argmax _ev_score for F4.

    If ``forced_champion`` is provided, it replaces the F4 pick in its
    region regardless of the F4 score. The caller guarantees the forced
    champion exists in the seed map.

    ``max_one_seeds`` caps how many 1-seeds can appear in the F4.
    Historical base rate is ~1.5 one-seeds per F4; default cap of 2
    prevents the all-chalk consensus bracket that looks like everyone
    else's in the pool.
    """
    # Build reverse lookup: team_id -> seed
    team_seed: Dict[str, int] = {}
    for region in _REGION_ORDER:
        for seed, tid in by_region[region].items():
            team_seed[tid] = seed

    # Phase 1: pick best team per region (unconstrained)
    picks: Dict[str, Tuple[str, float]] = {}  # region -> (team, score)
    forced_region = None
    if forced_champion is not None:
        for region in _REGION_ORDER:
            if forced_champion in by_region[region].values():
                forced_region = region
                picks[region] = (forced_champion, scorer(forced_champion, "F4"))
                break

    for region in _REGION_ORDER:
        if region == forced_region:
            continue
        region_teams = list(by_region[region].values())
        best_team = region_teams[0]
        best_score = scorer(best_team, "F4")
        for team in region_teams[1:]:
            s = scorer(team, "F4")
            if s > best_score or (s == best_score and team < best_team):
                best_team = team
                best_score = s
        picks[region] = (best_team, best_score)

    # Phase 2: enforce one-seed cap
    one_seed_regions = [r for r, (t, _) in picks.items() if team_seed.get(t) == 1 and r != forced_region]
    while (
        len(one_seed_regions) + (1 if forced_region and team_seed.get(picks[forced_region][0]) == 1 else 0)
        > max_one_seeds
    ):
        # Replace the one-seed with the weakest EV contribution
        weakest_region = min(one_seed_regions, key=lambda r: picks[r][1])
        # Pick best non-one-seed in that region
        region_teams = list(by_region[weakest_region].values())
        best_alt = None
        best_alt_score = -1.0
        for team in region_teams:
            if team_seed.get(team) == 1:
                continue
            s = scorer(team, "F4")
            if s > best_alt_score or (best_alt is None):
                best_alt = team
                best_alt_score = s
        if best_alt is not None:
            picks[weakest_region] = (best_alt, best_alt_score)
        one_seed_regions.remove(weakest_region)

    return {t for t, _ in picks.values()}


def _pick_s16_teams(
    scorer: Callable[[str, str], float],
    by_region: Dict[str, Dict[int, str]],
    forced_champion: Optional[str],
) -> Set[str]:
    """Pick 8 S16 winners, two per region (one per quadrant), by argmax _ev_score for S16.

    A region's 16 seeds split into 2 quadrants along the S16 bracket line:

      Top quadrant:    seeds {1, 16, 8, 9, 5, 12, 4, 13}
      Bottom quadrant: seeds {6, 11, 3, 14, 7, 10, 2, 15}

    These correspond to the first 4 and last 4 matchups in _SEED_MATCHUP_ORDER
    respectively. Each quadrant's 4 R64 teams compete in 3 games (2 R64, 1 R32)
    to produce one S16 winner.

    If ``forced_champion`` is provided, it replaces the quadrant winner in
    its own quadrant regardless of S16 score.
    """
    top_quadrant_seeds = {1, 16, 8, 9, 5, 12, 4, 13}
    bottom_quadrant_seeds = {6, 11, 3, 14, 7, 10, 2, 15}

    s16_teams: Set[str] = set()
    forced_quadrant: Optional[Tuple[str, str]] = None  # (region, quadrant_label)
    if forced_champion is not None:
        for region in _REGION_ORDER:
            for seed, tid in by_region[region].items():
                if tid == forced_champion:
                    quadrant = "top" if seed in top_quadrant_seeds else "bottom"
                    forced_quadrant = (region, quadrant)
                    s16_teams.add(forced_champion)
                    break
            if forced_quadrant is not None:
                break

    for region in _REGION_ORDER:
        for quadrant_label, quadrant_seeds in (("top", top_quadrant_seeds), ("bottom", bottom_quadrant_seeds)):
            if forced_quadrant == (region, quadrant_label):
                continue
            quadrant_teams = [by_region[region][seed] for seed in quadrant_seeds if seed in by_region[region]]
            if not quadrant_teams:
                continue
            best_team = quadrant_teams[0]
            best_score = scorer(best_team, "S16")
            for team in quadrant_teams[1:]:
                s = scorer(team, "S16")
                if s > best_score or (s == best_score and team < best_team):
                    best_team = team
                    best_score = s
            s16_teams.add(best_team)

    return s16_teams


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def construct_bracket(
    mode: str,
    seeds: Dict[str, int],
    regions: Dict[str, str],
    round_probs: Dict[str, Dict[str, float]],
    public_picks: Dict[str, Dict[str, float]],
    risk_level: float,
    pool_size: int = 100,
    scoring_system: Optional[Dict[str, int]] = None,
    forced_champion: Optional[str] = None,
    max_one_seeds_f4: int = 2,
    chalk_bias_table: Optional[Dict[int, Dict[str, float]]] = None,
) -> Tuple[Dict[str, str], str, List[str], float, float]:
    """Construct a complete 63-game bracket using the specified mode.

    Returns ``(picks, champion, final_four, expected_points, variance)``
    where ``picks`` is a dict mapping 63 game keys to winner team_ids.

    See the module docstring for a description of each mode. All modes
    share the same ``_ev_score``-based scorer (via ``_make_ev_scorer``) so
    comparisons between modes isolate the construction-order signal.

    Args:
        chalk_bias_table: Optional empirical chalk-bias ratios loaded from
            ``load_chalk_bias_table()``. Only used by ``champ_first_chalkfade``
            mode; auto-loaded from the latest artifact if None.

    Raises:
        ValueError: if mode is unknown, if the seed map can't be built to
        16-per-region, or if the forced_champion doesn't exist in seeds.
    """
    if mode not in CONSTRUCTION_MODES:
        raise ValueError(f"Unknown construction mode: {mode!r}. Valid: {CONSTRUCTION_MODES}")

    if scoring_system is None:
        scoring_system = dict(_DEFAULT_SCORING)

    # forward_greedy + forced_champion falls back to champ_first (decided in
    # the design doc — forward_greedy has no natural way to inject a forced
    # champion, and champ_first is the most direct way to honor it).
    if mode == "forward_greedy" and forced_champion is not None:
        mode = "champ_first"

    if forced_champion is not None and forced_champion not in seeds:
        raise ValueError(f"forced_champion={forced_champion!r} not in seeds")

    by_region = _build_seed_map(seeds, regions)
    all_teams: List[str] = [tid for region in _REGION_ORDER for tid in by_region[region].values()]

    scorer = _make_ev_scorer(round_probs, public_picks, risk_level, pool_size, scoring_system)

    # Determine locked teams and lock_through_round per mode.
    if mode == "champ_first":
        champion = _pick_champion(scorer, all_teams, forced_champion)
        locked_teams: Set[str] = {champion}
        lock_through_round: Optional[str] = "CHAMP"
    elif mode == "champ_first_chalkfade":
        if chalk_bias_table is None:
            from ..data.seed_pick_model import load_chalk_bias_table

            chalk_bias_table = load_chalk_bias_table()
        chalkfade_scorer = _make_chalkfade_champ_scorer(scorer, seeds, chalk_bias_table)
        champion = _pick_champion(chalkfade_scorer, all_teams, forced_champion)
        locked_teams = {champion}
        lock_through_round = "CHAMP"
    elif mode == "f4_first":
        locked_teams = _pick_f4_teams(scorer, by_region, forced_champion, max_one_seeds=max_one_seeds_f4)
        lock_through_round = "E8"
    elif mode == "e8_first":
        locked_teams = _pick_s16_teams(scorer, by_region, forced_champion)
        lock_through_round = "S16"
    else:  # forward_greedy
        locked_teams = set()
        lock_through_round = None

    picks, champion, final_four = _walk_bracket(
        by_region,
        locked_teams,
        lock_through_round,
        scorer,
        forced_champion=forced_champion,
    )

    expected_points, variance = _compute_expected_points(picks, round_probs, scoring_system)

    return picks, champion, final_four, expected_points, variance
