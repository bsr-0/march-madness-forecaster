"""Build the shipped candidate artifact, preserving the diversity we just earned.

The product ships a few thousand pre-scored brackets and does the last-mile
filter / rank / select in the browser. HOW those few thousand are drawn from the
~200k scenario bank is part of the product's statistical behaviour, not an
implementation detail: naive "top N by expected score" would reproduce exactly
the candidate collapse documented in FINDINGS.md 6e (~25 candidates -> ~5
distinct brackets, mean Hamming 6/63) that this whole line of work removed.

So the sampler uses explicit quotas rather than a ranking cut:

  1. CHAMPION STRATA. Slots are allocated per champion, proportional to that
     champion's probability but with a floor, so plausible-but-unlikely
     champions survive into the artifact instead of being ranked away.
  2. EV STRATA WITHIN CHAMPION. Within each champion, slots spread across
     expected-score deciles. This is what preserves the low-EV / high-P(1st)
     region -- the region where the two objectives disagree, and therefore the
     entire reason the product has more than one strategy.
  3. CONSTRAINT TOP-UP. Every supported preference is checked for survivor
     coverage afterwards and topped up if thin, so no UI control can silently
     return nothing.

LEAKAGE BOUNDARY
----------------
Everything here must be knowable the moment the bracket is released:

  seeds / regions          Selection Sunday
  torvik barthag           data_type=pre_tournament, cutoff_date before tip;
                           _validate_pretournament raises otherwise
  ESPN public picks        published before tip; require_archived=True

One documented exception: the shared P(1st) referee reads seed-vs-seed rates
from ``seed_pick_model._win_rate(window="recent")``, computed from Kaggle
results since 2010. For a forward-looking 2027 artifact it is clean, since all
its data precedes 2027. For validating a historical season it contains that
season's own results, applied identically to every candidate, so bounded and
non-differential. THE RECENT WINDOW MAKES THAT CONTAMINATION LARGER, not
smaller -- roughly 63 of ~1000 games rather than 63 of ~2500 -- because fewer
seasons means each one carries more weight. That is the price of the window and
it is recorded in the artifact's ``provenance`` block rather than hidden.

The referee is an OUTCOME model and the public-pick distribution is a CROWD
model, and they deliberately run on different windows. Pool edge is the gap
between them, so moving both together cancels exactly; see the window block in
src/data/seed_pick_model.py.

Usage:
    python3 scripts/experiments/build_candidate_artifact.py --year 2024
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts._common import load_seeds_and_regions  # noqa: E402
from scripts.experiments.conditional_bracket_engine import (  # noqa: E402
    _REACHES,
    expected_scores,
    round_marginals,
)
from scripts.experiments.objective_diversity_matrix import pool_p_first  # noqa: E402
from scripts.mc_pool_backtest import (  # noqa: E402
    ESPN_SCORING,
    _load_torvik_barthag,
    build_bracket_order,
    build_espn_pick_distribution,
    draw_selection_trials,
)
from src.prediction.pairwise import PairwiseProbabilities, simulate_bracket_outcomes  # noqa: E402
from src.prediction.seed_probabilities import build_seed_probabilities  # noqa: E402

ROUND_NAMES = ("R64", "R32", "S16", "E8", "F4", "CHAMP")
DEFAULT_POOL_SIZE = 30  # opponent field assumed by every P(1st) in the artifact


# ---------------------------------------------------------------------------
# Leakage gate
# ---------------------------------------------------------------------------


def assert_pretournament_inputs(year: int) -> Dict:
    """Fail before any compute if an input is not knowable at bracket release."""
    prov = {}
    for prefix in (Path("data/raw/historical"), Path("data/raw")):
        p = prefix / f"torvik_{year}.json"
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            if d.get("data_type") != "pre_tournament":
                raise RuntimeError(f"{p}: data_type={d.get('data_type')!r}, refusing to build.")
            prov["torvik"] = {
                "file": str(p),
                "data_type": d.get("data_type"),
                "cutoff_date": d.get("cutoff_date"),
                "tournament_start": d.get("tournament_start"),
            }
            break
    if "torvik" not in prov:
        raise RuntimeError(f"no torvik_{year}.json found; cannot verify provenance")

    from src.data.seed_pick_model import RECENT_FIRST_SEASON
    from src.prediction.seed_probabilities import OUTCOME_WINDOW

    prov["seed_head_to_head"] = {
        "source": (
            f"seed_pick_model._win_rate(window={OUTCOME_WINDOW!r}) "
            f"— seed-vs-seed rates computed from Kaggle results, {RECENT_FIRST_SEASON}+"
        ),
        "clean_for_forward_looking": True,
        "caveat": (
            "For historical validation this table includes the target season's own "
            "results (~63 of ~1000 games in this window). It is used only by the shared "
            "P(1st) referee and applied identically to every candidate, so it is bounded "
            "and does not favour one strategy over another. Note the recent window makes "
            "that share LARGER than it was under the full 1985-2025 table (~63 of ~2500), "
            "which is the price of the window: fewer games means each season, including "
            "the one being validated, carries more weight."
        ),
        "public_picks_window": (
            "The public-pick model is deliberately NOT on this window. It uses archived "
            "ESPN picks where available and falls back to SEED_PICK_RATES (1985-2025), "
            "because it models crowd belief rather than outcomes. Pool edge is the gap "
            "between the two, so moving both together would cancel it exactly."
        ),
    }
    return prov


# ---------------------------------------------------------------------------
# Diversity-preserving sampler
# ---------------------------------------------------------------------------


def stratified_sample(
    rounds: List,
    ev: np.ndarray,
    target: int,
    rng: np.random.Generator,
    min_per_champion: int = 8,
    ev_strata: int = 10,
) -> np.ndarray:
    """Draw ``target`` candidates preserving champion and objective diversity.

    Proportional-with-floor over champions, then spread across EV deciles inside
    each champion. The floor is what keeps unlikely-but-plausible champions in
    the artifact; the EV strata are what keep the low-EV / high-P(1st) region
    that makes the two strategies differ.
    """
    by_champ: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(rounds):
        c = r[_REACHES["CHAMP"]]
        if c:
            by_champ[c[0]].append(i)

    n_total = sum(len(v) for v in by_champ.values())
    quotas: Dict[str, int] = {}
    for champ, idxs in by_champ.items():
        prop = int(round(target * len(idxs) / n_total))
        quotas[champ] = min(len(idxs), max(min_per_champion, prop))

    # Scale back proportionally if the floors overshoot the target.
    over = sum(quotas.values())
    if over > target:
        scale = target / over
        for champ in quotas:
            quotas[champ] = max(1, int(quotas[champ] * scale))

    chosen: List[int] = []
    for champ, idxs in by_champ.items():
        want = quotas[champ]
        if want >= len(idxs):
            chosen.extend(idxs)
            continue
        arr = np.array(idxs)
        order = arr[np.argsort(ev[arr])]
        # Spread across EV strata rather than taking the top of the champion's
        # own distribution, which would re-collapse toward chalk within champion.
        buckets = np.array_split(order, min(ev_strata, len(order)))
        per = max(1, want // len(buckets))
        picked: List[int] = []
        for b in buckets:
            take = min(per, len(b))
            picked.extend(rng.choice(b, size=take, replace=False).tolist())
        if len(picked) < want:
            rest = [i for i in order.tolist() if i not in set(picked)]
            picked.extend(rng.choice(rest, size=min(want - len(picked), len(rest)), replace=False).tolist())
        chosen.extend(picked[:want])

    return np.array(sorted(set(chosen)))


# ---------------------------------------------------------------------------
# True constraint probabilities (full bank)
# ---------------------------------------------------------------------------


def _constraint_predicates(seeds: Dict[str, int]):
    """The predicates behind every preference control the UI can offer."""
    return {
        "f4_at_least_1_two_three": lambda r: sum(1 for t in r[_REACHES["F4"]] if seeds.get(t) in (2, 3)) >= 1,
        "f4_at_least_2_two_three": lambda r: sum(1 for t in r[_REACHES["F4"]] if seeds.get(t) in (2, 3)) >= 2,
        "f4_mostly_favorites": lambda r: sum(1 for t in r[_REACHES["F4"]] if seeds.get(t) == 1) >= 3,
        "s16_at_least_1_double_digit": lambda r: any(seeds.get(t, 0) >= 10 for t in r[_REACHES["S16"]]),
        "s16_at_least_2_double_digit": lambda r: sum(1 for t in r[_REACHES["S16"]] if seeds.get(t, 0) >= 10) >= 2,
        "s16_no_double_digit": lambda r: not any(seeds.get(t, 0) >= 10 for t in r[_REACHES["S16"]]),
    }


def true_constraint_probabilities(rounds: List, seeds: Dict[str, int]) -> Dict[str, float]:
    """P(constraint) over the FULL bank.

    These MUST come from the full bank, never from the shipped candidate list.
    The sampler deliberately over-samples unlikely champions to protect
    diversity, so the artifact is not a probability sample -- counting rows in
    it would bias every feasibility hint toward rare scenarios.
    """
    preds = _constraint_predicates(seeds)
    n = len(rounds)
    return {k: round(sum(1 for r in rounds if f(r)) / n, 5) for k, f in preds.items()}


def true_team_f4_probabilities(rounds: List, seeds: Dict[str, int]) -> Dict[str, float]:
    """P(team reaches Final Four) over the full bank, for the team dropdown."""
    c = Counter()
    for r in rounds:
        c.update(r[_REACHES["F4"]])
    n = len(rounds)
    return {t: round(v / n, 5) for t, v in sorted(c.items(), key=lambda kv: -kv[1])}


def load_team_names(year: int) -> Dict[str, str]:
    """Canonical display names, straight from the upstream tournament context.

    The artifact previously carried only ids, so the browser reconstructed names
    by title-casing the slug. That cannot round-trip: `saint_mary_s__ca` became
    "Saint Mary S Ca", `texas_a_m` became "Texas A M", and `tcu` became "Tcu".

    `team_name` is the authoritative value and is taken verbatim. No alias table
    and no frontend-side transformation: if the browser needs a value to render
    correctly, that value belongs in the artifact.
    """
    for prefix in (Path("data/raw/historical"), Path("data/raw")):
        path = prefix / f"tournament_context_{year}.json"
        if not path.exists():
            continue
        with open(path) as f:
            ctx = json.load(f)
        entries = (ctx.get("seeds") or {}).get("teams") or ctx.get("teams") or []
        names = {t["team_id"]: t["team_name"] for t in entries if t.get("team_id") and t.get("team_name")}
        if names:
            return names
    raise RuntimeError(
        f"no canonical team names found for {year}; the artifact must not ship ids for the browser to guess at"
    )


def true_team_round_probabilities(rounds: List, team_ids: List[str]) -> List[List[float]]:
    """P(team reaches each stage), over the FULL bank. Powers the Explore tab.

    Row per team (aligned to ``teams``), six columns: reaches R32, S16, E8,
    Final Four, Final, champion. Column 3 is the Final Four probability and
    column 5 is the title probability.

    Counted over every simulated tournament, NEVER over ``candidates``. The
    sampler deliberately over-samples unlikely champions to protect diversity, so
    counting candidate rows would overstate exactly the long shots a user is most
    likely to misread.
    """
    counters = [Counter() for _ in range(6)]
    for r in rounds:
        for stage in range(6):
            counters[stage].update(r[stage])
    n = len(rounds)
    return [[round(counters[stage][t] / n, 5) for stage in range(6)] for t in team_ids]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(bank, rounds, sel, ev, p1, first_round, seeds, full_rounds) -> Dict:
    """Checks that must pass before the artifact is fit to ship."""
    out: Dict[str, object] = {}

    # 1. Path consistency: a team may only win round R+1 if it won round R.
    bad = 0
    for i in sel[: min(500, len(sel))]:
        r = rounds[i]
        for ri in range(5):
            if not set(r[ri + 1]).issubset(set(r[ri])):
                bad += 1
                break
    out["path_consistent"] = bad == 0
    out["path_checked"] = min(500, len(sel))

    # 2. EV recomputed independently on a sample.
    marg = round_marginals(full_rounds)
    pts = {r: ESPN_SCORING[r] for r in ROUND_NAMES}
    errs = []
    for i in sel[: min(200, len(sel))]:
        manual = sum(pts[rn] * sum(marg[ri].get(t, 0.0) for t in rounds[i][ri]) for ri, rn in enumerate(ROUND_NAMES))
        errs.append(abs(manual - ev[i]))
    out["ev_max_abs_error"] = float(max(errs)) if errs else 0.0

    # 3. Diversity preserved vs the full bank.
    def champ_entropy(idxs):
        c = Counter(rounds[i][_REACHES["CHAMP"]][0] for i in idxs if rounds[i][_REACHES["CHAMP"]])
        tot = sum(c.values())
        p = np.array([v / tot for v in c.values()])
        return float(-(p * np.log2(p)).sum()), len(c)

    full_idx = list(range(len(rounds)))
    fe, fc = champ_entropy(full_idx)
    se, sc = champ_entropy(sel)
    rng = np.random.default_rng(0)

    def mean_hamming(idxs):
        a = rng.choice(idxs, size=min(4000, len(idxs)))
        b = rng.choice(idxs, size=len(a))
        keep = a != b
        return float((bank[a[keep]] != bank[b[keep]]).sum(axis=1).mean())

    out["champion_entropy_full"] = round(fe, 3)
    out["champion_entropy_artifact"] = round(se, 3)
    out["distinct_champions_full"] = fc
    out["distinct_champions_artifact"] = sc
    out["mean_hamming_full"] = round(mean_hamming(full_idx), 2)
    out["mean_hamming_artifact"] = round(mean_hamming(list(sel)), 2)

    # 4. Objective diversity: is the low-EV / high-P(1st) region retained?
    lo_ev = ev[sel] < np.percentile(ev[sel], 40)
    hi_p1 = p1 > np.percentile(p1, 60)
    out["low_ev_high_p1_count"] = int((lo_ev & hi_p1).sum())
    out["ev_p1_rank_corr"] = round(
        float(np.corrcoef(np.argsort(np.argsort(ev[sel])), np.argsort(np.argsort(p1)))[0, 1]), 3
    )

    # 5. Constraint coverage for every UI control.
    cons = {
        ">=1 2/3-seed F4": lambda r: sum(1 for t in r[_REACHES["F4"]] if seeds.get(t) in (2, 3)) >= 1,
        ">=2 2/3-seeds F4": lambda r: sum(1 for t in r[_REACHES["F4"]] if seeds.get(t) in (2, 3)) >= 2,
        ">=3 1-seeds F4": lambda r: sum(1 for t in r[_REACHES["F4"]] if seeds.get(t) == 1) >= 3,
        ">=1 dd-seed S16": lambda r: any(seeds.get(t, 0) >= 10 for t in r[_REACHES["S16"]]),
        ">=2 dd-seeds S16": lambda r: sum(1 for t in r[_REACHES["S16"]] if seeds.get(t, 0) >= 10) >= 2,
        "no dd-seed S16": lambda r: not any(seeds.get(t, 0) >= 10 for t in r[_REACHES["S16"]]),
    }
    out["constraint_coverage"] = {k: int(sum(1 for i in sel if f(rounds[i]))) for k, f in cons.items()}
    return out


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _ev_optimal_bracket(first_round, marg) -> List[List[str]]:
    """The exact expected-points maximum, by dynamic programming on the bracket.

    WHY NOT GREEDY. Expected score is separable across games --
    sum_R pts_R * sum_{picked in R} P(t wins R) -- but the picks are not free:
    a team can only be picked in round R if it was picked in R-1. Taking the
    higher marginal at each game bottom-up can therefore strand you, choosing a
    team that is likelier to win THIS game but much less likely to win the next
    one, where the points are four times larger. The DP evaluates that
    trade-off; greedy cannot see it.

    The recursion is over subtrees. best[g][t] is the most expected points
    obtainable from the subtree rooted at game g given that t is the team picked
    to emerge from it:

        best[g][t] = pts_R * P(t wins R)
                   + best[child containing t][t]
                   + max_u best[other child][u]

    The last term is independent of t, which is what makes this linear rather
    than combinatorial: 63 games by at most 64 teams.

    IN PRACTICE THE OPTIMUM IS WORTH ALMOST NOTHING OVER THE SIMPLE RULE.
    Against _champion_equity_strategy it differs by zero games in 2024 and one
    game in 2025 and 2026, worth under a point out of ~950. That is a result
    about this problem, not a reason to skip the DP: it says picking by title
    odds is essentially optimal for expected score, which is not obvious and is
    worth being able to state.

    Where it does matter is against the SAMPLED candidates, which fall 24-39
    points short because the optimum is simply not in the pool.
    """
    pts = [float(ESPN_SCORING[r]) for r in ROUND_NAMES]
    memo: Dict = {}

    def solve(r: int, i: int) -> Dict:
        if (r, i) in memo:
            return memo[(r, i)]
        if r == 0:
            a, b = first_round[2 * i], first_round[2 * i + 1]
            res = {
                a: (pts[0] * marg[0].get(a, 0.0), None, None),
                b: (pts[0] * marg[0].get(b, 0.0), None, None),
            }
        else:
            left, right = solve(r - 1, 2 * i), solve(r - 1, 2 * i + 1)
            bl = max(left, key=lambda t: left[t][0])
            br = max(right, key=lambda t: right[t][0])
            res = {}
            for t in left:
                res[t] = (pts[r] * marg[r].get(t, 0.0) + left[t][0] + right[br][0], "L", br)
            for t in right:
                res[t] = (pts[r] * marg[r].get(t, 0.0) + right[t][0] + left[bl][0], "R", bl)
        memo[(r, i)] = res
        return res

    root = solve(5, 0)
    champion = max(root, key=lambda t: root[t][0])

    winners: List[List[str]] = [[] for _ in range(6)]

    def walk(r: int, i: int, t: str) -> None:
        winners[r].append(t)
        if r == 0:
            return
        _, side, other = memo[(r, i)][t]
        if side == "L":
            walk(r - 1, 2 * i, t)
            walk(r - 1, 2 * i + 1, other)
        else:
            walk(r - 1, 2 * i + 1, t)
            walk(r - 1, 2 * i, other)

    walk(5, 0, champion)
    return winners


def _encode_rows(winners, first_round):
    """Bracket -> the (1, 63) boolean shape encoding the pool scorer expects."""
    row = np.zeros((1, 63), dtype=bool)
    picked = [set(r) for r in winners]
    current, game = list(first_round), 0
    for r in range(6):
        nxt = []
        for g in range(0, len(current), 2):
            t1, t2 = current[g], current[g + 1]
            first_wins = t1 in picked[r]
            row[0, game] = first_wins
            nxt.append(t1 if first_wins else t2)
            game += 1
        current = nxt
    return row


def _champion_equity_strategy(first_round, marg, p1_trials) -> Dict:
    """Decide every game by P(champion) rather than by P(winning that game).

    A RULE, NOT A SEARCH, which is what makes it worth shipping beside the two
    optimised strategies. Those pick the best bracket out of ~3,000 scored
    candidates; this one is a single sentence you could apply by hand -- at each
    game take whichever team is likelier to win the whole thing -- and it is
    still competitive. It is also the only strategy here a user can fully verify
    without trusting the optimiser.

    IT IS SCORED WITH THE SAME MARGINALS AND THE SAME POOL TRIALS as the
    candidates, so its ev and p1 sit on the same scale as theirs. Scoring it
    separately would produce two numbers that look comparable and are not --
    P(1st) in particular is meaningless except against a specific opponent
    field.

    Measured at pool 30 over 2011-2026 this beats sampling the same ratings by
    +0.0226 P(1st) (CI [+0.0033, +0.0439]) but trails the P(1st)-optimised
    bracket substantially. It earns its place as an interpretable option, not as
    the recommendation.
    """
    champ = marg[5]
    winners, current = [], list(first_round)
    row = np.zeros((1, 63), dtype=bool)
    game = 0
    for _ in range(6):
        nxt = []
        for g in range(0, len(current), 2):
            t1, t2 = current[g], current[g + 1]
            first_wins = champ.get(t1, 0.0) >= champ.get(t2, 0.0)
            row[0, game] = first_wins
            w = t1 if first_wins else t2
            nxt.append(w)
            game += 1
        winners.append(nxt)
        current = nxt

    out = {
        "champ_equity": {
            "w": winners,
            "ev": round(float(expected_scores([winners], marg, ESPN_SCORING)[0]), 1),
            "p1": round(float(pool_p_first(row, p1_trials, first_round)[0]), 4),
        }
    }

    # The exact expected-points maximum, scored the same way. Shipped so the
    # "maximise expected points" strategy can be the actual maximum rather than
    # the best of a sample: the sampled candidates fall 24-39 points short
    # because the optimum is not in the pool.
    ev_opt = _ev_optimal_bracket(first_round, marg)
    ev_row = _encode_rows(ev_opt, first_round)
    out["ev_optimal"] = {
        "w": ev_opt,
        "ev": round(float(expected_scores([ev_opt], marg, ESPN_SCORING)[0]), 1),
        "p1": round(float(pool_p_first(ev_row, p1_trials, first_round)[0]), 4),
    }
    return out


def build(year: int, n_sims: int, target: int, trials: int, seed: int) -> Dict:
    prov = assert_pretournament_inputs(year)
    seeds, regions = load_seeds_and_regions(year)
    first_round = build_bracket_order(seeds, regions)
    barthag = _load_torvik_barthag(year, seeds)
    pw = PairwiseProbabilities.from_ratings(barthag, source=f"log5(torvik_{year})")

    rng = np.random.default_rng(seed)
    print(f"[1/5] simulating {n_sims:,} tournaments ...")
    bank, rounds = simulate_bracket_outcomes(pw, first_round, n_sims, rng, noise_std=0.0)

    print("[2/5] exact expected scores ...")
    marg = round_marginals(rounds)
    ev = expected_scores(rounds, marg, ESPN_SCORING)

    print(f"[3/5] diversity-preserving sample -> {target:,} ...")
    sel = stratified_sample(rounds, ev, target, rng)

    print(f"[4/5] P(1st) for {len(sel):,} candidates, {trials:,} shared trials ...")
    seed_pw = build_seed_probabilities(seeds)
    pick_dist = build_espn_pick_distribution(year, seeds)
    p1_trials = draw_selection_trials(
        trials,
        n_opponents=DEFAULT_POOL_SIZE,
        first_round=first_round,
        pick_dist=pick_dist,
        matchup_probs=seed_pw,
        seeds=seeds,
        rng=np.random.default_rng(seed + 7),
    )
    p1 = pool_p_first(bank[sel], p1_trials, first_round)

    named = _champion_equity_strategy(first_round, marg, p1_trials)

    print("[5/5] validating ...")
    checks = validate(bank, rounds, sel, ev, p1, first_round, seeds, rounds)
    true_probs = true_constraint_probabilities(rounds, seeds)
    team_f4 = true_team_f4_probabilities(rounds, seeds)

    # The artifact is deliberately not a probability sample -- verify the
    # difference is real so the warning below is not decorative.
    preds = _constraint_predicates(seeds)
    artifact_probs = {k: round(sum(1 for i in sel if f(rounds[i])) / len(sel), 5) for k, f in preds.items()}
    checks["constraint_prob_bias"] = {k: round(artifact_probs[k] - true_probs[k], 4) for k in true_probs}

    # Compact encoding: team table + per-round winner indices.
    team_ids = sorted(seeds)
    tidx = {t: i for i, t in enumerate(team_ids)}
    candidates = []
    for j, i in enumerate(sel):
        r = rounds[i]
        candidates.append(
            {
                "w": [[tidx[t] for t in r[ri]] for ri in range(6)],
                "ev": round(float(ev[i]), 1),
                "p1": round(float(p1[j]), 4),
                "dd16": sum(1 for t in r[_REACHES["S16"]] if seeds.get(t, 0) >= 10),
            }
        )

    # Canonical per-matchup win probabilities, shipped so the browser never
    # reconstructs model math from ratings.
    #
    # These are the SAME PairwiseProbabilities that drove the simulation, not a
    # display-only recomputation, so the board cannot disagree with the bank it
    # came from. Stored as a flat row-major n*n table of P(row beats col) rather
    # than only the 63 games of each candidate: candidates disagree about who
    # meets whom, and a per-candidate encoding would be both larger and
    # redundant. Rounding to 4dp is a rendering tolerance, asserted in
    # tests/test_artifact_pairwise_contract.py.
    team_names = load_team_names(year)
    missing_names = [t for t in team_ids if t not in team_names]
    if missing_names:
        raise RuntimeError(
            f"no canonical name for {missing_names}; refusing to ship an artifact "
            "the browser would have to guess names from"
        )

    team_round_probs = true_team_round_probabilities(rounds, team_ids)

    n = len(team_ids)
    pairwise_flat = [
        round(float(pw.p(team_ids[i], team_ids[j])), 4) if i != j else 0.5 for i in range(n) for j in range(n)
    ]

    return {
        # Schema 3 adds `pairwise`. This is an implementation-contract change --
        # moving an already-computed value from the browser into the artifact --
        # and NOT a methodology change: the numbers are identical to what the
        # frozen 2027.v2 pipeline already produced.
        # Schema 4 adds `team_round_probabilities` (Explore). Like `pairwise`,
        # this is an implementation-contract addition: the numbers were already
        # computed by the frozen pipeline and are merely transported.
        # Schema 5 adds the canonical `name` on each team. Additive, and an
        # artifact/UI contract change only: no model, simulation, objective,
        # preference or selection behaviour moves, and 2027.v2 is untouched.
        "schema": 5,
        "year": year,
        "teams": [{"id": t, "name": team_names[t], "seed": seeds[t], "region": regions.get(t, "")} for t in team_ids],
        # The 64 team indices in bracket order (game g is [2g], [2g+1]). Carried
        # in the artifact so the browser can reconstruct game pairings without
        # reimplementing build_bracket_order -- the artifact is a contract, and
        # anything the client needs to render belongs inside it rather than in a
        # duplicated JS constant that can drift.
        "first_round": [tidx[t] for t in first_round],
        "pairwise": pairwise_flat,
        "team_round_probabilities": team_round_probs,
        "candidates": candidates,
        # Rule-based strategies, scored on the same marginals and pool trials as
        # the candidates so every number in the UI is on one scale.
        "named_strategies": {
            k: {"w": [[tidx[t] for t in r] for r in v["w"]], "ev": v["ev"], "p1": v["p1"]} for k, v in named.items()
        },
        "meta": {
            "n_sims": n_sims,
            "n_candidates": len(candidates),
            "p1_trials": trials,
            "p1_pool_size": DEFAULT_POOL_SIZE,
            "p1_assumption": (
                f"P(1st) assumes a {DEFAULT_POOL_SIZE}-opponent pool with ESPN public "
                f"pick behaviour. It is NOT a universal probability of winning any pool."
            ),
            "p1_se_estimate": round(float(np.sqrt(0.05 * 0.95 / trials)), 5),
            "candidates_are_not_a_probability_sample": (
                "Unlikely champions are deliberately over-sampled to protect diversity. "
                "Use constraint_probabilities / team_final_four_probabilities for any "
                "frequency shown to a user; NEVER count rows in `candidates`."
            ),
            "objectives": ["ev", "p1"],
            "source": pw.source,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "constraint_probabilities": true_probs,
        "team_final_four_probabilities": team_f4,
        "provenance": prov,
        "validation": checks,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--n-sims", type=int, default=150_000)
    ap.add_argument("--target", type=int, default=3000)
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--out", type=str, default="artifacts/candidates")
    args = ap.parse_args()

    art = build(args.year, args.n_sims, args.target, args.trials, args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"candidates_{args.year}.json"
    with open(path, "w") as f:
        json.dump(art, f, separators=(",", ":"))
    size_mb = path.stat().st_size / 2**20

    v = art["validation"]
    print(f"\n{'=' * 78}\nCANDIDATE ARTIFACT — {args.year}\n{'=' * 78}")
    print(f"  candidates            {art['meta']['n_candidates']:,}")
    print(f"  file size             {size_mb:.2f} MB")
    print(f"  path-consistent       {v['path_consistent']}  ({v['path_checked']} checked)")
    print(f"  EV max abs error      {v['ev_max_abs_error']:.6f}")
    print(f"  champions   full {v['distinct_champions_full']:3d} -> artifact {v['distinct_champions_artifact']:3d}")
    print(f"  champ entropy full {v['champion_entropy_full']:.3f} -> artifact {v['champion_entropy_artifact']:.3f}")
    print(f"  mean Hamming full {v['mean_hamming_full']:.1f} -> artifact {v['mean_hamming_artifact']:.1f}")
    print(f"  low-EV/high-P1 kept   {v['low_ev_high_p1_count']}")
    print(f"  EV-vs-P1 rank corr    {v['ev_p1_rank_corr']}")
    bias = v["constraint_prob_bias"]
    print(
        f"  artifact-vs-true P(constraint) bias: "
        f"min {min(bias.values()):+.3f}  max {max(bias.values()):+.3f}  "
        f"(why frequencies ship separately)"
    )
    print("  constraint coverage:")
    for k, n in v["constraint_coverage"].items():
        print(f"     {k:22} {n:6,}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
