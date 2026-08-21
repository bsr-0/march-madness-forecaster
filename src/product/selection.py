"""Canonical bracket selection — the semantics the browser must mirror.

Python is the reference implementation. The JavaScript in ``docs/`` is a
deliberately constrained mirror of *this* module: it does not invent selection
logic, and it is not permitted to diverge. ``tests/test_product_parity.py``
generates fixtures from these functions and asserts the JS reproduces them
exactly for representative objective/preference combinations.

If the JS implementation of something here turns out to be awkward, the fix is
to make the artifact contract clearer — not to let the two implementations drift
apart or to move canonical logic into the browser because it was convenient.

Everything here operates on the shipped candidate artifact and introduces no new
objective, preference, scoring rule or model parameter. The frozen 2027.v2
specification defines all of those; this module only applies them.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

# Index into a candidate's per-round winner lists.
R32, S16, E8, F4, FINAL, CHAMP = range(6)

# The frozen v1 objectives. No blend, no ownership penalty.
OBJECTIVES = ("ev", "p1")


# ---------------------------------------------------------------------------
# Preference predicates (frozen v1 set)
# ---------------------------------------------------------------------------


def _seed_of(artifact: Dict[str, Any], team_index: int) -> int:
    return artifact["teams"][team_index]["seed"]


def preference_predicates(artifact: Dict[str, Any]) -> Dict[str, Callable[[List[List[int]]], bool]]:
    """The frozen preference predicates, keyed as in the artifact.

    Each takes a candidate's ``w`` (per-round winner index lists) and returns
    whether that bracket satisfies the preference.
    """

    def seeds_in(w, rnd):
        return [_seed_of(artifact, t) for t in w[rnd]]

    return {
        "none": lambda w: True,
        "f4_at_least_1_two_three": lambda w: sum(1 for s in seeds_in(w, F4) if s in (2, 3)) >= 1,
        "f4_at_least_2_two_three": lambda w: sum(1 for s in seeds_in(w, F4) if s in (2, 3)) >= 2,
        "f4_mostly_favorites": lambda w: sum(1 for s in seeds_in(w, F4) if s == 1) >= 3,
        "s16_at_least_1_double_digit": lambda w: any(s >= 10 for s in seeds_in(w, S16)),
        "s16_at_least_2_double_digit": lambda w: sum(1 for s in seeds_in(w, S16) if s >= 10) >= 2,
        "s16_no_double_digit": lambda w: not any(s >= 10 for s in seeds_in(w, S16)),
    }


def team_reaches_final_four(artifact: Dict[str, Any], team_index: int) -> Callable:
    """The one parameterised predicate: a named team in the Final Four."""
    return lambda w: team_index in w[F4]


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def select(
    artifact: Dict[str, Any],
    objective: str = "ev",
    preference: str = "none",
    team_index: Optional[int] = None,
    k: int = 3,
) -> List[int]:
    """Return ``k`` candidate indices: highest-scoring, distinct compositions.

    Hierarchical diversity, exactly as frozen: distinct champion first, then
    distinct Final Four, then remaining highest scorers. Plain top-k would return
    k one-pick variations of the same bracket — the collapse this whole line of
    work exists to prevent (FINDINGS.md 6e-6h).

    Ties are broken by candidate index so the result is deterministic and the JS
    mirror can reproduce it without depending on sort stability.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; frozen v1 allows {OBJECTIVES}")

    preds = preference_predicates(artifact)
    if preference == "team_reaches_final_four":
        if team_index is None:
            raise ValueError("team_reaches_final_four requires team_index")
        pred = team_reaches_final_four(artifact, team_index)
    elif preference in preds:
        pred = preds[preference]
    else:
        raise ValueError(f"unknown preference {preference!r}")

    cands = artifact["candidates"]
    surviving = [i for i, c in enumerate(cands) if pred(c["w"])]
    # Descending by objective, ascending by index on ties.
    surviving.sort(key=lambda i: (-cands[i][objective], i))

    chosen: List[int] = []
    used_champs, used_f4 = set(), set()

    for i in surviving:  # tier 1 — distinct champion
        if len(chosen) >= k:
            break
        champ = cands[i]["w"][CHAMP][0]
        if champ not in used_champs:
            chosen.append(i)
            used_champs.add(champ)
            used_f4.add(tuple(sorted(cands[i]["w"][F4])))

    for i in surviving:  # tier 2 — distinct Final Four
        if len(chosen) >= k:
            break
        f4 = tuple(sorted(cands[i]["w"][F4]))
        if f4 not in used_f4 and i not in chosen:
            chosen.append(i)
            used_f4.add(f4)

    for i in surviving:  # tier 3 — top up
        if len(chosen) >= k:
            break
        if i not in chosen:
            chosen.append(i)

    return chosen


# ---------------------------------------------------------------------------
# Diverse selection (product.v2)
# ---------------------------------------------------------------------------

# PRODUCT SELECTION VERSION. This is deliberately separate from the frozen
# 2027.v2 methodology version: nothing here touches the model, the simulation,
# the objectives or P(1st). It changes only which of the already-scored
# candidates are shown to a user, so it is versioned on its own axis and the
# prospective freeze is NOT retroactively altered.
#
# product.v1 -- hierarchical: distinct champion, then distinct Final Four.
# product.v2 -- quality-first with tiered structural diversity (below).
# product.v3 -- v2 semantics unchanged, now hash-pinned in
#               configs/frozen/product_v3.json and derived from this module, so
#               editing a tier or threshold moves a hash instead of silently
#               diverging from a spec that transcribed it as a literal.
SELECTION_VERSION = "product.v3"

# Why v2 exists, from the 2026 artifact: v1's "distinct champion first" rule
# returned an alternative retaining 0.973 of baseline EV whose Final Four was
# IDENTICAL to the baseline's -- one differing pick presented as a strategy.
# A bracket retaining 0.995 EV with a changed Final Four and 3 changed Sweet 16
# teams was available and passed over. Champion diversity was being treated as
# intrinsically more valuable than diversity a user can actually see.
MIN_F4_CHANGES = 1
MIN_S16_CHANGES = 2

# Diversity tiers, most to least meaningful. A lower tier is only reached when no
# candidate qualifies at a higher one within the retention floor, which is what
# makes "champion difference alone" a last resort rather than the primary rule.
DIVERSITY_TIERS = ("final_four", "sweet_16", "champion")

# How much of the BASELINE's objective value a later bracket may give up.
# Not binding on 2026 EV (the chosen alternative retains 0.995); it exists to
# stop a poor bracket being promoted merely because it looks different.
DEFAULT_MIN_RETENTION = 0.97

# Thresholds are structural counts, not a distance metric. Hamming is
# deliberately unused: it weights all 63 games equally, so R64 is 50.8% of
# Hamming but 16.7% of the points (FINDINGS.md 6e-6h).


def difference_profile(artifact: Dict[str, Any], index: int, baseline_index: int) -> Dict[str, int]:
    """Structural differences between two candidates, as counts."""
    a = artifact["candidates"][index]["w"]
    b = artifact["candidates"][baseline_index]["w"]
    return {
        "champion": int(a[CHAMP][0] != b[CHAMP][0]),
        "final_four": len(set(a[F4]) - set(b[F4])),
        "sweet_16": len(set(a[S16]) - set(b[S16])),
    }


def differs_at_tier(artifact: Dict[str, Any], index: int, other_index: int, tier: str) -> bool:
    """Whether two candidates differ from each other at a specific tier."""
    d = difference_profile(artifact, index, other_index)
    if tier == "final_four":
        return d["final_four"] >= MIN_F4_CHANGES
    if tier == "sweet_16":
        return d["sweet_16"] >= MIN_S16_CHANGES
    if tier == "champion":
        return d["champion"] >= 1
    raise ValueError(f"unknown diversity tier {tier!r}")


def is_materially_different(artifact: Dict[str, Any], index: int, baseline_index: int) -> bool:
    """Whether a user would experience these as genuinely different brackets.

    A different champion is NOT sufficient on its own. That is the exact
    degenerate case this exists to catch: two brackets with the same Final Four
    read as one bracket with a different name on the trophy.
    """
    return differs_at_tier(artifact, index, baseline_index, "final_four") or differs_at_tier(
        artifact, index, baseline_index, "sweet_16"
    )


def select_diverse(
    artifact: Dict[str, Any],
    objective: str = "ev",
    k: int = 2,
    min_retention: float = DEFAULT_MIN_RETENTION,
) -> List[int]:
    """Up to ``k`` brackets: quality first, subject to meaningful structural diversity.

    The rule, in order:

    1. Respect the objective. Slot 1 is always the highest-scoring candidate, so
       the gate never costs the user the best bracket.
    2. Respect quality. Nothing below ``min_retention`` of the baseline's
       objective value is eligible, however different it looks.
    3. Require differentiation a user can see, preferring a changed Final Four,
       then changed Sweet 16 teams, and only then a changed champion.
    4. Champion difference is a diversity signal, not a requirement. It is never
       the reason a bracket is chosen while a materially different bracket is
       still available within the quality floor.

    Diversity is measured against the whole already-selected set, not just the
    baseline, so slot 3 cannot duplicate slot 2's shape.

    Returns FEWER than ``k`` indices when the field genuinely does not contain
    another distinguishable bracket. Showing one honest bracket beats
    manufacturing a second.

    This does not modify :func:`select`, which still implements the frozen v1
    hierarchical rule and remains available for research comparisons.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; frozen v1 allows {OBJECTIVES}")
    if k < 1:
        raise ValueError("k must be at least 1")

    cands = artifact["candidates"]
    if not cands:
        return []

    order = sorted(range(len(cands)), key=lambda i: (-cands[i][objective], i))
    chosen = [order[0]]
    floor = cands[order[0]][objective] * min_retention

    while len(chosen) < k:
        # Highest-scoring eligible candidate at the most meaningful tier that has
        # one. Tier order is what makes champion-only a last resort.
        pick = None
        for tier in DIVERSITY_TIERS:
            for i in order[1:]:
                if cands[i][objective] < floor:
                    break  # descending: nothing further can qualify
                if i in chosen:
                    continue
                if all(differs_at_tier(artifact, i, c, tier) for c in chosen):
                    pick = i
                    break
            if pick is not None:
                break
        if pick is None:
            break  # no distinguishable bracket left; return what we have
        chosen.append(pick)

    return chosen


def select_with_alternative(
    artifact: Dict[str, Any],
    objective: str = "ev",
    min_retention: float = DEFAULT_MIN_RETENTION,
) -> List[int]:
    """The Build flow's selector: one bracket, plus an alternative if one exists."""
    return select_diverse(artifact, objective, k=2, min_retention=min_retention)


def constraint_frequency(artifact: Dict[str, Any], preference: str, team_id: str = "") -> Optional[float]:
    """The user-facing "happens in X of 10 tournaments" figure.

    Read from the artifact's full-bank fields, NEVER by counting candidates. The
    sampler deliberately over-samples unlikely champions to protect diversity, so
    the candidate list is not a probability sample — counting it would bias every
    frequency shown to a user toward rare scenarios.
    """
    if preference == "none":
        return 1.0
    if preference == "team_reaches_final_four":
        return artifact.get("team_final_four_probabilities", {}).get(team_id)
    return artifact.get("constraint_probabilities", {}).get(preference)


# ---------------------------------------------------------------------------
# Presentation helpers (shared definitions, so JS and Python agree)
# ---------------------------------------------------------------------------


def candidate_summary(artifact: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Champion, Final Four and upset profile for one candidate."""
    c = artifact["candidates"][index]
    teams = artifact["teams"]
    champ = c["w"][CHAMP][0]
    return {
        "index": index,
        "champion_id": teams[champ]["id"],
        "champion_seed": teams[champ]["seed"],
        "final_four": sorted(
            ({"id": teams[t]["id"], "seed": teams[t]["seed"]} for t in c["w"][F4]),
            key=lambda x: (x["seed"], x["id"]),
        ),
        "double_digit_s16": c["dd16"],
        "ev": c["ev"],
        "p1": c["p1"],
    }


def pairwise_prob(artifact: Dict[str, Any], i: int, j: int) -> float:
    """Canonical ``P(team i beats team j)``, read from the artifact.

    Schema 3 ships the same ``PairwiseProbabilities`` table that drove the
    simulation. Nothing — least of all the browser — recomputes it from ratings:
    a second implementation of tournament math could drift from the bank the
    brackets were drawn out of, and the board would then contradict itself.
    """
    table = artifact.get("pairwise")
    if table is None:
        raise KeyError("artifact is missing the pairwise table (schema < 3)")
    n = len(artifact["teams"])
    return table[i * n + j]


def candidate_games(artifact: Dict[str, Any], index: int) -> List[Dict[str, Any]]:
    """The 63 games of one candidate, in bracket order, with display probabilities.

    This is the canonical definition of what the bracket board shows. The JS
    ``candidateToRounds`` mirrors it and ``tests/test_product_parity.py`` asserts
    the two agree game for game, so rendering cannot silently diverge.
    """
    c = artifact["candidates"][index]
    teams = artifact["teams"]
    current = list(artifact["first_round"])
    games: List[Dict[str, Any]] = []

    for rnd in range(6):
        winners = set(c["w"][rnd])
        nxt = []
        for g in range(0, len(current), 2):
            i1, i2 = current[g], current[g + 1]
            winner = i1 if i1 in winners else i2
            games.append(
                {
                    "round": rnd,
                    "team1": teams[i1]["id"],
                    "team2": teams[i2]["id"],
                    "win_prob": pairwise_prob(artifact, i1, i2),
                    "winner": teams[winner]["id"],
                }
            )
            nxt.append(winner)
        current = nxt
    return games


def why_this_differs(artifact: Dict[str, Any], index: int, baseline_index: int) -> List[str]:
    """Plain-language differences against the baseline bracket.

    Built only from candidate metadata already in the artifact. No new
    computation, and deliberately no claim about which bracket is *better* —
    the two objectives are near-orthogonal, so "different" is the honest framing.
    """
    a, b = candidate_summary(artifact, index), candidate_summary(artifact, baseline_index)
    out: List[str] = []
    if a["champion_id"] != b["champion_id"]:
        out.append(
            f"Takes {a['champion_id']} ({a['champion_seed']}) as champion "
            f"instead of {b['champion_id']} ({b['champion_seed']})."
        )
    a_f4 = {t["id"] for t in a["final_four"]}
    b_f4 = {t["id"] for t in b["final_four"]}
    added, dropped = a_f4 - b_f4, b_f4 - a_f4
    if added or dropped:
        bits = []
        if added:
            bits.append("adds " + ", ".join(sorted(added)))
        if dropped:
            bits.append("drops " + ", ".join(sorted(dropped)))
        out.append("Final Four " + " and ".join(bits) + ".")
    if a["double_digit_s16"] != b["double_digit_s16"]:
        out.append(
            f"Advances {a['double_digit_s16']} double-digit seed(s) to the Sweet 16 "
            f"rather than {b['double_digit_s16']}."
        )
    if not out:
        out.append("Differs only in individual game picks, not in overall shape.")
    return out
