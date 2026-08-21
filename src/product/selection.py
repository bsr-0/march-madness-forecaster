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
