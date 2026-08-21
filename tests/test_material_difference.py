"""Two brackets on screen must be two genuinely different brackets.

On the 2026 field, hierarchical selection's tier 1 (distinct champion) returned
a second EV bracket whose Final Four was *identical* to the first's. The user saw
"Model Favorite: Michigan" and "Close Alternative: Duke" with the same four teams
in the Final Four — one differing pick, presented as a strategic choice.

``select`` — the frozen v1 hierarchical rule — is NOT changed by any of this and
remains available for research comparisons. What changed is which selector the
product calls: ``select_diverse`` (SELECTION_VERSION "product.v2") maximises
quality subject to visible structural diversity, rather than maximising champion
count. Its tier ladder is Final Four, then Sweet 16, then champion, so champion
difference is a signal rather than the primary rule, and the answer to "is there
a second bracket worth showing at all?" is allowed to be no.

This is a product-selection version, deliberately separate from the frozen
2027.v2 methodology: no model, simulation, objective or P(1st) definition moves.

Deliberately NOT a distance metric. Hamming weights all 63 games equally, so R64
is 50.8% of Hamming but 16.7% of the points (FINDINGS.md 6e-6h).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.product.selection import (
    CHAMP,
    F4,
    S16,
    DEFAULT_MIN_RETENTION,
    difference_profile,
    is_materially_different,
    select_diverse,
    select_with_alternative,
)

REPO = Path(__file__).resolve().parent.parent
ARTIFACT = REPO / "docs" / "data" / "candidates_2026.json"

pytestmark = pytest.mark.skipif(not ARTIFACT.exists(), reason="candidate artifact not built")


@pytest.fixture(scope="module")
def artifact():
    return json.loads(ARTIFACT.read_text())


def _synthetic(pairs):
    """A minimal artifact with hand-built candidates, for the edge cases."""
    teams = [{"id": f"t{i}", "seed": (i % 16) + 1, "region": "X"} for i in range(64)]
    return {"teams": teams, "candidates": list(pairs)}


def _cand(champ, f4, s16, ev=100.0, p1=0.05):
    w = [[], s16, [], f4, [], [champ]]
    return {"w": w, "ev": ev, "p1": p1, "dd16": 0}


# ---------------------------------------------------------------------------
# The criterion itself
# ---------------------------------------------------------------------------


def test_champion_alone_is_not_material():
    """THE regression this module exists for.

    Same Final Four, same Sweet 16, different champion. The user experiences one
    bracket with a different name on the trophy.
    """
    art = _synthetic(
        [
            _cand(1, [1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8]),
            _cand(2, [1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8]),
        ]
    )
    assert difference_profile(art, 1, 0) == {"champion": 1, "final_four": 0, "sweet_16": 0}
    assert not is_materially_different(art, 1, 0)


def test_a_changed_final_four_is_material():
    art = _synthetic(
        [
            _cand(1, [1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8]),
            _cand(1, [1, 2, 3, 9], [1, 2, 3, 4, 5, 6, 7, 8]),
        ]
    )
    assert is_materially_different(art, 1, 0)


def test_two_changed_sweet_sixteen_teams_are_material():
    art = _synthetic(
        [
            _cand(1, [1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8]),
            _cand(1, [1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 20, 21]),
        ]
    )
    assert is_materially_different(art, 1, 0)


def test_one_changed_sweet_sixteen_team_is_not_material():
    """The threshold is two: a single early swap is not a different bracket."""
    art = _synthetic(
        [
            _cand(1, [1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8]),
            _cand(1, [1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 20]),
        ]
    )
    assert not is_materially_different(art, 1, 0)


def test_identical_brackets_are_not_material():
    art = _synthetic(
        [
            _cand(1, [1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8]),
            _cand(1, [1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8]),
        ]
    )
    assert not is_materially_different(art, 1, 0)


# ---------------------------------------------------------------------------
# Selection behaviour
# ---------------------------------------------------------------------------


def test_returns_one_bracket_when_no_alternative_qualifies():
    """The product must be willing to show a single bracket.

    Every eligible candidate here is identical in shape AND champion, so nothing
    qualifies at any tier; the only differentiated bracket is far below the
    retention floor. Honest output is one bracket, not a bad one promoted to fill
    a slot.
    """
    art = _synthetic(
        [
            _cand(1, [1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8], ev=100.0),
            _cand(1, [1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8], ev=99.5),  # identical
            _cand(9, [9, 10, 11, 12], [9, 10, 11, 12, 13, 14, 15, 16], ev=50.0),  # too weak
        ]
    )
    assert select_with_alternative(art, "ev") == [0]


def test_champion_only_is_the_last_resort_not_the_rule():
    """Tier ordering, stated as the two cases that distinguish it.

    Champion difference is a diversity *signal*: usable when nothing better
    exists, never preferred while something material is available. This is the
    single behaviour separating product.v2 from v1's champion-first rule.
    """
    baseline = _cand(1, [1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8], ev=100.0)

    # A champion-only bracket scores HIGHER than a materially different one.
    # Quality does not override tier: the material bracket still wins.
    art = _synthetic(
        [
            baseline,
            _cand(2, [1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8], ev=99.9),  # champion only
            _cand(1, [1, 2, 3, 9], [1, 2, 3, 4, 5, 6, 7, 8], ev=98.0),  # changed F4
        ]
    )
    assert select_diverse(art, "ev", k=2) == [0, 2]

    # With the material bracket gone, champion-only is accepted rather than
    # returning a single bracket.
    art = _synthetic([baseline, _cand(2, [1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8], ev=99.9)])
    assert select_diverse(art, "ev", k=2) == [0, 1]


def test_final_four_outranks_sweet_sixteen():
    """Tier 1 is exhausted before tier 2 is consulted, even at lower quality."""
    art = _synthetic(
        [
            _cand(1, [1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8], ev=100.0),
            _cand(1, [1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 20, 21], ev=99.9),  # S16 only
            _cand(1, [1, 2, 3, 9], [1, 2, 3, 4, 5, 6, 7, 8], ev=98.5),  # changed F4
        ]
    )
    assert select_diverse(art, "ev", k=2) == [0, 2]


def test_diversity_is_measured_against_the_whole_selected_set():
    """Slot 3 must differ from slot 2, not merely from the baseline.

    Candidates 1 and 2 share a Final Four with each other while both differing
    from the baseline. Only candidate 3 clears tier 1 against both.
    """
    art = _synthetic(
        [
            _cand(1, [1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8], ev=100.0),
            _cand(1, [1, 2, 3, 9], [1, 2, 3, 4, 5, 6, 7, 8], ev=99.5),
            _cand(1, [1, 2, 3, 9], [1, 2, 3, 4, 5, 6, 7, 8], ev=99.4),
            _cand(1, [1, 2, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8], ev=99.0),
        ]
    )
    sel = select_diverse(art, "ev", k=3)
    assert sel == [0, 1, 3]
    assert set(art["candidates"][sel[2]]["w"][F4]) - set(art["candidates"][sel[1]]["w"][F4])


def test_retention_floor_is_absolute_not_compounding():
    """Every slot is held to the same bar, measured from rank 1.

    Re-basing the floor on the most recent pick would let quality drift down slot
    by slot; at k=5 a 0.97 floor would compound to ~0.86.
    """
    cands = [_cand(1, [1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8], ev=100.0)]
    cands += [_cand(1, [1, 2, 3, 10 + n], [1, 2, 3, 4, 5, 6, 7, 8], ev=100.0 - 2.0 * (n + 1)) for n in range(4)]
    art = _synthetic(cands)
    sel = select_diverse(art, "ev", k=5, min_retention=0.97)
    assert all(art["candidates"][i]["ev"] >= 97.0 for i in sel)
    assert len(sel) == 2, "only one candidate sits within 3% of the baseline"


def test_k_of_one_returns_only_the_best():
    art = _synthetic(
        [
            _cand(1, [1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8], ev=100.0),
            _cand(1, [1, 2, 3, 9], [1, 2, 3, 4, 5, 6, 7, 8], ev=99.5),
        ]
    )
    assert select_diverse(art, "ev", k=1) == [0]


def test_invalid_k_raises():
    art = _synthetic([_cand(1, [1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8])])
    with pytest.raises(ValueError, match="k must be at least 1"):
        select_diverse(art, "ev", k=0)


def test_returns_two_when_a_qualifying_alternative_exists():
    art = _synthetic(
        [
            _cand(1, [1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8], ev=100.0),
            _cand(1, [1, 2, 3, 9], [1, 2, 3, 4, 5, 6, 7, 8], ev=99.0),
        ]
    )
    assert select_with_alternative(art, "ev") == [0, 1]


def test_alternative_is_the_highest_scoring_qualifier():
    """Not merely the first qualifier found in candidate order."""
    art = _synthetic(
        [
            _cand(1, [1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8], ev=100.0),
            _cand(1, [1, 2, 3, 9], [1, 2, 3, 4, 5, 6, 7, 8], ev=98.0),
            _cand(1, [1, 2, 3, 8], [1, 2, 3, 4, 5, 6, 7, 8], ev=99.0),
        ]
    )
    assert select_with_alternative(art, "ev") == [0, 2]


def test_retention_floor_is_enforced():
    art = _synthetic(
        [
            _cand(1, [1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8], ev=100.0),
            _cand(1, [1, 2, 3, 9], [1, 2, 3, 4, 5, 6, 7, 8], ev=90.0),
        ]
    )
    assert select_with_alternative(art, "ev", min_retention=0.95) == [0]
    assert select_with_alternative(art, "ev", min_retention=0.85) == [0, 1]


def test_baseline_is_always_the_top_scorer_on_the_objective(artifact):
    """The gate never costs the user the best bracket."""
    for objective in ("ev", "p1"):
        sel = select_with_alternative(artifact, objective)
        best = max(c[objective] for c in artifact["candidates"])
        assert artifact["candidates"][sel[0]][objective] == best


def test_unknown_objective_raises(artifact):
    with pytest.raises(ValueError, match="unknown objective"):
        select_with_alternative(artifact, "balanced")


# ---------------------------------------------------------------------------
# The real 2026 artifact
# ---------------------------------------------------------------------------


def test_2026_shipped_brackets_are_materially_different(artifact):
    """What the user will actually see on the site today."""
    for objective in ("ev", "p1"):
        sel = select_with_alternative(artifact, objective)
        if len(sel) == 2:
            assert is_materially_different(artifact, sel[1], sel[0]), (
                f"{objective} ships two brackets that are not materially different"
            )


def test_2026_ev_alternative_is_better_than_the_old_hierarchical_pick(artifact):
    """Documents the concrete improvement, so a regression is visible.

    The distinct-champion pick retained ~0.973 of baseline EV with an identical
    Final Four. The material-difference pick retains more EV *and* differs in
    shape — this is strictly better on both axes, not a trade.
    """
    cands = artifact["candidates"]
    sel = select_with_alternative(artifact, "ev")
    assert len(sel) == 2, "2026 should offer a qualifying EV alternative"
    baseline, alt = sel

    retention = cands[alt]["ev"] / cands[baseline]["ev"]
    assert retention >= DEFAULT_MIN_RETENTION

    champ = cands[baseline]["w"][CHAMP][0]
    base_f4 = set(cands[baseline]["w"][F4])
    order = sorted(range(len(cands)), key=lambda i: (-cands[i]["ev"], i))
    old_pick = next(i for i in order if cands[i]["w"][CHAMP][0] != champ)

    assert set(cands[old_pick]["w"][F4]) == base_f4, (
        "the degenerate case no longer reproduces; this test is now vacuous and "
        "should be re-derived against the current artifact"
    )
    assert retention > cands[old_pick]["ev"] / cands[baseline]["ev"], (
        "the material-difference alternative should retain more EV than the distinct-champion pick it replaced"
    )
    assert len(set(cands[alt]["w"][S16]) - set(cands[baseline]["w"][S16])) >= 2
