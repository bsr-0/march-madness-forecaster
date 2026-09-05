"""The UI's filter attributes must describe the brackets they are attached to.

WHY THIS IS A TEST AND NOT A COMMENT. Every filter in the browser is a scan over
attributes shipped in the payload -- champion, one-seeds in the Final Four,
Final Four depth, preference predicates, provenance. The browser trusts them
completely: it never re-derives them from the bracket. So an attribute that
disagrees with its own bracket does not raise anything, it just quietly returns
the wrong bracket for a filter, and the page looks entirely healthy while doing
it.

The brackets ship as 63-character bit strings against the first-round order, so
the check is: decode the bracket, recompute every attribute from the decoded
rounds, and require agreement. That also covers the encoder, since a bug there
would show up as attributes that no longer match.

Exhaustive filter behaviour (77,112 axis combinations, alternates, chip
enablement) is verified in the browser; this guards the data those checks stand
on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SEASONS = [2024, 2025, 2026]


def _payload(year):
    p = REPO / "docs" / "data" / f"season_{year}.json"
    if not p.exists():
        pytest.skip(f"no payload for {year}")
    return json.loads(p.read_text())


def _decode(bits, first_round):
    rounds, current, i = [], list(first_round), 0
    for _ in range(6):
        nxt = []
        for g in range(0, len(current), 2):
            t1, t2 = current[g], current[g + 1]
            w = t1 if bits[i] == "1" else t2
            nxt.append(w)
            i += 1
        rounds.append(nxt)
        current = nxt
    return rounds


@pytest.mark.parametrize("year", SEASONS)
def test_attributes_match_the_bracket_they_describe(year):
    d = _payload(year)
    f = d.get("filters")
    assert f, "payload carries no filters block"
    teams, fr = d["teams"], d["first_round"]
    for row in f["candidates"]:
        w = _decode(row["b"], fr)
        f4 = [teams[i]["seed"] for i in w[3]]
        assert row["c"] == w[5][0], f"{year}: champion attribute disagrees with the bracket"
        assert row["o"] == sum(1 for s in f4 if s == 1), f"{year}: one-seed count disagrees"
        assert row["d"] == max(f4), f"{year}: Final Four depth disagrees"


@pytest.mark.parametrize("year", SEASONS)
def test_predicate_flags_match_the_frozen_predicates(year):
    """The bit flags must agree with src/product/selection.py, not approximate it.

    An earlier version hand-rolled a lookalike instead of calling the shipped
    predicates, which silently covered three of the six and missed the rest.
    """
    from src.product.selection import preference_predicates

    d = _payload(year)
    f = d["filters"]
    art = json.loads((REPO / "artifacts" / "candidates" / f"candidates_{year}.json").read_text())
    preds = {k: fn for k, fn in preference_predicates(art).items() if k != "none"}
    keys = [p["key"] for p in sorted(f["predicates"], key=lambda p: p["i"])]
    assert keys == sorted(preds), "shipped predicate keys drifted from selection.py"

    for row in f["candidates"]:
        w = _decode(row["b"], d["first_round"])
        for p in f["predicates"]:
            expected = preds[p["key"]](w)
            assert (row["k"][p["i"]] == "1") == expected, (
                f"{year}: flag for {p['key']} disagrees with the predicate"
            )


@pytest.mark.parametrize("year", SEASONS)
def test_every_offered_axis_value_has_a_bracket(year):
    """A chip with nothing behind it would enable a filter that returns nothing."""
    d = _payload(year)
    f = d["filters"]
    rows = f["candidates"]
    for c in f["champions"]:
        assert any(r["c"] == c["team"] for r in rows), f"{year}: champion {c['name']} has no bracket"
    for o in f["ones"]:
        assert any(r["o"] == o for r in rows), f"{year}: one-seed count {o} has no bracket"
    for dp in f["depths"]:
        assert any(r["d"] == dp for r in rows), f"{year}: depth {dp} has no bracket"
    for s in f["sources"]:
        assert any(r["s"] == s for r in rows), f"{year}: source {s} has no bracket"
    for p in f["predicates"]:
        assert any(r["k"][p["i"]] == "1" for r in rows), f"{year}: predicate {p['key']} has no bracket"


@pytest.mark.parametrize("year", SEASONS)
def test_the_recommended_strategies_are_pool_members(year):
    """Council item 1: filtering must be able to reach what the product recommends.

    They were absent twice -- first because only region_top_n over the rating
    sources was added while the shipped rule uses the seed/no-seed blend, then
    because the source axis floor hid the three that were finally there.
    """
    d = _payload(year)
    f = d["filters"]
    fr = d["first_round"]

    def encode(picks):
        chosen = [set(r) for r in picks]
        bits, current = [], list(fr)
        for ri in range(6):
            nxt = []
            for g in range(0, len(current), 2):
                t1, t2 = current[g], current[g + 1]
                first = t1 in chosen[ri]
                bits.append("1" if first else "0")
                nxt.append(t1 if first else t2)
            current = nxt
        return "".join(bits)

    shipped = {r["b"] for r in f["candidates"]}
    for st in d["strategies"]:
        assert encode(st["picks"]) in shipped, f"{year}: '{st['label']}' is not in the pool"
    assert "shipped" in f["sources"], f"{year}: no source chip selects the recommended brackets"


@pytest.mark.parametrize("year", SEASONS)
def test_p1_standard_error_is_shipped(year):
    """The browser defines "near-tied" from this, not from a fixed count."""
    f = _payload(year)["filters"]
    assert isinstance(f.get("p1_se"), float) and 0 < f["p1_se"] < 0.05
