"""The specification boundary: methodology vs product selection.

THE BUG THIS EXISTS TO PREVENT
------------------------------
``candidate_selection.diversity_algorithm``, ``candidate_selection.k_returned``
and ``product.strategies`` were recorded in the methodology spec as hardcoded
string literals, and ``capture_live_spec()`` emitted the same literals. The
drift gate therefore compared a constant to itself. It reported "no drift" while
all three had stopped describing the product — the product had moved to
product.v2 tiered selection, was returning k=2, and had stopped exposing "Your
Preference".

An integrity control that cannot fail is worse than none, because it earns trust
it has not paid for. So the tests below are deliberately *mutation* tests: each
perturbs the live implementation and asserts the relevant hash actually moves.
A spec that survives every mutation is a spec that is not watching anything.

THE BOUNDARY
------------
    methodology (2027.v2)   determines the model, the simulation and the
                            candidate artifact — and therefore the prospective
                            claim.
    selection (product.vN)  turns the already-produced candidates into the
                            brackets on screen. Cannot move a probability, an
                            expected score or a P(1st).

The methodology hash changed when the boundary was corrected. That is NOT a
methodology change, and ``test_methodology_values_are_unchanged_by_the_boundary_correction``
proves it field-by-field against the original immutable file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest import mock

import pytest

from src.governance import product_spec as ps
from src.governance.frozen_spec import (
    FROZEN_SPEC_PATH,
    ORIGINAL_V2_SPEC_PATH,
    SCOPE_CORRECTION,
    SPEC_VERSION,
    canonical_hash,
    capture_live_spec,
    diff_against_frozen,
    load_frozen_spec,
)

REPO = Path(__file__).resolve().parent.parent

# Fields moved out of the methodology spec by the boundary correction.
MOVED_FIELDS = {
    ("candidate_selection", "diversity_algorithm"),
    ("candidate_selection", "k_returned"),
    ("product", "strategies"),
}


@pytest.fixture(scope="module")
def original_v2():
    return json.loads((REPO / ORIGINAL_V2_SPEC_PATH).read_text())


@pytest.fixture(scope="module")
def scoped_v2():
    return load_frozen_spec(REPO / FROZEN_SPEC_PATH)


# ---------------------------------------------------------------------------
# The boundary itself
# ---------------------------------------------------------------------------


def test_original_v2_specification_is_immutable(original_v2):
    """The prospective record is never rewritten, only superseded in scope."""
    # Byte digest of the file as frozen. Pinned so that ANY edit fails here,
    # including one that keeps the JSON semantically equal.
    digest = hashlib.sha256((REPO / ORIGINAL_V2_SPEC_PATH).read_bytes()).hexdigest()
    assert digest == "9143f9ee0aa7cc90442b9547c670f16cd62ca91b5258cceb6f259376401d172b", (
        "the original 2027.v2 file changed on disk. It is the prospective record "
        "and must stay byte-identical; supersede its scope instead of editing it."
    )
    body = {k: v for k, v in original_v2.items() if k != "spec_hash"}
    assert canonical_hash(body) == original_v2["spec_hash"], (
        "the original 2027.v2 file no longer hashes to its recorded spec_hash — "
        "it was edited. It must remain byte-identical."
    )


def test_methodology_spec_no_longer_carries_selection_fields(scoped_v2):
    assert "diversity_algorithm" not in scoped_v2["candidate_selection"]
    assert "k_returned" not in scoped_v2["candidate_selection"]
    assert "strategies" not in scoped_v2["product"]


def test_product_spec_carries_them_instead():
    live = ps.capture_live_product_spec()
    assert live["selection"]["diversity_tiers"]
    assert isinstance(live["selection"]["k_returned_by_build"], int)
    assert live["strategies"]


def test_methodology_values_are_unchanged_by_the_boundary_correction(original_v2):
    """THE AUDIT-TRAIL PROOF.

    Every methodology field that survived the correction must hold exactly the
    value it held in the original. If this passes, the hash change is attributable
    solely to removing the three selection-owned fields — not to any methodology
    edit smuggled in alongside.
    """
    live = capture_live_spec()
    diffs = []

    def walk(a, b, trail):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in a:
                if len(trail) == 1 and (trail[0], key) in MOVED_FIELDS:
                    continue
                if key in ("spec_hash", "scope_correction"):
                    continue
                walk(a[key], b.get(key), trail + [key])
        elif a != b:
            diffs.append({"path": ".".join(trail), "original": a, "live": b})

    walk({k: v for k, v in original_v2.items() if k != "spec_hash"}, live, [])
    assert not diffs, (
        "methodology values changed during a correction that was supposed to move "
        f"only the specification boundary: {diffs}"
    )


def test_scope_correction_is_recorded_in_the_spec(scoped_v2):
    """The audit trail must be in the artifact, not only in a commit message."""
    sc = scoped_v2["scope_correction"]
    assert sc["methodology_unchanged"] is True
    assert set(sc["fields_moved"]) == {
        "candidate_selection.diversity_algorithm",
        "candidate_selection.k_returned",
        "product.strategies",
    }
    assert "boundary" in sc["reason"].lower()
    assert scoped_v2["spec_version"] == SPEC_VERSION == "2027.v2", (
        "the methodology version must NOT be bumped: the methodology did not change"
    )


def test_product_version_was_bumped():
    assert ps.PRODUCT_SPEC_VERSION == "product.v3"
    assert ps.RECLASSIFIED_FROM["methodology_unchanged"] is True


# ---------------------------------------------------------------------------
# Both gates are self-consistent and currently clean
# ---------------------------------------------------------------------------


def test_both_pinned_specs_hash_to_their_bodies():
    for path, mod in ((FROZEN_SPEC_PATH, None), (ps.PRODUCT_SPEC_PATH, ps)):
        spec = json.loads((REPO / path).read_text())
        body = {k: v for k, v in spec.items() if k != "spec_hash"}
        hasher = ps.canonical_hash if mod else canonical_hash
        assert hasher(body) == spec["spec_hash"], f"{path} was hand-edited"


def test_no_drift_in_either_gate():
    assert diff_against_frozen(REPO / FROZEN_SPEC_PATH)["drifted"] == []
    assert ps.diff_against_product_spec(REPO / ps.PRODUCT_SPEC_PATH)["drifted"] == []


# ---------------------------------------------------------------------------
# MUTATION TESTS — the gates must actually be able to fail
# ---------------------------------------------------------------------------


def _product_hash_under(**attrs):
    with mock.patch.multiple("src.product.selection", **attrs):
        return ps.canonical_hash(ps.capture_live_product_spec())


def test_changing_the_diversity_tier_order_moves_the_product_hash():
    """The exact field that used to be a dead literal."""
    baseline = ps.canonical_hash(ps.capture_live_product_spec())
    mutated = _product_hash_under(DIVERSITY_TIERS=("champion", "final_four", "sweet_16"))
    assert mutated != baseline, (
        "reordering the diversity tiers did not move the product spec hash. The "
        "field is not being derived from the implementation — this is the exact "
        "defect the boundary correction was made to fix."
    )


def test_changing_a_diversity_threshold_moves_the_product_hash():
    baseline = ps.canonical_hash(ps.capture_live_product_spec())
    assert _product_hash_under(MIN_S16_CHANGES=3) != baseline
    assert _product_hash_under(MIN_F4_CHANGES=2) != baseline
    assert _product_hash_under(DEFAULT_MIN_RETENTION=0.90) != baseline


def test_changing_the_selection_version_moves_the_product_hash():
    baseline = ps.canonical_hash(ps.capture_live_product_spec())
    assert _product_hash_under(SELECTION_VERSION="product.v99") != baseline


def test_changing_k_in_the_shipped_build_flow_moves_the_product_hash():
    """k is read from docs/build.js, so shipping a different k is detectable."""
    baseline = ps.canonical_hash(ps.capture_live_product_spec())
    with mock.patch.object(ps, "_build_flow_k", return_value=3):
        assert ps.canonical_hash(ps.capture_live_product_spec()) != baseline


def test_changing_the_strategy_list_moves_the_product_hash():
    baseline = ps.canonical_hash(ps.capture_live_product_spec())
    fake = [
        {"name": "Trust the Model", "objective": "ev", "constraint": None},
        {"name": "Win My Pool", "objective": "p1", "constraint": None},
        {"name": "Your Preference", "objective": "user-selected", "constraint": "user-selected"},
    ]
    with mock.patch.object(ps, "_live_strategies", return_value=fake):
        assert ps.canonical_hash(ps.capture_live_product_spec()) != baseline, (
            "re-adding 'Your Preference' did not move the hash — the strategy list "
            "is being transcribed rather than read from the shipped UI"
        )


def test_a_methodology_change_moves_the_methodology_hash():
    """The methodology gate must be falsifiable too."""
    baseline = canonical_hash(capture_live_spec())
    with mock.patch("src.prediction.noseed_model.TRAIN_YEARS", (2015, 2016, 2017)):
        assert canonical_hash(capture_live_spec()) != baseline


def test_selection_changes_do_not_move_the_methodology_hash():
    """The boundary holds in the other direction.

    Selection is not methodology: retiering diversity must leave the prospective
    claim untouched, otherwise every product tweak would invalidate the freeze.
    """
    baseline = canonical_hash(capture_live_spec())
    with mock.patch.multiple(
        "src.product.selection",
        DIVERSITY_TIERS=("champion",),
        MIN_S16_CHANGES=9,
        SELECTION_VERSION="product.v99",
    ):
        assert canonical_hash(capture_live_spec()) == baseline


def test_the_product_spec_refuses_to_guess():
    """If build.js becomes unparseable, fail loudly rather than pin a guess."""
    with mock.patch.object(ps.Path, "read_text", return_value="// nothing here"):
        with pytest.raises(RuntimeError):
            ps._build_flow_k()
        with pytest.raises(RuntimeError):
            ps._live_strategies()


def test_javascript_mirror_declares_the_same_selection_version():
    """A version split between Python and the shipped mirror is the same class of
    bug as the literal-transcribed spec: two sources of truth, silently diverging.
    """
    import re

    js = (REPO / "docs" / "selection.js").read_text()
    m = re.search(r"const SELECTION_VERSION = '([^']+)'", js)
    assert m, "docs/selection.js no longer declares SELECTION_VERSION"

    from src.product.selection import SELECTION_VERSION

    assert m.group(1) == SELECTION_VERSION == ps.PRODUCT_SPEC_VERSION, (
        f"selection version disagrees across sources: js={m.group(1)}, "
        f"python={SELECTION_VERSION}, spec={ps.PRODUCT_SPEC_VERSION}"
    )


def test_the_version_bump_did_not_change_selected_brackets():
    """product.v3 re-specifies v2; it must not move a single bracket.

    If this ever fails, the bump stopped being a specification change and became
    a behaviour change, which needs its own evidence and fixture review.
    """
    fixture = json.loads((REPO / "tests" / "fixtures" / "product" / "parity_2026.json").read_text())
    assert fixture["selection_version"] == ps.PRODUCT_SPEC_VERSION
    expected = {
        ("ev", 1): [2814], ("ev", 2): [2814, 914], ("ev", 3): [2814, 914, 2785],
        ("p1", 1): [1216], ("p1", 2): [1216, 1540], ("p1", 3): [1216, 1540],
    }
    for case in fixture["diverse_cases"]:
        key = (case["objective"], case["k"])
        assert case["expected_indices"] == expected[key], (
            f"{key} moved from {expected[key]} to {case['expected_indices']}; the "
            "v2->v3 bump was supposed to be specification-only"
        )
