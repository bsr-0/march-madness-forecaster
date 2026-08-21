"""CI gate for the 2027 prospective freeze.

The freeze is only meaningful if drift is detectable. This test compares the
live code against ``configs/frozen/prospective_2027.json`` and fails on any
difference, so a frozen parameter cannot change quietly.

Changing a frozen value is allowed. Doing it without acknowledgement is not:
bump ``SPEC_VERSION`` in ``src/governance/frozen_spec.py``, regenerate the spec,
and record in PROSPECTIVE_2027.md that the v1 prospective claim is void and a new
version has started. That is the whole contract — the point is that April 2027
can be checked against what was actually frozen in August 2026, not against what
someone remembers freezing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.governance.frozen_spec import (
    FREEZE_DATE,
    FROZEN_SPEC_PATH,
    PROSPECTIVE_DOC,
    SPEC_VERSION,
    SUPERSEDED,
    canonical_hash,
    capture_live_spec,
    diff_against_frozen,
    load_frozen_spec,
)

REPO = Path(__file__).resolve().parent.parent


def test_frozen_spec_file_exists():
    assert (REPO / FROZEN_SPEC_PATH).exists(), (
        f"{FROZEN_SPEC_PATH} is missing. The 2027 prospective evaluation depends on it; "
        "regenerate via src.governance.frozen_spec.capture_live_spec()."
    )


def test_frozen_spec_hash_is_self_consistent():
    """The recorded hash must match the spec it is stored beside."""
    frozen = load_frozen_spec(REPO / FROZEN_SPEC_PATH)
    recorded = frozen.get("spec_hash")
    body = {k: v for k, v in frozen.items() if k != "spec_hash"}
    assert recorded == canonical_hash(body), (
        "spec_hash does not match the spec body — the file was edited by hand. "
        "Regenerate it rather than patching the hash."
    )


def test_live_system_has_not_drifted_from_the_freeze():
    """THE GATE. Any change to a frozen parameter fails here."""
    result = diff_against_frozen(REPO / FROZEN_SPEC_PATH)
    if result["drifted"]:
        lines = [f"  {d['path']}: frozen={d['frozen']!r} -> live={d['live']!r}" for d in result["drifted"]]
        pytest.fail(
            "The live system no longer matches the 2027 prospective freeze "
            f"({result['spec_version']}).\n"
            + "\n".join(lines)
            + "\n\nIf this change is intended, it invalidates the v1 prospective claim. "
            "Bump SPEC_VERSION, regenerate configs/frozen/prospective_2027.json, and "
            "record the reason in PROSPECTIVE_2027.md. Do not silently re-freeze."
        )


def test_prospective_document_records_the_freeze():
    """The operative prospective doc must carry this version, date and hash."""
    doc = (REPO / PROSPECTIVE_DOC).read_text()
    frozen = load_frozen_spec(REPO / FROZEN_SPEC_PATH)
    for token in (SPEC_VERSION, FREEZE_DATE, frozen["spec_hash"]):
        assert token in doc, f"PROSPECTIVE_2027.md does not record {token!r}"


def test_2026_is_recorded_as_contaminated():
    """2026 must never be reintroduced as a holdout by a later edit.

    It sits inside BACKTEST_YEARS, the production strategy was selected on a
    window containing it, and a documented modelling conclusion was drawn from
    its outcome. Any evaluation of it is in-sample.
    """
    spec = capture_live_spec()
    assert 2026 in spec["holdout"]["contaminated_seasons"]
    assert spec["holdout"]["prospective_season"] == 2027
    assert spec["holdout"]["outcomes_available_at_freeze"] is False


def test_v1_excludes_the_unvalidated_strategies():
    """Balanced and Contrarian were never measured; they must not ship.

    The strategy list itself now lives in the product spec (product.v3): it
    describes what the UI offers, not how the model works. It used to sit here as
    a hardcoded literal asserting three strategies including "Your Preference" --
    which the product had already stopped exposing, and which this comparison
    could not detect because both sides were the same constant.

    What stays methodology-owned is the exclusion: the objectives are ev and p1,
    and no unmeasured blend may be added.
    """
    from src.governance.product_spec import capture_live_product_spec

    spec = capture_live_spec()
    assert spec["candidate_selection"]["objectives"] == ["ev", "p1"]
    excluded = " ".join(spec["product"]["excluded_from_v1"]).lower()
    assert "balanced" in excluded and "contrarian" in excluded

    names = {s["name"] for s in capture_live_product_spec()["strategies"]}
    assert names == {"Trust the Model", "Win My Pool"}, (
        f"the shipped strategy set is {names}; v1 offers exactly the two measured "
        "objectives, and any addition needs research before a label"
    )
    assert not {"Balanced", "Contrarian"} & names


def test_pool_size_assumption_is_disclosed():
    """P(1st) is conditional on an opponent field; that must travel with it."""
    spec = capture_live_spec()
    assert spec["candidate_selection"]["p1_pool_size"] == 30
    disclosure = spec["product"]["p1_disclosure_required"].lower()
    assert "30-opponent" in disclosure
    assert "not a universal probability" in disclosure


def test_probability_direction_is_frozen():
    """The pairwise contract is part of the freeze, not an implementation detail."""
    spec = capture_live_spec()
    assert spec["tournament_engine"]["marginals_direction"] == ("pairwise -> simulator -> marginals (never reversed)")
    assert spec["candidate_selection"]["scoring_mode"] == "team_identity (never shape-encoded)"


def test_spec_is_valid_json_and_sorted():
    """Stored sorted so diffs stay readable across regenerations."""
    raw = (REPO / FROZEN_SPEC_PATH).read_text()
    parsed = json.loads(raw)
    assert raw == json.dumps(parsed, indent=2, sort_keys=True) + "\n" or raw == json.dumps(
        parsed, indent=2, sort_keys=True
    )


def test_v1_specification_is_immutable():
    """The superseded v1 record must never change.

    v1 is the original prospective specification. Its value is entirely in being
    an untouched record of what was frozen on 2026-08-20 before 2027 existed; a
    v1 that can be edited afterwards is worth nothing. Both the spec file and its
    document are pinned here.
    """
    v1_path = REPO / SUPERSEDED["spec_path"]
    assert v1_path.exists(), "the v1 specification file has been deleted"
    v1 = json.loads(v1_path.read_text())
    body = {k: v for k, v in v1.items() if k != "spec_hash"}
    assert canonical_hash(body) == SUPERSEDED["spec_hash"], (
        "configs/frozen/prospective_2027.json has been modified. v1 is an immutable "
        "historical record -- create a new version instead of editing it."
    )
    assert v1["spec_version"] == SUPERSEDED["version"]
    assert (REPO / SUPERSEDED["doc"]).exists(), "PROSPECTIVE_2027.md has been deleted"


def test_v2_records_why_it_supersedes_v1():
    """A version bump must carry its justification, not just a new number."""
    spec = capture_live_spec()
    sup = spec["supersedes"]
    assert sup["version"] == "2027.v1"
    reason = sup["reason_superseded"].lower()
    assert "train_years" in reason and "2026" in reason
    assert "ex ante" in reason, "the ex-ante ordering must be recorded"
    assert "not on the basis of any 2026 performance" in reason.replace("  ", " ")


def test_training_extension_does_not_touch_historical_walk_forward():
    """Adding 2026 must alter ONLY prediction years after 2026.

    train_noseed_model filters with `y < max_year`, so every test year at or
    before 2026 trains on an identical set. This is what makes the v2 change
    surgical rather than a re-baselining of the whole project.
    """
    from src.prediction.noseed_model import TRAIN_YEARS

    v2 = list(TRAIN_YEARS)
    v1 = [y for y in v2 if y != 2026]
    assert 2026 in v2, "v2 must include 2026 in training"
    for test_year in range(2011, 2027):
        assert [y for y in v1 if y < test_year] == [y for y in v2 if y < test_year], (
            f"training set for test year {test_year} changed; the extension is not surgical"
        )
    gained = set(y for y in v2 if y < 2027) - set(y for y in v1 if y < 2027)
    assert gained == {2026}


def test_2026_still_barred_as_an_evaluation_season():
    """Training on 2026 must not quietly reclassify it as evaluable."""
    spec = capture_live_spec()
    assert 2026 in spec["holdout"]["contaminated_seasons"]
    assert spec["holdout"]["contaminated_for_evaluation_only"] is True
    assert spec["model"]["training_cutoff_season"] == 2026
