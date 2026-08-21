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
    SPEC_VERSION,
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
    """PROSPECTIVE_2027.md must carry the same version, date and hash."""
    doc = (REPO / "PROSPECTIVE_2027.md").read_text()
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
    """Balanced and Contrarian were never measured; they must not ship in v1."""
    spec = capture_live_spec()
    names = {s["name"] for s in spec["product"]["strategies"]}
    assert names == {"Trust the Model", "Win My Pool", "Your Preference"}
    assert spec["candidate_selection"]["objectives"] == ["ev", "p1"]
    excluded = " ".join(spec["product"]["excluded_from_v1"]).lower()
    assert "balanced" in excluded and "contrarian" in excluded


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
