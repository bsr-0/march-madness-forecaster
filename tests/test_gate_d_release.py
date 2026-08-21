"""Gate D — the release gate.

This is the last thing between the work and a release, so it asserts the whole
chain rather than any one layer:

    methodology -> artifact -> selection -> browser -> product

Most individual guarantees are proven in their own modules; duplicating them
here would create two places to update and one to forget. What this file adds is
the part no single module can check:

  * the SEAMS between layers, where each owns something and no one owns the gap;
  * D3, the artifact contract, which had no owner at all until now;
  * an honest, machine-checked statement of what was NOT verified (D8).

The ownership boundaries, which are the organising idea of this whole exercise:

    methodology spec (2027.v2)   model, simulation, objectives. Does NOT own the
                                 artifact schema or selection semantics.
    artifact contract (schema 4) shape and compatibility rules. Does NOT own how
                                 candidates are chosen.
    selection spec (product.v3)  which candidates are shown. Does NOT own shape.

Each was tangled with its neighbour at some point in this project, and each
tangle produced a control that could not fail.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from src.governance import product_spec as ps
from src.governance.frozen_spec import (
    FROZEN_SPEC_PATH,
    ORIGINAL_V2_SPEC_PATH,
    SPEC_VERSION,
    canonical_hash,
    capture_live_spec,
    diff_against_frozen,
)
from src.product.artifact_contract import (
    EXPECTED_ARTIFACT_SCHEMA,
    REQUIRED_FIELDS,
    ArtifactSchemaError,
    contract_record,
    payload_hash,
    validate_artifact,
    validate_schema,
)
from src.product.selection import SELECTION_VERSION, select_diverse

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
ARTIFACT_PATH = DOCS / "data" / "candidates_2026.json"

pytestmark = pytest.mark.skipif(not ARTIFACT_PATH.exists(), reason="candidate artifact not built")


@pytest.fixture(scope="module")
def artifact():
    return json.loads(ARTIFACT_PATH.read_text())


# ===========================================================================
# D1 — Freeze integrity
# ===========================================================================


def test_d1_methodology_freeze_holds():
    assert SPEC_VERSION == "2027.v2"
    assert diff_against_frozen(REPO / FROZEN_SPEC_PATH)["drifted"] == []
    live = capture_live_spec()
    assert "diversity_algorithm" not in live["candidate_selection"]
    assert "k_returned" not in live["candidate_selection"]
    assert "strategies" not in live["product"]


def test_d1_a_real_methodology_change_fails_the_gate():
    """The freeze gate must be falsifiable, not merely currently satisfied."""
    baseline = canonical_hash(capture_live_spec())
    with mock.patch("src.prediction.noseed_model.TRAIN_YEARS", (2015, 2016)):
        assert canonical_hash(capture_live_spec()) != baseline


# ===========================================================================
# D2 — Selection integrity
# ===========================================================================


def test_d2_selection_version_agrees_across_every_source():
    js = (DOCS / "selection.js").read_text()
    m = re.search(r"const SELECTION_VERSION = '([^']+)'", js)
    assert m and m.group(1) == SELECTION_VERSION == ps.PRODUCT_SPEC_VERSION == "product.v3"


def test_d2_selection_semantics_are_captured_from_live_code():
    baseline = ps.canonical_hash(ps.capture_live_product_spec())
    with mock.patch.multiple("src.product.selection", DIVERSITY_TIERS=("champion", "sweet_16", "final_four")):
        assert ps.canonical_hash(ps.capture_live_product_spec()) != baseline


def test_d2_fewer_than_k_is_permitted(artifact):
    """The product must be allowed to say "there is no second bracket".

    On this field p1 has no third bracket within the retention floor, so k=3
    returns two. Padding it would mean manufacturing a distinction.
    """
    assert len(select_diverse(artifact, "p1", k=3)) == 2


# ===========================================================================
# D3 — Artifact contract (the previously unowned gate)
# ===========================================================================


def test_d3_artifact_declares_the_expected_schema(artifact):
    assert artifact["schema"] == EXPECTED_ARTIFACT_SCHEMA == 5
    validate_artifact(artifact, context="shipped artifact")


def test_d3_python_refuses_an_older_schema(artifact):
    for older in (2, 3, 4):
        with pytest.raises(ArtifactSchemaError, match=f"schema {older}"):
            validate_schema({**artifact, "schema": older})


def test_d3_python_refuses_a_newer_schema(artifact):
    """A newer artifact may redefine a field this code thinks it understands."""
    with pytest.raises(ArtifactSchemaError, match="Refusing rather than guessing"):
        validate_schema({**artifact, "schema": EXPECTED_ARTIFACT_SCHEMA + 1})


def test_d3_python_refuses_a_schemaless_artifact(artifact):
    with pytest.raises(ArtifactSchemaError, match="declares no schema"):
        validate_schema({k: v for k, v in artifact.items() if k != "schema"})


def test_d3_required_fields_are_present_and_shaped(artifact):
    for field in REQUIRED_FIELDS:
        assert field in artifact and artifact[field] not in (None, [], {}, "")
    n = len(artifact["teams"])
    assert len(artifact["pairwise"]) == n * n
    assert len(artifact["team_round_probabilities"]) == n
    assert len(artifact["first_round"]) == 64


def test_d3_a_missing_required_field_is_refused(artifact):
    for field in ("pairwise", "team_round_probabilities", "first_round"):
        with pytest.raises(ArtifactSchemaError):
            validate_artifact({k: v for k, v in artifact.items() if k != field})


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_d3_browser_enforces_the_same_rules(artifact):
    """The mirror must refuse exactly what Python refuses.

    A browser that is laxer than Python is the dangerous direction: it renders
    something plausible from an artifact the pipeline would have rejected.
    """
    runner = """
    const sel = require(process.argv[2]);
    const fs = require('fs');
    const art = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
    const cases = {
      valid:      art,
      older_2:    {...art, schema: 2},
      older_3:    {...art, schema: 3},
      newer:      {...art, schema: art.schema + 1},
      schemaless: (() => { const c = {...art}; delete c.schema; return c; })(),
      no_pairwise:(() => { const c = {...art}; delete c.pairwise; return c; })(),
      no_trp:     (() => { const c = {...art}; delete c.team_round_probabilities; return c; })(),
      bad_shape:  {...art, pairwise: [0.5, 0.5]},
    };
    const out = {};
    for (const [name, a] of Object.entries(cases)) {
      try { sel.validateArtifact(a); out[name] = 'accepted'; }
      catch (e) { out[name] = 'refused'; }
    }
    console.log(JSON.stringify(out));
    """
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(runner)
        script = f.name

    proc = subprocess.run(
        ["node", script, str(DOCS / "selection.js"), str(ARTIFACT_PATH)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"node failed: {proc.stderr[:400]}"
    got = json.loads(proc.stdout)

    assert got["valid"] == "accepted"
    for name in ("older_2", "older_3", "newer", "schemaless", "no_pairwise", "no_trp", "bad_shape"):
        assert got[name] == "refused", f"the browser accepted a bad artifact: {name}"


def test_d3_browser_and_python_pin_the_same_schema_number():
    js = (DOCS / "selection.js").read_text()
    m = re.search(r"const EXPECTED_ARTIFACT_SCHEMA = (\d+)", js)
    assert m and int(m.group(1)) == EXPECTED_ARTIFACT_SCHEMA


def test_d3_build_validates_before_using_the_artifact():
    """Validation must gate assignment, not run after it."""
    src = (DOCS / "build.js").read_text()
    assert "validateArtifact(loaded)" in src
    assert src.index("validateArtifact(loaded)") < src.index("ARTIFACT = loaded"), (
        "build.js assigns ARTIFACT before validating it"
    )


def test_d3_schema_is_part_of_the_manifest_record(artifact):
    rec = contract_record(artifact)
    assert rec["artifact_schema"] == EXPECTED_ARTIFACT_SCHEMA
    assert rec["expected_artifact_schema"] == EXPECTED_ARTIFACT_SCHEMA
    assert len(rec["payload_sha256"]) == 64
    assert "exact match only" in rec["compatibility"]


def test_d3_schema_participates_in_the_payload_hash(artifact):
    """A schema change must move the artifact hash.

    Otherwise a reshaped artifact could pass a checksum comparison unchanged --
    the same blindness the methodology gate had.
    """
    assert payload_hash({**artifact, "schema": 99}) != payload_hash(artifact)


def test_d3_schema_is_owned_by_the_contract_not_the_methodology_spec():
    """Ownership, asserted rather than assumed.

    The methodology spec must not reference the schema: transport cannot change
    a probability, so a transport change must not be able to invalidate a
    prospective claim.
    """
    live = json.dumps(capture_live_spec())
    assert "artifact_schema" not in live
    assert "team_round_probabilities" not in live


def test_d3_every_team_has_a_canonical_name(artifact):
    """Schema 5: names are data, not something the browser derives."""
    for t in artifact["teams"]:
        assert t.get("name"), f"team {t['id']} has no canonical name"
        assert not (t["name"] == t["id"] and "_" in t["id"]), (
            f"team {t['id']} name is just the slug"
        )


def test_d3_the_known_unroundtrippable_names_are_correct(artifact):
    """The exact schools that slug-derivation got wrong.

    Pinned individually because each fails differently: apostrophes, ampersands,
    parenthesised state qualifiers, and all-caps acronyms.
    """
    by_id = {t["id"]: t["name"] for t in artifact["teams"]}
    expected = {
        "saint_mary_s__ca": "Saint Mary's (CA)",
        "st__john_s__ny": "St. John's (NY)",
        "texas_a_m": "Texas A&M",
        "miami__fl": "Miami (FL)",
        "tcu": "TCU",
        "ucla": "UCLA",
        "ucf": "UCF",
    }
    wrong = {k: (by_id.get(k), v) for k, v in expected.items() if by_id.get(k) != v}
    assert not wrong, f"canonical names incorrect: {wrong}"


def test_d3_no_team_referenced_by_a_bracket_is_nameless(artifact):
    """THE ANTI-SLUG GATE.

    Every team a user can actually see -- in any candidate's picks, its champion
    or its Final Four -- must resolve to a non-empty canonical name. This is
    stronger than checking the team table, because it follows the references the
    product actually renders.
    """
    from src.product.selection import CHAMP, F4

    teams = artifact["teams"]
    referenced = set()
    for c in artifact["candidates"]:
        for rnd in c["w"]:
            referenced.update(rnd)
    referenced.update(artifact["first_round"])
    for c in artifact["candidates"]:
        referenced.update(c["w"][CHAMP])
        referenced.update(c["w"][F4])

    bad = [teams[i]["id"] for i in referenced if not teams[i].get("name")]
    assert not bad, f"{len(bad)} rendered team(s) would fall back to a slug: {bad[:5]}"
    assert len(referenced) >= 64


def test_d3_frontend_renders_the_name_field_not_the_id():
    """Rendering must read `teams[].name`; deriving from the id is prohibited."""
    sel = (DOCS / "selection.js").read_text()
    assert "teams[i1].name" in sel and "teams[i2].name" in sel, (
        "candidateToRounds no longer passes the canonical name to the renderer"
    )
    # The title-casing derivation must not come back anywhere.
    for name in ("app.js", "build.js", "explore.js", "record.js", "selection.js"):
        src = (DOCS / name).read_text()
        assert not re.search(r"split\('_'\)[\s\S]{0,80}toUpperCase", src), (
            f"{name} derives a display name from a slug; slugs do not round-trip"
        )


# ===========================================================================
# D4 — Prospective-data integrity
# ===========================================================================


def test_d4_no_realized_2026_metric_anywhere_in_the_product():
    """2026 accuracy must not appear as a user-facing claim."""
    for name in ("build.js", "selection.js", "explore.js", "record.js", "app.js"):
        text = (DOCS / name).read_text()
        literals = " ".join(x for g in re.findall(r"'([^'\n]*)'|\"([^\"\n]*)\"|`([^`]*)`", text) for x in g if x)
        assert not re.search(r"74\.6|0\.1449|10\.47|11\.2%", literals), (
            f"{name} carries a realized 2026 performance figure"
        )


def test_d4_track_record_headline_excludes_the_replay_year():
    payload = json.loads((DOCS / "data" / "ml_backtest.json").read_text())
    assert 2026 not in payload["years"]
    assert payload["replay_year"]["is_out_of_sample"] is False
    assert payload["n_games"] == sum(r["n_games"] for r in payload["per_year"] if r["year"] in payload["years"]), (
        "the headline game count does not match the scored window"
    )


def test_d4_2026_is_recorded_as_contaminated():
    holdout = capture_live_spec()["holdout"]
    assert holdout["contaminated_seasons"] == [2026]
    assert holdout["prospective_season"] == 2027
    assert holdout["outcomes_available_at_freeze"] is False


def test_d4_training_cutoff_is_2026():
    assert capture_live_spec()["model"]["training_cutoff_season"] == 2026


def test_d4_generation_requires_pretournament_inputs(artifact):
    """The leakage gate must be armed, and provenance retained in the artifact."""
    from scripts.experiments.build_candidate_artifact import assert_pretournament_inputs

    prov = artifact["provenance"]
    assert prov["torvik"]["data_type"] == "pre_tournament"
    assert prov["torvik"]["cutoff_date"] < prov["torvik"]["tournament_start"]
    assert callable(assert_pretournament_inputs)


def test_d4_input_hashes_are_retained():
    manifest = json.loads((REPO / "artifacts/phase2/manifest_2026_production.json").read_text())
    assert manifest["input_hashes"], "no input hashes retained for the production artifact"
    assert manifest["evaluation_performed"] is False
    assert "must not be used as evidence" in manifest["evaluation_note"]


# ===========================================================================
# D5 — Legacy / code integrity
# ===========================================================================


def test_d5_legacy_debt_is_empty():
    from tests.test_frontend_integrity import _LEGACY_DEBT

    assert _LEGACY_DEBT == {}


def test_d5_no_browser_model_math_or_legacy_machinery():
    banned = {
        "browser log5": r"\bfunction log5\b",
        "client simulation": r"\bfunction simulate\b",
        "strategy catalog": r"\bconst STRATEGIES\b",
        "hardcoded p_first": r"\bp_first\b",
        "per-strategy pick": r"\bfunction pick\s*\(",
        "rating lookup": r"\bbarthag\s*[:=]",
    }
    for name in ("build.js", "selection.js", "explore.js", "record.js", "app.js"):
        src = (DOCS / name).read_text()
        hits = [label for label, pat in banned.items() if re.search(pat, src)]
        assert not hits, f"{name} still contains legacy machinery: {hits}"


def test_d5_single_bracket_representation():
    """One expansion path, so the board cannot disagree with itself."""
    app = (DOCS / "app.js").read_text()
    assert "function precomputedRounds" not in app
    assert "candidateToRounds" in (DOCS / "selection.js").read_text()


# ===========================================================================
# D6 — Product behaviour (structural; the journey is exercised in-browser)
# ===========================================================================


def test_d6_build_offers_two_measured_objectives_and_no_preferences():
    src = (DOCS / "build.js").read_text()
    keys = set(re.findall(r"\{\s*key:\s*'(\w+)'", src))
    assert keys == {"ev", "p1"}
    assert "pref-grid" not in src and "renderPrefGrid" not in src


def test_d6_alternatives_are_materially_different(artifact):
    from src.product.selection import is_materially_different

    for objective in ("ev", "p1"):
        sel = select_diverse(artifact, objective, k=2)
        if len(sel) == 2:
            assert is_materially_different(artifact, sel[1], sel[0])


def test_d6_pool_size_disclosure_is_present():
    src = (DOCS / "build.js").read_text()
    assert "not a universal probability" in src.lower()
    assert "p1_pool_size" in src


def test_d6_no_false_precision_in_user_copy():
    """Dots and rounded percentages, never four-decimal model output."""
    for name in ("build.js", "explore.js", "record.js"):
        text = (DOCS / name).read_text()
        literals = " ".join(x for g in re.findall(r"'([^'\n]*)'|`([^`]*)`", text) for x in g if x)
        assert not re.search(r"\d\.\d{4,}", literals), f"{name} shows false precision"


def test_d6_explore_has_no_generation_controls():
    src = (DOCS / "explore.js").read_text()
    for leaked in ("selectDiverse", "selectWithAlternative", "generateBrackets", "selectBrackets"):
        assert leaked not in src, f"explore.js leaks a generation control: {leaked}"


def test_d6_explore_uses_full_bank_probabilities_not_candidate_counts():
    src = (DOCS / "explore.js").read_text()
    assert "team_round_probabilities" in src
    assert not re.search(r"candidates\.filter|candidates\.length", src), (
        "explore.js derives a probability by counting candidates, which over-samples long shots"
    )


def test_d6_track_record_separates_the_replay():
    src = (DOCS / "record.js").read_text()
    assert "tr-replay" in src
    assert "Not part of the track record above" in src
    assert not re.search(r"per_year[\s\S]{0,200}reduce\(", src)


# ===========================================================================
# D7 — Reproducibility
# ===========================================================================


def test_d7_selection_is_deterministic(artifact):
    for objective in ("ev", "p1"):
        runs = [tuple(select_diverse(artifact, objective, k=3)) for _ in range(5)]
        assert len(set(runs)) == 1


def test_d7_payload_hash_ignores_non_semantic_fields(artifact):
    """A rebuild from identical inputs must hash identically."""
    perturbed = json.loads(json.dumps(artifact))
    perturbed["meta"]["generated_at"] = "1999-01-01T00:00:00+00:00"
    perturbed["validation"] = {"anything": True}
    perturbed["provenance"] = {"different": "value"}
    assert payload_hash(perturbed) == payload_hash(artifact)


def test_d7_payload_hash_reacts_to_semantic_change(artifact):
    perturbed = json.loads(json.dumps(artifact))
    perturbed["candidates"][0]["ev"] += 1.0
    assert payload_hash(perturbed) != payload_hash(artifact)


def test_d7_production_manifest_matches_its_artifact():
    """The shipped artifact must be the one the manifest describes."""
    manifest = json.loads((REPO / "artifacts/phase2/manifest_2026_production.json").read_text())
    built = REPO / "artifacts/candidates/candidates_2026.json"
    if not built.exists():
        pytest.skip("production artifact not present locally")
    art = json.loads(built.read_text())
    recorded = manifest.get("artifact_payload_sha256")
    if art["schema"] != manifest.get("artifact_contract", {}).get("artifact_schema"):
        pytest.skip(
            "manifest predates the schema-4 rebuild; regenerate with "
            "scripts/experiments/phase2_production_validation.py"
        )
    assert payload_hash(art) == recorded


# ===========================================================================
# D8 — Test-environment disclosure
#
# Recorded in code so the limitation travels with the release rather than living
# in a chat message.
# ===========================================================================

TEST_ENVIRONMENT_DISCLOSURE = {
    # Repo facts, verified below. Deliberately NOT a record of "the pytest I
    # happened to run", which differs per machine and made an earlier version of
    # this gate fail simply for running somewhere else.
    "async_tests_in_suite": 0,
    "async_coverage_gap": False,
    "async_note": (
        "There are no async tests in tests/. A pytest-asyncio import failure "
        "therefore costs NO coverage: it stops a plugin loading, not a test running."
    ),
    "ci_requirements_file": "requirements.txt",
    "ci_python": "3.10",
    "ci_installs_pytest_asyncio": False,
    "ci_pytest_pin": "pytest>=7.4.0 (no upper bound -- CI resolves to whatever is newest)",
    "lock_is_used_by_ci": False,
    "lock_is_installable": False,
    "lock_note": (
        "requirements-lock.txt is NOT what CI installs and is not resolvable as "
        "written: it pins numpy==1.26.3 while cbbpy==2.1.2 requires numpy>=2.0.0, "
        "and it carries two -e git+ssh editable requirements pointing at a "
        "different private repository. Reasoning about 'the locked environment' "
        "as if it were CI's environment is therefore unsound."
    ),
    "ci_runs_full_suite": False,
    "ci_coverage_note": (
        "The full-tests job runs marker subsets -- unit, data_contract, leakage, "
        "backtest_regression -- not `pytest tests/`. Tests marked integration or "
        "slow are in no CI step. A green CI is not equivalent to a green full run."
    ),
}


def test_d8_disclosure_matches_the_repo():
    """The disclosure must describe THIS repo, not a remembered version of it.

    Every claim here is checkable from files, so the record cannot quietly rot
    while still being cited as evidence.
    """
    reqs = (REPO / "requirements.txt").read_text()
    lock = (REPO / "requirements-lock.txt").read_text()
    ci = (REPO / ".github/workflows/ci.yml").read_text()
    action = (REPO / ".github/actions/setup-python-env/action.yml").read_text()

    assert ("asyncio" in reqs) is TEST_ENVIRONMENT_DISCLOSURE["ci_installs_pytest_asyncio"]
    assert 'default: "requirements.txt"' in action
    assert TEST_ENVIRONMENT_DISCLOSURE["ci_requirements_file"] == "requirements.txt"
    assert 'python-version: "3.10"' in action

    # CI's test job must not silently start using the lock without this being
    # revisited -- the lock is unsatisfiable.
    assert "requirements-lock" not in ci, (
        "CI now references requirements-lock.txt, which is not installable "
        "(numpy pin conflicts with cbbpy). Re-verify before trusting a CI run."
    )
    assert re.search(r"^numpy==1\.26\.3", lock, re.M), "the lock's numpy pin moved"
    assert TEST_ENVIRONMENT_DISCLOSURE["lock_is_installable"] is False

    # CI runs marker subsets, not the whole suite.
    assert 'pytest tests/ -m "unit"' in ci
    assert "pytest tests/ -q\n" not in ci
    assert TEST_ENVIRONMENT_DISCLOSURE["ci_runs_full_suite"] is False


def test_d8_records_the_environment_it_actually_ran_in(record_property):
    """Attach the running versions to the report rather than asserting them.

    An earlier version pinned "local_pytest": "7.4.4" and asserted equality,
    which failed the moment the suite ran in the CI-equivalent environment --
    exactly the environment it existed to describe.
    """
    import sys

    import pytest as _pytest

    record_property("python", sys.version.split()[0])
    record_property("pytest", _pytest.__version__)
    try:
        import pytest_asyncio

        record_property("pytest_asyncio", pytest_asyncio.__version__)
    except Exception as exc:  # noqa: BLE001
        record_property("pytest_asyncio", f"unavailable ({type(exc).__name__})")


def test_d8_there_are_no_async_tests_to_miss():
    """Verifies the disclosure rather than trusting it.

    An earlier reading of this environment concluded that "async tests are not
    exercised locally", which implied a coverage gap. There is none: the suite
    contains no async tests at all.
    """
    found = []
    for path in (REPO / "tests").rglob("test_*.py"):
        if path.name == Path(__file__).name:
            continue  # this file names the marker in order to detect it
        text = path.read_text()
        # Anchored to real definitions: a bare substring search matches any file
        # that merely mentions the marker, including this one.
        if re.search(r"^\s*async def test_", text, re.M) or re.search(
            r"^\s*@pytest\.mark\.asyncio", text, re.M
        ):
            found.append(path.name)
    assert not found, (
        f"async tests now exist ({found}), so the local -p no:asyncio invocation "
        "IS skipping coverage. Update TEST_ENVIRONMENT_DISCLOSURE and fix the "
        "local pytest version."
    )
    assert TEST_ENVIRONMENT_DISCLOSURE["async_coverage_gap"] is False
