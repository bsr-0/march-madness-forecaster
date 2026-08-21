"""The candidate artifact contract — schema version and compatibility rules.

OWNERSHIP, STATED EXPLICITLY
---------------------------
This is the same separation that fixed the freeze gate, applied one layer down:

    methodology spec (2027.v2)   does NOT own the schema version. How data is
                                 transported cannot change a probability, and a
                                 transport change must not be able to invalidate
                                 a prospective claim.
    artifact contract (here)     owns the schema version and the compatibility
                                 rules. This is the only place that decides what
                                 the product will load.
    selection spec (product.v3)  owns selection semantics. It consumes the
                                 artifact; it does not define its shape.

WHY THIS EXISTS
---------------
The schema moved 2 -> 3 -> 4 while nothing pinned it. Schema 3 added the
canonical pairwise table (so the browser stopped computing log5) and schema 4
added per-team advancement probabilities (so Explore stopped needing anything
else). Both were additive and correct, but neither drift gate could see them:
the methodology spec does not reference the schema at all. That is the same
blind spot as the literal-transcribed selection fields — a contract no control
was watching.

The rule is deliberately strict in BOTH directions. Refusing an older schema is
obvious. Refusing a NEWER one matters just as much: a newer artifact may carry
fields this code does not understand, or may have changed the meaning of one it
does, and silently rendering it would be a correctness failure that looks like a
success.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping

# The one schema this codebase understands. Bumping it is a deliberate act that
# requires updating REQUIRED_FIELDS, the JS mirror, and the fixtures.
EXPECTED_ARTIFACT_SCHEMA = 5

# Fields schema 4 must carry. Absence is a hard failure, never a silent default:
# a missing `pairwise` once meant the browser fell back to computing its own
# probabilities, which is the class of bug this contract exists to prevent.
REQUIRED_FIELDS: Dict[str, str] = {
    "schema": "contract version",
    "year": "season the field belongs to",
    "teams": "team table; every index in `candidates` and `first_round` refers to it",
    "first_round": "64 team indices in bracket order, so pairings need no client logic",
    "pairwise": "canonical P(row beats col), n*n row-major — the browser must never recompute this",
    "candidates": "pre-scored brackets",
    "team_round_probabilities": "per-team reach probability for six stages, from the FULL bank",
    "constraint_probabilities": "full-bank preference frequencies; never counted from candidates",
    "meta": "provenance and disclosure",
}

# What changed at each version, so a mismatch reports something actionable.
SCHEMA_HISTORY: Dict[int, str] = {
    2: "candidates + teams + first_round",
    3: "added `pairwise`; browser-side log5 removed",
    4: "added `team_round_probabilities`; powers Explore",
    5: "added canonical `name` on each team; the browser stopped guessing names from slugs",
}


class ArtifactSchemaError(RuntimeError):
    """Raised when an artifact's schema is not the one this code understands."""


def validate_schema(artifact: Mapping[str, Any], *, context: str = "") -> None:
    """Refuse anything that is not exactly the expected schema.

    Both directions are errors. Older artifacts lack fields the product needs;
    newer ones may redefine fields it thinks it understands.
    """
    where = f" ({context})" if context else ""
    declared = artifact.get("schema")

    if declared is None:
        raise ArtifactSchemaError(
            f"artifact{where} declares no schema. It predates the contract and "
            f"cannot be loaded; rebuild it at schema {EXPECTED_ARTIFACT_SCHEMA}."
        )
    if not isinstance(declared, int):
        raise ArtifactSchemaError(f"artifact{where} declares a non-integer schema {declared!r}")

    if declared < EXPECTED_ARTIFACT_SCHEMA:
        missing = SCHEMA_HISTORY.get(declared + 1, "later additions")
        raise ArtifactSchemaError(
            f"artifact{where} is schema {declared}, expected {EXPECTED_ARTIFACT_SCHEMA}. "
            f"It is missing: {missing}. Rebuild with "
            f"scripts/experiments/build_candidate_artifact.py."
        )
    if declared > EXPECTED_ARTIFACT_SCHEMA:
        raise ArtifactSchemaError(
            f"artifact{where} is schema {declared}, but this code understands "
            f"{EXPECTED_ARTIFACT_SCHEMA}. Refusing rather than guessing: a newer "
            f"artifact may carry fields this code does not know, or may have "
            f"redefined one it thinks it does."
        )


def validate_required_fields(artifact: Mapping[str, Any], *, context: str = "") -> None:
    """Every schema 4 field must be present and non-empty."""
    where = f" ({context})" if context else ""
    missing = [f for f in REQUIRED_FIELDS if f not in artifact]
    if missing:
        raise ArtifactSchemaError(
            f"artifact{where} is missing required schema {EXPECTED_ARTIFACT_SCHEMA} field(s): {missing}"
        )
    empty = [f for f in REQUIRED_FIELDS if f != "schema" and not artifact[f]]
    if empty:
        raise ArtifactSchemaError(f"artifact{where} has empty required field(s): {empty}")


def validate_internal_consistency(artifact: Mapping[str, Any], *, context: str = "") -> None:
    """Shapes must agree with the team table.

    Cheap, and it catches a truncated or half-regenerated artifact before any of
    it reaches a user.
    """
    where = f" ({context})" if context else ""
    n = len(artifact["teams"])

    if len(artifact["pairwise"]) != n * n:
        raise ArtifactSchemaError(
            f"artifact{where}: pairwise has {len(artifact['pairwise'])} entries, expected {n * n} for {n} teams"
        )
    if len(artifact["team_round_probabilities"]) != n:
        raise ArtifactSchemaError(
            f"artifact{where}: team_round_probabilities has "
            f"{len(artifact['team_round_probabilities'])} rows, expected {n}"
        )
    if any(len(row) != 6 for row in artifact["team_round_probabilities"]):
        raise ArtifactSchemaError(f"artifact{where}: every advancement row needs 6 stages")
    if len(artifact["first_round"]) != 64:
        raise ArtifactSchemaError(f"artifact{where}: first_round has {len(artifact['first_round'])} teams, expected 64")


def validate_artifact(artifact: Mapping[str, Any], *, context: str = "") -> None:
    """The full gate: schema, then fields, then shapes."""
    validate_schema(artifact, context=context)
    validate_required_fields(artifact, context=context)
    validate_internal_consistency(artifact, context=context)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def deterministic_payload(artifact: Mapping[str, Any]) -> Dict[str, Any]:
    """The artifact stripped of everything that is not semantic content.

    ``generated_at`` and the ``validation`` block are excluded: a rebuild from
    identical inputs must hash identically, and a wall-clock timestamp would make
    every rebuild look like a change. The schema version IS included -- it is
    semantic, and a schema change must move the hash.
    """
    payload = {k: v for k, v in artifact.items() if k not in ("validation", "provenance")}
    meta = dict(payload.get("meta", {}))
    meta.pop("generated_at", None)
    payload["meta"] = meta
    return payload


def payload_hash(artifact: Mapping[str, Any]) -> str:
    """SHA-256 over the deterministic payload."""
    return hashlib.sha256(
        json.dumps(deterministic_payload(artifact), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def contract_record(artifact: Mapping[str, Any]) -> Dict[str, Any]:
    """The block a manifest should carry so the schema is part of the record."""
    return {
        "artifact_schema": artifact["schema"],
        "expected_artifact_schema": EXPECTED_ARTIFACT_SCHEMA,
        "schema_history": SCHEMA_HISTORY,
        "required_fields": sorted(REQUIRED_FIELDS),
        "payload_sha256": payload_hash(artifact),
        "payload_excludes": ["validation", "provenance", "meta.generated_at"],
        "compatibility": "exact match only; older and newer are both refused",
    }
