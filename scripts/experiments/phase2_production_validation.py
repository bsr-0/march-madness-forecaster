"""Phase 2 — production-scale validation of the frozen 2027.v2 pipeline.

Runs the frozen methodology at production parameters using 2026 strictly as an
integration fixture. **This does not create or claim a 2027 artifact.** No 2027
inputs exist at the time of writing; the point is to prove that in March 2027 the
same command against 2027 pre-tournament inputs yields a reproducible, auditable
artifact.

The deliverable is the manifest, not a pass count. The manifest is the template
for the official 2027 prediction record, and it is what makes April 2027 able to
verify that the published brackets came from the frozen system rather than from
something that resembled it.

DETERMINISM
-----------
The artifact is built twice with identical inputs and seeds, and the two payload
hashes must match exactly. Timestamps and other non-reproducible fields are
excluded from the hashed payload, since hashing them would make every build
trivially unique and prove nothing.

NO 2026 EVALUATION
------------------
Nothing here computes realized bracket score, champion accuracy, Final Four
accuracy or any other outcome metric for 2026, and no result is compared against
the historical 11.2% / 10.47% production figures. 2026 is a fixture. Its
predictive performance is not measured, so it cannot leak into a claim.

Usage:
    python3 scripts/experiments/phase2_production_validation.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts._common import load_seeds_and_regions  # noqa: E402
from scripts.experiments.build_candidate_artifact import (  # noqa: E402
    _constraint_predicates,
    build,
)
from scripts.experiments.conditional_bracket_engine import _REACHES  # noqa: E402
from scripts.experiments.integration_test_2026 import _select_node, _select_python  # noqa: E402
from src.governance.frozen_spec import (  # noqa: E402
    FROZEN_SPEC_PATH,
    SPEC_VERSION,
    diff_against_frozen,
    load_frozen_spec,
)

FIXTURE_YEAR = 2026

# Production parameters, taken from the frozen v2 specification rather than
# chosen here, so this script cannot silently validate something else.
_spec = load_frozen_spec(Path(FROZEN_SPEC_PATH))
BANK_SIMS = _spec["tournament_engine"]["scenario_bank_size"]
TARGET_CANDIDATES = _spec["candidate_selection"]["target_candidates"]
P1_TRIALS = _spec["candidate_selection"]["p1_trials"]
POOL_SIZE = _spec["candidate_selection"]["p1_pool_size"]
RANDOM_SEED = 20260820

_gates: List[Dict] = []


def gate(name: str, ok: bool, detail: str = "") -> bool:
    _gates.append({"gate": name, "pass": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def deterministic_payload(art: Dict[str, Any]) -> Dict[str, Any]:
    """The reproducible content, with non-deterministic fields removed.

    `generated_at` and the validation block are excluded: hashing a timestamp
    would make every build unique and prove nothing about reproducibility.
    """
    payload = {k: v for k, v in art.items() if k != "validation"}
    meta = dict(payload.get("meta", {}))
    meta.pop("generated_at", None)
    payload["meta"] = meta
    return payload


def payload_hash(art: Dict[str, Any]) -> str:
    blob = json.dumps(deterministic_payload(art), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def input_hashes(year: int) -> Dict[str, str]:
    """Hash every input file the artifact depends on."""
    out = {}
    for label, path in (
        ("torvik", Path(f"data/raw/historical/torvik_{year}.json")),
        ("tournament_context", Path(f"data/raw/historical/tournament_context_{year}.json")),
        ("espn_picks", Path(f"data/raw/historical_public_picks/espn_picks_{year}.json")),
    ):
        out[label] = sha256_file(path) if path.exists() else "ABSENT"
    return out


def source_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    print(f"PHASE 2 — production-scale validation of {SPEC_VERSION}")
    print(f"fixture season {FIXTURE_YEAR} (integration only; no evaluation)\n{'=' * 74}")

    # 1. frozen-spec match ------------------------------------------------
    print("\nfrozen specification")
    drift = diff_against_frozen(Path(FROZEN_SPEC_PATH))
    if not gate(
        "live code matches the frozen spec exactly",
        not drift["drifted"],
        drift["spec_version"] if not drift["drifted"] else f"{len(drift['drifted'])} drifted",
    ):
        return _finish(None, {})

    print(
        f"  parameters from spec: bank={BANK_SIMS:,} candidates={TARGET_CANDIDATES:,} "
        f"trials={P1_TRIALS:,} pool={POOL_SIZE}"
    )

    ihash = input_hashes(FIXTURE_YEAR)

    # 2. determinism ------------------------------------------------------
    print("\ndeterministic reproducibility (building twice at production scale)")
    art_a = build(FIXTURE_YEAR, BANK_SIMS, TARGET_CANDIDATES, P1_TRIALS, RANDOM_SEED)
    h_a = payload_hash(art_a)
    art_b = build(FIXTURE_YEAR, BANK_SIMS, TARGET_CANDIDATES, P1_TRIALS, RANDOM_SEED)
    h_b = payload_hash(art_b)
    gate("identical seeds produce a byte-identical payload", h_a == h_b, f"{h_a[:16]}…")

    art = art_a
    v = art["validation"]
    seeds, _regions = load_seeds_and_regions(FIXTURE_YEAR)

    # 3-9. structural gates -----------------------------------------------
    print("\nstructural gates")
    gate("every candidate is a feasible bracket", v["path_consistent"], f"{v['path_checked']} checked")
    gate("EV independently recomputes", v["ev_max_abs_error"] < 1e-9, f"max err {v['ev_max_abs_error']:.2e}")
    p1s = [c["p1"] for c in art["candidates"]]
    gate("P(1st) values are valid probabilities", all(0.0 <= p <= 1.0 for p in p1s), f"n={len(p1s):,}")
    gate(
        "P(1st) standard error recorded",
        "p1_se_estimate" in art["meta"],
        f"SE ~{art['meta'].get('p1_se_estimate', 0) * 100:.2f}pp — differences below this are not rankable",
    )
    gate(
        "champion coverage preserved by the sampler",
        v["distinct_champions_artifact"] >= v["distinct_champions_full"] * 0.95,
        f"{v['distinct_champions_full']} -> {v['distinct_champions_artifact']}",
    )
    gate(
        "EV/P(1st) disagreement region survives sampling",
        v["low_ev_high_p1_count"] > 0,
        f"{v['low_ev_high_p1_count']} low-EV/high-P(1st) candidates",
    )
    gate(
        "bracket diversity preserved",
        v["mean_hamming_artifact"] >= v["mean_hamming_full"] * 0.9,
        f"Hamming {v['mean_hamming_full']} -> {v['mean_hamming_artifact']}",
    )
    cov = v["constraint_coverage"]
    gate("every preference predicate has coverage", all(x > 50 for x in cov.values()), f"min {min(cov.values()):,}")
    gate(
        "user-facing frequencies come from the full bank",
        "constraint_probabilities" in art and "candidates_are_not_a_probability_sample" in art["meta"],
        f"sampling bias {min(v['constraint_prob_bias'].values()):+.3f}..{max(v['constraint_prob_bias'].values()):+.3f}",
    )

    # 10-11. parity and schema --------------------------------------------
    print("\nparity and schema")
    tidx = {t["id"]: i for i, t in enumerate(art["teams"])}
    rev = {i: t for t, i in tidx.items()}
    decoded = [[[rev[i] for i in rnd] for rnd in c["w"]] for c in art["candidates"]]
    preds = _constraint_predicates(seeds)
    gate(
        "preference predicates reproduce on the decoded artifact",
        all(sum(1 for r in decoded if f(r)) > 0 for f in preds.values()),
    )
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / f"candidates_{FIXTURE_YEAR}.json"
        with open(p, "w") as f:
            json.dump(art, f, separators=(",", ":"))
        py = {o: _select_python(art, decoded, o, seeds) for o in ("ev", "p1")}
        js = _select_node(p)
        if js is None:
            gate("Python/browser selection parity", False, "node unavailable")
        else:
            for o in ("ev", "p1"):
                gate(f"Python/browser agree on '{o}'", js[o] == py[o], f"{py[o]}")
        for o, sel in py.items():
            champs = {decoded[i][_REACHES["CHAMP"]][0] for i in sel}
            gate(f"'{o}' returns 3 distinct-champion brackets", len(sel) == 3 and len(champs) == 3)

    required = {
        "schema",
        "year",
        "teams",
        "candidates",
        "meta",
        "provenance",
        "validation",
        "constraint_probabilities",
        "team_final_four_probabilities",
    }
    gate("artifact schema complete", required <= set(art), f"{len(art['candidates']):,} candidates")
    gate("pool-size assumption disclosed", "not a universal probability" in art["meta"]["p1_assumption"].lower())
    gate("objectives limited to the two measured", art["meta"]["objectives"] == ["ev", "p1"])

    # write + manifest -----------------------------------------------------
    out_dir = Path("artifacts/phase2")
    out_dir.mkdir(parents=True, exist_ok=True)
    art_path = out_dir / f"candidates_{FIXTURE_YEAR}_production.json"
    with open(art_path, "w") as f:
        json.dump(art, f, separators=(",", ":"))
    size = art_path.stat().st_size

    gate("artifact size acceptable for the browser", size < 3_000_000, f"{size / 2**20:.2f} MB")

    manifest = {
        "artifact_schema": art["schema"],
        "purpose": "Phase 2 production-scale validation. Integration fixture, NOT a 2027 artifact.",
        "spec_version": SPEC_VERSION,
        "spec_hash": _spec["spec_hash"],
        "source_commit": source_commit(),
        "fixture_season": FIXTURE_YEAR,
        "input_hashes": ihash,
        "generation_parameters": {
            "bank_sims": BANK_SIMS,
            "candidates_target": TARGET_CANDIDATES,
            "candidates_actual": len(art["candidates"]),
            "p1_trials": P1_TRIALS,
            "p1_pool_size": POOL_SIZE,
        },
        "random_seed": RANDOM_SEED,
        "artifact_payload_sha256": h_a,
        "artifact_file_sha256": sha256_file(art_path),
        "artifact_size_bytes": size,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_performed": False,
        "evaluation_note": (
            "No 2026 outcome metric was computed. 2026 is an integration fixture; its "
            "predictive performance must not be used as evidence of out-of-sample skill."
        ),
    }
    with open(out_dir / f"manifest_{FIXTURE_YEAR}_production.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    return _finish(manifest, art)


def _finish(manifest, art) -> int:
    passed = sum(1 for g in _gates if g["pass"])
    total = len(_gates)
    failed = [g for g in _gates if not g["pass"]]

    if manifest:
        print(f"\n{'=' * 74}\nARTIFACT MANIFEST\n{'=' * 74}")
        rows = [
            ("spec", manifest["spec_version"]),
            ("spec_hash", manifest["spec_hash"]),
            ("source_commit", manifest["source_commit"]),
            ("fixture_season", str(manifest["fixture_season"])),
            ("bank_sims", f"{manifest['generation_parameters']['bank_sims']:,}"),
            ("candidates", f"{manifest['generation_parameters']['candidates_actual']:,}"),
            ("p1_trials", f"{manifest['generation_parameters']['p1_trials']:,}"),
            ("pool_size", str(manifest["generation_parameters"]["p1_pool_size"])),
            ("random_seed", str(manifest["random_seed"])),
            ("artifact_sha256", manifest["artifact_payload_sha256"]),
            ("artifact_size", f"{manifest['artifact_size_bytes'] / 2**20:.2f} MB"),
            ("generated_at", manifest["generated_at"]),
            ("evaluation_performed", str(manifest["evaluation_performed"])),
        ]
        for k, val in rows:
            print(f"  {k:22} {val}")
        print("\n  input_hashes:")
        for k, val in manifest["input_hashes"].items():
            print(f"    {k:20} {val[:32]}…" if val != "ABSENT" else f"    {k:20} ABSENT")

    print(f"\n{'=' * 74}\n{passed}/{total} gates passed")
    if failed:
        print("\nFAILURES — stop and report; do not tune against 2026:")
        for g in failed:
            print(f"  - {g['gate']}: {g['detail']}")
    print(
        "\nNo 2026 outcome metric was computed. 2026 is an integration fixture; its\n"
        "predictive performance is not evidence of out-of-sample skill, and no result\n"
        "here is comparable to the historical 11.2% / 10.47% production figures."
    )
    Path("artifacts/phase2").mkdir(parents=True, exist_ok=True)
    with open("artifacts/phase2/gates.json", "w") as f:
        json.dump({"passed": passed, "total": total, "gates": _gates}, f, indent=2)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
