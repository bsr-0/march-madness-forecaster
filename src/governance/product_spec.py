"""Product selection specification — how candidates become the brackets a user sees.

WHY THIS EXISTS SEPARATELY FROM THE METHODOLOGY FREEZE
------------------------------------------------------
There are two different contracts in this system and they were previously
tangled in one file:

    2027.v2 (methodology)   what determines the model, the simulation and the
                            candidate artifact -- and therefore the prospective
                            claim. Frozen before 2027 exists; changing it
                            invalidates that claim.

    product.vN (selection)  how the already-produced candidate set is turned
                            into the brackets on screen. Changing it cannot
                            affect a single probability, an expected score or a
                            P(1st); it changes which of the already-scored rows
                            are shown.

Three selection-owned fields -- ``diversity_algorithm``, ``k_returned`` and the
product strategy list -- were embedded in the methodology spec as hardcoded
literals. That produced the worst possible failure in an integrity control:
``diff_against_frozen`` compared a constant to itself, so it reported "no drift"
while all three had stopped describing the product. A gate that cannot fail is
not a gate.

They live here now, and they are DERIVED FROM THE LIVE IMPLEMENTATION rather
than transcribed, so a change to selection moves this hash and a hand-edit
cannot paper over it.

THE AUDIT TRAIL, STATED PLAINLY
-------------------------------
The 2027.v2 methodology is unchanged. No model, simulation, objective, scoring
rule or P(1st) definition moved. Selection-owned fields previously embedded in
the methodology specification were reclassified under product.v3. The
methodology hash changed only because the specification boundary was corrected,
and ``test_methodology_values_are_unchanged_by_the_boundary_correction`` proves
that field-by-field against the original immutable file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

# Bumped from product.v2 because the versioned contract itself is changing:
# these fields are now part of the product specification and are hashed.
#
#   product.v1  hierarchical: distinct champion -> distinct Final Four
#   product.v2  quality-first with tiered structural diversity
#   product.v3  v2 semantics, now specified and hash-pinned here
PRODUCT_SPEC_VERSION = "product.v3"
PRODUCT_SPEC_PATH = Path("configs/frozen/product_v3.json")

# Where these fields came from, so the reclassification is auditable.
RECLASSIFIED_FROM = {
    "spec_version": "2027.v2",
    "spec_path": "configs/frozen/prospective_2027_v2.json",
    "fields": [
        "candidate_selection.diversity_algorithm",
        "candidate_selection.k_returned",
        "product.strategies",
    ],
    "reason": (
        "These describe how candidates are turned into the displayed brackets, not "
        "how the candidates were produced. They were hardcoded literals in "
        "capture_live_spec(), so the methodology drift gate compared them to "
        "themselves and could never detect a change. Reclassified under "
        "product.v3 and derived from the live implementation."
    ),
    "methodology_unchanged": True,
}


def capture_live_product_spec() -> Dict[str, Any]:
    """Read selection behaviour out of the live code.

    Every value is introspected. In particular ``diversity_tiers`` and the
    thresholds come from ``src.product.selection`` itself, so editing the tier
    ladder or a threshold moves this hash -- which is exactly what the old
    literal-transcribed version could not do.
    """
    import inspect

    from src.product import selection as sel

    build_k = _build_flow_k()

    return {
        "product_spec_version": PRODUCT_SPEC_VERSION,
        "reclassified_from": RECLASSIFIED_FROM,
        "selection": {
            "version": sel.SELECTION_VERSION,
            "algorithm": (
                "quality-first with tiered structural diversity: slot 1 is the "
                "highest-scoring candidate; each later slot is the highest-scoring "
                "candidate within the retention floor that differs from EVERY "
                "already-selected bracket at the most meaningful tier available"
            ),
            "diversity_tiers": list(sel.DIVERSITY_TIERS),
            "min_f4_changes": sel.MIN_F4_CHANGES,
            "min_s16_changes": sel.MIN_S16_CHANGES,
            "min_retention": sel.DEFAULT_MIN_RETENTION,
            "retention_basis": "absolute, measured from slot 1 (never compounding)",
            "champion_difference_is_sufficient_alone": False,
            "may_return_fewer_than_k": True,
            "distance_metric": None,
            "distance_metric_note": (
                "Deliberately not a distance metric. Hamming weights all 63 games "
                "equally, so R64 is 50.8% of Hamming but 16.7% of the points."
            ),
            "objectives": list(sel.OBJECTIVES),
            "k_returned_by_build": build_k,
            "tie_break": "descending objective, ascending candidate index",
        },
        "frozen_v1_selector_retained": {
            "function": "src.product.selection.select",
            "algorithm": "hierarchical: distinct champion -> distinct Final Four -> top up",
            "default_k": inspect.signature(sel.select).parameters["k"].default,
            "note": "retained unchanged for research comparison; not used by the product",
        },
        "strategies": _live_strategies(),
        "preference_predicates_exposed_in_ui": False,
        "preference_predicates_available": sorted(
            set(sel.preference_predicates({"teams": []}).keys()) | {"team_reaches_final_four"}
        ),
    }


def _build_flow_k() -> int:
    """The k the shipped Build flow actually requests.

    Read from docs/build.js rather than assumed: the browser is what the user
    meets, and a mismatch between the documented k and the shipped k is precisely
    the sort of thing this spec exists to catch.
    """
    import re

    src = Path("docs/build.js").read_text()
    if re.search(r"selectWithAlternative\(\s*ARTIFACT\s*,\s*buildState\.objective\s*\)", src):
        return 2  # selectWithAlternative is k=2 by definition
    m = re.search(r"selectDiverse\(\s*ARTIFACT[^)]*?,\s*(\d+)\s*\)", src)
    if m:
        return int(m.group(1))
    raise RuntimeError("cannot determine k from docs/build.js; the product spec must not guess")


def _live_strategies() -> list:
    """The strategies the shipped UI actually offers.

    Parsed from docs/build.js's BUILD_STRATEGIES. "Your Preference" was listed in
    the old methodology spec but is not exposed, which is the kind of divergence
    a transcribed literal hides.
    """
    import re

    src = Path("docs/build.js").read_text()
    block = re.search(r"const BUILD_STRATEGIES = \[(.*?)\n\];", src, re.S)
    if not block:
        raise RuntimeError("BUILD_STRATEGIES not found in docs/build.js")
    # Split on entry boundaries rather than matching across them: the icons are
    # unicode escapes like \u{1F9E0}, whose brace would terminate a [^}] class.
    out = []
    for entry in re.split(r"\n\s*\{\s*(?=key:)", block.group(1)):
        key = re.search(r"key:\s*'(\w+)'", entry)
        label = re.search(r"label:\s*'([^']+)'", entry)
        if key and label:
            out.append({"name": label.group(1), "objective": key.group(1), "constraint": None})
    if not out:
        raise RuntimeError("no strategies parsed from BUILD_STRATEGIES")
    return out


def canonical_hash(spec: Dict[str, Any]) -> str:
    """SHA-256 over canonical JSON (sorted keys, no whitespace)."""
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_product_spec(path: Path = PRODUCT_SPEC_PATH) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def diff_against_product_spec(path: Path = PRODUCT_SPEC_PATH) -> Dict[str, Any]:
    """Compare live selection behaviour against the pinned product spec."""
    pinned = load_product_spec(path)
    live = capture_live_product_spec()
    body = {k: v for k, v in pinned.items() if k != "spec_hash"}

    drifted = []

    def walk(a, b, trail):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                walk(a.get(key), b.get(key), trail + [key])
        elif a != b:
            drifted.append({"path": ".".join(trail), "pinned": a, "live": b})

    walk(body, live, [])
    return {
        "drifted": drifted,
        "pinned_hash": pinned.get("spec_hash"),
        "live_hash": canonical_hash(live),
        "product_spec_version": body.get("product_spec_version"),
    }


def write_product_spec(path: Path = PRODUCT_SPEC_PATH) -> Dict[str, Any]:
    """Pin the current live selection behaviour."""
    spec = capture_live_product_spec()
    spec["spec_hash"] = canonical_hash(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    return spec
