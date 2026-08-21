"""Generate the Python->JS parity fixture.

Python is canonical. This records the selection Python produces for
representative objective/preference combinations; tests/test_product_parity.py
asserts the JavaScript mirror reproduces it exactly.

Committing the fixture means semantic drift appears as a reviewable diff rather
than only as a red test.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.product.selection import select, select_diverse, constraint_frequency, SELECTION_VERSION

ART = Path("docs/data/candidates_2026.json")
OUT = Path("tests/fixtures/product/parity_2026.json")

def main() -> None:
    art = json.loads(ART.read_text())
    top_team = max(
        art.get("team_final_four_probabilities", {}).items(), key=lambda kv: kv[1], default=("", 0)
    )[0]
    tidx = {t["id"]: i for i, t in enumerate(art["teams"])}

    cases = []
    prefs = ["none", "f4_at_least_1_two_three", "f4_at_least_2_two_three",
             "f4_mostly_favorites", "s16_at_least_1_double_digit",
             "s16_at_least_2_double_digit", "s16_no_double_digit"]
    for obj in ("ev", "p1"):
        for pref in prefs:
            cases.append({
                "objective": obj, "preference": pref, "team_id": None,
                "expected_indices": select(art, obj, pref, k=3),
                "frequency": constraint_frequency(art, pref),
            })
        cases.append({
            "objective": obj, "preference": "team_reaches_final_four", "team_id": top_team,
            "expected_indices": select(art, obj, "team_reaches_final_four",
                                       team_index=tidx[top_team], k=3),
            "frequency": constraint_frequency(art, "team_reaches_final_four", top_team),
        })

    # product.v2 selector cases. This is what the Build flow actually calls, so
    # it needs the same drift protection as the frozen preference predicates.
    # k=3 is included because p1 legitimately returns fewer than k on this
    # artifact -- the mirror must reproduce the short result, not pad it.
    diverse = [
        {"objective": obj, "k": k, "expected_indices": select_diverse(art, obj, k=k)}
        for obj in ("ev", "p1")
        for k in (1, 2, 3)
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "generated_from": "src/product/selection.py (canonical)",
        "artifact": str(ART),
        "artifact_schema": art["schema"],
        "k": 3,
        "selection_version": SELECTION_VERSION,
        "cases": cases,
        "diverse_cases": diverse,
    }, indent=2) + "\n")
    print(f"wrote {OUT} — {len(cases)} cases")
    for c in cases[:4]:
        print(f"  {c['objective']:3} {c['preference']:28} -> {c['expected_indices']}")
    print(f"selection {SELECTION_VERSION} — {len(diverse)} diverse cases")
    for c in diverse:
        print(f"  {c['objective']:3} k={c['k']} -> {c['expected_indices']}")


if __name__ == "__main__":
    main()
