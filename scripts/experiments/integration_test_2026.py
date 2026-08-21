"""2026 end-to-end integration test. NOT an evaluation.

2026 is contaminated as a benchmark: it sits inside BACKTEST_YEARS, the
production strategy was selected on a window containing it, and a documented
modelling conclusion was drawn from its outcome. See PROSPECTIVE_2027.md.

This script therefore checks that the pipeline *works*, never how well it
predicts. It deliberately does not compute or print realized bracket score,
champion accuracy, or how the 2026 P(1st) figures compare to anything. P(1st) is
verified to *compute correctly* — the value is machinery output, not a result.

If a check fails in a way that can only be fixed by changing a frozen parameter,
STOP and report it. That is a v2 event (PROSPECTIVE_2027_CHECKPOINTS.md,
Checkpoint 3), not something to absorb. A bug in the *implementation* of a frozen
parameter may be fixed under v1; a change to a parameter's *value or definition*
may not.

Usage:
    python3 scripts/experiments/integration_test_2026.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts._common import load_seeds_and_regions  # noqa: E402
from scripts.experiments.build_candidate_artifact import (  # noqa: E402
    _constraint_predicates,
    assert_pretournament_inputs,
    build,
)
from scripts.experiments.conditional_bracket_engine import _REACHES  # noqa: E402
from src.governance.frozen_spec import diff_against_frozen  # noqa: E402
from src.prediction.noseed_model import (  # noqa: E402
    REQUIRED_FEATURE_KEYS,
    validate_stats_payload,
)

YEAR = 2026
ROUND_NAMES = ("R64", "R32", "S16", "E8", "F4", "CHAMP")

_results: List[Dict] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def main() -> int:
    print(f"2026 INTEGRATION TEST (not an evaluation)\n{'=' * 70}")

    # --- frozen-spec integrity -------------------------------------------
    print("\nfrozen specification")
    drift = diff_against_frozen(Path("configs/frozen/prospective_2027.json"))
    check(
        "system matches the 2027.v1 freeze",
        not drift["drifted"],
        f"{len(drift['drifted'])} drifted" if drift["drifted"] else drift["spec_version"],
    )

    # --- data provenance --------------------------------------------------
    print("\ndata provenance")
    try:
        prov = assert_pretournament_inputs(YEAR)
        check(
            "pre-tournament gate accepts 2026 inputs",
            True,
            f"cutoff {prov['torvik']['cutoff_date']} < tip {prov['torvik']['tournament_start']}",
        )
    except Exception as exc:
        check("pre-tournament gate accepts 2026 inputs", False, str(exc)[:90])
        return _finish()

    # --- train/serve feature parity ---------------------------------------
    print("\ntrain/serve feature parity")
    from scripts.mc_pool_backtest import _load_team_stats

    stats = _load_team_stats(YEAR)
    seeds, _regions = load_seeds_and_regions(YEAR)
    teams = [t for t in stats.values() if isinstance(t, dict)]
    worst = min(sum(1 for t in teams if t.get(k) is not None) / len(teams) for k in REQUIRED_FEATURE_KEYS)
    check("serving payload carries all 12 features", worst > 0.9, f"worst coverage {worst:.0%}")
    try:
        validate_stats_payload(stats, context="integration 2026")
        check("validate_stats_payload accepts the serving payload", True)
    except Exception as exc:
        check("validate_stats_payload accepts the serving payload", False, str(exc)[:90])

    # --- full artifact build ---------------------------------------------
    print("\nscenario bank + candidate artifact")
    art = build(YEAR, n_sims=60_000, target=1500, trials=800, seed=20260820)
    v = art["validation"]
    n = len(art["candidates"])
    check("artifact produced candidates", n > 1000, f"{n:,}")
    check("every candidate path-consistent", v["path_consistent"], f"{v['path_checked']} checked")
    check("EV recomputes exactly", v["ev_max_abs_error"] < 1e-9, f"max err {v['ev_max_abs_error']:.2e}")
    check(
        "champion diversity preserved",
        v["distinct_champions_artifact"] >= v["distinct_champions_full"] * 0.95,
        f"{v['distinct_champions_full']} -> {v['distinct_champions_artifact']}",
    )
    check(
        "bracket diversity preserved",
        v["mean_hamming_artifact"] >= v["mean_hamming_full"] * 0.9,
        f"Hamming {v['mean_hamming_full']} -> {v['mean_hamming_artifact']}",
    )
    check(
        "objective-disagreement region retained",
        v["low_ev_high_p1_count"] > 0,
        f"{v['low_ev_high_p1_count']} low-EV/high-P1 candidates",
    )
    check(
        "P(1st) computed for every candidate",
        all(0.0 <= c["p1"] <= 1.0 for c in art["candidates"]),
        "all in [0,1]",
    )
    check(
        "preference coverage non-thin",
        all(x > 20 for x in v["constraint_coverage"].values()),
        f"min {min(v['constraint_coverage'].values())}",
    )

    # --- preference predicates agree with the shipped frequencies ---------
    print("\npreference predicates")
    preds = _constraint_predicates(seeds)
    tidx = {t["id"]: i for i, t in enumerate(art["teams"])}
    rev = {i: t for t, i in tidx.items()}
    decoded = [[[rev[i] for i in rnd] for rnd in c["w"]] for c in art["candidates"]]
    ok = True
    for key, fn in preds.items():
        got = sum(1 for r in decoded if fn(r))
        if got != v["constraint_coverage"].get(
            {
                "f4_at_least_1_two_three": ">=1 2/3-seed F4",
                "f4_at_least_2_two_three": ">=2 2/3-seeds F4",
                "f4_mostly_favorites": ">=3 1-seeds F4",
                "s16_at_least_1_double_digit": ">=1 dd-seed S16",
                "s16_at_least_2_double_digit": ">=2 dd-seeds S16",
                "s16_no_double_digit": "no dd-seed S16",
            }[key],
            -1,
        ):
            ok = False
    check("predicates reproduce on the decoded artifact", ok)
    check(
        "frequencies sourced from full bank, not artifact",
        "constraint_probabilities" in art and "candidates_are_not_a_probability_sample" in art["meta"],
        f"bias recorded: {min(v['constraint_prob_bias'].values()):+.3f}..{max(v['constraint_prob_bias'].values()):+.3f}",
    )

    # --- schema -----------------------------------------------------------
    print("\nartifact schema")
    for key in ("schema", "year", "teams", "candidates", "meta", "provenance", "validation"):
        check(f"top-level key '{key}'", key in art)
    check("teams carry id/seed/region", all({"id", "seed", "region"} <= set(t) for t in art["teams"]))
    check(
        "candidates carry w/ev/p1/dd16",
        all({"w", "ev", "p1", "dd16"} <= set(c) for c in art["candidates"]),
    )
    check("pool-size assumption disclosed", "not a universal probability" in art["meta"]["p1_assumption"].lower())
    check("objectives limited to the two measured", art["meta"]["objectives"] == ["ev", "p1"])

    # --- three-strategy output + Python/browser parity --------------------
    print("\nstrategy selection and Python/browser parity")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "candidates_2026.json"
        with open(path, "w") as f:
            json.dump(art, f, separators=(",", ":"))

        py = {obj: _select_python(art, decoded, obj, seeds) for obj in ("ev", "p1")}
        for obj, sel in py.items():
            champs = {decoded[i][_REACHES["CHAMP"]][0] for i in sel}
            check(f"'{obj}' returns 3 brackets", len(sel) == 3, f"champions {len(champs)}")
            check(f"'{obj}' champions are distinct", len(champs) == 3)

        js = _select_node(path)
        if js is None:
            check("browser parity", False, "node unavailable")
        else:
            for obj in ("ev", "p1"):
                check(
                    f"Python/browser agree on '{obj}' selection",
                    js[obj] == py[obj],
                    f"py={py[obj]} js={js[obj]}",
                )

    check(
        "artifact size acceptable for the browser",
        len(json.dumps(art, separators=(",", ":"))) < 3_000_000,
        f"{len(json.dumps(art, separators=(',', ':'))) / 2**20:.2f} MB",
    )
    return _finish()


def _select_python(art, decoded, objective, seeds, k=3):
    """Hierarchical selection: distinct champion, then distinct Final Four."""
    order = sorted(range(len(art["candidates"])), key=lambda i: -art["candidates"][i][objective])
    chosen, used_c, used_f = [], set(), set()
    for i in order:
        if len(chosen) >= k:
            break
        c = decoded[i][_REACHES["CHAMP"]][0]
        if c not in used_c:
            chosen.append(i)
            used_c.add(c)
            used_f.add(frozenset(decoded[i][_REACHES["F4"]]))
    for i in order:
        if len(chosen) >= k:
            break
        f = frozenset(decoded[i][_REACHES["F4"]])
        if f not in used_f:
            chosen.append(i)
            used_f.add(f)
    return chosen


def _select_node(path: Path):
    """Same algorithm in JS, to prove the browser will agree with Python."""
    js = """
    import fs from 'fs';
    const art = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
    const C = art.candidates, CH = 5, F4 = 3;
    function sel(obj, k = 3) {
      const order = C.map((_, i) => i).sort((a, b) => C[b][obj] - C[a][obj]);
      const chosen = [], uc = new Set(), uf = new Set();
      for (const i of order) { if (chosen.length >= k) break;
        const c = C[i].w[CH][0];
        if (!uc.has(c)) { chosen.push(i); uc.add(c); uf.add(C[i].w[F4].slice().sort().join(',')); } }
      for (const i of order) { if (chosen.length >= k) break;
        const f = C[i].w[F4].slice().sort().join(',');
        if (!uf.has(f)) { chosen.push(i); uf.add(f); } }
      return chosen;
    }
    console.log(JSON.stringify({ ev: sel('ev'), p1: sel('p1') }));
    """
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False) as f:
            f.write(js)
            script = f.name
        out = subprocess.run(["node", script, str(path)], capture_output=True, text=True, timeout=120)
        return json.loads(out.stdout) if out.returncode == 0 else None
    except Exception:
        return None


def _finish() -> int:
    passed = sum(1 for r in _results if r["pass"])
    total = len(_results)
    print(f"\n{'=' * 70}\n{passed}/{total} checks passed")
    failed = [r for r in _results if not r["pass"]]
    if failed:
        print("\nFAILURES:")
        for r in failed:
            print(f"  - {r['check']}: {r['detail']}")
        print(
            "\nIf a fix requires changing a frozen PARAMETER VALUE, stop and report it as a\n"
            "v2 event (PROSPECTIVE_2027_CHECKPOINTS.md, Checkpoint 3). Implementation bugs\n"
            "may be fixed under v1."
        )
    print(
        "\nNOTE: no 2026 predictive performance is reported here, by design. 2026 is an\n"
        "in-sample integration season; its accuracy, bracket score and P(1st) must not be\n"
        "used as evidence of out-of-sample performance."
    )
    Path("artifacts/integration").mkdir(parents=True, exist_ok=True)
    with open("artifacts/integration/integration_2026.json", "w") as f:
        json.dump({"year": YEAR, "passed": passed, "total": total, "checks": _results}, f, indent=2)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
