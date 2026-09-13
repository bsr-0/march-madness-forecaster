"""The 2027 A/B: power analysis, freeze, verify, record (audit recommendation 16).

Recommendation 16 asked for a pre-registered A/B between the production
per-season selector and a fixed rule. The first thing a protocol needs is a
power calculation, and doing it BEFORE writing the protocol -- rather than
after the data disappoints, which is when it usually happens -- showed the
proposed test cannot conclude: 181 seasons at 80% power on P(1st), the metric
the pool actually pays on.

So what is pre-registered is an accumulating ledger with a stopping rule, not a
test that will resolve. See ``src/governance/ab_2027.py`` for the reasoning and
``PROSPECTIVE_2027_AB.md`` for the human-readable protocol.

    --power     Recompute the historical effect sizes and required sample
                sizes from the 14 evaluation seasons. This is what the frozen
                HISTORICAL_ESTIMATE was derived from; re-running it is how you
                check that estimate rather than trusting it.
    --freeze    Write configs/frozen/prospective_2027_ab.json. Refuses to
                overwrite a differing protocol.
    --verify    Check the live protocol against the frozen one.
    --record    Record one prospective season, once 2027 has been played.
    --tally     Describe the prospective series so far. Not a test.

WHY NOT JUST RUN THE A/B IN 2027 AND SEE. Because "see" would mean looking at
one observation from a series that needs 181, and one observation will look
like something. Whichever arm wins 2027 will be quotable as evidence by anyone
who wants it to be, and the only defence against that is a rule written down
first that says what one season is worth. That rule is the deliverable here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.canonical_contract import as_backtest_kwargs  # noqa: E402
from src.governance.ab_2027 import (  # noqa: E402
    ALPHA,
    ARM_CONTROL,
    ARM_TREATMENT,
    BARRED_CONTROLS,
    METRIC_DIRECTION,
    PRIMARY_METRICS,
    freeze,
    record_season,
    tally,
    verify,
)

POWER_ARTIFACT = Path("artifacts/headline_measurement/ab_2027_power.json")


def _seasons_for_power(effect_d: float, power: float = 0.80, alpha: float = ALPHA, cap: int = 100_000) -> int:
    """Paired-t sample size for a given standardised effect, by exact search.

    Uses the noncentral t rather than a normal approximation: at these effect
    sizes the normal approximation understates the requirement, and the whole
    point of this calculation is not to understate it.
    """
    import numpy as np
    from scipy import stats

    if effect_d == 0:
        return cap
    n = 2
    while n <= cap:
        crit = stats.t.ppf(1 - alpha / 2, n - 1)
        ncp = abs(effect_d) * np.sqrt(n)
        achieved = 1 - stats.nct.cdf(crit, n - 1, ncp) + stats.nct.cdf(-crit, n - 1, ncp)
        if achieved >= power:
            return n
        n += 1
    return cap


def run_power(workers: int, out_path: Path) -> Dict[str, Any]:
    """Effect sizes and required sample sizes, from the historical window."""
    import numpy as np
    from scipy import stats

    from scripts.mc_pool_backtest import EVALUATION_YEARS, StrategiesFitter, run_backtest

    print(f"[power] {ARM_TREATMENT} vs {ARM_CONTROL} over {len(EVALUATION_YEARS)} seasons")
    t0 = time.time()
    results = run_backtest(
        years=list(EVALUATION_YEARS),
        hparam_fitter=StrategiesFitter((ARM_TREATMENT, ARM_CONTROL, "seed")),
        workers=workers,
        save_brackets=False,
        **as_backtest_kwargs(),
    )
    by: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for r in results:
        by.setdefault(str(r["mode"]), {})[int(r["year"])] = r
    seasons = sorted(by[ARM_TREATMENT])

    per_metric: Dict[str, Any] = {}
    print(f"\n{'metric':<14} {'treat':>9} {'control':>9} {'diff':>10} {'SD':>9} {'d':>7} {'p':>7} {'n@80%':>8}")
    for metric in PRIMARY_METRICS:
        direction = METRIC_DIRECTION[metric]
        a = np.array([float(by[ARM_TREATMENT][y][metric]) for y in seasons])
        b = np.array([float(by[ARM_CONTROL][y][metric]) for y in seasons])
        signed = (a - b) * direction  # positive => treatment better
        sd = float(signed.std(ddof=1))
        d = float(signed.mean() / sd) if sd else 0.0
        _t, p = stats.ttest_rel(a, b)
        n80 = _seasons_for_power(d, 0.80)
        n90 = _seasons_for_power(d, 0.90)
        per_metric[metric] = {
            "direction": direction,
            "treatment_mean": float(a.mean()),
            "control_mean": float(b.mean()),
            "paired_diff_raw": float((a - b).mean()),
            "paired_diff_signed": float(signed.mean()),
            "paired_sd": sd,
            "cohens_d": d,
            "p_value": float(p),
            "treatment_better_seasons": int((signed > 0).sum()),
            "control_better_seasons": int((signed < 0).sum()),
            "n_seasons": len(seasons),
            "seasons_for_80pct_power": n80,
            "seasons_for_90pct_power": n90,
            "per_season": {
                str(y): {"treatment": float(by[ARM_TREATMENT][y][metric]), "control": float(by[ARM_CONTROL][y][metric])}
                for y in seasons
            },
        }
        print(
            f"{metric:<14} {a.mean():>9.4f} {b.mean():>9.4f} {(a - b).mean():>+10.4f} "
            f"{sd:>9.4f} {d:>7.3f} {p:>7.3f} {n80:>8}"
        )

    worst = max(per_metric[m]["seasons_for_80pct_power"] for m in PRIMARY_METRICS)
    payload = {
        "measurement": "power analysis for the pre-registered 2027 A/B",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "arm_treatment": ARM_TREATMENT,
        "arm_control": ARM_CONTROL,
        "barred_controls": dict(BARRED_CONTROLS),
        "seasons": seasons,
        "alpha": ALPHA,
        "per_metric": per_metric,
        "seasons_needed_worst_metric": worst,
        "calendar_year_of_resolution": 2027 + worst - 1,
        "verdict": (
            "not_resolvable_prospectively"
            if worst > 40
            else "resolvable_but_slowly"
        ),
        "interpretation": (
            "The two primary metrics disagree in SIGN on the historical window: the treatment "
            "arm is nominally ahead on p_first and clearly behind on mean_rank. Both are "
            "pre-declared primary for that reason -- dropping the inconvenient one after the "
            "fact is the behaviour this pre-registration exists to prevent. A winner-take-all "
            "pool pays on p_first, so the production choice is defensible; but the selector is "
            "credited with an edge that is not distinguishable from zero on its own objective."
        ),
        "wall_time_minutes": round((time.time() - t0) / 60, 1),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp.replace(out_path)
    print(f"\n[power] worst metric needs {worst} seasons -> resolution around {2027 + worst - 1}")
    print(f"[power] VERDICT: {payload['verdict']}")
    print(f"  [artifact] {out_path}")
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--power", action="store_true", help="Recompute effect sizes and required sample sizes.")
    parser.add_argument("--freeze", action="store_true", help="Write the frozen protocol.")
    parser.add_argument("--verify", action="store_true", help="Check the live protocol against the frozen one.")
    parser.add_argument("--tally", action="store_true", help="Describe the prospective series (not a test).")
    parser.add_argument("--record", type=int, default=None, metavar="YEAR", help="Record one played season.")
    parser.add_argument("--treatment", type=json.loads, default=None, help='e.g. \'{"p_first":0.12,"mean_rank":9.8}\'')
    parser.add_argument("--control", type=json.loads, default=None, help="As --treatment, for the control arm.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=POWER_ARTIFACT)
    args = parser.parse_args(argv)

    if not any([args.power, args.freeze, args.verify, args.tally, args.record]):
        parser.error("pick at least one of --power / --freeze / --verify / --tally / --record")

    if args.power:
        run_power(args.workers, args.out)
    if args.freeze:
        payload = freeze()
        print(f"[freeze] {payload['spec_version']} hash {payload['spec_hash'][:16]}")
    if args.verify:
        result = verify()
        print(json.dumps(result, indent=2))
        if result.get("frozen") and not result["matches"]:
            print("\nPROTOCOL DRIFT: the live protocol differs from the frozen one.")
            return 1
    if args.record is not None:
        if args.treatment is None or args.control is None:
            parser.error("--record needs --treatment and --control")
        entry = record_season(args.record, args.treatment, args.control)
        print(f"[record] {entry['year']} recorded under {entry['spec_hash'][:12]}")
    if args.tally or args.record is not None:
        print(json.dumps(tally(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
