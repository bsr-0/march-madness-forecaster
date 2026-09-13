"""Do the pool-tool features change anything? (audit recommendation 13, parts 2-4)

Recommendation 13 asks for "pool-size and payout inputs, multi-entry support --
the features that separate a recommender from a pool tool". They are now built.
Building a feature is not evidence that it helps, so this measures each one
against the same 14 evaluation seasons under the canonical contract.

  --pool-factor   Part 2. Construction currently ignores pool size below 51
                  entries: `_make_ev_scorer`'s duplicate discount is gated on
                  `pool_size > 50`, so at 19-33 entries -- every real pool this
                  project has -- the same bracket is built regardless of size.
                  Compares threshold (production) / continuous / off.

  --payout        Part 3. Selection maximised P(1st) unconditionally, which is
                  the right objective for exactly one payout structure.
                  Measures whether a prize-weighted objective picks a different
                  bracket, and what it is worth, under each structure.

  --multi-entry   Part 4. Measures the marginal value of a 2nd, 3rd and 4th
                  entry -- the diminishing-returns curve a user needs to decide
                  whether more entries are worth the entry fee.

PRE-REGISTERED DECISION RULES, fixed before the first run.

  Pool factor. Adopt "continuous" only if it beats "threshold" on mean P(1st)
  by more than one season-level standard error (1.5pp, audit H1) AND wins in at
  least 9 of 14 seasons. Anything less is noise, and changing how the shipped
  bracket is built on noise is precisely the behaviour audit finding H2 exists
  to flag. Expected outcome is no adoption: the pool-size sweep already
  recorded in mc_pool_backtest.py:189-242 found the strategy's edge flat from
  pool 20 to 100.

  Payout. No threshold to meet -- this is a correctness feature, not an
  improvement. Reporting P(1st) to a user whose pool pays top 3 answers a
  question they did not ask, whatever the numbers say. What is measured is how
  OFTEN the objective changes the selected bracket, because a feature that
  never changes anything is worth knowing about too.

  Multi-entry. Report the marginal gains; adopt nothing. Whether a second entry
  is worth its fee depends on the fee, which this project does not know.

Every run uses the canonical contract (29 opponents, 100 repeats, team
identity, pool opponents, 500 pa_trials) so numbers are comparable to the
published headline. `--n-entries k` holds the pool size fixed by drawing k-1
fewer opponents: entering more brackets takes seats, it does not grow the pool.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.canonical_contract import CANONICAL_N_OPPONENTS

OUT_DIR = Path("artifacts/headline_measurement")

HEADLINE_MODE = "meta_region_poolaware"
BASELINE_MODE = "seed"

CANONICAL = dict(
    n_opponents=CANONICAL_N_OPPONENTS,
    n_repeats=100,
    opponent_source="pool",
    team_identity=True,
    pa_trials=500,
)

# Structures to compare. `tiered` is an alias of top_5 and adds nothing here.
PAYOUT_STRUCTURES = ("winner_take_all", "top_3", "top_5", "top_10pct", "top_25pct")

POOL_FACTOR_MODES = ("threshold", "continuous", "off")

ENTRY_COUNTS = (1, 2, 3, 4)

# One season-level standard error on the headline (audit H1). The finest
# distinction 14 seasons supports.
SEASON_SE_PP = 1.5


def _run(modes: Sequence[str], years: Sequence[int], workers: int, **overrides: Any) -> List[Dict[str, Any]]:
    from scripts.mc_pool_backtest import StrategiesFitter, run_backtest

    kwargs = dict(CANONICAL)
    kwargs.update(overrides)
    return run_backtest(
        years=list(years),
        hparam_fitter=StrategiesFitter(tuple(modes)),
        workers=workers,
        save_brackets=False,  # never; see scripts/pool_rdof_audit.py's docstring
        **kwargs,
    )


def _by_mode(results: Sequence[Dict[str, Any]], mode: str, key: str) -> Dict[int, float]:
    return {int(r["year"]): float(r[key]) for r in results if r["mode"] == mode}


def _mean(d: Dict[int, float]) -> float:
    return sum(d.values()) / len(d) if d else float("nan")


def _provenance() -> Dict[str, Any]:
    import subprocess

    def _git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        except Exception:
            return "unknown"

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "canonical_contract": dict(CANONICAL),
    }


def _write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp.replace(path)
    print(f"  [artifact] {path}")


# ---------------------------------------------------------------------------


def run_pool_factor(years, workers, out_dir) -> Dict[str, Any]:
    """Does letting pool size reach construction below 51 entries help?"""
    per_mode: Dict[str, Any] = {}
    for mode in POOL_FACTOR_MODES:
        print(f"\n[pool-factor] mode={mode}")
        t0 = time.time()
        res = _run([HEADLINE_MODE, BASELINE_MODE], years, workers, pool_factor_mode=mode)
        p1 = _by_mode(res, HEADLINE_MODE, "p_first")
        per_mode[mode] = {
            "per_year_p_first": {str(y): v for y, v in sorted(p1.items())},
            "mean_p_first": _mean(p1),
            "baseline_mean_p_first": _mean(_by_mode(res, BASELINE_MODE, "p_first")),
            "wall_time_minutes": round((time.time() - t0) / 60, 1),
        }
        print(f"[pool-factor] {mode}: mean P(1st) = {per_mode[mode]['mean_p_first']:.4f}")

    base = per_mode["threshold"]["per_year_p_first"]
    cont = per_mode["continuous"]["per_year_p_first"]
    wins = sum(1 for y in base if cont[y] > base[y])
    delta_pp = (per_mode["continuous"]["mean_p_first"] - per_mode["threshold"]["mean_p_first"]) * 100

    adopt = delta_pp > SEASON_SE_PP and wins >= 9
    payload = {
        "provenance": _provenance(),
        "measurement": "pool size in bracket construction: threshold vs continuous vs off",
        "seasons": list(years),
        "modes": per_mode,
        "continuous_minus_threshold_pp": round(delta_pp, 3),
        "continuous_wins_seasons": wins,
        "n_seasons": len(base),
        "adopt_continuous": adopt,
        "decision_rule": (
            f"Adopt continuous only if it beats threshold by more than {SEASON_SE_PP}pp "
            "(one season-level SE, audit H1) AND wins at least 9 of 14 seasons. "
            "Pre-registered before the run."
        ),
        "verdict": (
            "adopt_continuous"
            if adopt
            else "keep_threshold — the continuous discount does not clear the noise floor"
        ),
        "what_this_does_not_settle": (
            "Both arms are scored by the same simulated referee the strategy is selected "
            "against (audit C2), so this compares two construction rules under one model of "
            "how tournaments and opponents behave. It cannot say which builds better brackets "
            "against reality; only the real-outcome record can speak to that, and n=4 there."
        ),
    }
    _write(out_dir / "pool_factor_sensitivity.json", payload)
    return payload


def run_payout(years, workers, out_dir) -> Dict[str, Any]:
    """Does a prize-weighted objective pick a different bracket?"""
    from src.optimization.payout import describe, payout_shares, shares_summary

    per_structure: Dict[str, Any] = {}
    for structure in PAYOUT_STRUCTURES:
        print(f"\n[payout] structure={structure} — {describe(structure, 30)}")
        t0 = time.time()
        res = _run([HEADLINE_MODE, BASELINE_MODE], years, workers, payout=structure)
        p1 = _by_mode(res, HEADLINE_MODE, "p_first")
        prize = _by_mode(res, HEADLINE_MODE, "expected_prize")
        base_prize = _by_mode(res, BASELINE_MODE, "expected_prize")
        per_structure[structure] = {
            "shares": shares_summary(payout_shares(structure, 30)),
            "per_year_p_first": {str(y): v for y, v in sorted(p1.items())},
            "per_year_expected_prize": {str(y): v for y, v in sorted(prize.items())},
            "mean_p_first": _mean(p1),
            "mean_expected_prize": _mean(prize),
            "baseline_mean_expected_prize": _mean(base_prize),
            "prize_multiple_over_seed": (_mean(prize) / _mean(base_prize)) if _mean(base_prize) else None,
            "wall_time_minutes": round((time.time() - t0) / 60, 1),
        }
        print(
            f"[payout] {structure}: mean E[prize] = {per_structure[structure]['mean_expected_prize']:.4f} "
            f"vs seed {per_structure[structure]['baseline_mean_expected_prize']:.4f}"
        )

    # How often does optimising for the payout change the bracket? Detected via
    # the P(1st) of the selected bracket: selection under a broader payout can
    # only lower P(1st) if it picked something different.
    wta = per_structure["winner_take_all"]["per_year_p_first"]
    changed = {
        s: sum(1 for y in wta if abs(per_structure[s]["per_year_p_first"][y] - wta[y]) > 1e-12)
        for s in PAYOUT_STRUCTURES
        if s != "winner_take_all"
    }
    payload = {
        "provenance": _provenance(),
        "measurement": "prize-weighted selection objective vs P(1st)",
        "seasons": list(years),
        "structures": per_structure,
        "seasons_where_selection_changed": changed,
        "n_seasons": len(wta),
        "interpretation": (
            "A season counts as 'changed' when selecting for the payout produced a bracket "
            "with a different P(1st) from the winner-take-all pick -- a sufficient but not "
            "necessary signal, since two different brackets can share a P(1st)."
        ),
        "why_no_adoption_threshold": (
            "This is a correctness feature, not an improvement. Reporting P(1st) to a user "
            "whose pool pays its top three answers a question they did not ask, whatever the "
            "numbers say. The default stays winner_take_all so the published headline is "
            "unchanged; the flag exists so a user with a different pool can ask their own "
            "question."
        ),
    }
    _write(out_dir / "payout_objective.json", payload)
    return payload


def _marginal_stats(per_k: Dict[str, Any]) -> Dict[str, Any]:
    """Paired season-by-season statistics for each additional entry.

    Paired, not a difference of means: every entry count is measured on the
    same 14 seasons, and season-to-season variance dwarfs the effect. Comparing
    the unpaired means makes a real +4pp gain look like noise, and makes noise
    look like a gain -- the raw sequence here is +4.2, +5.0, +1.9pp, which reads
    as a curve that goes up before it goes down until the pairing is restored.
    """
    import numpy as np
    from scipy import stats as sp_stats

    out: Dict[str, Any] = {}
    counts = sorted(int(k) for k in per_k)
    for a, b in zip(counts, counts[1:]):
        va = per_k[str(a)]["per_year_p_any_first"]
        vb = per_k[str(b)]["per_year_p_any_first"]
        seasons = sorted(set(va) & set(vb))
        diff = np.array([vb[s] - va[s] for s in seasons])
        n = len(diff)
        half = float(sp_stats.t.ppf(0.975, n - 1) * diff.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        _t, p = sp_stats.ttest_rel([vb[s] for s in seasons], [va[s] for s in seasons])
        out[f"{a}->{b}"] = {
            "p_any_first_pp": round(float(diff.mean()) * 100, 3),
            "ci95_pp": [round((float(diff.mean()) - half) * 100, 3), round((float(diff.mean()) + half) * 100, 3)],
            "p_value": round(float(p), 4),
            "seasons_improved": int((diff > 0).sum()),
            "n_seasons": n,
            "distinguishable_from_zero": bool(p < 0.05),
            "expected_prize": round(
                per_k[str(b)]["mean_expected_prize"] - per_k[str(a)]["mean_expected_prize"], 5
            ),
        }

    # The economics a user actually needs: total prize rises with every entry,
    # but each entry costs a fee, so what matters is the return PER entry.
    out["expected_prize_per_entry"] = {
        str(k): round(per_k[str(k)]["mean_expected_prize"] / k, 5) for k in counts
    }
    return out


def run_multi_entry(years, workers, out_dir) -> Dict[str, Any]:
    """What is a 2nd, 3rd, 4th bracket worth?"""
    per_k: Dict[str, Any] = {}
    for k in ENTRY_COUNTS:
        print(f"\n[multi-entry] n_entries={k}")
        t0 = time.time()
        res = _run([HEADLINE_MODE, BASELINE_MODE], years, workers, n_entries=k)
        p1 = _by_mode(res, HEADLINE_MODE, "p_first")
        prize = _by_mode(res, HEADLINE_MODE, "expected_prize")
        per_k[str(k)] = {
            "per_year_p_any_first": {str(y): v for y, v in sorted(p1.items())},
            "mean_p_any_first": _mean(p1),
            "mean_expected_prize": _mean(prize),
            "wall_time_minutes": round((time.time() - t0) / 60, 1),
        }
        print(
            f"[multi-entry] k={k}: P(any 1st) = {per_k[str(k)]['mean_p_any_first']:.4f}, "
            f"E[prize] = {per_k[str(k)]['mean_expected_prize']:.4f}"
        )

    marginal = _marginal_stats(per_k)

    payload = {
        "provenance": _provenance(),
        "measurement": "marginal value of additional pool entries",
        "seasons": list(years),
        "entry_counts": list(ENTRY_COUNTS),
        "by_entry_count": per_k,
        "marginal_gains": marginal,
        "pool_size_held_fixed": (
            "k entries occupy k of the pool's seats, so the opponent count drops by k-1 rather "
            "than the field growing. Letting the field grow instead would flatter every "
            "additional entry by diluting the competition."
        ),
        "not_best_of_k": (
            "The k brackets are chosen jointly and BEFORE any outcome is known, and every one "
            "is scored. This is not best-of-k with hindsight -- the error "
            "scripts/real_pool_placement.py exists to prevent ('nobody submits 50 brackets and "
            "keeps only the winner')."
        ),
        "no_adoption": (
            "Whether another entry is worth its fee depends on the fee, which this project "
            "does not know. The marginal-gain curve is the deliverable; the decision is the "
            "user's."
        ),
    }
    _write(out_dir / "multi_entry_value.json", payload)
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool-factor", action="store_true")
    parser.add_argument("--payout", action="store_true")
    parser.add_argument("--multi-entry", action="store_true")
    parser.add_argument(
        "--reanalyse-multi-entry",
        action="store_true",
        help="Recompute the derived statistics from the stored per-season numbers without "
        "re-running the backtest. Changing how a measurement is ANALYSED should not require "
        "repeating the measurement.",
    )
    parser.add_argument("--years", type=int, nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    if not any([args.pool_factor, args.payout, args.multi_entry, args.reanalyse_multi_entry]):
        parser.error("pick at least one of --pool-factor / --payout / --multi-entry")

    if args.reanalyse_multi_entry:
        path = args.out_dir / "multi_entry_value.json"
        with open(path) as f:
            payload = json.load(f)
        payload["marginal_gains"] = _marginal_stats(payload["by_entry_count"])
        payload["reanalysed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _write(path, payload)
        for k, v in payload["marginal_gains"].items():
            if k == "expected_prize_per_entry":
                print(f"  E[prize] per entry: {v}")
            else:
                print(
                    f"  {k}: {v['p_any_first_pp']:+.2f}pp  95% CI {v['ci95_pp']}  "
                    f"p={v['p_value']}  wins {v['seasons_improved']}/{v['n_seasons']}"
                )
        if not (args.pool_factor or args.payout or args.multi_entry):
            return 0

    from scripts.mc_pool_backtest import EVALUATION_YEARS
    from src.governance.pool_rdof_audit import assert_not_sequestered

    years = args.years or list(EVALUATION_YEARS)
    assert_not_sequestered(years, context="pool-tool feature measurement")

    if args.pool_factor:
        run_pool_factor(years, args.workers, args.out_dir)
    if args.payout:
        run_payout(years, args.workers, args.out_dir)
    if args.multi_entry:
        run_multi_entry(years, args.workers, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
