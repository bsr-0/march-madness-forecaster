"""Referee qualification audit, then the referee matrix over the qualified set.

    python -m scripts.referee_qualification_audit [--years ...]

Pre-registration: artifacts/referee_audit/PREREGISTRATION_QUALIFICATION.md.

Phase 1 builds every season's referees and scores their RAW pairwise tables
on the real games. The gate (G1 beat a coin flip, G2 not worse than the
incumbent seed table, season-level paired) is applied to that and nothing
else; no strategy P(1st) exists yet when the gate is decided.

Phase 2 runs the first audit's matrix and LORO with the criterion set equal
to the qualified full-coverage referees and the independent referee chosen
by the pre-registered order. Every referee, qualified or not, stays in the
report.

Outputs under artifacts/referee_audit/:
    qualification.json, seasons_qualified/season_{year}.json,
    referee_matrix_qualified.json, REPORT_QUALIFIED.md, run_qualification_{stamp}.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from scripts.referee_robustness_audit import CANONICAL_LOG, OUT_DIR, _ci, _git_commit, write_report  # noqa: E402
from src.evaluation import referee_audit as ra  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", default=None)
    ap.add_argument("--n-eval-trials", type=int, default=ra.CRITERIA["n_eval_trials"])
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--no-parity", action="store_true")
    args = ap.parse_args()

    from scripts.mc_pool_backtest import EVALUATION_YEARS

    years = [int(y) for y in args.years.split(",")] if args.years else list(EVALUATION_YEARS)
    out_dir = Path(args.out_dir)
    (out_dir / "seasons_qualified").mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, s):
            for st in self.streams:
                st.write(s)
                st.flush()

        def flush(self):
            for st in self.streams:
                st.flush()

    with open(out_dir / f"run_qualification_{stamp}.txt", "w") as log:
        sys.stdout = Tee(sys.__stdout__, log)
        try:
            print(f"Referee qualification audit  {stamp}  commit {_git_commit()[:12]}  years={years}")
            # ---------------- Phase 1: calibration only ----------------
            print("Phase 1: referee calibration on the real games (no strategy P(1st) computed)")
            rows = []
            for y in years:
                r = ra.calibration_rows(y)
                rows.append(r)
                cal = r["referee_calibration"]
                print(f"  {y}  " + "  ".join(f"{k}={v['log_loss']:.3f}" for k, v in cal.items())
                      + f"  market_v2 fallback={len(r['market_v2_diagnostics'].get('fallback_teams', []))}")
            gate = ra.qualification_gate(rows, years)
            independent = ra.choose_independent_referee(gate)
            criterion = [r for r in ra.ALL_REFEREE_ORDER if gate.get(r, {}).get("primary_eligible")]
            print()
            print(f"{'referee':10s} {'status':13s} primary  log_loss  brier    G1 vs coin flip              G2 vs seed (log loss)")
            for r, g in gate.items():
                print(f"{r:10s} {g['status']:13s} {'yes' if g['primary_eligible'] else 'no ':7s}  {g['mean_log_loss']:.4f}   {g['mean_brier']:.4f}   "
                      f"{_ci(g['G1_vs_coin_flip_log_loss']):28s} {_ci(g['G2_vs_incumbent_log_loss'])}")
            print(f"criterion referees: {criterion}   independent referee: {independent}")
            (out_dir / "qualification.json").write_text(json.dumps({
                "generated_at": stamp, "commit": _git_commit(), "years": years, "spec": ra.QUALIFICATION,
                "rows": rows, "gate": gate, "criterion_referees": criterion, "independent_referee": independent,
            }, indent=1))
            if not criterion or independent is None:
                print("No qualified full-coverage referee beyond the incumbent, or no independent referee: phase 2 not run.")
                return 0

            # ---------------- Phase 2: matrix + LORO over the qualified set ----------------
            print("\nPhase 2: referee matrix and LORO over the qualified set")
            expected = {} if args.no_parity else ra.expected_labels_from_log(CANONICAL_LOG)
            full_refs = tuple(r for r in ra.ALL_REFEREE_ORDER if r not in ra.SUPPLEMENTARY_REFEREES and r not in ra.REFEREE_ONLY)
            cfg = ra.AuditConfig(
                n_eval_trials=args.n_eval_trials, expected_labels=expected, candidates_dir=str(out_dir / "candidates"),
                criterion_referees=tuple(criterion), independent_referee=independent, selection_referees=full_refs,
            )
            seasons = []
            for y in years:
                rec = ra.run_season(y, cfg)
                seasons.append(rec)
                (out_dir / "seasons_qualified" / f"season_{y}.json").write_text(json.dumps(rec, indent=1))
                m = rec["metrics"]
                print(f"  {y}  pool={rec['pool_size']}  chosen={rec['choices']['seed']['production']['label']:<28} parity={'ok' if rec['parity']['ok'] else 'FAIL'}  "
                      + "  ".join(f"{r}={m[r][ra.PRODUCTION_STRATEGY]['p_first']:.3f}" for r in rec["referees_available"]))
            tables = {m: ra.aggregate(seasons, m) for m in ("p_first", "mean_rank", "top3", "top10", "mean_score")}
            strategies = sorted({st for s in seasons for r in s["metrics"].values() for st in r})
            premium = ra.self_referee_premium(seasons, strategies, referees=criterion)
            loro = ra.loro_table(seasons, referees=criterion)
            criteria = ra.evaluate_criteria(tables["p_first"], premium, loro, criterion_referees=criterion, independent_referee=independent)
            wc = ra.paired_bootstrap(np.array([
                s["selection_p1"]["seed"][s["choices"]["seed"]["production"]["index"]] - s["metrics"]["seed"][ra.PRODUCTION_STRATEGY]["p_first"]
                for s in seasons
            ]))
            result = {
                "generated_at": stamp, "commit": _git_commit(), "years": [s["year"] for s in seasons],
                "config": {"n_eval_trials": cfg.n_eval_trials, "pa_trials": cfg.pa_trials, "n_stochastic": cfg.n_stochastic,
                           "eval_seed": cfg.eval_seed, "stochastic_seed": cfg.stochastic_seed, "selection_referees": list(full_refs)},
                "criteria_spec": ra.CRITERIA, "qualification_spec": ra.QUALIFICATION, "qualification": gate,
                "criterion_referees": criterion, "referees": list(tables["p_first"].keys()), "strategies": strategies,
                "parity_all_ok": all(s["parity"]["ok"] for s in seasons), "tables": tables,
                "self_referee_premium": premium, "loro": loro, "winners_curse": wc,
                "referee_calibration": ra.pooled_calibration(seasons), "criteria": criteria,
                "non_independence": ra.NON_INDEPENDENCE, "seasons": seasons,
            }
            (out_dir / "referee_matrix_qualified.json").write_text(json.dumps(result, indent=1))
            write_report(result, out_dir / "REPORT_QUALIFIED.md")
            print()
            print(f"VERDICT (qualified set {criterion}, independent={independent}): {criteria['verdict']}   "
                  f"C1={criteria['C1_cross_referee_edge']['status']}  C2={criteria['C2_self_referee_premium']['status']}  "
                  f"C3={criteria['C3_leave_one_referee_out']['status']}")
            print(f"wrote {out_dir / 'referee_matrix_qualified.json'} and {out_dir / 'REPORT_QUALIFIED.md'}")
        finally:
            sys.stdout = sys.__stdout__
    return 0


if __name__ == "__main__":
    sys.exit(main())
