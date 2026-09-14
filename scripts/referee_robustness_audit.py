"""Referee robustness audit: is meta_region_poolaware's edge referee-specific?

    python -m scripts.referee_robustness_audit [--years 2011,2013] [--workers 4]

Pre-registration (thresholds fixed before the first run):
    artifacts/referee_audit/PREREGISTRATION.md
Library (all logic, unit-tested):
    src/evaluation/referee_audit.py

Outputs, all under artifacts/referee_audit/:
    seasons/season_{year}.json     raw per-season records (metrics, candidate
                                   labels, selection P(1st) per referee, choices)
    candidates/candidates_{year}.json  the frozen candidate set, pick-level
    referee_matrix.json            aggregated tables, premiums, LORO, criteria
    REPORT.md                      the human-readable report
    run_{timestamp}.txt            the console log
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import referee_audit as ra  # noqa: E402

OUT_DIR = PROJECT_ROOT / "artifacts" / "referee_audit"
CANONICAL_LOG = PROJECT_ROOT / "artifacts" / "headline_measurement" / "canonical_2011_2025_n14.txt"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def _ci(d: dict) -> str:
    return f"{d['mean']:+.3f} [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]"


def write_report(result: dict, path: Path) -> None:
    refs = result["referees"]
    crit_refs = [r for r in ra.CRITERION_REFEREES if r in refs]
    L: list[str] = []
    L.append("# Referee robustness audit: results\n")
    L.append(f"Run {result['generated_at']} at commit `{result['commit'][:12]}`. "
             f"Seasons: {', '.join(str(y) for y in result['years'])} (n = {len(result['years'])}). "
             f"{result['config']['n_eval_trials']} evaluation trials per season per referee, "
             f"{result['config']['pa_trials']} selection trials. Pre-registration: PREREGISTRATION.md.\n")
    L.append("Parity of the reproduced production selection against the canonical log: "
             + ("**all seasons match**" if result["parity_all_ok"] else "**MISMATCH**") + ".\n")

    v = result["criteria"]
    L.append("## Verdict\n")
    L.append(f"**{v['verdict']}** -- C1 {v['C1_cross_referee_edge']['status']}, "
             f"C2 {v['C2_self_referee_premium']['status']}, C3 {v['C3_leave_one_referee_out']['status']}.\n")

    # C1 table
    L.append("## Production vs seed baseline, by referee (pooled P(1st), paired delta, seasons won)\n")
    L.append("| referee | P1 seed | P1 production | delta [95% CI] | won |")
    L.append("|---|---|---|---|---|")
    for r in refs:
        t = result["tables"]["p_first"][r]
        if ra.PRODUCTION_STRATEGY not in t:
            continue
        p = t[ra.PRODUCTION_STRATEGY]
        b = t[ra.BASELINE_STRATEGY]["mean"]
        d = p.get("delta_vs_baseline")
        L.append(f"| {r} | {_fmt(b)} | {_fmt(p['mean'])} | {_ci(d) if d else 'n/a'} | {p.get('seasons_won_vs_baseline','-')}/{p['n_seasons']} |")
    L.append("")

    # Full matrix
    L.append("## Referee matrix: pooled P(1st) per strategy (rows) and referee (columns)\n")
    strategies = [s for s in result["strategies"] if not s.startswith("sel:")]
    core = [s for s in strategies if not s.startswith("cand:")]
    cands = [s for s in strategies if s.startswith("cand:")]
    for title, group in (("Strategies", core), ("Individual poolaware candidates", cands)):
        L.append(f"### {title}\n")
        L.append("| strategy | " + " | ".join(refs) + " |")
        L.append("|---|" + "---|" * len(refs))
        for s in group:
            cells = []
            for r in refs:
                e = result["tables"]["p_first"][r].get(s)
                cells.append(_fmt(e["mean"]) if e else "-")
            L.append(f"| {s} | " + " | ".join(cells) + " |")
        L.append("")

    for metric, label in (("mean_rank", "mean rank"), ("top3", "top-3 rate"), ("top10", "top-10 rate"), ("mean_score", "mean simulated score")):
        L.append(f"### {label}, core strategies\n")
        L.append("| strategy | " + " | ".join(refs) + " |")
        L.append("|---|" + "---|" * len(refs))
        for s in core:
            cells = []
            for r in refs:
                e = result["tables"][metric][r].get(s)
                cells.append((f"{e['mean']:.1f}" if metric in ("mean_rank", "mean_score") else _fmt(e["mean"])) if e else "-")
            L.append(f"| {s} | " + " | ".join(cells) + " |")
        L.append("")

    L.append("## Paired season-level deltas vs seed, P(1st), core strategies\n")
    L.append("| strategy | " + " | ".join(refs) + " |")
    L.append("|---|" + "---|" * len(refs))
    for s in core:
        if s == ra.BASELINE_STRATEGY:
            continue
        cells = []
        for r in refs:
            e = result["tables"]["p_first"][r].get(s, {})
            d = e.get("delta_vs_baseline")
            cells.append(_ci(d) if d else "-")
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## Self-referee premium (P1 under own referee minus mean under the other criterion referees)\n")
    L.append("| strategy | own | P1 own | P1 others | premium [95% CI] | relative | material | reversal under |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s, p in result["self_referee_premium"].items():
        L.append(f"| {s} | {p['own_referee']} | {_fmt(p['p1_own'])} | {_fmt(p['p1_other_mean'])} | {_ci(p['premium'])} | "
                 f"{p['relative_premium']:+.2f} | {'YES' if p['material'] else 'no'} | {', '.join(p['reversal_under']) or '-'} |")
    L.append("")

    L.append("## Leave-one-referee-out\n")
    L.append("Each row: the referee held out of selection. Columns: pooled P(1st) under that referee "
             "of the bracket chosen by each rule, with its paired edge over seed.\n")
    L.append("| held out | seed | LORO choice | self choice (in-sample) | production (seed-selected) | average-all | LORO - self | LORO=prod |")
    L.append("|---|---|---|---|---|---|---|---|")
    for h in crit_refs:
        row = result["loro"][h]
        def cell(k):
            e = row.get(k)
            return f"{_fmt(e['p1'])} ({_ci(e['edge_vs_baseline'])})" if e else "-"
        L.append(f"| {h} | {_fmt(row['baseline_p1'])} | {cell('loro')} | {cell('self')} | {cell('production')} | {cell('average_all')} | "
                 f"{_ci(row['loro_minus_self']) if 'loro_minus_self' in row else '-'} | {row.get('choice_agreement_loro_vs_production', float('nan')):.2f} |")
    L.append("")

    L.append("## Referee calibration on the real games (raw pairwise tables, play-ins excluded)\n")
    L.append("Lower log loss and Brier are better. Sharpness is mean |p - 0.5|: how far from a coin flip the referee's "
             "probabilities sit. A sharp referee that is also poorly calibrated rewards whoever agrees with it, not whoever wins pools.\n")
    L.append("| referee | log loss | Brier | sharpness | games | seasons |")
    L.append("|---|---|---|---|---|---|")
    for r in refs:
        c = result["referee_calibration"].get(r)
        if c:
            L.append(f"| {r} | {c['log_loss']:.4f} | {c['brier']:.4f} | {c['sharpness']:.3f} | {c['n_games']} | {c['n_seasons']} |")
    L.append("")

    L.append("## Winner's curse on the selection trials\n")
    wc = result["winners_curse"]
    L.append(f"Selection-trial P(1st) of the chosen candidate under seed minus its fresh-trial P(1st) under seed: "
             f"{_ci(wc)} (pooled over seasons).\n")

    L.append("## Non-independence of candidate sources and referees\n")
    for r, rows in ra.NON_INDEPENDENCE.items():
        L.append(f"**{r}**")
        for k, why in rows.items():
            L.append(f"- {k}: {why}")
        L.append("")

    L.append("## Per-season production choice and its P(1st) under each referee\n")
    L.append("| season | pool | chosen candidate | " + " | ".join(refs) + " |")
    L.append("|---|---|---|" + "---|" * len(refs))
    for s in result["seasons"]:
        y = s["year"]
        cells = [_fmt(s["metrics"][r][ra.PRODUCTION_STRATEGY]["p_first"]) if r in s["metrics"] else "-" for r in refs]
        L.append(f"| {y} | {s['pool_size']} | {s['choices'][ra.SELECTION_REFEREE]['production']['label']} | " + " | ".join(cells) + " |")
    L.append("")
    path.write_text("\n".join(L))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", default=None, help="comma-separated; default: the canonical evaluation seasons")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n-eval-trials", type=int, default=ra.CRITERIA["n_eval_trials"])
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--no-parity", action="store_true", help="skip the canonical-log parity check (smoke tests only)")
    args = ap.parse_args()

    from scripts.mc_pool_backtest import EVALUATION_YEARS

    years = [int(y) for y in args.years.split(",")] if args.years else list(EVALUATION_YEARS)
    out_dir = Path(args.out_dir)
    (out_dir / "seasons").mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"run_{stamp}.txt"

    expected = {} if args.no_parity else ra.expected_labels_from_log(CANONICAL_LOG)
    cfg = ra.AuditConfig(n_eval_trials=args.n_eval_trials, expected_labels=expected, candidates_dir=str(out_dir / "candidates"))

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

    with open(log_path, "w") as log:
        sys.stdout = Tee(sys.__stdout__, log)
        try:
            print(f"Referee robustness audit  {stamp}  commit {_git_commit()[:12]}")
            print(f"  years={years}  n_eval_trials={cfg.n_eval_trials}  pa_trials={cfg.pa_trials}  workers={args.workers}")
            print(f"  parity check: {'ON (' + str(len(expected)) + ' expected labels)' if expected else 'OFF'}")
            seasons = []

            def absorb(rec):
                seasons.append(rec)
                (out_dir / "seasons" / f"season_{rec['year']}.json").write_text(json.dumps(rec, indent=1))
                p = rec["parity"]
                m = rec["metrics"]
                cells = "  ".join(f"{r}={m[r][ra.PRODUCTION_STRATEGY]['p_first']:.3f}" for r in rec["referees_available"])
                print(f"  {rec['year']}  pool={rec['pool_size']}  chosen={rec['choices']['seed']['production']['label']:<28} "
                      f"parity={'ok' if p['ok'] else 'FAIL'}{'' if p['checked'] else '(unchecked)'}  P1(prod): {cells}")

            if args.workers <= 1:
                for y in years:
                    absorb(ra.run_season(y, cfg))
            else:
                from concurrent.futures import ProcessPoolExecutor, as_completed

                with ProcessPoolExecutor(max_workers=min(args.workers, len(years))) as ex:
                    futs = {ex.submit(ra.run_season, y, cfg): y for y in years}
                    for f in as_completed(futs):
                        absorb(f.result())
            seasons.sort(key=lambda s: s["year"])

            tables = {m: ra.aggregate(seasons, m) for m in ("p_first", "mean_rank", "top3", "top10", "mean_score")}
            strategies = sorted({st for s in seasons for r in s["metrics"].values() for st in r})
            premium = ra.self_referee_premium(seasons, strategies)
            loro = ra.loro_table(seasons)
            criteria = ra.evaluate_criteria(tables["p_first"], premium, loro)
            import numpy as np

            wc = ra.paired_bootstrap(np.array([
                s["selection_p1"]["seed"][s["choices"]["seed"]["production"]["index"]] - s["metrics"]["seed"][ra.PRODUCTION_STRATEGY]["p_first"]
                for s in seasons
            ]))
            result = {
                "generated_at": stamp,
                "commit": _git_commit(),
                "years": [s["year"] for s in seasons],
                "config": {"n_eval_trials": cfg.n_eval_trials, "pa_trials": cfg.pa_trials, "n_stochastic": cfg.n_stochastic,
                           "eval_seed": cfg.eval_seed, "stochastic_seed": cfg.stochastic_seed},
                "criteria_spec": ra.CRITERIA,
                "referees": list(tables["p_first"].keys()),
                "strategies": strategies,
                "parity_all_ok": all(s["parity"]["ok"] for s in seasons),
                "tables": tables,
                "self_referee_premium": premium,
                "loro": loro,
                "winners_curse": wc,
                "referee_calibration": ra.pooled_calibration(seasons),
                "criteria": criteria,
                "non_independence": ra.NON_INDEPENDENCE,
                "seasons": seasons,
            }
            (out_dir / "referee_matrix.json").write_text(json.dumps(result, indent=1))
            write_report(result, out_dir / "REPORT.md")
            print()
            print(f"VERDICT: {criteria['verdict']}   C1={criteria['C1_cross_referee_edge']['status']}  "
                  f"C2={criteria['C2_self_referee_premium']['status']}  C3={criteria['C3_leave_one_referee_out']['status']}")
            print(f"wrote {out_dir / 'referee_matrix.json'} and {out_dir / 'REPORT.md'}")
        finally:
            sys.stdout = sys.__stdout__
    return 0


if __name__ == "__main__":
    sys.exit(main())
