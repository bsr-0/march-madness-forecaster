"""Step 8: apply the FROZEN qualification rule (PREREGISTRATION_QUALIFICATION.md, pinned in
referee_audit.QUALIFICATION) to the corrected, walk-forward referee tables. Nothing about the rule,
the incumbent, the referee set or the ordering is changed. No P(1st) is computed."""
import json, sys, logging, subprocess, datetime, numpy as np
sys.path.insert(0, '.'); logging.disable(logging.WARNING)
from src.evaluation import referee_audit as ra
from src.data.seed_pick_model import _recent_win_rates
import scripts.mc_pool_backtest as M
from scripts._common import load_tournament_results

years = [y for y in range(2011, 2027) if y != 2020 and y != 2026]   # evaluation seasons; 2026 excluded as integration season
print("== 1. frozen rule (from code; tests pin it to PREREGISTRATION_QUALIFICATION.md) ==")
print(json.dumps(ra.QUALIFICATION, indent=1, default=str))
print("G1: paired season-bootstrap CI of (log_loss - log2) entirely < 0.  G2: mean(log_loss - incumbent) <= 0 AND mean(brier - incumbent) <= 0.")
print("DISQUALIFIED: G1 fails or log-loss delta vs incumbent CI entirely > 0. PROVISIONAL: otherwise. Primary requires full coverage. seed = incumbent, QUALIFIED by construction.")

print("\n== 2. incumbent: walk-forward check ==")
inc_ok = True
for y in years:
    seeds, regions = M.load_seeds_and_regions(y); games = load_tournament_results(y); M.resolve_first_four(games, seeds, regions)
    tab = _recent_win_rates(y)
    # the table for Y must equal a tally that excludes seasons >= Y: check it changes when Y's season is added
    nxt = _recent_win_rates(y + 1)
    inc_ok &= (tab != nxt) or (y == 2011 and tab == {})
print("   seed table for season Y is built with as_of=Y (excludes Y and later):", inc_ok, "| 2011 table cells:", len(_recent_win_rates(2011)))

print("\n== 3/4. referee inventory and coverage per season (before any qualification) ==")
rows = []
for y in years:
    r = ra.calibration_rows(y); rows.append(r)
    cal = r['referee_calibration']; d = r['seed_table_diagnostics']; mv2 = r['market_v2_diagnostics']
    print(f"   {y}: games={cal['seed']['n_games']} referees={sorted(cal)} | seed: {d['games_on_empirical_cell']} empirical/{d['games_on_logistic_curve']} logistic | market_v2 fallback teams: {len(mv2.get('fallback_teams', []))} ({mv2.get('n_games','?')} games)")
present = {}
for r in rows:
    for ref in r['referee_calibration']: present.setdefault(ref, []).append(r['year'])
print("   coverage:", {k: f"{len(v)}/{len(years)}" for k, v in present.items()})

print("\n== 5-7. frozen gate on the corrected tables ==")
gate = ra.qualification_gate(rows, years)
indep = ra.choose_independent_referee(gate)
hdr = f"   {'referee':10} {'cov':6} {'logloss':8} {'brier':7} {'G1 CI(ll-log2)':22} {'G2 mean dLL':11} {'G2 dLL CI':22} {'G2 mean dBr':11} status"
print(hdr)
for ref, g in gate.items():
    def ci(x): return f"[{x['ci_lo']:+.4f},{x['ci_hi']:+.4f}]" if x else "-"
    print(f"   {ref:10} {len(g.get('seasons',[])):2d}/{len(years):<3} {g.get('mean_log_loss', float('nan')):.4f}   {g.get('mean_brier', float('nan')):.4f}  {ci(g.get('g1')):22} {g.get('g2_ll',{}).get('mean',float('nan')):+.4f}     {ci(g.get('g2_ll')):22} {g.get('g2_br',{}).get('mean',float('nan')):+.4f}     {g['status']}{' (primary-eligible)' if g.get('primary_eligible') else ''}")
print("   independent referee by the frozen order", ra.QUALIFICATION['independent_referee_order'], "->", indep)
out = {"generated_at": datetime.datetime.utcnow().isoformat(), "commit_base": subprocess.run(['git','rev-parse','--short','HEAD'],capture_output=True,text=True).stdout.strip(),
       "note": "Step 8 of the 2026-09 methodology audit. Frozen rule applied unchanged to walk-forward referees (seed as_of=year, noseed as_of=year, real F4 topology). The invalidated 2026-09-14 run is preserved at artifacts/referee_audit/qualification.json.",
       "years": years, "frozen_rule": ra.QUALIFICATION, "criteria": ra.CRITERIA, "rows": rows, "gate": gate, "independent_referee": indep,
       "fte_provenance": ra.FTE_PROVENANCE}
json.dump(out, open('artifacts/methodology_audit/step8/qualification_v2.json', 'w'), indent=1, default=str)
print("\nwrote artifacts/methodology_audit/step8/qualification_v2.json")
