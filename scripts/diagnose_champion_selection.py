"""Diagnostic: compare champion selection strategies across all backtest years.

For each year, loads the 12 probability bases and ESPN public picks, then
computes top-6 champion candidates with:
  - Mean P(CHAMP) across bases
  - ESPN public CHAMP pick percentage
  - Leverage ratio: P(champ) / public_pct
  - Competitive EV: P(champ) * (1 - public_pct) * 320

Shows which champion each selector (top1, leverage, competitive_EV) would
pick, plus the actual tournament champion for comparison.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction.meta_selector import (
    _load_year_data,
    _rank_champion_candidates,
)

BACKTEST_YEARS = (2011, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026)
DATA_ROOT = Path("data")

# Known actual champions for all backtest years
ACTUAL_CHAMPIONS = {
    2011: "connecticut",
    2013: "louisville",
    2014: "connecticut",
    2015: "duke",
    2016: "villanova",
    2017: "north_carolina",
    2018: "villanova",
    2019: "virginia",
    2021: "baylor",
    2022: "kansas",
    2023: "connecticut",
    2024: "connecticut",
    2025: "florida",
    2026: "florida",
}


def get_public_champ_pct(pick_dist, team_id):
    """Extract CHAMP pick percentage from pick_dist.

    pick_dist format: {team_id: {round: pct}}
    """
    if not pick_dist:
        return None
    team_data = pick_dist.get(team_id, {})
    if isinstance(team_data, dict):
        return team_data.get("CHAMP", None)
    return None


def analyze_year(year):
    """Load data for one year, compute champion candidate metrics."""
    try:
        brp, pick_dist, seeds, context, first_round, games = _load_year_data(year, DATA_ROOT)
    except Exception as e:
        return {"year": year, "error": str(e)}

    # Get top-6 candidates by mean P(CHAMP)
    candidates = _rank_champion_candidates(brp, first_round, top_n=6)

    # Get all candidates with full details
    rows = []
    for team_id, mean_champ_prob in candidates:
        pub_pct = get_public_champ_pct(pick_dist, team_id)
        seed = seeds.get(team_id, "?")

        # Collect per-base CHAMP probs
        base_probs = {}
        for base_name, team_rounds in brp.items():
            if team_id in team_rounds:
                cp = team_rounds[team_id].get("CHAMP", 0.0)
                if cp > 0:
                    base_probs[base_name] = cp

        # Compute metrics
        if pub_pct is not None and pub_pct > 0:
            leverage = mean_champ_prob / pub_pct
            competitive_ev = mean_champ_prob * (1.0 - pub_pct) * 320
        else:
            leverage = None
            competitive_ev = None

        rows.append(
            {
                "team_id": team_id,
                "seed": seed,
                "mean_champ_prob": mean_champ_prob,
                "public_pct": pub_pct,
                "leverage": leverage,
                "competitive_ev": competitive_ev,
                "n_bases": len(base_probs),
                "base_probs": base_probs,
            }
        )

    # Determine what each selector would pick
    actual = ACTUAL_CHAMPIONS.get(year, "???")

    # Top-1 by mean P(CHAMP)
    top1_pick = rows[0]["team_id"] if rows else "???"

    # Leverage selector: highest P(champ) / public_pct
    leverage_pick = top1_pick  # fallback
    best_lev = -1.0
    for r in rows:
        if r["leverage"] is not None and r["leverage"] > best_lev:
            best_lev = r["leverage"]
            leverage_pick = r["team_id"]

    # Competitive EV selector: highest P(champ) * (1 - public_pct) * 320
    cev_pick = top1_pick  # fallback
    best_cev = -1.0
    for r in rows:
        if r["competitive_ev"] is not None and r["competitive_ev"] > best_cev:
            best_cev = r["competitive_ev"]
            cev_pick = r["team_id"]

    # Backtest leverage (what the backtest code actually computes).
    # The backtest code has a bug: it checks "CHAMP" in pick_dist, but
    # pick_dist is {team_id: {round: pct}}, so "CHAMP" is never a key.
    # This means the leverage code falls through to the elif branch
    # which also doesn't find round names as team_id keys. So it
    # defaults to pub_pct=0.5 for all candidates.
    bt_lev_pick = top1_pick
    best_bt_lev = -1.0
    for r in rows:
        # Replicate the backtest's actual behavior
        pub_pct = 0.5  # backtest default
        # "CHAMP" in pick_dist is False (pick_dist keys are team_ids)
        # elif pick_dist: for rnd in ("CHAMP", "F4", "E8"):
        #   if rnd in pick_dist and cid in pick_dist[rnd]:  <-- rnd is never a key
        # So pub_pct stays at 0.5 for everyone
        pub_pct = max(pub_pct, 0.01)
        bt_lev = r["mean_champ_prob"] / pub_pct
        if bt_lev > best_bt_lev:
            best_bt_lev = bt_lev
            bt_lev_pick = r["team_id"]

    return {
        "year": year,
        "candidates": rows,
        "selectors": {
            "top1": top1_pick,
            "leverage": leverage_pick,
            "competitive_ev": cev_pick,
            "bt_leverage_actual": bt_lev_pick,
        },
        "actual": actual,
        "has_espn": pick_dist is not None,
        "n_bases": len(brp),
    }


def fmt_pct(val, width=7):
    if val is None:
        return " " * (width - 3) + "n/a"
    return f"{val * 100:>{width}.1f}%"


def fmt_float(val, width=7, decimals=2):
    if val is None:
        return " " * (width - 3) + "n/a"
    return f"{val:>{width}.{decimals}f}"


def main():
    print("=" * 130)
    print("CHAMPION SELECTION DIAGNOSTIC — All Backtest Years")
    print("=" * 130)

    summary = []
    all_results = []

    for year in BACKTEST_YEARS:
        result = analyze_year(year)
        all_results.append(result)

        if "error" in result:
            print(f"\n{year}: ERROR — {result['error']}")
            continue

        actual = result["actual"]
        sel = result["selectors"]

        print(f"\n{'=' * 130}")
        print(
            f"  {year}  |  Bases: {result['n_bases']}  |  ESPN picks: {'YES' if result['has_espn'] else 'NO'}  |  Actual champion: {actual}"
        )
        print(f"{'=' * 130}")
        print(
            f"  {'Rank':>4}  {'Team':<25} {'Seed':>4}  {'P(CHAMP)':>9} {'Public%':>9} {'Leverage':>9} {'Comp.EV':>9} {'#Bases':>6}  Selection"
        )
        print(
            f"  {'----':>4}  {'----':<25} {'----':>4}  {'---------':>9} {'-------':>9} {'--------':>9} {'-------':>9} {'------':>6}  ---------"
        )

        for i, r in enumerate(result["candidates"]):
            tid = r["team_id"]
            markers = []
            if tid == sel["top1"]:
                markers.append("T1")
            if tid == sel["leverage"]:
                markers.append("LEV")
            if tid == sel["competitive_ev"]:
                markers.append("CEV")
            if tid == actual:
                markers.append("ACTUAL")
            marker_str = " ".join(markers)

            print(
                f"  {i + 1:>4}  {tid:<25} {r['seed']:>4}  "
                f"{fmt_pct(r['mean_champ_prob'], 8)} "
                f"{fmt_pct(r['public_pct'], 8)} "
                f"{fmt_float(r['leverage'], 8)} "
                f"{fmt_float(r['competitive_ev'], 8)} "
                f"{r['n_bases']:>6}  {marker_str}"
            )

        # Show per-base breakdown for top-3
        print(f"\n  Per-base P(CHAMP) for top-3 candidates:")
        for i, r in enumerate(result["candidates"][:3]):
            probs_str = "  ".join(f"{b}={p:.3f}" for b, p in sorted(r["base_probs"].items()))
            print(f"    {r['team_id']}: {probs_str}")

        # Selection summary
        top1_correct = sel["top1"] == actual
        lev_correct = sel["leverage"] == actual
        cev_correct = sel["competitive_ev"] == actual
        bt_correct = sel["bt_leverage_actual"] == actual

        summary.append(
            {
                "year": year,
                "actual": actual,
                "top1": sel["top1"],
                "top1_correct": top1_correct,
                "leverage": sel["leverage"],
                "lev_correct": lev_correct,
                "cev": sel["competitive_ev"],
                "cev_correct": cev_correct,
                "bt_leverage": sel["bt_leverage_actual"],
                "bt_correct": bt_correct,
            }
        )

        print(f"\n  Selector picks:")
        print(f"    Top-1 (highest mean P):   {sel['top1']:<25} {'CORRECT' if top1_correct else 'wrong'}")
        print(f"    Leverage (P/pub%):        {sel['leverage']:<25} {'CORRECT' if lev_correct else 'wrong'}")
        print(f"    Competitive EV:           {sel['competitive_ev']:<25} {'CORRECT' if cev_correct else 'wrong'}")
        print(f"    BT leverage (buggy):      {sel['bt_leverage_actual']:<25} {'CORRECT' if bt_correct else 'wrong'}")

    # Summary table
    print(f"\n\n{'=' * 130}")
    print("SUMMARY: Selector Performance Across All Years")
    print(f"{'=' * 130}")
    print(f"  {'Year':>6}  {'Actual':<20} {'Top-1':<20} {'Leverage':<20} {'Comp.EV':<20} {'BT-Lev(buggy)':<20}")
    print(f"  {'----':>6}  {'------':<20} {'-----':<20} {'--------':<20} {'-------':<20} {'-------------':<20}")

    for s in summary:

        def mark(pick, correct):
            sym = "+" if correct else "-"
            return f"{sym} {pick}"

        print(
            f"  {s['year']:>6}  {s['actual']:<20} "
            f"{mark(s['top1'], s['top1_correct']):<20} "
            f"{mark(s['leverage'], s['lev_correct']):<20} "
            f"{mark(s['cev'], s['cev_correct']):<20} "
            f"{mark(s['bt_leverage'], s['bt_correct']):<20}"
        )

    # Totals
    n = len(summary)
    t1_count = sum(1 for s in summary if s["top1_correct"])
    lev_count = sum(1 for s in summary if s["lev_correct"])
    cev_count = sum(1 for s in summary if s["cev_correct"])
    bt_count = sum(1 for s in summary if s["bt_correct"])

    print(f"  {'':>6}  {'':>20} {'':>20} {'':>20} {'':>20}")
    print(f"  {'Total':>6}  {'':>20} {t1_count}/{n:<18} {lev_count}/{n:<18} {cev_count}/{n:<18} {bt_count}/{n:<18}")

    # Disagreement analysis
    print(f"\n\n{'=' * 130}")
    print("DISAGREEMENT ANALYSIS: Years where selectors diverge")
    print(f"{'=' * 130}")
    for s in summary:
        picks = {s["top1"], s["leverage"], s["cev"]}
        if len(picks) > 1:
            print(f"\n  {s['year']}: Actual={s['actual']}")
            print(f"    Top-1:       {s['top1']:<25} {'CORRECT' if s['top1_correct'] else 'wrong'}")
            print(f"    Leverage:    {s['leverage']:<25} {'CORRECT' if s['lev_correct'] else 'wrong'}")
            print(f"    Comp.EV:     {s['cev']:<25} {'CORRECT' if s['cev_correct'] else 'wrong'}")

    all_agree = all(len({s["top1"], s["leverage"], s["cev"]}) == 1 for s in summary)
    if all_agree:
        print("  No disagreements found — all three selectors pick the same champion every year.")

    # Bug analysis
    print(f"\n\n{'=' * 130}")
    print("BUG ANALYSIS: Backtest leverage code behavior")
    print(f"{'=' * 130}")
    print("  The backtest's meta_gbm_champ_leverage code checks 'CHAMP' in pick_dist,")
    print("  but pick_dist is {team_id: {round: pct}}, so 'CHAMP' is never a top-level key.")
    print("  The elif branch checks 'rnd in pick_dist' which also fails (round names != team IDs).")
    print("  Result: pub_pct defaults to 0.5 for ALL candidates, making leverage = P(champ) / 0.5")
    print("  which is monotonic with P(champ) — so leverage always picks the same as top-1.")

    bt_lev_same_as_top1 = all(s["bt_leverage"] == s["top1"] for s in summary)
    print(f"\n  BT-leverage == Top-1 in all years: {bt_lev_same_as_top1}")
    if not bt_lev_same_as_top1:
        for s in summary:
            if s["bt_leverage"] != s["top1"]:
                print(f"    {s['year']}: top1={s['top1']}, bt_leverage={s['bt_leverage']}")

    # Actual champion analysis
    print(f"\n\n{'=' * 130}")
    print("ACTUAL CHAMPION IN CANDIDATE LIST?")
    print(f"{'=' * 130}")
    for r in all_results:
        if "error" in r:
            continue
        actual = r["actual"]
        cand_ids = [c["team_id"] for c in r["candidates"]]
        if actual in cand_ids:
            rank = cand_ids.index(actual) + 1
            cand = r["candidates"][rank - 1]
            print(
                f"  {r['year']}: {actual} ranked #{rank} "
                f"(P={cand['mean_champ_prob']:.3f}, "
                f"pub={cand['public_pct'] * 100:.1f}% "
                if cand["public_pct"] is not None
                else f"pub=n/a ",
                end="",
            )
            if cand["competitive_ev"] is not None:
                print(f"CEV={cand['competitive_ev']:.1f})")
            else:
                print("CEV=n/a)")
        else:
            print(f"  {r['year']}: {actual} NOT in top-6 candidates")

    # Save artifact
    out = {
        "summary": summary,
        "years": [],
    }
    for r in all_results:
        if "error" in r:
            out["years"].append({"year": r["year"], "error": r["error"]})
            continue
        out["years"].append(
            {
                "year": r["year"],
                "n_bases": r["n_bases"],
                "has_espn": r["has_espn"],
                "actual": r["actual"],
                "selectors": r["selectors"],
                "candidates": [
                    {
                        "team_id": c["team_id"],
                        "seed": c["seed"],
                        "mean_champ_prob": round(c["mean_champ_prob"], 4),
                        "public_pct": round(c["public_pct"], 4) if c["public_pct"] is not None else None,
                        "leverage": round(c["leverage"], 4) if c["leverage"] is not None else None,
                        "competitive_ev": round(c["competitive_ev"], 2) if c["competitive_ev"] is not None else None,
                    }
                    for c in r["candidates"]
                ],
            }
        )

    out_path = PROJECT_ROOT / "artifacts" / "champion_selection_diagnostic.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved artifact to {out_path}")


if __name__ == "__main__":
    main()
