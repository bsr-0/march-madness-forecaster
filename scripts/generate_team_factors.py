"""Generate docs/data/team_factors.json — per-team roster/coaching deltas.

This is NOT a bracket strategy. roster_adj and coach_adj were backtested
2026-08-13 (artifacts/experiments/experiment_custom_20260813_222650.json)
and found statistically indistinguishable from plain torvik on P(1st) —
they don't move a bracket's odds. What they do is nudge individual teams'
round-advancement probabilities by a few points based on roster talent
(top-5 WARP) or coaching tournament experience, which can highlight teams
worth a second look even though it doesn't change which bracket wins pools.

Exports, per team: baseline/roster-adjusted/coach-adjusted Final Four and
Elite Eight probability, plus the raw inputs (WARP z-score, prior coach
tournament appearances) so the UI can show why a team moved.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._bracket_export_common import load_team_names
from scripts.mc_pool_backtest import (
    _load_torvik_barthag,
    build_torvik_round_probabilities,
    load_seeds_and_regions,
)
from src.prediction.coach_adj_probabilities import (
    build_coach_adj_round_probs,
    load_coach_experience,
)
from src.prediction.roster_adj_probabilities import (
    build_roster_adj_round_probs,
    load_team_talent,
)

YEAR = 2026
OUT_PATH = PROJECT_ROOT / "docs" / "data" / "team_factors.json"
DATA_ROOT = PROJECT_ROOT / "data"


def main():
    seeds, regions = load_seeds_and_regions(YEAR)
    if not seeds:
        print(f"ERROR: no seeds found for {YEAR}")
        sys.exit(1)

    barthag = _load_torvik_barthag(YEAR, seeds)
    baseline_rp = build_torvik_round_probabilities(seeds, regions, barthag)

    team_talent = load_team_talent(YEAR, seeds.keys(), DATA_ROOT)
    roster_rp = build_roster_adj_round_probs(baseline_rp, team_talent)

    coach_experience = load_coach_experience(YEAR, seeds.keys(), DATA_ROOT)
    coach_rp = build_coach_adj_round_probs(baseline_rp, coach_experience)

    # League mean/std for WARP, matching build_roster_adj_round_probs'
    # internal z-score so the UI can show "why" (raw z, not just the delta).
    talent_vals = list(team_talent.values())
    league_mean = sum(talent_vals) / len(talent_vals) if talent_vals else 0.0
    league_std = (
        (sum((v - league_mean) ** 2 for v in talent_vals) / len(talent_vals)) ** 0.5 if talent_vals else 0.0
    )

    team_names = load_team_names()

    teams = []
    for tid, seed in seeds.items():
        warp = team_talent.get(tid)
        z = (warp - league_mean) / league_std if warp is not None and league_std > 1e-9 else 0.0
        teams.append(
            {
                "team_id": tid,
                "team_name": team_names.get(tid, tid),
                "seed": seed,
                "region": regions.get(tid, ""),
                "baseline_f4_pct": round(baseline_rp.get(tid, {}).get("F4", 0.0) * 100, 2),
                "baseline_e8_pct": round(baseline_rp.get(tid, {}).get("E8", 0.0) * 100, 2),
                "roster_f4_pct": round(roster_rp.get(tid, {}).get("F4", 0.0) * 100, 2),
                "roster_e8_pct": round(roster_rp.get(tid, {}).get("E8", 0.0) * 100, 2),
                "roster_warp_z": round(z, 2),
                "coach_f4_pct": round(coach_rp.get(tid, {}).get("F4", 0.0) * 100, 2),
                "coach_e8_pct": round(coach_rp.get(tid, {}).get("E8", 0.0) * 100, 2),
                "coach_prior_apps": coach_experience.get(tid, 0),
            }
        )

    output = {
        "season": YEAR,
        "generated_at": datetime.now().strftime("%Y-%m-%d"),
        "note": (
            "roster_adj and coach_adj are backtested-neutral on P(1st) "
            "(experiment_custom_20260813_222650.json) — these numbers highlight "
            "teams these factors favor, they do not represent a validated "
            "alternative strategy or change your odds of winning a pool."
        ),
        "teams": teams,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    top_roster = sorted(teams, key=lambda t: t["roster_f4_pct"] - t["baseline_f4_pct"], reverse=True)[:5]
    top_coach = sorted(teams, key=lambda t: t["coach_f4_pct"] - t["baseline_f4_pct"], reverse=True)[:5]
    print("Top roster-talent movers (F4%):")
    for t in top_roster:
        print(f"  {t['team_name']:25s} {t['baseline_f4_pct']:.1f} -> {t['roster_f4_pct']:.1f}")
    print("Top coaching-experience movers (F4%):")
    for t in top_coach:
        print(f"  {t['team_name']:25s} {t['baseline_f4_pct']:.1f} -> {t['coach_f4_pct']:.1f} ({t['coach_prior_apps']} apps)")
    print(f"\nWritten to {OUT_PATH}")


if __name__ == "__main__":
    main()
