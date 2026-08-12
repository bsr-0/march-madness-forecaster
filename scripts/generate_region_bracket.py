"""Generate the meta_region bracket for 2026 and export to docs/data/.

Uses the region_top_n construction mode (region-level beam search on Torvik
round probabilities) — the same algorithm meta_region_poolaware is built on
top of. This replaces the client-side Barthag-only approximation the UI's
"stat" tab used to fall back on.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._bracket_export_common import (
    build_bracket_json,
    load_team_names,
    resolve_pool_consensus,
)
from scripts.mc_pool_backtest import (
    ESPN_SCORING,
    _load_torvik_barthag,
    build_torvik_round_probabilities,
    build_espn_pick_distribution,
    load_seeds_and_regions,
)
from src.optimization.bracket_construction import construct_bracket

YEAR = 2026
OUT_PATH = PROJECT_ROOT / "docs" / "data" / "bracket_2026_region.json"


def main():
    seeds, regions = load_seeds_and_regions(YEAR)
    if not seeds:
        print(f"ERROR: no seeds found for {YEAR}")
        sys.exit(1)

    barthag = _load_torvik_barthag(YEAR, seeds)
    round_probs = build_torvik_round_probabilities(seeds, regions, barthag)

    try:
        pick_dist = build_espn_pick_distribution(YEAR, seeds)
    except FileNotFoundError:
        pick_dist = {}

    picks, champion, final_four, ev, _var = construct_bracket(
        mode="region_top_n",
        seeds=seeds,
        regions=regions,
        round_probs=round_probs,
        public_picks=pick_dist,
        risk_level=0.5,
        pool_size=30,
        scoring_system=dict(ESPN_SCORING),
    )

    team_names = load_team_names()
    pool_pick_dist, opponent_source = resolve_pool_consensus(seeds, YEAR)
    print(f"  Opponent field for display: {opponent_source or 'unavailable'}")

    rounds = build_bracket_json(seeds, regions, barthag, round_probs, picks, team_names, pool_pick_dist)

    output = {
        "season": YEAR,
        "generated_at": datetime.now().strftime("%Y-%m-%d"),
        "model": "Torvik Barthag — Region Top-N Beam Search",
        "n_simulations": 10000,
        "opponent_source": opponent_source,
        "rounds": rounds,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Champion: {team_names.get(champion, champion)}")
    print(f"Final Four: {[team_names.get(t, t) for t in final_four]}")
    print(f"Expected points: {ev:.1f}")
    print()
    for rnd in rounds:
        print(f"  {rnd['round_name']}: {len(rnd['games'])} games")
    print(f"\nWritten to {OUT_PATH}")


if __name__ == "__main__":
    main()
