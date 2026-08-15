"""Generate the meta_region bracket for 2026 and export to docs/data/.

Uses the region_top_n construction mode (region-level beam search) on the
selected probability base (--prob-base torvik|elo|ap|upset) — the same
algorithm meta_region_poolaware is built on top of. Torvik is the default,
backtested base (docs/app.js STRATEGIES['stat'], 8.0% P(1st)). elo/ap
are exploratory lenses on the same construction; upset additionally
forces risk_level=1.0 (max contrarian weighting) instead of the normal
0.5 — see scripts/prob_base_variants.py.
"""

import argparse
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
from scripts.prob_base_variants import load_prob_base, MODEL_LABELS, RISK_LEVEL
from src.optimization.bracket_construction import construct_bracket

YEAR = 2026
OUT_DIR = PROJECT_ROOT / "docs" / "data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prob-base", choices=["torvik", "elo", "ap", "upset"], default="torvik")
    args = parser.parse_args()

    seeds, regions = load_seeds_and_regions(YEAR)
    if not seeds:
        print(f"ERROR: no seeds found for {YEAR}")
        sys.exit(1)

    barthag = _load_torvik_barthag(YEAR, seeds)
    torvik_rp = build_torvik_round_probabilities(seeds, regions, barthag)
    rating, round_probs = load_prob_base(args.prob_base, YEAR, seeds, regions, torvik_rp, barthag)

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
        risk_level=RISK_LEVEL[args.prob_base],
        pool_size=30,
        scoring_system=dict(ESPN_SCORING),
    )

    team_names = load_team_names()
    pool_pick_dist, opponent_source = resolve_pool_consensus(seeds, YEAR)
    print(f"  Opponent field for display: {opponent_source or 'unavailable'}")

    rounds = build_bracket_json(seeds, regions, rating, round_probs, picks, team_names, pool_pick_dist)

    suffix = "" if args.prob_base == "torvik" else f"_{args.prob_base}"
    out_path = OUT_DIR / f"bracket_2026_region{suffix}.json"

    output = {
        "season": YEAR,
        "generated_at": datetime.now().strftime("%Y-%m-%d"),
        "model": f"{MODEL_LABELS[args.prob_base]} — Region Top-N Beam Search",
        "n_simulations": 10000,
        "opponent_source": opponent_source,
        "rounds": rounds,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Champion: {team_names.get(champion, champion)}")
    print(f"Final Four: {[team_names.get(t, t) for t in final_four]}")
    print(f"Expected points: {ev:.1f}")
    print()
    for rnd in rounds:
        print(f"  {rnd['round_name']}: {len(rnd['games'])} games")
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
