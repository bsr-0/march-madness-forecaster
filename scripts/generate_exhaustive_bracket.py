"""Generate the meta_exhaustive bracket for 2026 and export to docs/data/.

Uses exhaustive_champion construction mode (tries all 64 possible champions,
picks the bracket with the highest expected points) on the selected
probability base (--prob-base torvik|elo|ap). Torvik is the default,
backtested base (docs/app.js STRATEGIES['exhaustive'], 7.7% P(1st)).
elo/ap swap in a fully independent rating system (see
scripts/prob_base_variants.py) so the same construction algorithm can be
viewed through that lens — exploratory, not a claim that either beats
Torvik on P(1st).
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
from scripts.prob_base_variants import load_prob_base, MODEL_LABELS
from src.optimization.bracket_construction import construct_bracket

YEAR = 2026
OUT_DIR = PROJECT_ROOT / "docs" / "data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prob-base", choices=["torvik", "elo", "ap"], default="torvik")
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
        mode="exhaustive_champion",
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

    rounds = build_bracket_json(seeds, regions, rating, round_probs, picks, team_names, pool_pick_dist)

    suffix = "" if args.prob_base == "torvik" else f"_{args.prob_base}"
    out_path = OUT_DIR / f"bracket_2026_exhaustive{suffix}.json"

    output = {
        "season": YEAR,
        "generated_at": datetime.now().strftime("%Y-%m-%d"),
        "model": f"{MODEL_LABELS[args.prob_base]} — Exhaustive Champion Search",
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
