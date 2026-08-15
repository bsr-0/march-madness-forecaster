"""Generate the meta_region_poolaware bracket for 2026 and export to docs/data/.

Replicates the meta_region_poolaware candidate-generation + pool-simulation
selection block from mc_pool_backtest.py (search "meta_region_poolaware") for
a single live year instead of the 14-year LOYO backtest loop: build ~15-25
diverse candidate brackets (forced 1-seed champions, risk sweeps, prob-base
sweeps, exhaustive-champion sweeps), score each by simulating against a
realistic opponent pool, and keep the highest binary-P(1st) candidate.

--prob-base torvik (default) is THE production strategy (11.9% P(1st),
14-yr LOYO) and is what docs/data/bracket_2026.json is supposed to contain.
--prob-base elo/ap swap the primary probability base (the "tv" slot) for
a fully independent rating system (see scripts/prob_base_variants.py),
keeping every other candidate-generation detail (forced champs, risk
sweeps, alt bases, pool-aware selection) identical — this is an
exploratory lens on the SAME construction, not a claim it wins on P(1st).

Torvik output previously held stale output from a discontinued Elo+seed
blend pipeline (generate_web_data.py) — see docs/app.js STRATEGIES['pool']
for the UI-facing claim this file backs.

Owns docs/data/bracket_2026*.json. generate_web_data.py no longer writes
these files — do not reintroduce that.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._bracket_export_common import build_bracket_json, load_team_names
from scripts.mc_pool_backtest import (
    ESPN_SCORING,
    N_OPPONENTS,
    POOL_HIST_PATH,
    _load_torvik_barthag,
    _picks_dict_to_bool_array,
    build_espn_pick_distribution,
    build_first_round_matchups,
    build_torvik_round_probabilities,
    load_seeds_and_regions,
)
from scripts.prob_base_variants import load_prob_base, MODEL_LABELS
from src.optimization.bracket_construction import construct_bracket
from src.prediction.massey_probabilities import load_massey_avg_barthag
from src.prediction.seed_probabilities import build_seed_probabilities
from src.simulation.pool_competition import (
    ROUND_NAMES,
    generate_opponent_brackets,
    score_brackets_team_identity,
    simulate_tournament_outcomes,
)
from src.simulation.pool_history_opponent_model import (
    build_pool_pick_distribution,
    load_pool_brackets,
)

YEAR = 2026
OUT_DIR = PROJECT_ROOT / "docs" / "data"
PA_TRIALS = 500
RISK_LEVELS = (0.1, 0.3, 0.5, 0.7, 0.9)


def resolve_opponents(seeds):
    """Real pool history (N≈30) if available, else ESPN picks, else seed model."""
    try:
        pool_brackets, group_size = load_pool_brackets(POOL_HIST_PATH, YEAR)
        pick_dist = build_pool_pick_distribution(pool_brackets, seeds)
        source = f"your pool history (N={group_size})"
        print(f"  Opponent source: pool history ({group_size - 1} opponents)")
        return pick_dist, group_size - 1, source
    except (FileNotFoundError, KeyError):
        pass
    try:
        pick_dist = build_espn_pick_distribution(YEAR, seeds)
        print(f"  Opponent source: ESPN picks ({N_OPPONENTS} opponents)")
        return pick_dist, N_OPPONENTS, "ESPN public picks"
    except FileNotFoundError:
        print(f"  Opponent source: none available — using empty pick distribution ({N_OPPONENTS} opponents)")
        return {}, N_OPPONENTS, None


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
    primary_rating, primary_rp = load_prob_base(args.prob_base, YEAR, seeds, regions, torvik_rp, barthag)

    # (name, round_probs, rating_dict) — rating_dict is carried alongside each
    # prob base so the winning candidate's *own* ratings drive its displayed
    # win_prob (via Log5), instead of every candidate showing generic Torvik
    # barthag regardless of which base actually picked it. Only the PRIMARY
    # slot changes with --prob-base; alt bases (massey_avg) stay torvik-
    # derived diversity candidates regardless, same as the production mode.
    prob_bases = [(args.prob_base, primary_rp, primary_rating)]
    massey_barthag = load_massey_avg_barthag(YEAR, seeds, PROJECT_ROOT / "data")
    if massey_barthag is not None:
        prob_bases.append(
            ("mass_avg", build_torvik_round_probabilities(seeds, regions, massey_barthag), massey_barthag)
        )

    pick_dist, n_opponents, opponent_source = resolve_opponents(seeds)
    seed_pw = build_seed_probabilities(seeds)
    first_round = build_first_round_matchups(seeds, regions)
    scoring = dict(ESPN_SCORING)
    rng = np.random.default_rng(77777 + YEAR)

    one_seed_teams = [tid for tid, s in seeds.items() if s == 1]
    candidates: list[tuple[dict, np.ndarray, str, dict]] = []  # (picks, bool_vector, label, rating_dict)

    def try_add(label: str, rating: dict, **kwargs) -> None:
        try:
            picks, _champ, _f4, _ev, _var = construct_bracket(
                seeds=seeds,
                regions=regions,
                public_picks=pick_dist,
                pool_size=n_opponents + 1,
                scoring_system=scoring,
                **kwargs,
            )
            bvec = _picks_dict_to_bool_array(picks, first_round)
            candidates.append((picks, bvec, label, rating))
        except Exception as exc:
            print(f"  candidate '{label}' skipped: {exc}")

    # (a) Forced 1-seed champions x region_top_n (primary prob base, risk=0.5)
    for forced in one_seed_teams:
        try_add(
            f"{args.prob_base}_champ={forced}",
            primary_rating,
            mode="region_top_n",
            round_probs=primary_rp,
            risk_level=0.5,
            forced_champion=forced,
        )

    # (b) Risk sweeps x prob bases x region_top_n (no forced champ)
    for risk in RISK_LEVELS:
        for pb_name, pb_rp, pb_rating in prob_bases:
            try_add(
                f"{pb_name}_region_risk={risk}",
                pb_rating,
                mode="region_top_n",
                round_probs=pb_rp,
                risk_level=risk,
            )

    # (c) Exhaustive champion search x prob bases x select risks
    for risk in (0.3, 0.5, 0.7):
        for pb_name, pb_rp, pb_rating in prob_bases:
            try_add(
                f"{pb_name}_exhaust_risk={risk}",
                pb_rating,
                mode="exhaustive_champion",
                round_probs=pb_rp,
                risk_level=risk,
            )

    # De-duplicate identical brackets (keeps first label)
    seen: set[bytes] = set()
    unique = []
    for picks, bvec, label, rating in candidates:
        key = bvec.tobytes()
        if key not in seen:
            seen.add(key)
            unique.append((picks, bvec, label, rating))
    candidates = unique

    if not candidates:
        print("  No candidates built — falling back to plain region_top_n bracket")
        best_picks, _champ, _f4, _ev, _var = construct_bracket(
            mode="region_top_n",
            seeds=seeds,
            regions=regions,
            round_probs=primary_rp,
            public_picks=pick_dist,
            risk_level=0.5,
            pool_size=n_opponents + 1,
            scoring_system=scoring,
        )
        best_label = "fallback"
        best_rating = primary_rating
    else:
        # Score each candidate via pool simulation (binary P(1st) estimator) —
        # mirrors mc_pool_backtest.py's meta_region_poolaware selection exactly.
        best_p1 = -1.0
        best_idx = 0
        for ci, (_picks, bvec, _label, _rating) in enumerate(candidates):
            wins = 0
            for _ in range(PA_TRIALS):
                opp = generate_opponent_brackets(
                    n_opponents=n_opponents,
                    first_round_matchups=first_round,
                    pick_distribution=pick_dist,
                    matchup_probs=seed_pw,
                    seeds=seeds,
                    rng=rng,
                    chalk_noise_std=0.0,
                )
                _out, br = simulate_tournament_outcomes(
                    n_tournaments=1,
                    first_round_matchups=first_round,
                    matchup_probs=seed_pw,
                    seeds=seeds,
                    noise_std=0.16,
                    rng=rng,
                )
                sim_winners = {rnd: set(br[0][ri]) for ri, rnd in enumerate(ROUND_NAMES)}
                c_score = score_brackets_team_identity(
                    bvec.reshape(1, 63), sim_winners, first_round, scoring
                )[0]
                opp_scores = score_brackets_team_identity(opp, sim_winners, first_round, scoring)
                if c_score >= opp_scores.max():
                    wins += 1
            p1 = wins / PA_TRIALS
            if p1 > best_p1:
                best_p1 = p1
                best_idx = ci
        best_picks, _bvec, best_label, best_rating = candidates[best_idx]
        print(
            f"  Selected {best_label} (best of {len(candidates)} candidates, "
            f"simulated P(1st)={best_p1:.3f})"
        )

    team_names = load_team_names()
    # Use the winning candidate's own ratings for displayed win_prob, not
    # always generic Torvik barthag — so the percentage actually reflects
    # whichever prob base drove this specific bracket's picks. pick_dist here
    # is already the real opponent field (pool history preferred over ESPN),
    # so it doubles as the pool-consensus annotation for display.
    rounds = build_bracket_json(seeds, regions, best_rating, primary_rp, best_picks, team_names, pick_dist)

    suffix = "" if args.prob_base == "torvik" else f"_{args.prob_base}"
    out_path = OUT_DIR / f"bracket_2026{suffix}.json"

    output = {
        "season": YEAR,
        "generated_at": datetime.now().strftime("%Y-%m-%d"),
        "model": f"{MODEL_LABELS[args.prob_base]} — Region Top-N x Multi-Candidate Pool-Aware Selection",
        "n_simulations": PA_TRIALS,
        "opponent_source": opponent_source,
        "rounds": rounds,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    champion = best_picks.get("CHAMP")
    print(f"Champion: {team_names.get(champion, champion)}")
    print()
    for rnd in rounds:
        print(f"  {rnd['round_name']}: {len(rnd['games'])} games")
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
