"""Build the meta_region_poolaware bracket for a live year -- a reference tool,
not part of the live site.

Replicates the meta_region_poolaware candidate-generation + pool-simulation
selection block from mc_pool_backtest.py (search "meta_region_poolaware") for
a single live year instead of the 15-year LOYO backtest loop: build ~15-25
diverse candidate brackets (forced 1-seed champions, risk sweeps, prob-base
sweeps, exhaustive-champion sweeps), score each by simulating against a
realistic opponent pool, and keep the highest binary-P(1st) candidate.

WHAT THIS IS FOR. This is the strategy `mc_pool_backtest.py` measures at
~12% P(1st) (README "What the backtest number means"). Nothing else builds
that exact strategy for a live year, so this script is the only way to see
the concrete bracket the headline number describes, or to spot-check it
against a season's real outcome.

WHAT THIS IS NOT. Its output, docs/data/bracket_2026.json, is not read by
docs/app.js and never has been since the current UI shipped. The live site
is built by scripts/experiments/build_candidate_artifact.py and
scripts/build_ui_payload.py, which serve a different, separately-validated
fixed rule (`blend_region_35`, ~10-11% P(1st) -- see the comment on
`_blend_region_bracket` in build_candidate_artifact.py) via
docs/data/season_*.json. The two numbers are not the same strategy and
should not be quoted for each other. This is audit finding C3; the README's
"What the backtest number means" section carries the same caveat for users.

Kept, with `src/optimization/poolaware_recipe.py` and
`tests/test_poolaware_recipe.py`, as the guard against the recipe drifting
from what the backtest actually measures -- that drift is exactly what C3
first found. `generate_region_bracket.py` and `generate_exhaustive_bracket.py`
served the same "reference build" role for the other two backtest strategies
but had no such guard and no other purpose; both were retired 2026-09-11
along with the unreachable generate-web-data.yml / deploy-pages.yml
(orphaned when run-pipeline.yml, their only caller, was removed with the ML
pipeline -- see audit H10).
"""

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
    load_tournament_results,
    resolve_first_four,
)
from src.optimization.bracket_construction import construct_bracket
from src.optimization.poolaware_recipe import (
    POOLAWARE_EXHAUSTIVE_RISKS,
    POOLAWARE_RISK_LEVELS,
    build_poolaware_prob_bases,
    poolaware_base_names,
)
from src.prediction.massey_best_probabilities import build_massey_best_round_probabilities
from src.prediction.massey_probabilities import load_massey_avg_barthag
from src.prediction.noseed_model import (
    _load_team_stats,
    build_blend_round_probabilities,
    build_noseed_round_probabilities,
    train_noseed_model,
)
from src.prediction.seed_probabilities import (
    build_seed_probabilities,
    build_seed_round_probabilities,
)
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
# Risk sweeps come from the shared recipe, not a local copy — see
# src/optimization/poolaware_recipe.py for why.
RISK_LEVELS = POOLAWARE_RISK_LEVELS


def _build_blend_round_probs(seeds):
    """The `blend` base: alpha*seed + (1-alpha)*noseed round probabilities.

    Mirrors the backtest's construction, including walk-forward training
    (`max_year=YEAR`, so the model never sees the year it is picking for) and
    the canonical `blend_alpha=0.5`. Returns None if the no-seed model or its
    stats payload is unavailable, which drops `blend` from the sweep the same
    way a missing Massey file drops `mass_avg`.
    """
    try:
        model = train_noseed_model(max_year=YEAR)
        assert all(y < YEAR for y in model.train_years), (
            f"walk-forward violation: noseed model for {YEAR} trained on {model.train_years}"
        )
        stats = _load_team_stats(YEAR)
        noseed_rp = build_noseed_round_probabilities(model, seeds, stats)
        seed_rp = build_seed_round_probabilities(seeds)
        return build_blend_round_probabilities(seed_rp, noseed_rp, alpha=0.5)
    except Exception as exc:
        print(f"  blend base unavailable ({type(exc).__name__}: {exc}) — dropped from sweep")
        return None


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
    seeds, regions = load_seeds_and_regions(YEAR)
    if not seeds:
        print(f"ERROR: no seeds found for {YEAR}")
        sys.exit(1)

    # Resolve the play-in games, exactly as the backtest does before building
    # any bracket (mc_pool_backtest._run_one_year). Without this the seeds file
    # still holds both teams for each play-in slot and build_bracket_order
    # refuses to guess the draw — this script has been unable to run at all
    # since that guard landed. Play-in games finish before brackets lock, so
    # their winners are ordinary pre-tournament information.
    games = load_tournament_results(YEAR)
    n_resolved = resolve_first_four(games, seeds, regions) if games else 0
    if n_resolved:
        print(f"  Resolved {n_resolved} play-in slot(s) -> {len(seeds)}-team field")

    barthag = _load_torvik_barthag(YEAR, seeds)
    torvik_rp = build_torvik_round_probabilities(seeds, regions, barthag)

    # Sweep the SAME probability bases the backtest sweeps. This script used
    # to build its own two-base list (tv, mass_avg) while the backtest swept
    # five, so the strategy measured at ~11% P(1st) and the strategy that
    # shipped were different strategies: over the 15-season reference run the
    # backtest picked a base this script could not build in 11 of 15 seasons,
    # including tv_mass80 for 2026. The recipe now lives in one place.
    massey_barthag = load_massey_avg_barthag(YEAR, seeds, PROJECT_ROOT / "data")
    massey_avg_rp = (
        build_torvik_round_probabilities(seeds, regions, massey_barthag)
        if massey_barthag is not None
        else None
    )
    massey_best_rp = build_massey_best_round_probabilities(
        seeds, regions, test_year=YEAR, data_root=PROJECT_ROOT / "data"
    )
    blend_rp = _build_blend_round_probs(seeds)

    prob_bases = build_poolaware_prob_bases(
        torvik_rp,
        massey_avg=massey_avg_rp,
        massey_best=massey_best_rp,
        blend=blend_rp,
    )
    # Ratings for display, keyed by base name. Bases built in marginal space
    # (blend, tv_mass80) have no rating vector of their own; they fall back to
    # torvik barthag, which build_bracket_json only uses where round_probs
    # coverage is degenerate.
    ratings_by_base = {
        "tv": barthag,
        "mass_avg": massey_barthag if massey_barthag is not None else barthag,
        "mass_best": barthag,
        "blend": barthag,
        "tv_mass80": barthag,
    }
    print(f"  Probability bases: {', '.join(poolaware_base_names(prob_bases))}")

    pick_dist, n_opponents, opponent_source = resolve_opponents(seeds)
    seed_pw = build_seed_probabilities(seeds)
    first_round = build_first_round_matchups(seeds, regions)
    scoring = dict(ESPN_SCORING)
    rng = np.random.default_rng(77777 + YEAR)

    one_seed_teams = [tid for tid, s in seeds.items() if s == 1]
    # (picks, bool_vector, label, rating_dict, display_round_probs)
    candidates: list[tuple[dict, np.ndarray, str, dict, dict]] = []

    def try_add(label: str, rating: dict, _display_rp=None, **kwargs) -> None:
        """Build one candidate. `_display_rp` is the base's own round probs,
        carried so the shipped JSON shows the probabilities of the base that
        actually picked the bracket rather than always showing torvik's."""
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
            candidates.append((picks, bvec, label, rating, _display_rp or torvik_rp))
        except Exception as exc:
            print(f"  candidate '{label}' skipped: {exc}")

    # (a) Forced 1-seed champions x region_top_n (torvik, risk=0.5)
    for forced in one_seed_teams:
        try_add(
            f"tv_champ={forced}",
            barthag,
            mode="region_top_n",
            round_probs=torvik_rp,
            risk_level=0.5,
            forced_champion=forced,
        )

    # (b) Risk sweeps x prob bases x region_top_n (no forced champ)
    for risk in RISK_LEVELS:
        for pb_name, pb_rp in prob_bases:
            try_add(
                f"{pb_name}_region_risk={risk}",
                ratings_by_base.get(pb_name, barthag),
                mode="region_top_n",
                round_probs=pb_rp,
                risk_level=risk,
                _display_rp=pb_rp,
            )

    # (c) Exhaustive champion search x prob bases x select risks
    for risk in POOLAWARE_EXHAUSTIVE_RISKS:
        for pb_name, pb_rp in prob_bases:
            try_add(
                f"{pb_name}_exhaust_risk={risk}",
                ratings_by_base.get(pb_name, barthag),
                mode="exhaustive_champion",
                round_probs=pb_rp,
                risk_level=risk,
                _display_rp=pb_rp,
            )

    # De-duplicate identical brackets (keeps first label)
    seen: set[bytes] = set()
    unique = []
    for picks, bvec, label, rating, display_rp in candidates:
        key = bvec.tobytes()
        if key not in seen:
            seen.add(key)
            unique.append((picks, bvec, label, rating, display_rp))
    candidates = unique

    if not candidates:
        print("  No candidates built — falling back to plain region_top_n bracket")
        best_picks, _champ, _f4, _ev, _var = construct_bracket(
            mode="region_top_n",
            seeds=seeds,
            regions=regions,
            round_probs=torvik_rp,
            public_picks=pick_dist,
            risk_level=0.5,
            pool_size=n_opponents + 1,
            scoring_system=scoring,
        )
        best_label = "fallback"
        best_rating = barthag
        best_display_rp = torvik_rp
    else:
        # Score each candidate via pool simulation (binary P(1st) estimator) —
        # mirrors mc_pool_backtest.py's meta_region_poolaware selection exactly.
        best_p1 = -1.0
        best_idx = 0
        for ci, (_picks, bvec, _label, _rating, _display_rp) in enumerate(candidates):
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
        best_picks, _bvec, best_label, best_rating, best_display_rp = candidates[best_idx]
        print(
            f"  Selected {best_label} (best of {len(candidates)} candidates, "
            f"simulated P(1st)={best_p1:.3f})"
        )

    team_names = load_team_names()
    # Display the winning candidate's OWN probabilities. build_bracket_json
    # derives each game's win_prob from round_probs (barthag is only a
    # degenerate-coverage fallback), so passing torvik_rp unconditionally --
    # as this script used to -- annotated a bracket picked by, say, blend or
    # tv_mass80 with torvik's numbers. That was harmless while only torvik and
    # mass_avg were swept and torvik usually won; it is not harmless now that
    # the full backtest sweep is available and tv_mass80 wins 2026. pick_dist
    # is the real opponent field (pool history preferred over ESPN), so it
    # doubles as the pool-consensus annotation for display.
    rounds = build_bracket_json(
        seeds, regions, best_rating, best_display_rp, best_picks, team_names, pick_dist
    )

    out_path = OUT_DIR / "bracket_2026.json"

    output = {
        "season": YEAR,
        "generated_at": datetime.now().strftime("%Y-%m-%d"),
        "model": "Torvik Barthag — Region Top-N x Multi-Candidate Pool-Aware Selection",
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
