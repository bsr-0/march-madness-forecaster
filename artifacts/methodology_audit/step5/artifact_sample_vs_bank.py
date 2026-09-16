"""Step 5, Path 2: does the artifact's 3,000-bracket stratified sample lose the P(1st)-optimal
region of the ~150k bank it is drawn from? Rebuild the 2026 bank (30k sims for speed), draw the
production stratified sample and an independent one, plus a uniform 3,000 draw; score all on one
shared 1,000-trial set and compare the top-P(1st) and the P(1st) distribution tails."""
import json, sys, logging, numpy as np; sys.path.insert(0, '.'); logging.disable(logging.WARNING)
from scripts.experiments import build_candidate_artifact as B
from scripts.experiments.conditional_bracket_engine import round_marginals, expected_scores
from scripts.experiments.objective_diversity_matrix import pool_p_first
from src.prediction.pairwise import PairwiseProbabilities, simulate_bracket_outcomes
from src.prediction.seed_probabilities import build_seed_probabilities
from scripts.mc_pool_backtest import draw_selection_trials, build_espn_pick_distribution, resolve_first_four, ESPN_SCORING
from scripts._common import load_seeds_and_regions, load_tournament_results, load_seeds_block
from src.simulation import bracket_topology as bt
year = 2026
seeds, regions = load_seeds_and_regions(year); games = load_tournament_results(year); resolve_first_four(games, seeds, regions)
ro = bt.resolve_region_order(year, games=games, regions=regions, seeds_block=load_seeds_block(year)); fr = bt.build_bracket_order(seeds, regions, region_order=ro)
barthag = B._load_torvik_barthag(year, seeds)
pw = PairwiseProbabilities.from_ratings(barthag, source='torvik')
N = 30_000
bank, rounds = simulate_bracket_outcomes(pw, fr, N, np.random.default_rng(20260820), noise_std=0.0)
marg = round_marginals(rounds); ev = expected_scores(rounds, marg, ESPN_SCORING)
sel_a = B.stratified_sample(rounds, ev, 3000, np.random.default_rng(1))
sel_b = B.stratified_sample(rounds, ev, 3000, np.random.default_rng(2))
sel_u = np.random.default_rng(3).choice(N, 3000, replace=False)
seed_pw = build_seed_probabilities(seeds, as_of=year); pick = build_espn_pick_distribution(year, seeds)
trials = draw_selection_trials(1000, n_opponents=29, first_round=fr, pick_dist=pick, matchup_probs=seed_pw, seeds=seeds, rng=np.random.default_rng(99))
idx = np.unique(np.concatenate([sel_a, sel_b, sel_u, np.random.default_rng(4).choice(N, 6000, replace=False)]))
p1 = pool_p_first(bank[idx], trials, fr); lookup = dict(zip(idx.tolist(), p1))
def stats(name, sel):
    v = np.array([lookup[i] for i in sel]); return f"{name:22} n={len(sel):4d} max={v.max():.4f} p99={np.quantile(v,.99):.4f} p90={np.quantile(v,.9):.4f} mean={v.mean():.4f}"
print(stats('stratified (prod seed)', sel_a)); print(stats('stratified (seed 2)', sel_b)); print(stats('uniform 3000', sel_u)); print(stats('uniform 6000+', idx))
top = idx[np.argsort(-p1)[:20]]
print("top-20 of the scored bank present in the production stratified sample:", int(np.isin(top, sel_a).sum()), "/ 20;  in an independent stratified sample:", int(np.isin(top, sel_b).sum()), "/ 20")
print("bank distinct champions:", len({r[5][0] for r in rounds}), "| in stratified sample:", len({rounds[i][5][0] for i in sel_a}))
