"""Step 6: opponent and ownership machinery, tested on 2026 data and synthetic pools."""
import json, sys, logging, numpy as np
sys.path.insert(0, '.'); logging.disable(logging.WARNING)
import scripts.mc_pool_backtest as M
from src.simulation.pool_competition import generate_opponent_brackets, simulate_tournament_outcomes, score_brackets_team_identity, picks_by_round, ROUND_NAMES
from src.simulation.bracket_topology import resolve_region_order, build_bracket_order, picks_to_bool_vector
from src.prediction.seed_probabilities import build_seed_probabilities
from src.optimization.payout import first_place_share
from src.optimization.bracket_construction import construct_bracket
from scripts._common import load_tournament_results, load_seeds_block
from scripts.experiments.objective_diversity_matrix import pool_p_first
PTS = M.ESPN_SCORING
year = 2026
seeds, regions = M.load_seeds_and_regions(year); games = load_tournament_results(year); M.resolve_first_four(games, seeds, regions)
ro = resolve_region_order(year, games=games, regions=regions, seeds_block=load_seeds_block(year)); fr = build_bracket_order(seeds, regions, region_order=ro)
seed_pw = build_seed_probabilities(seeds, as_of=year); pick = M.build_espn_pick_distribution(year, seeds)
art = json.load(open('artifacts/candidates/candidates_2026.json')); ids = [t['id'] for t in art['teams']]
from scripts.experiments.build_candidate_artifact import _encode_rows
cand = _encode_rows([[ids[i] for i in r] for r in art['named_strategies']['blend_region_35']['w']], fr)[0]

def walk(vec):
    cur=list(fr); gi=0; out=[]
    for r in range(6):
        nxt=[cur[g] if vec[gi+g//2] else cur[g+1] for g in range(0,len(cur),2)]; gi+=len(cur)//2; out.append(nxt); cur=nxt
    return out

print("== 8. legality of generated opponents ==")
opp = generate_opponent_brackets(20000, fr, seed_pw, pick, seeds, np.random.default_rng(1))
assert opp.shape == (20000, 63)
# legality is structural in the 63-bool walk; verify decode gives 32/16/8/4/2/1 distinct winners drawn from the right slots
bad = 0
for i in range(2000):
    w = walk(opp[i]); ok = all(len(set(w[r])) == 32 >> r for r in range(6)) and all(set(w[r+1]) <= set(w[r]) for r in range(5))
    bad += not ok
print(f"   2000 decoded opponents: illegal = {bad}; each has 32/16/8/4/2/1 winners, every later winner won the earlier round")

print("== 9. does the ratio heuristic reproduce the input marginal pick shares? ==")
counts = {t: np.zeros(6) for t in fr}
for i in range(20000):
    w = walk(opp[i])
    for r in range(6):
        for t in w[r]: counts[t][r] += 1
rows=[]
for r, R in enumerate(ROUND_NAMES):
    inp = np.array([pick.get(t, {}).get(R, 0.0) for t in fr]); got = np.array([counts[t][r]/20000 for t in fr])
    rows.append((R, float(np.abs(got-inp).max()), float(np.corrcoef(inp, got)[0,1]), float(inp.sum()), float(got.sum())))
    print(f"   {R:5} max|realised - ESPN share| = {rows[-1][1]:.3f}  corr {rows[-1][2]:.4f}  (ESPN sum {rows[-1][3]:.2f}, realised sum {rows[-1][4]:.2f})")
worst = max(((t, R, pick.get(t,{}).get(R,0), counts[t][r]/20000) for t in fr for r,R in enumerate(ROUND_NAMES)), key=lambda x: abs(x[2]-x[3]))
print(f"   worst cell: {worst[0]} {worst[1]} ESPN {worst[2]:.3f} realised {worst[3]:.3f}")

print("== 3. pool-size behaviour of P(1st) for the shipped bracket ==")
for n in (1, 5, 10, 20, 29, 50, 100):
    tr = M.draw_selection_trials(1000, n_opponents=n, first_round=fr, pick_dist=pick, matchup_probs=seed_pw, seeds=seeds, rng=np.random.default_rng(5))
    p = pool_p_first(cand.reshape(1,63), tr, fr)[0]
    print(f"   n_opponents={n:3d}: P(1st) share = {p:.4f}   (1/(n+1) = {1/(n+1):.4f})")

print("== 5. shared vs independent tournament realisation ==")
tr = M.draw_selection_trials(2000, n_opponents=29, first_round=fr, pick_dist=pick, matchup_probs=seed_pw, seeds=seeds, rng=np.random.default_rng(7))
shared = pool_p_first(cand.reshape(1,63), tr, fr)[0]
# independent: candidate scored on tournament t, opponents on tournament t+1 (a different draw)
indep = 0.0
for i in range(len(tr)):
    opp_i, sw_i = tr[i]; _, sw_j = tr[(i+1) % len(tr)]
    c = score_brackets_team_identity(cand.reshape(1,63), sw_j, fr, PTS)[0]; o = score_brackets_team_identity(opp_i, sw_i, fr, PTS)
    indep += first_place_share(c, o)
indep /= len(tr)
print(f"   shared realisation P(1st) = {shared:.4f};  independent realisations = {indep:.4f}  (these are DIFFERENT estimands; a pool shares one tournament)")

print("== 10. synthetic pools with a known answer ==")
def chalk_vec():
    cur=list(fr); v=np.zeros(63,dtype=bool); gi=0
    for r in range(6):
        nxt=[]
        for g in range(0,len(cur),2):
            a,b=cur[g],cur[g+1]; w=a if seeds[a]<=seeds[b] else b; v[gi]=(w==a); nxt.append(w); gi+=1
        cur=nxt
    return v
chalk = chalk_vec()
sims = [simulate_tournament_outcomes(1, fr, seed_pw, seeds, 0.0, np.random.default_rng(100+i))[1][0] for i in range(1500)]
sw = [{R: set(s[r]) for r, R in enumerate(ROUND_NAMES)} for s in sims]
def indep_share(cvec, opp_mat):
    tot=0.0
    for w in sw:
        c=score_brackets_team_identity(cvec.reshape(1,63), w, fr, PTS)[0]; o=score_brackets_team_identity(opp_mat, w, fr, PTS)
        top=o.max(); k=int((o==top).sum()); tot += 1.0 if c>top else (1.0/(1+k) if c==top else 0.0)
    return tot/len(sw)
pools = {}
pools['A all chalk (29)'] = np.repeat(chalk.reshape(1,63), 29, axis=0)
pools['B identical = candidate (29)'] = np.repeat(cand.reshape(1,63), 29, axis=0)
rng=np.random.default_rng(3); pools['C uniform random legal (29)'] = rng.random((29,63)) < 0.5
mixed = np.vstack([np.repeat(chalk.reshape(1,63), 10, axis=0), np.repeat(cand.reshape(1,63), 5, axis=0), rng.random((14,63))<0.5]); pools['D mixed 10 chalk/5 cand/14 random'] = mixed
for name, opp_m in pools.items():
    trials=[(opp_m, w) for w in sw]
    prod = pool_p_first(cand.reshape(1,63), trials, fr)[0]; ref = indep_share(cand, opp_m)
    print(f"   {name:36} production {prod:.4f}  independent {ref:.4f}  diff {prod-ref:+.1e}")
print(f"   pool B sanity: candidate ties every opponent in every tournament -> share must be exactly 1/30 = {1/30:.4f}")
# generator-level: an all-chalk pick distribution must yield identical chalk opponents
allchalk = {t: {R: (1.0 if True else 0.0) for R in ROUND_NAMES} for t in fr}
# make favourites 1.0 and underdogs 0.0 per game slot is not expressible per round for later rounds without a bracket; use extreme shares by seed instead
byseed = {t: {R: 1.0/seeds[t] for R in ROUND_NAMES} for t in fr}
o = generate_opponent_brackets(200, fr, seed_pw, {t:{R:(1.0 if seeds[t]<=8 else 0.0) for R in ROUND_NAMES} for t in fr}, seeds, np.random.default_rng(2))
print(f"   generator with shares 1/0 by favourite: distinct opponents = {len({o[i].tobytes() for i in range(200)})} (R64 games between two favourites/underdogs fall to 0.5)")

print("== 12. ownership pathways ==")
pk_off = {}
p_on, ch_on, *_ = construct_bracket(mode='region_top_n', seeds=seeds, regions=regions, round_probs=art and None or None, public_picks=pick, risk_level=0.35, pool_size=30, scoring_system=dict(PTS), region_order=ro) if False else (None, None)
from src.evaluation.referee_audit import build_season_context
ctx = build_season_context(year)
b_on = picks_to_bool_vector(construct_bracket(mode='region_top_n', seeds=seeds, regions=regions, round_probs=ctx.blend_rp, public_picks=pick, risk_level=0.35, pool_size=30, scoring_system=dict(PTS), region_order=ro)[0], fr)
b_off = picks_to_bool_vector(construct_bracket(mode='region_top_n', seeds=seeds, regions=regions, round_probs=ctx.blend_rp, public_picks={}, risk_level=0.35, pool_size=30, scoring_system=dict(PTS), region_order=ro)[0], fr)
print(f"   construction pathway: bracket differs with ownership on vs off in {int((b_on!=b_off).sum())} of 63 games")
tr_on = M.draw_selection_trials(1500, n_opponents=29, first_round=fr, pick_dist=pick, matchup_probs=seed_pw, seeds=seeds, rng=np.random.default_rng(11))
tr_off = M.draw_selection_trials(1500, n_opponents=29, first_round=fr, pick_dist={}, matchup_probs=seed_pw, seeds=seeds, rng=np.random.default_rng(11))
for lab, v in (('ownership-built bracket', b_on), ('ownership-free bracket', b_off)):
    print(f"   opponent pathway, {lab:24}: P(1st) vs ESPN-share opponents {pool_p_first(v.reshape(1,63), tr_on, fr)[0]:.4f} | vs seed-model opponents (pick_dist={{}}) {pool_p_first(v.reshape(1,63), tr_off, fr)[0]:.4f}")
