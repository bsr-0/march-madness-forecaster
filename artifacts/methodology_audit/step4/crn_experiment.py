"""Step 4 item 11: common random numbers must change variance, not the estimand.
20 candidates from the 2026 artifact; P(1st) under (a) one shared trial set of 400 (CRN) and
(b) an independent 400-trial set per candidate, repeated over 12 seeds. Report the mean
difference (should be ~0 within MC error) and the variance of pairwise candidate differences."""
import json, logging, sys, numpy as np; sys.path.insert(0, '.'); logging.disable(logging.WARNING)
import scripts.mc_pool_backtest as M
from scripts.experiments.objective_diversity_matrix import pool_p_first
from scripts.experiments.build_candidate_artifact import _encode_rows
from src.prediction.seed_probabilities import build_seed_probabilities
a=json.load(open('artifacts/candidates/candidates_2026.json')); ids=[t['id'] for t in a['teams']]; fr=[ids[i] for i in a['first_round']]
seeds={t['id']:t['seed'] for t in a['teams']}; seed_pw=build_seed_probabilities(seeds, as_of=2026); pick=M.build_espn_pick_distribution(2026, seeds)
rng=np.random.default_rng(11); idx=rng.choice(len(a['candidates']), 20, replace=False)
rows=np.vstack([_encode_rows([[ids[i] for i in r] for r in a['candidates'][k]['w']], fr) for k in idx])
T=400; R=12; crn=np.zeros((R,20)); ind=np.zeros((R,20))
for r in range(R):
    tr=M.draw_selection_trials(T, n_opponents=29, first_round=fr, pick_dist=pick, matchup_probs=seed_pw, seeds=seeds, rng=np.random.default_rng(1000+r))
    crn[r]=pool_p_first(rows, tr, fr)
    for c in range(20):
        trc=M.draw_selection_trials(T, n_opponents=29, first_round=fr, pick_dist=pick, matchup_probs=seed_pw, seeds=seeds, rng=np.random.default_rng(5000+r*20+c))
        ind[r,c]=pool_p_first(rows[c:c+1], trc, fr)[0]
print(f"mean P(1st): CRN {crn.mean():.4f}  independent {ind.mean():.4f}  diff {crn.mean()-ind.mean():+.4f}  (SE of diff ~{np.sqrt(crn.mean(axis=0).var()/R+ind.mean(axis=0).var()/R)/np.sqrt(20):.4f})")
d_crn=(crn[:,:,None]-crn[:,None,:]); d_ind=(ind[:,:,None]-ind[:,None,:])
print(f"var of pairwise candidate differences across seeds: CRN {d_crn.var(axis=0).mean():.2e}  independent {d_ind.var(axis=0).mean():.2e}  ratio {d_ind.var(axis=0).mean()/d_crn.var(axis=0).mean():.2f}x")
print(f"argmax candidate stable across seeds: CRN {len(set(crn.argmax(axis=1)))} distinct  independent {len(set(ind.argmax(axis=1)))} distinct")
