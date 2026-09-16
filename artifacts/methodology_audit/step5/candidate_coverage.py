"""Step 5: is meta_region_poolaware's selection constrained by candidate availability?

For each season: S = the production candidate set (build_production_candidates, ~20-30 brackets).
A = S + the season's artifact sample (~3,000 log5-simulated brackets over three rating sources,
same tree) + 300 brackets sampled game-by-game from the seed pairwise table.
Select argmax P(1st) on 500 SELECTION trials (production pa_trials, seed 77777+year);
evaluate every selection on 1,500 INDEPENDENT trials (seed 424242+year) so maxima are not
winner's-curse inflated. Report P(1st)_eval(S*) vs P(1st)_eval(A*), the best-in-A's provenance,
and the fraction of A that beats S* on evaluation. Also: distinct brackets in S, Hamming spread,
and whether shuffling S changes the pick (tie-break sensitivity).

Run: PYTHONPATH=. python3 artifacts/methodology_audit/step5/candidate_coverage.py [--years ...]
"""
import argparse, json, logging, sys, time
import numpy as np
sys.path.insert(0, '.'); logging.disable(logging.WARNING)
from src.evaluation.referee_audit import build_season_context, build_production_candidates
from scripts.mc_pool_backtest import draw_selection_trials, sample_model_brackets, ESPN_SCORING, N_OPPONENTS
from scripts.experiments.objective_diversity_matrix import pool_p_first
from src.simulation.bracket_topology import winner_sets_to_bool_vector

def artifact_rows(year, first_round):
    try:
        a = json.load(open(f'artifacts/candidates/candidates_{year}.json'))
    except FileNotFoundError:
        return np.zeros((0, 63), dtype=bool), []
    ids = [t['id'] for t in a['teams']]
    if [ids[i] for i in a['first_round']] != list(first_round):
        raise RuntimeError(f"{year}: artifact tree differs from the season context tree")
    rows, src = [], []
    for c in a['candidates']:
        rows.append(winner_sets_to_bool_vector([[ids[i] for i in r] for r in c['w']], first_round)); src.append(c.get('src', '?'))
    return np.stack(rows), src

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--years', type=int, nargs='*'); ap.add_argument('--sel', type=int, default=500); ap.add_argument('--eval', type=int, default=1500)
    args = ap.parse_args()
    years = args.years or [y for y in range(2011, 2027) if y != 2020]
    out = []
    for year in years:
        t0 = time.time()
        ctx = build_season_context(year)
        S = build_production_candidates(ctx)
        S_rows = np.stack([b for b, _ in S]); S_labels = [l for _, l in S]
        A_rows, A_src = artifact_rows(year, ctx.first_round)
        seed_base = ctx.bases['seed']
        R_rows = sample_model_brackets(ctx.first_round, seed_base, 300, np.random.default_rng(31 + year), ctx.seeds, ctx.regions, ctx.pick_dist, ctx.pool_size) if False else None
        # sampler signature varies; draw seed-pairwise brackets directly
        rng = np.random.default_rng(31 + year); R = []
        for _ in range(300):
            cur = list(ctx.first_round); vec = np.zeros(63, dtype=bool); gi = 0
            for r in range(6):
                nxt = []
                for g in range(0, len(cur), 2):
                    a, b = cur[g], cur[g + 1]; p = ctx.seed_pw.get((a, b), 0.5)
                    w = a if rng.random() < p else b; vec[gi] = (w == a); nxt.append(w); gi += 1
                cur = nxt
            R.append(vec)
        R_rows = np.stack(R)
        ALL = np.vstack([S_rows, A_rows, R_rows]); prov = ['production'] * len(S_rows) + [f'artifact:{s}' for s in A_src] + ['seed_sample'] * len(R_rows)
        sel = draw_selection_trials(args.sel, n_opponents=ctx.n_opponents, first_round=ctx.first_round, pick_dist=ctx.pick_dist, matchup_probs=ctx.seed_pw, seeds=ctx.seeds, rng=np.random.default_rng(77777 + year), chalk_noise_std=ctx.chalk_noise_std)
        ev = draw_selection_trials(args.eval, n_opponents=ctx.n_opponents, first_round=ctx.first_round, pick_dist=ctx.pick_dist, matchup_probs=ctx.seed_pw, seeds=ctx.seeds, rng=np.random.default_rng(424242 + year), chalk_noise_std=ctx.chalk_noise_std)
        p_sel = pool_p_first(ALL, sel, ctx.first_round); p_ev = pool_p_first(ALL, ev, ctx.first_round)
        nS = len(S_rows)
        s_star = int(np.argmax(p_sel[:nS])); a_star = int(np.argmax(p_sel))
        se = float(np.sqrt(0.1 * 0.9 / args.eval))
        # tie-break sensitivity: shuffle S and re-select
        perm = np.random.default_rng(7).permutation(nS); s_star_shuf = int(perm[np.argmax(p_sel[:nS][perm])])
        # diversity of S
        ham = [int((S_rows[i] != S_rows[j]).sum()) for i in range(nS) for j in range(i + 1, nS)]
        champs = {tuple(np.where(S_rows[i])[0][-1:]) for i in range(nS)}
        beat = float(np.mean(p_ev[nS:] > p_ev[s_star])) if len(ALL) > nS else float('nan')
        row = dict(year=year, n_S=nS, n_A=len(ALL), S_star=S_labels[s_star], S_star_sel=float(p_sel[s_star]), S_star_eval=float(p_ev[s_star]),
                   A_star=prov[a_star] + ('' if a_star >= nS else f'({S_labels[a_star]})'), A_star_sel=float(p_sel[a_star]), A_star_eval=float(p_ev[a_star]),
                   gain_eval=float(p_ev[a_star] - p_ev[s_star]), eval_se=se, frac_A_beats_S_star_on_eval=beat,
                   S_hamming_mean=float(np.mean(ham)), S_hamming_min=int(min(ham)), S_shuffle_changes_pick=bool(s_star_shuf != s_star),
                   S_top_eval_max=float(p_ev[:nS].max()), A_top_eval_max=float(p_ev.max()))
        out.append(row)
        print(f"{year}: S={nS:2d} A={len(ALL):5d} | S* {row['S_star']:28} sel {row['S_star_sel']:.3f} eval {row['S_star_eval']:.3f} | A* {row['A_star'][:36]:36} sel {row['A_star_sel']:.3f} eval {row['A_star_eval']:.3f} | gain {row['gain_eval']:+.3f} (SE {se:.3f}) | {100*beat:.1f}% of A beat S* | Hamming mean {row['S_hamming_mean']:.1f} min {row['S_hamming_min']} | shuffle changes pick: {row['S_shuffle_changes_pick']}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(out, open('artifacts/methodology_audit/step5/candidate_coverage.json', 'w'), indent=1)
    g = np.array([r['gain_eval'] for r in out]); print(f"\nmean eval gain from the augmented set: {g.mean():+.4f} (per-season SE {out[0]['eval_se']:.3f}); seasons with gain > 2 SE: {(g > 2*out[0]['eval_se']).sum()}/{len(g)}")

if __name__ == '__main__':
    main()
