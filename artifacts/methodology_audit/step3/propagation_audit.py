"""Step 3 propagation audit: pairwise -> simulated games -> advancement marginals, on REAL 2026 tables.

Independent reference: the positional analytic recursion (copied from Step 2, not imported from repo
marginalizers). Pre-registered PASS for a simulator/marginalizer: min Bonferroni-adjusted exact-binomial
p over 384 (team, round) cells > 0.01 and fraction of |z|>2 within [0.02, 0.08].

Run: PYTHONPATH=. python3 artifacts/methodology_audit/step3/propagation_audit.py [--post-fix] [--n 200000]
"""
import sys, json, time, argparse, logging, subprocess
import numpy as np
from scipy.stats import binomtest
sys.path.insert(0, '.')
logging.disable(logging.WARNING)
import scripts.mc_pool_backtest as M
from scripts._common import load_tournament_results
from src.prediction.pairwise import PairwiseProbabilities, marginals_from_pairwise, ROUND_NAMES
from src.simulation.pool_competition import simulate_tournament_outcomes
from src.prediction.seed_probabilities import build_seed_probabilities, build_seed_round_probabilities
from src.prediction.pit_production_model import pairwise_for_year
from src.prediction.noseed_model import train_noseed_model, build_noseed_probabilities, build_blend_probabilities, build_noseed_round_probabilities
from src.data.seed_pick_model import _compute_advancement_rates

def analytic(pw, first_round):
    n = len(first_round); reach = np.ones(n); out = []
    for r in range(6):
        block, half = 2**(r+1), 2**r; win = np.zeros(n)
        for i, t in enumerate(first_round):
            b0 = (i//block)*block; mine = (i//half)*half; opp0 = b0 if mine != b0 else b0+half
            win[i] = reach[i]*sum(reach[j]*pw.p(t, first_round[j]) for j in range(opp0, opp0+half))
        out.append(win); reach = win
    return {t: {ROUND_NAMES[r]: out[r][i] for r in range(6)} for i, t in enumerate(first_round)}

def compare(name, A, counts, N, teams, skip_floor=None):
    zs, worst = [], []
    for t in teams:
        for r, R in enumerate(ROUND_NAMES):
            p = A[t][R]; k = counts[t][r]
            if skip_floor is not None and abs(k/N - skip_floor) < 1e-12:
                continue   # cell reported AT the floor: the marginalizer's floor, not its propagation, is being measured
            if 0 < p < 1:
                z = (k/N - p)/np.sqrt(p*(1-p)/N); zs.append(z)
                worst.append((abs(z), t, R, p, k))
    zs = np.array(zs); worst.sort(reverse=True)
    top = []
    for az, t, R, p, k in worst[:10]:
        pv = binomtest(int(k), N, p).pvalue
        top.append(dict(team=t, round=R, p=p, k=int(k), z=float(np.sign(k/N-p)*az), exact_p=pv, bonf_p=min(1.0, pv*len(zs))))
    frac2 = float((abs(zs) > 2).mean()); minbonf = min(x['bonf_p'] for x in top)
    verdict = 'PASS' if (minbonf > 0.01 and 0.02 <= frac2 <= 0.08) else 'FAIL'
    print(f"  {name:38} cells={len(zs)} max|z|={abs(zs).max():.2f} frac|z|>2={frac2:.3f} min Bonferroni p={minbonf:.3f} -> {verdict}")
    return dict(name=name, cells=len(zs), max_abs_z=float(abs(zs).max()), frac_z_gt_2=frac2, min_bonf_p=minbonf, verdict=verdict, worst=top[:3])

def counts_from_rounds(rounds, teams):
    c = {t: np.zeros(6) for t in teams}
    for sr in rounds:
        for r in range(6):
            for w in sr[r]:
                if w in c: c[w][r] += 1
    return c

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--post-fix', action='store_true'); ap.add_argument('--n', type=int, default=200_000); args = ap.parse_args()
    N = args.n; year = 2026
    seeds, regions = M.load_seeds_and_regions(year); games = load_tournament_results(year)
    M.resolve_first_four(games, seeds, regions)
    ro = tuple(M.derive_f4_region_pairing(games, regions)); default = tuple(M.REGION_ORDER)
    real_fr = M.build_first_round_matchups(seeds, regions, region_order=ro)
    def_fr = M.build_first_round_matchups(seeds, regions, region_order=default)
    teams = list(real_fr)
    barthag = M._load_torvik_barthag(year, seeds); stats = M._load_team_stats(year); model = train_noseed_model(max_year=year)
    seed_pw = build_seed_probabilities(seeds, as_of=year)
    noseed_pw = build_noseed_probabilities(model, seeds, stats)
    tables = {
        'torvik': PairwiseProbabilities.from_ratings(barthag, source='torvik'),
        'seed': PairwiseProbabilities.from_dict(seed_pw, 'seed'),
        'pit': PairwiseProbabilities.from_dict(pairwise_for_year(year, teams), 'pit'),
        'blend': PairwiseProbabilities.from_dict(build_blend_probabilities(seed_pw, noseed_pw, 0.5), 'blend'),
    }
    report = dict(commit=subprocess.run(['git','rev-parse','--short','HEAD'],capture_output=True,text=True).stdout.strip(),
                  post_fix=args.post_fix, N=N, real_order=ro, default_order=default, results=[])
    print(f"2026 real F4 order {ro}; default {default}; N={N:,}; post_fix={args.post_fix}")
    for name, pw in tables.items():
        A = analytic(pw, real_fr)
        for r, slots in enumerate((32, 16, 8, 4, 2, 1)):
            assert abs(sum(A[t][ROUND_NAMES[r]] for t in teams) - slots) < 1e-9
        for t in teams:
            s = [A[t][R] for R in ROUND_NAMES]; assert all(x >= y - 1e-12 for x, y in zip(s, s[1:])) and all(0 <= x <= 1 for x in s)
        print(f"[{name}] analytic invariants ok (sums 32/16/8/4/2/1, monotone, bounded)")
        t0 = time.time()
        # The referee simulator caps every game at [0.01, 0.99] even with noise 0 (pool_competition.py:481). Propagation
        # is judged against the table it actually walks; the cap itself is recorded separately (F3-5).
        clipped = PairwiseProbabilities.from_dict({k: min(0.99, max(0.01, v)) for k, v in pw.as_dict().items()}, name+'_clipped')
        n_capped = sum(1 for v in pw.as_dict().values() if v > 0.99 or v < 0.01)
        A_clip = analytic(clipped, real_fr)
        _, rounds = simulate_tournament_outcomes(N, real_fr, pw.as_dict(), seeds, 0.0, np.random.default_rng(7))
        cnt = counts_from_rounds(rounds, teams)
        report['results'].append(compare(f'{name}: simulate_tournament_outcomes n=0 (vs clipped table)', A_clip, cnt, N, teams))
        if n_capped:
            report['results'].append(compare(f'{name}: same, vs UNclipped table', A, cnt, N, teams))
            print(f"  {name}: referee cap [0.01,0.99] binds on {n_capped} ordered pairs; max|p - clip| = {max(abs(v-min(0.99,max(0.01,v))) for v in pw.as_dict().values()):.4f}")
        Mg = marginals_from_pairwise(pw, real_fr, teams, n_sims=10_000, seed=42, floor=0.0)
        c = {t: np.array([Mg[t][R]*10_000 for R in ROUND_NAMES]) for t in teams}
        report['results'].append(compare(f'{name}: marginals_from_pairwise 10k', A, c, 10_000, teams))
        # noise 0.16, descriptive
        _, rn = simulate_tournament_outcomes(N//4, real_fr, pw.as_dict(), seeds, 0.16, np.random.default_rng(8))
        cn = counts_from_rounds(rn, teams)
        l1 = [sum(abs(cn[t][r]/(N//4) - A[t][ROUND_NAMES[r]]) for t in teams) for r in range(6)]
        print(f"  {name}: noise 0.16 per-round L1 shift vs analytic: {[round(x,3) for x in l1]}  ({time.time()-t0:.0f}s)")
        report['results'].append(dict(name=f'{name}: noise 0.16 L1 by round', l1=l1))
    # build_torvik_round_probabilities: which tree does it marginalise?
    A_real, A_def = analytic(tables['torvik'], real_fr), analytic(tables['torvik'], def_fr)
    try:
        rp = M.build_torvik_round_probabilities(seeds, regions, barthag, n_sims=100_000, region_order=ro); which = 'region_order=real'
    except TypeError:
        rp = M.build_torvik_round_probabilities(seeds, regions, barthag, n_sims=100_000); which = 'no region_order arg (pre-fix)'
    c = {t: np.array([rp[t][R]*100_000 for R in ROUND_NAMES]) for t in teams}
    print(f"build_torvik_round_probabilities ({which}):")
    report['results'].append(compare('  vs analytic on REAL tree (floored cells skipped)', A_real, c, 100_000, teams, skip_floor=0.001))
    report['results'].append(compare('  vs analytic on DEFAULT tree (floored cells skipped)', A_def, c, 100_000, teams, skip_floor=0.001))
    # seed-level recursion vs positional recursion with the seed table
    car = _compute_advancement_rates('recent', as_of=year); A_seed = analytic(tables['seed'], real_fr)
    gap = max(abs(car[seeds[t]][R] - A_seed[t][R]) for t in teams for R in ROUND_NAMES)
    print(f"_compute_advancement_rates('recent', as_of=2026) vs positional recursion on real tree: max abs gap = {gap:.2e}")
    report['seed_recursion_gap'] = gap
    # noseed heuristic vs propagated noseed pairwise (R-1 evidence, descriptive)
    try:
        ns_rp = build_noseed_round_probabilities(model, seeds, stats, as_of=year)
    except TypeError:
        ns_rp = build_noseed_round_probabilities(model, seeds, stats)
    A_ns = analytic(PairwiseProbabilities.from_dict(noseed_pw, 'noseed'), real_fr)
    d = [max(abs(ns_rp[t][R] - A_ns[t][R]) for t in teams) for R in ROUND_NAMES]
    print(f"noseed heuristic marginals vs propagated noseed pairwise: max abs diff by round {[round(x,3) for x in d]}; CHAMP mass {sum(ns_rp[t]['CHAMP'] for t in teams):.3f}")
    report['noseed_heuristic_vs_propagated_max_diff_by_round'] = d
    tag = 'post_fix' if args.post_fix else 'pre_fix'
    json.dump(report, open(f'artifacts/methodology_audit/step3/propagation_audit_{tag}.json', 'w'), indent=1, default=float)

if __name__ == '__main__':
    main()
