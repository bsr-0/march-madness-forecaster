"""Pre-fix evidence for Step 3 findings F3-1 / F3-2: bracket-topology mismatch.

For every backtest season: the real Final Four pairing vs the hardcoded default;
Torvik log5 analytic marginals on both trees; and, for the Torvik and blend
bases, whether the bracket the backtest SCORED (old `_picks_dict_to_bool_array`
projection onto the real tree) carries the F4 winners / champion that
construction actually chose.

Run: PYTHONPATH=. python3 artifacts/methodology_audit/step3/topology_impact.py [--post-fix]
"""
import json, sys, logging, argparse
import numpy as np
sys.path.insert(0, '.')
logging.disable(logging.WARNING)
import scripts.mc_pool_backtest as M
from scripts._common import load_tournament_results
from src.optimization.bracket_construction import construct_bracket
from src.prediction.pairwise import PairwiseProbabilities, ROUND_NAMES
from src.prediction.noseed_model import train_noseed_model, build_noseed_round_probabilities, build_blend_round_probabilities
from src.prediction.seed_probabilities import build_seed_round_probabilities

def analytic(pw, first_round):
    n = len(first_round); reach = np.ones(n); out = []
    for r in range(6):
        block, half = 2**(r+1), 2**r; win = np.zeros(n)
        for i, t in enumerate(first_round):
            b0 = (i//block)*block; mine = (i//half)*half; opp0 = b0 if mine != b0 else b0+half
            win[i] = reach[i]*sum(reach[j]*pw.p(t, first_round[j]) for j in range(opp0, opp0+half))
        out.append(win); reach = win
    return {t: {ROUND_NAMES[r]: out[r][i] for r in range(6)} for i, t in enumerate(first_round)}

def decode(vec, first_round):
    cur = list(first_round); gi = 0; rounds = []
    for r in range(6):
        nxt = []
        for g in range(0, len(cur), 2):
            t = cur[g] if vec[gi] else cur[g+1]; nxt.append(t); gi += 1
        rounds.append(nxt); cur = nxt
    return rounds

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--post-fix', action='store_true'); args = ap.parse_args()
    rows = []
    for year in [y for y in range(2011, 2027) if y != 2020]:
        seeds, regions = M.load_seeds_and_regions(year); games = load_tournament_results(year)
        M.resolve_first_four(games, seeds, regions)
        ro = tuple(M.derive_f4_region_pairing(games, regions))
        default = tuple(M.REGION_ORDER)
        same = {frozenset(ro[:2]), frozenset(ro[2:])} == {frozenset(default[:2]), frozenset(default[2:])}
        real_fr = M.build_first_round_matchups(seeds, regions, region_order=ro)
        def_fr = M.build_first_round_matchups(seeds, regions, region_order=default)
        barthag = M._load_torvik_barthag(year, seeds)
        pw = PairwiseProbabilities.from_ratings(barthag, source='torvik')
        A_real, A_def = analytic(pw, real_fr), analytic(pw, def_fr)
        dF4 = max(abs(A_real[t]['F4']-A_def[t]['F4']) for t in real_fr); dCH = max(abs(A_real[t]['CHAMP']-A_def[t]['CHAMP']) for t in real_fr)
        row = dict(year=year, real_order=ro, matches_default=same, max_dF4=round(dF4, 4), max_dCHAMP=round(dCH, 4), bases={})
        kw = dict(region_order=ro) if args.post_fix else {}
        # bases as the backtest builds them
        try:
            torvik_base = M.build_base_from_ratings('torvik', seeds, regions, barthag, **kw)
        except TypeError:
            torvik_base = M.build_base_from_ratings('torvik', seeds, regions, barthag)
        stats = M._load_team_stats(year); model = train_noseed_model(max_year=year)
        seed_rp = build_seed_round_probabilities(seeds, as_of=year)
        try:
            noseed_rp = build_noseed_round_probabilities(model, seeds, stats, as_of=year)
        except TypeError:
            noseed_rp = build_noseed_round_probabilities(model, seeds, stats)
        blend_rp = build_blend_round_probabilities(seed_rp, noseed_rp, alpha=0.5)
        for name, rp in (('torvik', torvik_base.rp if hasattr(torvik_base, 'rp') else torvik_base.round_probs), ('blend', blend_rp)):
            ckw = dict(region_order=ro) if args.post_fix else {}
            picks, champ, f4, ev, var = construct_bracket(seeds=seeds, regions=regions, round_probs=rp, public_picks={}, pool_size=30,
                                                           scoring_system=dict(M.ESPN_SCORING), mode='region_top_n', risk_level=0.35, **ckw)
            f4_picks = {v for k, v in picks.items() if k.startswith('F4')}
            vec = M._picks_dict_to_bool_array(picks, real_fr)
            dec = decode(vec, real_fr)
            row['bases'][name] = dict(construction_f4=sorted(f4_picks), construction_champ=champ,
                                      scored_f4=sorted(dec[4]), scored_champ=dec[5][0],
                                      mismatch=(dec[5][0] != champ or set(dec[4]) != f4_picks))
        rows.append(row)
        b = row['bases']
        print(f"{year} {ro} {'same' if same else 'DIFF'} dF4={dF4:.3f} dCH={dCH:.3f} | torvik {'MISMATCH' if b['torvik']['mismatch'] else 'ok'} ({b['torvik']['construction_champ']}->{b['torvik']['scored_champ']}) | blend {'MISMATCH' if b['blend']['mismatch'] else 'ok'} ({b['blend']['construction_champ']}->{b['blend']['scored_champ']})", flush=True)
    tag = 'post_fix' if args.post_fix else 'pre_fix'
    json.dump(rows, open(f'artifacts/methodology_audit/step3/topology_impact_{tag}.json', 'w'), indent=1, default=str)
    n_diff = sum(not r['matches_default'] for r in rows); n_mm = {k: sum(r['bases'][k]['mismatch'] for r in rows) for k in ('torvik', 'blend')}
    print(f"\nseasons with real pairing != default: {n_diff}/{len(rows)}; construction-vs-scored mismatches: {n_mm}")

if __name__ == '__main__':
    main()
