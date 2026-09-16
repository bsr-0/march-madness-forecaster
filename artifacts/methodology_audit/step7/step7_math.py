"""Step 7 items 5-7, 15, 16: referee table mathematics, topology and coupling, on the real 2026 field."""
import sys, logging, itertools, numpy as np; sys.path.insert(0, '.'); logging.disable(logging.WARNING)
from scipy.stats import norm
from src.evaluation.referee_audit import build_season_context
from src.simulation.bracket_topology import resolve_region_order, build_bracket_order
from scripts._common import load_tournament_results, load_seeds_block
ctx = build_season_context(2026)
teams = ctx.first_round
print("referees:", list(ctx.referees))
ro = resolve_region_order(2026, games=ctx.games, regions=ctx.regions, seeds_block=load_seeds_block(2026))
assert ctx.first_round == build_bracket_order(ctx.seeds, ctx.regions, region_order=ro), "referee context tree != canonical tree"
print("topology: ctx.first_round == canonical real-tree bracket order:", True, ro)
P = {}
for name, tab in ctx.referees.items():
    M = np.array([[tab[(a, b)] if a != b else 0.5 for b in teams] for a in teams])
    comp = np.abs(M + M.T - 1.0); np.fill_diagonal(comp, 0)
    lo, hi = M[~np.eye(64, dtype=bool)].min(), M[~np.eye(64, dtype=bool)].max()
    # equal-strength check: two teams with the same seed AND same rating source value? use symmetry instead: p(a,a) undefined; check p(a,b) vs p(b,a) mirrors
    print(f"  {name:12} max|p(a,b)+p(b,a)-1| = {comp.max():.1e}   range [{lo:.4f}, {hi:.4f}]   >0.99: {(M>0.99).sum()//1}  <0.01: {(M<0.01).sum()//1}")
    P[name] = M
# FTE link: independent check of Phi((rA-rB)/11)
import json
doc = json.load(open('data/kaggle/fivethirtyeight_ratings.json')); cols = doc['columns']
if 'fte' in P:
    from src.data.normalize import normalize_team_id
    r = {normalize_team_id(row[cols.index('team')]): float(row[cols.index('power_rating')]) for row in doc['data'] if row[cols.index('year')] == 2026}
    a, b = teams[0], teams[1]
    print(f"  fte check: {a} {r[a]:.2f} vs {b} {r[b]:.2f}: table {P['fte'][0,1]:.4f} vs Phi((rA-rB)/11) {norm.cdf((r[a]-r[b])/11):.4f}; equal ratings -> {norm.cdf(0):.2f}")
# coupling: pairwise correlation of logit tables over the 2016 ordered pairs
iu = np.triu_indices(64, 1)
names = list(P); L = {n: np.log(np.clip(P[n][iu], 1e-6, 1-1e-6) / (1 - np.clip(P[n][iu], 1e-6, 1-1e-6))) for n in names}
print("\nlogit-scale correlation between referee tables (2016 unordered pairs, 2026):")
print("             " + " ".join(f"{n[:8]:>8}" for n in names))
for a in names:
    print(f"  {a[:10]:10} " + " ".join(f"{np.corrcoef(L[a], L[b])[0,1]:8.3f}" for b in names))
# Path 3 production probabilities (blend base) vs each referee
