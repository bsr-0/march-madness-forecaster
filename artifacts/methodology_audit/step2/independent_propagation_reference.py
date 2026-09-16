"""Independent analytic bracket propagation (item 17): exact recursion over opponent mixtures.
P_r(t) = P(t wins its round-r game) = P_{r-1}(t) * sum_{o in opponents_r(t)} P_{r-1}(o) * p(t beats o)
where opponents_r(t) are the teams in the adjacent block of size 2^(r-1). Written from the definition, not from repo code."""
import numpy as np, sys, time
sys.path.insert(0, '/Users/benrosen/Documents/march-madness-forecaster')
from src.prediction.pairwise import PairwiseProbabilities, simulate_bracket_outcomes, marginals_from_pairwise, ROUND_NAMES

def analytic(pw, first_round):
    n = len(first_round); idx = {t:i for i,t in enumerate(first_round)}
    reach = np.ones(n)                      # P(reach round 1 game) = 1
    out = []
    for r in range(6):
        block = 2**(r+1); half = 2**r
        win = np.zeros(n)
        for i,t in enumerate(first_round):
            b0 = (i//block)*block; mine = (i//half)*half
            opp0 = b0 if mine != b0 else b0+half
            s = 0.0
            for j in range(opp0, opp0+half):
                s += reach[j] * pw.p(t, first_round[j])
            win[i] = reach[i]*s
        out.append(win); reach = win
    return {t:{ROUND_NAMES[r]: out[r][i] for r in range(6)} for i,t in enumerate(first_round)}

rng = np.random.default_rng(123)
teams = [f"t{i:02d}" for i in range(64)]
ratings = {t: float(np.clip(rng.beta(2,2), 0.02, 0.98)) for t in teams}
# Make a few extreme teams to hit boundary behaviour
ratings['t00']=0.995; ratings['t63']=0.01; ratings['t01']=0.5; ratings['t02']=0.5
pw = PairwiseProbabilities.from_ratings(ratings, source='synthetic')

A = analytic(pw, teams)
# invariants of the analytic solution
for r in range(6):
    tot = sum(A[t][ROUND_NAMES[r]] for t in teams)
    assert abs(tot - 2**(5-r)) < 1e-9, (r, tot)          # exactly 32,16,8,4,2,1 winners per round
for t in teams:
    seq=[A[t][R] for R in ROUND_NAMES]; assert all(x>=y-1e-12 for x,y in zip(seq,seq[1:])), t   # monotone non-increasing
    assert all(0<=x<=1 for x in seq)
print("analytic: per-round totals exact (32,16,8,4,2,1); monotone; bounded  -- OK")

N=200_000; t0=time.time()
_, rounds = simulate_bracket_outcomes(pw, teams, N, np.random.default_rng(7), noise_std=0.0)
print(f"simulated {N:,} tournaments in {time.time()-t0:.0f}s")
cnt = {t:np.zeros(6) for t in teams}
for sr in rounds:
    for r in range(6):
        for w in sr[r]: cnt[w][r]+=1
worst_z=0; worst=None; nz=0
for t in teams:
    for r in range(6):
        p=A[t][ROUND_NAMES[r]]; phat=cnt[t][r]/N; se=np.sqrt(max(p*(1-p),1e-12)/N)
        z=(phat-p)/se if se>0 else 0
        nz+=1
        if abs(z)>abs(worst_z): worst_z, worst = z, (t, ROUND_NAMES[r], p, phat)
zs=[]
for t in teams:
    for r in range(6):
        p=A[t][ROUND_NAMES[r]]; phat=cnt[t][r]/N; se=np.sqrt(p*(1-p)/N) if 0<p<1 else None
        if se: zs.append((phat-p)/se)
zs=np.array(zs)
print(f"simulate_bracket_outcomes vs analytic: {len(zs)} (team,round) cells; max|z|={abs(zs).max():.2f} at {worst}; frac |z|>2 = {(abs(zs)>2).mean():.3f} (expect ~0.046); frac |z|>3 = {(abs(zs)>3).mean():.4f}")
# marginals_from_pairwise (the round_probs source used by region_top_n construction)
M = marginals_from_pairwise(pw, teams, teams, n_sims=100_000, seed=3, floor=0.0)
zs2=[]
for t in teams:
    for R in ROUND_NAMES:
        p=A[t][R]; 
        if 0<p<1: zs2.append((M[t][R]-p)/np.sqrt(p*(1-p)/100_000))
zs2=np.array(zs2); print(f"marginals_from_pairwise vs analytic: max|z|={abs(zs2).max():.2f}; frac|z|>2={(abs(zs2)>2).mean():.3f}")
# item 7: pairwise unchanged by marginals -- pw is frozen dataclass; check p(t,o) identical before/after
print("pairwise table unchanged after simulation (frozen, no write path):", all(abs(pw.p(a,b)-PairwiseProbabilities.from_ratings(ratings,source='x').p(a,b))<1e-15 for a in teams[:8] for b in teams[:8] if a!=b))
# item 9: conditional opponent mixture -- show that using P(advance) instead of pairwise would be wrong
t='t05'; i=teams.index(t)
mix = sum(A[o]['R64']*pw.p(t,o) for o in teams[(i//4)*4:(i//4)*4+4] if (o==teams[(i//2)*2] or o==teams[(i//2)*2+1])==False)
print(f"conditional R32 for {t}: analytic P(win R32) = {A[t]['R32']:.5f}; = P(win R64) {A[t]['R64']:.5f} x opponent-mixture {mix:.5f} = {A[t]['R64']*mix:.5f}; sim {cnt[t][1]/N:.5f}")

# locate the marginals_from_pairwise worst cell and evaluate with an exact binomial tail instead of a z
from scipy.stats import binom
worst=None
for t in teams:
    for R in ROUND_NAMES:
        p=A[t][R]
        if 0<p<1:
            k=round(M[t][R]*100_000); z=(M[t][R]-p)/np.sqrt(p*(1-p)/100_000)
            if worst is None or abs(z)>abs(worst[0]): worst=(z,t,R,p,k)
z,t,R,p,k=worst
tail = 2*min(binom.cdf(k,100_000,p), binom.sf(k-1,100_000,p))
print(f"worst marginals cell: {t} {R} analytic p={p:.3e}, observed k={k} of 100000, two-sided exact binomial p-value={tail:.3f}  (384 cells -> Bonferroni floor {1/384:.4f})")
