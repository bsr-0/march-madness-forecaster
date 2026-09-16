"""R-4: training weights inside the tournament-trained PIT model. Executes PREREGISTRATION_R4.md
(sha256 0d73668e...) exactly: same gates as Steps 10-12, weights only, frozen choice on the tuning window
before the recent regime is scored. M0 is reproduced through the same weighted code with unit weights
and checked against the production walk_forward() so the reference is exact."""
import json, sys, logging, hashlib, subprocess, datetime
import numpy as np
from scipy.stats import t as student_t, norm
from scipy.optimize import minimize
sys.path.insert(0, '.'); logging.disable(logging.WARNING)
from src.prediction import pit_production_model as PM

OUT = 'artifacts/methodology_audit/r4/'
TUNING = [2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022]; RECENT = [2023, 2024, 2025]; EXTRA = [2026]
B = 5000; SEED = 42; VARIANTS = ['M2(0.85)', 'M2(0.70)', 'M3', 'MU(2)', 'MU(4)']
print("== R-4 gate definitions (frozen; identical to Steps 10-12) ==")
print("dLL = LL(variant) - LL(M0) per season; gate1 tuning mean<=0 & 2.5th pct<=0; gate2 recent mean<0 & 97.5th pct<0; gate3 |subset| tolerance +0.01 ABSOLUTE LL; gate4 slope in [0.8,1.2]")
print("upset subset = games won by the lower dated-barthag team (pre-tournament sign); MU weight on training rows where the lower-barthag team won\n")

keys, X, m, y = PM.load_training(); cols = PM.resolve_cols(keys, PM.CANONICAL_KEYS); Xc = X[:, cols]
bi = keys.index('barthag'); fav_lost = (X[:, bi] > 0) != (m > 0); zero_gap = X[:, bi] == 0
w_out = (m > 0).astype(float)

def fit_w(Xt, mt, wt):
    n, k = Xt.shape; A = (Xt * wt[:, None]).T @ Xt; A[np.diag_indices(k)] += PM.LAMBDA * (wt.sum() / 1000.0)
    beta = np.linalg.solve(A, (Xt * wt[:, None]).T @ mt)
    sigma = max(float(np.sqrt(np.average((mt - Xt @ beta) ** 2, weights=wt))), 1e-6)
    return beta, sigma

def weights(v, ys, fl, Y):
    if v == 'M0': return np.ones(len(ys))
    if v.startswith('M2'): return float(v[3:-1]) ** (Y - 1 - ys)
    if v == 'M3': return (ys >= Y - 3).astype(float)
    return np.where(fl, float(v[3:-1]), 1.0)

def run(v):
    """Walk-forward predictions per season with the production's causal link (calibrate on prior OOS rows, a shrunk n/(n+63))."""
    P = {}; oos_y, oos_m, oos_p, oos_s = [], [], [], []
    for Y in sorted(set(y)):
        tr = y < Y; te = y == Y
        wt = weights(v, y[tr], fav_lost[tr], Y)
        if tr.sum() < len(cols) * PM.MIN_ROWS_PER_COL or wt.sum() < len(cols) * PM.MIN_ROWS_PER_COL or wt[wt > 0].size < len(cols) * PM.MIN_ROWS_PER_COL: continue
        beta, sigma = fit_w(Xc[tr], m[tr], wt); pred = Xc[te] @ beta
        prior = np.array(oos_y) < Y
        if prior.sum() >= 1:
            c = PM.calibrate(np.array(oos_m)[prior], np.array(oos_p)[prior], np.array(oos_s)[prior]); n = int(prior.sum())
            a, nu = (n * c['a'] + PM.CAL_PRIOR_STRENGTH) / (n + PM.CAL_PRIOR_STRENGTH), c['nu']
        else: a, nu = 1.0, np.inf
        P[Y] = PM.clip_prob(PM.student_t_cdf(a * pred / sigma, nu))
        if Y >= PM.MIN_TEST_YEAR:   # production's walk_forward only records OOS rows from MIN_TEST_YEAR
            oos_y += [Y] * te.sum(); oos_m += list(m[te]); oos_p += list(pred); oos_s += [sigma] * te.sum()
    return P

def ll(p, w): return -(w * np.log(p) + (1 - w) * np.log(1 - p))
def season_ll(P, years, mask=None):
    out = {}
    for Y in years:
        sel = (y == Y) if mask is None else ((y == Y) & mask); idx = np.where(sel)[0]; pos = np.searchsorted(np.where(y == Y)[0], idx)
        out[Y] = float(ll(P[Y][pos], w_out[idx]).mean()) if len(idx) else float('nan')
    return out
def boot(d):
    d = np.array(d); rng = np.random.default_rng(SEED); means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
def slope(P, years):
    lp = np.concatenate([np.log(P[Y] / (1 - P[Y])) for Y in years]); w = np.concatenate([w_out[y == Y] for Y in years])
    nll = lambda th: -(w * (th[0] + th[1] * lp) - np.log1p(np.exp(th[0] + th[1] * lp))).sum()
    return float(minimize(nll, [0.0, 1.0], method='BFGS').x[1])

P0 = run('M0')
# exactness of the reference against the production walk-forward code
py, pm, pp, ps = PM.walk_forward(X, m, y, cols); cal = PM.walk_forward_calibration(py, pm, pp, ps)
ref = {Y: PM.clip_prob(PM.student_t_cdf(cal[Y]['a'] * pp[py == Y] / ps[py == Y], cal[Y]['nu'])) for Y in TUNING + RECENT}
print("M0 reproduces production walk_forward probabilities: max abs diff =", max(float(np.abs(P0[Y] - ref[Y]).max()) for Y in TUNING + RECENT))
upset = fav_lost & ~zero_gap; chalk = ~fav_lost & ~zero_gap
print(f"subsets over all rows: upset {upset.sum()}, chalk {chalk.sum()}, zero-gap {zero_gap.sum()}\n")

ll0 = season_ll(P0, TUNING); print(f"== tuning 2014-2022: M0 mean LL {np.mean(list(ll0.values())):.4f}")
res = {}
for v in VARIANTS:
    P = run(v); l = season_ll(P, TUNING); mean, lo, hi = boot([l[Y] - ll0[Y] for Y in TUNING]); g1 = mean <= 0 and lo <= 0
    res[v] = dict(P=P, tuning=dict(mean_ll=float(np.mean(list(l.values()))), dLL=mean, ci=[lo, hi], gate1=g1))
    print(f"  {v:9} LL {res[v]['tuning']['mean_ll']:.4f}  dLL {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]  gate1 {'PASS' if g1 else 'FAIL'}")
passers = [v for v in VARIANTS if res[v]['tuning']['gate1']]
chosen = min(passers, key=lambda v: (res[v]['tuning']['dLL'], VARIANTS.index(v))) if passers else None
json.dump({'chosen': chosen, 'tuning': {v: res[v]['tuning'] for v in VARIANTS}, 'frozen_at': datetime.datetime.utcnow().isoformat()}, open(OUT + 'frozen_choice.json', 'w'), indent=1)
print(f"FROZEN choice (written before recent regime): {chosen}\n")

ll0r = season_ll(P0, RECENT); u0 = season_ll(P0, RECENT, upset); c0 = season_ll(P0, RECENT, chalk)
print(f"== recent 2023-2025: M0 LL {np.mean(list(ll0r.values())):.4f}  upset {np.nanmean(list(u0.values())):.4f}  chalk {np.nanmean(list(c0.values())):.4f}  slope {slope(P0, RECENT):.3f}")
rep = {}
for v in VARIANTS:
    P = res[v]['P']; l = season_ll(P, RECENT); u = season_ll(P, RECENT, upset); c = season_ll(P, RECENT, chalk)
    mean, lo, hi = boot([l[Y] - ll0r[Y] for Y in RECENT]); du = float(np.nanmean([u[Y] - u0[Y] for Y in RECENT])); dc = float(np.nanmean([c[Y] - c0[Y] for Y in RECENT])); sl = slope(P, RECENT)
    g2 = mean < 0 and hi < 0; g3 = du <= 0.01 and dc <= 0.01; g4 = 0.8 <= sl <= 1.2
    rep[v] = dict(mean_ll=float(np.mean(list(l.values()))), dLL=mean, ci=[lo, hi], dLL_upset=du, dLL_chalk=dc, slope=sl, gate2=g2, gate3=g3, gate4=g4, per_season={Y: l[Y] - ll0r[Y] for Y in RECENT})
    print(f"  {v:9} LL {rep[v]['mean_ll']:.4f}  dLL {mean:+.4f} [{lo:+.4f}, {hi:+.4f}] g2 {'P' if g2 else 'F'} | upset {du:+.4f} chalk {dc:+.4f} g3 {'P' if g3 else 'F'} | slope {sl:.3f} g4 {'P' if g4 else 'F'}{'  <- FROZEN' if v == chosen else ''}")
print("\n== 2026 (integration season, transparency only, no gate) ==")
l0x = season_ll(P0, EXTRA)[2026]; print(f"  M0 LL {l0x:.4f}; " + "  ".join(f"{v} {season_ll(res[v]['P'], EXTRA)[2026] - l0x:+.4f}" for v in VARIANTS))
verdict = 'NO VARIANT PASSED GATE 1 -> negative' if chosen is None else ('ADOPT' if all(rep[chosen][g] for g in ('gate2', 'gate3', 'gate4')) else ('INDETERMINATE (gate 2 CI spans zero)' if rep[chosen]['gate3'] and rep[chosen]['gate4'] and rep[chosen]['dLL'] < 0 and not rep[chosen]['gate2'] else 'REJECT'))
print(f"\nVERDICT for frozen choice {chosen}: {verdict}")
json.dump({'prereg_sha256_16': hashlib.sha256(open(OUT + 'PREREGISTRATION_R4.md', 'rb').read()).hexdigest()[:16], 'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(), 'chosen': chosen, 'recent': rep, 'M0_recent_ll': float(np.mean(list(ll0r.values()))), 'verdict': verdict}, open(OUT + 'results.json', 'w'), indent=1, default=float)
