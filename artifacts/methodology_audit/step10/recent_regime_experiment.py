"""Steps 10-12: the pre-registered recent-regime / upset-sensitive experiment.

Executes PREREGISTRATION_RECENT_REGIME.md (sha256 fc7bc93b...) exactly, in the registered order:
prints the gate definitions, scores the tuning window, FREEZES the variant choice to disk, then
scores the recent regime and applies gates 2-4. Nothing is tuned on the recent regime.
"""
import csv, json, sys, logging, hashlib, subprocess, datetime
import numpy as np
from scipy.stats import t as student_t, norm
from scipy.optimize import minimize
sys.path.insert(0, '.'); logging.disable(logging.WARNING)
from src.prediction.pit_production_model import pairwise_for_year

OUT = 'artifacts/methodology_audit/step10/'
TUNING = [2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022]
RECENT = [2023, 2024, 2025]
LAMBDA = 1.0; CAL_PRIOR = 63; B = 5000; SEED = 42
VARIANTS = ['M1', 'M2(0.85)', 'M2(0.70)', 'M3', 'MU(2)', 'MU(4)']

print("== gate definitions (frozen; see PREREGISTRATION_RECENT_REGIME.md) ==")
print("dLL = LL(variant) - LL(M0) per season (mean per-game LL); negative = variant better")
print("gate1 tuning 2014-2022: mean dLL <= 0 and 2.5th pct of paired season bootstrap <= 0")
print("gate2 recent 2023-2025: mean dLL < 0 and 97.5th pct < 0 (n=3 seasons, 5000 resamples, seed 42)")
print("gate3 subsets: mean dLL_upset <= +0.01 and dLL_chalk <= +0.01, ABSOLUTE log-loss units; upset/chalk by Selection Sunday seed; First Four excluded from subsets")
print("gate4 calibration slope b in w ~ a + b*logit(p), MLE on pooled 2023-2025 games incl. First Four, in [0.8, 1.2]")
print("MU weight: training rows where the LOWER dated-barthag team won get weight w (favourite by snapshot, not result)")
print("selection: lowest tuning mean dLL among gate-1 passers, ties -> simpler; frozen to disk before recent regime is scored\n")

tr = json.load(open('docs/data/training_pit.json')); ev = json.load(open('docs/data/eval_pit_tournament.json'))
keys = tr['keys']; assert keys == ev['keys']
use = [i for i, k in enumerate(keys) if not k.startswith('venue')]
bi = keys.index('barthag')
X = np.array([g['x'] for g in tr['games']])[:, use]; m = np.array([g['m'] for g in tr['games']], float); y = np.array([g['y'] for g in tr['games']])
fav_lost = (np.array([g['x'][bi] for g in tr['games']]) > 0) != (m > 0)   # team1 higher barthag XOR team1 won -> lower-barthag team won
Xe = np.array([g['x'] for g in ev['games']])[:, use]; me = np.array([g['m'] for g in ev['games']], float); ye = np.array([g['y'] for g in ev['games']]); we = (me > 0).astype(float)
gde = np.array([g['gd'] for g in ev['games']])
seeds = {}
for r in csv.DictReader(open('data/kaggle/MNCAATourneySeeds.csv')): seeds[(int(r['Season']), r['TeamID'])] = int(r['Seed'][1:3])
names = {r['TeamName']: r['TeamID'] for r in csv.DictReader(open('data/kaggle/MTeams.csv'))}
from src.data.features.custom_ratings import KAGGLE_TEAMNAME_ALIASES
from src.data.normalize import normalize_team_id
canon2kid = {}
for nm, kid in names.items(): canon2kid[KAGGLE_TEAMNAME_ALIASES.get(nm) or normalize_team_id(nm)] = kid
def seed_of(yr, t): return seeds.get((yr, canon2kid.get(t)))
s1 = np.array([seed_of(g['y'], g['t1']) or -1 for g in ev['games']]); s2 = np.array([seed_of(g['y'], g['t2']) or -1 for g in ev['games']])
print(f"eval rows {len(Xe)}; seed lookup failures {(s1 < 0).sum() + (s2 < 0).sum()}; First Four {(gde < 136).sum()}")
upset = (s1 != s2) & (gde >= 136) & (((s1 > s2) & (we == 1)) | ((s2 > s1) & (we == 0)))
chalk = (s1 != s2) & (gde >= 136) & ~upset
print(f"subset sizes: upset {upset.sum()}, chalk {chalk.sum()}, same-seed/FF excluded {(~upset & ~chalk).sum()}")

def fit_ridge(Xtr, mtr, wts):
    n, k = Xtr.shape; A = np.column_stack([np.ones(n), Xtr]); W = wts[:, None]
    G = (A * W).T @ A; G[np.arange(1, k + 1), np.arange(1, k + 1)] += LAMBDA * (wts.sum() / 1000.0)
    beta = np.linalg.solve(G, (A * W).T @ mtr)
    return lambda Q: np.column_stack([np.ones(len(Q)), Q]) @ beta

def weights(variant, ytr, fav_lost_tr, Y):
    w = np.ones(len(ytr))
    if variant.startswith('M2'): rho = float(variant[3:-1]); w = rho ** (Y - 1 - ytr)
    if variant == 'M3': w = (ytr >= Y - 3).astype(float)
    if variant.startswith('MU'): w = np.where(fav_lost_tr, float(variant[3:-1]), 1.0)
    return w

def link_calibrate(prior_m, prior_pred, prior_sigma):
    """Student-t link (a, nu) fitted on prior out-of-sample rows by log loss, a shrunk toward 1 with n/(n+63)."""
    if len(prior_m) < 100: return 1.0, np.inf
    yv = (prior_m > 0).astype(float); best = (9e9, 1.0, np.inf)
    for nu in (2, 3, 4, 6, 8, 12, 20, 40, np.inf):
        for a in np.arange(0.2, 3.0, 0.02):
            z = a * prior_pred / prior_sigma; p = np.clip(norm.cdf(z) if np.isinf(nu) else student_t.cdf(z, nu), 1e-6, 1 - 1e-6)
            ll = -(yv * np.log(p) + (1 - yv) * np.log(1 - p)).mean()
            if ll < best[0]: best = (ll, a, nu)
    n = len(prior_m); a = (n * best[1] + CAL_PRIOR) / (n + CAL_PRIOR)
    return a, best[2]

def probs_for(variant, years):
    """Walk-forward: for each Y, fit on training seasons < Y; link calibrated on out-of-sample predictions of seasons < Y."""
    out = {}
    oos = []   # (season, true margin, predicted margin, sigma) from earlier fits, for the link
    for Y in sorted(set(list(range(2011, 2026)) ) - {2020}):
        trm = y < Y; tem = ye == Y
        if trm.sum() < 500: continue
        w = weights(variant, y[trm], fav_lost[trm], Y)
        if w.sum() < 500: continue
        f = fit_ridge(X[trm], m[trm], w)
        sigma = float(np.sqrt(np.average((m[trm] - f(X[trm])) ** 2, weights=w)))
        pred = f(Xe[tem])
        prior = [o for o in oos if o[0] < Y]
        a, nu = link_calibrate(np.array([o[1] for o in prior]), np.array([o[2] for o in prior]), np.array([o[3] for o in prior])) if prior else (1.0, np.inf)
        z = a * pred / sigma; p = norm.cdf(z) if np.isinf(nu) else student_t.cdf(z, nu)
        if Y in years: out[Y] = np.clip(p, 1e-6, 1 - 1e-6)
        oos += [(Y, me[tem][i], pred[i], sigma) for i in range(tem.sum())]
    return out

ID_FIX = {'massachusetts': 'umass'}   # one id differs between the PIT matrix and the production stats table (2014)
def m0_probs(years):
    out = {}
    for Y in years:
        tem = ye == Y; rows = np.array(ev['games'], dtype=object)[tem]
        ids = sorted({ID_FIX.get(g['t1'], g['t1']) for g in rows} | {ID_FIX.get(g['t2'], g['t2']) for g in rows})
        pw = pairwise_for_year(Y, ids)
        out[Y] = np.clip(np.array([pw[(ID_FIX.get(g['t1'], g['t1']), ID_FIX.get(g['t2'], g['t2']))] for g in rows]), 1e-6, 1 - 1e-6)
    return out

def ll(p, w): return -(w * np.log(p) + (1 - w) * np.log(1 - p))
def season_ll(P, years, mask=None):
    res = {}
    for Y in years:
        sel = ye == Y
        if mask is not None: sel &= mask
        idx = np.where(sel)[0]
        pos = np.searchsorted(np.where(ye == Y)[0], idx)
        res[Y] = float(ll(P[Y][pos], we[idx]).mean()) if len(idx) else float('nan')
    return res
def boot(deltas):
    d = np.array(deltas); rng = np.random.default_rng(SEED)
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
def cal_slope(P, years):
    lp = np.concatenate([np.log(P[Y] / (1 - P[Y])) for Y in years]); w = np.concatenate([we[ye == Y] for Y in years])
    nll = lambda th: -(w * (th[0] + th[1] * lp) - np.log1p(np.exp(th[0] + th[1] * lp))).sum()
    return float(minimize(nll, [0.0, 1.0], method='BFGS').x[1])

# ---- tuning window ----
P0 = m0_probs(TUNING + RECENT)
ll0 = season_ll(P0, TUNING)
print("== tuning window 2014-2022 (walk-forward) ==")
print(f"M0 (production pit) mean LL over 8 seasons: {np.mean(list(ll0.values())):.4f}")
tuning = {}
for v in VARIANTS:
    P = probs_for(v, TUNING + RECENT); llv = season_ll(P, TUNING)
    mean, lo, hi = boot([llv[Y] - ll0[Y] for Y in TUNING])
    gate1 = mean <= 0 and lo <= 0
    tuning[v] = dict(mean_ll=float(np.mean(list(llv.values()))), dLL=mean, ci=[lo, hi], gate1=gate1, P=P)
    print(f"  {v:9} LL {tuning[v]['mean_ll']:.4f}  dLL {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]  gate1 {'PASS' if gate1 else 'FAIL'}")
passers = [v for v in VARIANTS if tuning[v]['gate1']]
chosen = min(passers, key=lambda v: (tuning[v]['dLL'], VARIANTS.index(v))) if passers else None
frozen = {'chosen_variant': chosen, 'tuning': {v: {k: x for k, x in tuning[v].items() if k != 'P'} for v in VARIANTS}, 'frozen_at': datetime.datetime.utcnow().isoformat()}
json.dump(frozen, open(OUT + 'frozen_choice.json', 'w'), indent=1)
print(f"FROZEN variant choice (written before any recent-regime number): {chosen}\n")

# ---- recent regime ----
print("== recent regime 2023-2025 ==")
ll0r = season_ll(P0, RECENT); up0 = season_ll(P0, RECENT, upset); ch0 = season_ll(P0, RECENT, chalk)
print(f"M0: LL {np.mean(list(ll0r.values())):.4f}  upset-subset LL {np.nanmean(list(up0.values())):.4f}  chalk-subset LL {np.nanmean(list(ch0.values())):.4f}  slope {cal_slope(P0, RECENT):.3f}")
report = {}
for v in VARIANTS:
    P = tuning[v]['P']; llv = season_ll(P, RECENT); upv = season_ll(P, RECENT, upset); chv = season_ll(P, RECENT, chalk)
    mean, lo, hi = boot([llv[Y] - ll0r[Y] for Y in RECENT])
    du = float(np.nanmean([upv[Y] - up0[Y] for Y in RECENT])); dc = float(np.nanmean([chv[Y] - ch0[Y] for Y in RECENT])); sl = cal_slope(P, RECENT)
    g2 = mean < 0 and hi < 0; g3 = du <= 0.01 and dc <= 0.01; g4 = 0.8 <= sl <= 1.2
    report[v] = dict(mean_ll=float(np.mean(list(llv.values()))), dLL=mean, ci=[lo, hi], dLL_upset=du, dLL_chalk=dc, slope=sl, gate2=g2, gate3=g3, gate4=g4, per_season_dLL={Y: llv[Y] - ll0r[Y] for Y in RECENT})
    tag = ' <- FROZEN CHOICE' if v == chosen else ''
    print(f"  {v:9} LL {report[v]['mean_ll']:.4f}  dLL {mean:+.4f} [{lo:+.4f}, {hi:+.4f}] g2 {'P' if g2 else 'F'} | upset {du:+.4f} chalk {dc:+.4f} g3 {'P' if g3 else 'F'} | slope {sl:.3f} g4 {'P' if g4 else 'F'}{tag}")
verdict = 'NO VARIANT PASSED GATE 1 -> negative result' if chosen is None else ('ADOPT' if all((report[chosen][g] for g in ('gate2', 'gate3', 'gate4'))) else ('INDETERMINATE (gate 2 CI spans zero)' if report[chosen]['gate3'] and report[chosen]['gate4'] and report[chosen]['dLL'] < 0 else 'REJECT'))
print(f"\nVERDICT for the frozen choice {chosen}: {verdict}")
json.dump({'preregistration_sha256_16': hashlib.sha256(open(OUT + 'PREREGISTRATION_RECENT_REGIME.md', 'rb').read()).hexdigest()[:16], 'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
           'M0_tuning_mean_ll': float(np.mean(list(ll0.values()))), 'M0_recent': {'mean_ll': float(np.mean(list(ll0r.values()))), 'upset_ll': float(np.nanmean(list(up0.values()))), 'chalk_ll': float(np.nanmean(list(ch0.values()))), 'slope': cal_slope(P0, RECENT)},
           'chosen': chosen, 'recent': report, 'verdict': verdict, 'subset_sizes': {'upset': int(upset.sum()), 'chalk': int(chalk.sum())}}, open(OUT + 'results.json', 'w'), indent=1, default=float)
