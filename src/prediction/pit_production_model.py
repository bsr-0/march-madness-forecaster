"""Python port of the browser model that ships in docs/fit.js.

WHY THIS EXISTS. The model the UI ships -- ridge on eleven standardised
feature differences, read out through a Student-t link -- is the only model in
this repo whose held-out accuracy has been measured end to end (log loss
0.45296 on 630 walk-forward tournament games). The pool optimizer does not use
it. It runs on `noseed_model` and on Torvik barthag, neither of which has been
scored by that harness, so the best-validated probabilities available have
never been tested for pool EV. This module is the bridge.

IT IS A PORT, NOT A REIMPLEMENTATION, and the distinction is the whole point.
Every constant here is copied from docs/fit.js and scripts/model_baseline.js
rather than chosen: no intercept, ridge 1.0 * n/1000 on every diagonal, sigma
as the in-sample training residual, the nine-value nu grid, golden-section
search on a over [0.2, 3.0], and shrinkage of a toward 1 with weight
n / (n + 63). Several of those are defensible rather than optimal -- an
in-sample sigma is not the held-out one, and the research harness in
scripts/experiment_model_families.py deliberately differs -- but matching the
shipped model matters more than improving it, because a bridge that quietly
predicts something else would make every pool number downstream describe a
model nobody ships.

scripts/verify_pit_port.py is the gate: it reproduces the frozen baseline in
artifacts/model_baseline.json from this code. If that check fails, nothing
built on top of this module means anything, and it should fail loudly.

THE LEAKAGE RULE, which is easy to lose when crossing a language boundary. For
tournament year Y, everything -- beta, sigma, and the link's (a, nu) -- comes
from seasons strictly before Y. `pairwise_for_year` enforces it rather than
trusting the caller.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm, t as student_t

REPO = Path(__file__).resolve().parent.parent.parent
TRAINING = REPO / "docs" / "data" / "training.json"
STATS = REPO / "docs" / "data" / "team_stats_by_year.json"

# --- constants copied from docs/fit.js (FIT) and scripts/model_baseline.js ---
LAMBDA = 1.0
MIN_ROWS_PER_COL = 5
MIN_TEST_YEAR = 2014
PROB_CLIP = 1e-3
CAL_PRIOR_STRENGTH = 63
NUS: Tuple[float, ...] = (2, 3, 4, 6, 8, 12, 20, 40, np.inf)
WARM_PRIOR_N = 100

CANONICAL_KEYS: Tuple[str, ...] = (
    "barthag",
    "t_rank",
    "sos_avg_opp_barthag",
    "adj_offensive_efficiency",
    "adj_defensive_efficiency",
    "adj_tempo",
    "effective_fg_pct",
    "three_pt_pct",
    "three_pt_rate",
    "offensive_reb_rate",
    "turnover_rate",
)


def clip_prob(p):
    return np.clip(p, PROB_CLIP, 1 - PROB_CLIP)


def student_t_cdf(x, nu):
    """docs/fit.js studentTCdf: normal once nu is large enough to not matter."""
    if not (nu < 1e6):
        return norm.cdf(x)
    return student_t.cdf(x, nu)


# ------------------------------------------------------------------ data
def load_training(path: Path = TRAINING):
    """Return (keys, X, m, y) from the shipped training matrix."""
    doc = json.loads(path.read_text())
    keys = list(doc["keys"])
    X = np.array([g["x"] for g in doc["games"]], dtype=float)
    m = np.array([g["m"] for g in doc["games"]], dtype=float)
    y = np.array([g["y"] for g in doc["games"]], dtype=int)
    return keys, X, m, y


def resolve_cols(keys: Sequence[str], wanted: Sequence[str]) -> List[int]:
    """Port of resolveCols: fail loudly on a renamed or dropped variable.

    The baseline is defined by key, not by position, so a silent reordering
    upstream must not quietly change which model is being measured.
    """
    missing = [k for k in wanted if k not in keys]
    if missing:
        raise KeyError(
            f"training.json is missing baseline variables: {', '.join(missing)}. "
            "The baseline is defined by key, so a renamed or dropped variable "
            "must be dealt with deliberately."
        )
    return [list(keys).index(k) for k in wanted]


# ------------------------------------------------------------------ fit
def fit_linear(X: np.ndarray, m: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    """Ridge with NO INTERCEPT, exactly as docs/fit.js fitLinear.

    The missing intercept is deliberate upstream: rows are oriented by seed and
    mirrored, so the model claims "these two teams differ by this many points"
    and its null is a tie, not a mean margin. Adding an intercept here would fit
    a house edge that the antisymmetry of the row layout says cannot exist.

    sigma is the IN-SAMPLE residual spread over the training rows. That is
    optimistic relative to a held-out estimate, and it is what ships.
    """
    n, k = X.shape
    if not k or n < k * MIN_ROWS_PER_COL:
        return None
    A = X.T @ X
    A[np.diag_indices(k)] += LAMBDA * (n / 1000.0)
    try:
        beta = np.linalg.solve(A, X.T @ m)
    except np.linalg.LinAlgError:
        return None
    sigma = max(float(np.sqrt(((m - X @ beta) ** 2).mean())), 1e-6)
    return beta, sigma


def log_loss_for(m: np.ndarray, pred: np.ndarray, sigma: np.ndarray, a: float, nu: float) -> float:
    """Port of logLossFor.

    NOTE THE DENOMINATOR. The JS divides by rows.length, which counts tied
    games that the numerator skips. On this matrix nothing ties so the two
    agree, but the port copies the shipped arithmetic rather than the intended
    arithmetic -- if a tie ever appears, both implementations should be wrong
    in the same way and the gate should keep catching drift.
    """
    keep = m != 0
    p = clip_prob(student_t_cdf(a * pred[keep] / sigma[keep], nu))
    w = (m[keep] > 0).astype(float)
    return float(-(w * np.log(p) + (1 - w) * np.log(1 - p)).sum() / len(m))


def calibrate(m: np.ndarray, pred: np.ndarray, sigma: np.ndarray) -> Dict[str, float]:
    """Port of calibrate: nu over a fixed grid, a by golden-section search."""
    gr = (np.sqrt(5) - 1) / 2
    best = {"a": 1.0, "nu": np.inf, "logLoss": np.inf}
    for nu in NUS:
        lo, hi = 0.2, 3.0
        c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
        fc, fd = log_loss_for(m, pred, sigma, c, nu), log_loss_for(m, pred, sigma, d, nu)
        for _ in range(30):
            if hi - lo <= 1e-3:
                break
            if fc < fd:
                hi, d, fd = d, c, fc
                c = hi - gr * (hi - lo)
                fc = log_loss_for(m, pred, sigma, c, nu)
            else:
                lo, c, fc = c, d, fd
                d = lo + gr * (hi - lo)
                fd = log_loss_for(m, pred, sigma, d, nu)
        a = (lo + hi) / 2
        ll = log_loss_for(m, pred, sigma, a, nu)
        if ll < best["logLoss"]:
            best = {"a": a, "nu": nu, "logLoss": ll}
    return best


# ------------------------------------------------------- walk-forward
def walk_forward(X: np.ndarray, m: np.ndarray, y: np.ndarray, cols: Sequence[int], min_year: int = MIN_TEST_YEAR):
    """Port of walkForward: fit on strictly earlier seasons, predict one season.

    sigma travels per row because it is refit per fold; collapsing it to a
    global value would be a different model.
    """
    Xc = X[:, list(cols)]
    out_y, out_m, out_pred, out_sigma = [], [], [], []
    for yr in sorted(set(y.tolist())):
        if yr < min_year:
            continue
        tr, te = y < yr, y == yr
        fit = fit_linear(Xc[tr], m[tr])
        if fit is None or not te.sum():
            continue
        beta, sigma = fit
        out_y.append(np.full(int(te.sum()), yr))
        out_m.append(m[te])
        out_pred.append(Xc[te] @ beta)
        out_sigma.append(np.full(int(te.sum()), sigma))
    return (np.concatenate(out_y), np.concatenate(out_m), np.concatenate(out_pred), np.concatenate(out_sigma))


def walk_forward_calibration(py, pm, pp, ps) -> Dict[int, Dict[str, float]]:
    """Port of walkForwardCalibration: per-year link fit on EARLIER years only.

    a is shrunk toward 1 rather than toward the global fit, because the global
    fit is computed from every prediction including the year being scored --
    the leak this replaces. Per-year a is noisy on ~63 binary outcomes, and the
    shrinkage is what keeps that noise out of the scored numbers.
    """
    by_year: Dict[int, Dict[str, float]] = {}
    for yr in sorted(set(py.tolist())):
        prior = py < yr
        n = int(prior.sum())
        if not n:
            by_year[yr] = {"a": 1.0, "nu": np.inf, "priorN": 0}
            continue
        fit = calibrate(pm[prior], pp[prior], ps[prior])
        w = n / (n + CAL_PRIOR_STRENGTH)
        by_year[yr] = {"a": w * fit["a"] + (1 - w) * 1.0, "nu": fit["nu"], "priorN": n}
    return by_year


def score(py, pm, pp, ps, cal: Dict[int, Dict[str, float]]) -> Dict[str, float]:
    """Port of score(). Metrics over non-tied games; rmse/mae over all rows."""
    p = np.empty(len(pm))
    for yr, c in cal.items():
        k = py == yr
        if k.any():
            p[k] = clip_prob(student_t_cdf(c["a"] * pp[k] / ps[k], c["nu"]))
    keep = pm != 0
    w = (pm[keep] > 0).astype(float)
    pk = p[keep]
    return {
        "n": int(keep.sum()),
        "logLoss": float(-(w * np.log(pk) + (1 - w) * np.log(1 - pk)).mean()),
        "brier": float(((pk - w) ** 2).mean()),
        "accuracy": float(((pp[keep] > 0) == (pm[keep] > 0)).mean()),
        "rmse": float(np.sqrt(((pm - pp) ** 2).mean())),
        "mae": float(np.abs(pm - pp).mean()),
    }


# ------------------------------------------------ per-team feature vectors
def season_z(rows: List[Dict], keys: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Within-season z-scores, matching scripts/build_training_matrix.season_z.

    Standardisation is within the 68-team field of that season, which is what
    training.json does, so a difference vector built here is on the same scale
    as the rows beta was fitted on. Getting this wrong would not error -- it
    would silently rescale every prediction -- so it is imported rather than
    reimplemented wherever possible.
    """
    from scripts.build_training_matrix import season_z as _sz

    return _sz(rows)


def pairwise_for_year(
    year: int, team_ids: Sequence[str], keys_wanted: Sequence[str] = CANONICAL_KEYS
) -> Dict[Tuple[str, str], float]:
    """P(a beats b) for every ordered pair, fit strictly on seasons before `year`.

    Returns {(a, b): p}. Antisymmetric by construction: the model has no
    intercept, so margin(a, b) = -margin(b, a) and the link is symmetric about
    zero, giving p(a, b) + p(b, a) = 1 exactly up to the probability clip.
    """
    keys, X, m, y = load_training()
    cols = resolve_cols(keys, keys_wanted)

    fit = fit_linear(X[:, cols][y < year], m[y < year])
    if fit is None:
        raise ValueError(f"not enough training data before {year}")
    beta, sigma = fit

    py, pm, pp, ps = walk_forward(X, m, y, cols)
    prior = py < year
    if int(prior.sum()) >= 1:
        c = calibrate(pm[prior], pp[prior], ps[prior])
        n = int(prior.sum())
        w = n / (n + CAL_PRIOR_STRENGTH)
        a, nu = w * c["a"] + (1 - w) * 1.0, c["nu"]
    else:
        a, nu = 1.0, np.inf

    stats = json.loads(STATS.read_text())["stats_by_year"]
    rows = stats.get(str(year))
    if not rows:
        raise ValueError(f"no team stats for {year}")
    z = season_z(rows, keys_wanted)

    missing = [t for t in team_ids if t not in z]
    if missing:
        raise KeyError(f"{len(missing)} teams have no {year} stats: {missing[:5]}")

    out: Dict[Tuple[str, str], float] = {}
    for i, ta in enumerate(team_ids):
        for tb in team_ids[i + 1 :]:
            x = np.array([z[ta][k] - z[tb][k] for k in keys_wanted])
            p = float(clip_prob(student_t_cdf(a * float(x @ beta) / sigma, nu)))
            out[(ta, tb)] = p
            out[(tb, ta)] = 1.0 - p
    return out
