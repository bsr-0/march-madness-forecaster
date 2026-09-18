"""Women's tournament win-probability model for the Kaggle submission.

WHY THIS EXISTS. Kaggle's March Machine Learning Mania scores one CSV across
both tournaments, so a submission needs P(a beats b) for women's teams too.
Nothing else in the repo models the women's game: the site, the pool
optimizer and every feature source (Torvik, KenPom, Massey, coaches) are
men's-only. This module fills the women's half of the CSV with the SAME model
family the site ships for the men's field -- ridge on standardised feature
differences, read out through a Student-t link, fit strictly on earlier
seasons -- with the features rebuilt from the only women's data the repo has,
Kaggle's regular-season box scores.

IT REUSES THE PORT RATHER THAN COPYING IT. fit_linear, calibrate, walk_forward
and the link come from src/prediction/pit_production_model.py, so the women's
model changes if and only if the shipped men's model does. What is new here is
the feature table. Kaggle publishes no Torvik for the women's game, so barthag
/ t_rank / the adjusted efficiencies are replaced by what the box scores
support: a Massey-style margin rating (srs) and its strength of schedule,
opponent-adjusted offensive and defensive efficiency, tempo, and the four
factors. Names are shared with the men's table where the meaning is the same
so the two feature lists read side by side.

THE STANDARDISATION POPULATION DIFFERS FROM THE MEN'S, DELIBERATELY. docs/fit.js
standardises within the 68-team field. That needs the field, and the Kaggle
snapshot downloaded before Stage 2 carries no seeds for the season being
predicted. Standardising within the season's full D1 population needs nothing
that arrives late, and is applied identically to training and prediction
rows, which is the property that matters. It does mean a women's z-score and a
men's z-score are on different scales; nothing compares them.

THE LEAKAGE RULE is the men's one: for tournament year Y, beta, sigma and the
link's (a, nu) come from seasons strictly before Y, enforced by
pairwise_for_year rather than trusted to the caller.

scripts/backtest_womens_brier.py is the measurement: walk-forward Brier
against a seed-only baseline, written to artifacts/. If that number is not
better than the seed baseline, this model should not be in the CSV.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.prediction.pit_production_model import (
    CAL_PRIOR_STRENGTH,
    MIN_TEST_YEAR,
    calibrate,
    clip_prob,
    fit_linear,
    score,
    student_t_cdf,
    walk_forward,
    walk_forward_calibration,
)

REPO = Path(__file__).resolve().parent.parent.parent
DEFAULT_KAGGLE_DIR = REPO / "data" / "kaggle"

REG_SEASON_FILE = "WRegularSeasonDetailedResults.csv"
TOURNEY_FILE = "WNCAATourneyCompactResults.csv"
SEEDS_FILE = "WNCAATourneySeeds.csv"

# (key, higher_is_better). Sign only affects readability of the z-scores; a
# zero-intercept linear fit is invariant to it. Signs copied from the men's
# VARIABLES table in scripts/build_ui_payload.py for the shared names.
WOMENS_KEYS: Tuple[Tuple[str, bool], ...] = (
    ("srs", True),
    ("sos", True),
    ("adj_offensive_efficiency", True),
    ("adj_defensive_efficiency", False),
    ("adj_tempo", True),
    ("effective_fg_pct", True),
    ("three_pt_pct", True),
    ("three_pt_rate", True),
    ("offensive_reb_rate", True),
    ("turnover_rate", False),
)
KEY_NAMES: Tuple[str, ...] = tuple(k for k, _ in WOMENS_KEYS)

# Teams with fewer regular-season box scores than this get no feature row.
# They are almost always non-D1 opponents that appear in a handful of games.
MIN_GAMES = 10

# Ridge on the Massey system. Small: its job is to pin the rating scale (the
# system is otherwise only determined up to a constant) and to keep a
# disconnected schedule graph solvable, not to shrink anything meaningful.
SRS_RIDGE = 1.0
ADJ_ITERATIONS = 10

_SEED_RE = re.compile(r"^[A-Z](\d{2})")


# ------------------------------------------------------------------ box scores
@dataclass
class _GameLine:
    """One team's side of one regular-season game."""

    season: int
    team: int
    opp: int
    pts: int
    opp_pts: int
    poss: float
    home: int  # +1 home, -1 away, 0 neutral
    fgm: int
    fga: int
    fgm3: int
    fga3: int
    fta: int
    oreb: int
    to: int
    opp_dreb: int


def _possessions(fga: int, oreb: int, to: int, fta: int) -> float:
    """Standard possession estimate; the 0.475 is the usual FTA weight."""
    return fga - oreb + to + 0.475 * fta


def _read_regular_season(path: Path) -> Dict[int, List[_GameLine]]:
    """Every regular-season game as two _GameLine rows, keyed by season."""
    out: Dict[int, List[_GameLine]] = defaultdict(list)
    with open(path, newline="", encoding="latin-1") as f:
        for r in csv.DictReader(f):
            season = int(r["Season"])
            win, los = int(r["WTeamID"], 10), int(r["LTeamID"], 10)
            ws, ls = int(r["WScore"]), int(r["LScore"])
            wposs = _possessions(int(r["WFGA"]), int(r["WOR"]), int(r["WTO"]), int(r["WFTA"]))
            lposs = _possessions(int(r["LFGA"]), int(r["LOR"]), int(r["LTO"]), int(r["LFTA"]))
            poss = 0.5 * (wposs + lposs)
            loc = r["WLoc"]
            whome = 1 if loc == "H" else (-1 if loc == "A" else 0)
            out[season].append(
                _GameLine(
                    season,
                    win,
                    los,
                    ws,
                    ls,
                    poss,
                    whome,
                    int(r["WFGM"]),
                    int(r["WFGA"]),
                    int(r["WFGM3"]),
                    int(r["WFGA3"]),
                    int(r["WFTA"]),
                    int(r["WOR"]),
                    int(r["WTO"]),
                    int(r["LDR"]),
                )
            )
            out[season].append(
                _GameLine(
                    season,
                    los,
                    win,
                    ls,
                    ws,
                    poss,
                    -whome,
                    int(r["LFGM"]),
                    int(r["LFGA"]),
                    int(r["LFGM3"]),
                    int(r["LFGA3"]),
                    int(r["LFTA"]),
                    int(r["LOR"]),
                    int(r["LTO"]),
                    int(r["WDR"]),
                )
            )
    return dict(out)


# ------------------------------------------------------------------ ratings
def _massey(lines: Sequence[_GameLine], teams: Sequence[int]) -> Tuple[Dict[int, float], float]:
    """Ridge least-squares margin ratings with one home-court term.

    margin(i over j) = r_i - r_j + h * home. Each game appears once, from
    the lower-numbered team's side; its mirrored line is the same equation
    negated and would only double-count, exactly as with the men's
    zero-intercept fit.
    """
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    rows, rhs = [], []
    for g in lines:
        if g.team > g.opp or g.team not in idx or g.opp not in idx:
            continue
        row = np.zeros(n + 1)
        row[idx[g.team]] = 1.0
        row[idx[g.opp]] = -1.0
        row[n] = g.home
        rows.append(row)
        rhs.append(g.pts - g.opp_pts)
    if not rows:
        return {t: 0.0 for t in teams}, 0.0
    A = np.array(rows)
    b = np.array(rhs, dtype=float)
    M = A.T @ A
    M[np.diag_indices(n + 1)] += SRS_RIDGE
    sol = np.linalg.solve(M, A.T @ b)
    return {t: float(sol[idx[t]]) for t in teams}, float(sol[n])


def _adjusted_efficiency(
    by_team: Dict[int, List[_GameLine]], teams: Sequence[int]
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Opponent-adjusted points per 100 possessions, KenPom-style iteration.

    adjO_i = mean over games of (game offence + (league_avg - adjD_opp)); adjD
    symmetric. Ten rounds is well past convergence on a ~5,000-game season.
    """
    tset = set(teams)
    per_game: Dict[int, List[Tuple[int, float, float]]] = {}
    for t in teams:
        per_game[t] = [
            (g.opp, 100.0 * g.pts / g.poss, 100.0 * g.opp_pts / g.poss)
            for g in by_team[t]
            if g.poss > 0 and g.opp in tset
        ]
    all_off = [o for gs in per_game.values() for _, o, _ in gs]
    avg = float(np.mean(all_off)) if all_off else 100.0
    adj_o = {t: float(np.mean([o for _, o, _ in gs])) if gs else avg for t, gs in per_game.items()}
    adj_d = {t: float(np.mean([d for _, _, d in gs])) if gs else avg for t, gs in per_game.items()}
    for _ in range(ADJ_ITERATIONS):
        new_o, new_d = {}, {}
        for t, gs in per_game.items():
            if not gs:
                new_o[t], new_d[t] = avg, avg
                continue
            new_o[t] = float(np.mean([o + (avg - adj_d[opp]) for opp, o, _ in gs]))
            new_d[t] = float(np.mean([d + (avg - adj_o[opp]) for opp, _, d in gs]))
        adj_o, adj_d = new_o, new_d
    return adj_o, adj_d


# ------------------------------------------------------------------ feature table
_SEASON_CACHE: Dict[Tuple[str, int], List[Dict[str, float]]] = {}
_REG_CACHE: Dict[str, Dict[int, List[_GameLine]]] = {}


def _regular_season(data_dir: Path) -> Dict[int, List[_GameLine]]:
    key = str(Path(data_dir).resolve())
    if key not in _REG_CACHE:
        _REG_CACHE[key] = _read_regular_season(Path(data_dir) / REG_SEASON_FILE)
    return _REG_CACHE[key]


def season_rows(season: int, data_dir: Path = DEFAULT_KAGGLE_DIR) -> List[Dict[str, float]]:
    """Raw (unstandardised) feature row per D1 team for one season.

    Each row carries ``team_id`` (the Kaggle TeamID as a string, the only
    identity the women's data has) plus every key in WOMENS_KEYS.
    """
    cache_key = (str(Path(data_dir).resolve()), season)
    if cache_key in _SEASON_CACHE:
        return _SEASON_CACHE[cache_key]

    lines = _regular_season(data_dir).get(season, [])
    by_team: Dict[int, List[_GameLine]] = defaultdict(list)
    for g in lines:
        by_team[g.team].append(g)
    teams = sorted(t for t, gs in by_team.items() if len(gs) >= MIN_GAMES)
    if not teams:
        _SEASON_CACHE[cache_key] = []
        return []

    srs, _home = _massey(lines, teams)
    adj_o, adj_d = _adjusted_efficiency(by_team, teams)

    rows: List[Dict[str, float]] = []
    for t in teams:
        gs = by_team[t]
        fga = sum(g.fga for g in gs)
        fga3 = sum(g.fga3 for g in gs)
        oreb = sum(g.oreb for g in gs)
        poss = sum(g.poss for g in gs)
        opp_srs = [srs[g.opp] for g in gs if g.opp in srs]
        rows.append(
            {
                "team_id": str(t),
                "games": len(gs),
                "srs": srs[t],
                "sos": float(np.mean(opp_srs)) if opp_srs else 0.0,
                "adj_offensive_efficiency": adj_o[t],
                "adj_defensive_efficiency": adj_d[t],
                "adj_tempo": poss / len(gs),
                "effective_fg_pct": (sum(g.fgm for g in gs) + 0.5 * sum(g.fgm3 for g in gs)) / fga if fga else 0.0,
                "three_pt_pct": sum(g.fgm3 for g in gs) / fga3 if fga3 else 0.0,
                "three_pt_rate": fga3 / fga if fga else 0.0,
                "offensive_reb_rate": oreb / (oreb + sum(g.opp_dreb for g in gs)) if oreb else 0.0,
                "turnover_rate": sum(g.to for g in gs) / poss if poss else 0.0,
            }
        )
    _SEASON_CACHE[cache_key] = rows
    return rows


def season_z(
    rows: List[Dict[str, float]], keys: Sequence[Tuple[str, bool]] = WOMENS_KEYS
) -> Dict[str, Dict[str, float]]:
    """Within-season z-scores over the full D1 population, sign-corrected.

    Same arithmetic as scripts/build_ui_payload.zscores (population sd,
    higher-is-better sign flip) so a women's z means what a men's z means,
    just over a different population.
    """
    from scripts.build_ui_payload import zscores

    ids = [r["team_id"] for r in rows]
    out: Dict[str, Dict[str, float]] = {tid: {} for tid in ids}
    for key, higher_better in keys:
        vals = [r.get(key) for r in rows]
        for tid, z in zip(ids, zscores(vals, higher_better)):
            out[tid][key] = z
    return out


# ------------------------------------------------------------------ training matrix
def _parse_seed(code: str) -> Optional[int]:
    m = _SEED_RE.match(code.strip())
    return int(m.group(1)) if m else None


def load_seeds(data_dir: Path = DEFAULT_KAGGLE_DIR) -> Dict[int, Dict[str, int]]:
    """{season: {team_id: seed}} from WNCAATourneySeeds.csv."""
    out: Dict[int, Dict[str, int]] = defaultdict(dict)
    with open(Path(data_dir) / SEEDS_FILE, newline="", encoding="latin-1") as f:
        for r in csv.DictReader(f):
            seed = _parse_seed(r["Seed"])
            if seed is not None:
                out[int(r["Season"])][str(int(r["TeamID"]))] = seed
    return dict(out)


@dataclass
class TrainingMatrix:
    X: np.ndarray
    m: np.ndarray
    y: np.ndarray
    seed1: np.ndarray  # better (or alphabetically-first) seed, per row
    seed2: np.ndarray
    keys: Tuple[str, ...] = KEY_NAMES
    skipped: Dict[int, int] = field(default_factory=dict)  # season -> games without features


def build_training(data_dir: Path = DEFAULT_KAGGLE_DIR) -> TrainingMatrix:
    """One row per tournament game, x = z(team1) - z(team2), m = margin.

    Orientation follows scripts/build_training_matrix._orient: the better
    seed first, ties broken by team id. Both are settled before tip-off so
    the layout carries no information about the result. Play-in games are
    kept -- Kaggle scores them -- which is the one place this matrix differs
    from the men's bracket-only one.
    """
    seeds = load_seeds(data_dir)
    seasons_with_stats = set(_regular_season(data_dir))
    X, m, y, s1, s2 = [], [], [], [], []
    skipped: Dict[int, int] = defaultdict(int)
    with open(Path(data_dir) / TOURNEY_FILE, newline="", encoding="latin-1") as f:
        for r in csv.DictReader(f):
            season = int(r["Season"])
            if season not in seasons_with_stats:
                continue
            z = season_z(season_rows(season, data_dir))
            win, los = str(int(r["WTeamID"])), str(int(r["LTeamID"]))
            if win not in z or los not in z:
                skipped[season] += 1
                continue
            sw, sl = seeds.get(season, {}).get(win), seeds.get(season, {}).get(los)
            margin = int(r["WScore"]) - int(r["LScore"])
            if sw is not None and sl is not None and sw != sl:
                swap = sw > sl
            else:
                swap = win > los
            a, b = (los, win) if swap else (win, los)
            sa, sb = (sl, sw) if swap else (sw, sl)
            X.append([z[a][k] - z[b][k] for k in KEY_NAMES])
            m.append(-margin if swap else margin)
            y.append(season)
            s1.append(sa if sa is not None else 0)
            s2.append(sb if sb is not None else 0)
    return TrainingMatrix(
        np.array(X, dtype=float),
        np.array(m, dtype=float),
        np.array(y, dtype=int),
        np.array(s1, dtype=int),
        np.array(s2, dtype=int),
        skipped=dict(skipped),
    )


# ------------------------------------------------------------------ prediction
def teams_with_stats(year: int, data_dir: Path = DEFAULT_KAGGLE_DIR) -> List[str]:
    return [r["team_id"] for r in season_rows(year, data_dir)]


def pairwise_for_year(
    year: int, team_ids: Sequence[str], data_dir: Path = DEFAULT_KAGGLE_DIR
) -> Dict[Tuple[str, str], float]:
    """P(a beats b) for every ordered pair, fit strictly on seasons before `year`.

    Mirrors pit_production_model.pairwise_for_year: ridge beta and sigma on
    prior seasons, the link's (a, nu) from prior walk-forward predictions
    shrunk toward a=1 with weight n/(n+63), antisymmetric by construction.
    """
    tm = build_training(data_dir)
    cols = list(range(len(KEY_NAMES)))
    prior_rows = tm.y < year
    fit = fit_linear(tm.X[prior_rows], tm.m[prior_rows])
    if fit is None:
        raise ValueError(f"not enough women's training data before {year}")
    beta, sigma = fit

    py, pm, pp, ps = walk_forward(tm.X, tm.m, tm.y, cols)
    prior = py < year
    if int(prior.sum()) >= 1:
        c = calibrate(pm[prior], pp[prior], ps[prior])
        n = int(prior.sum())
        w = n / (n + CAL_PRIOR_STRENGTH)
        a, nu = w * c["a"] + (1 - w) * 1.0, c["nu"]
    else:
        a, nu = 1.0, np.inf

    rows = season_rows(year, data_dir)
    if not rows:
        raise ValueError(f"no women's regular-season box scores for {year}")
    z = season_z(rows)
    missing = [t for t in team_ids if t not in z]
    if missing:
        raise KeyError(f"{len(missing)} women's teams have no {year} stats: {missing[:5]}")

    out: Dict[Tuple[str, str], float] = {}
    for i, ta in enumerate(team_ids):
        for tb in team_ids[i + 1 :]:
            x = np.array([z[ta][k] - z[tb][k] for k in KEY_NAMES])
            p = float(clip_prob(student_t_cdf(a * float(x @ beta) / sigma, nu)))
            out[(ta, tb)] = p
            out[(tb, ta)] = 1.0 - p
    return out


# ------------------------------------------------------------------ backtest
def _seed_baseline(tm: TrainingMatrix, min_year: int) -> Dict[int, np.ndarray]:
    """Walk-forward seed-only probabilities: logistic in (seed2 - seed1).

    The slope is the one value fit, on strictly earlier seasons, by a grid
    over log loss. This is the baseline the model has to beat to earn its
    place in the CSV; a seed-only submission is what anyone can file.
    """
    diff = (tm.seed2 - tm.seed1).astype(float)
    won = (tm.m > 0).astype(float)
    out: Dict[int, np.ndarray] = {}
    grid = np.linspace(0.02, 0.6, 59)
    for yr in sorted(set(tm.y.tolist())):
        if yr < min_year:
            continue
        prior = tm.y < yr
        best_k, best_ll = 0.175, np.inf
        for k in grid:
            p = clip_prob(1.0 / (1.0 + np.exp(-k * diff[prior])))
            ll = float(-(won[prior] * np.log(p) + (1 - won[prior]) * np.log(1 - p)).mean())
            if ll < best_ll:
                best_k, best_ll = k, ll
        out[yr] = clip_prob(1.0 / (1.0 + np.exp(-best_k * diff[tm.y == yr])))
    return out


def walk_forward_report(data_dir: Path = DEFAULT_KAGGLE_DIR, min_year: int = MIN_TEST_YEAR) -> Dict[str, object]:
    """Per-season and pooled Brier / log loss, model vs seed-only, walk-forward.

    Every number here is out of sample under the leakage rule: season Y is
    scored with a fit and a link from seasons < Y.
    """
    tm = build_training(data_dir)
    cols = list(range(len(KEY_NAMES)))
    py, pm, pp, ps = walk_forward(tm.X, tm.m, tm.y, cols, min_year=min_year)
    cal = walk_forward_calibration(py, pm, pp, ps)
    seed_p = _seed_baseline(tm, min_year)

    per_year = []
    for yr in sorted(set(py.tolist())):
        k = py == yr
        s = score(py[k], pm[k], pp[k], ps[k], {yr: cal[yr]})
        won = (tm.m[tm.y == yr] > 0).astype(float)
        sp = seed_p[yr]
        seed_brier = float(((sp - won) ** 2).mean())
        per_year.append(
            {
                "year": int(yr),
                "n_games": s["n"],
                "model_brier": round(s["brier"], 4),
                "model_log_loss": round(s["logLoss"], 4),
                "seed_brier": round(seed_brier, 4),
                "bss_vs_seed": round(1.0 - s["brier"] / seed_brier, 4),
                "accuracy": round(s["accuracy"], 4),
                "link_a": round(cal[yr]["a"], 4),
                "link_nu": None if not np.isfinite(cal[yr]["nu"]) else float(cal[yr]["nu"]),
            }
        )
    pooled = score(py, pm, pp, ps, cal)
    all_won = np.concatenate([(tm.m[tm.y == yr] > 0).astype(float) for yr in sorted(set(py.tolist()))])
    all_seed = np.concatenate([seed_p[yr] for yr in sorted(set(py.tolist()))])
    seed_pooled = float(((all_seed - all_won) ** 2).mean())
    return {
        "keys": list(KEY_NAMES),
        "min_test_year": min_year,
        "n_training_games": int(len(tm.y)),
        "skipped_games": tm.skipped,
        "per_year": per_year,
        "pooled": {
            "n_games": pooled["n"],
            "model_brier": round(pooled["brier"], 4),
            "model_log_loss": round(pooled["logLoss"], 4),
            "seed_brier": round(seed_pooled, 4),
            "bss_vs_seed": round(1.0 - pooled["brier"] / seed_pooled, 4),
            "seasons_model_beats_seed": sum(1 for r in per_year if r["model_brier"] < r["seed_brier"]),
            "n_seasons": len(per_year),
        },
    }
