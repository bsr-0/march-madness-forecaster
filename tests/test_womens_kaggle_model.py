"""Women's Kaggle model on a synthetic season: features, orientation, antisymmetry.

The fit machinery is the men's port and is gated by tests/test_pit_production_port.py;
what is tested here is the part that is new -- the box-score feature table, the
training-row orientation, and that pairwise_for_year keeps the men's contracts.
"""

import csv
import random

import numpy as np
import pytest

from src.prediction import womens_kaggle_model as W

REG_COLS = [
    "Season",
    "DayNum",
    "WTeamID",
    "WScore",
    "LTeamID",
    "LScore",
    "WLoc",
    "NumOT",
    "WFGM",
    "WFGA",
    "WFGM3",
    "WFGA3",
    "WFTM",
    "WFTA",
    "WOR",
    "WDR",
    "WAst",
    "WTO",
    "WStl",
    "WBlk",
    "WPF",
    "LFGM",
    "LFGA",
    "LFGM3",
    "LFGA3",
    "LFTM",
    "LFTA",
    "LOR",
    "LDR",
    "LAst",
    "LTO",
    "LStl",
    "LBlk",
    "LPF",
]


def _box(score, rng):
    fga = rng.randint(50, 70)
    fgm = max(10, min(fga, int(score * 0.42)))
    fga3 = rng.randint(12, 25)
    return [
        fgm,
        fga,
        min(fga3, fgm // 3),
        fga3,
        10,
        rng.randint(12, 20),
        rng.randint(6, 14),
        rng.randint(20, 30),
        12,
        rng.randint(10, 20),
        6,
        3,
        15,
    ]


def _write_season(kaggle_dir, season, strengths, rng, games_per_pair=2):
    """Round-robin season; team strength sets expected margin and win chance."""
    teams = sorted(strengths)
    reg = []
    day = 10
    for _ in range(games_per_pair):
        for i, a in enumerate(teams):
            for b in teams[i + 1 :]:
                margin = strengths[a] - strengths[b] + rng.gauss(0, 8)
                a_pts = 65 + int(margin / 2)
                b_pts = 65 - int(margin / 2)
                if a_pts == b_pts:
                    a_pts += 1
                win, los = (a, b) if a_pts > b_pts else (b, a)
                ws, ls = max(a_pts, b_pts), min(a_pts, b_pts)
                loc = rng.choice(["H", "A", "N"])
                reg.append([season, day, win, ws, los, ls, loc, 0] + _box(ws, rng) + _box(ls, rng))
                day += 1
    with open(kaggle_dir / W.REG_SEASON_FILE, "a", newline="") as f:
        csv.writer(f).writerows(reg)


@pytest.fixture
def synthetic(tmp_path):
    """Four seasons, 32 teams (31 tournament games each; fit_linear needs 5 rows per feature),
    strengths fixed across seasons so the model has something to learn."""
    rng = random.Random(7)
    kaggle_dir = tmp_path
    strengths = {3100 + i: 20 - 1.25 * i for i in range(32)}  # 3100 strongest
    with open(kaggle_dir / W.REG_SEASON_FILE, "w", newline="") as f:
        csv.writer(f).writerow(REG_COLS)
    seeds, tourney = (
        [["Season", "Seed", "TeamID"]],
        [["Season", "DayNum", "WTeamID", "WScore", "LTeamID", "LScore", "WLoc", "NumOT"]],
    )
    for season in (2011, 2012, 2013, 2014):
        _write_season(kaggle_dir, season, strengths, rng)
        ranked = sorted(strengths, key=lambda t: -strengths[t])
        for s, t in enumerate(ranked, start=1):
            seeds.append([season, f"W{s:02d}", t])
        # single-elimination: 1v16, 2v15 ... then by strength, the stronger team usually wins
        alive = ranked[:]
        day = 137
        while len(alive) > 1:
            nxt = []
            for i in range(len(alive) // 2):
                a, b = alive[i], alive[-(i + 1)]
                margin = strengths[a] - strengths[b] + rng.gauss(0, 8)
                win, los = (a, b) if margin > 0 else (b, a)
                m = abs(int(margin)) or 1
                tourney.append([season, day, win, 60 + m, los, 60, "N", 0])
                nxt.append(win)
            alive = sorted(nxt, key=lambda t: -strengths[t])
            day += 1
    with open(kaggle_dir / W.SEEDS_FILE, "w", newline="") as f:
        csv.writer(f).writerows(seeds)
    with open(kaggle_dir / W.TOURNEY_FILE, "w", newline="") as f:
        csv.writer(f).writerows(tourney)
    W._SEASON_CACHE.clear()
    W._REG_CACHE.clear()
    return kaggle_dir, strengths


def test_season_rows_cover_every_team_with_every_key(synthetic):
    kaggle_dir, strengths = synthetic
    rows = W.season_rows(2014, kaggle_dir)
    assert {r["team_id"] for r in rows} == {str(t) for t in strengths}
    for r in rows:
        assert set(W.KEY_NAMES) <= set(r)
        assert all(np.isfinite(r[k]) for k in W.KEY_NAMES)


def test_srs_recovers_strength_order(synthetic):
    kaggle_dir, strengths = synthetic
    rows = W.season_rows(2014, kaggle_dir)
    by_srs = [int(r["team_id"]) for r in sorted(rows, key=lambda r: -r["srs"])]
    truth = sorted(strengths, key=lambda t: -strengths[t])
    # Spearman on 32 teams; margins are noisy so demand strong, not perfect, agreement.
    rho = np.corrcoef([truth.index(t) for t in by_srs], range(32))[0, 1]
    assert rho > 0.9
    assert by_srs[0] == truth[0]


def test_season_z_is_standardised_and_sign_corrected(synthetic):
    kaggle_dir, _ = synthetic
    z = W.season_z(W.season_rows(2014, kaggle_dir))
    for key, higher_better in W.WOMENS_KEYS:
        vals = np.array([z[t][key] for t in z])
        assert abs(vals.mean()) < 1e-3
        assert abs(vals.std() - 1) < 1e-2
    # adj_defensive_efficiency is lower-is-better: the best defence gets the highest z
    rows = W.season_rows(2014, kaggle_dir)
    best_d = min(rows, key=lambda r: r["adj_defensive_efficiency"])["team_id"]
    assert z[best_d]["adj_defensive_efficiency"] == max(z[t]["adj_defensive_efficiency"] for t in z)


def test_training_rows_are_better_seed_first(synthetic):
    kaggle_dir, _ = synthetic
    tm = W.build_training(kaggle_dir)
    assert len(tm.y) == 4 * 31
    assert tm.X.shape == (124, len(W.KEY_NAMES))
    assert (tm.seed1 <= tm.seed2).all()
    assert not tm.skipped
    # better seed first, so the mean margin is positive but not every row is a win
    assert tm.m.mean() > 0
    assert (tm.m > 0).mean() < 1.0


def test_pairwise_is_antisymmetric_and_ranks_strength(synthetic):
    kaggle_dir, strengths = synthetic
    teams = [str(t) for t in strengths]
    pw = W.pairwise_for_year(2014, teams, kaggle_dir)
    assert len(pw) == 32 * 31
    for (a, b), p in pw.items():
        assert 0 < p < 1
        assert abs(p + pw[(b, a)] - 1.0) < 1e-9
    assert pw[("3100", "3131")] > 0.8
    assert pw[("3100", "3101")] > 0.5


def test_pairwise_refuses_unknown_team_and_too_early_year(synthetic):
    kaggle_dir, strengths = synthetic
    with pytest.raises(KeyError):
        W.pairwise_for_year(2014, ["3100", "9999"], kaggle_dir)
    with pytest.raises(ValueError):
        W.pairwise_for_year(2011, ["3100", "3101"], kaggle_dir)


def test_walk_forward_report_shape(synthetic):
    kaggle_dir, _ = synthetic
    rep = W.walk_forward_report(kaggle_dir, min_year=2013)
    assert [r["year"] for r in rep["per_year"]] == [2013, 2014]
    assert rep["pooled"]["n_games"] == 62
    for r in rep["per_year"]:
        assert 0 <= r["model_brier"] <= 1 and 0 <= r["seed_brier"] <= 1
