#!/usr/bin/env python3
"""Does opponent-adjusted road margin beat the true_road_win_pct column?

WHY THIS COMPARISON. generate_team_stats_table builds true_road_win_pct as a
win RATE over away-and-neutral games. A rate discards margin, which 735df4a
deliberately kept, and its denominator is pathological early on -- one road
game reads as 0.000 or 1.000. The proposed replacement is an opponent-adjusted
average road margin with per-team shrinkage. Whether that actually predicts
better, and on which venue population, is measured here rather than argued.

WHAT IS HELD FIXED. Every candidate is scored as an ADDITION to the same
baseline: a point-in-time SRS differential. That matters because these
features are never used alone -- they sit alongside opponent-adjusted ratings,
so the only question that counts is what they add once strength is already
accounted for. A feature that looks strong alone and adds nothing on top of
SRS is measuring strength a second time, not road performance.

Everything stays in Kaggle team-ID space on purpose: it needs no name
normalisation and does not read docs/data/team_stats_by_year.json, so this
measurement is independent of the torvik vintage reconciliation.

Scoring is leave-one-year-out: fit on every other season, predict the held-out
one. Target is tournament game margin, oriented by team ID so it cannot encode
the winner.

Run: python3 scripts/evaluate_road_features.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.features.point_in_time_ratings import (  # noqa: E402
    SELECTION_SUNDAY_DAY,
    games_before,
    load_season_games,
    road_margin_adjusted,
    srs,
)

KAGGLE = REPO / "data" / "kaggle"
SEASONS = [y for y in range(2010, 2027) if y != 2020]


def road_win_rate(games, include_neutral: bool) -> dict[int, float]:
    """The existing feature's definition, recomputed here for a like-for-like
    comparison. include_neutral=True reproduces true_road_win_pct."""
    wins: dict[int, int] = {}
    played: dict[int, int] = {}

    def bump(team: int, won: bool) -> None:
        played[team] = played.get(team, 0) + 1
        wins[team] = wins.get(team, 0) + (1 if won else 0)

    for g in games:
        if g.winner_loc == "A":
            bump(g.winner, True)
        elif g.winner_loc == "H":
            bump(g.loser, False)
        elif include_neutral:
            bump(g.winner, True)
            bump(g.loser, False)
    return {t: wins[t] / played[t] for t in played if played[t]}


def tourney_games(season: int):
    path = KAGGLE / "MNCAATourneyCompactResults.csv"
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if int(r["Season"]) != season:
                continue
            w, l = int(r["WTeamID"]), int(r["LTeamID"])
            ws, ls = int(r["WScore"]), int(r["LScore"])
            a, b = sorted((w, l))
            margin = (ws - ls) if a == w else (ls - ws)
            out.append((a, b, float(margin)))
    return out


def main() -> int:
    if not (KAGGLE / "MNCAATourneyCompactResults.csv").exists():
        print("missing MNCAATourneyCompactResults.csv")
        return 1

    def residualised(games, ratings, include_neutral: bool):
        """Road margin above the team's own overall rating.

        The plain adjusted margin turns out to be a near-restatement of team
        strength, which is why it adds nothing on top of SRS. Subtracting the
        team's own SRS asks the question the feature was actually meant to
        ask: does this team hold up away from home relative to how good it is?

        This is NOT the actual-minus-expected form that was rejected as
        architecturally incompatible. That one is defined against the fitted
        model's expectation, so it would change whenever the UI refits on a
        different variable subset. SRS is a fixed statistic of the game
        results, so this stays computable once per as-of date.
        """
        rma = road_margin_adjusted(games, ratings, include_neutral=include_neutral)
        return {t: v - ratings.get(t, 0.0) for t, v in rma.items()}

    candidates = {
        "road_wr_away+neutral": lambda g, r: road_win_rate(g, True),
        "road_wr_true_only": lambda g, r: road_win_rate(g, False),
        "adj_margin_true_only": lambda g, r: road_margin_adjusted(g, r, include_neutral=False),
        "adj_margin_away+neutral": lambda g, r: road_margin_adjusted(g, r, include_neutral=True),
        "adj_margin_resid_true": lambda g, r: residualised(g, r, False),
        "adj_margin_resid_away+neut": lambda g, r: residualised(g, r, True),
    }

    # rows[season] = list of (srs_diff, {name: feat_diff}, margin)
    rows: dict[int, list] = {}
    for season in SEASONS:
        games = load_season_games(season)
        if not games:
            continue
        past = games_before(games, SELECTION_SUNDAY_DAY)
        ratings = srs(past)
        feats = {name: fn(past, ratings) for name, fn in candidates.items()}

        season_rows = []
        for a, b, margin in tourney_games(season):
            if a not in ratings or b not in ratings:
                continue
            diffs = {}
            ok = True
            for name, table in feats.items():
                if a not in table or b not in table:
                    ok = False
                    break
                diffs[name] = table[a] - table[b]
            if ok:
                season_rows.append((ratings[a] - ratings[b], diffs, margin))
        if season_rows:
            rows[season] = season_rows

    years = sorted(rows)
    total = sum(len(rows[y]) for y in years)
    print(f"\n{total:,} tournament games across {len(years)} seasons, leave-one-year-out\n")

    def design(r, names: list[str], with_srs: bool) -> list[float]:
        head = [1.0, r[0]] if with_srs else [1.0]
        return head + [r[1][n] for n in names]

    def loyo_errors(names: list[str], with_srs: bool = True) -> np.ndarray:
        """Held-out residuals, one per tournament game, in a fixed game order
        so residuals from different feature sets are paired elementwise.

        with_srs=False drops the strength baseline, which separates "this
        feature is uninformative" from "this feature is informative but
        redundant with strength" -- opposite conclusions that the incremental
        test alone cannot tell apart.
        """
        errs: list[float] = []
        for held in years:
            train = [r for y in years if y != held for r in rows[y]]
            X = np.array([design(r, names, with_srs) for r in train])
            y = np.array([r[2] for r in train])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            Xh = np.array([design(r, names, with_srs) for r in rows[held]])
            yh = np.array([r[2] for r in rows[held]])
            errs.extend((yh - Xh @ beta).tolist())
        return np.array(errs)

    def loyo_rmse(names: list[str]) -> float:
        return float(np.sqrt(np.mean(np.square(loyo_errors(names)))))

    def bootstrap_ci(base_err: np.ndarray, cand_err: np.ndarray, n: int = 5000):
        """Paired bootstrap over games on the RMSE difference. Paired because
        both models are scored on the identical games; resampling them
        independently would swamp a small real difference with between-game
        variance that cancels in the pairing."""
        rng = np.random.default_rng(0)
        m = len(base_err)
        out = np.empty(n)
        for i in range(n):
            idx = rng.integers(0, m, m)
            out[i] = np.sqrt(np.mean(base_err[idx] ** 2)) - np.sqrt(np.mean(cand_err[idx] ** 2))
        return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

    base_err = loyo_errors([])
    base = float(np.sqrt(np.mean(np.square(base_err))))
    print(f"  {'baseline: SRS differential only':<34} RMSE {base:7.4f}")
    print()

    results = []
    for name in candidates:
        cand_err = loyo_errors([name])
        r = float(np.sqrt(np.mean(np.square(cand_err))))
        results.append((r, name))
        lo, hi = bootstrap_ci(base_err, cand_err)
        verdict = "significant" if lo > 0 else "not distinguishable from 0"
        print(
            f"  + {name:<24} RMSE {r:7.4f}  delta {base - r:+7.4f}  "
            f"95% CI [{lo:+.4f}, {hi:+.4f}]  {verdict}"
        )

    both = loyo_rmse(["adj_margin_true_only", "road_wr_away+neutral"])
    print(f"\n  + both adj_margin_true_only and road_wr_away+neutral: RMSE {both:7.4f} "
          f"(delta {base - both:+7.4f})")

    # Informative-but-redundant, or uninformative? Score each feature with no
    # strength term at all, against an intercept-only floor.
    floor = float(np.sqrt(np.mean(np.square(loyo_errors([], with_srs=False)))))
    print(f"\n  alone, without any strength term (intercept-only floor = {floor:.4f}):")
    for name in candidates:
        r = float(np.sqrt(np.mean(np.square(loyo_errors([name], with_srs=False)))))
        print(f"  + {name:<24} RMSE {r:7.4f}  delta vs floor {floor - r:+7.4f}")

    best_r, best_name = min(results)
    print()
    if best_r < base:
        print(f"  best single addition: {best_name} ({base - best_r:+.4f} RMSE improvement)")
    else:
        print(f"  NO candidate improves on the SRS-only baseline (best was {best_name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
