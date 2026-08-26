#!/usr/bin/env python3
"""Measure whether point-in-time SRS is worth using, week by week, out of sample.

WHY THIS EXISTS. Point-in-time SRS is only useful if it beats the things it
would replace, at the dates it would actually be used. Early in a season it is
noise-dominated: a team with three games has a rating built from three games.
The honest test is out-of-sample and forward-looking -- ratings solved from
games before a week boundary, scored against the margins of games played after
it -- because a rating scored on the games that built it will always look good.

WHAT IS COMPARED, all at the same boundary
  zero    predict every margin as 0. The floor. Any feature that cannot beat
          this is not a feature.
  prior   prior season's final SRS. Knows nothing about this season but is
          built on a full season of games.
  pit     this season's SRS solved from games before the boundary only.
  blend   per-team shrinkage of pit toward prior, lambda = k/(k+games).

Games are oriented by team ID, not by who won, so the target carries no
information about the outcome. Home court is not modelled; it is a constant
offset that penalises all four methods identically and would not change the
ranking between them.

Run: python3 scripts/validate_point_in_time_srs.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.features.point_in_time_ratings import (  # noqa: E402
    SELECTION_SUNDAY_DAY,
    game_counts,
    games_before,
    load_season_games,
    shrink_to_prior,
    srs,
)

SEASONS = [y for y in range(2011, 2027) if y not in (2020, 2021)]
WEEKS = list(range(2, 12))
DAYS_PER_WEEK = 7


def rmse(errors: list[float]) -> float:
    return float(np.sqrt(np.mean(np.square(errors)))) if errors else float("nan")


def main() -> int:
    # errors[method][week] accumulates across seasons
    errors: dict[str, dict[int, list[float]]] = {m: defaultdict(list) for m in ("zero", "prior", "pit", "blend")}
    ks = [4, 5, 6, 7, 8, 10]
    k_errors: dict[int, dict[int, list[float]]] = {k: defaultdict(list) for k in ks}

    for season in SEASONS:
        games = load_season_games(season)
        prev = load_season_games(season - 1)
        if not games or not prev:
            continue
        prior = srs(games_before(prev, SELECTION_SUNDAY_DAY))

        for week in WEEKS:
            cutoff = week * DAYS_PER_WEEK
            past = games_before(games, cutoff)
            if not past:
                continue
            # score against the NEXT week only: strictly out of sample
            future = [g for g in games if cutoff <= g.day < cutoff + DAYS_PER_WEEK]
            if not future:
                continue

            pit = srs(past)
            counts = game_counts(past)
            blend = shrink_to_prior(pit, prior, counts)
            # one blend per k per boundary, not per game
            blends_k = {k: shrink_to_prior(pit, prior, counts, k=float(k)) for k in ks}

            for g in future:
                # orient by team id so the target cannot encode the winner
                a, b = sorted((g.winner, g.loser))
                if a == g.winner:
                    margin = g.winner_score - g.loser_score
                else:
                    margin = g.loser_score - g.winner_score

                errors["zero"][week].append(margin)
                errors["prior"][week].append(margin - (prior.get(a, 0.0) - prior.get(b, 0.0)))
                errors["pit"][week].append(margin - (pit.get(a, 0.0) - pit.get(b, 0.0)))
                errors["blend"][week].append(margin - (blend.get(a, 0.0) - blend.get(b, 0.0)))

                for k, bk in blends_k.items():
                    k_errors[k][week].append(margin - (bk.get(a, 0.0) - bk.get(b, 0.0)))

    print("\nOut-of-sample RMSE predicting next week's margins\n")
    header = "  week " + " ".join(f"{w:>7}" for w in WEEKS)
    print(header)
    for method in ("zero", "prior", "pit", "blend"):
        row = " ".join(f"{rmse(errors[method][w]):>7.2f}" for w in WEEKS)
        print(f"  {method:<5}{row}")

    n = sum(len(errors["zero"][w]) for w in WEEKS)
    print(f"\n  {n:,} game-predictions across {len(SEASONS)} seasons")

    print("\nBlend sensitivity to k\n")
    print(header)
    for k in ks:
        row = " ".join(f"{rmse(k_errors[k][w]):>7.2f}" for w in WEEKS)
        print(f"  k={k:<3}{row}")

    # The two claims this script exists to check.
    print()
    blend_beats = [
        w for w in WEEKS if rmse(errors["blend"][w]) <= min(rmse(errors["pit"][w]), rmse(errors["prior"][w])) + 1e-9
    ]
    print(f"  blend <= both components at {len(blend_beats)}/{len(WEEKS)} weeks: {blend_beats}")
    crossover = [w for w in WEEKS if rmse(errors["pit"][w]) < rmse(errors["prior"][w])]
    first = min(crossover) if crossover else None
    print(f"  pit first beats prior at week {first}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
