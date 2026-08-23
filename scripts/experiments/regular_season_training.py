#!/usr/bin/env python3
"""Does training on regular-season games help predict tournament games?

THE QUESTION
The tournament matrix is 1,008 games -- tiny for a 26-parameter model. Regular
season games are ~90x more numerous. If they carry the same relationship between
team quality and margin, adding them should stabilise the coefficients. If they
carry a DIFFERENT relationship (neutral floors, one-and-done intensity, no
travel advantage), pooling them will bias the tournament fit instead.

The flag column is how you find out. Five specifications are compared:

  A  tournament only                  the current model, the baseline
  B  pooled, no flag                  regular + tournament, one set of slopes
  C  pooled + additive flag           adds a plain is_tournament column
  D  pooled + flag interactions       separate slopes per game type
  E  neutral-site regular only        pooled, but only the ~10% of regular
                                      season games played on neutral floors

ANSWER: regular-season games hurt, substantially, and no flag rescues them.
A scores 77.6%; pooling drops it to 72.1%, barely above the 71.7% you get by
always picking the better seed. Interactions recover only part of it (73.5%).

E identifies the main culprit. Restricting the regular-season rows to neutral
floors recovers most of the gap (75.4%), which says the damage is largely HOME
COURT: ~90% of regular-season games carry a ~3.5 point advantage that is not in
the model and is not present in the tournament, where every game is neutral.
Pooling them estimates the slopes on games whose margins contain a large
omitted term. That E still trails A says the rest of the harm is real too --
the circularity described below, and 82,919 regular rows simply outvoting 1,005
tournament rows by 80 to 1.

C is included specifically to demonstrate a trap. The model has no intercept
because the rows are differentials and must be antisymmetric: swapping the two
teams negates x and negates the margin. A game-type flag does NOT negate -- it
is a property of the fixture, identical in both orientations. Under the
symmetric augmentation that forces the intercept to zero, such a column is
forced to zero as well -- exactly zero, not merely small, and C's metrics come
out identical to B's to every digit. The interaction form in D is the one that
actually asks the question, because flag * x DOES negate and so is admissible.

Fit WITHOUT that augmentation and the same flag takes a nonzero value (~1.5
points), which is a trap worth naming: it is measuring the difference in
ORIENTATION CONVENTION between the two subsets, not anything about game type.

EVALUATION
Everything is scored the same way: walk-forward, fit on strictly earlier
seasons, score that season's TOURNAMENT games only. Regular season games are a
training resource here, never a test set -- nobody fills in a bracket for
January.

THE CIRCULARITY, STATED UP FRONT
The predictors are Torvik's end-of-season ratings, captured the day before the
tournament. For a tournament game that is clean point-in-time data: the season
is over, the games being predicted have not been played. For a regular-season
game in December it is not -- the rating used to predict that game was computed
partly FROM that game. Specification B/C/D therefore train on a target the
features already know about, and their training fit will look better than it
has any right to. This does not contaminate the TEST games, which are always
tournament games in a later season, so the comparison below remains valid; but
it does mean a coefficient learned from regular-season rows is a coefficient
learned under look-ahead, and should not be read as a causal effect.

VARIABLE SET
Restricted to the 13 Torvik fields, because those are the only ones available
for all ~360 D1 teams. The other 13 (form, box score, roster, coach) exist only
for tournament qualifiers, so including them would force regular-season games
down to the handful played between two teams that both later made the field --
a sample selected on exactly the quality being modelled. Holding the variable
set fixed at 13 across all five specifications is what makes them comparable.
A on 13 variables scores 77.6% against the shipped model's 78.2% on 26, so
this restriction costs little and is not what drives the result.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

HIST = REPO / "data" / "raw" / "historical"
KAGGLE = REPO / "data" / "kaggle"

# Torvik's numeric fields: present for every D1 team, not just the 68.
TORVIK_KEYS = [
    "t_rank",
    "barthag",
    "adj_offensive_efficiency",
    "adj_defensive_efficiency",
    "adj_tempo",
    "effective_fg_pct",
    "turnover_rate",
    "offensive_reb_rate",
    "free_throw_rate",
    "opp_effective_fg_pct",
    "opp_turnover_rate",
    "defensive_reb_rate",
    "opp_free_throw_rate",
]

# Lower is better for these, so they are sign-flipped when standardised and
# every coefficient reads "more of this is better".
LOWER_IS_BETTER = {
    "t_rank",
    "adj_defensive_efficiency",
    "turnover_rate",
    "opp_effective_fg_pct",
    "opp_free_throw_rate",
}

SKIP_ROUNDS = {"FF"}
YEARS = range(2010, 2027)
MIN_TEST_YEAR = 2014
RIDGE_PER_1000 = 1.0


def zscores(values: List[Any], higher_better: bool) -> List[float]:
    present = [v for v in values if isinstance(v, (int, float))]
    if not present:
        return [0.0] * len(values)
    mean = sum(present) / len(present)
    sd = (sum((v - mean) ** 2 for v in present) / len(present)) ** 0.5
    if sd == 0:
        return [0.0] * len(values)
    sign = 1.0 if higher_better else -1.0
    return [0.0 if not isinstance(v, (int, float)) else sign * (v - mean) / sd for v in values]


def season_z(year: int) -> Dict[str, np.ndarray]:
    """Standardise the 13 Torvik fields across that season's whole D1 field."""
    path = HIST / f"torvik_{year}.json"
    if not path.exists():
        return {}
    teams = json.loads(path.read_text()).get("teams", [])
    if not teams:
        return {}
    cols = {
        k: zscores([t.get(k) for t in teams], k not in LOWER_IS_BETTER)
        for k in TORVIK_KEYS
    }
    return {
        t["team_id"]: np.array([cols[k][i] for k in TORVIK_KEYS])
        for i, t in enumerate(teams)
    }


def kaggle_bridge(canonical_ids) -> Dict[str, str]:
    import re

    def norm(s):
        return re.sub(r"[^a-z0-9]", "", s.lower())

    by_norm = {norm(c): c for c in canonical_ids}
    out = {}
    with open(KAGGLE / "MTeamSpellings.csv", encoding="latin-1") as f:
        for r in csv.DictReader(f):
            c = by_norm.get(norm(r["TeamNameSpelling"]))
            if c is not None:
                out[r["TeamID"]] = c
    return out


def load_regular(year: int, z: Dict[str, np.ndarray]) -> List[Tuple[np.ndarray, float]]:
    """Regular-season + conference-tournament games, oriented deterministically.

    Kaggle's regular-season file contains no NCAA tournament games, so this
    cannot accidentally pull in the thing being predicted.

    Orientation is alphabetical by team id -- a fact of the fixture, not the
    result. Winner-first ordering would put the answer in the row layout.
    """
    bridge = kaggle_bridge(z.keys())
    rows = []
    with open(KAGGLE / "MRegularSeasonCompactResults.csv") as f:
        for r in csv.DictReader(f):
            if int(r["Season"]) != year:
                continue
            w, l = bridge.get(r["WTeamID"]), bridge.get(r["LTeamID"])
            if w is None or l is None or w not in z or l not in z:
                continue
            ws, ls = int(r["WScore"]), int(r["LScore"])
            a, b, sa, sb = (w, l, ws, ls) if w < l else (l, w, ls, ws)
            rows.append((z[a] - z[b], float(sa - sb), r["WLoc"] == "N"))
    return rows


def load_tournament(year: int, z: Dict[str, np.ndarray]) -> List[Tuple[np.ndarray, float]]:
    """Bracket games, oriented better-seed-first (a Selection Sunday fact)."""
    path = HIST / f"tournament_context_{year}.json"
    if not path.exists():
        return []
    res = json.loads(path.read_text()).get("results") or {}
    rows = []
    for g in res.get("games", []):
        if g.get("round_name") in SKIP_ROUNDS:
            continue
        a, b = g.get("team1_id"), g.get("team2_id")
        s1, s2 = g.get("team1_score"), g.get("team2_score")
        if a not in z or b not in z or s1 is None or s2 is None:
            continue
        sa, sb = g.get("team1_seed"), g.get("team2_seed")
        swap = (sa > sb) if (sa is not None and sb is not None and sa != sb) else (a > b)
        if swap:
            a, b, s1, s2 = b, a, s2, s1
        rows.append((z[a] - z[b], float(s1 - s2), True))
    return rows


def design(rows, is_tourney: np.ndarray, spec: str) -> np.ndarray:
    """Build the design matrix for one specification."""
    X = np.array([r[0] for r in rows])
    if spec in ("A", "B"):
        return X
    if spec == "C":
        # Additive flag: a property of the fixture, identical in both
        # orientations. See mirror() for why that makes it inadmissible.
        return np.hstack([X, is_tourney.reshape(-1, 1)])
    if spec in ("D", "E"):
        # Interactions: separate slopes per game type. These DO negate with the
        # differential, so unlike the additive flag they are admissible.
        return np.hstack([X, X * is_tourney.reshape(-1, 1)])
    raise ValueError(spec)


# Columns that are properties of the fixture rather than of the differential,
# and so do not change sign when the two teams are swapped.
INVARIANT_COLS = {"C": [-1]}


def mirror(X: np.ndarray, y: np.ndarray, spec: str):
    """Symmetric augmentation: every row (x, m) also appears as (-x, -m).

    This is the fitting contract the training payload declares, and it is what
    forces the intercept to zero. For the antisymmetric columns it is exactly
    redundant -- X'X and X'y both simply double -- which is why the browser
    skips it. It is applied here because of what it does to the OTHER kind of
    column.

    An orientation-invariant column (the additive flag) keeps its value in the
    mirrored row while the target negates. Its entry in X'y becomes
    sum(f*m) + sum(f*-m) = 0, and its cross terms with every differential
    column become sum(f*x) + sum(f*-x) = 0. The column is orthogonal to
    everything with a right-hand side of zero, so its coefficient is exactly
    zero. Not small -- zero.

    Without mirroring the same column fits a nonzero value, but what it is
    measuring is the ORIENTATION CONVENTION, not the game type: tournament rows
    are ordered better-seed-first (so their margins skew positive) while
    regular-season rows are ordered alphabetically (so theirs are symmetric).
    A coefficient that changes when you rename the teams is not a finding.
    """
    Xm = X.copy()
    keep = INVARIANT_COLS.get(spec, [])
    neg = np.full(X.shape[1], -1.0)
    for c in keep:
        neg[c] = 1.0
    Xm = X * neg
    return np.vstack([X, Xm]), np.concatenate([y, -y])


def fit(X: np.ndarray, y: np.ndarray, spec: str) -> np.ndarray:
    Xa, ya = mirror(X, y, spec)
    lam = RIDGE_PER_1000 * (len(ya) / 1000)
    return np.linalg.solve(Xa.T @ Xa + lam * np.eye(Xa.shape[1]), Xa.T @ ya)


def main() -> int:
    print("Loading seasons ...")
    z_by_year, tourney, regular = {}, {}, {}
    for y in YEARS:
        z = season_z(y)
        if not z:
            continue
        t = load_tournament(y, z)
        if not t:
            continue
        z_by_year[y] = z
        tourney[y] = t
        regular[y] = load_regular(y, z)
    years = sorted(tourney)
    print(f"  {len(years)} seasons: {years[0]}-{years[-1]}")
    print(f"  tournament games {sum(len(v) for v in tourney.values()):,}")
    print(f"  regular games    {sum(len(v) for v in regular.values()):,}")
    print(f"  variables        {len(TORVIK_KEYS)} (Torvik, all-D1)\n")

    specs = {
        "A  tournament only": "A",
        "B  pooled, no flag": "B",
        "C  pooled + additive flag": "C",
        "D  pooled + flag interactions": "D",
        "E  neutral-site regular only": "E",
    }
    results = {}
    flag_coefs = []

    for label, spec in specs.items():
        sse = sae = sst = 0.0
        correct = n = 0
        for ty in years:
            if ty < MIN_TEST_YEAR:
                continue
            train = []
            flags = []
            for y in years:
                if y >= ty:
                    continue
                train += tourney[y]
                flags += [1.0] * len(tourney[y])
                if spec != "A":
                    reg = [r for r in regular[y] if r[2]] if spec == "E" else regular[y]
                    train += reg
                    flags += [0.0] * len(reg)
            if not train:
                continue
            fl = np.array(flags)
            Xtr, ytr = design(train, fl, spec), np.array([r[1] for r in train])
            beta = fit(Xtr, ytr, spec)
            if spec == "C":
                # Also fit it unmirrored, to show what the column picks up when
                # the symmetry is not enforced.
                lam = RIDGE_PER_1000 * (len(ytr) / 1000)
                raw = np.linalg.solve(
                    Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ ytr
                )
                flag_coefs.append((beta[-1], raw[-1]))

            test = tourney[ty]
            fte = np.ones(len(test))
            Xte = design(test, fte, spec)
            yte = np.array([r[1] for r in test])
            pred = Xte @ beta

            sse += float(((yte - pred) ** 2).sum())
            sae += float(np.abs(yte - pred).sum())
            sst += float((yte**2).sum())
            correct += int(((pred > 0) == (yte > 0)).sum())
            n += len(yte)

        results[label] = {
            "acc": 100 * correct / n,
            "rmse": (sse / n) ** 0.5,
            "mae": sae / n,
            "r2": 1 - sse / sst,
            "n": n,
        }

    base = [r for r in tourney.values() for r in []]  # noqa: F841
    seed_correct = sum(
        1 for y in years if y >= MIN_TEST_YEAR for r in tourney[y] if r[1] > 0
    )
    seed_n = sum(len(tourney[y]) for y in years if y >= MIN_TEST_YEAR)

    print(f"Scored on {seed_n} held-out TOURNAMENT games, walk-forward\n")
    print(f"{'specification':32}{'acc':>8}{'RMSE':>8}{'MAE':>8}{'R2(0)':>9}")
    print(f"{'better seed always wins':32}{100 * seed_correct / seed_n:>7.1f}%{'-':>8}{'-':>8}{'-':>9}")
    for label, r in results.items():
        print(f"{label:32}{r['acc']:>7.1f}%{r['rmse']:>8.2f}{r['mae']:>8.2f}{r['r2']:>9.3f}")

    if flag_coefs:
        mirrored = max(abs(c[0]) for c in flag_coefs)
        raw = max(abs(c[1]) for c in flag_coefs)
        print("\nThe additive is_tournament flag, across folds (max |coefficient|):")
        print(f"  under the declared symmetric fit  {mirrored:.2e}   <- exactly zero")
        print(f"  without symmetric augmentation    {raw:.2f}")
        print(
            "  The column is a property of the fixture, so it does not change sign\n"
            "  when the two teams are swapped. Mirroring makes it orthogonal to every\n"
            "  other column with a right-hand side of zero, so it cannot carry any\n"
            "  signal. The nonzero value it takes without mirroring is measuring the\n"
            "  ORIENTATION CONVENTION -- tournament rows are ordered better-seed-first,\n"
            "  regular-season rows alphabetically -- not anything about game type.\n"
            "  Specification D is the admissible way to ask the question."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
