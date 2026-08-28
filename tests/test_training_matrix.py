"""Guards for the browser's training matrix, docs/data/training.json.

The matrix is one row per bracket game: a standardised differential `x`, the
final margin `m`, and the binary result `w`. The UI fits a zero-intercept
spread regression on it live.

WHAT CAN GO WRONG HERE
The source records (`tournament_context_*.json`) are stored WINNER-FIRST: the
championship game is 100% team1-won, and the First Four is 95.5% team1-won
despite being same-seed matchups where seeding explains nothing. Copying that
order into the matrix writes the answer into the row layout.

That is survivable for some consumers and not for others. A zero-intercept
antisymmetric model is invariant to orientation -- flipping a row negates both
`x` and `m`, leaving the normal equations unchanged -- so coefficients, RMSE,
MAE and accuracy are all unaffected. Anything measured ABOUT THE MEAN of the
target is not: winner-first rows have a mean margin near +9.5, so a
variance-explained figure computed against that mean is scored against a
baseline ("the first-listed team wins by 9.5") that is only available to
someone who already knows the result.

These tests pin the orientation to pre-tournament facts so that neither kind of
consumer has to know about the hazard.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

MATRIX = Path("docs/data/training.json")

# Better seeds win roughly 71-72% of bracket games. Winner-first ordering shows
# up as ~87-90%. The gap between those is far larger than year-to-year noise,
# so a loose band is enough to catch a regression without being brittle.
SEED_BASE_RATE = (0.66, 0.78)


def _matrix() -> dict:
    if not MATRIX.exists():
        pytest.skip(f"{MATRIX} not built")
    return json.loads(MATRIX.read_text())


def test_rows_carry_margin_and_result():
    d = _matrix()
    assert d["games"], "no games"
    for r in d["games"]:
        assert set(r) == {"y", "x", "w", "m"}, f"unexpected row shape: {sorted(r)}"
        assert len(r["x"]) == len(d["keys"])


def test_margin_and_result_agree():
    """`w` must be exactly the sign of `m`, or the two targets disagree.

    They are not independent facts -- `w` is a coarsening of `m` -- so any
    disagreement means one of them was written from the wrong side of a swap.
    """
    d = _matrix()
    for r in d["games"]:
        assert r["m"] != 0, "a tied game cannot be labelled with a winner"
        assert (r["m"] > 0) == (r["w"] == 1), f"sign(m)={r['m']} disagrees with w={r['w']}"


def test_rows_are_not_oriented_by_the_result():
    """The headline guard: row order must not encode who won.

    If it did, `w` would be near 1.0 rather than near the rate at which the
    better seed actually wins.
    """
    d = _matrix()
    rate = sum(r["w"] for r in d["games"]) / len(d["games"])
    lo, hi = SEED_BASE_RATE
    assert lo <= rate <= hi, (
        f"P(w=1) = {rate:.3f}, outside the {lo}-{hi} band expected when rows are "
        "oriented by seed. A value near 0.87-0.90 means the matrix inherited the "
        "source records' winner-first ordering."
    )


def test_no_single_season_is_oriented_by_the_result():
    """A pooled rate can hide a season that is entirely winner-first.

    The source data is winner-first for whole seasons at a time (every game of
    2005-2015 has team1_won=true), so the pooled check above could pass while
    one season is fully leaked.
    """
    d = _matrix()
    by_year: dict[int, list[int]] = {}
    for r in d["games"]:
        by_year.setdefault(r["y"], []).append(r["w"])
    for year, ws in sorted(by_year.items()):
        rate = sum(ws) / len(ws)
        assert rate < 0.95, (
            f"{year}: {rate:.1%} of rows have team1 winning — that season looks winner-first rather than seed-oriented."
        )


def test_orientation_is_invariant_for_a_zero_intercept_fit():
    """Flipping rows must not move the coefficients, RMSE, MAE or accuracy.

    This is the property that lets the browser skip symmetric augmentation, and
    it is why the winner-first ordering was harmless to the old logistic model
    even though it was wrong. Asserted rather than assumed, because everything
    downstream leans on it.
    """
    numpy = pytest.importorskip("numpy")
    d = _matrix()
    X = numpy.array([r["x"] for r in d["games"]], dtype=float)
    m = numpy.array([r["m"] for r in d["games"]], dtype=float)

    # Flip a deterministic half of the rows, exactly as a different orientation
    # convention would have written them.
    flip = numpy.array([i % 2 == 0 for i in range(len(m))])
    Xf, mf = X.copy(), m.copy()
    Xf[flip] *= -1
    mf[flip] *= -1

    def fit(A, y):
        lam = 1.0 * (len(y) / 1000)
        return numpy.linalg.solve(A.T @ A + lam * numpy.eye(A.shape[1]), A.T @ y)

    b1, b2 = fit(X, m), fit(Xf, mf)
    assert numpy.allclose(b1, b2, atol=1e-9), "coefficients moved under reorientation"

    r1 = numpy.sqrt(((m - X @ b1) ** 2).mean())
    r2 = numpy.sqrt(((mf - Xf @ b2) ** 2).mean())
    assert r1 == pytest.approx(r2, abs=1e-9), "RMSE moved under reorientation"

    a1 = ((X @ b1 > 0) == (m > 0)).mean()
    a2 = ((Xf @ b2 > 0) == (mf > 0)).mean()
    assert a1 == pytest.approx(a2, abs=1e-12), "accuracy moved under reorientation"


def test_mean_dependent_statistics_are_the_reason_orientation_matters():
    """The converse: show that a mean-centred figure DOES move.

    Without this the invariance test above reads as "orientation never matters",
    which is the wrong lesson and the one that let winner-first ordering survive.
    """
    numpy = pytest.importorskip("numpy")
    d = _matrix()
    m = numpy.array([r["m"] for r in d["games"]], dtype=float)

    mf = m.copy()
    mf[numpy.array([i % 2 == 0 for i in range(len(m))])] *= -1

    assert abs(float(m.mean())) > 3, "seed-oriented margins should have a clear positive mean"
    assert abs(float(mf.mean())) < abs(float(m.mean())), (
        "reorientation must change the mean of the target — that is precisely the "
        "quantity a mean-centred R2 would be scored against"
    )


# Variables that must not reach the model or the menu. Two different reasons
# are bundled here, and the distinction matters if either is ever revisited.
#
# outcome_* and hist_residual are RESULTS. They can never be inputs.
#
# The roster shares were originally here as CONTAMINATED: every
# cbbpy_rosters_*.json was scraped after the fact, so per-player minute averages
# were computed over a game count that grows with how far a team advanced --
# r = +0.71 to +0.96 between "extra games on the roster" and "rounds actually
# won". That is no longer true as of 2026-08-27: build_roster_minutes weights by
# pre-tournament box-score minutes, so the window no longer straddles the
# prediction point.
#
# They stay excluded on a DIFFERENT and now-measured ground: they are null.
# Added to the fit, walk-forward warm n=630, log loss 0.45296 -> 0.45015 with a
# paired bootstrap of [-0.00469, +0.00681], straddling zero. A variable that
# contributes nothing but appears in the menu is worse than an absent one,
# because a reader reasonably infers the model accounts for it.
CONTAMINATED = ("returning_minutes_pct", "freshman_minutes_pct")


def test_post_hoc_variables_are_absent_from_the_matrix():
    d = _matrix()
    for key in CONTAMINATED:
        assert key not in d["keys"], (
            f"{key} is back in the training matrix. Its minute weights include "
            "that season's tournament games for every year before 2026."
        )
    # hist_residual was on this list and was deliberately removed from it in
    # 94aa15d. It is NOT a result in the sense the others are: it averages a
    # team's outcome_vs_seed_delta over its appearances STRICTLY BEFORE the
    # season being predicted, and generate_team_stats_table reads that history
    # before folding the current year in. The tell that it is backward-looking
    # is that it is null for 255 of 1,085 team-seasons -- every team's first
    # appearance, which has no prior history to average.
    #
    # The genuine results stay: they describe the very tournament being
    # predicted and can never be inputs.
    for key in ("outcome_rounds_won", "outcome_vs_seed_delta"):
        assert key not in d["keys"], f"{key} is a result, not a pre-tournament property"


def test_post_hoc_variables_are_absent_from_the_menu():
    """The matrix and the menu are built separately and can drift apart."""
    seasons = Path("docs/data/seasons.json")
    if not seasons.exists():
        pytest.skip("seasons.json not built")
    excluded = set(json.loads(seasons.read_text())["variables_excluded_as_leakage"])
    for key in CONTAMINATED:
        assert key in excluded, f"{key} is no longer declared as excluded"

    for payload in sorted(Path("docs/data").glob("season_*.json")):
        data = json.loads(payload.read_text())
        offered = {v["key"] for v in data.get("variables", [])}
        leaked = offered & excluded
        assert not leaked, f"{payload.name} offers excluded variables: {sorted(leaked)}"


def test_every_season_has_a_full_bracket():
    d = _matrix()
    for year, n in d["per_year"].items():
        assert n == 63, f"{year}: {n} games, expected the 63-game bracket"


def test_fitting_contract_is_declared():
    """The payload must state the contract the browser relies on."""
    d = _matrix()
    c = d["fitting_contract"]
    assert c["intercept"] == 0
    assert c["mirror_rows"] is True
    assert c["leave_one_year_out"] is True
    assert d["target"] == "m"


def test_margins_are_plausible_basketball_scores():
    d = _matrix()
    m = [r["m"] for r in d["games"]]
    assert max(abs(v) for v in m) < 80, "a bracket game is not decided by 80 points"
    assert 8 < statistics.pstdev(m) < 20, "margin spread is implausible for tournament games"
