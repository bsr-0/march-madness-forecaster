"""Canonical, outcome-independent orientation for tournament game records.

**The problem this solves.** In ``tournament_context_{year}.json`` →
``results.games[]`` the two teams are frequently stored *winner-first*: every
one of the 63 games in each of 2005-2015 has ``team1_won == true``, and 90.2%
of all 1,323 games across 2005-2026 do. The records themselves are not
corrupt — ``team1_won`` agrees with ``team1_score > team2_score`` in all 21
years, with no missing scores — but the *ordering* correlates with the result.

That is a silent trap for anything that treats ``(team1, team2)`` as an
outcome-independent ordering and scores ``P(team1 wins)`` against
``outcome``. Such a consumer is graded against a ~90% base rate instead of
the true ~72%.

**What is and isn't affected.** Brier score, log loss and BSS are invariant
under flipping both the label and the probability, so raw-model Brier figures
computed from these records were never wrong. Accuracy, calibration/ECE, and
— most consequentially — anything that *learns from residuals* are not
invariant. Fitting a correction or calibration layer on the stored
orientation teaches it that the mean residual is +0.238 when the true value
is -0.0005, i.e. that every probability should be shifted up by ~24 points.

**The fix.** Re-orient every record so team1 is the better seed, with an
alphabetical tiebreak on equal seeds — a rule that depends only on inputs,
never on the result. After orientation the favourite win rate lands at 72.0%,
matching the published historical rate.

Orientation is applied at the *evaluation/training* boundary rather than by
rewriting the ground-truth files, because the pool backtest and bracket
scorers read those files for winner *identity*, where ordering is irrelevant
and a rewrite would be needless churn.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

# Probability fields in a per-game prediction record. Each is P(team1 wins),
# so each must be inverted whenever the pair is flipped.
PROBABILITY_FIELDS = (
    "torvik",
    "seed",
    "elo",
    "massey_avg",
    "knn",
    "closing_market",
    "opening_market",
    "pipeline",
)


def should_flip(seed1: Any, seed2: Any, team1: str, team2: str) -> bool:
    """True when a game is stored with the worse seed first.

    Depends only on seeds and team ids — never on the outcome — so applying
    it cannot leak the result. Equal seeds (1-vs-1 Final Fours) fall back to
    an alphabetical tiebreak so the choice is deterministic and arbitrary
    with respect to who won.
    """
    try:
        s1, s2 = int(seed1), int(seed2)
    except (TypeError, ValueError):
        s1 = s2 = 0
    return (s1, str(team1)) > (s2, str(team2))


def orient_result_game(game: Dict) -> Dict:
    """Return a ``results.games[]`` record oriented better-seed-first.

    Swaps the ``team1_*``/``team2_*`` field pairs and inverts ``team1_won``.
    The input is not mutated.
    """
    if not should_flip(
        game.get("team1_seed", 8),
        game.get("team2_seed", 8),
        game.get("team1_id", ""),
        game.get("team2_id", ""),
    ):
        return dict(game)

    out = dict(game)
    for a, b in (
        ("team1_id", "team2_id"),
        ("team1_seed", "team2_seed"),
        ("team1_score", "team2_score"),
    ):
        if a in game or b in game:
            out[a], out[b] = game.get(b), game.get(a)
    if "team1_won" in game:
        out["team1_won"] = not bool(game["team1_won"])
    return out


def orient_prediction_record(record: Dict, probability_fields: Iterable[str] = PROBABILITY_FIELDS) -> Dict:
    """Return a per-game *prediction* record oriented better-seed-first.

    Swaps ``team1``/``team2`` and ``seed1``/``seed2``, inverts ``outcome``,
    and inverts every probability field present (each is P(team1 wins)).
    The input is not mutated.
    """
    if not should_flip(
        record.get("seed1", 8),
        record.get("seed2", 8),
        record.get("team1", ""),
        record.get("team2", ""),
    ):
        return dict(record)

    out = dict(record)
    out["team1"], out["team2"] = record.get("team2"), record.get("team1")
    out["seed1"], out["seed2"] = record.get("seed2"), record.get("seed1")
    if "outcome" in record:
        out["outcome"] = 1 - int(record["outcome"])
    for field in probability_fields:
        value = record.get(field)
        if value is not None:
            out[field] = 1.0 - float(value)
    if "market_movement" in record and record["market_movement"] is not None:
        # A closing-minus-opening delta flips sign, not complement.
        out["market_movement"] = -float(record["market_movement"])
    return out


def orient_result_games(games: Iterable[Dict]) -> List[Dict]:
    return [orient_result_game(g) for g in games]


def orient_prediction_records(records: Iterable[Dict]) -> List[Dict]:
    return [orient_prediction_record(r) for r in records]


def favorite_won(record: Dict) -> Optional[bool]:
    """Whether the better seed won, for an already-oriented prediction record.

    Returns None for same-seed matchups, where there is no favourite.
    """
    try:
        if int(record["seed1"]) == int(record["seed2"]):
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return int(record.get("outcome", 0)) == 1
