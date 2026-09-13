"""One opponent count, not seven (audit recommendation 15).

Recommendation 15 flagged `run_experiment.py` for defaulting to 30 opponents --
a 31-person pool -- while the canonical contract is 29 opponents in a 30-person
pool, making its numbers quietly incomparable to the published headline. It was
not one module. Six declared their own, all 30, none with a recorded reason,
and the clearest symptom was an instance named `recency_fitter_3yr_pool30`
holding `n_opponents=30`: the name asserting the thing the value contradicted.

30 is a plausible value for either quantity, which is what makes this drift
invisible by inspection. So it gets a test rather than a convention: the scan
below fails when a module in the pool pipeline hardcodes an opponent count
instead of importing the contract.

This is the same failure class as audit finding H6 (`N_OPPONENTS` was 999, so
every pre-2023 season was measured in a pool 33x too large) -- just one off
instead of thirty-three, and therefore harder to notice.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.evaluation.canonical_contract import (
    CANONICAL_N_ENTRIES,
    CANONICAL_N_OPPONENTS,
    CANONICAL_POOL_SIZE,
    as_backtest_kwargs,
    describe,
    is_comparable_to_headline,
)

ROOT = Path(__file__).resolve().parents[1]

# Modules in the pool-measurement path: anything whose numbers are, or could be,
# compared against the published P(1st).
SCANNED = [
    "scripts/mc_pool_backtest.py",
    "scripts/run_experiment.py",
    "scripts/retrospective_scorer.py",
    "scripts/rank_correlation_diagnostic.py",
    "scripts/validate_opponent_model.py",
    "scripts/pool_retrospective.py",
    "scripts/pool_rdof_audit.py",
    "scripts/pool_tool_features.py",
    "scripts/independent_referee_check.py",
    "src/optimization/recency_hparam_fitter.py",
]

# An opponent count written as a literal rather than imported.
_LITERAL_OPPONENTS = re.compile(r"\bn_opponents\s*(?::\s*int\s*)?=\s*(\d+)")
_LITERAL_CONSTANT = re.compile(r"^N_OPPONENTS\s*=\s*(\d+)")

# Deliberate departures. Each needs a reason, because "it is different" and
# "it is wrong" look identical in a diff.
ALLOWED_LITERALS = {
    # src/simulation/pool_competition.py's own simulator defaults are a
    # different layer (a generic pool simulator, not this project's contract)
    # and are not scanned.
}


def test_the_contract_is_internally_consistent():
    """pool_size counts everyone; n_opponents counts the others."""
    assert CANONICAL_POOL_SIZE == CANONICAL_N_OPPONENTS + CANONICAL_N_ENTRIES
    assert CANONICAL_N_OPPONENTS == 29
    assert CANONICAL_POOL_SIZE == 30


def test_no_pool_module_hardcodes_its_own_opponent_count():
    """The recommendation-15 scan.

    A module needing a different field size should import the contract and
    depart from it visibly, so the departure reads as a departure.
    """
    offenders = []
    for rel in SCANNED:
        path = ROOT / rel
        if not path.exists():  # pragma: no cover - repo layout guard
            continue
        lines = path.read_text().splitlines()
        for lineno, text in enumerate(lines, start=1):
            # Comments routinely quote the old value while explaining why it
            # was wrong -- flagging those would make the fix un-documentable.
            if text.lstrip().startswith("#"):
                continue
            if f"{rel}:{lineno}" in ALLOWED_LITERALS:
                continue
            for pattern, label in ((_LITERAL_OPPONENTS, "n_opponents"), (_LITERAL_CONSTANT, "N_OPPONENTS")):
                match = pattern.search(text)
                if match:
                    offenders.append(f"{rel}:{lineno} {label}={match.group(1)}")

    assert not offenders, (
        "opponent count hardcoded instead of imported from "
        "src/evaluation/canonical_contract.py. 30 is a plausible value for both "
        "`n_opponents` and `pool_size`, so a literal here is drift nobody will spot by "
        f"reading. Offenders: {offenders}"
    )


def test_the_canonical_constant_is_the_one_the_backtest_uses():
    """mc_pool_backtest.N_OPPONENTS must BE the contract, not merely equal it."""
    from scripts.mc_pool_backtest import N_OPPONENTS

    assert N_OPPONENTS is CANONICAL_N_OPPONENTS


def test_the_recency_fitter_is_on_the_contract():
    """The instance whose name promised a 30-person pool and delivered 31."""
    from src.optimization import recency_hparam_fitter as mod

    assert not hasattr(mod, "recency_fitter_3yr_pool30"), (
        "recency_fitter_3yr_pool30 is back. Its name says pool 30 and n_opponents=30 means "
        "pool 31; it was renamed to recency_fitter_3yr_canonical for that reason."
    )
    assert mod.recency_fitter_3yr_canonical.n_opponents == CANONICAL_N_OPPONENTS


def test_as_backtest_kwargs_matches_the_published_contract():
    kwargs = as_backtest_kwargs()
    assert kwargs["n_opponents"] == 29
    assert kwargs["n_repeats"] == 100
    assert kwargs["opponent_source"] == "pool"
    assert kwargs["team_identity"] is True
    assert kwargs["pa_trials"] == 500


def test_as_backtest_kwargs_lets_a_departure_be_explicit():
    kwargs = as_backtest_kwargs(n_opponents=999)
    assert kwargs["n_opponents"] == 999
    assert kwargs["n_repeats"] == 100, "an override must not disturb the rest of the contract"


@pytest.mark.parametrize(
    "n_opponents,n_entries,expected",
    [(29, 1, True), (30, 1, False), (29, 2, False), (999, 1, False)],
)
def test_comparability_check_rejects_the_off_by_one(n_opponents, n_entries, expected):
    """A 31-person pool is not the published quantity, however close it looks."""
    assert is_comparable_to_headline(n_opponents, n_entries) is expected


def test_describe_states_the_pool_size_not_just_the_opponent_count():
    """Headers must name the number a reader would otherwise have to derive."""
    text = describe()
    assert "30-person pool" in text
    assert "29 opponents" in text
