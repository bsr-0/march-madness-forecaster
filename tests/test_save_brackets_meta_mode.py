"""``--save-brackets`` must not crash on meta modes.

``meta_region_poolaware`` (and other meta modes) build exactly one
deterministic bracket per year — ``model_brackets.shape[0] == 1`` — while
stochastic modes build ``n_model`` (default 50). The save-brackets
serialization loop in ``_run_one_year`` used to iterate ``range(n_model)``
unconditionally and index ``model_brackets[m]``, which raised
``IndexError`` for any meta mode as soon as m reached 1. Found running the
real-outcome placement analysis (scripts/real_pool_placement.py) for
2023-2026 with meta_region_poolaware, which is exactly the mode most
likely to be run with --save-brackets and least likely to be covered by a
default (stochastic-mode) smoke test.

A full call to `_run_one_year` needs a season's worth of data fixtures, so
this is a static source check rather than an integration test — it pins
the invariant "the save-brackets loop bound must come from the bracket
array, not the outer n_model" directly against a regression.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts" / "mc_pool_backtest.py").read_text()


def _save_brackets_loop_body():
    marker = "# Serialize pick-level brackets when --save-brackets is active."
    start = SOURCE.index(marker)
    end = SOURCE.index("return {", start)
    return SOURCE[start:end]


def test_save_brackets_loop_bound_is_not_the_stochastic_n_model():
    body = _save_brackets_loop_body()
    assert "range(n_model)" not in body, (
        "save-brackets serialization must not use the outer n_model as its loop "
        "bound — it is 1 for meta modes and crashes indexing model_brackets[1]"
    )


def test_save_brackets_loop_bound_matches_the_actual_bracket_count():
    body = _save_brackets_loop_body()
    assert re.search(r"for m in range\(model_brackets\.shape\[0\]\)", body), (
        "expected the save-brackets loop to iterate range(model_brackets.shape[0]), "
        "matching how ti_scores/score_brackets_team_identity are already sized"
    )
