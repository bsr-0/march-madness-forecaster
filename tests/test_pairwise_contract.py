"""Regression tests for the pairwise probability contract.

The contract (see ``src/prediction/pairwise.py``):

    LEGAL:    pairwise -> simulator -> outcomes -> marginals
    ILLEGAL:  marginals -> "pairwise"

These tests exist to make the illegal direction impossible to reintroduce.
There are three layers:

1. :func:`test_marginal_ratio_is_invalid` — a numerical proof that the old
   ``p1 / (p1 + p2)`` reconstruction is wrong, and by how much. This is the
   evidence the whole contract rests on; it is pinned so nobody has to
   re-derive it.
2. :func:`test_pairwise_roundtrip_is_exact` — the corrected path recovers the
   generating probabilities to within Monte Carlo error.
3. :func:`test_no_marginal_to_pairwise_reconstruction` — a static AST scan of
   ``src/`` and ``scripts/`` that fails when new code divides one round-probability
   lookup by the sum of two.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from src.prediction.pairwise import (
    MissingPairwiseSource,
    PairwiseProbabilities,
    ProbabilityBase,
    log5,
    marginals_from_pairwise,
    simulate_bracket_outcomes,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REGION_ORDER = ["East", "West", "South", "Midwest"]
SEED_MATCHUP_ORDER = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
ROUND_NAMES = ("R64", "R32", "S16", "E8", "F4", "CHAMP")


def _synthetic_bracket():
    """A full 64-team bracket with the project's seed->barthag fallback ladder.

    Matches ``mc_pool_backtest._load_torvik_barthag``'s fallback so the numbers
    here correspond to a realistic strength spread.
    """
    ratings, seeds, order = {}, {}, []
    for region in REGION_ORDER:
        for seed in range(1, 17):
            tid = f"{region}_{seed}"
            ratings[tid] = max(0.10, 1.0 - seed * 0.04)
            seeds[tid] = seed
    for region in REGION_ORDER:
        for high, low in SEED_MATCHUP_ORDER:
            order.extend([f"{region}_{high}", f"{region}_{low}"])
    return ratings, seeds, order


# ---------------------------------------------------------------------------
# 1. The defect, pinned
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_marginal_ratio_is_invalid():
    """``p1/(p1+p2)`` over marginals is exact in R64 and badly wrong after.

    Self-consistency test: generate marginals *from* a known pairwise table by
    simulation, then try to recover the pairwise table by normalizing pairs of
    marginals. A valid conversion would return the inputs. It does not.

    The error is one-directional — the signed error (oriented toward the
    stronger team) equals the absolute error at every round, meaning every
    single matchup is distorted toward the favorite. That is why this defect
    suppressed exactly the upset/Cinderella variation the bracket product
    needs.

    Originally measured in ``/tmp/audit_norm_bias.py`` during the 2026-08-19
    architecture audit; moved here so the analysis cannot be lost.
    """
    ratings, _seeds, order = _synthetic_bracket()
    pw = PairwiseProbabilities.from_ratings(ratings, "log5(synthetic)")

    n_sims = 60_000
    rng = np.random.default_rng(42)

    # Track marginals AND which pairs actually met in each round.
    counts = {t: {r: 0 for r in ROUND_NAMES} for t in ratings}
    met: dict = {}
    for _ in range(n_sims):
        current = list(order)
        for round_idx, rname in enumerate(ROUND_NAMES):
            nxt = []
            for g in range(0, len(current), 2):
                t1, t2 = current[g], current[g + 1]
                met[(round_idx, *sorted((t1, t2)))] = met.get((round_idx, *sorted((t1, t2))), 0) + 1
                winner = t1 if rng.random() < pw.p(t1, t2) else t2
                counts[winner][rname] += 1
                nxt.append(winner)
            current = nxt

    rp = {t: {r: max(1e-6, counts[t][r] / n_sims) for r in ROUND_NAMES} for t in ratings}

    # Expected mean-absolute error per round, from the audit. Generous
    # tolerances — the point is the shape (near-zero in R64, an order of
    # magnitude larger from R32 on), not the third decimal.
    expected = {
        "R64": (0.000, 0.010),
        "R32": (0.045, 0.100),
        "S16": (0.080, 0.150),
        "E8": (0.080, 0.155),
        "F4": (0.090, 0.170),
        "CHAMP": (0.090, 0.180),
    }

    for round_idx, rname in enumerate(ROUND_NAMES):
        errs, signed = [], []
        for key, n in met.items():
            if key[0] != round_idx or n < 200:
                continue
            a, b = key[1], key[2]
            truth = log5(ratings[a], ratings[b])
            p1, p2 = rp[a][rname], rp[b][rname]
            approx = p1 / (p1 + p2) if p1 + p2 > 1e-9 else 0.5
            errs.append(abs(approx - truth))
            signed.append((approx - truth) if ratings[a] >= ratings[b] else (truth - approx))

        assert errs, f"no sampled matchups for {rname}"
        mean_abs = float(np.mean(errs))
        lo, hi = expected[rname]
        assert lo <= mean_abs <= hi, f"{rname}: mean|err|={mean_abs:.4f} outside [{lo}, {hi}]"

        # The bias is toward the favorite: signed error ~= absolute error.
        mean_signed = float(np.mean(signed))
        if rname != "R64":
            assert mean_signed > 0.9 * mean_abs, (
                f"{rname}: expected a one-directional chalk bias, got signed={mean_signed:.4f} vs abs={mean_abs:.4f}"
            )


def test_marginal_ratio_is_exact_in_r64_only():
    """Cheap structural version of the above: p1+p2 == 1 in R64, < 1 after.

    This is the whole mechanism in two assertions, and it runs in a second.
    """
    ratings, _seeds, order = _synthetic_bracket()
    pw = PairwiseProbabilities.from_ratings(ratings, "log5(synthetic)")
    rp = marginals_from_pairwise(pw, order, ratings.keys(), n_sims=4000, seed=7)

    # R64: the two teams in a game are the only ones who can win it.
    for g in range(0, len(order), 2):
        t1, t2 = order[g], order[g + 1]
        assert rp[t1]["R64"] + rp[t2]["R64"] == pytest.approx(1.0, abs=1e-9)

    # R32: four teams contest each slot, so any two of them sum to < 1.
    for g in range(0, len(order), 4):
        t1, t2 = order[g], order[g + 2]
        assert rp[t1]["R32"] + rp[t2]["R32"] < 0.995


# ---------------------------------------------------------------------------
# 2. The corrected path
# ---------------------------------------------------------------------------


def test_pairwise_roundtrip_is_exact():
    """Simulating from a pairwise table reproduces that table's win rates.

    This is the acceptance gate the audit called A9: the corrected path must
    recover its inputs to within Monte Carlo error at *every* round, versus the
    0.07-0.14 error of the marginal-ratio reconstruction.
    """
    ratings, _seeds, order = _synthetic_bracket()
    pw = PairwiseProbabilities.from_ratings(ratings, "log5(synthetic)")

    n_sims = 40_000
    rng = np.random.default_rng(11)
    _outcomes, by_round = simulate_bracket_outcomes(pw, order, n_sims, rng)

    # Empirical head-to-head rate for every pair that actually met.
    met: dict = {}
    won: dict = {}
    for sim in range(n_sims):
        current = list(order)
        for round_idx in range(6):
            winners = by_round[sim][round_idx]
            for gi, g in enumerate(range(0, len(current), 2)):
                t1, t2 = current[g], current[g + 1]
                key = (round_idx, *sorted((t1, t2)))
                met[key] = met.get(key, 0) + 1
                if winners[gi] == sorted((t1, t2))[0]:
                    won[key] = won.get(key, 0) + 1
            current = winners

    errs = []
    for key, n in met.items():
        if n < 400:
            continue
        a, b = key[1], key[2]
        empirical = won.get(key, 0) / n
        errs.append(abs(empirical - log5(ratings[a], ratings[b])))

    assert errs
    mean_abs = float(np.mean(errs))
    assert mean_abs < 0.01, f"pairwise roundtrip should be exact, got mean|err|={mean_abs:.4f}"


def test_marginals_are_normalized_per_round():
    """Sum of marginals across all teams equals the number of winners per round."""
    ratings, _seeds, order = _synthetic_bracket()
    pw = PairwiseProbabilities.from_ratings(ratings, "log5(synthetic)")
    rp = marginals_from_pairwise(pw, order, ratings.keys(), n_sims=3000, seed=3)

    for rname, n_games in zip(ROUND_NAMES, (32, 16, 8, 4, 2, 1)):
        total = sum(rp[t][rname] for t in ratings)
        assert total == pytest.approx(n_games, abs=0.05 * n_games + 0.05)


# ---------------------------------------------------------------------------
# 3. Type invariants and the loud-failure guard
# ---------------------------------------------------------------------------


def test_pairwise_validates_antisymmetry():
    pw = PairwiseProbabilities.from_dict({("a", "b"): 0.7, ("b", "a"): 0.3}, "ok")
    pw.validate()

    with pytest.raises(ValueError, match="antisymmetric"):
        PairwiseProbabilities.from_dict({("a", "b"): 0.7, ("b", "a"): 0.7}, "bad")

    with pytest.raises(ValueError, match="out of range"):
        PairwiseProbabilities.from_dict({("a", "b"): 1.4}, "bad")


def test_pairwise_infers_reverse_pair():
    pw = PairwiseProbabilities.from_dict({("a", "b"): 0.75}, "one-way", validate=False)
    assert pw.p("b", "a") == pytest.approx(0.25)
    assert pw.p("x", "y") == 0.5  # unknown pair -> default


def test_log5_is_symmetric_and_monotone():
    assert log5(0.9, 0.9) == pytest.approx(0.5)
    assert log5(0.9, 0.5) + log5(0.5, 0.9) == pytest.approx(1.0)
    assert log5(0.9, 0.3) > log5(0.7, 0.3) > log5(0.5, 0.3)


def test_probability_base_is_a_mapping_over_marginals():
    """Legacy round_probs read paths must keep working unchanged."""
    pw = PairwiseProbabilities.from_ratings({"a": 0.9, "b": 0.4}, "t")
    base = ProbabilityBase("t", {"a": {"F4": 0.3}, "b": {"F4": 0.1}}, pw)

    assert base["a"]["F4"] == 0.3
    assert base.get("a", {}).get("F4") == 0.3
    assert base.get("zz", {}) == {}
    assert set(base.keys()) == {"a", "b"}
    assert sorted(base) == ["a", "b"]
    assert dict(base) == {"a": {"F4": 0.3}, "b": {"F4": 0.1}}
    assert len(base) == 2


def test_base_without_pairwise_fails_loudly():
    """A base with no head-to-head model must raise, never fabricate."""
    base = ProbabilityBase("contrarian", {"a": {"F4": 0.3}}, None)
    assert base.has_pairwise is False
    with pytest.raises(MissingPairwiseSource, match="no pairwise source"):
        _ = base.pairwise


def test_samplers_reject_bare_round_probs():
    """The stochastic samplers must refuse a plain marginal mapping."""
    from scripts.mc_pool_backtest import sample_model_brackets

    ratings, _seeds, order = _synthetic_bracket()
    bare = {t: {r: 0.5 for r in ROUND_NAMES} for t in ratings}

    with pytest.raises(MissingPairwiseSource):
        sample_model_brackets(order, bare, 2, np.random.default_rng(0))

    # ... and accept a properly-formed base.
    pw = PairwiseProbabilities.from_ratings(ratings, "log5(synthetic)")
    base = ProbabilityBase("torvik", bare, pw)
    out = sample_model_brackets(order, base, 3, np.random.default_rng(0))
    assert out.shape == (3, 63)


def test_simulated_annealing_rejects_bare_round_probs():
    """SA needs real head-to-head probabilities, not round-collapsed marginals."""
    from src.optimization.bracket_construction import construct_bracket

    ratings, seeds, _order = _synthetic_bracket()
    regions = {t: t.split("_")[0] for t in ratings}
    bare = {t: {r: 0.5 for r in ROUND_NAMES} for t in ratings}

    with pytest.raises(MissingPairwiseSource):
        construct_bracket(
            mode="simulated_annealing",
            seeds=seeds,
            regions=regions,
            round_probs=bare,
            public_picks={},
            risk_level=0.5,
        )


# ---------------------------------------------------------------------------
# 4. Static scan — stop it coming back
# ---------------------------------------------------------------------------

# Call sites that match the pattern but are NOT the defect. Each entry must
# carry a justification; adding one is a deliberate act, not a rubber stamp.
_ALLOWLIST = {
    # Public *ownership* shares, not model probabilities. Normalizing two pick
    # percentages answers "which team does a typical bracket advance here",
    # which is genuinely a ratio of two shares of the same population.
    ("src/simulation/pool_competition.py", "_get_pick_prob"),
    # Same quantity (SEED_PICK_RATES ownership), isolated into its own
    # function precisely so this allowlist entry is narrow.
    ("scripts/divergence_diagnostic.py", "field_pick_share"),
    # KNOWN DEFECT, DELIBERATELY NOT FIXED HERE (2026-08-19).
    # This is the bracket export that feeds the web UI's displayed win_prob.
    # It is the same invalid reconstruction, but changing it changes numbers
    # rendered to users, and the current work is explicitly scoped to leave
    # the UI alone. Tracked in ARCHITECTURE_AUDIT_PREFERENCE_BRACKETS.md §3.
    ("scripts/_bracket_export_common.py", "build_bracket_json"),
    # KNOWN INVALID AS A PROBABILITY, RETAINED AS A GBM FEATURE (2026-08-19).
    # Consumed by _game_features as one input among many, computed identically
    # across every probability base so the model sees base *disagreement*.
    # Changing it retrains every meta_gbm* mode. See the function's docstring.
    ("src/prediction/meta_selector.py", "_pairwise_prob"),
}


def _round_key_names(tree: ast.AST) -> set:
    """Variables that look like they hold a per-round probability lookup.

    Matches assignments of the shape ``x = m.get(team, {}).get(round_name, d)``
    or ``x = m[team][round_name]`` where the final key is a round name or a
    variable whose name mentions ``round``.
    """
    round_literals = set(ROUND_NAMES)
    found = set()

    def _final_key(node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            if node.args:
                return node.args[0]
        if isinstance(node, ast.Subscript):
            return node.slice
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        key = _final_key(node.value)
        if key is None:
            continue
        if isinstance(key, ast.Constant) and key.value in round_literals:
            found.add(target.id)
        elif isinstance(key, ast.Name) and "round" in key.id.lower():
            found.add(target.id)
    return found


def test_no_marginal_to_pairwise_reconstruction():
    """Fail if any code divides a round-probability by the sum of two.

    This is the exact textual signature of the defect:

        p1 = round_probs.get(t1, {}).get(round_name, 0.0)
        p2 = round_probs.get(t2, {}).get(round_name, 0.0)
        p_t1 = p1 / (p1 + p2)          # <-- invalid from R32 onward
    """
    violations = []

    for path in sorted(list((PROJECT_ROOT / "src").rglob("*.py")) + list((PROJECT_ROOT / "scripts").rglob("*.py"))):
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue

        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if (rel, func.name) in _ALLOWLIST:
                continue

            marginal_vars = _round_key_names(func)
            if not marginal_vars:
                continue

            # Variables assigned from a sum of marginal-derived values, e.g.
            # ``total = p1 + p2``. Dividing by one of these is the same defect
            # written across two statements.
            sum_vars = set()
            for node in ast.walk(func):
                if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
                    continue
                target = node.targets[0]
                if not isinstance(target, ast.Name):
                    continue
                val = node.value
                if isinstance(val, ast.BinOp) and isinstance(val.op, ast.Add):
                    if {n.id for n in ast.walk(val) if isinstance(n, ast.Name)} & marginal_vars:
                        sum_vars.add(target.id)

            for node in ast.walk(func):
                if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)):
                    continue
                denom = node.right
                if isinstance(denom, ast.Call) and getattr(denom.func, "id", None) in ("max", "min"):
                    denom = denom.args[0] if denom.args else denom

                flagged = False
                if isinstance(denom, ast.BinOp) and isinstance(denom.op, ast.Add):
                    names = {n.id for n in ast.walk(denom) if isinstance(n, ast.Name)}
                    flagged = bool(names & marginal_vars)
                elif isinstance(denom, ast.Name) and denom.id in sum_vars:
                    # ``p1 / total`` where ``total = p1 + p2``
                    flagged = True

                if flagged:
                    violations.append(f"{rel}:{node.lineno} in {func.name}(): {ast.unparse(node)}")

    assert not violations, (
        "marginal advancement probabilities are being converted into pairwise "
        "matchup probabilities. This is invalid from R32 onward (see "
        "src/prediction/pairwise.py). Use the base's PairwiseProbabilities "
        "instead, or add a justified entry to _ALLOWLIST.\n  " + "\n  ".join(violations)
    )
