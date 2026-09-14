"""Pins for the referee robustness audit (src/evaluation/referee_audit.py).

What is worth pinning: the vectorised scorer must equal the production
scorer bit-for-bit, the rank convention must equal the harness's, the
tie-break must be production's strict-greater-first, the criteria must be
the pre-registered ones (numbers in the markdown and in CRITERIA agree), and
the criteria logic must produce each of its three statuses on inputs built
to trigger them.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from src.evaluation import referee_audit as ra
from src.simulation.pool_competition import score_brackets_team_identity

ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "artifacts" / "referee_audit" / "PREREGISTRATION.md"


def _fake_field(n=64):
    return [f"t{i}" for i in range(n)]


def _random_winners(first_round, rng):
    current = list(first_round)
    winners = {}
    for rn in ra.ROUND_NAMES:
        nxt = [current[g] if rng.random() < 0.5 else current[g + 1] for g in range(0, len(current), 2)]
        winners[rn] = set(nxt)
        current = nxt
    return winners


def test_vectorised_scorer_matches_production_scorer():
    rng = np.random.default_rng(0)
    first_round = _fake_field()
    team_index = {t: i for i, t in enumerate(first_round)}
    brackets = rng.random((40, 63)) < 0.5
    points = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
    membership = ra.decode_membership(brackets, first_round, team_index)
    for _ in range(5):
        winners = _random_winners(first_round, rng)
        expected = score_brackets_team_identity(brackets, winners, first_round, points)
        got = ra.score_membership(membership, ra.winners_vector(winners, team_index), [points[r] for r in ra.ROUND_NAMES])
        np.testing.assert_array_equal(got, expected)


def test_rank_convention_matches_harness():
    rng = np.random.default_rng(1)
    opp = rng.integers(0, 200, 29).astype(float) * 10
    ours = rng.integers(0, 200, 7).astype(float) * 10
    got = ra.ranks_against(opp, ours)
    for i, s in enumerate(ours):
        better = np.sum(opp > s)
        tied = np.sum(opp == s)
        assert got[i] == better + 1 + tied / 2.0


def test_argmax_first_is_strict_greater_tie_break():
    assert ra.argmax_first([0.1, 0.3, 0.3, 0.2]) == 1
    assert ra.argmax_first([0.0, 0.0]) == 0


def test_loro_choices_exclude_the_held_out_referee():
    p1 = {
        "seed": np.array([0.9, 0.1, 0.1]),
        "torvik": np.array([0.1, 0.9, 0.1]),
        "blend": np.array([0.1, 0.9, 0.1]),
        "pit": np.array([0.1, 0.1, 0.9]),
        "market": np.array([0.1, 0.1, 0.9]),
    }
    c = ra.loro_choices(p1, list(p1))
    assert c["seed"]["production"] == 0  # production is always the seed argmax
    assert c["seed"]["self"] == 0
    # Holding seed out, candidates 1 and 2 tie on the mean; strict-first keeps 1.
    assert c["seed"]["loro"] == 1
    assert c["pit"]["loro"] == 1  # seed+torvik+blend+market -> index 1 (0.3) vs index 2 (0.3)... tie -> 1
    assert c["market"]["self"] == 2


def test_expected_labels_parse_from_canonical_log_snippet(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text(
        "  2011   meta_region_poolaware    selected=blend_region_risk=0.5 (best of 20, P1=0.056)\n"
        "  2011   meta_region_poolaware META bracket built (1 deterministic)\n"
        "  2012   meta_region_poolaware    selected=tv_champ=syracuse (best of 22, P1=0.058)\n"
    )
    assert ra.expected_labels_from_log(log) == {2011: "blend_region_risk=0.5", 2012: "tv_champ=syracuse"}


def test_parity_check_semantics():
    assert ra.check_parity(2024, "a", {2024: "a"})["ok"]
    assert not ra.check_parity(2024, "a", {2024: "b"})["ok"]
    unchecked = ra.check_parity(2030, "a", {2024: "b"})
    assert unchecked["ok"] and not unchecked["checked"]


def test_criteria_constants_match_the_preregistration_document():
    text = PREREG.read_text()
    assert re.search(r"rel > 0\.33", text) and ra.CRITERIA["C2_material_relative_premium"] == 0.33
    assert re.search(r"e_loro >= 0\.5 \* e_self", text) and ra.CRITERIA["C3_min_retained_fraction_of_self_edge"] == 0.5
    assert "N_EVAL_TRIALS = 300" in text and ra.CRITERIA["n_eval_trials"] == 300
    assert "500 selection trials" in text and ra.CRITERIA["selection_trials"] == 500
    assert "5000-resample" in text and ra.CRITERIA["bootstrap_resamples"] == 5000
    assert "sigma = 11" in text and ra.CRITERIA["fte_margin_sigma"] == 11.0
    assert "50 forward-sampled brackets" in text and ra.CRITERIA["n_stochastic_samples"] == 50
    for r in ra.CRITERION_REFEREES:
        assert f"| {r} " in text
    assert ra.CRITERIA["criterion_referees"] == ["seed", "torvik", "blend", "pit", "market"]


def _season(year, metrics_by_ref, sel_p1, choices):
    return {"year": year, "metrics": metrics_by_ref, "selection_p1": sel_p1, "choices": choices}


def _synthetic_seasons(prod_p1, base_p1=0.04, loro_p1=None, self_p1=None, n=6):
    """Seasons where production scores `prod_p1[ref]` under each criterion referee."""
    refs = list(ra.CRITERION_REFEREES)
    seasons = []
    rng = np.random.default_rng(3)
    for k in range(n):
        m = {}
        for r in refs:
            noise = rng.normal(0, 0.005)
            row = {
                ra.BASELINE_STRATEGY: {"p_first": base_p1 + noise},
                ra.PRODUCTION_STRATEGY: {"p_first": prod_p1[r] + noise},
            }
            for h in refs:
                row[f"sel:loro:{h}"] = {"p_first": (loro_p1 or prod_p1)[r] + noise}
                row[f"sel:self:{h}"] = {"p_first": (self_p1 or prod_p1)[r] + noise}
                row[f"sel:average_all:{h}"] = {"p_first": prod_p1[r] + noise}
            m[r] = row
        choices = {h: {"loro": {"index": 0}, "self": {"index": 0}, "production": {"index": 0}, "average_all": {"index": 0}} for h in refs}
        seasons.append(_season(2011 + k, m, {}, choices))
    return seasons


def _run_criteria(seasons):
    table = ra.aggregate(seasons, "p_first")
    prem = ra.self_referee_premium(seasons, [ra.PRODUCTION_STRATEGY, ra.BASELINE_STRATEGY])
    loro = ra.loro_table(seasons)
    return ra.evaluate_criteria(table, prem, loro)


def test_criteria_robust_when_edge_holds_under_every_referee():
    prod = {r: 0.12 for r in ra.CRITERION_REFEREES}
    out = _run_criteria(_synthetic_seasons(prod))
    assert out["C1_cross_referee_edge"]["status"] == "PASS"
    assert out["C2_self_referee_premium"]["status"] == "PASS"
    assert out["C3_leave_one_referee_out"]["status"] == "PASS"
    assert out["verdict"] == "ROBUST"


def test_criteria_fail_when_independent_referee_erases_the_edge():
    prod = {r: 0.12 for r in ra.CRITERION_REFEREES}
    prod["market"] = 0.03  # below the 0.04 baseline
    out = _run_criteria(_synthetic_seasons(prod))
    assert out["C1_cross_referee_edge"]["status"] == "FAIL"
    assert out["verdict"] == "NOT ROBUST"


def test_criteria_flag_a_material_self_referee_premium():
    prod = {r: 0.06 for r in ra.CRITERION_REFEREES}
    prod["seed"] = 0.15  # 60% of the own-referee figure is premium
    out = _run_criteria(_synthetic_seasons(prod))
    c2 = out["C2_self_referee_premium"]
    assert c2["status"] == "FAIL" and c2["detail"]["material"]
    assert c2["detail"]["relative_premium"] > ra.CRITERIA["C2_material_relative_premium"]


def test_criteria_loro_fails_when_held_out_edge_vanishes():
    prod = {r: 0.12 for r in ra.CRITERION_REFEREES}
    loro = {r: 0.12 for r in ra.CRITERION_REFEREES}
    loro["pit"] = 0.03  # LORO choice loses to seed when pit is held out
    out = _run_criteria(_synthetic_seasons(prod, loro_p1=loro))
    assert out["C3_leave_one_referee_out"]["status"] == "FAIL"


def test_criteria_loro_indeterminate_when_less_than_half_the_self_edge_survives():
    prod = {r: 0.12 for r in ra.CRITERION_REFEREES}
    loro = {r: 0.12 for r in ra.CRITERION_REFEREES}
    loro["market"] = 0.06  # edge 0.02 vs self edge 0.08: retains 25%
    out = _run_criteria(_synthetic_seasons(prod, loro_p1=loro))
    assert out["C3_leave_one_referee_out"]["status"] == "INDETERMINATE"
    assert out["verdict"] == "INDETERMINATE"


def test_self_referee_premium_reports_reversal():
    prod = {r: 0.12 for r in ra.CRITERION_REFEREES}
    prod["torvik"] = 0.04  # equal to baseline -> delta <= 0 -> reversal under torvik
    seasons = _synthetic_seasons(prod)
    prem = ra.self_referee_premium(seasons, [ra.PRODUCTION_STRATEGY])
    assert "torvik" in prem[ra.PRODUCTION_STRATEGY]["reversal_under"]


def test_referee_game_scores_pool_by_game_and_skip_play_ins():
    games = [
        {"round_name": "R64", "team1_id": "a", "team2_id": "b", "team1_won": True},
        {"round_name": "R64", "team1_id": "c", "team2_id": "d", "team1_won": False},
        {"round_name": "FF", "team1_id": "a", "team2_id": "b", "team1_won": True},
    ]
    idx = {"a": 0, "b": 1, "c": 2, "d": 3}
    refs = {"sharp": {("a", "b"): 0.9, ("c", "d"): 0.1}, "flat": {("a", "b"): 0.5, ("c", "d"): 0.5}}
    out = ra.referee_game_scores(refs, games, idx)
    assert out["sharp"]["n_games"] == 2 and out["flat"]["n_games"] == 2
    assert out["sharp"]["log_loss"] == pytest.approx(-np.log(0.9))
    assert out["flat"]["log_loss"] == pytest.approx(np.log(2))
    assert out["sharp"]["sharpness"] == pytest.approx(0.4) and out["flat"]["sharpness"] == 0.0
    pooled = ra.pooled_calibration([{"referee_calibration": out}, {"referee_calibration": out}])
    assert pooled["sharp"]["n_games"] == 4 and pooled["sharp"]["log_loss"] == pytest.approx(-np.log(0.9))


def test_fte_pairwise_is_symmetric_and_monotone(monkeypatch, tmp_path):
    doc = {"columns": ["year", "team_no", "team", "seed", "round", "power_rating", "power_rating_rank"],
           "data": [[2024, 1, "Alpha", 1, 1, 95.0, 1], [2024, 2, "Beta", 2, 1, 90.0, 2], [2024, 3, "Gamma", 3, 1, 80.0, 3]]}
    (tmp_path / "data" / "kaggle").mkdir(parents=True)
    (tmp_path / "data" / "kaggle" / "fivethirtyeight_ratings.json").write_text(__import__("json").dumps(doc))
    monkeypatch.setattr(ra, "PROJECT_ROOT", tmp_path)
    pw = ra.load_fte_pairwise(2024, ["alpha", "beta", "gamma"])
    assert pw is not None
    assert pw.p("alpha", "beta") == pytest.approx(1 - pw.p("beta", "alpha"))
    assert pw.p("alpha", "gamma") > pw.p("alpha", "beta") > 0.5
    assert ra.load_fte_pairwise(2024, ["alpha", "beta", "delta"]) is None  # unmapped team -> no referee
