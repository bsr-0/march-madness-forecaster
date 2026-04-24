"""Unit tests for the Tournament Oracle.

Covers ground-truth loading from the real 2026 results file (one of the
few places in the suite where a known outcome is stable data) plus the
pure scoring helpers. The 2026 ground-truth test doubles as a regression
anchor: if anyone touches the ground-truth parser, the Illinois-F4/UConn-
finals/Michigan-champion fact must still come back correctly.
"""

from pathlib import Path

import pytest

from src.evaluation.tournament_oracle import (
    BracketOracleScore,
    OracleGroundTruth,
    PortfolioOracleReport,
    load_ground_truth,
    report_to_dict,
    score_bracket,
    score_portfolio,
)

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def test_load_ground_truth_2026_locks_actual_outcome():
    """The 2026 tournament is recent/public ground truth — lock it as a fixture."""
    gt = load_ground_truth(2026, DATA_ROOT)
    assert gt.year == 2026
    assert gt.final_four == frozenset({"arizona", "michigan", "connecticut", "illinois"})
    assert gt.finalists == frozenset({"michigan", "connecticut"})
    assert gt.champion == "michigan"
    assert gt.champion_seed == 1


def test_load_ground_truth_2025_chalk_year():
    """2025 — all four 1-seeds to F4, 1v1 championship."""
    gt = load_ground_truth(2025, DATA_ROOT)
    assert gt.final_four == frozenset({"auburn", "duke", "florida", "houston"})
    assert gt.finalists == frozenset({"florida", "houston"})
    assert gt.champion == "florida"
    assert gt.champion_seed == 1


def test_score_bracket_exact_match():
    truth = OracleGroundTruth(
        year=2026,
        final_four=frozenset({"arizona", "michigan", "connecticut", "illinois"}),
        finalists=frozenset({"michigan", "connecticut"}),
        champion="michigan",
        champion_seed=1,
    )
    s = score_bracket(
        semifinalists=["arizona", "michigan", "connecticut", "illinois"],
        finalists=["michigan", "connecticut"],
        champion="michigan",
        truth=truth,
    )
    assert s == BracketOracleScore(f4_hits=4, finals_hits=2, champ_hit=True)


def test_score_bracket_partial_match():
    """Reproduces the f4_first_tv idx=46 2026 bracket: 4/4 F4, 1/2 finals, champ hit."""
    truth = OracleGroundTruth(
        year=2026,
        final_four=frozenset({"arizona", "michigan", "connecticut", "illinois"}),
        finalists=frozenset({"michigan", "connecticut"}),
        champion="michigan",
        champion_seed=1,
    )
    s = score_bracket(
        semifinalists=["arizona", "michigan", "connecticut", "illinois"],
        finalists=["illinois", "michigan"],  # picked Illinois instead of UConn
        champion="michigan",
        truth=truth,
    )
    assert s.f4_hits == 4
    assert s.finals_hits == 1
    assert s.champ_hit is True


def test_score_bracket_all_wrong():
    truth = OracleGroundTruth(
        year=2026,
        final_four=frozenset({"arizona", "michigan", "connecticut", "illinois"}),
        finalists=frozenset({"michigan", "connecticut"}),
        champion="michigan",
        champion_seed=1,
    )
    s = score_bracket(
        semifinalists=["duke", "florida", "kansas", "houston"],
        finalists=["duke", "florida"],
        champion="duke",
        truth=truth,
    )
    assert s == BracketOracleScore(f4_hits=0, finals_hits=0, champ_hit=False)


def _mk_bracket(idx, e8, f4, champ, espn_score, mean_rank):
    return {
        "bracket_idx": idx,
        "picks": {"E8": list(e8), "F4": list(f4)},
        "champion": champ,
        "score_team_identity": float(espn_score),
        "mean_rank": float(mean_rank),
    }


def test_score_portfolio_reproduces_2026_f4_first_tv_findings():
    """Portfolio reduced from the real 2026 f4_first_tv artifact.

    Locks the core diagnostic from the A-investigation: the portfolio's
    best-scoring bracket got 4/4 F4 + correct champion, but the ranker
    promoted a worse bracket — leaving ESPN points on the table.
    """
    truth = OracleGroundTruth(
        year=2026,
        final_four=frozenset({"arizona", "michigan", "connecticut", "illinois"}),
        finalists=frozenset({"michigan", "connecticut"}),
        champion="michigan",
        champion_seed=1,
    )
    portfolio = [
        # idx=11: what the ranker chose (submitted). Low ESPN score, 1/4 F4.
        _mk_bracket(
            idx=11,
            e8=["arizona", "duke", "florida", "iowa_state"],
            f4=["arizona", "florida"],
            champ="florida",
            espn_score=620,
            mean_rank=418.1,  # lowest = submitted
        ),
        # idx=46: the one we should have submitted. 4/4 F4, correct champ, highest ESPN.
        _mk_bracket(
            idx=46,
            e8=["arizona", "connecticut", "illinois", "michigan"],
            f4=["illinois", "michigan"],
            champ="michigan",
            espn_score=1450,
            mean_rank=603.3,
        ),
        # Middling filler to make sure max aggregation isn't fooled.
        _mk_bracket(
            idx=33,
            e8=["arizona", "connecticut", "florida", "michigan"],
            f4=["connecticut", "michigan"],
            champ="michigan",
            espn_score=1420,
            mean_rank=544.5,
        ),
    ]
    report = score_portfolio(portfolio, truth)
    assert report.max_f4_hits == 4  # idx=46 hit 4/4
    assert report.max_finals_hits == 2  # idx=33 hit both finalists
    assert report.portfolio_had_champ is True
    assert report.best_score_bracket_idx == 1  # idx=46 is portfolio[1]
    assert report.best_score_oracle.f4_hits == 4
    assert report.best_score_oracle.champ_hit is True
    assert report.submitted_idx == 0  # idx=11 has lowest mean_rank
    assert report.submitted_oracle.f4_hits == 1
    assert report.submitted_oracle.champ_hit is False
    assert report.ranker_gap_espn_pts == pytest.approx(830.0)  # 1450 - 620


def test_score_portfolio_empty_raises():
    truth = OracleGroundTruth(
        year=2026,
        final_four=frozenset(),
        finalists=frozenset(),
        champion="",
        champion_seed=None,
    )
    with pytest.raises(ValueError):
        score_portfolio([], truth)


def test_report_to_dict_is_json_serializable():
    report = PortfolioOracleReport(
        max_f4_hits=4,
        max_finals_hits=2,
        portfolio_had_champ=True,
        best_score_bracket_idx=1,
        best_score_oracle=BracketOracleScore(f4_hits=4, finals_hits=1, champ_hit=True),
        submitted_idx=0,
        submitted_oracle=BracketOracleScore(f4_hits=1, finals_hits=0, champ_hit=False),
        ranker_gap_espn_pts=830.0,
    )
    import json

    s = json.dumps(report_to_dict(report))
    parsed = json.loads(s)
    assert parsed["ranker_gap_espn_pts"] == 830.0
    assert parsed["max_f4_hits"] == 4
    assert parsed["submitted_champ_hit"] is False
