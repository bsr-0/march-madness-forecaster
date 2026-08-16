"""Tournament Oracle: compare a bracket against actual tournament outcomes.

Given sparse tournament data (~14 years in the backtest window), the
ledger of "did the portfolio contain a bracket that caught the actual
F4 / finals / champion?" is one of the few qualitative diagnostics that
scales. This module exposes pure functions that derive ground truth
from existing ``data/raw/historical/tournament_results_{year}.json``
files and score portfolio brackets against it.

Paired with the ``memory/tournament_oracle.md`` ledger (curated notes
on defining upsets per year), it gives every new strategy a fixed
checklist: did the portfolio's best bracket actually line up with the
year's chalk/chaos pattern?

See STRATEGY_CATALOG.md § "Metrics Reported" for how the oracle hooks
into the tier reports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# Rounds as named in tournament_results_{year}.json. F4 = semifinal games
# (4 teams, 2 games); NCG = championship game (2 teams, 1 game).
_SEMI_ROUND = "F4"
_FINAL_ROUND = "NCG"


@dataclass(frozen=True)
class OracleGroundTruth:
    """Actual tournament outcome for one year.

    Attributes:
        year: Tournament year.
        final_four: 4 teams that reached the semifinals.
        finalists: 2 teams that reached the championship game.
        champion: Winner of the championship game.
        champion_seed: Seed of the champion (for chalk/upset annotation).
    """

    year: int
    final_four: frozenset
    finalists: frozenset
    champion: str
    champion_seed: Optional[int]


@dataclass(frozen=True)
class BracketOracleScore:
    """How well a single bracket matched the actual outcome."""

    f4_hits: int  # 0-4
    finals_hits: int  # 0-2
    champ_hit: bool


@dataclass(frozen=True)
class PortfolioOracleReport:
    """Aggregate oracle diagnostics over a full portfolio.

    Attributes:
        max_f4_hits: Best F4-hit count across the portfolio.
        max_finals_hits: Best finals-hit count across the portfolio.
        portfolio_had_champ: True if any bracket picked the real champion.
        best_score_bracket_idx: Portfolio index of the highest-ESPN-score
            bracket (the one we wish we'd submitted).
        best_score_oracle: Oracle score of the best-ESPN bracket.
        submitted_idx: Portfolio index of the bracket the ranker selected
            (#1 by mean_rank ascending).
        submitted_oracle: Oracle score of the submitted bracket.
        ranker_gap_espn_pts: ESPN points left on the table (best_score −
            submitted_score). This is the KPI for the ranker/opponent-model
            problem — if it shrinks over time, the ranker is learning.
    """

    max_f4_hits: int
    max_finals_hits: int
    portfolio_had_champ: bool
    best_score_bracket_idx: int
    best_score_oracle: BracketOracleScore
    submitted_idx: int
    submitted_oracle: BracketOracleScore
    ranker_gap_espn_pts: float


def load_ground_truth(year: int, data_root: Path) -> OracleGroundTruth:
    """Derive F4 / finalists / champion from ``tournament_results_{year}.json``.

    Prefers the consolidated ``tournament_context_{year}.json`` (key
    "results") when present, falling back to the old per-type file."""
    ctx_path = Path(data_root) / "raw" / "historical" / f"tournament_context_{year}.json"
    raw = None
    if ctx_path.exists():
        with open(ctx_path) as f:
            ctx = json.load(f)
        if "results" in ctx:
            raw = ctx["results"]
    if raw is None:
        path = Path(data_root) / "raw" / "historical" / f"tournament_results_{year}.json"
        with open(path) as f:
            raw = json.load(f)
    games = raw["games"] if isinstance(raw, dict) and "games" in raw else raw

    f4_teams = set()
    finalists: set = set()
    champion: Optional[str] = None
    champion_seed: Optional[int] = None

    for g in games:
        round_name = g.get("round_name") or g.get("round")
        if round_name == _SEMI_ROUND:
            f4_teams.add(g["team1_id"])
            f4_teams.add(g["team2_id"])
        elif round_name == _FINAL_ROUND:
            finalists.add(g["team1_id"])
            finalists.add(g["team2_id"])
            winner_is_team1 = bool(g.get("team1_won"))
            champion = g["team1_id"] if winner_is_team1 else g["team2_id"]
            champion_seed = g.get("team1_seed") if winner_is_team1 else g.get("team2_seed")

    if not f4_teams or not finalists or champion is None:
        raise ValueError(
            f"Incomplete tournament_results for {year} — "
            f"F4={len(f4_teams)}, finalists={len(finalists)}, champion={champion}"
        )

    return OracleGroundTruth(
        year=year,
        final_four=frozenset(f4_teams),
        finalists=frozenset(finalists),
        champion=champion,
        champion_seed=champion_seed,
    )


def score_bracket(
    semifinalists: Iterable[str],
    finalists: Iterable[str],
    champion: str,
    truth: OracleGroundTruth,
) -> BracketOracleScore:
    """Score one bracket against the ground truth.

    Args:
        semifinalists: The 4 teams the bracket picked to reach the semifinal
            round (i.e., Elite Eight winners — in the backtest schema this is
            ``picks["E8"]``).
        finalists: The 2 teams picked to reach the championship game
            (``picks["F4"]`` in the backtest schema).
        champion: The team picked to win it all.
        truth: Ground-truth outcome.
    """
    semi_set = set(semifinalists)
    final_set = set(finalists)
    return BracketOracleScore(
        f4_hits=len(semi_set & truth.final_four),
        finals_hits=len(final_set & truth.finalists),
        champ_hit=champion == truth.champion,
    )


def _bracket_to_picks(bracket: dict) -> Tuple[List[str], List[str], str]:
    """Extract (semifinalists, finalists, champion) from a backtest bracket dict.

    The backtest harness stores ``picks["E8"]`` as the 4 semifinalists
    (winners of the Elite Eight), ``picks["F4"]`` as the 2 championship-
    game participants, and a top-level ``champion`` field. Submission
    artifacts use a different layout handled by the caller.
    """
    picks = bracket["picks"]
    return list(picks.get("E8", [])), list(picks.get("F4", [])), bracket["champion"]


def score_portfolio(
    portfolio: Sequence[dict],
    truth: OracleGroundTruth,
    *,
    espn_score_key: str = "score_team_identity",
    rank_key: str = "mean_rank",
) -> PortfolioOracleReport:
    """Score an entire portfolio against the ground truth.

    Args:
        portfolio: List of bracket dicts in the
            ``artifacts/backtest_brackets/backtest_brackets_{year}.json``
            schema — each item has ``picks``, ``champion``, the ESPN score
            under ``score_team_identity`` and an internal ranker score
            under ``mean_rank`` (lower = submitted first).
        truth: Ground-truth outcome.
        espn_score_key: Field name for ESPN score (default matches the
            backtest harness).
        rank_key: Field name for the internal ranker's ordering (default
            matches the backtest harness).

    Returns:
        ``PortfolioOracleReport``. The ``ranker_gap_espn_pts`` field is the
        diagnostic KPI for the selection/ranking problem: points left on
        the table because the ranker did not promote the best bracket.
    """
    if not portfolio:
        raise ValueError("empty portfolio")

    best_score_idx = max(range(len(portfolio)), key=lambda i: portfolio[i][espn_score_key])
    submitted_idx = min(range(len(portfolio)), key=lambda i: portfolio[i][rank_key])

    max_f4 = 0
    max_finals = 0
    had_champ = False
    for b in portfolio:
        semi, finals, champ = _bracket_to_picks(b)
        s = score_bracket(semi, finals, champ, truth)
        if s.f4_hits > max_f4:
            max_f4 = s.f4_hits
        if s.finals_hits > max_finals:
            max_finals = s.finals_hits
        if s.champ_hit:
            had_champ = True

    best_semi, best_finals, best_champ = _bracket_to_picks(portfolio[best_score_idx])
    sub_semi, sub_finals, sub_champ = _bracket_to_picks(portfolio[submitted_idx])

    return PortfolioOracleReport(
        max_f4_hits=max_f4,
        max_finals_hits=max_finals,
        portfolio_had_champ=had_champ,
        best_score_bracket_idx=best_score_idx,
        best_score_oracle=score_bracket(best_semi, best_finals, best_champ, truth),
        submitted_idx=submitted_idx,
        submitted_oracle=score_bracket(sub_semi, sub_finals, sub_champ, truth),
        ranker_gap_espn_pts=float(portfolio[best_score_idx][espn_score_key] - portfolio[submitted_idx][espn_score_key]),
    )


def report_to_dict(report: PortfolioOracleReport) -> Dict:
    """JSON-serializable view for ``_save_budget_summary`` and artifacts."""
    return {
        "max_f4_hits": report.max_f4_hits,
        "max_finals_hits": report.max_finals_hits,
        "portfolio_had_champ": report.portfolio_had_champ,
        "best_score_bracket_idx": report.best_score_bracket_idx,
        "best_score_f4_hits": report.best_score_oracle.f4_hits,
        "best_score_finals_hits": report.best_score_oracle.finals_hits,
        "best_score_champ_hit": report.best_score_oracle.champ_hit,
        "submitted_idx": report.submitted_idx,
        "submitted_f4_hits": report.submitted_oracle.f4_hits,
        "submitted_finals_hits": report.submitted_oracle.finals_hits,
        "submitted_champ_hit": report.submitted_oracle.champ_hit,
        "ranker_gap_espn_pts": report.ranker_gap_espn_pts,
    }
