"""Tests for round segmentation correctness.

Verifies:
- Games are correctly categorized by seed matchup and round
- All seed-pair segments are populated with correct matchups
- Round name normalization works across variants
- Segment coverage: every tournament game falls into at least one segment
"""

from __future__ import annotations

import pytest

from src.evaluation.round_analysis import (
    EvalGame,
    ROUND_NAME_MAP,
    ROUND_SEGMENTS,
    RoundAnalysisReport,
    RoundSegmenter,
    SEED_MATCHUP_SEGMENTS,
    SegmentMetrics,
    compute_round_analysis,
    compute_segment_metrics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_r64_game(seed1: int, seed2: int, prediction: float = 0.5, outcome: float = 1.0) -> EvalGame:
    """Helper to create a Round of 64 game."""
    return EvalGame(
        team1_id=f"team_{seed1}", team2_id=f"team_{seed2}",
        team1_seed=seed1, team2_seed=seed2,
        round_name="R64", prediction=prediction, outcome=outcome, year=2024,
    )


def _make_game(round_name: str, seed1: int = 1, seed2: int = 2,
               prediction: float = 0.7, outcome: float = 1.0) -> EvalGame:
    """Helper to create a game in any round."""
    return EvalGame(
        team1_id=f"team_{seed1}", team2_id=f"team_{seed2}",
        team1_seed=seed1, team2_seed=seed2,
        round_name=round_name, prediction=prediction, outcome=outcome, year=2024,
    )


def _make_full_tournament_games() -> list[EvalGame]:
    """Create a representative set of tournament games across all segments."""
    games = []
    # R64 — all seed matchups
    for fav, dog in [(1, 16), (2, 15), (3, 14), (4, 13), (5, 12), (6, 11), (7, 10), (8, 9)]:
        games.append(_make_r64_game(fav, dog, prediction=0.8, outcome=1.0))
        games.append(_make_r64_game(fav, dog, prediction=0.7, outcome=1.0))
        games.append(_make_r64_game(fav, dog, prediction=0.6, outcome=0.0))  # upset
        games.append(_make_r64_game(fav, dog, prediction=0.55, outcome=1.0))

    # R32
    for _ in range(16):
        games.append(_make_game("R32", 1, 8, 0.75, 1.0))

    # Sweet 16
    for _ in range(8):
        games.append(_make_game("S16", 1, 4, 0.65, 1.0))

    # Elite 8
    for _ in range(4):
        games.append(_make_game("E8", 1, 3, 0.6, 1.0))

    # Final Four
    games.append(_make_game("F4", 1, 2, 0.55, 1.0))
    games.append(_make_game("F4", 1, 3, 0.6, 0.0))

    # Championship
    games.append(_make_game("NCG", 1, 1, 0.5, 1.0))

    return games


# ---------------------------------------------------------------------------
# Seed matchup segmentation tests
# ---------------------------------------------------------------------------

class TestSeedMatchupSegmentation:
    """Verify first-round games are grouped by seed matchup."""

    def test_heavy_favorites_correct(self):
        """1v16, 2v15, 3v14 go to heavy_favorites."""
        segmenter = RoundSegmenter()
        games = [
            _make_r64_game(1, 16),
            _make_r64_game(2, 15),
            _make_r64_game(3, 14),
        ]
        result = segmenter.segment_by_seed_matchup(games)
        assert len(result["heavy_favorites"]) == 3

    def test_mid_tier_correct(self):
        """4v13, 5v12 go to mid_tier."""
        segmenter = RoundSegmenter()
        games = [
            _make_r64_game(4, 13),
            _make_r64_game(5, 12),
        ]
        result = segmenter.segment_by_seed_matchup(games)
        assert len(result["mid_tier"]) == 2

    def test_toss_ups_correct(self):
        """6v11, 7v10, 8v9 go to toss_ups."""
        segmenter = RoundSegmenter()
        games = [
            _make_r64_game(6, 11),
            _make_r64_game(7, 10),
            _make_r64_game(8, 9),
        ]
        result = segmenter.segment_by_seed_matchup(games)
        assert len(result["toss_ups"]) == 3

    def test_seed_pair_order_independent(self):
        """(16, 1) should be treated same as (1, 16)."""
        segmenter = RoundSegmenter()
        # Team with seed 16 listed first
        game = EvalGame(
            team1_id="underdog", team2_id="favorite",
            team1_seed=16, team2_seed=1,
            round_name="R64", prediction=0.1, outcome=0.0, year=2024,
        )
        result = segmenter.segment_by_seed_matchup([game])
        assert len(result["heavy_favorites"]) == 1

    def test_non_r64_games_excluded_from_seed_segments(self):
        """Only R64 games appear in seed matchup segments."""
        segmenter = RoundSegmenter()
        games = [
            _make_game("S16", 1, 4),
            _make_game("E8", 2, 3),
        ]
        result = segmenter.segment_by_seed_matchup(games)
        for seg_games in result.values():
            assert len(seg_games) == 0

    def test_all_r64_seed_pairs_covered(self):
        """Every standard R64 seed pair maps to exactly one segment."""
        segmenter = RoundSegmenter()
        all_pairs = [(1, 16), (2, 15), (3, 14), (4, 13), (5, 12), (6, 11), (7, 10), (8, 9)]
        games = [_make_r64_game(fav, dog) for fav, dog in all_pairs]
        result = segmenter.segment_by_seed_matchup(games)

        total = sum(len(g) for g in result.values())
        assert total == 8, f"Expected 8 games categorized, got {total}"


# ---------------------------------------------------------------------------
# Round segmentation tests
# ---------------------------------------------------------------------------

class TestRoundSegmentation:
    """Verify games are grouped by tournament round."""

    def test_all_rounds_populated(self):
        """Full tournament produces games in every round segment."""
        segmenter = RoundSegmenter()
        games = _make_full_tournament_games()
        result = segmenter.segment_by_round(games)

        for seg_name in ROUND_SEGMENTS:
            assert len(result[seg_name]) > 0, f"Segment {seg_name} is empty"

    def test_round_name_normalization(self):
        """Various round name formats are normalized correctly."""
        assert ROUND_NAME_MAP["R64"] == "R64"
        assert ROUND_NAME_MAP["Round of 64"] == "R64"
        assert ROUND_NAME_MAP["S16"] == "S16"
        assert ROUND_NAME_MAP["Sweet 16"] == "S16"
        assert ROUND_NAME_MAP["E8"] == "E8"
        assert ROUND_NAME_MAP["Elite 8"] == "E8"
        assert ROUND_NAME_MAP["F4"] == "F4"
        assert ROUND_NAME_MAP["Final Four"] == "F4"
        assert ROUND_NAME_MAP["NCG"] == "NCG"
        assert ROUND_NAME_MAP["CHAMP"] == "NCG"
        assert ROUND_NAME_MAP["Championship"] == "NCG"

    def test_alternate_round_names_work(self):
        """Games with alternate round names are correctly segmented."""
        segmenter = RoundSegmenter()
        games = [
            _make_game("Sweet 16", 1, 4),
            _make_game("Elite 8", 1, 2),
            _make_game("Final Four", 1, 1),
            _make_game("Championship", 1, 2),
        ]
        result = segmenter.segment_by_round(games)
        assert len(result["sweet_16"]) == 1
        assert len(result["elite_8"]) == 1
        assert len(result["final_four"]) == 1
        assert len(result["championship"]) == 1


# ---------------------------------------------------------------------------
# Game categorisation tests
# ---------------------------------------------------------------------------

class TestGameCategorisation:
    """Verify categorise_game assigns correct segment names."""

    def test_r64_game_gets_round_and_seed_segment(self):
        """R64 1v16 game should be in both round_of_64 and heavy_favorites."""
        segmenter = RoundSegmenter()
        game = _make_r64_game(1, 16)
        categories = segmenter.categorise_game(game)
        assert "round_of_64" in categories
        assert "heavy_favorites" in categories

    def test_later_round_game_gets_round_only(self):
        """Sweet 16 game only gets round segment, not seed segment."""
        segmenter = RoundSegmenter()
        game = _make_game("S16", 1, 4)
        categories = segmenter.categorise_game(game)
        assert "sweet_16" in categories
        # Seed matchup segments only apply to R64
        assert "heavy_favorites" not in categories
        assert "mid_tier" not in categories
        assert "toss_ups" not in categories


# ---------------------------------------------------------------------------
# EvalGame property tests
# ---------------------------------------------------------------------------

class TestEvalGameProperties:
    """Verify EvalGame computed properties."""

    def test_seed_pair_normalized(self):
        """seed_pair always returns (lower, higher)."""
        game = EvalGame(
            team1_id="a", team2_id="b",
            team1_seed=16, team2_seed=1,
            round_name="R64", prediction=0.1, outcome=0.0,
        )
        assert game.seed_pair == (1, 16)

    def test_favorite_prediction_oriented(self):
        """favorite_prediction returns P(lower seed wins)."""
        # Team1 is the favorite (lower seed)
        game1 = EvalGame(
            team1_id="a", team2_id="b",
            team1_seed=1, team2_seed=16,
            round_name="R64", prediction=0.95, outcome=1.0,
        )
        assert game1.favorite_prediction == 0.95

        # Team2 is the favorite (lower seed)
        game2 = EvalGame(
            team1_id="a", team2_id="b",
            team1_seed=16, team2_seed=1,
            round_name="R64", prediction=0.1, outcome=0.0,
        )
        assert game2.favorite_prediction == 0.9  # 1 - 0.1

    def test_is_upset(self):
        """is_upset is True when higher seed wins."""
        # Favorite wins — not an upset
        game1 = _make_r64_game(1, 16, prediction=0.9, outcome=1.0)
        assert not game1.is_upset

        # Underdog wins — upset
        game2 = _make_r64_game(1, 16, prediction=0.9, outcome=0.0)
        assert game2.is_upset

    def test_canonical_round(self):
        """canonical_round normalizes round names."""
        game = _make_game("Sweet 16")
        assert game.canonical_round == "S16"


# ---------------------------------------------------------------------------
# Segment metrics computation tests
# ---------------------------------------------------------------------------

class TestSegmentMetrics:
    """Verify per-segment metric computation."""

    def test_empty_segment_returns_zero_games(self):
        """Empty segment produces n_games=0."""
        result = compute_segment_metrics([], "empty")
        assert result.n_games == 0
        assert result.brier is None

    def test_metrics_computed_for_nonempty_segment(self):
        """Non-empty segment produces all metric fields."""
        games = [
            _make_r64_game(1, 16, 0.9, 1.0),
            _make_r64_game(1, 16, 0.85, 1.0),
            _make_r64_game(1, 16, 0.8, 0.0),
            _make_r64_game(1, 16, 0.7, 1.0),
            _make_r64_game(1, 16, 0.75, 1.0),
        ]
        result = compute_segment_metrics(games, "heavy_favorites", n_bootstrap=100)
        assert result.n_games == 5
        assert result.brier is not None
        assert result.log_loss_val is not None
        assert result.accuracy_val is not None
        assert result.ece is not None

    def test_upset_metrics_correct(self):
        """Upset count and recall computed correctly."""
        games = [
            _make_r64_game(1, 16, 0.9, 1.0),   # fav wins, no upset
            _make_r64_game(1, 16, 0.9, 0.0),   # upset, not predicted
            _make_r64_game(1, 16, 0.3, 0.0),   # upset, predicted (fav_pred < 0.5)
        ]
        result = compute_segment_metrics(games, "test", n_bootstrap=100)
        assert result.n_upsets == 2
        assert result.upset_recall_val is not None
        assert result.upset_recall_val == 0.5  # 1 of 2 upsets predicted


# ---------------------------------------------------------------------------
# Full round analysis report tests
# ---------------------------------------------------------------------------

class TestRoundAnalysisReport:
    """Verify the full compute_round_analysis function."""

    def test_full_report_structure(self):
        """Report has all expected sections."""
        games = _make_full_tournament_games()
        report = compute_round_analysis(games, n_bootstrap=100)

        assert report.n_total_games == len(games)
        assert len(report.global_metrics) > 0
        assert "brier_score" in report.global_metrics
        assert len(report.seed_matchup_segments) > 0
        assert len(report.round_segments) > 0

    def test_report_serializable(self):
        """Report can be serialized to dict."""
        games = _make_full_tournament_games()
        report = compute_round_analysis(games, n_bootstrap=50)
        d = report.to_dict()
        assert "global_metrics" in d
        assert "seed_matchup_segments" in d
        assert "round_segments" in d

    def test_empty_games_returns_empty_report(self):
        """Empty game list produces empty report."""
        report = compute_round_analysis([])
        assert report.n_total_games == 0
        assert report.global_metrics == {}

    def test_years_tracked(self):
        """Report tracks which years were evaluated."""
        games = _make_full_tournament_games()
        report = compute_round_analysis(games, n_bootstrap=50)
        assert 2024 in report.years_evaluated
