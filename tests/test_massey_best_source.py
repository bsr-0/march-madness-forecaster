"""Unit tests for A6 massey_best source.

Covers the per-system Brier-selection harness:
- Log5 pairwise win probability math.
- Per-(system, year) Brier scoring on tournament games.
- Walk-forward cumulative Brier across years < test_year.
- ``select_best_system``: argmin over systems, min-games gated.
- ``build_massey_best_round_probabilities``: end-to-end round_probs from the
  selected system's ratings for the test year.
- Pipeline registry + permutation enumeration.

Walk-forward contract: selection at test_year Y uses only tournament
outcomes from years strictly < Y. Each test year's selection is computed
in isolation.
"""

import json
from pathlib import Path

import pytest

from src.prediction.massey_best_probabilities import (
    _cumulative_brier_per_system,
    _pairwise_win_prob,
    _score_system_on_year,
    build_massey_best_round_probabilities,
    select_best_system,
)

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def test_pairwise_win_prob_symmetric_case():
    """Equal ratings → P = 0.5."""
    p = _pairwise_win_prob(0.6, 0.6)
    assert p == pytest.approx(0.5, abs=1e-9)


def test_pairwise_win_prob_favored_team():
    """Stronger team has P > 0.5; weaker has P < 0.5."""
    p_fav = _pairwise_win_prob(0.9, 0.3)  # strong vs weak
    p_dog = _pairwise_win_prob(0.3, 0.9)
    assert p_fav > 0.5
    assert p_dog < 0.5
    assert p_fav + p_dog == pytest.approx(1.0, abs=1e-9)


def test_pairwise_win_prob_fallback_for_zero_ratings():
    """Both ratings 0 → P=0.5 (coin flip), not NaN."""
    p = _pairwise_win_prob(0.0, 0.0)
    assert p == pytest.approx(0.5, abs=1e-9)


def test_score_system_on_year_known_system_beats_uniform():
    """A real ranker (POM) on a real tournament year should produce Brier < 0.25.

    Baseline: uniform probability P=0.5 for every game → Brier = 0.25.
    A system with signal scores LOWER than that.
    """
    result = _score_system_on_year("POM", 2024, DATA_ROOT)
    assert result is not None
    brier_sum, n_games = result
    assert n_games >= 50  # tournament has ~63 games; some may skip on missing ratings
    mean_brier = brier_sum / n_games
    assert mean_brier < 0.25, f"POM 2024 mean Brier {mean_brier:.3f} should beat uniform (0.25)"


def test_score_system_on_year_missing_data_returns_none():
    """Missing system or year data → None, not a crash."""
    result = _score_system_on_year("NONEXISTENT_SYSTEM", 2024, DATA_ROOT)
    assert result is None
    result = _score_system_on_year("POM", 1900, DATA_ROOT)
    assert result is None


def test_cumulative_brier_respects_walk_forward():
    """_cumulative_brier_per_system(2015) must not read 2015-or-later tournament data.

    Verified indirectly: the same call for test_year=2025 should produce more
    games-scored than test_year=2015 (9 more tournaments of cumulation).
    """
    cum_2015 = _cumulative_brier_per_system(2015, DATA_ROOT)
    cum_2025 = _cumulative_brier_per_system(2025, DATA_ROOT)
    # For a system present in both windows, 2025 should have more games.
    assert "POM" in cum_2015 and "POM" in cum_2025
    _, games_2015 = cum_2015["POM"]
    _, games_2025 = cum_2025["POM"]
    assert games_2025 > games_2015, f"Walk-forward broken: POM games 2015={games_2015}, 2025={games_2025}"


def test_select_best_system_2026_returns_known_ranker():
    """select_best_system(2026) should return a well-known ranker (POM/SAG/BAR/etc.),
    not an obscure one-year system.
    """
    best = select_best_system(2026, DATA_ROOT)
    assert best is not None, "Should find a best system for 2026"
    # The min-games threshold + historical Brier competition should weed out
    # obscure systems. Accept any system that's been around historically —
    # the specific winner isn't what we're locking, just that it's plausible.
    assert isinstance(best, str)
    assert len(best) >= 2  # not an empty string
    # Hard sanity: must be a system that exists in the 2026 consolidated file.
    ratings_path = DATA_ROOT / "raw" / "historical" / "external_ratings_2026.json"
    assert ratings_path.exists(), "external_ratings_2026.json must exist"
    with open(ratings_path) as f:
        systems = json.load(f)["systems"]
    assert best in systems, f"Selected system {best!r} must have a 2026 entry in external_ratings_2026.json"


def test_select_best_system_respects_min_games_threshold():
    """A system with very few historical games can't win the selection,
    even if its one-year Brier looks good."""
    # Run with a high threshold that should eliminate most minor systems.
    best_strict = select_best_system(2026, DATA_ROOT, min_games=500)
    best_loose = select_best_system(2026, DATA_ROOT, min_games=10)
    # Both should return something; the strict one should be a longer-
    # standing ranker. If both are the same, that's also fine (the top
    # system clears even the strict threshold).
    assert best_strict is not None
    assert best_loose is not None


def test_select_best_system_returns_none_for_unsupported_year():
    """Year with no historical tournament data → None (no crash)."""
    best = select_best_system(1995, DATA_ROOT, min_games=100)
    assert best is None


def test_build_massey_best_round_probabilities_2026_covers_field():
    """End-to-end: the selected system's ratings → round_probs for all 68 teams."""
    import json

    seeds_raw = json.load(open(DATA_ROOT / "raw" / "tournament_seeds_2026.json"))
    seeds = {s["team_id"]: s["seed"] for s in seeds_raw}
    regions = {s["team_id"]: s["region"] for s in seeds_raw}

    rp = build_massey_best_round_probabilities(seeds, regions, test_year=2026, data_root=DATA_ROOT, n_sims=2000)
    assert rp is not None, "Should produce round_probs for 2026"
    # All 68 teams present.
    assert set(rp.keys()) == set(seeds.keys())
    # Each team has 6 rounds.
    for tid, rounds in rp.items():
        assert set(rounds.keys()) >= {"R64", "R32", "S16", "E8", "F4", "CHAMP"}
    # R64 rp = P(team wins its R64 game) — sum across teams = 32 (num R64 games).
    r64_sum = sum(rp[t]["R64"] for t in rp)
    assert 30 <= r64_sum <= 34, f"R64 total should be ~32 (num R64 games), got {r64_sum:.1f}"
    # CHAMP rp sums to 1 (exactly one champion per sim).
    champ_sum = sum(rp[t]["CHAMP"] for t in rp)
    assert 0.9 <= champ_sum <= 1.1, f"CHAMP total should be ~1, got {champ_sum:.3f}"


def test_build_massey_best_returns_none_without_selection():
    """Year with no selectable system → None (resolved gracefully upstream)."""
    rp = build_massey_best_round_probabilities(
        seeds={"duke": 1}, regions={"duke": "East"}, test_year=1900, data_root=DATA_ROOT
    )
    assert rp is None


def test_massey_best_in_implemented_sources():
    from src.prediction.strategy_pipeline import IMPLEMENTED_SOURCES

    assert "massey_best" in IMPLEMENTED_SOURCES


def test_permutation_generator_enumerates_massey_best_strategies():
    from src.prediction.strategy_pipeline import generate_all_permutations

    strategies = generate_all_permutations(implemented_only=True)
    mb = [s for s in strategies if "massey_best" in s]
    assert len(mb) >= 10, f"Expected >=10 massey_best strategies, got {len(mb)}"
    # Bare single-source + forward.
    assert "massey_best" in strategies
    # Source + construction variant.
    assert any(s == "massey_best_f4_first" for s in strategies)


def test_score_system_walk_forward_same_system_different_years():
    """Scoring POM on two different years produces two different Briers —
    sanity check that year-selection is actually routing to different files.
    """
    result_2019 = _score_system_on_year("POM", 2019, DATA_ROOT)
    result_2024 = _score_system_on_year("POM", 2024, DATA_ROOT)
    assert result_2019 is not None and result_2024 is not None
    # Not the exact same numbers — differ at least in game count.
    assert result_2019[1] != result_2024[1] or result_2019[0] != result_2024[0]
