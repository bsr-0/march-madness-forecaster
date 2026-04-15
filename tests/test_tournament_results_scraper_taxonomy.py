"""Structural-fix regression tests for the ESPN tournament-results scraper.

Closes the O22 root cause: the fragile first-match-wins substring matcher
in ``src/data/scrapers/tournament_results.py::_parse_espn_event`` silently
collapsed 48 non-NCG 2026 games onto ``NCG`` because the bare
``"championship"`` pattern matched any event-note headline that contained
the word "championship".

These tests assert three structural properties of the replacement:

1. **Specificity ordering.** The matcher picks the *most specific* round
   pattern that appears anywhere in the joined notes — so a headline like
   "NCAA Men's Basketball Championship - First Round" must resolve to
   R64, never NCG. Node ordering (which note is first) must not affect
   the answer.

2. **No silent default.** When *no* pattern matches, the matcher returns
   ``None`` and the caller drops the event. R64 is no longer a
   "we don't know" fallback.

3. **Post-scrape guardrails.** ``_assert_canonical_labels`` raises on any
   non-canonical label; ``_assert_canonical_distribution`` raises when a
   complete-sized scrape (67 games for 2011+, 64 for pre-2011) carries
   the O22 distribution shape. Partial scrapes (< full count) are
   skipped so mid-tournament calls don't spuriously fail.
"""

from __future__ import annotations

import pytest

from src.data.scrapers.tournament_results import (
    CANONICAL_ROUND_LABELS,
    TaxonomyValidationError,
    _assert_canonical_distribution,
    _assert_canonical_labels,
    _round_name_from_headlines,
)


# ---------------------------------------------------------------------------
# Matcher — specificity ordering
# ---------------------------------------------------------------------------


class TestHeadlineMatcherSpecificity:
    """The matcher must prefer the most-specific round phrase anywhere in notes."""

    def test_championship_plus_first_round_resolves_to_r64(self) -> None:
        """The exact O22 failure headline shape. Pre-fix this returned NCG."""
        assert _round_name_from_headlines(["NCAA Men's Basketball Championship - First Round"]) == "R64"

    def test_championship_plus_second_round_resolves_to_r32(self) -> None:
        assert _round_name_from_headlines(["NCAA Men's Basketball Championship - Second Round"]) == "R32"

    def test_sweet_16_beats_generic_championship_across_notes(self) -> None:
        """A generic 'Championship' note must not suppress a specific round note."""
        assert _round_name_from_headlines(["NCAA Men's Basketball Championship", "Sweet 16"]) == "S16"

    def test_final_four_beats_championship_across_notes(self) -> None:
        assert _round_name_from_headlines(["NCAA Men's Basketball Championship", "Final Four"]) == "F4"

    def test_national_championship_resolves_to_ncg(self) -> None:
        assert _round_name_from_headlines(["2026 NCAA Men's National Championship"]) == "NCG"

    def test_championship_game_resolves_to_ncg(self) -> None:
        assert _round_name_from_headlines(["Championship Game"]) == "NCG"

    def test_first_four_beats_first_round_when_both_present(self) -> None:
        """Historical ESPN corner case: First Four games are sometimes
        tagged 'First Four / First Round'. The play-in label wins."""
        assert _round_name_from_headlines(["First Four - First Round"]) == "FF"

    def test_note_order_is_irrelevant(self) -> None:
        """Swapping the order of notes must not change the answer."""
        a = _round_name_from_headlines(["NCAA Championship", "Sweet 16"])
        b = _round_name_from_headlines(["Sweet 16", "NCAA Championship"])
        assert a == b == "S16"

    def test_elite_eight_digit_and_word_forms(self) -> None:
        """Both 'Elite 8' and 'Elite Eight' must resolve to E8."""
        assert _round_name_from_headlines(["Elite 8"]) == "E8"
        assert _round_name_from_headlines(["Elite Eight"]) == "E8"

    def test_sweet_sixteen_digit_and_word_forms(self) -> None:
        assert _round_name_from_headlines(["Sweet 16"]) == "S16"
        assert _round_name_from_headlines(["Sweet Sixteen"]) == "S16"


# ---------------------------------------------------------------------------
# Matcher — no silent default
# ---------------------------------------------------------------------------


class TestHeadlineMatcherReturnsNoneOnNoMatch:
    """Silent R64 default was the O22 root cause — verify it's gone."""

    def test_unrelated_text_returns_none(self) -> None:
        assert _round_name_from_headlines(["Some unrelated string"]) is None

    def test_empty_list_returns_none(self) -> None:
        assert _round_name_from_headlines([]) is None

    def test_only_empty_strings_returns_none(self) -> None:
        assert _round_name_from_headlines(["", "", None]) is None

    def test_bare_championship_does_not_leak_to_ncg(self) -> None:
        """The bare 'championship' catch-all is deliberately removed.

        Pre-fix, ``championship`` appeared in ``_ESPN_ROUND_MAP`` and
        collapsed any headline containing the word onto NCG. A headline
        that ONLY says 'Men's Basketball Championship' (no 'National',
        no specific round phrase) must now return None so the event
        gets dropped rather than mislabeled.
        """
        assert _round_name_from_headlines(["Men's Basketball Championship"]) is None


# ---------------------------------------------------------------------------
# Post-scrape label guardrail
# ---------------------------------------------------------------------------


def _canonical_complete_post_2011() -> list[dict]:
    """Return 67 canonical-shaped games for distribution tests."""
    shape = {"FF": 4, "R64": 32, "R32": 16, "S16": 8, "E8": 4, "F4": 2, "NCG": 1}
    games: list[dict] = []
    for label, count in shape.items():
        games.extend({"round_name": label} for _ in range(count))
    return games


def _canonical_complete_pre_2011() -> list[dict]:
    shape = {"FF": 1, "R64": 32, "R32": 16, "S16": 8, "E8": 4, "F4": 2, "NCG": 1}
    games: list[dict] = []
    for label, count in shape.items():
        games.extend({"round_name": label} for _ in range(count))
    return games


class TestAssertCanonicalLabels:
    def test_passes_on_canonical_labels(self) -> None:
        _assert_canonical_labels(_canonical_complete_post_2011())  # no raise

    def test_raises_on_r68_label(self) -> None:
        games = _canonical_complete_post_2011()
        games[0]["round_name"] = "R68"  # the exact O22 non-canonical label
        with pytest.raises(TaxonomyValidationError) as excinfo:
            _assert_canonical_labels(games)
        assert "R68" in str(excinfo.value)
        assert "O22" in str(excinfo.value)

    def test_raises_on_unknown_label(self) -> None:
        games = _canonical_complete_post_2011()
        games[5]["round_name"] = "CHAMP"
        with pytest.raises(TaxonomyValidationError) as excinfo:
            _assert_canonical_labels(games)
        assert "CHAMP" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Post-scrape distribution guardrail
# ---------------------------------------------------------------------------


class TestAssertCanonicalDistribution:
    def test_passes_on_canonical_post_2011(self) -> None:
        _assert_canonical_distribution(_canonical_complete_post_2011(), 2026)

    def test_passes_on_canonical_pre_2011(self) -> None:
        _assert_canonical_distribution(_canonical_complete_pre_2011(), 2008)

    def test_raises_on_exact_o22_defect_shape(self) -> None:
        """{R68:4, NCG:49, S16:8, E8:4, F4:2} — the literal malformed shape
        the scraper produced for 2026 before the structural fix."""
        shape = {"R68": 4, "NCG": 49, "S16": 8, "E8": 4, "F4": 2}
        games: list[dict] = []
        for label, count in shape.items():
            games.extend({"round_name": label} for _ in range(count))
        assert len(games) == 67  # canonical total — full-scrape gate engaged
        with pytest.raises(TaxonomyValidationError) as excinfo:
            _assert_canonical_distribution(games, 2026)
        msg = str(excinfo.value)
        assert "O22" in msg
        assert "non-canonical round distribution" in msg

    def test_skips_partial_scrape(self) -> None:
        """Mid-tournament scrape (fewer than 67 games) must NOT raise —
        downstream expects to be able to poll during a live tournament."""
        partial = [{"round_name": "R64"} for _ in range(12)]
        _assert_canonical_distribution(partial, 2026)  # no raise

    def test_raises_on_off_by_one_count_at_full_total(self) -> None:
        """67 canonical-labeled games but wrong per-round counts must raise."""
        games = _canonical_complete_post_2011()
        # swap: drop one R64, add one R32 (still 67 total).
        r64_idx = next(i for i, g in enumerate(games) if g["round_name"] == "R64")
        games[r64_idx]["round_name"] = "R32"
        with pytest.raises(TaxonomyValidationError):
            _assert_canonical_distribution(games, 2026)


# ---------------------------------------------------------------------------
# Canonical label set — no drift
# ---------------------------------------------------------------------------


def test_canonical_labels_match_historical_loader_contract() -> None:
    """The label set must match the contract in
    src/data/historical_tournament_results.py::TournamentGame.
    If this drifts, downstream consumers break silently."""
    expected = {"FF", "R64", "R32", "S16", "E8", "F4", "NCG"}
    assert CANONICAL_ROUND_LABELS == expected
