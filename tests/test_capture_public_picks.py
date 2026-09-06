"""The 2027 public-picks capture: the one step with a deadline attached.

CHECKPOINT 2 fixes a capture instant and the provenance gate refuses to build
without a checkable capture time. Nothing could produce one until this script
existed, so these tests cover the parts that will run exactly once, under time
pressure, on a March afternoon -- where a silent scale error or a quietly empty
payload is far more expensive than a refusal.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from scripts.capture_public_picks import (
    CaptureError,
    _assert_not_already_captured,
    _assert_usable,
    _assert_within_cutoff,
    to_archive_teams,
)
from src.data.season_calendar import get_public_picks_cutoff, get_round_of_64_tip

DECLARED_SEASON = 2027
UNDECLARED_SEASON = 2026


class _Picks:
    """Stand-in for PublicPicks: only ``as_dict`` and ``seed`` are read."""

    def __init__(self, champ=5.0, seed=1):
        self.seed = seed
        self._d = {"R64": 99.0, "R32": 80.0, "S16": 50.0, "E8": 25.0, "F4": 12.0, "CHAMP": champ}

    @property
    def as_dict(self):
        return dict(self._d)


class _Consensus:
    def __init__(self, teams):
        self.teams = teams
        self.sources = ["test"]


def _consensus(n=64, champ=5.0):
    return _Consensus({f"team-{i}": _Picks(champ=champ) for i in range(n)})


class TestScaleConversion:
    def test_percentages_become_fractions(self):
        """ESPN gives 0-100; every consumer of the archive reads 0-1.

        Getting this backwards is a silent 100x error that makes every team
        look certain, and nothing downstream would flag it -- a confident
        probability is a plausible probability.
        """
        teams = to_archive_teams(_consensus(n=1, champ=42.0))
        row = teams["team-0"]
        assert row["CHAMP"] == pytest.approx(0.42)
        assert row["R64"] == pytest.approx(0.99)
        assert all(0.0 <= v <= 1.0 for k, v in row.items() if k != "seed")

    def test_out_of_range_input_is_refused(self):
        """A scale change at the source must fail loudly, not be normalised away."""
        bad = _Consensus({"team-0": _Picks(champ=250.0)})
        with pytest.raises(CaptureError, match="outside 0-100"):
            to_archive_teams(bad)

    def test_the_seed_rides_along(self):
        assert to_archive_teams(_consensus(n=1))["team-0"]["seed"] == 1


class TestTheDeadlineBinds:
    def test_capture_before_the_cutoff_is_allowed(self):
        _assert_within_cutoff(
            DECLARED_SEASON, get_public_picks_cutoff(DECLARED_SEASON) - timedelta(hours=2)
        )

    def test_capture_exactly_at_the_cutoff_is_allowed(self):
        _assert_within_cutoff(DECLARED_SEASON, get_public_picks_cutoff(DECLARED_SEASON))

    def test_capture_after_the_cutoff_is_refused(self):
        late = get_public_picks_cutoff(DECLARED_SEASON) + timedelta(minutes=1)
        with pytest.raises(CaptureError, match="past the declared cutoff"):
            _assert_within_cutoff(DECLARED_SEASON, late)

    def test_there_is_no_override(self):
        """Deliberate: a late capture is not the capture that was promised."""
        late = get_public_picks_cutoff(DECLARED_SEASON) + timedelta(minutes=1)
        with pytest.raises(CaptureError, match="no --force"):
            _assert_within_cutoff(DECLARED_SEASON, late)

    def test_an_undeclared_season_still_cannot_capture_after_tip(self):
        after = get_round_of_64_tip(UNDECLARED_SEASON) + timedelta(hours=1)
        with pytest.raises(CaptureError, match="past the .* R64 tip"):
            _assert_within_cutoff(UNDECLARED_SEASON, after)


class TestOneCaptureOnly:
    def test_declared_season_refuses_to_recapture(self, tmp_path):
        (tmp_path / f"espn_picks_{DECLARED_SEASON}.json").write_text("{}")
        with pytest.raises(CaptureError, match="calls fatal"):
            _assert_not_already_captured(DECLARED_SEASON, tmp_path)

    def test_any_accepted_filename_counts_as_captured(self, tmp_path):
        """Writing under a second accepted name would not be a second capture."""
        (tmp_path / f"public_picks_{DECLARED_SEASON}.json").write_text("{}")
        with pytest.raises(CaptureError):
            _assert_not_already_captured(DECLARED_SEASON, tmp_path)

    def test_development_season_says_delete_it(self, tmp_path):
        (tmp_path / f"espn_picks_{UNDECLARED_SEASON}.json").write_text("{}")
        with pytest.raises(CaptureError, match="Delete it"):
            _assert_not_already_captured(UNDECLARED_SEASON, tmp_path)

    def test_a_clean_directory_is_fine(self, tmp_path):
        _assert_not_already_captured(DECLARED_SEASON, tmp_path)


class TestAPayloadMustBeWorthKeeping:
    def test_too_few_teams_is_refused(self):
        with pytest.raises(CaptureError, match="expected at least"):
            _assert_usable(to_archive_teams(_consensus(n=10)), DECLARED_SEASON)

    def test_an_all_zero_champion_column_is_refused(self):
        """The parse failing looks exactly like nobody picking a champion."""
        with pytest.raises(CaptureError, match="CHAMP=0"):
            _assert_usable(to_archive_teams(_consensus(champ=0.0)), DECLARED_SEASON)

    def test_a_full_field_passes(self):
        _assert_usable(to_archive_teams(_consensus()), DECLARED_SEASON)
