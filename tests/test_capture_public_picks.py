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


class TestManualFallback:
    """The capture happens once, so a moved endpoint must not be terminal.

    ESPN's pick endpoints are undocumented and have changed domains more than
    once. If the live fetch fails at 11:58 on the day, there is no second
    attempt at 12:05 -- the cutoff does not move. ``--from-json`` lets the
    operator paste in what they can see in a browser and still record a capture
    that satisfies every other rule.

    It is a fallback for the transport, not for the deadline: the cutoff still
    binds, the payload is still validated, and ``captured_at`` is still the
    moment the script runs rather than anything the file claims.
    """

    @pytest.fixture
    def before_the_cutoff(self, monkeypatch):
        """Freeze the clock, or these tests start failing in March 2027."""
        import scripts.capture_public_picks as cap

        instant = get_public_picks_cutoff(DECLARED_SEASON) - timedelta(hours=1)
        monkeypatch.setattr(cap, "_now_eastern", lambda: instant)
        return instant

    @staticmethod
    def _valid_payload(path):
        import json

        # Round sums must be plausible or the scraper's own bracket-structure
        # check rejects the payload: 64 teams -> R64 ~3200%, halving each round.
        teams = {}
        for i in range(64):
            advances = i % 2 == 0
            teams[f"team-{i}"] = {
                "team_name": f"T{i}",
                "seed": (i % 16) + 1,
                "region": "East",
                "round_of_64_pct": 100.0 if advances else 0.0,
                "round_of_32_pct": 50.0 if advances else 0.0,
                "sweet_16_pct": 25.0 if advances else 0.0,
                "elite_8_pct": 12.5 if advances else 0.0,
                "final_four_pct": 6.25 if advances else 0.0,
                "champion_pct": 3.125 if advances else 0.0,
            }
        path.write_text(json.dumps({"teams": teams, "sources": ["manual"]}))
        return path

    def test_a_hand_saved_payload_can_be_captured(self, tmp_path, before_the_cutoff):
        from scripts.capture_public_picks import capture

        src = self._valid_payload(tmp_path / "manual.json")
        payload = capture(DECLARED_SEASON, tmp_path, dry_run=True, from_json=src)

        assert len(payload["teams"]) == 64
        assert "manual" in payload["source"]
        assert payload["teams"]["team-0"]["CHAMP"] == pytest.approx(0.03125)

    def test_captured_at_is_the_run_time_not_the_file(self, tmp_path, before_the_cutoff):
        """The capture instant is when the operator captured, not what a file says."""
        import json

        src = self._valid_payload(tmp_path / "manual.json")
        stale = json.loads(src.read_text())
        stale["timestamp"] = "2020-01-01T00:00:00-05:00"
        src.write_text(json.dumps(stale))

        from scripts.capture_public_picks import capture

        payload = capture(DECLARED_SEASON, tmp_path, dry_run=True, from_json=src)
        assert payload["captured_at"] == before_the_cutoff.isoformat()

    def test_the_deadline_still_binds(self, tmp_path, monkeypatch):
        """The escape hatch is for the transport, not the cutoff."""
        import scripts.capture_public_picks as cap

        late = get_public_picks_cutoff(DECLARED_SEASON) + timedelta(minutes=1)
        monkeypatch.setattr(cap, "_now_eastern", lambda: late)
        src = self._valid_payload(tmp_path / "manual.json")

        with pytest.raises(CaptureError, match="past the declared cutoff"):
            cap.capture(DECLARED_SEASON, tmp_path, dry_run=True, from_json=src)

    def test_a_missing_file_is_named(self, tmp_path, before_the_cutoff):
        from scripts.capture_public_picks import capture

        with pytest.raises(CaptureError, match="does not exist"):
            capture(DECLARED_SEASON, tmp_path, dry_run=True, from_json=tmp_path / "nope.json")
