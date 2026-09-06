"""PROSPECTIVE_2027 CHECKPOINT 2 — the prediction-time cutoff, enforced.

Two rules, both of which rested on operator discipline until 2026-09-05:

1. Public pick shares must carry a checkable capture time for any season with a
   declared cutoff, and must not post-date the R64 tip for any season at all.
   Until now the gate verified Torvik's provenance and described the seed
   head-to-head table, while the picks archive -- the input whose value moves
   right up to tip, and therefore the one the checkpoint singles out -- entered
   later with no timestamp recorded anywhere.

2. An official artifact, once generated, is not overwritten.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from scripts.experiments.build_candidate_artifact import (
    _public_picks_provenance,
    _refuse_to_overwrite,
)
from src.data.season_calendar import (
    EASTERN,
    get_public_picks_cutoff,
    get_round_of_64_tip,
)

DECLARED_SEASON = 2027  # has a cutoff in PUBLIC_PICKS_CUTOFFS
UNDECLARED_SEASON = 2026  # historical validation


@pytest.fixture
def picks_dir(tmp_path, monkeypatch):
    """Redirect the archive lookup at a directory the test controls."""
    import src.data.historical_picks as hp

    monkeypatch.setattr(hp, "_DEFAULT_PICKS_DIR", tmp_path)
    return tmp_path


def _write_archive(picks_dir, year, captured_at=None):
    payload = {"year": year, "source": "test fixture", "teams": {}}
    if captured_at is not None:
        payload["captured_at"] = captured_at
    (picks_dir / f"espn_picks_{year}.json").write_text(json.dumps(payload))


class TestDeclaredSeasonRequiresACheckableCaptureTime:
    def test_missing_timestamp_is_refused(self, picks_dir):
        _write_archive(picks_dir, DECLARED_SEASON, captured_at=None)
        with pytest.raises(RuntimeError, match="declared public-picks cutoff"):
            _public_picks_provenance(DECLARED_SEASON)

    def test_capture_after_the_cutoff_is_refused(self, picks_dir):
        late = get_public_picks_cutoff(DECLARED_SEASON) + timedelta(minutes=1)
        _write_archive(picks_dir, DECLARED_SEASON, captured_at=late.isoformat())
        with pytest.raises(RuntimeError):
            _public_picks_provenance(DECLARED_SEASON)

    def test_the_declared_cutoff_binds_independently_of_the_tip(self, picks_dir, monkeypatch):
        """2027's cutoff sits exactly on the R64 tip, so the two rules coincide.

        A season whose cutoff is deliberately earlier than tip separates them --
        otherwise the declared-cutoff branch would be covered only by accident,
        and moving the cutoff earlier in some future season would silently stop
        being enforced.
        """
        import src.data.season_calendar as cal

        early = get_round_of_64_tip(DECLARED_SEASON) - timedelta(days=1)
        monkeypatch.setitem(cal.PUBLIC_PICKS_CUTOFFS, DECLARED_SEASON, early)

        late = early + timedelta(hours=1)  # past the cutoff, still well before tip
        _write_archive(picks_dir, DECLARED_SEASON, captured_at=late.isoformat())
        with pytest.raises(RuntimeError, match="after the declared cutoff"):
            _public_picks_provenance(DECLARED_SEASON)

    def test_capture_at_the_cutoff_is_accepted(self, picks_dir):
        """The deadline is inclusive, or the declared instant is unmeetable.

        2027's cutoff lands exactly on the R64 tip boundary, so an exclusive
        comparison would reject a capture taken precisely when instructed.
        """
        on_time = get_public_picks_cutoff(DECLARED_SEASON)
        _write_archive(picks_dir, DECLARED_SEASON, captured_at=on_time.isoformat())
        prov = _public_picks_provenance(DECLARED_SEASON)
        assert prov["capture_time_verified"] is True
        assert prov["declared_cutoff"] == on_time.isoformat()

    def test_naive_timestamp_is_refused(self, picks_dir):
        """Four hours of ambiguity on either side of the deadline is not a rounding error."""
        _write_archive(picks_dir, DECLARED_SEASON, captured_at="2027-03-18T11:00:00")
        with pytest.raises(RuntimeError, match="no timezone"):
            _public_picks_provenance(DECLARED_SEASON)

    def test_unparseable_timestamp_is_refused(self, picks_dir):
        _write_archive(picks_dir, DECLARED_SEASON, captured_at="last tuesday")
        with pytest.raises(RuntimeError, match="ISO-8601"):
            _public_picks_provenance(DECLARED_SEASON)


class TestUndeclaredSeasonToleratesOnlyTheUnrecordedCase:
    def test_missing_timestamp_is_recorded_rather_than_refused(self, picks_dir):
        """Historical archives predate anyone thinking to write a capture time."""
        _write_archive(picks_dir, UNDECLARED_SEASON, captured_at=None)
        prov = _public_picks_provenance(UNDECLARED_SEASON)
        assert prov["capture_time_verified"] is False
        assert "cannot be placed on a clock" in prov["caveat"]

    def test_a_post_tip_timestamp_is_still_refused(self, picks_dir):
        """Tolerating an unrecorded capture time is not tolerating a leak."""
        after = get_round_of_64_tip(UNDECLARED_SEASON) + timedelta(hours=6)
        _write_archive(picks_dir, UNDECLARED_SEASON, captured_at=after.isoformat())
        with pytest.raises(RuntimeError, match="after the .* R64 tip"):
            _public_picks_provenance(UNDECLARED_SEASON)

    def test_a_pre_tip_timestamp_is_verified(self, picks_dir):
        before = datetime(2026, 3, 18, 8, 0, tzinfo=EASTERN)
        _write_archive(picks_dir, UNDECLARED_SEASON, captured_at=before.isoformat())
        prov = _public_picks_provenance(UNDECLARED_SEASON)
        assert prov["capture_time_verified"] is True


class TestMissingArchiveFailsBeforeCompute:
    def test_absent_file_is_refused(self, picks_dir):
        with pytest.raises(RuntimeError, match="no archived public picks"):
            _public_picks_provenance(UNDECLARED_SEASON)


class TestOfficialArtifactsAreImmutable:
    def test_official_season_is_never_overwritten(self, tmp_path):
        path = tmp_path / f"candidates_{DECLARED_SEASON}.json"
        path.write_text("{}")
        with pytest.raises(SystemExit, match="official prospective season"):
            _refuse_to_overwrite(path, DECLARED_SEASON, force=False)

    def test_force_does_not_lift_it(self, tmp_path):
        """--force is for development seasons; the official record is not negotiable."""
        path = tmp_path / f"candidates_{DECLARED_SEASON}.json"
        path.write_text("{}")
        with pytest.raises(SystemExit, match="with or without --force"):
            _refuse_to_overwrite(path, DECLARED_SEASON, force=True)

    def test_development_season_needs_force(self, tmp_path):
        path = tmp_path / f"candidates_{UNDECLARED_SEASON}.json"
        path.write_text("{}")
        with pytest.raises(SystemExit, match="Pass --force"):
            _refuse_to_overwrite(path, UNDECLARED_SEASON, force=False)
        _refuse_to_overwrite(path, UNDECLARED_SEASON, force=True)

    def test_a_fresh_path_is_always_allowed(self, tmp_path):
        _refuse_to_overwrite(tmp_path / "candidates_2027.json", DECLARED_SEASON, force=False)
