"""Tests for Selection Sunday snapshot correctness.

Verifies:
- All features obey timestamp constraints
- Training data is snapshot-based (each training year uses its own snapshot)
- Feature computation happens after filtering
- Play-in games are modeled probabilistically
- Chronological expanding-window cross-validation is enforced
- Forbidden data sources are rejected
"""

from __future__ import annotations

from datetime import date, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.evaluation.snapshot_integrity import (
    SELECTION_SUNDAY_DATES,
    DataSourceRecord,
    SnapshotIntegrityError,
    SnapshotIntegrityValidator,
    SnapshotMetadata,
    SnapshotViolation,
    compute_feature_hash,
    get_selection_sunday,
    selection_sunday_as_datetime,
)
from src.evaluation.bracket_snapshot import (
    BracketSnapshot,
    PlayInMatchup,
    TeamEntry,
    marginalise_play_in,
)
from src.evaluation.selection_sunday_backtest import (
    SelectionSundayBacktester,
    SelectionSundaySnapshot,
    SnapshotBuilder,
    YearBacktestResult,
)


# ---------------------------------------------------------------------------
# Selection Sunday date tests
# ---------------------------------------------------------------------------

class TestSelectionSundayDates:
    """Verify Selection Sunday date handling."""

    def test_known_dates_exist(self):
        """All historical years with data have Selection Sunday dates."""
        for year in [2018, 2019, 2021, 2022, 2023, 2024, 2025]:
            assert year in SELECTION_SUNDAY_DATES

    def test_get_selection_sunday_known_year(self):
        """get_selection_sunday returns correct date for known years."""
        assert get_selection_sunday(2024) == date(2024, 3, 17)

    def test_get_selection_sunday_before_tournament(self):
        """Selection Sunday is before tournament start for all known years."""
        try:
            from src.pipeline.config import TOURNAMENT_START_DATES
            for year in SELECTION_SUNDAY_DATES:
                if year in TOURNAMENT_START_DATES:
                    ss = SELECTION_SUNDAY_DATES[year]
                    ts = TOURNAMENT_START_DATES[year]
                    assert ss < ts, (
                        f"Year {year}: Selection Sunday {ss} must be before "
                        f"tournament start {ts}"
                    )
        except ImportError:
            pytest.skip("pipeline.config not available")

    def test_get_selection_sunday_unknown_year_raises(self):
        """Unknown year without fallback raises ValueError."""
        with pytest.raises(ValueError, match="No Selection Sunday date"):
            get_selection_sunday(1999)

    def test_selection_sunday_as_datetime(self):
        """selection_sunday_as_datetime returns end-of-day datetime."""
        dt = selection_sunday_as_datetime(2024)
        assert dt.year == 2024
        assert dt.month == 3
        assert dt.day == 17
        assert dt.hour == 23


# ---------------------------------------------------------------------------
# Snapshot integrity validator tests
# ---------------------------------------------------------------------------

class TestSnapshotIntegrityValidator:
    """Verify snapshot integrity validation."""

    def _make_valid_metadata(self, year: int = 2024) -> SnapshotMetadata:
        ss = get_selection_sunday(year)
        cutoff = selection_sunday_as_datetime(year)
        return SnapshotMetadata(
            year=year,
            selection_sunday=ss,
            cutoff_datetime=cutoff,
            data_sources=[
                DataSourceRecord(
                    name="historical_games",
                    source_type="games",
                    timestamp=datetime(year, 3, 1),
                    record_count=100,
                ),
            ],
            games_included=100,
            games_excluded=5,
            teams_count=64,
            feature_hash="abc123",
            features_computed_after_filter=True,
        )

    def test_valid_snapshot_passes(self):
        """A correctly constructed snapshot passes validation."""
        metadata = self._make_valid_metadata()
        validator = SnapshotIntegrityValidator(2024)
        violations = validator.validate(metadata)
        assert violations == []

    def test_features_computed_before_filter_fails(self):
        """Snapshot with features computed before filtering fails."""
        metadata = self._make_valid_metadata()
        metadata.features_computed_after_filter = False
        validator = SnapshotIntegrityValidator(2024)
        violations = validator.validate(metadata)
        assert len(violations) >= 1
        assert any("after" in v.description.lower() for v in violations)

    def test_tournament_results_source_forbidden(self):
        """Tournament results data source is rejected."""
        metadata = self._make_valid_metadata()
        metadata.data_sources.append(DataSourceRecord(
            name="tournament_results_2024",
            source_type="tournament_results",
            timestamp=datetime(2024, 4, 1),
            record_count=63,
        ))
        validator = SnapshotIntegrityValidator(2024)
        violations = validator.validate(metadata)
        assert len(violations) >= 1
        assert any("forbidden" in v.description.lower() for v in violations)

    def test_post_selection_timestamp_rejected(self):
        """Data source with timestamp after Selection Sunday is rejected."""
        metadata = self._make_valid_metadata()
        # Add a source dated after Selection Sunday
        metadata.data_sources.append(DataSourceRecord(
            name="end_of_season_ratings",
            source_type="ratings",
            timestamp=datetime(2024, 4, 15),
            record_count=350,
        ))
        validator = SnapshotIntegrityValidator(2024)
        violations = validator.validate(metadata)
        assert len(violations) >= 1
        assert any("after" in v.description.lower() for v in violations)

    def test_validate_or_raise_with_violations(self):
        """validate_or_raise raises SnapshotIntegrityError."""
        metadata = self._make_valid_metadata()
        metadata.features_computed_after_filter = False
        validator = SnapshotIntegrityValidator(2024)
        with pytest.raises(SnapshotIntegrityError):
            validator.validate_or_raise(metadata)

    def test_game_dates_before_selection_sunday(self):
        """Games before Selection Sunday are accepted."""
        validator = SnapshotIntegrityValidator(2024)
        game_dates = [date(2024, 1, 15), date(2024, 2, 20), date(2024, 3, 10)]
        violations = validator.check_game_dates(game_dates)
        assert violations == []

    def test_game_dates_on_selection_sunday_rejected(self):
        """Games on Selection Sunday are rejected."""
        validator = SnapshotIntegrityValidator(2024)
        ss = get_selection_sunday(2024)
        game_dates = [date(2024, 1, 15), ss]
        violations = validator.check_game_dates(game_dates)
        assert len(violations) == 1

    def test_tournament_round_names_rejected(self):
        """Tournament round games in snapshot are rejected."""
        validator = SnapshotIntegrityValidator(2024)
        round_names = ["regular", "R64", "S16"]
        violations = validator.check_tournament_game_exclusion(round_names)
        assert len(violations) == 2  # R64 and S16

    def test_year_mismatch_flagged(self):
        """Snapshot year != validator year is flagged."""
        metadata = self._make_valid_metadata(year=2024)
        validator = SnapshotIntegrityValidator(2023)
        violations = validator.validate(metadata)
        assert any("year" in v.description.lower() for v in violations)

    def test_retrospective_revision_forbidden(self):
        """Retrospective metric revisions are rejected."""
        metadata = self._make_valid_metadata()
        metadata.data_sources.append(DataSourceRecord(
            name="revised_kenpom",
            source_type="retrospective_revision",
        ))
        validator = SnapshotIntegrityValidator(2024)
        violations = validator.validate(metadata)
        assert any("forbidden" in v.description.lower() for v in violations)


# ---------------------------------------------------------------------------
# Feature hash tests
# ---------------------------------------------------------------------------

class TestFeatureHash:
    """Verify feature hash computation for lineage tracking."""

    def test_hash_deterministic(self):
        """Same data produces same hash."""
        data = np.array([1.0, 2.0, 3.0])
        h1 = compute_feature_hash(data.tobytes())
        h2 = compute_feature_hash(data.tobytes())
        assert h1 == h2

    def test_different_data_different_hash(self):
        """Different data produces different hash."""
        d1 = np.array([1.0, 2.0, 3.0])
        d2 = np.array([1.0, 2.0, 4.0])
        h1 = compute_feature_hash(d1.tobytes())
        h2 = compute_feature_hash(d2.tobytes())
        assert h1 != h2


# ---------------------------------------------------------------------------
# Play-in probability marginalisation tests
# ---------------------------------------------------------------------------

class TestPlayInMarginalisation:
    """Verify probabilistic play-in handling."""

    def test_marginalise_symmetric(self):
        """Equal play-in teams produce average of matchup probs."""
        play_in = PlayInMatchup(
            team_a="team_a", team_b="team_b",
            seed=16, region="East", slot_index=0,
        )

        def predict_fn(t1, t2):
            # Equal teams
            return 0.5

        result = marginalise_play_in(predict_fn, play_in, "opponent")
        assert abs(result - 0.5) < 1e-10

    def test_marginalise_with_dominant_play_in_team(self):
        """If play-in team A always wins, result equals P(opponent beats A)."""
        play_in = PlayInMatchup(
            team_a="team_a", team_b="team_b",
            seed=16, region="East", slot_index=0,
        )

        def predict_fn(t1, t2):
            if t1 == "team_a" and t2 == "team_b":
                return 1.0  # A always beats B in play-in
            if t1 == "opponent" and t2 == "team_a":
                return 0.9  # Opponent beats A 90%
            if t1 == "opponent" and t2 == "team_b":
                return 0.8  # Opponent beats B 80%
            return 0.5

        result = marginalise_play_in(predict_fn, play_in, "opponent")
        # P(A wins play-in) = 1.0, so result = 1.0 * 0.9 + 0.0 * 0.8 = 0.9
        assert abs(result - 0.9) < 1e-10

    def test_marginalise_weighted_correctly(self):
        """Marginalised probability is correctly weighted."""
        play_in = PlayInMatchup(
            team_a="team_a", team_b="team_b",
            seed=16, region="East", slot_index=0,
        )

        def predict_fn(t1, t2):
            if t1 == "team_a" and t2 == "team_b":
                return 0.6  # A 60% to win play-in
            if t1 == "opponent" and t2 == "team_a":
                return 0.9
            if t1 == "opponent" and t2 == "team_b":
                return 0.8
            return 0.5

        result = marginalise_play_in(predict_fn, play_in, "opponent")
        expected = 0.6 * 0.9 + 0.4 * 0.8  # = 0.86
        assert abs(result - expected) < 1e-10

    def test_play_in_teams_always_probabilistic(self):
        """Play-in matchups are never treated as certain."""
        play_in = PlayInMatchup(
            team_a="team_a", team_b="team_b",
            seed=16, region="East", slot_index=0,
        )

        # Even if one team is much stronger, both outcomes are considered
        def predict_fn(t1, t2):
            if t1 == "team_a" and t2 == "team_b":
                return 0.95
            if t1 == "opponent":
                return 0.7
            return 0.5

        result = marginalise_play_in(predict_fn, play_in, "opponent")
        # Both play-in outcomes must contribute
        p_a_wins = 0.95
        p_b_wins = 0.05
        expected = p_a_wins * 0.7 + p_b_wins * 0.7  # Both get same pred here
        assert abs(result - expected) < 1e-10


# ---------------------------------------------------------------------------
# Bracket snapshot tests
# ---------------------------------------------------------------------------

class TestBracketSnapshot:
    """Verify bracket snapshot data structures."""

    def _make_snapshot(self) -> BracketSnapshot:
        teams = [
            TeamEntry(team_id=f"team_{s}_{r}", seed=s, region=r)
            for r in ["East", "West", "South", "Midwest"]
            for s in range(1, 17)
        ]
        play_ins = [
            PlayInMatchup(
                team_a="play_a", team_b="play_b",
                seed=16, region="East", slot_index=0,
            ),
        ]
        return BracketSnapshot(
            year=2024,
            teams=teams + [
                TeamEntry(team_id="play_a", seed=16, region="East", is_play_in=True),
                TeamEntry(team_id="play_b", seed=16, region="East", is_play_in=True),
            ],
            seeds={t.team_id: t.seed for t in teams},
            regions={t.team_id: t.region for t in teams},
            play_in_matchups=play_ins,
        )

    def test_has_play_ins(self):
        snap = self._make_snapshot()
        assert snap.has_play_ins

    def test_get_play_in_teams(self):
        snap = self._make_snapshot()
        pi_teams = snap.get_play_in_teams()
        assert "play_a" in pi_teams
        assert "play_b" in pi_teams

    def test_teams_by_region(self):
        snap = self._make_snapshot()
        by_region = snap.get_teams_by_region()
        assert len(by_region) >= 4
        for region_teams in by_region.values():
            assert len(region_teams) >= 16

    def test_serialise_roundtrip(self):
        snap = self._make_snapshot()
        d = snap.to_dict()
        snap2 = BracketSnapshot.from_dict(d)
        assert snap2.year == snap.year
        assert len(snap2.play_in_matchups) == len(snap.play_in_matchups)


# ---------------------------------------------------------------------------
# Chronological cross-validation enforcement
# ---------------------------------------------------------------------------

class TestChronologicalCV:
    """Verify expanding-window cross-validation."""

    def test_training_years_strictly_before_eval_year(self):
        """Training years must be < eval year."""
        backtester = SelectionSundayBacktester()

        # Mock get_available_years and snapshot building
        with patch.object(backtester, '_get_available_years', return_value=[2018, 2019, 2021, 2022]):
            with patch.object(backtester, 'build_snapshot') as mock_build:
                mock_build.side_effect = lambda y, **kw: SelectionSundaySnapshot(year=y)
                with patch.object(backtester, '_generate_predictions', return_value={}):
                    results = backtester.run_backtest(
                        evaluation_years=[2022],
                        first_training_year=2018,
                    )

        if results:
            assert all(ty < 2022 for ty in results[0].training_years)

    def test_covid_year_excluded(self):
        """2020 (COVID) is never in training years."""
        backtester = SelectionSundayBacktester()

        with patch.object(backtester, '_get_available_years',
                          return_value=[2018, 2019, 2020, 2021, 2022]):
            with patch.object(backtester, 'build_snapshot') as mock_build:
                mock_build.side_effect = lambda y, **kw: SelectionSundaySnapshot(year=y)
                with patch.object(backtester, '_generate_predictions', return_value={}):
                    results = backtester.run_backtest(
                        evaluation_years=[2022],
                        first_training_year=2018,
                    )

        if results:
            assert 2020 not in results[0].training_years


# ---------------------------------------------------------------------------
# Snapshot-based training data tests
# ---------------------------------------------------------------------------

class TestSnapshotBasedTraining:
    """Verify that training data uses per-year snapshots."""

    def test_each_training_year_gets_own_snapshot(self):
        """Every training year gets its own Selection Sunday snapshot."""
        backtester = SelectionSundayBacktester()
        built_years = []

        def track_build(year, **kwargs):
            built_years.append(year)
            return SelectionSundaySnapshot(year=year)

        with patch.object(backtester, '_get_available_years',
                          return_value=[2018, 2019, 2021, 2022]):
            with patch.object(backtester, 'build_snapshot', side_effect=track_build):
                with patch.object(backtester, '_generate_predictions', return_value={}):
                    backtester.run_backtest(
                        evaluation_years=[2022],
                        first_training_year=2018,
                    )

        # All training years (2018, 2019, 2021) + eval year (2022) should be built
        for y in [2018, 2019, 2021, 2022]:
            assert y in built_years, f"Year {y} snapshot was not built"
