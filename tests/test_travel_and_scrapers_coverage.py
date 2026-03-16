"""Tests for travel_distance, contract_validator dataclasses, and tournament_results scraper.

Targets modules with 0% coverage to bring them into the test suite.
"""

import math
import tempfile

import pytest

from src.data.features.travel_distance import (
    TEAM_COORDINATES,
    VENUE_COORDINATES,
    compute_travel_advantage,
    compute_travel_distance,
    haversine_miles,
)
from src.data.contract_validator import ValidationReport, ValidationViolation
from src.data.scrapers.tournament_results import TournamentResultsScraper, _ROUND_LABELS


# ---------------------------------------------------------------------------
# haversine_miles
# ---------------------------------------------------------------------------

class TestHaversineMiles:
    """Tests for haversine_miles distance computation."""

    def test_same_point_returns_zero(self):
        assert haversine_miles(40.0, -80.0, 40.0, -80.0) == 0.0

    def test_same_point_origin(self):
        assert haversine_miles(0.0, 0.0, 0.0, 0.0) == 0.0

    def test_duke_to_indianapolis(self):
        """Duke (Durham, NC) to Indianapolis is roughly 500-600 miles."""
        duke = TEAM_COORDINATES["duke"]
        indy = VENUE_COORDINATES["indianapolis"]
        dist = haversine_miles(duke[0], duke[1], indy[0], indy[1])
        assert 400 <= dist <= 550, f"Duke->Indy was {dist:.1f} miles"

    def test_cross_country_duke_to_gonzaga(self):
        """Duke to Gonzaga (Spokane) should be 2000+ miles."""
        duke = TEAM_COORDINATES["duke"]
        gonzaga = TEAM_COORDINATES["gonzaga"]
        dist = haversine_miles(duke[0], duke[1], gonzaga[0], gonzaga[1])
        assert dist >= 2000, f"Duke->Gonzaga was {dist:.1f} miles"

    def test_symmetry(self):
        """Distance A->B equals B->A."""
        d1 = haversine_miles(36.0, -78.0, 47.0, -117.0)
        d2 = haversine_miles(47.0, -117.0, 36.0, -78.0)
        assert d1 == pytest.approx(d2)

    def test_always_non_negative(self):
        dist = haversine_miles(-33.0, 151.0, 51.5, -0.1)
        assert dist >= 0

    def test_short_distance(self):
        """Duke to UNC is roughly 8-12 miles."""
        duke = TEAM_COORDINATES["duke"]
        unc = TEAM_COORDINATES["north_carolina"]
        dist = haversine_miles(duke[0], duke[1], unc[0], unc[1])
        assert 5 <= dist <= 15, f"Duke->UNC was {dist:.1f} miles"

    def test_returns_float(self):
        result = haversine_miles(0.0, 0.0, 1.0, 1.0)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# TEAM_COORDINATES / VENUE_COORDINATES dicts
# ---------------------------------------------------------------------------

class TestCoordinateDicts:
    """Verify key entries exist in coordinate dictionaries."""

    def test_team_coords_has_duke(self):
        assert "duke" in TEAM_COORDINATES

    def test_team_coords_has_gonzaga(self):
        assert "gonzaga" in TEAM_COORDINATES

    def test_team_coords_has_kansas(self):
        assert "kansas" in TEAM_COORDINATES

    def test_team_coords_tuple_format(self):
        lat, lon = TEAM_COORDINATES["duke"]
        assert isinstance(lat, float)
        assert isinstance(lon, float)

    def test_venue_coords_has_indianapolis(self):
        assert "indianapolis" in VENUE_COORDINATES

    def test_venue_coords_has_columbus(self):
        assert "columbus" in VENUE_COORDINATES

    def test_venue_coords_tuple_format(self):
        lat, lon = VENUE_COORDINATES["indianapolis"]
        assert isinstance(lat, float)
        assert isinstance(lon, float)


# ---------------------------------------------------------------------------
# compute_travel_distance
# ---------------------------------------------------------------------------

class TestComputeTravelDistance:
    """Tests for compute_travel_distance."""

    def test_known_team_and_venue(self):
        dist = compute_travel_distance("duke", "indianapolis")
        assert isinstance(dist, float)
        assert 400 <= dist <= 550

    def test_unknown_team_returns_none(self):
        assert compute_travel_distance("fake_team_xyz", "indianapolis") is None

    def test_unknown_venue_returns_none(self):
        assert compute_travel_distance("duke", "fake_venue_xyz") is None

    def test_both_unknown_returns_none(self):
        assert compute_travel_distance("fake_team", "fake_venue") is None

    def test_custom_coords(self):
        custom_teams = {"team_a": (40.0, -80.0)}
        custom_venues = {"venue_a": (40.0, -80.0)}
        dist = compute_travel_distance(
            "team_a", "venue_a",
            team_coords=custom_teams, venue_coords=custom_venues,
        )
        assert dist == pytest.approx(0.0)

    def test_custom_coords_unknown_team(self):
        custom_teams = {"team_a": (40.0, -80.0)}
        custom_venues = {"venue_a": (40.0, -80.0)}
        result = compute_travel_distance(
            "team_b", "venue_a",
            team_coords=custom_teams, venue_coords=custom_venues,
        )
        assert result is None


# ---------------------------------------------------------------------------
# compute_travel_advantage
# ---------------------------------------------------------------------------

class TestComputeTravelAdvantage:
    """Tests for compute_travel_advantage."""

    def test_closer_team_gets_positive(self):
        """Kentucky is much closer to Indianapolis than Gonzaga -> positive."""
        adv = compute_travel_advantage("kentucky", "gonzaga", "indianapolis")
        assert adv > 0

    def test_farther_team_gets_negative(self):
        """Gonzaga is farther from Indianapolis than Kentucky -> negative."""
        adv = compute_travel_advantage("gonzaga", "kentucky", "indianapolis")
        assert adv < 0

    def test_both_unknown_returns_zero(self):
        adv = compute_travel_advantage("fake_a", "fake_b", "indianapolis")
        assert adv == 0.0

    def test_one_unknown_returns_zero(self):
        adv = compute_travel_advantage("duke", "fake_b", "indianapolis")
        assert adv == 0.0

    def test_unknown_venue_returns_zero(self):
        adv = compute_travel_advantage("duke", "kentucky", "fake_venue")
        assert adv == 0.0

    def test_same_team_returns_near_zero(self):
        adv = compute_travel_advantage("duke", "duke", "indianapolis")
        assert adv == pytest.approx(0.0, abs=1e-9)

    def test_bounded_between_neg1_and_pos1(self):
        """Even extreme distances should stay in [-1, 1]."""
        adv = compute_travel_advantage("gonzaga", "miami_fl", "indianapolis")
        assert -1.0 <= adv <= 1.0

    def test_returns_float(self):
        result = compute_travel_advantage("duke", "kansas", "indianapolis")
        assert isinstance(result, float)

    def test_custom_coords_advantage(self):
        teams = {"near": (39.77, -86.16), "far": (47.66, -117.40)}
        venues = {"venue": (39.77, -86.16)}
        adv = compute_travel_advantage(
            "near", "far", "venue",
            team_coords=teams, venue_coords=venues,
        )
        assert adv > 0


# ---------------------------------------------------------------------------
# ValidationViolation dataclass
# ---------------------------------------------------------------------------

class TestValidationViolation:
    """Tests for the ValidationViolation dataclass."""

    def test_create_with_defaults(self):
        v = ValidationViolation(
            check="test_check",
            feature_name="feat",
            message="something broke",
        )
        assert v.severity == "error"

    def test_create_with_warning(self):
        v = ValidationViolation(
            check="test_check",
            feature_name="feat",
            message="not ideal",
            severity="warning",
        )
        assert v.severity == "warning"

    def test_fields_accessible(self):
        v = ValidationViolation(
            check="c", feature_name="f", message="m", severity="error",
        )
        assert v.check == "c"
        assert v.feature_name == "f"
        assert v.message == "m"


# ---------------------------------------------------------------------------
# ValidationReport dataclass
# ---------------------------------------------------------------------------

class TestValidationReport:
    """Tests for the ValidationReport dataclass."""

    def test_empty_report_passes(self):
        r = ValidationReport()
        assert r.passed is True

    def test_warning_only_still_passes(self):
        r = ValidationReport(violations=[
            ValidationViolation("c", "f", "msg", severity="warning"),
        ])
        assert r.passed is True

    def test_error_causes_failure(self):
        r = ValidationReport(violations=[
            ValidationViolation("c", "f", "msg", severity="error"),
        ])
        assert r.passed is False

    def test_mixed_severity_fails(self):
        r = ValidationReport(violations=[
            ValidationViolation("c1", "f1", "ok", severity="warning"),
            ValidationViolation("c2", "f2", "bad", severity="error"),
        ])
        assert r.passed is False

    def test_summary_contains_passed(self):
        r = ValidationReport(checks_run=3, checks_passed=3)
        s = r.summary()
        assert "PASSED" in s
        assert "Checks run: 3" in s
        assert "Checks passed: 3" in s

    def test_summary_contains_failed(self):
        r = ValidationReport(
            checks_run=2,
            checks_passed=1,
            violations=[
                ValidationViolation("c", "f", "boom", severity="error"),
            ],
        )
        s = r.summary()
        assert "FAILED" in s
        assert "Errors: 1" in s

    def test_summary_lists_violations(self):
        r = ValidationReport(violations=[
            ValidationViolation("chk", "feat_x", "details here", severity="warning"),
        ])
        s = r.summary()
        assert "[WARNING]" in s
        assert "feat_x" in s
        assert "details here" in s

    def test_defaults(self):
        r = ValidationReport()
        assert r.violations == []
        assert r.checks_run == 0
        assert r.checks_passed == 0


# ---------------------------------------------------------------------------
# TournamentResultsScraper and _ROUND_LABELS
# ---------------------------------------------------------------------------

class TestTournamentResultsScraper:
    """Tests for TournamentResultsScraper class and _ROUND_LABELS constant."""

    def test_round_labels_r64(self):
        assert _ROUND_LABELS[0] == "R64"

    def test_round_labels_r32(self):
        assert _ROUND_LABELS[1] == "R32"

    def test_round_labels_s16(self):
        assert _ROUND_LABELS[2] == "S16"

    def test_round_labels_e8(self):
        assert _ROUND_LABELS[3] == "E8"

    def test_round_labels_f4(self):
        assert _ROUND_LABELS[4] == "F4"

    def test_round_labels_ncg(self):
        assert _ROUND_LABELS[5] == "NCG"

    def test_round_labels_length(self):
        assert len(_ROUND_LABELS) == 6

    def test_init_no_cache(self):
        scraper = TournamentResultsScraper(cache_dir=None)
        assert scraper.cache_dir is None

    def test_init_with_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scraper = TournamentResultsScraper(cache_dir=tmpdir)
            assert scraper.cache_dir is not None
            assert scraper.cache_dir.exists()

    def test_has_session(self):
        scraper = TournamentResultsScraper()
        assert scraper.session is not None

    def test_base_url(self):
        assert "sports-reference.com" in TournamentResultsScraper.BASE_URL
