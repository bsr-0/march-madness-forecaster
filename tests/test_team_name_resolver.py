"""Tests for the canonical team name resolution system."""

import pytest

from src.data.team_name_resolver import (
    MatchResult,
    TeamNameResolver,
    _normalize_str,
    _to_canonical_id,
)


# ---------------------------------------------------------------------------
# Unit tests for normalization helpers
# ---------------------------------------------------------------------------


class TestNormalizeStr:
    """Tests for _normalize_str()."""

    def test_basic_lowercase(self):
        assert _normalize_str("Arizona") == "arizona"

    def test_ampersand_replacement(self):
        assert _normalize_str("Texas A&M") == "texas a and m"

    def test_html_unescape(self):
        assert _normalize_str("Texas A&amp;M") == "texas a and m"

    def test_strip_special_chars(self):
        result = _normalize_str("St. John's (NY)")
        assert result == "st john s ny"

    def test_collapse_whitespace(self):
        assert _normalize_str("  North   Carolina  ") == "north carolina"

    def test_empty_string(self):
        assert _normalize_str("") == ""


class TestToCanonicalId:
    """Tests for _to_canonical_id()."""

    def test_basic(self):
        assert _to_canonical_id("North Carolina") == "north_carolina"

    def test_ampersand(self):
        assert _to_canonical_id("Texas A&M") == "texas_a_m"

    def test_html_entity(self):
        # html.unescape converts &amp; -> & which then becomes _
        assert _to_canonical_id("Texas A&amp;M") == "texas_a_m"

    def test_special_chars(self):
        assert _to_canonical_id("St. John's") == "st_john_s"

    def test_strips_leading_trailing_underscores(self):
        assert _to_canonical_id("  Alabama  ") == "alabama"

    def test_no_double_underscores(self):
        result = _to_canonical_id("Miami (FL)")
        assert "__" not in result


# ---------------------------------------------------------------------------
# TeamNameResolver core tests
# ---------------------------------------------------------------------------


class TestTeamNameResolverExactID:
    """Tests for pass 1: exact canonical ID match."""

    def test_exact_canonical_id(self):
        resolver = TeamNameResolver()
        result = resolver.resolve("alabama")
        assert result.canonical_id == "alabama"
        assert result.confidence == 1.0
        assert result.method == "exact_id"

    def test_canonical_id_with_underscores(self):
        resolver = TeamNameResolver()
        result = resolver.resolve("north_carolina")
        assert result.canonical_id == "north_carolina"
        assert result.method == "exact_id"

    def test_canonical_id_case_insensitive(self):
        resolver = TeamNameResolver()
        result = resolver.resolve("Alabama")
        assert result.canonical_id == "alabama"


class TestTeamNameResolverAlias:
    """Tests for pass 2: alias table lookup."""

    def test_common_abbreviation(self):
        resolver = TeamNameResolver()
        result = resolver.resolve("UConn")
        assert result.canonical_id == "connecticut"
        assert result.confidence >= 0.98
        assert "alias" in result.method

    def test_full_name(self):
        resolver = TeamNameResolver()
        result = resolver.resolve("Connecticut")
        assert result.canonical_id == "connecticut"

    def test_abbreviation_lsu(self):
        resolver = TeamNameResolver()
        result = resolver.resolve("LSU")
        assert result.canonical_id == "louisiana_state"

    def test_nickname(self):
        resolver = TeamNameResolver()
        result = resolver.resolve("Zags")
        assert result.canonical_id == "gonzaga"

    def test_nickname_ole_miss(self):
        resolver = TeamNameResolver()
        result = resolver.resolve("Ole Miss")
        assert result.canonical_id == "mississippi"

    def test_display_name_returned(self):
        resolver = TeamNameResolver()
        result = resolver.resolve("UConn")
        assert result.display_name == "Connecticut"

    def test_byu(self):
        resolver = TeamNameResolver()
        result = resolver.resolve("BYU")
        assert result.canonical_id == "brigham_young"

    def test_usc(self):
        resolver = TeamNameResolver()
        result = resolver.resolve("USC")
        assert result.canonical_id == "southern_california"

    def test_with_state_suffix(self):
        resolver = TeamNameResolver()
        result = resolver.resolve("Michigan State")
        assert result.canonical_id == "michigan_state"

    def test_tcu(self):
        resolver = TeamNameResolver()
        result = resolver.resolve("TCU")
        assert result.canonical_id == "texas_christian"


class TestTeamNameResolverSlug:
    """Tests for pass 3: Sports Reference slug match."""

    def test_slug_format(self):
        resolver = TeamNameResolver()
        # Slug would be "north-carolina"
        result = resolver.resolve("north-carolina")
        assert result.canonical_id == "north_carolina"
        assert result.confidence >= 0.90


class TestTeamNameResolverContainment:
    """Tests for pass 4: token containment."""

    def test_partial_name_match(self):
        resolver = TeamNameResolver()
        result = resolver.resolve("University of Alabama")
        assert result.canonical_id == "alabama"
        assert result.confidence >= 0.85

    def test_long_form_name(self):
        resolver = TeamNameResolver()
        result = resolver.resolve("University of Connecticut Huskies")
        assert result.canonical_id == "connecticut"
        assert "containment" in result.method


class TestTeamNameResolverFuzzy:
    """Tests for pass 5: fuzzy string matching."""

    def test_misspelling(self):
        resolver = TeamNameResolver()
        result = resolver.resolve("Gonzaga")
        assert result.canonical_id == "gonzaga"
        assert result.confidence >= 0.80

    def test_slight_misspelling(self):
        resolver = TeamNameResolver()
        result = resolver.resolve("Marquete")
        assert result.canonical_id == "marquette"
        assert result.method == "fuzzy"
        assert result.confidence >= 0.80


class TestTeamNameResolverEdgeCases:
    """Edge cases and error handling."""

    def test_empty_string(self):
        resolver = TeamNameResolver()
        result = resolver.resolve("")
        assert result.confidence == 0.0
        assert result.method == "empty"

    def test_whitespace_only(self):
        resolver = TeamNameResolver()
        result = resolver.resolve("   ")
        assert result.confidence == 0.0
        assert result.method == "empty"

    def test_nonsense_string(self):
        resolver = TeamNameResolver()
        result = resolver.resolve("xyzzy_not_a_team_12345")
        assert result.method == "unresolved"
        assert result.confidence < 0.80

    def test_miami_fl_vs_oh(self):
        """Miami is ambiguous — should resolve to one consistently."""
        resolver = TeamNameResolver()
        result = resolver.resolve("Miami (FL)")
        assert result.canonical_id == "miami_fl"
        result_oh = resolver.resolve("Miami (OH)")
        assert result_oh.canonical_id == "miami_oh"

    def test_saint_vs_st(self):
        """St. and Saint should match the same team."""
        resolver = TeamNameResolver()
        r1 = resolver.resolve("Saint Mary's")
        r2 = resolver.resolve("St. Mary's")
        assert r1.canonical_id == r2.canonical_id == "saint_marys"


# ---------------------------------------------------------------------------
# Batch resolution and utility methods
# ---------------------------------------------------------------------------


class TestTeamNameResolverBatch:
    """Tests for batch resolution and utility methods."""

    def test_resolve_batch(self):
        resolver = TeamNameResolver()
        names = ["Duke", "UNC", "Kentucky", "UConn"]
        results = resolver.resolve_batch(names)
        assert len(results) == 4
        assert results[0].canonical_id == "duke"
        assert results[1].canonical_id == "north_carolina"
        assert results[2].canonical_id == "kentucky"
        assert results[3].canonical_id == "connecticut"

    def test_get_display_name(self):
        resolver = TeamNameResolver()
        assert resolver.get_display_name("connecticut") == "Connecticut"
        assert resolver.get_display_name("louisiana_state") == "LSU"

    def test_get_display_name_unknown(self):
        resolver = TeamNameResolver()
        assert resolver.get_display_name("unknown_team") == "unknown_team"

    def test_add_alias(self):
        resolver = TeamNameResolver()
        resolver.add_alias("duke", "Blue Devils")
        result = resolver.resolve("Blue Devils")
        assert result.canonical_id == "duke"

    def test_add_alias_new_team(self):
        resolver = TeamNameResolver()
        resolver.add_alias("brand_new_team", "New Team")
        result = resolver.resolve("New Team")
        assert result.canonical_id == "brand_new_team"
        assert "brand_new_team" in resolver.known_teams

    def test_known_teams_nonempty(self):
        resolver = TeamNameResolver()
        teams = resolver.known_teams
        assert len(teams) > 100  # We have ~150 teams in the alias table
        assert "alabama" in teams
        assert "duke" in teams

    def test_extra_aliases_in_constructor(self):
        extra = {"duke": ["Dookies", "Blue Devilz"]}
        resolver = TeamNameResolver(extra_aliases=extra)
        result = resolver.resolve("Dookies")
        assert result.canonical_id == "duke"

    def test_extra_aliases_new_team(self):
        extra = {"middle_tennessee": ["Middle Tennessee", "MTSU", "Blue Raiders"]}
        resolver = TeamNameResolver(extra_aliases=extra)
        result = resolver.resolve("MTSU")
        assert result.canonical_id == "middle_tennessee"


# ---------------------------------------------------------------------------
# Cross-source consistency tests
# ---------------------------------------------------------------------------


class TestCrossSourceConsistency:
    """Verify that different source formats for the same team resolve to the same ID."""

    @pytest.fixture
    def resolver(self):
        return TeamNameResolver()

    @pytest.mark.parametrize(
        "variants,expected_id",
        [
            (["Connecticut", "UConn", "Conn"], "connecticut"),
            (["North Carolina", "UNC", "N Carolina"], "north_carolina"),
            (["Michigan State", "Michigan St", "MSU"], "michigan_state"),
            (["Texas A&M", "Texas A and M", "TAMU"], "texas_am"),
            (["St. John's", "Saint John's"], "saint_johns"),
            (["Virginia Tech", "Va Tech", "VT"], "virginia_tech"),
            (["Florida Atlantic", "FAU", "Fla Atlantic"], "florida_atlantic"),
        ],
    )
    def test_variants_resolve_same(self, resolver, variants, expected_id):
        for variant in variants:
            result = resolver.resolve(variant)
            assert result.canonical_id == expected_id, (
                f"'{variant}' resolved to '{result.canonical_id}' "
                f"(expected '{expected_id}', method={result.method})"
            )
