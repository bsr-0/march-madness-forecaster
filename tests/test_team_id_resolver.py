"""Tests for GameToTorvikResolver — shared team ID matching."""

import pytest

from src.data.team_id_resolver import GameToTorvikResolver

# Representative Torvik ID set (subset used for testing)
_TORVIK_IDS = {
    "michigan", "michigan_st", "duke", "connecticut", "miami_fl", "miami_oh",
    "n_c__state", "mcneese_st", "appalachian_st", "cal_st__bakersfield",
    "cal_st__fullerton", "cal_st__northridge", "fiu", "grambling_st",
    "illinois_chicago", "iu_indy", "liu", "louisiana_monroe", "loyola_md",
    "sam_houston_st", "nebraska_omaha", "nicholls_st", "san_jose_st",
    "southeastern_louisiana", "tennessee_martin", "texas_a_m_corpus_chris",
    "umkc", "usc_upstate", "william___mary", "texas", "texas_tech",
    "texas_a_m", "texas_southern", "alabama", "alabama_st", "alabama_a_m",
    "penn", "penn_st", "georgia", "georgia_tech", "georgia_southern",
    "iowa_st", "cal_baptist", "florida", "florida_st", "florida_a_m",
    "houston", "houston_christian", "virginia", "virginia_tech",
    "hawaii", "colorado_st",
}


@pytest.fixture
def resolver():
    return GameToTorvikResolver(_TORVIK_IDS)


class TestExactMatch:
    def test_exact_torvik_id(self, resolver):
        assert resolver.resolve("duke") == "duke"

    def test_exact_michigan(self, resolver):
        assert resolver.resolve("michigan") == "michigan"


class TestPrefixMatching:
    def test_mascot_suffix(self, resolver):
        assert resolver.resolve("duke_blue_devils") == "duke"

    def test_michigan_wolverines(self, resolver):
        assert resolver.resolve("michigan_wolverines") == "michigan"

    def test_longest_prefix_wins(self, resolver):
        """texas_a_m_aggies should match texas_a_m, not texas."""
        assert resolver.resolve("texas_a_m_aggies") == "texas_a_m"

    def test_texas_tech_over_texas(self, resolver):
        assert resolver.resolve("texas_tech_red_raiders") == "texas_tech"

    def test_georgia_tech_over_georgia(self, resolver):
        assert resolver.resolve("georgia_tech_yellow_jackets") == "georgia_tech"

    def test_virginia_tech_over_virginia(self, resolver):
        assert resolver.resolve("virginia_tech_hokies") == "virginia_tech"

    def test_houston_christian_over_houston(self, resolver):
        assert resolver.resolve("houston_christian_huskies") == "houston_christian"

    def test_florida_a_m_over_florida(self, resolver):
        assert resolver.resolve("florida_a_m_rattlers") == "florida_a_m"


class TestStStateExpansion:
    def test_state_to_st(self, resolver):
        """alabama_state_hornets should match alabama_st."""
        assert resolver.resolve("alabama_state_hornets") == "alabama_st"

    def test_michigan_state_spartans(self, resolver):
        assert resolver.resolve("michigan_state_spartans") == "michigan_st"

    def test_iowa_state(self, resolver):
        assert resolver.resolve("iowa_state") == "iowa_st"

    def test_colorado_state_rams(self, resolver):
        assert resolver.resolve("colorado_state_rams") == "colorado_st"


class TestDoubleUnderscore:
    def test_cal_state_bakersfield(self, resolver):
        assert resolver.resolve("cal_state_bakersfield_roadrunners") == "cal_st__bakersfield"

    def test_cal_state_fullerton(self, resolver):
        assert resolver.resolve("cal_state_fullerton_titans") == "cal_st__fullerton"


class TestExplicitAliases:
    """Tournament-critical teams that require explicit aliases."""

    def test_uconn(self, resolver):
        assert resolver.resolve("uconn_huskies") == "connecticut"

    def test_nc_state(self, resolver):
        assert resolver.resolve("nc_state_wolfpack") == "n_c__state"

    def test_mcneese(self, resolver):
        assert resolver.resolve("mcneese_cowboys") == "mcneese_st"

    def test_miami_fl(self, resolver):
        assert resolver.resolve("miami_hurricanes") == "miami_fl"

    def test_miami_oh(self, resolver):
        assert resolver.resolve("miami_oh_redhawks") == "miami_oh"

    def test_fiu(self, resolver):
        assert resolver.resolve("florida_international_panthers") == "fiu"

    def test_grambling(self, resolver):
        assert resolver.resolve("grambling_tigers") == "grambling_st"

    def test_sam_houston(self, resolver):
        assert resolver.resolve("sam_houston_bearkats") == "sam_houston_st"

    def test_nicholls(self, resolver):
        assert resolver.resolve("nicholls_colonels") == "nicholls_st"

    def test_william_mary(self, resolver):
        assert resolver.resolve("william_mary_tribe") == "william___mary"

    def test_iu_indy(self, resolver):
        assert resolver.resolve("iu_indianapolis_jaguars") == "iu_indy"

    def test_umkc(self, resolver):
        assert resolver.resolve("kansas_city_roos") == "umkc"


class TestDisplayNameResolution:
    def test_uconn_display(self, resolver):
        assert resolver.resolve_display_name("UConn Huskies") == "connecticut"

    def test_mcneese_display(self, resolver):
        assert resolver.resolve_display_name("McNeese Cowboys") == "mcneese_st"

    def test_app_state_display(self, resolver):
        assert resolver.resolve_display_name("App State Mountaineers") == "appalachian_st"

    def test_simple_display(self, resolver):
        assert resolver.resolve_display_name("Duke Blue Devils") == "duke"


class TestRemapDict:
    def test_remap_basic(self, resolver):
        ff = {
            "michigan_wolverines": {"to_rate": 0.15},
            "duke_blue_devils": {"to_rate": 0.12},
        }
        result = resolver.remap_dict(ff)
        assert "michigan" in result
        assert "duke" in result
        assert result["michigan"]["to_rate"] == 0.15

    def test_remap_preserves_unresolved(self, resolver):
        ff = {"unknown_team_xyz": {"to_rate": 0.2}}
        result = resolver.remap_dict(ff)
        assert "unknown_team_xyz" in result

    def test_remap_empty(self, resolver):
        assert resolver.remap_dict({}) == {}

    def test_remap_no_duplicates(self, resolver):
        """If two game IDs map to the same Torvik ID, first one wins."""
        ff = {
            "michigan_wolverines": {"to_rate": 0.15},
            "michigan": {"to_rate": 0.20},
        }
        result = resolver.remap_dict(ff)
        assert "michigan" in result


class TestNoMatch:
    def test_completely_unknown(self, resolver):
        assert resolver.resolve("xyz_nonexistent_team") is None

    def test_d2_team(self, resolver):
        """D2 teams not in Torvik set should return None."""
        assert resolver.resolve("adams_state_grizzlies") is None
