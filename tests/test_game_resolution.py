"""Tests for historical game ID → metric ID resolution.

The historical data pipeline produces two files per season:
  - historical_games_YYYY.json: game IDs use ESPN/CBBpy mascot-suffixed
    format (e.g. "duke_blue_devils", "uconn_huskies")
  - team_metrics_YYYY.json: metric IDs use Sports Reference canonical
    format (e.g. "duke", "connecticut")

Sports Reference also appends "NCAA" to tournament-qualifying teams
(e.g. "ArizonaNCAA" → team_id "arizonancaa").

These tests verify that:
1. The ingestion pipeline strips NCAA suffixes at scrape time
2. The downstream resolver handles both mascot suffixes and any
   residual NCAA suffixes in legacy data
3. Known problem cases (UConn, BYU, LSU, etc.) all resolve correctly
"""

import pytest

from src.data.team_name_resolver import TeamNameResolver


# ---------------------------------------------------------------------------
# 1. NCAA suffix stripping in ingestion (_ensure_team_ids)
# ---------------------------------------------------------------------------


class TestNCAASuffixStripping:
    """Verify that _ensure_team_ids strips NCAA suffixes from Sports Reference data."""

    @pytest.fixture
    def pipeline(self):
        """Minimal HistoricalDataPipeline for unit-testing _ensure_team_ids."""
        from src.data.ingestion.historical_pipeline import HistoricalDataPipeline
        return HistoricalDataPipeline()

    def test_strips_ncaa_from_team_name(self, pipeline):
        rows = [{"team_name": "ArizonaNCAA", "off_rtg": 100}]
        result = pipeline._ensure_team_ids(rows)
        assert result[0]["team_name"] == "Arizona"
        assert result[0]["name"] == "Arizona"

    def test_strips_ncaa_from_team_id(self, pipeline):
        rows = [{"team_name": "Baylor", "team_id": "baylorncaa", "off_rtg": 100}]
        result = pipeline._ensure_team_ids(rows)
        assert result[0]["team_id"] == "baylor"

    def test_strips_ncaa_parenthetical_team(self, pipeline):
        """Teams with parentheticals like Albany (NY)NCAA should be cleaned."""
        rows = [{"team_name": "Albany (NY)NCAA", "off_rtg": 100}]
        result = pipeline._ensure_team_ids(rows)
        assert result[0]["team_name"] == "Albany (NY)"
        assert "ncaa" not in result[0]["team_id"].lower()

    def test_preserves_non_ncaa_names(self, pipeline):
        rows = [{"team_name": "Duke", "team_id": "duke", "off_rtg": 100}]
        result = pipeline._ensure_team_ids(rows)
        assert result[0]["team_name"] == "Duke"
        assert result[0]["team_id"] == "duke"

    def test_no_ncaa_suffix_in_output(self, pipeline):
        """No team in the output should have an NCAA suffix."""
        rows = [
            {"team_name": "DukeNCAA", "off_rtg": 100},
            {"team_name": "KansasNCAA", "off_rtg": 110},
            {"team_name": "Gonzaga", "off_rtg": 105},
            {"team_name": "ConnecticutNCAA", "off_rtg": 108},
        ]
        result = pipeline._ensure_team_ids(rows)
        for row in result:
            assert not row["team_name"].endswith("NCAA"), (
                f"team_name '{row['team_name']}' still has NCAA suffix"
            )
            assert not row["team_id"].endswith("ncaa"), (
                f"team_id '{row['team_id']}' still has NCAA suffix"
            )


# ---------------------------------------------------------------------------
# 2. Mascot-suffixed game IDs resolve via TeamNameResolver
# ---------------------------------------------------------------------------


class TestMascotSuffixResolution:
    """Verify that mascot-suffixed game IDs can be resolved to canonical IDs.

    This tests the TeamNameResolver's ability to handle display names
    from ESPN/CBBpy (with mascots) and resolve them to the canonical
    form used by Sports Reference metrics.
    """

    @pytest.fixture
    def resolver(self):
        return TeamNameResolver()

    @pytest.mark.parametrize(
        "display_name,expected_id",
        [
            # Major conference teams — school name is a prefix of the display name,
            # so containment matching works directly.
            ("Duke Blue Devils", "duke"),
            ("North Carolina Tar Heels", "north_carolina"),
            ("Kentucky Wildcats", "kentucky"),
            ("Kansas Jayhawks", "kansas"),
            ("Alabama Crimson Tide", "alabama"),
            ("Michigan State Spartans", "michigan_state"),
            ("Gonzaga Bulldogs", "gonzaga"),
            ("Florida Atlantic Owls", "florida_atlantic"),
            ("Houston Cougars", "houston"),
            # Names with special characters
            ("Texas A&M Aggies", "texas_a_m"),
            ("Hawai'i Rainbow Warriors", "hawaii"),
        ],
    )
    def test_display_name_resolves(self, resolver, display_name, expected_id):
        """Display names where the school name is a substring resolve via containment."""
        result = resolver.resolve(display_name)
        assert result.canonical_id == expected_id, (
            f"'{display_name}' resolved to '{result.canonical_id}' "
            f"(expected '{expected_id}', method={result.method})"
        )
        assert result.confidence >= 0.80

    @pytest.mark.parametrize(
        "abbreviation,expected_id",
        [
            # Abbreviation-only lookups (without mascot) resolve correctly.
            # In the full pipeline, the CBBpy CSV maps display names like
            # "UConn Huskies" → location "UConn" → metric_id "connecticut"
            # before the resolver is ever called. These test the resolver's
            # ability to handle the abbreviation alone.
            ("UConn", "connecticut"),
            ("BYU", "brigham_young"),
            ("LSU", "louisiana_state"),
            ("USC", "southern_california"),
            ("SMU", "southern_methodist"),
            ("UCF", "ucf"),
            ("VCU", "virginia_commonwealth"),
            ("UNLV", "nevada_las_vegas"),
            ("UMBC", "maryland_baltimore_county"),
            ("UTEP", "utep"),
        ],
    )
    def test_abbreviation_resolves(self, resolver, abbreviation, expected_id):
        """Common abbreviations resolve correctly via alias table."""
        result = resolver.resolve(abbreviation)
        assert result.canonical_id == expected_id, (
            f"'{abbreviation}' resolved to '{result.canonical_id}' "
            f"(expected '{expected_id}', method={result.method})"
        )
        assert result.confidence >= 0.90


# ---------------------------------------------------------------------------
# 3. End-to-end resolution rate on actual historical data
# ---------------------------------------------------------------------------


class TestHistoricalResolutionRate:
    """Verify that historical game files achieve >95% resolution rate.

    This test reads actual data files and simulates the resolution
    pipeline from sota.py. It acts as a regression guard against
    future changes that could degrade resolution quality.
    """

    @staticmethod
    def _team_id(name: str) -> str:
        return "".join(
            c.lower() if c.isalnum() else "_" for c in name
        ).strip("_")

    def _compute_resolution_rate(self, year: int) -> float:
        """Compute the resolution rate for a single year.

        Returns the fraction of game team references that resolve to
        a valid metric key (0.0 to 1.0), or -1.0 if the year has
        insufficient metrics data.
        """
        import json
        import os

        from src.data.features.proprietary_metrics import _load_cbbpy_team_map

        games_dir = "data/raw/historical"
        gp = os.path.join(games_dir, f"historical_games_{year}.json")
        mp = os.path.join(games_dir, f"team_metrics_{year}.json")

        if not os.path.isfile(gp) or not os.path.isfile(mp):
            return -1.0

        with open(gp) as f:
            games_data = json.load(f)
        with open(mp) as f:
            metrics_data = json.load(f)

        # Build metric keys (with NCAA suffix stripping)
        team_metrics = set()
        for tm in metrics_data.get("teams", []):
            tid = self._team_id(str(tm.get("team_id") or tm.get("name", "")))
            off = float(tm.get("off_rtg", 0))
            drt = float(tm.get("def_rtg", 0))
            if not tid or off < 1e-6 or drt < 1e-6:
                continue
            team_metrics.add(tid)
            if tid.endswith("ncaa"):
                base = tid[:-4].rstrip("_")
                if base not in team_metrics:
                    team_metrics.add(base)

        # Not enough metrics data — skip
        if len(team_metrics) < 50:
            return -1.0

        metric_keys = sorted(team_metrics, key=len, reverse=True)

        # Build CBBpy display_name → metric_id lookup
        _LOCATION_TO_METRIC = {
            "american university": "american",
            "app state": "appalachian_state",
            "byu": "brigham_young",
            "central connecticut": "central_connecticut_state",
            "charleston": "college_of_charleston",
            "hawai'i": "hawaii",
            "iu indianapolis": "iu_indy",
            "loyola chicago": "loyola__il",
            "loyola maryland": "loyola__md",
            "lsu": "louisiana_state",
            "mcneese": "mcneese_state",
            "miami": "miami__fl",
            "nicholls": "nicholls_state",
            "ole miss": "mississippi",
            "prairie view a&m": "prairie_view",
            "queens university": "queens__nc",
            "saint mary's": "saint_mary_s__ca",
            "san jose state": "san_jose_state",
            "san josé state": "san_jose_state",
            "se louisiana": "southeastern_louisiana",
            "seattle u": "seattle",
            "smu": "southern_methodist",
            "southern miss": "southern_mississippi",
            "st. francis brooklyn": "st__francis__ny",
            "st. john's": "st__john_s__ny",
            "ualbany": "albany__ny",
            "uconn": "connecticut",
            "uic": "illinois_chicago",
            "ul monroe": "louisiana_monroe",
            "umass lowell": "massachusetts_lowell",
            "umbc": "maryland_baltimore_county",
            "unlv": "nevada_las_vegas",
            "usc": "southern_california",
            "ut martin": "tennessee_martin",
            "ut rio grande valley": "texas_rio_grande_valley",
            "vcu": "virginia_commonwealth",
            "fairleigh dickinson": "fairleigh_dickinson",
        }

        cbbpy_map = _load_cbbpy_team_map()
        display_to_metric = {}
        for dn, loc in cbbpy_map.items():
            ll = loc.lower().strip()
            mid = _LOCATION_TO_METRIC.get(ll)
            if mid and mid in team_metrics:
                display_to_metric[dn] = mid
                continue
            lid = self._team_id(loc)
            if lid in team_metrics:
                display_to_metric[dn] = lid

        # Pre-scan games
        resolver = TeamNameResolver()
        games = games_data.get("games", [])
        gid2disp = {}
        for g in games:
            for s in ["1", "2"]:
                raw = self._team_id(
                    str(g.get(f"team{s}_id") or g.get(f"team{s}") or "")
                )
                nm = str(g.get(f"team{s}_name") or "")
                if raw and nm and raw not in gid2disp:
                    gid2disp[raw] = nm

        cache = {}
        for gid, dn in gid2disp.items():
            if gid in team_metrics:
                cache[gid] = gid
            elif dn in display_to_metric:
                cache[gid] = display_to_metric[dn]

        def _resolve(gid):
            if gid in cache:
                return cache[gid]
            if gid in team_metrics:
                cache[gid] = gid
                return gid
            for mk in metric_keys:
                if gid.startswith(mk + "_") or gid.startswith(mk):
                    cache[gid] = mk
                    return mk
            dn = gid2disp.get(gid, "")
            for c in [dn, gid.replace("_", " ")]:
                if not c:
                    continue
                r = resolver.resolve(c)
                if r.method != "unresolved" and r.confidence >= 0.80:
                    if r.canonical_id in team_metrics:
                        cache[gid] = r.canonical_id
                        return r.canonical_id
            cache[gid] = None
            return None

        total = 0
        resolved = 0
        for g in games:
            for s in ["1", "2"]:
                raw = self._team_id(
                    str(g.get(f"team{s}_id") or g.get(f"team{s}") or "")
                )
                if not raw:
                    continue
                total += 1
                r = _resolve(raw)
                if r and r in team_metrics:
                    resolved += 1

        return resolved / total if total > 0 else 0.0

    @pytest.mark.parametrize("year", list(range(2010, 2026)))
    def test_resolution_rate_above_95_percent(self, year):
        """Each valid year must achieve >95% game→metric resolution."""
        rate = self._compute_resolution_rate(year)
        if rate < 0:
            pytest.skip(f"Year {year}: insufficient metrics data")
        assert rate >= 0.95, (
            f"Year {year}: resolution rate {rate:.1%} is below 95% threshold"
        )
