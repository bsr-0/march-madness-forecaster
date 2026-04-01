"""Tests for ESPN 'Who Picked Whom' HTML scraper."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data.scrapers.espn_picks import ESPNPicksScraper


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def script_embed_html():
    return (FIXTURES / "espn_wpw_script_embed.html").read_text()


@pytest.fixture
def table_html():
    return (FIXTURES / "espn_wpw_table.html").read_text()


class TestParseWpwHtmlScriptEmbed:
    """Test parsing of __NEXT_DATA__ script-embedded JSON format."""

    def test_extracts_all_16_teams(self, script_embed_html):
        teams = ESPNPicksScraper.parse_wpw_html(script_embed_html)
        assert teams is not None
        assert len(teams) == 16

    def test_duke_round_values(self, script_embed_html):
        teams = ESPNPicksScraper.parse_wpw_html(script_embed_html)
        duke = next(t for t in teams if t["name"] == "Duke")
        assert duke["seed"] == 1
        assert duke["region"] == "East"
        assert duke["round_of_64_pct"] == 98.7
        assert duke["round_of_32_pct"] == 92.1
        assert duke["sweet_16_pct"] == 78.4
        assert duke["elite_8_pct"] == 55.2
        assert duke["final_four_pct"] == 38.6
        assert duke["champion_pct"] == 22.3

    def test_16_seed_low_values(self, script_embed_html):
        teams = ESPNPicksScraper.parse_wpw_html(script_embed_html)
        norfolk = next(t for t in teams if t["name"] == "Norfolk State")
        assert norfolk["seed"] == 16
        assert norfolk["round_of_64_pct"] == 1.3
        assert norfolk["champion_pct"] == 0.0002

    def test_rates_decrease_by_round(self, script_embed_html):
        teams = ESPNPicksScraper.parse_wpw_html(script_embed_html)
        round_keys = [
            "round_of_64_pct", "round_of_32_pct", "sweet_16_pct",
            "elite_8_pct", "final_four_pct", "champion_pct",
        ]
        for team in teams:
            for i in range(len(round_keys) - 1):
                assert team[round_keys[i]] >= team[round_keys[i + 1]], (
                    f"{team['name']}: {round_keys[i]}={team[round_keys[i]]} "
                    f"< {round_keys[i+1]}={team[round_keys[i+1]]}"
                )


class TestParseWpwHtmlTable:
    """Test parsing of HTML table format."""

    def test_extracts_6_teams(self, table_html):
        teams = ESPNPicksScraper.parse_wpw_html(table_html)
        assert teams is not None
        assert len(teams) == 6

    def test_duke_values_from_table(self, table_html):
        teams = ESPNPicksScraper.parse_wpw_html(table_html)
        duke = next(t for t in teams if t["name"] == "Duke")
        assert duke["seed"] == 1
        assert duke["region"] == "East"
        assert duke["round_of_64_pct"] == 98.7
        assert duke["champion_pct"] == 22.3

    def test_oregon_12_seed(self, table_html):
        teams = ESPNPicksScraper.parse_wpw_html(table_html)
        oregon = next(t for t in teams if t["name"] == "Oregon")
        assert oregon["seed"] == 12
        assert oregon["round_of_64_pct"] == 34.2
        assert oregon["champion_pct"] == 0.03

    def test_pct_sign_stripped(self, table_html):
        """Percentage signs in HTML cells should be stripped correctly."""
        teams = ESPNPicksScraper.parse_wpw_html(table_html)
        for team in teams:
            # All values should be plain floats, not strings
            assert isinstance(team["round_of_64_pct"], float)
            assert isinstance(team["champion_pct"], float)


class TestParseWpwHtmlEdgeCases:
    """Edge cases for parse_wpw_html."""

    def test_empty_html_returns_none(self):
        assert ESPNPicksScraper.parse_wpw_html("") is None

    def test_no_data_returns_none(self):
        html = "<html><body><p>No pick data here</p></body></html>"
        assert ESPNPicksScraper.parse_wpw_html(html) is None

    def test_malformed_json_in_script_returns_none(self):
        html = """
        <html><body>
        <script id="__NEXT_DATA__">not valid json{{{</script>
        </body></html>
        """
        assert ESPNPicksScraper.parse_wpw_html(html) is None

    def test_empty_teams_list_returns_none(self):
        html = """
        <html><body>
        <script id="__NEXT_DATA__" type="application/json">
        {"props":{"pageProps":{"picks":{"teams":[]}}}}
        </script>
        </body></html>
        """
        assert ESPNPicksScraper.parse_wpw_html(html) is None


class TestWpwTeamsToDict:
    """Test conversion from parsed teams to standard dict format."""

    def test_creates_team_ids(self, script_embed_html):
        teams = ESPNPicksScraper.parse_wpw_html(script_embed_html)
        payload = ESPNPicksScraper._wpw_teams_to_dict(teams)
        assert "duke" in payload["teams"]
        assert "connecticut" in payload["teams"]
        assert "michigan_state" in payload["teams"]
        assert "st_johns" in payload["teams"]

    def test_preserves_values(self, script_embed_html):
        teams = ESPNPicksScraper.parse_wpw_html(script_embed_html)
        payload = ESPNPicksScraper._wpw_teams_to_dict(teams)
        duke = payload["teams"]["duke"]
        assert duke["team_name"] == "Duke"
        assert duke["seed"] == 1
        assert duke["champion_pct"] == 22.3

    def test_has_sources_key(self, script_embed_html):
        teams = ESPNPicksScraper.parse_wpw_html(script_embed_html)
        payload = ESPNPicksScraper._wpw_teams_to_dict(teams)
        assert "espn_wpw" in payload["sources"]


class TestTryHtmlPageIntegration:
    """Integration test: _try_html_page is deprecated; JSON extraction works."""

    def test_try_html_page_deprecated_returns_none(self, script_embed_html):
        """_try_html_page is deprecated and always returns None."""
        scraper = ESPNPicksScraper()
        result = scraper._try_html_page(2026)
        assert result is None

    def test_script_embed_extracted_via_json(self, script_embed_html):
        """_extract_json_from_html extracts teams from <script> embeds."""
        scraper = ESPNPicksScraper()
        teams = scraper._extract_json_from_html(script_embed_html)
        assert teams is not None
        assert len(teams) > 0
        duke_teams = [t for t in teams if "duke" in t.get("name", "").lower()]
        assert len(duke_teams) >= 1
        assert duke_teams[0]["champion_pct"] == 22.3
        assert duke_teams[0]["seed"] == 1

    def test_try_json_url_handles_html_with_embedded_json(self, script_embed_html):
        """_try_json_url extracts JSON from HTML responses."""
        scraper = ESPNPicksScraper()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = script_embed_html
        mock_resp.headers = {"content-type": "text/html"}
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(side_effect=ValueError("not json"))

        with patch.object(scraper.session, "get", return_value=mock_resp):
            result = scraper._try_json_url("http://test", 2026, "test")

        assert result is not None
        assert len(result.teams) > 0

    def test_403_returns_none(self):
        scraper = ESPNPicksScraper()
        mock_resp = MagicMock()
        mock_resp.status_code = 403

        with patch.object(scraper.session, "get", return_value=mock_resp):
            result = scraper._try_html_page(2026)

        assert result is None

    def test_network_error_returns_none(self):
        scraper = ESPNPicksScraper()
        with patch.object(
            scraper.session, "get", side_effect=ConnectionError("no network")
        ):
            result = scraper._try_html_page(2026)

        assert result is None

    def test_caches_successful_result(self, script_embed_html, tmp_path):
        scraper = ESPNPicksScraper(cache_dir=str(tmp_path))
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = script_embed_html
        mock_resp.headers = {"content-type": "text/html"}
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(side_effect=ValueError("not json"))

        with patch.object(scraper.session, "get", return_value=mock_resp):
            result = scraper._try_json_url("http://test", 2026, "test")

        assert result is not None
        cache_file = tmp_path / "espn_picks_2026.json"
        assert cache_file.exists()
        cached = json.loads(cache_file.read_text())
        assert "duke" in cached["teams"]
