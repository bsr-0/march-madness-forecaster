"""Unit tests for tournament seed parsing."""

from src.data.scrapers.tournament_bracket import TournamentSeedScraper


def test_parse_seed_teams_from_bracket_html():
    html = """
    <div id="brackets">
      <div id="east">
        <div class="round">
          <div>
            <span>1</span>
            <a href="/cbb/schools/duke/men/2025.html">Duke</a>
            <a href="/cbb/boxscores/2025-03-21-14-duke.html">93</a>
          </div>
          <div>
            <span>16</span>
            <a href="/cbb/schools/mount-st-marys/men/2025.html">Mount St. Mary's</a>
            <a href="/cbb/boxscores/2025-03-21-14-duke.html">49</a>
          </div>
        </div>
      </div>
    </div>
    """
    scraper = TournamentSeedScraper()
    teams = scraper._parse_seed_teams(html, 2025)

    assert len(teams) == 2
    assert teams[0]["region"] == "East"
    assert {t["school_slug"] for t in teams} == {"duke", "mount-st-marys"}
    assert {t["seed"] for t in teams} == {1, 16}
