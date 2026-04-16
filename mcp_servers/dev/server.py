"""
Dev MCP server — tools backed by local filesystem and live HTTP scrapers.

Backend priority for each tool:
  1. Read from the project's data/raw/ cache (fast, no network)
  2. Fall back to the scraper if the cache file is absent

Run:
    python3 -m mcp_servers.dev.server
    python3 mcp_servers/dev/server.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from mcp_servers.shared.schemas import SAFE_CLI_COMMANDS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_ROOT = PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

mcp = FastMCP("march-madness-dev")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def _torvik_cache(year: int) -> Path:
    return DATA_RAW / f"torvik_{year}.json"


def _picks_cache(year: int) -> Path:
    return DATA_RAW / f"public_picks_{year}.json"


def _bracket_cache(year: int) -> Path:
    return DATA_RAW / f"bracket_{year}.json"


def _odds_cache(year: int) -> Path:
    return DATA_RAW / "betting_odds" / f"theoddsapi_{year}.json"


def _forecast_path() -> Path:
    return DATA_ROOT / "forecaster_output.json"


def _seed_map(year: int) -> dict[str, int]:
    """Return {team_id: seed} from bracket file, empty dict if absent."""
    p = _bracket_cache(year)
    if not p.exists():
        return {}
    data = _load_json(p)
    teams = data.get("teams", [])
    return {t["team_id"]: t["seed"] for t in teams if "seed" in t and "team_id" in t}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def get_team_ratings(year: int, team_id: str | None = None) -> dict:
    """
    Return Torvik efficiency metrics for all tournament teams in a given year.

    Reads from data/raw/torvik_{year}.json (scraper cache). Falls back to
    the live BartTorvikScraper if the cache is absent.

    Args:
        year:    Season year (e.g. 2026).
        team_id: Optional team slug (e.g. "duke"). When provided, returns a
                 single-entry list filtered to that team.

    Returns:
        {"year": int, "teams": [TeamRating, ...]}
    """
    cache = _torvik_cache(year)

    if cache.exists():
        raw = _load_json(cache)
        raw_teams = raw.get("teams", [])
    else:
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.data.scrapers.torvik import BartTorvikScraper

        scraper = BartTorvikScraper(cache_dir=str(DATA_RAW))
        torvik_teams = scraper.fetch_current_rankings(year)
        raw_teams = [t.to_dict() for t in torvik_teams]

    seeds = _seed_map(year)

    teams = [
        {
            "team_id": t.get("team_id", ""),
            "team_name": t.get("team_name") or t.get("name", ""),
            "adj_off_eff": t.get("adj_offensive_efficiency", 0.0),
            "adj_def_eff": t.get("adj_defensive_efficiency", 0.0),
            "adj_tempo": t.get("adj_tempo", 0.0),
            "barthag": t.get("barthag", 0.0),
            "seed": seeds.get(t.get("team_id", "")),
            "conference": t.get("conference", ""),
        }
        for t in raw_teams
    ]

    if team_id is not None:
        teams = [t for t in teams if t["team_id"] == team_id]

    return {"year": year, "teams": teams}


@mcp.tool()
def get_public_picks(year: int) -> dict:
    """
    Return ESPN public pick percentages for every team in a given year.

    Reads from data/raw/public_picks_{year}.json. Falls back to a live
    ESPNPicksScraper call if the cache is absent.

    Args:
        year: Season year (e.g. 2026).

    Returns:
        {"year": int, "teams": {team_id: PublicPickEntry, ...}}
    """
    cache = _picks_cache(year)

    if cache.exists():
        raw = _load_json(cache)
        raw_teams: dict = raw.get("teams", {})
    else:
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.data.scrapers.espn_picks import ESPNPicksScraper

        scraper = ESPNPicksScraper(cache_dir=str(DATA_RAW))
        consensus = scraper.fetch_picks(year)
        raw_teams = {
            pick.team_id: {
                "team_name": pick.team_name,
                "seed": pick.seed,
                "region": pick.region,
                "round_of_64_pct": pick.round_of_64_pct,
                "round_of_32_pct": pick.round_of_32_pct,
                "sweet_16_pct": pick.sweet_16_pct,
                "elite_8_pct": pick.elite_8_pct,
                "final_four_pct": pick.final_four_pct,
                "champion_pct": pick.champion_pct,
            }
            for pick in consensus.picks.values()
        }

    teams = {
        tid: {
            "team_id": tid,
            "team_name": entry.get("team_name", ""),
            "seed": entry.get("seed", 0),
            "region": entry.get("region", ""),
            "R64": entry.get("round_of_64_pct", 0.0),
            "R32": entry.get("round_of_32_pct", 0.0),
            "S16": entry.get("sweet_16_pct", 0.0),
            "E8": entry.get("elite_8_pct", 0.0),
            "F4": entry.get("final_four_pct", 0.0),
            "CHAMP": entry.get("champion_pct", 0.0),
        }
        for tid, entry in raw_teams.items()
    }

    return {"year": year, "teams": teams}


@mcp.tool()
def get_bracket(year: int) -> dict:
    """
    Return the tournament bracket structure for a given year.

    Reads data/raw/bracket_{year}.json directly. Returns the full bracket
    including seeds, regions, first-four games, and resolution warnings.

    Args:
        year: Season year (e.g. 2026).

    Returns:
        Raw bracket dict with keys: season, teams, first_four,
        resolution_warnings.
    """
    p = _bracket_cache(year)
    if not p.exists():
        return {"error": f"No bracket file found for {year}", "path": str(p)}

    data = _load_json(p)
    teams = [
        {
            "team_id": t.get("team_id", ""),
            "team_name": t.get("team_name", ""),
            "seed": t.get("seed"),
            "region": t.get("region", ""),
            "conference": t.get("conference", ""),
            "school_slug": t.get("school_slug", ""),
        }
        for t in data.get("teams", [])
    ]
    return {
        "year": year,
        "season": data.get("season"),
        "source": data.get("source", ""),
        "teams": teams,
        "first_four": data.get("first_four", []),
        "resolution_warnings": data.get("resolution_warnings", []),
    }


@mcp.tool()
def get_market_odds(year: int) -> dict:
    """
    Return betting market implied championship probabilities for a given year.

    Reads from data/raw/betting_odds/theoddsapi_{year}.json if present.
    Live scraping requires ODDS_API_KEY env var and is only attempted when
    no cache exists.

    Args:
        year: Season year (e.g. 2026).

    Returns:
        {"year": int, "teams": {team_id: MarketOddsEntry, ...}}
    """
    cache = _odds_cache(year)

    if cache.exists():
        raw = _load_json(cache)
        # Cache may be stored as list or dict keyed by team_id
        if isinstance(raw, list):
            entries = raw
        else:
            entries = list(raw.values())
    else:
        api_key = os.environ.get("ODDS_API_KEY", "")
        if not api_key:
            return {
                "year": year,
                "teams": {},
                "warning": (
                    f"No cache at {cache} and ODDS_API_KEY env var not set. Set ODDS_API_KEY to fetch live odds."
                ),
            }
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.data.scrapers.betting_markets import TheOddsAPIScraper

        scraper = TheOddsAPIScraper(api_key=api_key, cache_dir=str(DATA_RAW / "betting_odds"))
        raw_odds = scraper.scrape(year)
        entries = [vars(v) for v in raw_odds.values()]

    teams = {}
    for entry in entries:
        if isinstance(entry, dict):
            tid = entry.get("team_id", "")
            teams[tid] = {
                "team_id": tid,
                "team_name": entry.get("team_name", ""),
                "source": entry.get("source", "theoddsapi"),
                "championship_odds": entry.get("championship_odds", 0.0),
                "implied_probability": entry.get("implied_probability", 0.0),
            }

    return {"year": year, "teams": teams}


@mcp.tool()
def get_forecast(year: int) -> dict:
    """
    Return model matchup win probabilities for a given year.

    Reads data/forecaster_output.json and returns the predictions_{year}
    sub-dict plus top-level metadata (calibration_method, metrics).

    Args:
        year: Season year (e.g. 2026).

    Returns:
        {"year": int, "predictions": {"team_a_team_b": prob, ...},
         "calibration_method": str, "metrics": {...}}
    """
    p = _forecast_path()
    if not p.exists():
        return {"error": "forecaster_output.json not found", "path": str(p)}

    data = _load_json(p)
    predictions = data.get(f"predictions_{year}", {})
    return {
        "year": year,
        "predictions": predictions,
        "calibration_method": data.get("calibration_method", ""),
        "metrics": data.get("metrics", {}),
        "status": data.get("status", ""),
    }


@mcp.tool()
def list_data(year: int) -> dict:
    """
    List available raw data files for a given year.

    Scans data/raw/ for files whose names contain the year, and checks for
    the canonical files used by the pipeline.

    Args:
        year: Season year (e.g. 2026).

    Returns:
        {"year": int, "files": [...], "torvik": bool, "bracket": bool,
         "public_picks": bool, "advanced_metrics": bool, "forecast": bool}
    """
    year_str = str(year)
    files = sorted(p.name for p in DATA_RAW.iterdir() if p.is_file() and year_str in p.name)

    return {
        "year": year,
        "files": files,
        "torvik": _torvik_cache(year).exists(),
        "bracket": _bracket_cache(year).exists(),
        "public_picks": _picks_cache(year).exists(),
        "advanced_metrics": (DATA_RAW / f"advanced_metrics_{year}.json").exists(),
        "forecast": _forecast_path().exists(),
    }


@mcp.tool()
def run_cli(command: str) -> dict:
    """
    Run a read-only project CLI command and return its output.

    Allowed subcommands: status, pool-report, eval.
    Anything else is rejected to prevent pipeline mutations from MCP.

    Args:
        command: Subcommand and arguments, e.g. "status" or "pool-report --year 2026".

    Returns:
        {"returncode": int, "stdout": str, "stderr": str}
    """
    parts = command.strip().split()
    if not parts:
        return {"error": "Empty command"}

    subcmd = parts[0]
    if subcmd not in SAFE_CLI_COMMANDS:
        return {"error": (f"'{subcmd}' is not an allowed command. Permitted: {sorted(SAFE_CLI_COMMANDS)}")}

    result = subprocess.run(
        [sys.executable, "-m", "src"] + parts,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=120,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
