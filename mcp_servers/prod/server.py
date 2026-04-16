"""
Prod MCP server — tools backed by a REST API and a local SQLite cache.

Exposes identical tool names and signatures to mcp_servers.dev.server.
Operators must set the following env vars before deploying:

    DATA_API_URL   Base URL for the data API  (e.g. https://api.example.com/v1)
    ODDS_API_URL   Base URL for odds API      (defaults to TheOddsAPI v4)
    ODDS_API_KEY   API key for odds endpoint
    DB_PATH        Path to SQLite cache db    (default: data/prod.db)

Tools gracefully return an error dict when required env vars are absent
so the server starts cleanly with no configuration.

Run:
    python3 -m mcp_servers.prod.server
    python3 mcp_servers/prod/server.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx
from mcp.server.fastmcp import FastMCP

from mcp_servers.shared.schemas import SAFE_CLI_COMMANDS

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_API_URL: str = os.environ.get("DATA_API_URL", "")
ODDS_API_URL: str = os.environ.get("ODDS_API_URL", "https://api.the-odds-api.com/v4")
ODDS_API_KEY: str = os.environ.get("ODDS_API_KEY", "")
DB_PATH: Path = Path(os.environ.get("DB_PATH", str(PROJECT_ROOT / "data" / "prod.db")))

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

mcp = FastMCP("march-madness-prod")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def _db():
    """Open a short-lived SQLite connection, creating tables on first use."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        _ensure_schema(conn)
        yield conn
    finally:
        conn.close()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS team_ratings (
            year        INTEGER NOT NULL,
            team_id     TEXT    NOT NULL,
            payload     TEXT    NOT NULL,
            PRIMARY KEY (year, team_id)
        );
        CREATE TABLE IF NOT EXISTS public_picks (
            year        INTEGER NOT NULL,
            team_id     TEXT    NOT NULL,
            payload     TEXT    NOT NULL,
            PRIMARY KEY (year, team_id)
        );
        CREATE TABLE IF NOT EXISTS brackets (
            year        INTEGER PRIMARY KEY,
            payload     TEXT    NOT NULL
        );
        CREATE TABLE IF NOT EXISTS market_odds (
            year        INTEGER NOT NULL,
            team_id     TEXT    NOT NULL,
            payload     TEXT    NOT NULL,
            PRIMARY KEY (year, team_id)
        );
        CREATE TABLE IF NOT EXISTS forecasts (
            year        INTEGER PRIMARY KEY,
            payload     TEXT    NOT NULL
        );
    """)
    conn.commit()


def _api_get(path: str, params: dict | None = None) -> Any:
    """GET from DATA_API_URL, raise if not configured."""
    if not DATA_API_URL:
        raise RuntimeError("DATA_API_URL env var is not set. Configure it to point at your production data API.")
    url = urljoin(DATA_API_URL.rstrip("/") + "/", path.lstrip("/"))
    resp = httpx.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _cache_rows(table: str, year: int, conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(f"SELECT payload FROM {table} WHERE year = ?", (year,)).fetchall()
    return [json.loads(r["payload"]) for r in rows]


def _upsert_rows(table: str, year: int, rows: list[dict], key: str, conn: sqlite3.Connection) -> None:
    conn.executemany(
        f"INSERT OR REPLACE INTO {table} (year, team_id, payload) VALUES (?, ?, ?)",
        [(year, row[key], json.dumps(row)) for row in rows],
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Tools  (identical signatures to dev/server.py)
# ---------------------------------------------------------------------------


@mcp.tool()
def get_team_ratings(year: int, team_id: str | None = None) -> dict:
    """
    Return Torvik efficiency metrics for all tournament teams in a given year.

    Reads from SQLite cache (team_ratings table); on cache miss, fetches from
    DATA_API_URL/ratings/{year} and populates the cache.

    Args:
        year:    Season year (e.g. 2026).
        team_id: Optional team slug. When provided, returns a single-entry list.

    Returns:
        {"year": int, "teams": [TeamRating, ...]}
    """
    try:
        with _db() as conn:
            cached = _cache_rows("team_ratings", year, conn)
            if not cached:
                data = _api_get(f"ratings/{year}")
                teams_raw = data if isinstance(data, list) else data.get("teams", [])
                _upsert_rows("team_ratings", year, teams_raw, "team_id", conn)
                cached = teams_raw

        teams = cached
        if team_id is not None:
            teams = [t for t in teams if t.get("team_id") == team_id]
        return {"year": year, "teams": teams}

    except Exception as exc:
        return {"error": str(exc), "year": year, "teams": []}


@mcp.tool()
def get_public_picks(year: int) -> dict:
    """
    Return ESPN public pick percentages for every team in a given year.

    Reads from SQLite cache (public_picks table); on cache miss, fetches from
    DATA_API_URL/picks/{year}.

    Args:
        year: Season year (e.g. 2026).

    Returns:
        {"year": int, "teams": {team_id: PublicPickEntry, ...}}
    """
    try:
        with _db() as conn:
            cached = _cache_rows("public_picks", year, conn)
            if not cached:
                data = _api_get(f"picks/{year}")
                entries = data if isinstance(data, list) else list(data.get("teams", {}).values())
                _upsert_rows("public_picks", year, entries, "team_id", conn)
                cached = entries

        return {"year": year, "teams": {t["team_id"]: t for t in cached}}

    except Exception as exc:
        return {"error": str(exc), "year": year, "teams": {}}


@mcp.tool()
def get_bracket(year: int) -> dict:
    """
    Return the tournament bracket structure for a given year.

    Reads from SQLite cache (brackets table, single row per year); on cache
    miss, fetches from DATA_API_URL/bracket/{year}.

    Args:
        year: Season year (e.g. 2026).

    Returns:
        Bracket dict with keys: season, teams, first_four, resolution_warnings.
    """
    try:
        with _db() as conn:
            row = conn.execute("SELECT payload FROM brackets WHERE year = ?", (year,)).fetchone()

            if row is None:
                data = _api_get(f"bracket/{year}")
                conn.execute(
                    "INSERT OR REPLACE INTO brackets (year, payload) VALUES (?, ?)",
                    (year, json.dumps(data)),
                )
                conn.commit()
            else:
                data = json.loads(row["payload"])

        return data

    except Exception as exc:
        return {"error": str(exc), "year": year}


@mcp.tool()
def get_market_odds(year: int) -> dict:
    """
    Return betting market implied championship probabilities for a given year.

    Reads from SQLite cache (market_odds table); on cache miss, queries
    TheOddsAPI using ODDS_API_KEY.

    Args:
        year: Season year (e.g. 2026).

    Returns:
        {"year": int, "teams": {team_id: MarketOddsEntry, ...}}
    """
    try:
        with _db() as conn:
            cached = _cache_rows("market_odds", year, conn)
            if not cached:
                if not ODDS_API_KEY:
                    return {
                        "year": year,
                        "teams": {},
                        "warning": "ODDS_API_KEY env var is not set.",
                    }
                url = f"{ODDS_API_URL}/sports/basketball_ncaab/odds"
                resp = httpx.get(
                    url,
                    params={
                        "apiKey": ODDS_API_KEY,
                        "markets": "h2h",
                        "regions": "us",
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                raw = resp.json()
                # Flatten sportsbook entries to per-team MarketOddsEntry shape
                entries = _parse_odds_response(raw, year)
                _upsert_rows("market_odds", year, entries, "team_id", conn)
                cached = entries

        return {"year": year, "teams": {t["team_id"]: t for t in cached}}

    except Exception as exc:
        return {"error": str(exc), "year": year, "teams": {}}


def _parse_odds_response(raw: list, year: int) -> list[dict]:
    """Convert TheOddsAPI response to MarketOddsEntry list."""
    seen: dict[str, dict] = {}
    for game in raw:
        for bm in game.get("bookmakers", []):
            for market in bm.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name", "")
                    price = outcome.get("price", 0.0)
                    tid = name.lower().replace(" ", "_")
                    if tid not in seen:
                        # Convert decimal odds → implied prob
                        prob = 1 / price if price > 0 else 0.0
                        seen[tid] = {
                            "team_id": tid,
                            "team_name": name,
                            "source": bm.get("key", "unknown"),
                            "championship_odds": round((price - 1) * 100, 1),
                            "implied_probability": round(prob, 4),
                        }
    return list(seen.values())


@mcp.tool()
def get_forecast(year: int) -> dict:
    """
    Return model matchup win probabilities for a given year.

    Reads from SQLite cache (forecasts table); on cache miss, fetches from
    DATA_API_URL/forecast/{year}.

    Args:
        year: Season year (e.g. 2026).

    Returns:
        {"year": int, "predictions": {"team_a_team_b": prob, ...},
         "calibration_method": str, "metrics": {...}}
    """
    try:
        with _db() as conn:
            row = conn.execute("SELECT payload FROM forecasts WHERE year = ?", (year,)).fetchone()

            if row is None:
                data = _api_get(f"forecast/{year}")
                conn.execute(
                    "INSERT OR REPLACE INTO forecasts (year, payload) VALUES (?, ?)",
                    (year, json.dumps(data)),
                )
                conn.commit()
            else:
                data = json.loads(row["payload"])

        return data

    except Exception as exc:
        return {"error": str(exc), "year": year}


@mcp.tool()
def list_data(year: int) -> dict:
    """
    List datasets available from the API for a given year.

    Queries DATA_API_URL/datasets?year={year}. Falls back to checking the
    SQLite cache tables when the API is unreachable.

    Args:
        year: Season year (e.g. 2026).

    Returns:
        {"year": int, "files": [...], "torvik": bool, "bracket": bool,
         "public_picks": bool, "forecast": bool}
    """
    try:
        try:
            data = _api_get("datasets", params={"year": year})
            return {"year": year, **data}
        except RuntimeError:
            pass  # DATA_API_URL not configured — fall through to DB check

        with _db() as conn:
            has_ratings = bool(_cache_rows("team_ratings", year, conn))
            has_picks = bool(_cache_rows("public_picks", year, conn))
            has_bracket = bool(conn.execute("SELECT 1 FROM brackets WHERE year = ?", (year,)).fetchone())
            has_forecast = bool(conn.execute("SELECT 1 FROM forecasts WHERE year = ?", (year,)).fetchone())

        return {
            "year": year,
            "files": [],
            "torvik": has_ratings,
            "bracket": has_bracket,
            "public_picks": has_picks,
            "advanced_metrics": has_ratings,
            "forecast": has_forecast,
            "source": "sqlite_cache",
        }

    except Exception as exc:
        return {"error": str(exc), "year": year}


@mcp.tool()
def run_cli(command: str) -> dict:
    """
    Run a read-only project CLI command and return its output.

    Allowed subcommands: status, pool-report, eval.

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
