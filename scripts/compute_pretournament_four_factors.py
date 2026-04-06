#!/usr/bin/env python3
"""Compute pre-tournament four factors from game-level box score data.

The existing torvik_four_factors_{year}.json files were scraped from
barttorvik.com without date filtering, so they include tournament game
results — a look-ahead bias.

This script computes offensive and defensive four factors from the
team_games box score data in historical_games_{year}.json, using only
games played BEFORE the tournament start date.

Four Factors (per Dean Oliver):
  Offense: eFG%, TO%, ORB%, FTR
  Defense: opp_eFG%, opp_TO% (forced), DRB%, opp_FTR
"""

import json
import logging
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HIST_DIR = ROOT / "data" / "raw" / "historical"

TOURNAMENT_START_DATES = {
    2008: date(2008, 3, 18),
    2009: date(2009, 3, 17),
    2010: date(2010, 3, 16),
    2011: date(2011, 3, 15),
    2012: date(2012, 3, 13),
    2013: date(2013, 3, 19),
    2014: date(2014, 3, 18),
    2015: date(2015, 3, 17),
    2016: date(2016, 3, 15),
    2017: date(2017, 3, 14),
    2018: date(2018, 3, 13),
    2019: date(2019, 3, 19),
    2021: date(2021, 3, 18),
    2022: date(2022, 3, 15),
    2023: date(2023, 3, 14),
    2024: date(2024, 3, 19),
    2025: date(2025, 3, 18),
    2026: date(2026, 3, 17),
}


def _safe_div(num, denom, default=0.0):
    return num / denom if denom > 0 else default


def _build_id_map(year):
    """Build game-data ID to canonical ID mapping (same as barthag script)."""
    try:
        from scripts.compute_pretournament_barthag import _build_game_to_canonical_map

        return _build_game_to_canonical_map(year)
    except ImportError:
        return {}


def compute_four_factors_for_year(year):
    """Compute pre-tournament four factors from box score data."""
    games_path = HIST_DIR / f"historical_games_{year}.json"
    if not games_path.exists():
        logger.warning("No game data for %d", year)
        return None

    with open(games_path) as f:
        data = json.load(f)

    tourney_start = TOURNAMENT_START_DATES.get(year)
    if not tourney_start:
        logger.warning("No tournament start date for %d", year)
        return None

    team_games = data.get("team_games", [])
    if not team_games:
        logger.warning("No team_games box score data for %d", year)
        return None

    id_map = _build_id_map(year)

    # Accumulate stats per team (pre-tournament only)
    stats = defaultdict(
        lambda: {
            "fgm": 0,
            "fga": 0,
            "fg3m": 0,
            "fg3a": 0,
            "fta": 0,
            "turnovers": 0,
            "orb": 0,
            "drb": 0,
            "possessions": 0,
            "games": 0,
            # Opponent stats (for defensive four factors)
            "opp_fgm": 0,
            "opp_fga": 0,
            "opp_fg3m": 0,
            "opp_fg3a": 0,
            "opp_fta": 0,
            "opp_turnovers": 0,
            "opp_orb": 0,
            "opp_drb": 0,
        }
    )

    # Build a lookup of game_id -> opponent's box score for defensive stats
    game_teams = defaultdict(list)
    pre_tourney_games = []

    for tg in team_games:
        game_date_str = tg.get("date", "")
        if not game_date_str:
            continue
        try:
            gd = date.fromisoformat(game_date_str[:10])
        except ValueError:
            continue
        if gd >= tourney_start:
            continue
        pre_tourney_games.append(tg)
        game_teams[tg["game_id"]].append(tg)

    logger.info("Year %d: %d pre-tournament team-game records", year, len(pre_tourney_games))

    # Process each game — accumulate both offensive and defensive stats
    for game_id, entries in game_teams.items():
        if len(entries) != 2:
            continue  # Need both sides of the game

        for i, tg in enumerate(entries):
            opp = entries[1 - i]  # The other team's box score

            raw_id = tg.get("team_id", "")
            tid = id_map.get(raw_id, raw_id)
            s = stats[tid]

            fgm = tg.get("fgm", 0) or 0
            fga = tg.get("fga", 0) or 0
            fg3m = tg.get("fg3m", 0) or 0
            fta = tg.get("fta", 0) or 0
            turnovers = tg.get("turnovers", 0) or 0
            orb = tg.get("orb", 0) or 0
            drb = tg.get("drb", 0) or 0
            poss = tg.get("possessions", 0) or 0

            s["fgm"] += fgm
            s["fga"] += fga
            s["fg3m"] += fg3m
            s["fta"] += fta
            s["turnovers"] += turnovers
            s["orb"] += orb
            s["drb"] += drb
            s["possessions"] += poss
            s["games"] += 1

            # Opponent stats (for our defensive four factors)
            opp_fgm = opp.get("fgm", 0) or 0
            opp_fga = opp.get("fga", 0) or 0
            opp_fg3m = opp.get("fg3m", 0) or 0
            opp_fta = opp.get("fta", 0) or 0
            opp_turnovers = opp.get("turnovers", 0) or 0
            opp_orb = opp.get("orb", 0) or 0

            s["opp_fgm"] += opp_fgm
            s["opp_fga"] += opp_fga
            s["opp_fg3m"] += opp_fg3m
            s["opp_fta"] += opp_fta
            s["opp_turnovers"] += opp_turnovers
            s["opp_orb"] += opp_orb

    # Compute four factors per team
    result = {}
    for tid, s in stats.items():
        if s["games"] < 5:
            continue

        fga = s["fga"]
        fgm = s["fgm"]
        fg3m = s["fg3m"]
        fta = s["fta"]
        to = s["turnovers"]
        orb = s["orb"]
        drb = s["drb"]
        poss = s["possessions"]

        opp_fga = s["opp_fga"]
        opp_fgm = s["opp_fgm"]
        opp_fg3m = s["opp_fg3m"]
        opp_fta = s["opp_fta"]
        opp_to = s["opp_turnovers"]
        opp_orb = s["opp_orb"]

        # Offensive Four Factors
        efg = _safe_div(fgm + 0.5 * fg3m, fga, 0.50)
        to_rate = _safe_div(to, poss, 0.18) if poss > 0 else _safe_div(to, fga + 0.44 * fta + to, 0.18)
        orb_rate = _safe_div(orb, orb + (drb if drb > 0 else 1), 0.28)  # ORB / (ORB + opp DRB)
        ftr = _safe_div(fta, fga, 0.35)

        # Defensive Four Factors
        opp_efg = _safe_div(opp_fgm + 0.5 * opp_fg3m, opp_fga, 0.48)
        opp_to_rate = _safe_div(opp_to, poss, 0.18) if poss > 0 else 0.18
        drb_rate = _safe_div(drb, drb + opp_orb, 0.72)
        opp_ftr = _safe_div(opp_fta, opp_fga, 0.30)

        result[tid] = {
            "effective_fg_pct": round(efg, 4),
            "turnover_rate": round(to_rate, 4),
            "offensive_reb_rate": round(orb_rate, 4),
            "free_throw_rate": round(ftr, 4),
            "opp_effective_fg_pct": round(opp_efg, 4),
            "opp_turnover_rate": round(opp_to_rate, 4),
            "defensive_reb_rate": round(drb_rate, 4),
            "opp_free_throw_rate": round(opp_ftr, 4),
        }

    logger.info("Year %d: computed four factors for %d teams", year, len(result))
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, help="Single year")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    years = [args.year] if args.year else sorted(TOURNAMENT_START_DATES.keys())

    for year in years:
        result = compute_four_factors_for_year(year)
        if result is None:
            continue

        if args.dry_run:
            # Show a sample team
            sample_tid = next(iter(result))
            print(f"\n{year} ({len(result)} teams), sample '{sample_tid}':")
            for k, v in result[sample_tid].items():
                print(f"  {k}: {v}")
            continue

        # Write to historical dir
        outpath = HIST_DIR / f"torvik_four_factors_{year}.json"
        payload = {
            "data_type": "pre_tournament_computed",
            "cutoff_date": TOURNAMENT_START_DATES[year].isoformat(),
            "tournament_start": TOURNAMENT_START_DATES[year].isoformat(),
            "n_teams": len(result),
        }
        payload.update(result)
        with open(outpath, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Wrote %s (%d teams)", outpath, len(result))

        # Also write to data/raw/ if it exists there
        raw_path = ROOT / "data" / "raw" / f"torvik_four_factors_{year}.json"
        if raw_path.exists():
            with open(raw_path, "w") as f:
                json.dump(payload, f, indent=2)
            logger.info("Wrote %s", raw_path)


if __name__ == "__main__":
    main()
