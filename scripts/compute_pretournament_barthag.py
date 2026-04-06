#!/usr/bin/env python3
"""Compute pre-tournament barthag from game-level data.

Since we can't scrape barttorvik.com with date filtering from this environment,
we compute barthag ourselves from the historical game data, using only games
played BEFORE the NCAA tournament starts.

Method:
1. Load all games from historical_games_{year}.json
2. Filter to games before tournament start date
3. Estimate possessions per game using the Kenpom formula
4. Compute raw offensive/defensive efficiency (points per 100 possessions)
5. Run iterative adjustment to account for opponent strength (15 iterations)
6. Compute barthag from adjusted efficiencies using the Pythagorean formula:
   barthag = AdjOE^11.5 / (AdjOE^11.5 + AdjDE^11.5)
7. Overwrite data/raw/historical/torvik_{year}.json with pre-tournament data

The Pythagorean exponent of 11.5 matches Torvik's methodology for tempo-free
efficiency ratings.
"""

import json
import logging
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

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

# Pythagorean exponent for college basketball (Torvik uses ~11.5)
PYTH_EXP = 11.5

# Number of iterations for efficiency adjustment
N_ITERATIONS = 15

# Average D1 tempo (possessions per 40 minutes) — used as baseline
AVG_TEMPO = 68.0

# Average efficiency (points per 100 possessions)
AVG_EFF = 100.0


def _strip_mascot(game_id):
    """Strip common mascot suffixes from ESPN-style game IDs.

    'uconn_huskies' -> 'uconn'
    'kansas_state_wildcats' -> 'kansas_state'
    'alabama' -> 'alabama' (no mascot)

    Uses a heuristic: known mascot words are stripped from the end.
    """
    _MASCOTS = frozenset(
        {
            "aggies",
            "anteaters",
            "aztecs",
            "badgers",
            "banana_slugs",
            "bears",
            "beavers",
            "bearcats",
            "bearkats",
            "bees",
            "bengals",
            "billikens",
            "bison",
            "blazers",
            "blue_devils",
            "blue_hens",
            "blue_jays",
            "blue_raiders",
            "bobcats",
            "boilermakers",
            "bonnies",
            "braves",
            "broncos",
            "bruins",
            "buckeyes",
            "buffaloes",
            "bulldogs",
            "bulls",
            "bunnies",
            "cadets",
            "camels",
            "cardinals",
            "catamounts",
            "cavaliers",
            "chants",
            "chippewas",
            "colonels",
            "colonials",
            "comets",
            "commodores",
            "cornhuskers",
            "cougars",
            "cowboys",
            "crimson",
            "crimson_tide",
            "crusaders",
            "cyclones",
            "daemons",
            "deacons",
            "demon_deacons",
            "dolphins",
            "dons",
            "dragons",
            "dukes",
            "eagles",
            "explorers",
            "falcons",
            "fighting_illini",
            "fighting_irish",
            "flames",
            "flyers",
            "friars",
            "gaels",
            "gators",
            "generals",
            "golden_bears",
            "golden_eagles",
            "golden_flashes",
            "golden_griffins",
            "golden_hurricane",
            "golden_knights",
            "governors",
            "grizzlies",
            "hawks",
            "heels",
            "highlanders",
            "hilltoppers",
            "hokies",
            "hoosiers",
            "hornets",
            "horned_frogs",
            "huskies",
            "ichabods",
            "illini",
            "indians",
            "irish",
            "islanders",
            "jackrabbits",
            "jaguars",
            "jaspers",
            "javelinas",
            "jayhawks",
            "kangaroos",
            "keydets",
            "kingsmen",
            "knights",
            "leopards",
            "lions",
            "lopes",
            "lumberjacks",
            "mavericks",
            "mean_green",
            "midshipmen",
            "miners",
            "minutemen",
            "moccasins",
            "monarchs",
            "mountaineers",
            "musketeers",
            "mustangs",
            "nittany_lions",
            "norse",
            "oaks",
            "orange",
            "orangemen",
            "owls",
            "paladins",
            "panthers",
            "patriots",
            "peacocks",
            "penguins",
            "phoenix",
            "pilots",
            "pioneers",
            "pirates",
            "privateers",
            "purple_eagles",
            "quakers",
            "racers",
            "ragin_cajuns",
            "rails",
            "rainbow_warriors",
            "rams",
            "rattlers",
            "razorbacks",
            "rebels",
            "red_flash",
            "red_foxes",
            "red_hawks",
            "red_raiders",
            "red_storm",
            "red_wolves",
            "redhawks",
            "redbirds",
            "retrievers",
            "roadrunners",
            "rockets",
            "roos",
            "royals",
            "running_rebels",
            "sailfish",
            "saints",
            "salukis",
            "scarlet_hawks",
            "scarlet_knights",
            "scots",
            "sea_gulls",
            "seahawks",
            "shockers",
            "skyhawks",
            "sooners",
            "spartans",
            "spiders",
            "stags",
            "sun_devils",
            "tar_heels",
            "terrapins",
            "terriers",
            "thunderbirds",
            "thundering_herd",
            "tidal_wave",
            "tigers",
            "titans",
            "toreros",
            "trojans",
            "tribe",
            "tritons",
            "utes",
            "vikings",
            "volunteers",
            "vulcans",
            "warriors",
            "wasps",
            "wave",
            "wildcats",
            "wolf_pack",
            "wolverines",
            "wolves",
            "wren",
            "yellow_jackets",
            "zips",
        }
    )
    parts = game_id.split("_")
    # Try removing 1, 2, or 3 trailing words as mascot
    for n_suffix in range(3, 0, -1):
        if len(parts) <= n_suffix:
            continue
        suffix = "_".join(parts[-n_suffix:])
        if suffix in _MASCOTS:
            return "_".join(parts[:-n_suffix])
    return game_id


def _build_game_to_canonical_map(year):
    """Build mapping from game-data team IDs to canonical (seed/torvik) IDs.

    Game data uses ESPN-style IDs with mascot suffixes (e.g. 'uconn_huskies'),
    while seeds/torvik use canonical short IDs (e.g. 'connecticut').
    """
    try:
        from data.normalize import normalize_team_id
    except ImportError:
        normalize_team_id = None

    canonical_ids = set()

    # Collect canonical IDs from torvik and seed files
    torvik_path = HIST_DIR / f"torvik_{year}.json"
    if torvik_path.exists():
        with open(torvik_path) as f:
            tdata = json.load(f)
        for t in tdata.get("teams", []):
            cid = t.get("team_id", "")
            if cid:
                canonical_ids.add(cid)

    seed_path = HIST_DIR / f"tournament_seeds_{year}.json"
    if seed_path.exists():
        with open(seed_path) as f:
            sdata = json.load(f)
        for t in sdata.get("teams", []):
            cid = t.get("team_id", "")
            if cid:
                canonical_ids.add(cid)

    # Load game data team IDs
    games_path = HIST_DIR / f"historical_games_{year}.json"
    game_ids = set()
    if games_path.exists():
        with open(games_path) as f:
            gdata = json.load(f)
        for g in gdata.get("games", []):
            game_ids.add(g.get("team1_id", ""))
            game_ids.add(g.get("team2_id", ""))

    mapping = {}
    for gid in game_ids:
        if not gid:
            continue
        # Direct match
        if gid in canonical_ids:
            mapping[gid] = gid
            continue

        # Strip mascot suffix, then try normalize
        stripped = _strip_mascot(gid)
        if stripped in canonical_ids:
            mapping[gid] = stripped
            continue

        # Try normalize_team_id on the stripped version
        if normalize_team_id:
            normalized = normalize_team_id(stripped)
            if normalized in canonical_ids:
                mapping[gid] = normalized
                continue

        # Check if a canonical ID is a prefix of the game ID
        best = None
        best_len = 0
        for cid in canonical_ids:
            if gid.startswith(cid + "_") and len(cid) > best_len:
                best = cid
                best_len = len(cid)
        if best:
            mapping[gid] = best
            continue

        # No match — keep raw (non-tournament team, won't affect results)
        mapping[gid] = gid

    return mapping


def estimate_possessions(score1, score2, tempo_estimate=AVG_TEMPO):
    """Rough possession estimate from scores.

    In practice, possessions ~ (score1 + score2) / 2 * (AVG_TEMPO / AVG_PTS_PER_GAME).
    We use a simplified model: possessions ~ total_points * 0.475
    This is a reasonable approximation given that avg D1 games score ~140 total
    and have ~68 possessions.
    """
    total = score1 + score2
    return max(40.0, total * 0.475)


def compute_barthag_for_year(year):
    """Compute pre-tournament adjusted efficiencies and barthag for a year."""
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

    # Build mapping from game IDs to canonical IDs
    id_map = _build_game_to_canonical_map(year)

    # Collect pre-tournament games
    games = []
    for g in data.get("games", []):
        game_date_str = g.get("date", "")
        if not game_date_str:
            continue
        try:
            gd = date.fromisoformat(game_date_str[:10])
        except ValueError:
            continue
        if gd >= tourney_start:
            continue

        s1 = g.get("team1_score")
        s2 = g.get("team2_score")
        if s1 is None or s2 is None or s1 == 0 or s2 == 0:
            continue

        t1 = id_map.get(g.get("team1_id", ""), g.get("team1_id", ""))
        t2 = id_map.get(g.get("team2_id", ""), g.get("team2_id", ""))
        if not t1 or not t2 or t1 == t2:
            continue

        games.append((t1, t2, int(s1), int(s2)))

    logger.info("Year %d: %d pre-tournament games (cutoff %s)", year, len(games), tourney_start)
    if len(games) < 1000:
        logger.warning("Very few games for %d — results may be unreliable", year)

    # Compute raw efficiency per team
    team_off_pts = defaultdict(float)  # total offensive points
    team_def_pts = defaultdict(float)  # total defensive points
    team_poss = defaultdict(float)  # total possessions
    team_games_count = defaultdict(int)
    team_opponents = defaultdict(list)  # list of (opponent, off_eff, def_eff)

    for t1, t2, s1, s2 in games:
        poss = estimate_possessions(s1, s2)
        # Team 1 perspective
        team_off_pts[t1] += s1
        team_def_pts[t1] += s2
        team_poss[t1] += poss
        team_games_count[t1] += 1
        # Team 2 perspective
        team_off_pts[t2] += s2
        team_def_pts[t2] += s1
        team_poss[t2] += poss
        team_games_count[t2] += 1

    # Filter to teams with enough games
    teams = [t for t in team_poss if team_games_count[t] >= 5]
    teams.sort()
    n = len(teams)
    team_idx = {t: i for i, t in enumerate(teams)}

    logger.info("Year %d: %d teams with 5+ games", year, n)

    # Raw efficiency (points per 100 possessions)
    raw_oe = np.full(n, AVG_EFF)
    raw_de = np.full(n, AVG_EFF)
    for i, t in enumerate(teams):
        if team_poss[t] > 0:
            raw_oe[i] = team_off_pts[t] / team_poss[t] * 100.0
            raw_de[i] = team_def_pts[t] / team_poss[t] * 100.0

    # Build opponent lists for adjustment
    opp_lists = [[] for _ in range(n)]
    for t1, t2, s1, s2 in games:
        if t1 in team_idx and t2 in team_idx:
            poss = estimate_possessions(s1, s2)
            i1, i2 = team_idx[t1], team_idx[t2]
            oe1 = s1 / poss * 100.0
            de1 = s2 / poss * 100.0
            opp_lists[i1].append((i2, oe1, de1))
            opp_lists[i2].append((i1, de1, oe1))  # mirror

    # Iterative efficiency adjustment
    # Start with raw efficiencies, then adjust based on opponent strength
    adj_oe = raw_oe.copy()
    adj_de = raw_de.copy()

    for iteration in range(N_ITERATIONS):
        new_oe = np.full(n, AVG_EFF)
        new_de = np.full(n, AVG_EFF)

        for i in range(n):
            if not opp_lists[i]:
                continue
            oe_sum = 0.0
            de_sum = 0.0
            count = 0
            for opp_idx, game_oe, game_de in opp_lists[i]:
                # Adjust game OE by opponent's defensive strength
                # If opponent has good defense (low adj_de), our OE is more impressive
                opp_de_factor = adj_de[opp_idx] / AVG_EFF
                opp_oe_factor = adj_oe[opp_idx] / AVG_EFF
                oe_sum += game_oe / opp_de_factor
                de_sum += game_de / opp_oe_factor
                count += 1
            if count > 0:
                new_oe[i] = oe_sum / count
                new_de[i] = de_sum / count

        # Renormalize to keep average at AVG_EFF
        mean_oe = new_oe.mean()
        mean_de = new_de.mean()
        if mean_oe > 0:
            new_oe *= AVG_EFF / mean_oe
        if mean_de > 0:
            new_de *= AVG_EFF / mean_de

        adj_oe = new_oe
        adj_de = new_de

    # Compute barthag using Pythagorean formula
    # barthag = AdjOE^exp / (AdjOE^exp + AdjDE^exp)
    barthag = np.zeros(n)
    for i in range(n):
        oe_pow = adj_oe[i] ** PYTH_EXP
        de_pow = adj_de[i] ** PYTH_EXP
        if oe_pow + de_pow > 0:
            barthag[i] = oe_pow / (oe_pow + de_pow)
        else:
            barthag[i] = 0.5

    # Rank by barthag (descending)
    rankings = sorted(range(n), key=lambda i: -barthag[i])
    rank_map = {i: r + 1 for r, i in enumerate(rankings)}

    # Build output
    team_records = []
    for i, t in enumerate(teams):
        team_records.append(
            {
                "team_id": t,
                "team_name": t,
                "name": t,
                "conference": "",
                "t_rank": rank_map[i],
                "barthag": round(float(barthag[i]), 6),
                "adj_offensive_efficiency": round(float(adj_oe[i]), 2),
                "adj_defensive_efficiency": round(float(adj_de[i]), 2),
                "adj_tempo": AVG_TEMPO,
                "effective_fg_pct": 0.0,
                "turnover_rate": 0.0,
                "offensive_reb_rate": 0.0,
                "free_throw_rate": 0.0,
                "opp_effective_fg_pct": 0.0,
                "opp_turnover_rate": 0.0,
                "defensive_reb_rate": 0.0,
                "opp_free_throw_rate": 0.0,
            }
        )

    # Sort by rank
    team_records.sort(key=lambda x: x["t_rank"])

    logger.info(
        "Year %d top 5: %s",
        year,
        [(r["name"], r["t_rank"], f"{r['barthag']:.4f}") for r in team_records[:5]],
    )

    return {
        "data_type": "pre_tournament_computed",
        "method": "iterative_adjusted_efficiency_pythagorean",
        "cutoff_date": (tourney_start).isoformat(),
        "tournament_start": tourney_start.isoformat(),
        "n_games": len(games),
        "n_teams": n,
        "pyth_exponent": PYTH_EXP,
        "n_iterations": N_ITERATIONS,
        "teams": team_records,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, help="Single year")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    years = [args.year] if args.year else sorted(TOURNAMENT_START_DATES.keys())

    for year in years:
        result = compute_barthag_for_year(year)
        if result is None:
            continue

        if args.dry_run:
            top5 = result["teams"][:5]
            print(f"\n{year} Top 5 (pre-tournament):")
            for t in top5:
                print(
                    f"  #{t['t_rank']} {t['name']:25s} barthag={t['barthag']:.4f}  AdjOE={t['adj_offensive_efficiency']:.1f}  AdjDE={t['adj_defensive_efficiency']:.1f}"
                )
            continue

        outpath = HIST_DIR / f"torvik_{year}.json"
        with open(outpath, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Wrote %s (%d teams)", outpath, result["n_teams"])


if __name__ == "__main__":
    main()
