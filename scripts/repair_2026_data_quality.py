#!/usr/bin/env python3
"""Repair season data quality issues.

Addresses the following gaps in scraped season data:

1. Defensive Four Factors (opp_effective_fg_pct, opp_turnover_rate,
   opp_free_throw_rate) are all zero — computed from historical game box scores.
2. Coach tournament teams arrays are empty — populated from Barttorvik
   team-coach mappings, and head_coach injected into roster data.
3. Player RAPM values are null — estimated from BPM/WARP/usage priors.
4. Manifest is stale — updated with current timestamp and cleared errors.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.features.public_advanced_metrics import PublicAdvancedMetricsBuilder
from src.data.normalize import normalize_team_id, _raw_normalize

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_YEAR = 2026

# Cache of torvik team IDs, populated by _build_game_id_resolver()
_TORVIK_IDS: set[str] = set()


def _build_game_id_resolver(torvik_data: dict) -> callable:
    """Build a function that maps CBBpy game team names to Torvik team IDs.

    CBBpy uses "Houston Cougars" format while Torvik uses "houston".
    This resolver strips mascots and handles state/st abbreviations,
    double-underscore IDs, and other Torvik naming quirks.
    """
    teams = torvik_data.get("teams", [])
    torvik_ids = set(t["team_id"] for t in teams)
    _TORVIK_IDS.update(torvik_ids)

    import re

    # Build reverse lookup: collapsed form -> torvik_id
    # This handles double-underscore IDs like cal_st__bakersfield
    collapsed_to_torvik: dict[str, str] = {}
    for tid in torvik_ids:
        collapsed = tid.replace("_", "")
        collapsed_to_torvik[collapsed] = tid

    # Explicit aliases for CBBpy game names that don't match Torvik IDs
    # through mascot stripping or standard normalization
    _GAME_NAME_ALIASES: dict[str, str] = {
        "App State Mountaineers": "appalachian_st",
        "Cal Baptist Lancers": "cal_baptist",
        "California Baptist Lancers": "cal_baptist",
        "Cal State Bakersfield Roadrunners": "cal_st__bakersfield",
        "Cal State Fullerton Titans": "cal_st__fullerton",
        "Cal State Northridge Matadors": "cal_st__northridge",
        "UConn Huskies": "connecticut",
        "Florida International Panthers": "fiu",
        "Grambling Tigers": "grambling_st",
        "UIC Flames": "illinois_chicago",
        "IU Indianapolis Jaguars": "iu_indy",
        "Long Island University Sharks": "liu",
        "UL Monroe Warhawks": "louisiana_monroe",
        "Loyola Maryland Greyhounds": "loyola_md",
        "McNeese Cowboys": "mcneese_st",
        "Miami Hurricanes": "miami_fl",
        "Miami (OH) RedHawks": "miami_oh",
        "Omaha Mavericks": "nebraska_omaha",
        "Nicholls Colonels": "nicholls_st",
        "Sam Houston Bearkats": "sam_houston_st",
        "San Jose State Spartans": "san_jose_st",
        "SE Louisiana Lions": "southeastern_louisiana",
        "UT Martin Skyhawks": "tennessee_martin",
        "Texas A&M-Corpus Christi Islanders": "texas_a_m_corpus_chris",
        "Kansas City Roos": "umkc",
        "South Carolina Upstate Spartans": "usc_upstate",
    }

    def resolve(team_name: str) -> str:
        # Check explicit aliases first
        if team_name in _GAME_NAME_ALIASES:
            return _GAME_NAME_ALIASES[team_name]
        # First try the standard normalize_team_id (handles known aliases)
        normalized = normalize_team_id(team_name)
        if normalized in torvik_ids:
            return normalized

        # Strip mascot: try progressively shorter word prefixes
        parts = team_name.split()
        for i in range(len(parts) - 1, 0, -1):
            candidate = "_".join(parts[:i]).lower()
            candidate = re.sub(r"[^a-z0-9_]", "_", candidate)
            candidate = re.sub(r"_+", "_", candidate).strip("_")
            if candidate in torvik_ids:
                return candidate
            # Try state -> st abbreviation (Torvik convention)
            st_version = re.sub(r"_state$", "_st", candidate)
            if st_version in torvik_ids:
                return st_version
            # Try collapsed form (handles double underscores)
            collapsed = candidate.replace("_", "")
            if collapsed in collapsed_to_torvik:
                return collapsed_to_torvik[collapsed]

        return normalized  # Return best-effort even if not in torvik

    return resolve


def load_json(filename: str, required: bool = False) -> dict | list:
    path = DATA_DIR / filename
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required data file missing: {path}")
        logger.warning("Data file not found, using empty default: %s", path)
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        if required:
            raise
        logger.warning("Failed to read %s: %s — using empty default", path, exc)
        return {}


def save_json(filename: str, data: dict | list) -> None:
    path = DATA_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved %s", path)


# ---------------------------------------------------------------------------
# 1. Compute defensive Four Factors from historical game box scores
# ---------------------------------------------------------------------------

def compute_defensive_four_factors(
    torvik_data: dict,
    year: int = DEFAULT_YEAR,
) -> dict[str, dict[str, float]]:
    """Aggregate opponent box score stats into defensive Four Factors per team.

    Uses historical game box scores. For each team, we aggregate opponent
    shooting/turnover/rebounding across all games they played, giving us:
      - opp_effective_fg_pct: opponents' eFG% against this team
      - opp_turnover_rate: opponents' turnover rate against this team
      - opp_free_throw_rate: opponents' FTA/FGA against this team
    """
    resolve = _build_game_id_resolver(torvik_data)

    games_data = load_json(f"historical_games_{year}.json")
    games = games_data.get("games", [])
    logger.info("Loaded %d game records for defensive Four Factors", len(games))

    # Build a mapping: for each game_id, collect both sides
    # Games appear as one row per team, so game_id appears twice
    game_pairs: dict[str, list[dict]] = defaultdict(list)
    for g in games:
        game_pairs[g["game_id"]].append(g)

    # Accumulate opponent stats per team (using torvik-style team IDs)
    team_opp_stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {"fgm": 0, "fga": 0, "fg3m": 0, "fg3a": 0, "fta": 0,
                 "turnovers": 0, "orb": 0, "drb": 0, "possessions": 0, "games": 0}
    )

    for game_id, sides in game_pairs.items():
        if len(sides) != 2:
            continue
        for i, my_side in enumerate(sides):
            opp_side = sides[1 - i]
            my_id = resolve(my_side["team_name"])

            # Opponent's box score stats = what the opponent did against us
            opp_fgm = float(opp_side.get("fgm") or 0)
            opp_fga = float(opp_side.get("fga") or 0)
            opp_fg3m = float(opp_side.get("fg3m") or 0)
            opp_fta = float(opp_side.get("fta") or 0)
            opp_tov = float(opp_side.get("turnovers") or 0)
            opp_orb = float(opp_side.get("orb") or 0)
            my_drb = float(my_side.get("drb") or 0)
            possessions = float(opp_side.get("possessions") or 0)

            if opp_fga < 1:
                continue

            stats = team_opp_stats[my_id]
            stats["fgm"] += opp_fgm
            stats["fga"] += opp_fga
            stats["fg3m"] += opp_fg3m
            stats["fta"] += opp_fta
            stats["turnovers"] += opp_tov
            stats["orb"] += opp_orb
            stats["drb"] += my_drb
            stats["possessions"] += possessions
            stats["games"] += 1

    # Compute defensive Four Factors
    result: dict[str, dict[str, float]] = {}
    for team_id, s in team_opp_stats.items():
        if s["fga"] < 10:
            continue
        opp_efg = (s["fgm"] + 0.5 * s["fg3m"]) / s["fga"]
        # Turnover rate: TOV / (FGA + 0.44*FTA + TOV)
        denom_to = s["fga"] + 0.44 * s["fta"] + s["turnovers"]
        opp_to_rate = s["turnovers"] / denom_to if denom_to > 0 else 0.0
        opp_ftr = s["fta"] / s["fga"]

        result[team_id] = {
            "opp_effective_fg_pct": round(opp_efg, 4),
            "opp_turnover_rate": round(opp_to_rate, 4),
            "opp_free_throw_rate": round(opp_ftr, 4),
        }

    logger.info("Computed defensive Four Factors for %d teams", len(result))
    return result


def apply_defensive_four_factors(
    torvik_data: dict,
    ff_data: dict,
    def_ff: dict[str, dict[str, float]],
) -> tuple[int, int]:
    """Merge defensive Four Factors into torvik team data and four factors file.

    Returns (torvik_updated, ff_updated) counts.
    """
    # Build collapsed lookup for matching def_ff keys to torvik IDs
    collapsed_def_ff: dict[str, dict[str, float]] = {}
    for k, v in def_ff.items():
        collapsed_def_ff[k.replace("_", "")] = v

    torvik_updated = 0
    teams = torvik_data.get("teams", [])
    for team in teams:
        team_id = team.get("team_id", "")
        d_direct = def_ff.get(team_id)
        d_collapsed = collapsed_def_ff.get(team_id.replace("_", ""))
        d = d_direct or d_collapsed
        if d is None:
            continue
        changed = False
        for field in ("opp_effective_fg_pct", "opp_turnover_rate", "opp_free_throw_rate"):
            if team.get(field, 0.0) == 0.0 and d.get(field, 0.0) != 0.0:
                team[field] = d[field]
                changed = True
            # Also update enriched_stats
            enriched = team.get("enriched_stats", {})
            if enriched.get(field, 0.0) == 0.0 and d.get(field, 0.0) != 0.0:
                enriched[field] = d[field]
                team["enriched_stats"] = enriched
        if changed:
            torvik_updated += 1

    # Update four factors file too
    ff_updated = 0
    for team_id, d in def_ff.items():
        if team_id in ff_data:
            entry = ff_data[team_id]
            for field in ("opp_effective_fg_pct", "opp_turnover_rate", "opp_free_throw_rate"):
                if entry.get(field, 0.0) == 0.0 and d.get(field, 0.0) != 0.0:
                    entry[field] = d[field]
            ff_updated += 1
        else:
            # Try fuzzy lookup by collapsing underscores
            collapsed = team_id.replace("_", "")
            for key in ff_data:
                if key.replace("_", "") == collapsed:
                    entry = ff_data[key]
                    for field in ("opp_effective_fg_pct", "opp_turnover_rate", "opp_free_throw_rate"):
                        if entry.get(field, 0.0) == 0.0 and d.get(field, 0.0) != 0.0:
                            entry[field] = d[field]
                    ff_updated += 1
                    break

    return torvik_updated, ff_updated


# ---------------------------------------------------------------------------
# 1b. Rebuild advanced_metrics from historical game box scores
# ---------------------------------------------------------------------------

def rebuild_advanced_metrics(
    torvik_data: dict,
    year: int = DEFAULT_YEAR,
) -> int:
    """Regenerate advanced_metrics JSON by re-running PublicAdvancedMetricsBuilder.

    The builder pairs game records by game_id to fill opponent box score stats
    from the companion row, computes SOS-adjusted efficiency ratings, Four
    Factors, and Barthag.  Returns the number of teams in the output.
    """
    games_data = load_json(f"historical_games_{year}.json")
    game_records = games_data.get("games", [])
    if not game_records:
        logger.warning("No game records found — skipping advanced metrics rebuild")
        return 0

    torvik_teams = torvik_data.get("teams", [])
    builder = PublicAdvancedMetricsBuilder()
    result = builder.build(game_records, teams=torvik_teams)

    team_count = len(result.get("teams", []))
    save_json(f"advanced_metrics_{year}.json", result)
    return team_count


# ---------------------------------------------------------------------------
# 2. Map coaches to current teams
# ---------------------------------------------------------------------------

def fix_coach_tournament_teams(
    coach_data: dict,
    roster_data: dict,
    team_coach_map: dict[str, str],
) -> tuple[int, int]:
    """Populate coach tournament teams arrays and inject head_coach into rosters.

    Uses a team_coach_map (team_id -> coach_name) scraped from Barttorvik to:
    1. Set head_coach on each roster team block (enables data_loader.py lookup)
    2. Populate the teams array on coach tournament entries via reverse lookup

    Returns (roster_teams_updated, coaches_updated) counts.
    """
    # -- Inject head_coach into roster team blocks --
    roster_teams = roster_data.get("teams", [])
    roster_updated = 0

    # Build collapsed lookup for fuzzy matching
    collapsed_map: dict[str, tuple[str, str]] = {}
    for tid, coach in team_coach_map.items():
        collapsed_map[tid.replace("_", "")] = (tid, coach)

    for team in roster_teams:
        tid = team.get("team_id", "")
        coach = team_coach_map.get(tid)
        if not coach:
            # Try fuzzy match via collapsed form
            entry = collapsed_map.get(tid.replace("_", ""))
            if entry:
                coach = entry[1]
        if not coach:
            # Try normalizing team_name
            tname = team.get("team_name", "")
            norm = normalize_team_id(tname)
            coach = team_coach_map.get(norm)
            if not coach:
                entry = collapsed_map.get(norm.replace("_", ""))
                if entry:
                    coach = entry[1]
        if coach:
            team["head_coach"] = coach
            roster_updated += 1

    # -- Populate teams arrays on coach tournament entries --
    coaches = coach_data.get("coaches", {})

    # Build reverse map: normalized coach name -> [team_ids]
    coach_to_teams: dict[str, list[str]] = defaultdict(list)
    for tid, coach_name in team_coach_map.items():
        norm_coach = _normalize_coach_name(coach_name)
        coach_to_teams[norm_coach].append(tid)

    coaches_updated = 0
    for cid, info in coaches.items():
        coach_name = info.get("name", "")
        norm = _normalize_coach_name(coach_name)
        matched_teams = coach_to_teams.get(norm, [])

        # Fallback: try last-name-only matching
        if not matched_teams and " " in coach_name:
            last_name = coach_name.split()[-1].lower()
            for norm_key, teams_list in coach_to_teams.items():
                if norm_key.endswith(last_name) or last_name in norm_key.split("_"):
                    matched_teams = teams_list
                    break

        if matched_teams:
            info["teams"] = matched_teams
            info["teams_source"] = "barttorvik_team_coaches"
            coaches_updated += 1
        elif not info.get("teams"):
            info["teams_source"] = "unavailable_from_scrape"

    logger.info(
        "Coach mapping: %d roster teams updated, %d coaches mapped to teams",
        roster_updated, coaches_updated,
    )
    return roster_updated, coaches_updated


def _normalize_coach_name(name: str) -> str:
    """Normalize a coach name for fuzzy matching."""
    return "".join(
        c.lower() if c.isalnum() else "_" for c in (name or "")
    ).strip("_")


# ---------------------------------------------------------------------------
# 3. Estimate RAPM from BPM/WARP priors
# ---------------------------------------------------------------------------

def estimate_rapm_from_priors(roster_data: dict) -> int:
    """Fill null/zero RAPM values using BPM/WARP/usage proxy.

    Uses the same formula as enrich_roster_rapm() in data_loader.py:
      proxy = 0.6 * BPM + 0.3 * (4.0 * WARP) + 0.1 * ((usage - 20) / 25)
      off_share = 0.6 if usage >= 20 else 0.45

    Returns count of players updated.
    """
    teams = roster_data.get("teams", [])
    updated = 0

    for team in teams:
        players = team.get("players", [])
        for player in players:
            rapm_off = player.get("rapm_offensive")
            rapm_def = player.get("rapm_defensive")

            # Check if RAPM is null or zero
            if rapm_off is not None and rapm_def is not None:
                if abs(float(rapm_off or 0)) > 1e-8 or abs(float(rapm_def or 0)) > 1e-8:
                    continue

            bpm = float(player.get("box_plus_minus") or 0.0)
            warp = float(player.get("warp") or 0.0)
            usage = float(player.get("usage_rate") or 0.0)

            warp_signal = 4.0 * warp
            usage_signal = (usage - 20.0) / 25.0
            proxy = 0.6 * bpm + 0.3 * warp_signal + 0.1 * usage_signal

            off_share = 0.6 if usage >= 20.0 else 0.45
            player["rapm_offensive"] = round(proxy * off_share, 4)
            player["rapm_defensive"] = round(proxy * (1.0 - off_share), 4)
            updated += 1

    logger.info("Estimated RAPM for %d players from BPM/WARP priors", updated)
    return updated


# ---------------------------------------------------------------------------
# 4. Update manifest
# ---------------------------------------------------------------------------

def update_manifest(
    torvik_ff_fixed: int,
    rapm_fixed: int,
    def_ff_count: int,
    coaches_mapped: int = 0,
    year: int = DEFAULT_YEAR,
) -> None:
    """Update manifest with current timestamp and revised validation errors."""
    manifest = load_json(f"manifest_{year}.json")

    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()

    # Revise validation errors based on what we fixed
    errors = manifest.get("validation_errors", {})

    # Torvik: remove Four Factors errors if we populated them
    torvik_errors = errors.get("torvik_json", [])
    if torvik_ff_fixed > 0 or def_ff_count > 0:
        # Remove the "all zero" errors for fields we've now populated
        fixed_fields = set()
        if torvik_ff_fixed > 0:
            fixed_fields.update([
                "effective_fg_pct", "turnover_rate",
                "offensive_reb_rate", "free_throw_rate",
            ])
        if def_ff_count > 0:
            fixed_fields.update([
                "opp_effective_fg_pct", "opp_turnover_rate", "opp_free_throw_rate",
            ])
        torvik_errors = [
            e for e in torvik_errors
            if not any(f"'{f}'" in e for f in fixed_fields)
        ]
        errors["torvik_json"] = torvik_errors

    # Rosters: update RAPM errors
    if rapm_fixed > 0:
        errors["rosters_json"] = [
            f"RAPM estimated from BPM/WARP priors for {rapm_fixed} players "
            f"(no raw RAPM available from source)"
        ]

    manifest["validation_errors"] = errors

    # Add provenance for the repair
    provenance = manifest.get("provenance", {})
    provenance["data_quality_repair"] = {
        "script": "scripts/repair_2026_data_quality.py",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "defensive_four_factors_teams": def_ff_count,
        "rapm_players_estimated": rapm_fixed,
        "coaches_mapped_to_teams": coaches_mapped,
    }
    manifest["provenance"] = provenance

    save_json(f"manifest_{year}.json", manifest)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Repair season data quality issues")
    parser.add_argument(
        "--year", type=int, default=DEFAULT_YEAR,
        help=f"Season year to repair (default: {DEFAULT_YEAR})",
    )
    args = parser.parse_args()
    year = args.year

    logger.info("=== %d Data Quality Repair ===", year)

    # Load data files (torvik is required; others degrade gracefully)
    torvik_data = load_json(f"torvik_{year}.json", required=True)
    ff_data = load_json(f"torvik_four_factors_{year}.json")
    coach_data = load_json(f"coach_tournament_{year}.json")
    roster_data = load_json(f"rosters_{year}.json")

    # 1. Compute and apply defensive Four Factors
    logger.info("--- Step 1: Defensive Four Factors ---")
    def_ff = compute_defensive_four_factors(torvik_data, year=year)
    torvik_updated, ff_updated = apply_defensive_four_factors(
        torvik_data, ff_data, def_ff,
    )
    logger.info(
        "Applied defensive FF: %d torvik teams, %d four_factors entries",
        torvik_updated, ff_updated,
    )

    # 1b. Rebuild advanced_metrics from historical games
    logger.info("--- Step 1b: Rebuild Advanced Metrics ---")
    adv_metrics_updated = rebuild_advanced_metrics(torvik_data, year=year)
    logger.info("Rebuilt advanced_metrics with %d teams", adv_metrics_updated)

    # 2. Fix coach tournament teams
    logger.info("--- Step 2: Coach Tournament Teams ---")
    from src.data.scrapers.tournament_context import TournamentContextScraper
    try:
        team_coach_map = TournamentContextScraper(
            cache_dir=str(DATA_DIR / "cache"),
        ).fetch_team_coaches(year)
    except Exception as exc:
        logger.warning("Failed to fetch team coaches: %s", exc)
        team_coach_map = {}
    logger.info("Fetched %d team-coach mappings from Barttorvik", len(team_coach_map))

    # Fallback: if scraper returned empty, try coach fields from torvik data
    if not team_coach_map:
        logger.info("Scraper returned empty; trying torvik_data coach fallback")
        for team in torvik_data.get("teams", []):
            coach = team.get("coach", "")
            tid = team.get("team_id", "")
            if coach and tid:
                team_coach_map[tid] = coach
        logger.info("Recovered %d mappings from torvik_data", len(team_coach_map))

    roster_coaches_updated = 0
    coaches_updated = 0
    if team_coach_map:
        roster_coaches_updated, coaches_updated = fix_coach_tournament_teams(
            coach_data, roster_data, team_coach_map,
        )
    else:
        logger.warning(
            "No team-coach mappings available — skipping coach team "
            "population to preserve any existing data"
        )

    # 3. Estimate RAPM from priors
    logger.info("--- Step 3: RAPM Estimation ---")
    rapm_updated = estimate_rapm_from_priors(roster_data)

    # Save updated files (skip empty dicts to avoid overwriting existing data)
    logger.info("--- Saving repaired data ---")
    save_json(f"torvik_{year}.json", torvik_data)
    if ff_data:
        save_json(f"torvik_four_factors_{year}.json", ff_data)
    if coach_data:
        save_json(f"coach_tournament_{year}.json", coach_data)
    if roster_data:
        save_json(f"rosters_{year}.json", roster_data)

    # 4. Update manifest
    logger.info("--- Step 4: Update Manifest ---")
    update_manifest(
        torvik_ff_fixed=torvik_updated,
        rapm_fixed=rapm_updated,
        def_ff_count=len(def_ff),
        coaches_mapped=coaches_updated,
        year=year,
    )

    # Summary
    logger.info("=== Repair Complete ===")
    logger.info("  Defensive Four Factors computed for %d teams", len(def_ff))
    logger.info("  Torvik teams updated: %d", torvik_updated)
    logger.info("  Four Factors file entries updated: %d", ff_updated)
    logger.info("  Advanced metrics rebuilt for %d teams", adv_metrics_updated)
    logger.info("  Roster teams with head_coach: %d", roster_coaches_updated)
    logger.info("  Coach entries mapped to teams: %d", coaches_updated)
    logger.info("  Player RAPM estimated: %d", rapm_updated)

    # --- Data Quality Checks ---
    logger.info("--- Data Quality Checks ---")
    warnings = []

    adv_data = load_json(f"advanced_metrics_{year}.json")
    adv_teams = adv_data.get("teams", [])
    if adv_teams:
        barthag_ones = sum(
            1 for t in adv_teams if t.get("barthag", 0) >= 1.0
        )
        if barthag_ones > 10:
            warnings.append(
                f"{barthag_ones} teams have barthag >= 1.0 (expected <= 10)"
            )
        zero_opp_efg = sum(
            1 for t in adv_teams
            if t.get("opp_effective_fg_pct", 0) == 0.0
        )
        if zero_opp_efg > 5:
            warnings.append(
                f"{zero_opp_efg} teams have opp_effective_fg_pct == 0.0 "
                f"(expected <= 5)"
            )

    coach_entries = coach_data.get("coaches", {})
    if coach_entries:
        coaches_with_teams = sum(
            1 for c in coach_entries.values() if c.get("teams")
        )
        pct = coaches_with_teams / len(coach_entries) if coach_entries else 0
        if pct < 0.5:
            warnings.append(
                f"Only {coaches_with_teams}/{len(coach_entries)} "
                f"({pct:.0%}) coaches have non-empty teams"
            )

    if warnings:
        for w in warnings:
            logger.warning("DATA QUALITY: %s", w)
    else:
        logger.info("All data quality checks passed")


if __name__ == "__main__":
    main()
