"""Schema validators for ingested artifacts."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _to_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def validate_teams_payload(payload: Dict) -> List[str]:
    errors: List[str] = []
    teams = payload.get("teams")
    if not isinstance(teams, list) or not teams:
        return ["teams payload must include non-empty 'teams' list"]

    required = {"name", "seed", "region"}
    for idx, row in enumerate(teams):
        if not isinstance(row, dict):
            errors.append(f"teams[{idx}] must be an object")
            continue
        missing = [k for k in required if k not in row]
        if missing:
            errors.append(f"teams[{idx}] missing fields: {', '.join(missing)}")
    return errors


def validate_ratings_payload(
    payload: Dict,
    name_field: str = "name",
    required_numeric_fields: Optional[List[str]] = None,
    min_unique_team_ids: int = 1,
    variance_fields: Optional[List[str]] = None,
    stddev_thresholds: Optional[Dict[str, float]] = None,
) -> List[str]:
    errors: List[str] = []
    teams = payload.get("teams")
    if not isinstance(teams, list) or not teams:
        return ["ratings payload must include non-empty 'teams' list"]

    required = {"team_id", name_field}
    team_ids = set()
    field_values: Dict[str, List[float]] = {field: [] for field in (variance_fields or [])}
    required_numeric_fields = required_numeric_fields or []
    for idx, row in enumerate(teams):
        if not isinstance(row, dict):
            errors.append(f"teams[{idx}] must be an object")
            continue
        missing = [k for k in required if k not in row]
        if missing:
            errors.append(f"teams[{idx}] missing fields: {', '.join(missing)}")
            continue
        team_ids.add(str(row.get("team_id")))

        for field in required_numeric_fields:
            val = _to_float(row.get(field))
            if val is None:
                errors.append(f"teams[{idx}] missing/invalid numeric field '{field}'")

        for field in field_values:
            val = _to_float(row.get(field))
            if val is not None:
                field_values[field].append(val)

    if len(team_ids) < min_unique_team_ids:
        errors.append(
            f"ratings payload must include at least {min_unique_team_ids} unique teams; got {len(team_ids)}"
        )

    for field, values in field_values.items():
        uniques = {round(v, 6) for v in values}
        if len(uniques) <= 1:
            errors.append(f"ratings field '{field}' has insufficient variance")

    # Standard deviation minimum thresholds — catches near-constant distributions
    # that pass the unique-count check (e.g. all AdjOE within ±0.5 of mean).
    if stddev_thresholds and field_values:
        for field, min_std in stddev_thresholds.items():
            values = field_values.get(field, [])
            if len(values) >= 10:
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
                std = variance ** 0.5
                if std < min_std:
                    errors.append(
                        f"ratings field '{field}' std dev too low: "
                        f"{std:.4f} < {min_std} across {len(values)} teams "
                        f"— likely data corruption or scraper bug"
                    )

    # Reject payloads where critical numeric fields are all-zero/default.
    # This catches silent upstream failures where the HTML column is absent
    # and the scraper falls through to 0.0 for every team.
    critical_zero_fields = required_numeric_fields or []
    if teams and critical_zero_fields:
        for field in critical_zero_fields:
            vals = [_to_float(row.get(field)) for row in teams if isinstance(row, dict)]
            nonzero = [v for v in vals if v is not None and abs(v) > 1e-6]
            if vals and not nonzero:
                errors.append(
                    f"ratings field '{field}' is zero/missing for ALL {len(vals)} teams "
                    f"— likely a data source failure"
                )

    return errors


_CROSS_SOURCE_THRESHOLDS = {
    "adj_offensive_efficiency": 3.0,
    "adj_defensive_efficiency": 3.0,
    "barthag": 0.03,
}


def cross_validate_torvik_sources(
    primary: List[Dict],
    secondary: List[Dict],
    thresholds: Optional[Dict[str, float]] = None,
) -> List[str]:
    """Compare key metrics between two Torvik-source payloads.

    Advisory only — returns warnings, never blocks.  Compares up to 30 teams
    (by position in primary) on fields specified in *thresholds*.
    """
    thresholds = thresholds or _CROSS_SOURCE_THRESHOLDS
    warnings: List[str] = []
    sec_map: Dict[str, Dict] = {}
    for r in secondary:
        tid = r.get("team_id") or r.get("name", "")
        if tid:
            sec_map[tid] = r
    compared = 0
    for row in primary[:30]:
        tid = row.get("team_id") or row.get("name", "")
        sec_row = sec_map.get(tid)
        if sec_row is None:
            continue
        compared += 1
        for field, threshold in thresholds.items():
            p_val = _to_float(row.get(field))
            s_val = _to_float(sec_row.get(field))
            if p_val is not None and s_val is not None:
                diff = abs(p_val - s_val)
                if diff > threshold:
                    warnings.append(
                        f"Cross-source divergence: {tid}.{field} "
                        f"primary={p_val:.4f} secondary={s_val:.4f} "
                        f"diff={diff:.4f} (threshold={threshold})"
                    )
    if compared < 5:
        warnings.append(
            f"Cross-source comparison: only {compared}/30 teams matched — "
            f"name resolution may be inconsistent between sources"
        )
    return warnings


def validate_games_payload(payload: Dict) -> List[str]:
    errors: List[str] = []
    games = payload.get("games")
    if not isinstance(games, list) or not games:
        return ["games payload must include non-empty 'games' list"]

    for idx, row in enumerate(games):
        if not isinstance(row, dict):
            errors.append(f"games[{idx}] must be an object")
            continue
        game_id = row.get("game_id") or row.get("id")
        t1 = row.get("team1_id") or row.get("team1") or row.get("home_team") or row.get("team_id")
        t2 = row.get("team2_id") or row.get("team2") or row.get("away_team") or row.get("opponent_id")
        if not game_id:
            errors.append(f"games[{idx}] missing game id")
        if not t1:
            errors.append(f"games[{idx}] missing team1/home/team")
        if not t2:
            errors.append(f"games[{idx}] missing team2/away/opponent")
    return errors


def validate_public_picks_payload(payload: Dict) -> List[str]:
    teams = payload.get("teams")
    if not isinstance(teams, dict) or not teams:
        return ["public picks must include non-empty 'teams' object"]

    errors: List[str] = []
    for team_id, row in teams.items():
        if not isinstance(row, dict):
            errors.append(f"teams['{team_id}'] must be an object")
            continue
        for key in (
            "team_name",
            "seed",
            "region",
            "round_of_64_pct",
            "round_of_32_pct",
            "sweet_16_pct",
            "elite_8_pct",
            "final_four_pct",
            "champion_pct",
        ):
            if key not in row:
                errors.append(f"teams['{team_id}'] missing '{key}'")
    return errors


def validate_rosters_payload(payload: Dict) -> List[str]:
    teams = payload.get("teams")
    if not isinstance(teams, list) or not teams:
        return ["roster payload must include non-empty 'teams' list"]

    errors: List[str] = []
    for idx, team in enumerate(teams):
        if not isinstance(team, dict):
            errors.append(f"teams[{idx}] must be an object")
            continue

        team_id = team.get("team_id") or team.get("team_name") or team.get("name")
        if not team_id:
            errors.append(f"teams[{idx}] missing team identifier")

        players = team.get("players")
        if not isinstance(players, list) or not players:
            errors.append(f"teams[{idx}] missing players list")
            continue

        rapm_like_count = 0
        nonzero_rapm_count = 0
        for p_idx, player in enumerate(players):
            if not isinstance(player, dict):
                errors.append(f"teams[{idx}].players[{p_idx}] must be an object")
                continue
            if not (player.get("player_id") or player.get("name")):
                errors.append(f"teams[{idx}].players[{p_idx}] missing player_id/name")

            rapm_total = _to_float(player.get("rapm_total"))
            rapm_off = _to_float(player.get("rapm_offensive"))
            rapm_def = _to_float(player.get("rapm_defensive"))
            rapm_value = None
            if rapm_total is not None:
                rapm_value = rapm_total
            elif rapm_off is not None or rapm_def is not None:
                rapm_value = float((rapm_off or 0.0) + (rapm_def or 0.0))
            if rapm_value is not None and abs(rapm_value) > 1e-6:
                nonzero_rapm_count += 1
            if (
                player.get("rapm_total") is not None
                or player.get("rapm_offensive") is not None
                or player.get("rapm_defensive") is not None
                or player.get("box_plus_minus") is not None
            ):
                rapm_like_count += 1

        if rapm_like_count == 0:
            errors.append(
                f"teams[{idx}] has no RAPM-like player fields for RAPM derivation"
            )
        if rapm_like_count > 0 and nonzero_rapm_count == 0:
            errors.append(f"teams[{idx}] has RAPM fields but all RAPM values are zero")

    return errors


def validate_transfer_payload(payload: Dict) -> List[str]:
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        return ["transfer payload must include non-empty 'entries' list"]

    errors: List[str] = []
    for idx, row in enumerate(entries):
        if not isinstance(row, dict):
            errors.append(f"entries[{idx}] must be an object")
            continue
        if not (row.get("player_id") or row.get("player_name")):
            errors.append(f"entries[{idx}] missing player id/name")
    return errors


def validate_odds_payload(payload: Dict) -> List[str]:
    teams = payload.get("teams")
    if not isinstance(teams, list) or not teams:
        return ["odds payload must include non-empty 'teams' list"]

    errors: List[str] = []
    for idx, row in enumerate(teams):
        if not isinstance(row, dict):
            errors.append(f"teams[{idx}] must be an object")
            continue
        if not (row.get("team_id") or row.get("team_name") or row.get("name")):
            errors.append(f"teams[{idx}] missing team identifier")
        implied = _to_float(row.get("implied_win_probability"))
        title = _to_float(row.get("title_odds"))
        if implied is None and title is None:
            errors.append(f"teams[{idx}] missing implied_win_probability/title_odds")
    return errors


def validate_season_quality(
    records,
    min_games: int = 500,
    min_unique_teams: int = 200,
    min_unique_dates: int = 30,
    min_box_score_pct: float = 0.5,
    max_single_team_pct: float = 0.05,
) -> List[str]:
    """Validate statistical quality of ingested game records.

    Checks coverage, temporal spread, box-score availability, and team
    concentration.  Returns a list of error strings (empty = pass).

    Parameters
    ----------
    records : list of IngestionGameRecord
        Game-level records to validate.
    min_games : int
        Minimum number of games required.  NCAA D1 has ~5,500 per season;
        500 is a ~9 % coverage floor for stable Elo / SOS computation.
    min_unique_teams : int
        Minimum unique team IDs across all games.  D1 has ~360 teams;
        200 ensures multi-conference representation.
    min_unique_dates : int
        Minimum unique game dates.  A season spans ~160 days; 30 ensures
        point-in-time features capture season progression.
    min_box_score_pct : float
        Fraction of games that must have FGA > 0 for Four Factors
        (eFG%%, TOV%%, ORB%%, FTR) computation.
    max_single_team_pct : float
        Maximum fraction of games any single team may appear in.  Detects
        scraper bias returning team-specific rather than league-wide data.
    """
    errors: List[str] = []
    n = len(records)

    if n < min_games:
        errors.append(
            f"insufficient games: {n} < {min_games} minimum for stable "
            f"Elo/SOS computation"
        )
        # Further checks are unreliable with very few records.
        if n == 0:
            return errors

    # Team coverage
    team_ids: Dict[str, int] = {}
    for r in records:
        team_ids[r.home_team_id] = team_ids.get(r.home_team_id, 0) + 1
        team_ids[r.away_team_id] = team_ids.get(r.away_team_id, 0) + 1
    n_teams = len(team_ids)
    if n_teams < min_unique_teams:
        errors.append(
            f"insufficient team coverage: {n_teams} unique teams < "
            f"{min_unique_teams} minimum for conference representation"
        )

    # Temporal spread
    unique_dates = {r.date for r in records if r.date}
    if len(unique_dates) < min_unique_dates:
        errors.append(
            f"insufficient temporal spread: {len(unique_dates)} unique dates "
            f"< {min_unique_dates} minimum for point-in-time features"
        )

    # Box-score coverage — use the explicit has_box_score flag when
    # available (set by the ingestion layer), falling back to FGA > 0.
    if n > 0:
        has_box = sum(1 for r in records if getattr(r, 'has_box_score', False) or r.home_fga > 0 or r.away_fga > 0)
        pct = has_box / n
        if pct < min_box_score_pct:
            errors.append(
                f"insufficient box-score coverage: {pct:.1%} of games have "
                f"FGA > 0 vs {min_box_score_pct:.0%} required for Four Factors"
            )

    # Team concentration (detect scraper returning one team's games only)
    if n > 0 and team_ids:
        max_appearances = max(team_ids.values())
        max_team = max(team_ids, key=team_ids.get)
        if max_appearances / n > max_single_team_pct:
            errors.append(
                f"team concentration too high: {max_team} appears in "
                f"{max_appearances}/{n} games ({max_appearances / n:.1%}), "
                f"exceeding {max_single_team_pct:.0%} threshold"
            )

    return errors


# ---------------------------------------------------------------------------
# Four Factors quality gate
# ---------------------------------------------------------------------------

# Physically plausible bounds for NCAA D1 Four Factors.
_FF_RANGES = {
    "opp_effective_fg_pct": (0.30, 0.65),
    "opp_turnover_rate": (0.10, 0.35),
    "opp_free_throw_rate": (0.15, 0.55),
    "offensive_reb_rate": (0.15, 0.45),
    "defensive_reb_rate": (0.55, 0.85),
}


def validate_four_factors(
    torvik_payload: Dict,
    max_def_ff_zeros: int = 5,
    orb_median_range: tuple = (0.18, 0.40),
) -> List[str]:
    """Validate Four Factors quality after enrichment.

    Checks that defensive FF zeros are eliminated and offensive ORB%/DRB%
    are at team-level (not player-level).

    Parameters
    ----------
    torvik_payload : dict
        Torvik data dict with ``teams`` list.
    max_def_ff_zeros : int
        Maximum teams allowed with all defensive FF = 0.0.
    orb_median_range : tuple
        Expected (min, max) for median team-level ORB%.

    Returns
    -------
    list of str
        Error messages (empty = all checks pass).
    """
    errors: List[str] = []
    teams = torvik_payload.get("teams", [])
    if not teams:
        return ["torvik payload has no teams"]

    # Check defensive FF zeros
    def_ff_fields = ("opp_effective_fg_pct", "opp_turnover_rate", "opp_free_throw_rate")
    def_zero_count = sum(
        1 for t in teams
        if all(abs(_to_float(t.get(f, 0)) or 0) < 1e-6 for f in def_ff_fields)
    )
    if def_zero_count > max_def_ff_zeros:
        errors.append(
            f"{def_zero_count}/{len(teams)} teams have ALL defensive FF = 0.0 "
            f"(max allowed: {max_def_ff_zeros})"
        )

    # Check offensive ORB% for player-level bias
    orb_vals = sorted(
        _to_float(t.get("offensive_reb_rate", 0)) or 0
        for t in teams
        if (_to_float(t.get("offensive_reb_rate", 0)) or 0) > 1e-6
    )
    if orb_vals:
        median_orb = orb_vals[len(orb_vals) // 2]
        lo, hi = orb_median_range
        if median_orb < lo:
            errors.append(
                f"ORB% median = {median_orb:.4f} is below {lo} — "
                f"likely CSV player-level bias (not team-level)"
            )
        elif median_orb > hi:
            errors.append(
                f"ORB% median = {median_orb:.4f} is above {hi} — suspiciously high"
            )

    return errors


def validate_team_id_consistency(
    seeds_payload: Dict[str, Any],
    torvik_payload: Dict[str, Any],
    results_payload: Optional[Dict[str, Any]] = None,
    year: int = 0,
    max_missing_before_error: int = 3,
) -> List[str]:
    """Cross-validate team IDs across seeds, Torvik, and results for a season.

    Normalizes all team IDs through ``normalize_team_id`` and checks that:
    - All tournament seed teams exist in Torvik data
    - All tournament result teams exist in seeds data

    Returns a list of error strings.  If the number of missing teams exceeds
    *max_missing_before_error*, the errors are treated as blocking.
    """
    from ..normalize import normalize_team_id

    errors: List[str] = []
    prefix = f"[{year}] " if year else ""

    # Extract and normalize seed team IDs
    seed_teams_raw = seeds_payload.get("teams", []) if isinstance(seeds_payload, dict) else seeds_payload
    if isinstance(seed_teams_raw, list):
        seed_ids = {normalize_team_id(t["team_id"]) for t in seed_teams_raw
                    if isinstance(t, dict) and t.get("team_id")}
    else:
        seed_ids = set()

    # Extract and normalize Torvik team IDs
    torvik_teams = torvik_payload.get("teams", []) if isinstance(torvik_payload, dict) else torvik_payload
    if isinstance(torvik_teams, list):
        torvik_ids = {normalize_team_id(t["team_id"]) for t in torvik_teams
                      if isinstance(t, dict) and t.get("team_id")}
    else:
        torvik_ids = set()

    # Check seeds vs Torvik
    missing_in_torvik = seed_ids - torvik_ids
    if missing_in_torvik:
        msg = (f"{prefix}{len(missing_in_torvik)} seed team(s) not found in Torvik: "
               f"{sorted(missing_in_torvik)}")
        if len(missing_in_torvik) >= max_missing_before_error:
            errors.append(msg)
        else:
            errors.append(f"WARNING: {msg}")

    # Check results vs seeds (if results provided and non-empty)
    if results_payload:
        games = results_payload.get("games", [])
        if games:
            result_ids = set()
            for game in games:
                if not isinstance(game, dict):
                    continue
                for key in ("team1_id", "team2_id", "winner_id", "loser_id"):
                    tid = game.get(key)
                    if tid:
                        result_ids.add(normalize_team_id(tid))
            missing_in_seeds = result_ids - seed_ids
            # Allow up to 4 missing (First Four losers)
            if len(missing_in_seeds) > 4:
                errors.append(
                    f"{prefix}{len(missing_in_seeds)} result team(s) not in seeds: "
                    f"{sorted(missing_in_seeds)}"
                )

    return errors


def validate_tournament_results_completeness(
    year: int,
    games: List[Dict[str, Any]],
) -> List[str]:
    """Validate that tournament results have the expected game count with correct round counts.

    Game counts vary by era:
    - 2005-2010: 64 games (1 play-in + 63 bracket)
    - 2011+: 67 games (4 First Four + 63 bracket)

    Returns a list of error strings (empty if valid).
    """
    if year >= 2011:
        EXPECTED_ROUNDS = {"FF": 4, "R64": 32, "R32": 16, "S16": 8, "E8": 4, "F4": 2, "NCG": 1}
        expected_total = 67
    else:
        EXPECTED_ROUNDS = {"FF": 1, "R64": 32, "R32": 16, "S16": 8, "E8": 4, "F4": 2, "NCG": 1}
        expected_total = 64
    errors: List[str] = []

    if not games:
        errors.append(f"{year}: Tournament results are empty (0 games)")
        return errors

    if len(games) != expected_total:
        errors.append(f"{year}: Expected {expected_total} games, got {len(games)}")

    round_counts: Dict[str, int] = {}
    for g in games:
        rnd = g.get("round_name", "")
        round_counts[rnd] = round_counts.get(rnd, 0) + 1

    for rnd, expected in EXPECTED_ROUNDS.items():
        actual = round_counts.get(rnd, 0)
        if actual != expected:
            errors.append(f"{year}: {rnd} has {actual} games, expected {expected}")

    # Check required fields
    required_fields = {"year", "round_name", "region", "team1_id", "team1_seed",
                       "team1_score", "team2_id", "team2_seed", "team2_score", "team1_won"}
    for idx, g in enumerate(games):
        if not isinstance(g, dict):
            errors.append(f"{year}: games[{idx}] is not a dict")
            continue
        missing = required_fields - set(g.keys())
        if missing:
            errors.append(f"{year}: games[{idx}] missing fields: {sorted(missing)}")
            break  # Only report first missing-field game

    return errors
