"""Schema validators for ingested artifacts."""

from __future__ import annotations

from typing import Dict, List, Optional


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
    return errors


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


def validate_shotquality_games_payload(payload: Dict, min_xp_coverage: float = 0.8) -> List[str]:
    errors = validate_games_payload(payload)
    games = payload.get("games")
    if not isinstance(games, list) or not games:
        return errors

    total_possessions = 0
    xp_possessions = 0
    for idx, game in enumerate(games):
        if not isinstance(game, dict):
            continue
        possessions = game.get("possessions")
        if not isinstance(possessions, list) or not possessions:
            errors.append(f"games[{idx}] missing possession-level records")
            continue
        for p_idx, poss in enumerate(possessions):
            if not isinstance(poss, dict):
                errors.append(f"games[{idx}].possessions[{p_idx}] must be an object")
                continue
            total_possessions += 1
            if poss.get("xp") is not None:
                xp_possessions += 1
            if not poss.get("team_id"):
                errors.append(f"games[{idx}].possessions[{p_idx}] missing team_id")

    if total_possessions == 0:
        errors.append("shotquality games contain zero possessions")
    else:
        coverage = xp_possessions / total_possessions
        if coverage < min_xp_coverage:
            errors.append(
                f"shotquality possession xP coverage too low ({coverage:.1%}); expected >= {min_xp_coverage:.0%}"
            )
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

        if rapm_like_count == 0 and not isinstance(team.get("stints"), list):
            errors.append(
                f"teams[{idx}] has no RAPM-like player fields and no stint data for RAPM derivation"
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
