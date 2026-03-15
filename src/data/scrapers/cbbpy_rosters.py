"""Roster scraper that derives player databases from cbbpy schedule/boxscore/player endpoints."""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ._retry import rate_limited_call

logger = logging.getLogger(__name__)


class CBBpyRosterScraper:
    """Build canonical roster payloads from cbbpy game and player endpoints."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_rosters(self, year: int) -> Dict:
        cache_name = f"cbbpy_rosters_{year}.json"
        cached = self._load_cache(cache_name)
        if cached and isinstance(cached.get("teams"), list) and cached["teams"]:
            return cached

        scraper = self._import_module("cbbpy.mens_scraper")
        if scraper is None:
            return {}

        enable_pbp = str(os.getenv("CBBPY_ROSTER_ENABLE_PBP", "")).strip().lower() in {
            "1", "true", "yes", "y",
        }
        box_rows, pbp_rows, collection_mode = self._collect_box_and_pbp_rows(
            scraper, year, enable_pbp=enable_pbp,
        )
        if not box_rows:
            return {}

        payload = self._build_payload(year, box_rows, pbp_rows=pbp_rows)
        if not payload.get("teams"):
            return {}

        enrichment = self._enrich_from_player_endpoint(scraper, payload)
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()
        payload["source"] = "cbbpy_schedule_boxscore"
        total_pbp_stints = sum(
            len(t.get("stints", [])) for t in payload.get("teams", [])
        )
        payload["metadata"] = {
            "collection_mode": collection_mode,
            "raw_boxscore_rows": len(box_rows),
            "raw_pbp_rows": len(pbp_rows),
            "pbp_enabled": enable_pbp,
            "pbp_stints_total": total_pbp_stints,
            "player_endpoint_calls": enrichment[0],
            "player_endpoint_successes": enrichment[1],
            "player_endpoint_available": enrichment[2],
        }
        if enrichment[2]:
            payload["source"] = "cbbpy_schedule_boxscore_player_endpoint"

        self._save_cache(cache_name, payload)
        return payload

    # ------------------------------------------------------------------
    # Data collection (box scores + optional PBP)
    # ------------------------------------------------------------------

    def _collect_box_and_pbp_rows(
        self, scraper, year: int, *, enable_pbp: bool = False,
    ) -> Tuple[List[Dict], List[Dict], str]:
        """Collect box-score rows and optionally PBP rows.

        Returns ``(box_rows, pbp_rows, collection_mode)``.
        When *enable_pbp* is False, *pbp_rows* is always ``[]``.
        """
        max_games = self._safe_int(os.getenv("CBBPY_ROSTER_MAX_GAMES"), 0)
        force_schedule = str(os.getenv("CBBPY_ROSTER_FORCE_SCHEDULE", "")).strip().lower() in {
            "1", "true", "yes", "y",
        }
        get_games_season = getattr(scraper, "get_games_season", None)

        # Fast path: season endpoint (PBP disabled or bulk fetch)
        # Wrap in thread-based timeout — cbbpy's internal requests.get()
        # has no timeout and can hang indefinitely on ESPN.
        if callable(get_games_season) and not force_schedule and max_games <= 0:
            try:
                data = self._run_with_timeout(
                    get_games_season, args=(year,),
                    kwargs={"info": False, "box": True, "pbp": enable_pbp},
                    timeout=600,
                )
            except TypeError:
                try:
                    data = self._run_with_timeout(
                        get_games_season, args=(year,), timeout=600,
                    )
                except Exception:
                    data = None
            except Exception:
                data = None

            box_rows = self._extract_box_rows(data)
            pbp_rows = self._extract_pbp_rows(data) if enable_pbp else []
            if box_rows:
                if enable_pbp:
                    logger.info(
                        "Season endpoint returned %d box rows and %d PBP rows",
                        len(box_rows), len(pbp_rows),
                    )
                return box_rows, pbp_rows, "season_endpoint"

        # Slow path: game-by-game via schedule
        box_rows, pbp_rows = self._collect_rows_via_schedule(
            scraper, year, enable_pbp=enable_pbp,
        )
        if box_rows:
            return box_rows, pbp_rows, "schedule_game_endpoints"
        return [], [], "none"

    def _extract_box_rows(self, data) -> List[Dict]:
        if isinstance(data, tuple):
            box = data[1] if len(data) > 1 else None
            return self._frame_to_records(box)
        return self._frame_to_records(data)

    def _extract_pbp_rows(self, data) -> List[Dict]:
        """Extract PBP rows from a cbbpy result tuple."""
        if isinstance(data, tuple) and len(data) > 2:
            return self._frame_to_records(data[2])
        return []

    def _collect_rows_via_schedule(
        self, scraper, year: int, *, enable_pbp: bool = False,
    ) -> Tuple[List[Dict], List[Dict]]:
        get_game = getattr(scraper, "get_game", None)
        get_boxscore = getattr(scraper, "get_game_boxscore", None)

        box_rows: List[Dict] = []
        pbp_rows: List[Dict] = []
        seen_game_ids: set = set()
        max_games = self._safe_int(os.getenv("CBBPY_ROSTER_MAX_GAMES"), 0)

        for day in self._season_dates(year):
            if max_games > 0 and len(seen_game_ids) >= max_games:
                break
            try:
                # Use lightweight ESPN API instead of cbbpy's get_game_ids
                # which calls requests.get() with no timeout.
                game_ids = self._scrape_game_ids_for_date(day.isoformat())
            except Exception:
                continue
            if not isinstance(game_ids, list):
                continue

            for game_id in game_ids:
                gid = str(game_id).strip()
                if not gid or gid in seen_game_ids:
                    continue
                seen_game_ids.add(gid)
                if max_games > 0 and len(seen_game_ids) > max_games:
                    break
                game_box, game_pbp = self._fetch_single_game(
                    get_boxscore, get_game, gid, enable_pbp=enable_pbp,
                )
                if game_box:
                    box_rows.extend(game_box)
                if game_pbp:
                    pbp_rows.extend(game_pbp)
        return box_rows, pbp_rows

    def _fetch_single_game(
        self, get_boxscore, get_game, game_id: str, *, enable_pbp: bool = False,
    ) -> Tuple[List[Dict], List[Dict]]:
        """Fetch box score (and optionally PBP) for a single game."""
        # If PBP not needed, use the simpler boxscore endpoint first
        if not enable_pbp and callable(get_boxscore):
            try:
                return self._frame_to_records(
                    rate_limited_call(get_boxscore, game_id, delay=1.0, max_retries=2),
                ), []
            except Exception:
                pass

        if callable(get_game):
            try:
                data = rate_limited_call(
                    get_game, game_id, info=False, box=True, pbp=enable_pbp,
                    delay=1.0, max_retries=2,
                )
            except TypeError:
                try:
                    data = rate_limited_call(get_game, game_id, delay=1.0, max_retries=2)
                except Exception:
                    data = None
            except Exception:
                data = None
            if data is not None:
                box = self._extract_box_rows(data)
                pbp = self._extract_pbp_rows(data) if enable_pbp else []
                return box, pbp

        # Last resort: boxscore-only endpoint
        if callable(get_boxscore):
            try:
                return self._frame_to_records(
                    rate_limited_call(get_boxscore, game_id, delay=1.0, max_retries=2),
                ), []
            except Exception:
                pass
        return [], []

    def _build_payload(self, year: int, box_rows: List[Dict], *, pbp_rows: Optional[List[Dict]] = None) -> Dict:
        teams: Dict[str, Dict] = {}
        game_team_stats: Dict[Tuple[str, str], Dict] = {}
        game_teams: Dict[str, set] = defaultdict(set)

        for row in box_rows:
            team_name = str(row.get("team") or row.get("team_name") or "").strip()
            player_name = str(row.get("player") or row.get("player_name") or "").strip()
            game_id = str(row.get("game_id") or row.get("id") or "").strip()
            if not team_name or not player_name or not game_id:
                continue
            if player_name.lower() in {"team", "totals", "team totals"}:
                continue

            team_id = self._team_id(team_name)
            team_bucket = teams.setdefault(
                team_id,
                {
                    "team_id": team_id,
                    "team_name": team_name,
                    "players": {},
                    "stints": [],
                    "team_totals": {
                        "fga": 0.0,
                        "fta": 0.0,
                        "turnovers": 0.0,
                        "minutes": 0.0,
                        "possessions": 0.0,
                    },
                },
            )

            player_id = str(row.get("player_id") or "").strip()
            if not player_id:
                player_id = self._player_id(team_id, player_name)

            player_bucket = team_bucket["players"].setdefault(
                player_id,
                {
                    "player_id": player_id,
                    "name": player_name,
                    "team_id": team_id,
                    "position_raw": str(row.get("position") or "G"),
                    "games": set(),
                    "games_started": 0,
                    "minutes": 0.0,
                    "pts": 0.0,
                    "reb": 0.0,
                    "ast": 0.0,
                    "stl": 0.0,
                    "blk": 0.0,
                    "to": 0.0,
                    "fga": 0.0,
                    "fgm": 0.0,
                    "fg3m": 0.0,
                    "fta": 0.0,
                    "oreb": 0.0,
                    "dreb": 0.0,
                    "pf": 0.0,
                },
            )

            minutes = self._parse_minutes(row.get("min"))
            player_bucket["games"].add(game_id)
            player_bucket["games_started"] += 1 if self._to_bool(row.get("starter")) else 0
            player_bucket["minutes"] += minutes
            player_bucket["pts"] += self._to_float(row.get("pts"))
            player_bucket["reb"] += self._to_float(row.get("reb"))
            player_bucket["ast"] += self._to_float(row.get("ast"))
            player_bucket["stl"] += self._to_float(row.get("stl"))
            player_bucket["blk"] += self._to_float(row.get("blk"))
            player_bucket["to"] += self._to_float(row.get("to"))
            player_bucket["fga"] += self._to_float(row.get("fga"))
            player_bucket["fgm"] += self._to_float(row.get("fgm"))
            player_bucket["fg3m"] += self._to_float(row.get("3pm"))
            player_bucket["fta"] += self._to_float(row.get("fta"))
            player_bucket["oreb"] += self._to_float(row.get("oreb"))
            player_bucket["dreb"] += self._to_float(row.get("dreb"))
            player_bucket["pf"] += self._to_float(row.get("pf"))

            game_key = (game_id, team_id)
            game_stats = game_team_stats.setdefault(
                game_key,
                {
                    "game_id": game_id,
                    "team_id": team_id,
                    "team_name": team_name,
                    "player_rows": 0,
                    "points": 0.0,
                    "fga": 0.0,
                    "fta": 0.0,
                    "turnovers": 0.0,
                    "oreb": 0.0,
                    "player_minutes": {},
                },
            )
            game_stats["player_rows"] += 1
            game_stats["points"] += self._to_float(row.get("pts"))
            game_stats["fga"] += self._to_float(row.get("fga"))
            game_stats["fta"] += self._to_float(row.get("fta"))
            game_stats["turnovers"] += self._to_float(row.get("to"))
            game_stats["oreb"] += self._to_float(row.get("oreb"))
            game_stats["player_minutes"][player_id] = game_stats["player_minutes"].get(player_id, 0.0) + minutes

            game_teams[game_id].add(team_id)

        # Try PBP-derived stints first; fall back to synthetic box-score stints.
        pbp_stint_count = 0
        if pbp_rows:
            pbp_stint_count = self._attach_pbp_stints(
                teams, pbp_rows or [], box_rows, game_teams,
            )
        if pbp_stint_count == 0:
            self._attach_stints(teams, game_team_stats, game_teams)
        else:
            # Still need team_totals populated for usage rate calculations.
            self._accumulate_team_totals(teams, game_team_stats)

        out_teams: List[Dict] = []
        for team_id, team in teams.items():
            player_rows: List[Dict] = []
            team_totals = team["team_totals"]
            team_usage_denom = team_totals["fga"] + 0.44 * team_totals["fta"] + team_totals["turnovers"]
            team_minutes = max(team_totals["minutes"], 1.0)

            for p in team["players"].values():
                games_played = max(len(p["games"]), 1)
                minutes = p["minutes"]
                minutes_per_game = minutes / games_played
                fga = p["fga"]
                fta = p["fta"]
                turnovers = p["to"]
                usage_num = fga + 0.44 * fta + turnovers
                usage_rate = 100.0 * usage_num / max(team_usage_denom, 1.0)
                raw_impact = (
                    p["pts"]
                    + 0.7 * p["ast"]
                    + 0.8 * p["oreb"]
                    + 0.6 * p["dreb"]
                    + 1.2 * p["stl"]
                    + 1.0 * p["blk"]
                    - 0.9 * turnovers
                    - 0.35 * p["pf"]
                )
                bpm = raw_impact / games_played
                minute_share = minutes / team_minutes
                warp = max(0.0, bpm * minute_share / 12.0)
                true_shooting = p["pts"] / max(2.0 * (fga + 0.44 * fta), 1e-6)
                efg = (p["fgm"] + 0.5 * p["fg3m"]) / max(fga, 1e-6)

                player_rows.append(
                    {
                        "player_id": p["player_id"],
                        "name": p["name"],
                        "position": self._normalize_position(p["position_raw"]),
                        "minutes_per_game": round(minutes_per_game, 3),
                        "games_played": games_played,
                        "games_started": int(p["games_started"]),
                        # cbbpy boxscore-derived payload does not include possession RAPM estimates.
                        "rapm_offensive": None,
                        "rapm_defensive": None,
                        "warp": round(warp, 4),
                        "box_plus_minus": round(bpm, 4),
                        "usage_rate": round(usage_rate, 4),
                        "true_shooting_pct": round(true_shooting, 4),
                        "effective_fg_pct": round(efg, 4),
                        "points_per_game": round(p["pts"] / games_played, 3),
                        "rebounds_per_game": round(p["reb"] / games_played, 3),
                        "assists_per_game": round(p["ast"] / games_played, 3),
                        "steals_per_game": round(p["stl"] / games_played, 3),
                        "blocks_per_game": round(p["blk"] / games_played, 3),
                        "turnovers_per_game": round(turnovers / games_played, 3),
                        "injury_status": "healthy",
                        "is_transfer": False,
                        "eligibility_year": 1,
                    }
                )

            if not player_rows:
                continue
            player_rows.sort(key=lambda row: (-float(row.get("minutes_per_game", 0.0)), row.get("name", "")))
            team_stints = team.get("stints", [])
            if not team_stints:
                fallback_players = [row["player_id"] for row in player_rows[:5]]
                if fallback_players:
                    team_stints = [{"players": fallback_players, "plus_minus": 0.0, "possessions": 70.0}]

            out_teams.append(
                {
                    "team_id": team_id,
                    "team_name": team["team_name"],
                    "players": player_rows,
                    "stints": team_stints,
                }
            )

        out_teams.sort(key=lambda row: row.get("team_name", ""))
        return {"year": year, "teams": out_teams}

    def _attach_stints(
        self,
        teams: Dict[str, Dict],
        game_team_stats: Dict[Tuple[str, str], Dict],
        game_teams: Dict[str, set],
    ) -> None:
        by_game: Dict[str, List[Dict]] = defaultdict(list)
        for (game_id, _team_id), stats in game_team_stats.items():
            by_game[game_id].append(stats)

        for game_id, stats_rows in by_game.items():
            if len(stats_rows) < 2:
                continue
            stats_rows = sorted(stats_rows, key=lambda row: (-int(row["player_rows"]), row["team_name"]))[:2]
            first, second = stats_rows
            for team, opp in ((first, second), (second, first)):
                team_id = team["team_id"]
                team_bucket = teams.get(team_id)
                if team_bucket is None:
                    continue
                possessions = team["fga"] - team["oreb"] + team["turnovers"] + 0.475 * team["fta"]
                team_bucket["team_totals"]["fga"] += team["fga"]
                team_bucket["team_totals"]["fta"] += team["fta"]
                team_bucket["team_totals"]["turnovers"] += team["turnovers"]
                team_bucket["team_totals"]["minutes"] += sum(team["player_minutes"].values())
                team_bucket["team_totals"]["possessions"] += max(possessions, 0.0)

                top_lineup = sorted(team["player_minutes"].items(), key=lambda item: (-item[1], item[0]))[:5]
                lineup_ids = [pid for pid, _ in top_lineup if pid]
                if not lineup_ids:
                    continue
                team_bucket["stints"].append(
                    {
                        "game_id": game_id,
                        "players": lineup_ids,
                        "plus_minus": round(team["points"] - opp["points"], 4),
                        "possessions": round(max(possessions, 1.0), 4),
                    }
                )

    def _accumulate_team_totals(
        self,
        teams: Dict[str, Dict],
        game_team_stats: Dict[Tuple[str, str], Dict],
    ) -> None:
        """Populate team_totals without creating synthetic stints."""
        for (_game_id, _tid), stats in game_team_stats.items():
            team_bucket = teams.get(stats["team_id"])
            if team_bucket is None:
                continue
            team_bucket["team_totals"]["fga"] += stats["fga"]
            team_bucket["team_totals"]["fta"] += stats["fta"]
            team_bucket["team_totals"]["turnovers"] += stats["turnovers"]
            team_bucket["team_totals"]["minutes"] += sum(stats["player_minutes"].values())
            possessions = stats["fga"] - stats["oreb"] + stats["turnovers"] + 0.475 * stats["fta"]
            team_bucket["team_totals"]["possessions"] += max(possessions, 0.0)

    # ------------------------------------------------------------------
    # PBP-derived stint building
    # ------------------------------------------------------------------

    # Regex to detect ESPN substitution descriptions.
    # Common patterns: "Player A enters the game for Player B"
    _SUB_RE = re.compile(
        r"^(?P<player_in>.+?)\s+enters\s+the\s+game\s+for\s+(?P<player_out>.+?)\s*$",
        re.IGNORECASE,
    )

    def _attach_pbp_stints(
        self,
        teams: Dict[str, Dict],
        pbp_rows: List[Dict],
        box_rows: List[Dict],
        game_teams: Dict[str, set],
    ) -> int:
        """Build lineup stints from PBP substitution events.

        Returns the total number of stints created across all teams.
        Falls back gracefully when PBP data lacks substitution events.
        """
        if not pbp_rows:
            return 0

        # Group PBP and box rows by game_id.
        pbp_by_game: Dict[str, List[Dict]] = defaultdict(list)
        for row in pbp_rows:
            gid = str(row.get("game_id") or "").strip()
            if gid:
                pbp_by_game[gid].append(row)

        box_by_game_team: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
        for row in box_rows:
            gid = str(row.get("game_id") or row.get("id") or "").strip()
            team_name = str(row.get("team") or row.get("team_name") or "").strip()
            player_name = str(row.get("player") or row.get("player_name") or "").strip()
            if gid and team_name and player_name.lower() not in {"team", "totals", "team totals"}:
                box_by_game_team[gid][team_name].append(row)

        total_stints = 0
        for game_id, game_pbp in pbp_by_game.items():
            game_stints = self._build_game_stints(
                game_id, game_pbp, box_by_game_team.get(game_id, {}),
            )
            for team_id, stints in game_stints.items():
                team_bucket = teams.get(team_id)
                if team_bucket is not None:
                    team_bucket["stints"].extend(stints)
                    total_stints += len(stints)

        if total_stints > 0:
            logger.info("PBP stint builder created %d stints across all teams", total_stints)
        return total_stints

    def _build_game_stints(
        self,
        game_id: str,
        pbp_rows: List[Dict],
        box_by_team: Dict[str, List[Dict]],
    ) -> Dict[str, List[Dict]]:
        """Build stints for a single game from PBP events.

        Returns ``{team_id: [stint_dict, ...]}``.
        """
        if not pbp_rows or not box_by_team:
            return {}

        # Build name→player_id mapping and identify starters per team.
        team_name_map: Dict[str, str] = {}  # team_name -> team_id
        name_to_pid: Dict[str, Dict[str, str]] = {}  # team_name -> {normalized_name: player_id}
        starters_by_team: Dict[str, List[str]] = {}  # team_name -> [player_ids]

        for team_name, players in box_by_team.items():
            team_id = self._team_id(team_name)
            team_name_map[team_name] = team_id
            pid_map: Dict[str, str] = {}
            starter_ids: List[str] = []

            for row in players:
                player_name = str(row.get("player") or row.get("player_name") or "").strip()
                player_id = str(row.get("player_id") or "").strip()
                if not player_id:
                    player_id = self._player_id(team_id, player_name)

                norm_name = self._normalize_player_name(player_name)
                pid_map[norm_name] = player_id

                # Also map last name for fuzzy matching.
                parts = player_name.strip().split()
                if len(parts) > 1:
                    pid_map[self._normalize_player_name(parts[-1])] = player_id

                if self._to_bool(row.get("starter")):
                    starter_ids.append(player_id)

            name_to_pid[team_name] = pid_map
            starters_by_team[team_name] = starter_ids[:5]

        # Resolve PBP play_team to our team_name keys.
        # PBP play_team may not exactly match box score team names.
        pbp_team_resolve: Dict[str, str] = {}
        for team_name in box_by_team:
            pbp_team_resolve[team_name.lower()] = team_name
            # Also try abbreviations/partial matches.
            for word in team_name.split():
                if len(word) > 3:
                    pbp_team_resolve[word.lower()] = team_name

        def resolve_team(play_team: str) -> Optional[str]:
            if not play_team:
                return None
            key = play_team.strip().lower()
            if key in pbp_team_resolve:
                return pbp_team_resolve[key]
            # Fuzzy: check if any known team name is contained in play_team.
            for known_lower, known in pbp_team_resolve.items():
                if known_lower in key or key in known_lower:
                    return known
            return None

        def resolve_player(team_name: str, player_name: str) -> Optional[str]:
            pid_map = name_to_pid.get(team_name, {})
            norm = self._normalize_player_name(player_name)
            if norm in pid_map:
                return pid_map[norm]
            # Try last name only.
            parts = player_name.strip().split()
            if len(parts) > 1:
                last = self._normalize_player_name(parts[-1])
                if last in pid_map:
                    return pid_map[last]
            return None

        # Track current on-court lineups per team.
        current_lineup: Dict[str, set] = {}
        for team_name, starter_ids in starters_by_team.items():
            current_lineup[team_name] = set(starter_ids)

        # Track stint boundaries.
        # Each stint: {team_name, lineup (frozenset), start_score, end_score, possessions}
        active_stints: Dict[str, Dict] = {}
        completed_stints: Dict[str, List[Dict]] = defaultdict(list)

        def get_team_score(row: Dict, team_name: str) -> float:
            """Get the score for a team from a PBP row."""
            home_score = self._to_float(row.get("home_score"))
            away_score = self._to_float(row.get("away_score"))
            # We don't know which is home/away reliably, so we track
            # the score tuple and compute delta.
            return home_score, away_score

        def close_stint(team_name: str, home_score: float, away_score: float) -> None:
            stint = active_stints.get(team_name)
            if stint is None or not stint["lineup"]:
                return
            start_home, start_away = stint["start_score"]
            delta_home = home_score - start_home
            delta_away = away_score - start_away
            # Determine which score belongs to this team.
            plus_minus = self._compute_stint_plus_minus(
                team_name, box_by_team, delta_home, delta_away,
            )
            possessions = max(stint["possessions"], 1.0)
            completed_stints[team_name].append({
                "game_id": game_id,
                "players": sorted(stint["lineup"]),
                "plus_minus": round(plus_minus, 4),
                "possessions": round(possessions, 4),
            })

        def open_stint(team_name: str, home_score: float, away_score: float) -> None:
            active_stints[team_name] = {
                "lineup": set(current_lineup.get(team_name, set())),
                "start_score": (home_score, away_score),
                "possessions": 0.0,
            }

        # Initialize stints for each team.
        for team_name in box_by_team:
            open_stint(team_name, 0.0, 0.0)

        # Process PBP events chronologically.
        sub_count = 0
        for row in pbp_rows:
            play_desc = str(row.get("play_desc") or "").strip()
            play_team_raw = str(row.get("play_team") or "").strip()
            play_type = str(row.get("play_type") or "").strip().lower()
            home_score = self._to_float(row.get("home_score"))
            away_score = self._to_float(row.get("away_score"))

            resolved_team = resolve_team(play_team_raw)

            # Check for substitution event.
            sub_match = self._SUB_RE.match(play_desc)
            if sub_match and resolved_team:
                player_in_name = sub_match.group("player_in").strip()
                player_out_name = sub_match.group("player_out").strip()
                player_in_id = resolve_player(resolved_team, player_in_name)
                player_out_id = resolve_player(resolved_team, player_out_name)

                if player_in_id and player_out_id:
                    # Close current stint for this team.
                    close_stint(resolved_team, home_score, away_score)

                    # Update lineup.
                    lineup = current_lineup.get(resolved_team, set())
                    lineup.discard(player_out_id)
                    lineup.add(player_in_id)
                    current_lineup[resolved_team] = lineup
                    sub_count += 1

                    # Open new stint.
                    open_stint(resolved_team, home_score, away_score)

            # Count possession-ending events for the active stint.
            if resolved_team and resolved_team in active_stints:
                stint = active_stints[resolved_team]
                if self._is_possession_ending(play_type, play_desc, row):
                    stint["possessions"] += 1.0

        # Close final stints.
        for team_name in box_by_team:
            last_home = self._to_float(pbp_rows[-1].get("home_score")) if pbp_rows else 0.0
            last_away = self._to_float(pbp_rows[-1].get("away_score")) if pbp_rows else 0.0
            close_stint(team_name, last_home, last_away)

        if sub_count == 0:
            # No substitutions found — PBP data doesn't contain sub events.
            return {}

        # Convert to team_id keyed result.
        result: Dict[str, List[Dict]] = {}
        for team_name, stints in completed_stints.items():
            team_id = team_name_map.get(team_name)
            if team_id and stints:
                # Filter out trivially short stints.
                meaningful = [s for s in stints if s["possessions"] >= 1.0]
                if meaningful:
                    result[team_id] = meaningful

        logger.info(
            "Game %s: parsed %d substitutions, produced %d stints",
            game_id, sub_count, sum(len(s) for s in result.values()),
        )
        return result

    def _compute_stint_plus_minus(
        self,
        team_name: str,
        box_by_team: Dict[str, List[Dict]],
        delta_home: float,
        delta_away: float,
    ) -> float:
        """Determine +/- for a team from home/away score deltas.

        Uses the first box row's team name to guess home/away alignment.
        Since we can't always determine home vs away from PBP alone, we
        use a heuristic: if team is listed first in the box score, treat
        as home.
        """
        team_names = list(box_by_team.keys())
        if not team_names:
            return delta_home - delta_away
        # First team in box score is typically home.
        if team_name == team_names[0]:
            return delta_home - delta_away
        return delta_away - delta_home

    @staticmethod
    def _is_possession_ending(play_type: str, play_desc: str, row: Dict) -> bool:
        """Return True if this play ends a possession."""
        pt = play_type.lower()
        if "turnover" in pt:
            return True
        if "made" in pt or ("shooting" in pt and row.get("scoring_play")):
            return True
        # Defensive rebound after a miss ends the opponent's possession.
        if "rebound" in pt and "defensive" in play_desc.lower():
            return True
        # End of free throw sequence (last FT made or missed).
        if "free throw" in pt:
            desc_lower = play_desc.lower()
            if "2 of 2" in desc_lower or "3 of 3" in desc_lower or "1 of 1" in desc_lower:
                return True
        return False

    @staticmethod
    def _normalize_player_name(name: str) -> str:
        """Normalize a player name for fuzzy matching."""
        return re.sub(r"[^a-z]", "", name.lower())

    def _enrich_from_player_endpoint(self, scraper, payload: Dict) -> Tuple[int, int, bool]:
        get_player_info = getattr(scraper, "get_player_info", None)
        if not callable(get_player_info):
            return 0, 0, False

        max_calls = self._safe_int(os.getenv("CBBPY_PLAYER_INFO_MAX_CALLS"), 0)
        players: List[Dict] = []
        for team in payload.get("teams", []):
            for player in team.get("players", []):
                players.append(player)
        players.sort(key=lambda row: (-float(row.get("minutes_per_game", 0.0)), row.get("player_id", "")))

        calls = 0
        successes = 0
        cache = {}
        for player in players:
            if max_calls > 0 and calls >= max_calls:
                break
            player_id = str(player.get("player_id") or "").strip()
            if not player_id:
                continue
            if not player_id.isdigit():
                continue

            if player_id in cache:
                profile = cache[player_id]
            else:
                calls += 1
                try:
                    raw = rate_limited_call(
                        get_player_info, player_id, delay=1.0, max_retries=2,
                    )
                    profile = self._frame_to_single_row(raw)
                except Exception:
                    profile = None
                cache[player_id] = profile

            if not profile:
                continue
            successes += 1
            self._apply_player_profile(player, profile)

        return calls, successes, True

    def _apply_player_profile(self, player: Dict, profile: Dict) -> None:
        position = profile.get("position") or profile.get("pos")
        if position:
            player["position"] = self._normalize_position(str(position))

        class_value = (
            profile.get("class")
            or profile.get("year")
            or profile.get("class_year")
            or profile.get("experience")
            or profile.get("eligibility")
        )
        eligibility = self._parse_eligibility_year(class_value)
        if eligibility is not None:
            player["eligibility_year"] = eligibility

    @staticmethod
    def _frame_to_records(obj) -> List[Dict]:
        if obj is None:
            return []
        if isinstance(obj, list):
            return [row for row in obj if isinstance(row, dict)]
        if isinstance(obj, dict):
            return [obj]
        to_dict = getattr(obj, "to_dict", None)
        if callable(to_dict):
            try:
                rows = to_dict("records")
                if isinstance(rows, list):
                    return [row for row in rows if isinstance(row, dict)]
            except Exception:
                pass
        return []

    def _frame_to_single_row(self, obj) -> Optional[Dict]:
        rows = self._frame_to_records(obj)
        if not rows:
            return None
        return rows[0]

    @staticmethod
    def _normalize_position(value: str) -> str:
        raw = (value or "").strip().upper()
        if raw in {"PG", "POINT GUARD"}:
            return "PG"
        if raw in {"SG", "SHOOTING GUARD"}:
            return "SG"
        if raw in {"SF", "SMALL FORWARD"}:
            return "SF"
        if raw in {"PF", "POWER FORWARD"}:
            return "PF"
        if raw in {"C", "CENTER"}:
            return "C"
        if raw.startswith("G"):
            return "PG"
        if raw.startswith("F"):
            return "SF"
        if raw.startswith("C"):
            return "C"
        return "PG"

    @staticmethod
    def _parse_eligibility_year(value) -> Optional[int]:
        if value is None:
            return None
        raw = str(value).strip().upper()
        mapping = {
            "FR": 1,
            "FRESHMAN": 1,
            "RS FR": 1,
            "RFR": 1,
            "SO": 2,
            "SOPHOMORE": 2,
            "RS SO": 2,
            "RS-SO": 2,
            "JR": 3,
            "JUNIOR": 3,
            "RS JR": 3,
            "RS-JR": 3,
            "SR": 4,
            "SENIOR": 4,
            "GR": 5,
            "GRAD": 5,
            "GRADUATE": 5,
            "5TH": 5,
            "5TH YEAR": 5,
        }
        if raw in mapping:
            return mapping[raw]
        raw = raw.replace("-", " ").replace(".", "").strip()
        if raw in mapping:
            return mapping[raw]
        try:
            number = int(float(raw))
            if number > 0:
                return number
        except (TypeError, ValueError):
            pass
        return None

    @staticmethod
    def _parse_minutes(value) -> float:
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        raw = str(value).strip()
        if ":" in raw:
            try:
                minute_part, second_part = raw.split(":", 1)
                minutes = float(minute_part)
                seconds = float(second_part)
                return minutes + seconds / 60.0
            except (TypeError, ValueError):
                return 0.0
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _to_float(value) -> float:
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _to_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    @staticmethod
    def _safe_int(value, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _load_cache(self, filename: str) -> Optional[Dict]:
        if not self.cache_dir:
            return None
        path = self.cache_dir / filename
        if not path.exists():
            return None
        try:
            with open(path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None

    def _save_cache(self, filename: str, payload: Dict) -> None:
        if not self.cache_dir:
            return
        path = self.cache_dir / filename
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def _import_module(module_name: str):
        try:
            return importlib.import_module(module_name)
        except Exception:
            return None

    @staticmethod
    def _team_id(name: str) -> str:
        return "".join(ch.lower() if ch.isalnum() else "_" for ch in (name or "")).strip("_")

    @staticmethod
    def _player_id(team_id: str, name: str) -> str:
        normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in (name or "")).strip("_")
        return f"{team_id}_{normalized or 'player'}"

    def _season_dates(self, season: int) -> Iterable[date]:
        start = date(season - 1, 11, 1)
        end = date(season, 5, 1)
        current = start
        today = datetime.now(timezone.utc).date()
        stop = min(end, today)
        while current <= stop:
            yield current
            current += timedelta(days=1)

    @staticmethod
    def _scrape_game_ids_for_date(day_str: str, http_timeout: int = 15,
                                  session=None) -> List:
        """Lightweight ESPN API call for game IDs on a single date.

        Bypasses cbbpy's ``get_game_ids`` which uses ``requests.get()``
        with no timeout.  Pass a ``requests.Session`` for connection
        pooling across calls.
        """
        import requests as _requests

        d = day_str.replace("-", "")
        api_url = (
            f"https://site.api.espn.com/apis/site/v2/sports/basketball/"
            f"mens-college-basketball/scoreboard?dates={d}&groups=50&limit=200"
        )
        getter = session or _requests
        if session is None:
            resp = getter.get(api_url, headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            }, timeout=http_timeout)
        else:
            resp = getter.get(api_url, timeout=http_timeout)
        resp.raise_for_status()
        data = resp.json()
        return [str(e["id"]) for e in data.get("events", []) if "id" in e]

    @staticmethod
    def _run_with_timeout(fn, args=(), kwargs=None, timeout=120):
        """Run *fn* in a thread with a timeout to prevent indefinite hangs."""
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

        kwargs = kwargs or {}
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(fn, *args, **kwargs)
            try:
                return future.result(timeout=timeout)
            except FuturesTimeout:
                raise TimeoutError(
                    f"{getattr(fn, '__name__', fn)} timed out after {timeout}s"
                )
