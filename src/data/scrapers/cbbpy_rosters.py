"""Roster scraper that derives player databases from cbbpy schedule/boxscore/player endpoints."""

from __future__ import annotations

import importlib
import json
import os
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


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

        box_rows, collection_mode = self._collect_box_rows(scraper, year)
        if not box_rows:
            return {}

        payload = self._build_payload(year, box_rows)
        if not payload.get("teams"):
            return {}

        enrichment = self._enrich_from_player_endpoint(scraper, payload)
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()
        payload["source"] = "cbbpy_schedule_boxscore"
        payload["metadata"] = {
            "collection_mode": collection_mode,
            "raw_boxscore_rows": len(box_rows),
            "player_endpoint_calls": enrichment[0],
            "player_endpoint_successes": enrichment[1],
            "player_endpoint_available": enrichment[2],
        }
        if enrichment[2]:
            payload["source"] = "cbbpy_schedule_boxscore_player_endpoint"

        self._save_cache(cache_name, payload)
        return payload

    def _collect_box_rows(self, scraper, year: int) -> Tuple[List[Dict], str]:
        max_games = self._safe_int(os.getenv("CBBPY_ROSTER_MAX_GAMES"), 0)
        force_schedule = str(os.getenv("CBBPY_ROSTER_FORCE_SCHEDULE", "")).strip().lower() in {"1", "true", "yes", "y"}
        get_games_season = getattr(scraper, "get_games_season", None)
        if callable(get_games_season) and not force_schedule and max_games <= 0:
            try:
                data = get_games_season(year, info=False, box=True, pbp=False)
            except TypeError:
                try:
                    data = get_games_season(year)
                except Exception:
                    data = None
            except Exception:
                data = None

            rows = self._extract_box_rows(data)
            if rows:
                return rows, "season_endpoint"

        rows = self._collect_box_rows_via_schedule(scraper, year)
        if rows:
            return rows, "schedule_game_endpoints"
        return [], "none"

    def _extract_box_rows(self, data) -> List[Dict]:
        if isinstance(data, tuple):
            box = data[1] if len(data) > 1 else None
            return self._frame_to_records(box)
        return self._frame_to_records(data)

    def _collect_box_rows_via_schedule(self, scraper, year: int) -> List[Dict]:
        get_game_ids = getattr(scraper, "get_game_ids", None)
        if not callable(get_game_ids):
            return []

        get_boxscore = getattr(scraper, "get_game_boxscore", None)
        get_game = getattr(scraper, "get_game", None)

        rows: List[Dict] = []
        seen_game_ids = set()
        max_games = self._safe_int(os.getenv("CBBPY_ROSTER_MAX_GAMES"), 0)

        for day in self._season_dates(year):
            if max_games > 0 and len(seen_game_ids) >= max_games:
                break
            try:
                game_ids = get_game_ids(day.isoformat())
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
                game_rows = self._fetch_single_game_box_rows(get_boxscore, get_game, gid)
                if game_rows:
                    rows.extend(game_rows)
        return rows

    def _fetch_single_game_box_rows(self, get_boxscore, get_game, game_id: str) -> List[Dict]:
        if callable(get_boxscore):
            try:
                return self._frame_to_records(get_boxscore(game_id))
            except Exception:
                pass

        if callable(get_game):
            try:
                data = get_game(game_id, info=False, box=True, pbp=False)
            except TypeError:
                try:
                    data = get_game(game_id)
                except Exception:
                    data = None
            except Exception:
                data = None
            if data is not None:
                return self._extract_box_rows(data)
        return []

    def _build_payload(self, year: int, box_rows: List[Dict]) -> Dict:
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

        self._attach_stints(teams, game_team_stats, game_teams)

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
                    profile = self._frame_to_single_row(get_player_info(player_id))
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
