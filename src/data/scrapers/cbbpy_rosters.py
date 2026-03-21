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
        payload["metadata"] = {
            "collection_mode": collection_mode,
            "raw_boxscore_rows": len(box_rows),
            "raw_pbp_rows": len(pbp_rows),
            "pbp_enabled": enable_pbp,
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

        # Populate team_totals for usage rate calculations.
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
                # Box Plus-Minus: weighted per-game stats CENTERED on D1 averages
                # so an average rotation player produces BPM ≈ 0.
                # D1 rotation-player averages (≥5 mpg, ≥3 gp, from cbbpy 2024):
                # 7.1 ppg, 1.3 apg, 1.0 oreb/g, 2.2 dreb/g, 0.65 stl/g, 0.33 blk/g,
                # 1.15 tov/g, 1.7 pf/g.
                _AVG_IMPACT = (
                    7.1 + 0.7 * 1.3 + 0.8 * 1.0 + 0.6 * 2.2
                    + 1.2 * 0.65 + 1.0 * 0.33 - 0.9 * 1.15 - 0.35 * 1.7
                )  # ≈ 9.30
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
                bpm = raw_impact / games_played - _AVG_IMPACT
                minute_share = minutes / team_minutes
                warp = max(0.0, bpm * minute_share * games_played / 300.0)
                true_shooting = p["pts"] / max(2.0 * (fga + 0.44 * fta), 1e-6)
                efg = (p["fgm"] + 0.5 * p["fg3m"]) / max(fga, 1e-6)

                # Box-score RAPM estimate: split raw impact into O/D components
                # and scale to per-100-possession RAPM range (~-5 to +8).
                rapm_off, rapm_def = self._estimate_rapm_from_box_score(
                    p, games_played, minutes_per_game, true_shooting,
                )

                player_rows.append(
                    {
                        "player_id": p["player_id"],
                        "name": p["name"],
                        "position": self._normalize_position(p["position_raw"]),
                        "minutes_per_game": round(minutes_per_game, 3),
                        "games_played": games_played,
                        "games_started": int(p["games_started"]),
                        "rapm_offensive": round(rapm_off, 4),
                        "rapm_defensive": round(rapm_def, 4),
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

            out_teams.append(
                {
                    "team_id": team_id,
                    "team_name": team["team_name"],
                    "players": player_rows,
                }
            )

        out_teams.sort(key=lambda row: row.get("team_name", ""))
        return {"year": year, "teams": out_teams}

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

    @staticmethod
    def _estimate_rapm_from_box_score(
        p: Dict, games_played: int, minutes_per_game: float, true_shooting: float,
    ) -> Tuple[float, float]:
        """Estimate offensive/defensive RAPM from box-score stats.

        Uses per-minute production rates scaled to per-100-possession RAPM
        range.  Coefficients are calibrated so that an average D1 player
        produces ~0 and an elite player reaches ~+6.

        Returns (rapm_offensive, rapm_defensive).
        """
        if minutes_per_game < 5.0 or games_played < 3:
            return 0.0, 0.0

        gp = float(games_played)
        ppg = p["pts"] / gp
        apg = p["ast"] / gp
        orpg = p["oreb"] / gp
        drpg = (p["reb"] - p["oreb"]) / gp
        spg = p["stl"] / gp
        bpg = p["blk"] / gp
        topg = p["to"] / gp
        pfpg = p["pf"] / gp

        # --- Offensive RAPM ---
        # D1 rotation-player averages (≥5 mpg, ≥3 gp, from cbbpy 2024):
        # 7.1 ppg, 1.3 apg, 1.0 oreb/g, 1.15 tov/g, 0.52 TS%
        o_rapm = (
            (ppg - 7.1) * 0.25
            + (apg - 1.3) * 0.55
            + (orpg - 1.0) * 0.35
            - (topg - 1.15) * 0.45
            + (true_shooting - 0.52) * 12.0
        )

        # --- Defensive RAPM ---
        # D1 averages: 2.2 dreb/g, 0.65 stl/g, 0.33 blk/g, 1.7 pf/g
        d_rapm = (
            (drpg - 2.2) * 0.20
            + (spg - 0.65) * 1.2
            + (bpg - 0.33) * 0.9
            - (pfpg - 1.7) * 0.20
        )

        # Weight by minutes share: a 32+ mpg player gets full value;
        # a 10 mpg bench player gets proportional credit.
        min_weight = min(minutes_per_game / 32.0, 1.0)
        return o_rapm * min_weight, d_rapm * min_weight

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
