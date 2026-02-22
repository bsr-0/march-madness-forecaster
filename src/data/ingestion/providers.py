"""Provider adapters that prefer library-backed data over custom scraping."""

from __future__ import annotations

import csv
import importlib
import io
import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class ProviderResult:
    provider: str
    records: List[Dict]


class LibraryProviderHub:
    """Best-effort data provider hub with ordered fallback."""

    DEFAULT_PRIORITIES = {
        "historical_games": ["sportsdataverse", "cbbpy", "sportsipy", "cbbdata"],
        "team_metrics": ["sportsdataverse", "sportsipy", "cbbdata"],
        "torvik": ["barttorvik", "cbbdata"],
    }

    def fetch_historical_games(self, year: int, priority: Optional[List[str]] = None) -> ProviderResult:
        methods = {
            "sportsdataverse": self._from_sportsdataverse_pbp,
            "cbbpy": self._from_cbbpy_pbp,
            "sportsipy": self._from_sportsipy_games_api,
            "cbbdata": self._from_cbbdata_games_api,
        }
        for method in self._ordered_methods("historical_games", methods, priority):
            result = method(year)
            if result.records:
                return result
        return ProviderResult(provider="none", records=[])

    def fetch_team_box_metrics(self, year: int, priority: Optional[List[str]] = None) -> ProviderResult:
        methods = {
            "sportsdataverse": self._from_sportsdataverse_team_box,
            "sportsipy": self._from_sportsipy_team_metrics_api,
            "cbbdata": self._from_cbbdata_team_metrics_api,
        }
        for method in self._ordered_methods("team_metrics", methods, priority):
            result = method(year)
            if result.records:
                return result
        return ProviderResult(provider="none", records=[])

    def fetch_torvik_ratings(self, year: int, priority: Optional[List[str]] = None) -> ProviderResult:
        methods = {
            "barttorvik": self._from_barttorvik_csv,
            "cbbdata": self._from_cbbdata_torvik_api,
        }
        for method in self._ordered_methods("torvik", methods, priority):
            result = method(year)
            if result.records:
                return result
        return ProviderResult(provider="none", records=[])

    def credential_requirements(self) -> Dict[str, List[str]]:
        return {
            "cbbdata_api": ["CBBDATA_API_KEY", "CBBDATA_BASE_URL"],
            "sportsdataverse_py": [],
            "cbbpy": [],
            "sportsipy": [],
            "barttorvik": [],
        }

    def _ordered_methods(
        self,
        data_kind: str,
        methods: Dict[str, Callable[[int], ProviderResult]],
        priority: Optional[List[str]],
    ) -> List[Callable[[int], ProviderResult]]:
        ordered_names = [p.strip().lower() for p in (priority or self.DEFAULT_PRIORITIES[data_kind])]
        resolved: List[Callable[[int], ProviderResult]] = []
        for name in ordered_names:
            method = methods.get(name)
            if method is not None:
                resolved.append(method)
        return resolved

    def _from_sportsdataverse_pbp(self, year: int) -> ProviderResult:
        mbb = self._import_module("sportsdataverse.mbb")
        if mbb is None:
            return ProviderResult("sportsdataverse", [])

        call_candidates = [
            ("load_mbb_pbp", {"seasons": [year]}),
            ("load_mbb_pbp", {"seasons": year}),
            ("load_mbb_pbp", {"year": year}),
            ("espn_mbb_pbp", {"year": year}),
        ]
        for fn_name, kwargs in call_candidates:
            fn = getattr(mbb, fn_name, None)
            if fn is None:
                continue
            try:
                df = fn(**kwargs)
                records = self._frame_to_records(df)
                if records:
                    return ProviderResult("sportsdataverse", records)
            except Exception:
                continue
        return ProviderResult("sportsdataverse", [])

    def _from_sportsdataverse_team_box(self, year: int) -> ProviderResult:
        mbb = self._import_module("sportsdataverse.mbb")
        if mbb is None:
            return ProviderResult("sportsdataverse", [])

        call_candidates = [
            ("load_mbb_team_boxscore", {"seasons": [year]}),
            ("load_mbb_team_boxscore", {"year": year}),
            ("espn_mbb_team_boxscore", {"year": year}),
        ]
        for fn_name, kwargs in call_candidates:
            fn = getattr(mbb, fn_name, None)
            if fn is None:
                continue
            try:
                df = fn(**kwargs)
                records = self._frame_to_records(df)
                if records:
                    return ProviderResult("sportsdataverse", records)
            except Exception:
                continue
        return ProviderResult("sportsdataverse", [])

    def _from_cbbpy_pbp(self, year: int) -> ProviderResult:
        scraper = self._import_module("cbbpy.mens_scraper")
        if scraper is None:
            return ProviderResult("cbbpy", [])

        # CBBpy function names have changed across releases; probe common names.
        for fn_name in ("get_games_season", "get_games_range"):
            fn = getattr(scraper, fn_name, None)
            if fn is None:
                continue
            try:
                if fn_name == "get_games_season":
                    games = fn(year, info=False, box=True, pbp=False)
                else:
                    games = fn(f"{year-1}-11-01", f"{year}-04-15", info=False, box=True, pbp=False)
            except TypeError:
                try:
                    games = fn(year) if fn_name == "get_games_season" else fn(f"{year-1}-11-01", f"{year}-04-15")
                except Exception:
                    continue
            except Exception:
                continue

            game_rows = self._normalize_cbbpy_records(games)
            if not game_rows:
                continue
            return ProviderResult("cbbpy", game_rows)

        return ProviderResult("cbbpy", [])

    def _from_cbbdata_games_api(self, year: int) -> ProviderResult:
        payload = self._fetch_cbbdata_endpoint("CBBDATA_GAMES_URL", year)
        records = payload.get("games") or payload.get("records") or []
        return ProviderResult("cbbdata", records if isinstance(records, list) else [])

    def _from_sportsipy_games_api(self, year: int) -> ProviderResult:
        try:
            from sportsipy.ncaab.teams import Teams
        except ImportError:
            return ProviderResult("sportsipy", [])

        seen = set()
        records = []
        for team in Teams(year):
            team_id = self._normalize_team_name(team.name)
            for game in team.schedule:
                game_id = getattr(game, "boxscore", None)
                if not game_id or game_id in seen:
                    continue
                seen.add(game_id)
                opponent = getattr(game, "opponent_name", "")
                opp_id = self._normalize_team_name(opponent)
                records.append(
                    {
                        "game_id": game_id,
                        "team_id": team_id,
                        "team_name": team.name,
                        "opponent_id": opp_id,
                        "opponent_name": opponent,
                        "team_score": getattr(game, "points", 0),
                        "opponent_score": getattr(game, "opponent_points", 0),
                        "possessions": getattr(game, "possessions", 0),
                    }
                )
        return ProviderResult("sportsipy", records)

    def _from_cbbdata_team_metrics_api(self, year: int) -> ProviderResult:
        payload = self._fetch_cbbdata_endpoint("CBBDATA_TEAM_METRICS_URL", year)
        records = payload.get("teams") or payload.get("records") or []
        return ProviderResult("cbbdata", records if isinstance(records, list) else [])

    def _from_sportsipy_team_metrics_api(self, year: int) -> ProviderResult:
        try:
            from sportsipy.ncaab.teams import Teams
        except ImportError:
            return ProviderResult("sportsipy", [])

        records = []
        for team in Teams(year):
            team_id = self._normalize_team_name(team.name)
            records.append(
                {
                    "team_id": team_id,
                    "team_name": team.name,
                    "name": team.name,
                    "conference": getattr(team, "conference", ""),
                    # Expose both canonical materialization fields and adj_* aliases.
                    "off_rtg": getattr(team, "offensive_rating", None) or getattr(team, "points_per_game", 0.0),
                    "def_rtg": getattr(team, "defensive_rating", None) or getattr(team, "points_against", 0.0),
                    "pace": getattr(team, "pace", None) or getattr(team, "possessions_per_game", 70.0),
                    "adj_offensive_efficiency": getattr(team, "offensive_rating", None)
                    or getattr(team, "points_per_game", 0.0),
                    "adj_defensive_efficiency": getattr(team, "defensive_rating", None)
                    or getattr(team, "points_against", 0.0),
                    "adj_tempo": getattr(team, "pace", None) or getattr(team, "possessions_per_game", 70.0),
                }
            )
        return ProviderResult("sportsipy", records)

    def _from_cbbdata_torvik_api(self, year: int) -> ProviderResult:
        payload = self._fetch_cbbdata_endpoint("CBBDATA_TORVIK_URL", year)
        records = payload.get("teams") or payload.get("records") or []
        return ProviderResult("cbbdata", records if isinstance(records, list) else [])

    def _from_barttorvik_csv(self, year: int) -> ProviderResult:
        url_template = os.getenv("BARTTORVIK_TORVIK_URL")
        if url_template:
            url = url_template.format(year=year)
        else:
            url = f"https://barttorvik.com/{year}_team_results.csv"

        try:
            response = requests.get(url, timeout=45)
            response.raise_for_status()
        except Exception as exc:
            logger.warning("barttorvik request failed for %s: %s", url, exc)
            return ProviderResult("barttorvik", [])

        text = response.text.strip()
        if not text:
            return ProviderResult("barttorvik", [])

        try:
            sample = text[:4096]
            dialect = csv.Sniffer().sniff(sample)
        except Exception:
            dialect = csv.excel

        reader = csv.reader(io.StringIO(text), dialect)
        rows = list(reader)
        if not rows:
            return ProviderResult("barttorvik", [])

        header = [c.strip() for c in rows[0]]
        has_header = any(h and not h.replace(".", "", 1).isdigit() for h in header)
        if not has_header:
            logger.warning("barttorvik CSV appears to lack headers; skipping (set BARTTORVIK_TORVIK_URL to a headered feed).")
            return ProviderResult("barttorvik", [])

        records: List[Dict] = []
        dict_reader = csv.DictReader(io.StringIO(text), dialect=dialect)
        for row in dict_reader:
            record = self._map_barttorvik_row(row)
            if record:
                records.append(record)
        return ProviderResult("barttorvik", records)

    @staticmethod
    def _map_barttorvik_row(row: Dict[str, str]) -> Optional[Dict]:
        if not isinstance(row, dict):
            return None

        def pick(keys: List[str]) -> str:
            for key in keys:
                for k, v in row.items():
                    if k is None:
                        continue
                    if k.strip().lower() == key:
                        return str(v).strip()
            return ""

        def to_float(value: str) -> float:
            if value is None:
                return 0.0
            text = str(value).replace("%", "").strip()
            try:
                return float(text)
            except (TypeError, ValueError):
                return 0.0

        def normalize_rate(value: float) -> float:
            if value > 1.5:
                return value / 100.0
            return value

        name = pick(["team", "team_name", "team name", "school"])
        conf = pick(["conf", "conference"])
        if not name:
            return None

        team_id = "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")

        t_rank = int(to_float(pick(["rank", "rk", "t_rank"]))) if pick(["rank", "rk", "t_rank"]) else 999

        barthag = to_float(pick(["barthag", "bar_thag"]))
        adj_oe = to_float(pick(["adjoe", "adjo", "adj_o", "adj_oe", "adj_off", "adj_offense", "adj_offensive_efficiency"]))
        adj_de = to_float(pick(["adjde", "adjd", "adj_d", "adj_de", "adj_def", "adj_defense", "adj_defensive_efficiency"]))
        adj_t = to_float(pick(["adjt", "adj_t", "tempo", "adj_tempo"]))

        efg = normalize_rate(to_float(pick(["efg", "efg%", "effective_fg_pct", "efg_pct"])))
        to_rate = normalize_rate(to_float(pick(["to%", "to_rate", "turnover_rate", "to_pct"])))
        orb = normalize_rate(to_float(pick(["orb%", "orb", "offensive_reb_rate", "orb_pct"])))
        ftr = normalize_rate(to_float(pick(["ftr", "ft_rate", "free_throw_rate", "ft_rate_pct"])))

        record = {
            "team_id": team_id,
            "team_name": name,
            "name": name,
            "conference": conf,
            "t_rank": t_rank,
            "barthag": barthag,
            "adj_offensive_efficiency": adj_oe,
            "adj_defensive_efficiency": adj_de,
            "adj_tempo": adj_t,
            "effective_fg_pct": efg,
            "turnover_rate": to_rate,
            "offensive_reb_rate": orb,
            "free_throw_rate": ftr,
        }
        return record

    def _fetch_cbbdata_endpoint(self, url_env: str, year: int) -> Dict:
        url_template = os.getenv(url_env)
        api_key = os.getenv("CBBDATA_API_KEY")
        if not url_template or not api_key:
            return {}

        url = url_template.format(year=year)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "x-api-key": api_key,
        }
        try:
            response = requests.get(url, headers=headers, timeout=45)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict):
                return data
            if isinstance(data, list):
                return {"records": data}
        except Exception as exc:
            logger.warning("cbbdata request failed for %s: %s", url_env, exc)
        return {}

    def _normalize_cbbpy_records(self, obj) -> List[Dict]:
        if not isinstance(obj, tuple):
            return self._frame_to_records(obj)

        info_df = obj[0] if len(obj) > 0 else None
        box_df = obj[1] if len(obj) > 1 else None
        pbp_df = obj[2] if len(obj) > 2 else None

        box_rows = self._frame_to_records(box_df)
        team_games = self._aggregate_cbbpy_box_rows(box_rows)
        if team_games:
            return team_games

        info_rows = self._frame_to_records(info_df)
        if info_rows:
            return info_rows
        return self._frame_to_records(pbp_df)

    def _aggregate_cbbpy_box_rows(self, rows: List[Dict]) -> List[Dict]:
        by_game_team: Dict[str, Dict[str, Dict]] = {}
        for row in rows:
            game_id = str(row.get("game_id") or row.get("id") or "").strip()
            team_name = str(row.get("team") or row.get("team_name") or "").strip()
            player_name = str(row.get("player") or "").strip().lower()
            if not game_id or not team_name:
                continue
            if player_name in {"team", "totals", "team totals"}:
                continue

            game_bucket = by_game_team.setdefault(game_id, {})
            team_id = self._normalize_team_name(team_name)
            stats = game_bucket.setdefault(
                team_id,
                {
                    "team_id": team_id,
                    "team_name": team_name,
                    "team_score": 0.0,
                    "fgm": 0.0,
                    "fga": 0.0,
                    "fg3m": 0.0,
                    "fg3a": 0.0,
                    "fta": 0.0,
                    "turnovers": 0.0,
                    "orb": 0.0,
                    "drb": 0.0,
                    "player_rows": 0,
                },
            )
            stats["team_score"] += self._to_float(row.get("pts"))
            stats["fgm"] += self._to_float(row.get("fgm"))
            stats["fga"] += self._to_float(row.get("fga"))
            stats["fg3m"] += self._to_float(row.get("3pm"))
            stats["fg3a"] += self._to_float(row.get("3pa"))
            stats["fta"] += self._to_float(row.get("fta"))
            stats["turnovers"] += self._to_float(row.get("to"))
            stats["orb"] += self._to_float(row.get("oreb"))
            stats["drb"] += self._to_float(row.get("dreb"))
            stats["player_rows"] += 1

        out: List[Dict] = []
        for game_id, game_teams in by_game_team.items():
            team_list = sorted(game_teams.values(), key=lambda t: (-t["player_rows"], t["team_name"]))
            if len(team_list) < 2:
                continue
            team_list = team_list[:2]
            first, second = team_list
            for team, opp in ((first, second), (second, first)):
                poss = team["fga"] - team["orb"] + team["turnovers"] + 0.475 * team["fta"]
                out.append(
                    {
                        "game_id": game_id,
                        "team_id": team["team_id"],
                        "team_name": team["team_name"],
                        "opponent_id": opp["team_id"],
                        "opponent_name": opp["team_name"],
                        "team_score": int(round(team["team_score"])),
                        "opponent_score": int(round(opp["team_score"])),
                        "possessions": max(float(poss), 0.0),
                        "fgm": team["fgm"],
                        "fga": team["fga"],
                        "fg3m": team["fg3m"],
                        "fg3a": team["fg3a"],
                        "fta": team["fta"],
                        "turnovers": team["turnovers"],
                        "orb": team["orb"],
                        "drb": team["drb"],
                    }
                )
        return out

    @staticmethod
    def _import_module(module_name: str):
        try:
            return importlib.import_module(module_name)
        except Exception:
            return None

    @staticmethod
    def _frame_to_records(obj) -> List[Dict]:
        if obj is None:
            return []
        if isinstance(obj, list):
            return [r for r in obj if isinstance(r, dict)]
        if isinstance(obj, dict):
            return [obj]

        to_dict = getattr(obj, "to_dict", None)
        if callable(to_dict):
            try:
                records = to_dict("records")
                if isinstance(records, list):
                    return [r for r in records if isinstance(r, dict)]
            except Exception:
                pass
        return []

    @staticmethod
    def _normalize_team_name(name: str) -> str:
        if not name:
            return ""
        return "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")

    @staticmethod
    def _to_float(value) -> float:
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
