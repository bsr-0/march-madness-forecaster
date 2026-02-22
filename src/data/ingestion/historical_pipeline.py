"""Historical game ingestion pipeline focused on 2022-2025 real CBB data."""

from __future__ import annotations

import json
import logging
import re
import signal
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ..scrapers import SportsReferenceScraper, TournamentSeedScraper
from .providers import LibraryProviderHub
from .validators import validate_games_payload, validate_ratings_payload

logger = logging.getLogger(__name__)


@dataclass
class HistoricalIngestionConfig:
    """Configuration for multi-season historical data pulls."""

    start_season: int = 2022
    end_season: int = 2025
    output_dir: str = "data/raw/historical"
    cache_dir: str = "data/raw/cache"

    include_pbp: bool = False
    strict_validation: bool = True
    retry_attempts: int = 2
    per_game_timeout_seconds: int = 25
    max_games_per_season: Optional[int] = None
    include_tournament_context: bool = True
    include_torvik: bool = True
    team_metrics_provider_priority: Optional[List[str]] = None
    torvik_provider_priority: Optional[List[str]] = None


class HistoricalDataPipeline:
    """Collects real historical team/game data for model training."""

    def __init__(self, config: Optional[HistoricalIngestionConfig] = None):
        self.config = config or HistoricalIngestionConfig()
        self.output_dir = Path(self.config.output_dir)
        self.cache_dir = Path(self.config.cache_dir)
        self.providers = LibraryProviderHub()
        self.sports_reference = SportsReferenceScraper(str(self.cache_dir))
        self.tournament_seed_scraper = TournamentSeedScraper(str(self.cache_dir))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict:
        if self.config.start_season > self.config.end_season:
            raise ValueError("start_season must be <= end_season")

        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "start_season": self.config.start_season,
            "end_season": self.config.end_season,
            "artifacts": {},
            "providers": {},
            "validation_errors": {},
            "season_counts": {},
        }

        for season in range(self.config.start_season, self.config.end_season + 1):
            game_payload, game_provider = self._collect_season_games(season)
            game_errors = validate_games_payload({"games": game_payload["games"]})
            self._assert_valid(f"historical_games_{season}", game_errors)
            games_path = self._write_json(f"historical_games_{season}.json", game_payload)

            # Pass game records so the SR scraper can compute def_rtg when the
            # HTML page omits the column.
            all_game_rows = game_payload.get("games", []) + game_payload.get("team_games", [])
            team_payload, team_provider = self._collect_team_metrics(
                season, game_records=all_game_rows,
            )
            team_errors = validate_ratings_payload(team_payload, name_field="team_name")
            self._assert_valid(f"team_metrics_{season}", team_errors)
            teams_path = self._write_json(f"team_metrics_{season}.json", team_payload)

            manifest["artifacts"][str(season)] = {
                "historical_games_json": games_path,
                "team_metrics_json": teams_path,
            }
            manifest["providers"][str(season)] = {
                "historical_games_json": game_provider,
                "team_metrics_json": team_provider,
            }
            manifest["validation_errors"][str(season)] = {
                "historical_games_json": game_errors,
                "team_metrics_json": team_errors,
            }
            manifest["season_counts"][str(season)] = {
                "games": len(game_payload["games"]),
                "team_games": len(game_payload.get("team_games", [])),
                "teams": len(team_payload.get("teams", [])),
            }

            if self.config.include_torvik:
                torvik_payload, torvik_provider = self._collect_torvik(season)
                if torvik_payload.get("teams"):
                    torvik_errors = validate_ratings_payload(torvik_payload)
                    self._assert_valid(f"torvik_{season}", torvik_errors)
                    torvik_path = self._write_json(f"torvik_{season}.json", torvik_payload)
                    manifest["artifacts"][str(season)]["torvik_json"] = torvik_path
                    manifest["providers"][str(season)]["torvik_json"] = torvik_provider
                    manifest["validation_errors"][str(season)]["torvik_json"] = torvik_errors
                    manifest["season_counts"][str(season)]["torvik_teams"] = len(torvik_payload["teams"])

            if self.config.include_tournament_context:
                tournament_payload, tournament_provider = self._collect_tournament_context(season)
                if tournament_payload.get("teams"):
                    seeds_path = self._write_json(f"tournament_seeds_{season}.json", tournament_payload)
                    manifest["artifacts"][str(season)]["tournament_seeds_json"] = seeds_path
                    manifest["providers"][str(season)]["tournament_seeds_json"] = tournament_provider
                    manifest["validation_errors"][str(season)]["tournament_seeds_json"] = []
                    manifest["season_counts"][str(season)]["tournament_seed_teams"] = len(tournament_payload["teams"])

        manifest_path = self._write_json(
            f"historical_manifest_{self.config.start_season}_{self.config.end_season}.json",
            manifest,
        )
        manifest["manifest_path"] = manifest_path
        return manifest

    def _collect_torvik(self, season: int) -> Tuple[Dict, str]:
        provider = self.providers.fetch_torvik_ratings(
            season,
            priority=self.config.torvik_provider_priority,
        )
        teams = [t for t in provider.records if isinstance(t, dict)]
        return {"teams": teams}, provider.provider

    def _collect_season_games(self, season: int) -> Tuple[Dict, str]:
        season_cache = self.cache_dir / f"cbbpy_historical_games_{season}.json"
        games: List[Dict] = []
        team_games: List[Dict] = []
        pbp_rows: List[Dict] = []
        failed_game_ids: List[str] = []
        processed_game_ids = set()

        if season_cache.exists():
            with open(season_cache, "r") as f:
                cached = json.load(f)
            if isinstance(cached, dict):
                games = [g for g in cached.get("games", []) if isinstance(g, dict)]
                team_games = [g for g in cached.get("team_games", []) if isinstance(g, dict)]
                pbp_rows = [r for r in cached.get("pbp", []) if isinstance(r, dict)]
                failed_game_ids = [str(gid) for gid in cached.get("failed_game_ids", [])]
                processed_game_ids = {str(g["game_id"]) for g in games if g.get("game_id")}
                cache_cap = cached.get("max_games_per_season")
                current_cap = self.config.max_games_per_season
                cap_match = cache_cap == current_cap
                if cached.get("complete") and games and cap_match:
                    return cached, "cbbpy_cache"
                if not cap_match:
                    games = []
                    team_games = []
                    pbp_rows = []
                    failed_game_ids = []
                    processed_game_ids = set()

        scraper = self.providers._import_module("cbbpy.mens_scraper")
        if scraper is None:
            raise RuntimeError("cbbpy is not available. Install cbbpy before running historical ingestion.")

        if (
            self.config.max_games_per_season is None
            and not self.config.include_pbp
            and not processed_game_ids
        ):
            fast_payload = self._collect_season_games_fast(season, scraper)
            if fast_payload is not None and fast_payload.get("games"):
                with open(season_cache, "w") as f:
                    json.dump(fast_payload, f, indent=2)
                return fast_payload, "cbbpy"

        game_ids_by_date: Dict[str, str] = {}
        for day in self._season_dates(season):
            day_str = day.isoformat()
            try:
                ids = scraper.get_game_ids(day_str)
            except Exception:
                continue
            for game_id in ids:
                game_id_str = str(game_id).strip()
                if game_id_str and game_id_str not in game_ids_by_date:
                    game_ids_by_date[game_id_str] = day_str

        game_ids = sorted(game_ids_by_date.keys())
        game_ids = [gid for gid in game_ids if gid not in processed_game_ids]
        if self.config.max_games_per_season is not None:
            remaining = self.config.max_games_per_season - len(processed_game_ids)
            if remaining <= 0 and games:
                cached_payload = {
                    "season": season,
                    "provider": "cbbpy",
                    "games": games,
                    "team_games": team_games,
                    "failed_game_ids": failed_game_ids,
                    "complete": True,
                    "max_games_per_season": self.config.max_games_per_season,
                }
                if self.config.include_pbp:
                    cached_payload["pbp"] = pbp_rows
                return cached_payload, "cbbpy_cache"
            game_ids = game_ids[: max(remaining, 0)]

        for idx, game_id in enumerate(game_ids, start=1):
            payload = self._fetch_single_cbbpy_game(
                scraper=scraper,
                season=season,
                game_id=game_id,
                game_date=game_ids_by_date[game_id],
            )
            if payload is None:
                failed_game_ids.append(game_id)
                continue
            games.append(payload["game"])
            team_games.extend(payload["team_games"])
            if self.config.include_pbp:
                pbp_rows.extend(payload["pbp"])
            if idx % 100 == 0:
                self._save_season_cache(
                    season_cache,
                    season,
                    games,
                    team_games,
                    failed_game_ids,
                    pbp_rows,
                    complete=False,
                )

        out = self._save_season_cache(
            season_cache,
            season,
            games,
            team_games,
            failed_game_ids,
            pbp_rows,
            complete=True,
        )
        if not games:
            raise ValueError(f"No games collected from cbbpy for season {season}")
        return out, "cbbpy"

    def _collect_season_games_fast(self, season: int, scraper) -> Optional[Dict]:
        try:
            games_tuple = scraper.get_games_season(season, info=False, box=True, pbp=False)
        except TypeError:
            try:
                games_tuple = scraper.get_games_season(season)
            except Exception:
                return None
        except Exception:
            return None

        team_games = self.providers._normalize_cbbpy_records(games_tuple)
        if not team_games:
            return None
        for row in team_games:
            row["season"] = season
            # Preserve actual game dates from cbbpy when available.
            # The 'date' field may be populated by cbbpy's bulk API;
            # only fall back to season-start if truly missing.
            existing_date = row.get("date") or row.get("game_date") or ""
            if not existing_date or existing_date == f"{season-1}-11-01":
                row["date"] = row.get("date") or f"{season-1}-11-01"

        games = self._team_games_to_games(team_games, season)
        return {
            "season": season,
            "provider": "cbbpy",
            "games": games,
            "team_games": team_games,
            "failed_game_ids": [],
            "complete": True,
            "max_games_per_season": None,
        }

    def _save_season_cache(
        self,
        season_cache: Path,
        season: int,
        games: List[Dict],
        team_games: List[Dict],
        failed_game_ids: List[str],
        pbp_rows: List[Dict],
        complete: bool,
    ) -> Dict:
        out = {
            "season": season,
            "provider": "cbbpy",
            "games": games,
            "team_games": team_games,
            "failed_game_ids": failed_game_ids,
            "complete": complete,
            "max_games_per_season": self.config.max_games_per_season,
        }
        if self.config.include_pbp:
            out["pbp"] = pbp_rows

        with open(season_cache, "w") as f:
            json.dump(out, f, indent=2)
        return out

    def _fetch_single_cbbpy_game(
        self,
        scraper,
        season: int,
        game_id: str,
        game_date: str,
    ) -> Optional[Dict]:
        last_exc: Optional[Exception] = None
        for _ in range(max(1, self.config.retry_attempts)):
            try:
                with self._timeout(self.config.per_game_timeout_seconds):
                    _, box_df, pbp_df = scraper.get_game(
                        game_id,
                        info=False,
                        box=True,
                        pbp=self.config.include_pbp,
                    )
                return self._normalize_game_frames(
                    season=season,
                    game_id=game_id,
                    game_date=game_date,
                    box_df=box_df,
                    pbp_df=pbp_df,
                )
            except Exception as exc:
                last_exc = exc
                continue
        if last_exc is not None:
            return None
        return None

    def _normalize_game_frames(self, season: int, game_id: str, game_date: str, box_df, pbp_df) -> Optional[Dict]:
        box_rows = self.providers._frame_to_records(box_df)
        if not box_rows:
            return None

        by_team: Dict[str, Dict] = {}
        for row in box_rows:
            team_name = str(row.get("team") or row.get("team_name") or "").strip()
            player_name = str(row.get("player") or "").strip().lower()
            if not team_name:
                continue
            if player_name in {"team", "totals", "team totals"}:
                continue
            team_id = self._normalize_team_name(team_name)
            stats = by_team.setdefault(
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

        teams = sorted(by_team.values(), key=lambda x: (-x["player_rows"], x["team_name"]))
        if len(teams) < 2:
            return None
        teams = teams[:2]
        first, second = teams

        game = {
            "game_id": game_id,
            "season": season,
            "date": game_date,
            "team1_id": first["team_id"],
            "team1_name": first["team_name"],
            "team2_id": second["team_id"],
            "team2_name": second["team_name"],
            "team1_score": int(round(first["team_score"])),
            "team2_score": int(round(second["team_score"])),
        }

        team_games = []
        for team, opp in ((first, second), (second, first)):
            possessions = team["fga"] - team["orb"] + team["turnovers"] + 0.475 * team["fta"]
            team_games.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "date": game_date,
                    "team_id": team["team_id"],
                    "team_name": team["team_name"],
                    "opponent_id": opp["team_id"],
                    "opponent_name": opp["team_name"],
                    "team_score": int(round(team["team_score"])),
                    "opponent_score": int(round(opp["team_score"])),
                    "possessions": max(float(possessions), 0.0),
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

        pbp = []
        if self.config.include_pbp:
            pbp_rows = self.providers._frame_to_records(pbp_df)
            for row in pbp_rows:
                pbp.append(
                    {
                        "game_id": game_id,
                        "season": season,
                        "date": game_date,
                        "half": row.get("half"),
                        "secs_left_half": row.get("secs_left_half"),
                        "secs_left_reg": row.get("secs_left_reg"),
                        "play_desc": row.get("play_desc"),
                        "play_team": row.get("play_team"),
                        "play_type": row.get("play_type"),
                        "shooting_play": row.get("shooting_play"),
                        "scoring_play": row.get("scoring_play"),
                        "is_three": row.get("is_three"),
                        "home_score": row.get("home_score"),
                        "away_score": row.get("away_score"),
                        "shot_x": row.get("shot_x"),
                        "shot_y": row.get("shot_y"),
                    }
                )

        return {"game": game, "team_games": team_games, "pbp": pbp}

    def _team_games_to_games(self, team_games: List[Dict], season: int) -> List[Dict]:
        by_game: Dict[str, List[Dict]] = {}
        for row in team_games:
            game_id = str(row.get("game_id") or "").strip()
            if not game_id:
                continue
            by_game.setdefault(game_id, []).append(row)

        games: List[Dict] = []
        for game_id, rows in by_game.items():
            if len(rows) < 2:
                continue
            rows = sorted(rows, key=lambda r: str(r.get("team_id", "")))
            t1, t2 = rows[0], rows[1]
            games.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "date": t1.get("date") or t2.get("date") or f"{season-1}-11-01",
                    "team1_id": t1.get("team_id"),
                    "team1_name": t1.get("team_name"),
                    "team2_id": t2.get("team_id"),
                    "team2_name": t2.get("team_name"),
                    "team1_score": int(round(self._to_float(t1.get("team_score")))),
                    "team2_score": int(round(self._to_float(t2.get("team_score")))),
                }
            )
        return games

    def _collect_team_metrics(
        self,
        season: int,
        game_records: Optional[List[Dict]] = None,
    ) -> Tuple[Dict, str]:
        provider_result = self.providers.fetch_team_box_metrics(
            season,
            priority=self.config.team_metrics_provider_priority,
        )
        provider = provider_result.provider
        rows = self._ensure_team_ids(provider_result.records)

        if not rows:
            rows = self._ensure_team_ids(
                self.sports_reference.fetch_team_season_stats(
                    season, game_records=game_records,
                )
            )
            provider = "sports_reference_scraper"

        # If rows came from a non-SR provider, still check for zero def_rtg
        # and patch from game records.
        if rows and game_records:
            zero_count = sum(1 for r in rows if (r.get("def_rtg") or 0) <= 0)
            if zero_count > len(rows) * 0.5:
                from ..scrapers.sports_reference import SportsReferenceScraper
                game_def_rtg = SportsReferenceScraper._compute_def_rtg_from_games(
                    game_records,
                )
                for row in rows:
                    if (row.get("def_rtg") or 0) <= 0:
                        tid = self._normalize_team_name(row.get("team_name") or "")
                        if tid in game_def_rtg:
                            row["def_rtg"] = game_def_rtg[tid]

        if not rows:
            raise ValueError(f"No team metrics available for season {season}")
        return {"season": season, "teams": rows}, provider

    def _collect_tournament_context(self, season: int) -> Tuple[Dict, str]:
        try:
            teams = self.tournament_seed_scraper.fetch_tournament_seeds(season)
            return {"season": season, "teams": teams}, "sports_reference_tournament_scraper"
        except Exception:
            return {"season": season, "teams": []}, "none"

    _NCAA_SUFFIX_RE = re.compile(r"NCAA$", re.IGNORECASE)

    def _ensure_team_ids(self, rows: List[Dict]) -> List[Dict]:
        out = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if not row.get("team_name") and row.get("name"):
                row["team_name"] = row["name"]
            if not row.get("name") and row.get("team_name"):
                row["name"] = row["team_name"]
            # Strip "NCAA" suffix from team names (Sports Reference appends it
            # for tournament qualifiers, producing IDs like "akronncaa").
            for key in ("team_name", "name"):
                val = row.get(key)
                if val and self._NCAA_SUFFIX_RE.search(val):
                    row[key] = self._NCAA_SUFFIX_RE.sub("", val).rstrip()
            if not row.get("team_id"):
                row["team_id"] = self._normalize_team_name(str(row.get("team_name") or row.get("name") or ""))
            else:
                # Also strip NCAA suffix from pre-existing team_id.
                tid = row["team_id"]
                if tid.endswith("ncaa"):
                    row["team_id"] = tid[:-4].rstrip("_")
            if row.get("team_id") and row.get("team_name"):
                out.append(row)
        return out

    def _season_dates(self, season: int) -> Iterable[date]:
        start = date(season - 1, 11, 1)
        end = date(season, 5, 1)
        current = start
        today = datetime.now(timezone.utc).date()
        stop = min(end, today)
        while current <= stop:
            yield current
            current += timedelta(days=1)

    def _write_json(self, filename: str, payload: Dict) -> str:
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return str(path)

    def _assert_valid(self, artifact_name: str, errors: List[str]) -> None:
        if errors and self.config.strict_validation:
            raise ValueError(f"{artifact_name} validation failed: {errors[:5]}")

    @staticmethod
    def _normalize_team_name(name: str) -> str:
        return "".join(ch.lower() if ch.isalnum() else "_" for ch in (name or "")).strip("_")

    @staticmethod
    def _to_float(value) -> float:
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    @contextmanager
    def _timeout(seconds: int):
        if seconds <= 0:
            yield
            return

        def _raise_timeout(_signum, _frame):
            raise TimeoutError(f"Timed out after {seconds} seconds")

        previous_handler = signal.signal(signal.SIGALRM, _raise_timeout)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)
