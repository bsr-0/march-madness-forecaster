"""Resumable season-by-season historical backfill runner."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ingestion.historical_pipeline import HistoricalDataPipeline, HistoricalIngestionConfig


def _load_json(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def _season_files(base_dir: Path, season: int) -> Tuple[Path, Path, Path]:
    games = base_dir / f"historical_games_{season}.json"
    teams = base_dir / f"team_metrics_{season}.json"
    seeds = base_dir / f"tournament_seeds_{season}.json"
    return games, teams, seeds


def _season_complete(base_dir: Path, season: int, require_tournament: bool) -> Tuple[bool, str]:
    games_path, teams_path, seeds_path = _season_files(base_dir, season)
    required = [games_path, teams_path] + ([seeds_path] if require_tournament else [])
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        return False, f"missing files: {', '.join(missing)}"

    try:
        games_payload = _load_json(games_path)
        teams_payload = _load_json(teams_path)
        seeds_payload = _load_json(seeds_path) if require_tournament else {"teams": []}
    except Exception as exc:
        return False, f"json parse failure: {exc}"

    games = games_payload.get("games", [])
    team_games = games_payload.get("team_games", [])
    teams = teams_payload.get("teams", [])
    seeds = seeds_payload.get("teams", [])
    if not isinstance(games, list) or not games:
        return False, "games payload empty"
    if not isinstance(team_games, list) or not team_games:
        return False, "team_games payload empty"
    if not isinstance(teams, list) or not teams:
        return False, "team metrics payload empty"
    if require_tournament and (not isinstance(seeds, list) or len(seeds) < 64):
        return False, "tournament seeds payload undersized"
    return True, "ok"


def _run_single_season(args, season: int) -> Dict:
    cfg = HistoricalIngestionConfig(
        start_season=season,
        end_season=season,
        output_dir=str(args.output_dir),
        cache_dir=str(args.cache_dir),
        include_pbp=args.include_pbp,
        include_tournament_context=not args.skip_tournament_context,
        strict_validation=not args.allow_invalid_payloads,
        retry_attempts=args.retry_attempts,
        per_game_timeout_seconds=args.per_game_timeout_seconds,
        max_games_per_season=args.max_games_per_season,
        team_metrics_provider_priority=args.team_metrics_provider_priority,
    )
    return HistoricalDataPipeline(cfg).run()


def _write_rollup_manifest(
    output_dir: Path,
    start_season: int,
    end_season: int,
    seasons_completed: List[int],
    seasons_failed: List[Dict],
    config: Dict,
) -> Path:
    rollup = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "start_season": start_season,
        "end_season": end_season,
        "seasons_completed": seasons_completed,
        "seasons_failed": seasons_failed,
        "config": config,
    }
    out = output_dir / f"historical_backfill_manifest_{start_season}_{end_season}.json"
    with open(out, "w") as f:
        json.dump(rollup, f, indent=2)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resumable historical season backfill runner.")
    parser.add_argument("--start-season", type=int, required=True)
    parser.add_argument("--end-season", type=int, required=True)
    parser.add_argument("--output-dir", default="data/raw/historical")
    parser.add_argument("--cache-dir", default="data/raw/cache")
    parser.add_argument("--retry-attempts", type=int, default=2)
    parser.add_argument("--per-game-timeout-seconds", type=int, default=25)
    parser.add_argument("--max-games-per-season", type=int, default=None)
    parser.add_argument("--include-pbp", action="store_true")
    parser.add_argument("--skip-tournament-context", action="store_true")
    parser.add_argument("--allow-invalid-payloads", action="store_true")
    parser.add_argument(
        "--team-metrics-provider-priority",
        default=None,
        help="Comma-separated provider order: sportsdataverse,sportsipy,cbbdata",
    )
    parser.add_argument("--force", action="store_true", help="Re-run seasons even when outputs are complete.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue remaining seasons after a failure.")
    parser.add_argument("--season-retries", type=int, default=1, help="Whole-season retry attempts on failure.")
    parser.add_argument("--sleep-seconds", type=float, default=2.0, help="Sleep between retries.")
    return parser.parse_args()


def _parse_priority(raw: str | None):
    if raw is None:
        return None
    values = [p.strip() for p in raw.split(",") if p.strip()]
    return values or None


def main() -> int:
    args = parse_args()
    if args.start_season > args.end_season:
        print("start-season must be <= end-season", file=sys.stderr)
        return 2

    args.output_dir = Path(args.output_dir)
    args.cache_dir = Path(args.cache_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.team_metrics_provider_priority = _parse_priority(args.team_metrics_provider_priority)

    completed: List[int] = []
    failed: List[Dict] = []
    require_tournament = not args.skip_tournament_context

    for season in range(args.start_season, args.end_season + 1):
        complete, reason = _season_complete(args.output_dir, season, require_tournament=require_tournament)
        if complete and not args.force:
            print(f"[{season}] skip (already complete)")
            completed.append(season)
            continue

        print(f"[{season}] run (reason: {reason})")
        last_error = None
        for attempt in range(1, max(args.season_retries, 1) + 1):
            try:
                manifest = _run_single_season(args, season)
                counts = manifest.get("season_counts", {}).get(str(season), {})
                print(
                    f"[{season}] success games={counts.get('games', 0)} "
                    f"team_games={counts.get('team_games', 0)} teams={counts.get('teams', 0)}"
                )
                completed.append(season)
                last_error = None
                break
            except Exception as exc:
                last_error = str(exc)
                print(f"[{season}] attempt {attempt}/{args.season_retries} failed: {last_error}", file=sys.stderr)
                if attempt < args.season_retries:
                    time.sleep(max(args.sleep_seconds, 0.0))

        if last_error is not None:
            failed.append({"season": season, "error": last_error})
            if not args.continue_on_error:
                break

    rollup = _write_rollup_manifest(
        output_dir=args.output_dir,
        start_season=args.start_season,
        end_season=args.end_season,
        seasons_completed=completed,
        seasons_failed=failed,
        config=asdict(
            HistoricalIngestionConfig(
                start_season=args.start_season,
                end_season=args.end_season,
                output_dir=str(args.output_dir),
                cache_dir=str(args.cache_dir),
                include_pbp=args.include_pbp,
                strict_validation=not args.allow_invalid_payloads,
                retry_attempts=args.retry_attempts,
                per_game_timeout_seconds=args.per_game_timeout_seconds,
                max_games_per_season=args.max_games_per_season,
                include_tournament_context=not args.skip_tournament_context,
                team_metrics_provider_priority=args.team_metrics_provider_priority,
            )
        ),
    )
    print(f"Backfill rollup manifest: {rollup}")

    if failed:
        print(f"Seasons failed: {', '.join(str(item['season']) for item in failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
