#!/usr/bin/env python3
"""Audit the historical Odds API NCAAB archive.

Produces a compact JSON summary for the current backfill:

- snapshot completeness by season
- overall and tournament-window game counts
- bookmaker / market depth
- tournament game coverage against tournament_results_{year}.json

This is an archive-audit tool, not a normalization or modeling step.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import median
import sys
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.normalize import normalize_team_id

RAW_DIR = REPO_ROOT / "data/raw/odds_api/ncaab"
HIST_DIR = REPO_ROOT / "data/raw/historical"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/odds_api_archive_audit.json"

# Season windows mirror the scraper's configured backfill target.
SEASONS = [
    (2021, date(2020, 11, 1), date(2021, 4, 15)),
    (2022, date(2021, 11, 1), date(2022, 4, 10)),
    (2023, date(2022, 11, 1), date(2023, 4, 10)),
    (2024, date(2023, 11, 1), date(2024, 4, 10)),
    (2025, date(2024, 11, 1), date(2025, 4, 10)),
]

TOURNEY_START = (3, 15)
# Extended to (4, 9) because late-evening championship games (e.g. 2024 tip
# 9:20 PM ET) have UTC commence_time on April 9.
TOURNEY_END = (4, 9)

_ODDS_NAME_ALIASES = {
    "texas_a_m_cc": "texas_a_m_corpus_christi",
    "texas_a_m_c_c": "texas_a_m_corpus_christi",
    "texas_a_m_cc_islanders": "texas_a_m_corpus_christi",
    "texas_a_m_corpus_christi_islanders": "texas_a_m_corpus_christi",
    "mount_st_mary_s_mountaineers": "mount_st__mary_s",
    "mount_st_mary_s": "mount_st__mary_s",
    "saint_mary_s_gaels": "saint_mary_s__ca",
    "saint_mary_s": "saint_mary_s__ca",
    "st_john_s_red_storm": "st__john_s__ny",
    "st_john_s": "st__john_s__ny",
    "byu_cougars": "brigham_young",
    "vcu_rams": "virginia_commonwealth",
    "unc_tar_heels": "north_carolina",
    "uconn_huskies": "connecticut",
    "ole_miss_rebels": "mississippi",
    "miami_hurricanes": "miami__fl",
    # Odds API uses "Loyola (Chi) Ramblers", not "Loyola Chicago Ramblers".
    "loyola_chi_ramblers": "loyola__il",
    "loyola_chicago_ramblers": "loyola__il",
    "providence_friars": "providence",
    "southern_california_trojans": "southern_california",
    "georgia_tech_yellow_jackets": "georgia_tech",
    "oregon_state_beavers": "oregon_state",
    "cal_state_fullerton_titans": "cal_state_fullerton",
    "saint_francis_red_flash": "saint_francis",
}


@dataclass(frozen=True)
class SnapshotSummary:
    season: int
    timestamp: datetime
    path: Path


def _is_tournament_window(d: date) -> bool:
    start = date(d.year, *TOURNEY_START)
    end = date(d.year, *TOURNEY_END)
    return start <= d <= end


def _parse_snapshot_filename(path: Path) -> datetime:
    return datetime.strptime(path.stem, "%Y-%m-%d_%H%MZ").replace(tzinfo=timezone.utc)


def _planned_snapshot_timestamps() -> dict[int, list[datetime]]:
    per_season: dict[int, list[datetime]] = {}
    for season, start_date, end_date in SEASONS:
        current = start_date
        timestamps: list[datetime] = []
        while current <= end_date:
            timestamps.append(datetime(current.year, current.month, current.day, 17, 0, 0, tzinfo=timezone.utc))
            if _is_tournament_window(current):
                timestamps.append(datetime(current.year, current.month, current.day, 23, 0, 0, tzinfo=timezone.utc))
            current += timedelta(days=1)
        per_season[season] = timestamps
    return per_season


def _season_for_timestamp(ts: datetime) -> int | None:
    for season, start_date, end_date in SEASONS:
        if start_date <= ts.date() <= end_date:
            return season
    return None


def _load_archive_paths() -> list[SnapshotSummary]:
    snapshots: list[SnapshotSummary] = []
    for path in sorted(RAW_DIR.glob("*.json")):
        ts = _parse_snapshot_filename(path)
        season = _season_for_timestamp(ts)
        if season is None:
            continue
        snapshots.append(SnapshotSummary(season=season, timestamp=ts, path=path))
    return snapshots


def _load_tournament_pairs(season: int) -> set[tuple[str, str]]:
    path = HIST_DIR / f"tournament_results_{season}.json"
    payload = json.loads(path.read_text())
    pairs = set()
    for game in payload["games"]:
        team1 = normalize_team_id(str(game["team1_id"]))
        team2 = normalize_team_id(str(game["team2_id"]))
        pairs.add(tuple(sorted((team1, team2))))
    return pairs


def _load_tournament_team_ids(season: int) -> set[str]:
    path = HIST_DIR / f"tournament_results_{season}.json"
    payload = json.loads(path.read_text())
    teams = set()
    for game in payload["games"]:
        teams.add(normalize_team_id(str(game["team1_id"])))
        teams.add(normalize_team_id(str(game["team2_id"])))
    return teams


def _resolve_to_tournament_field(raw_name: str, tournament_team_ids: set[str]) -> str | None:
    norm = normalize_team_id(raw_name)
    if norm in tournament_team_ids:
        return norm
    if norm in _ODDS_NAME_ALIASES:
        alias = _ODDS_NAME_ALIASES[norm]
        if alias in tournament_team_ids:
            return alias

    parts = norm.split("_")
    # Odds API often appends mascots; progressively drop suffix tokens.
    for stop in range(len(parts) - 1, 0, -1):
        candidate = normalize_team_id("_".join(parts[:stop]))
        if candidate in tournament_team_ids:
            return candidate
        alias = _ODDS_NAME_ALIASES.get(candidate)
        if alias in tournament_team_ids:
            return alias
    return None


def _safe_median(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(median(values))


def audit_archive() -> dict[str, object]:
    planned = _planned_snapshot_timestamps()
    snapshots = _load_archive_paths()

    snapshot_counts = Counter(item.season for item in snapshots)
    by_season_paths: dict[int, list[SnapshotSummary]] = defaultdict(list)
    for item in snapshots:
        by_season_paths[item.season].append(item)

    archive_summary = {
        "snapshot_count": len(snapshots),
        "first_snapshot": snapshots[0].path.name if snapshots else None,
        "last_snapshot": snapshots[-1].path.name if snapshots else None,
    }

    season_summary: dict[str, object] = {}
    tournament_summary: dict[str, object] = {}

    for season, planned_ts in planned.items():
        actual_paths = by_season_paths.get(season, [])
        expected_names = {ts.strftime("%Y-%m-%d_%H%MZ") + ".json" for ts in planned_ts}
        actual_names = {item.path.name for item in actual_paths}
        missing_names = sorted(expected_names - actual_names)
        tournament_team_ids = _load_tournament_team_ids(season)

        unique_event_ids: set[str] = set()
        unique_tourney_event_ids: set[str] = set()
        books_per_event: list[int] = []
        books_per_tourney_event: list[int] = []
        event_market_presence = Counter()
        tourney_market_presence = Counter()
        tourney_pairs_seen: set[tuple[str, str]] = set()

        for item in actual_paths:
            payload = json.loads(item.path.read_text())
            events = payload.get("data", [])
            for event in events:
                event_id = str(event.get("id") or "")
                if not event_id:
                    continue
                unique_event_ids.add(event_id)

                bookmakers = event.get("bookmakers", []) or []
                books_per_event.append(len(bookmakers))
                market_keys = {
                    market.get("key")
                    for book in bookmakers
                    for market in (book.get("markets", []) or [])
                    if market.get("key")
                }
                for key in ("h2h", "spreads", "totals"):
                    if key in market_keys:
                        event_market_presence[key] += 1

                commence_raw = event.get("commence_time")
                try:
                    commence_dt = datetime.fromisoformat(str(commence_raw).replace("Z", "+00:00"))
                except ValueError:
                    continue

                if not _is_tournament_window(commence_dt.date()):
                    continue

                home = _resolve_to_tournament_field(str(event.get("home_team") or ""), tournament_team_ids)
                away = _resolve_to_tournament_field(str(event.get("away_team") or ""), tournament_team_ids)
                if home and away:
                    unique_tourney_event_ids.add(event_id)
                    books_per_tourney_event.append(len(bookmakers))
                    for key in ("h2h", "spreads", "totals"):
                        if key in market_keys:
                            tourney_market_presence[key] += 1
                    tourney_pairs_seen.add(tuple(sorted((home, away))))

        expected_pairs = _load_tournament_pairs(season)
        matched_pairs = expected_pairs & tourney_pairs_seen
        unmatched_expected = sorted(expected_pairs - tourney_pairs_seen)
        extra_pairs = sorted(tourney_pairs_seen - expected_pairs)

        season_summary[str(season)] = {
            "expected_snapshots": len(planned_ts),
            "actual_snapshots": len(actual_paths),
            "missing_snapshots": len(missing_names),
            "completion_pct": round(len(actual_paths) / max(len(planned_ts), 1) * 100.0, 2),
            "first_missing_snapshot": missing_names[0] if missing_names else None,
            "unique_event_ids": len(unique_event_ids),
            "unique_tournament_event_ids": len(unique_tourney_event_ids),
            "median_books_per_event": _safe_median(books_per_event),
            "median_books_per_tournament_event": _safe_median(books_per_tourney_event),
            "event_market_presence": dict(event_market_presence),
            "tournament_market_presence": dict(tourney_market_presence),
        }

        tournament_summary[str(season)] = {
            "expected_games": len(expected_pairs),
            "matched_games": len(matched_pairs),
            "coverage_pct": round(len(matched_pairs) / max(len(expected_pairs), 1) * 100.0, 2),
            "unmatched_expected_games": unmatched_expected[:20],
            "unmatched_expected_count": len(unmatched_expected),
            "extra_observed_pairs": extra_pairs[:20],
            "extra_observed_count": len(extra_pairs),
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "archive": archive_summary,
        "season_summary": season_summary,
        "tournament_summary": tournament_summary,
    }


def main() -> None:
    report = audit_archive()
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True))

    print(f"Archive snapshots: {report['archive']['snapshot_count']}")
    print(f"First snapshot: {report['archive']['first_snapshot']}")
    print(f"Last snapshot: {report['archive']['last_snapshot']}")
    for season, summary in report["season_summary"].items():
        print(
            f"Season {season}: {summary['actual_snapshots']}/{summary['expected_snapshots']} snapshots "
            f"({summary['completion_pct']}%), unique tournament events={summary['unique_tournament_event_ids']}"
        )
    for season, summary in report["tournament_summary"].items():
        print(
            f"Tournament {season}: {summary['matched_games']}/{summary['expected_games']} games "
            f"({summary['coverage_pct']}%)"
        )
    print(f"Wrote audit: {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
