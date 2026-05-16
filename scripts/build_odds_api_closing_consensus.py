#!/usr/bin/env python3
"""Build normalized closing consensus probabilities from Odds API snapshots.

Uses the latest pregame snapshot per event and computes no-vig implied
probabilities from each bookmaker's moneyline pair, then averages those
book-level probabilities into a consensus closing estimate.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.normalize import normalize_team_id
from src.data.scrapers.betting_markets import decimal_to_probability
from src.data.team_name_resolver import TeamNameResolver

RAW_DIR = REPO_ROOT / "data/raw/odds_api/ncaab"
OUTPUT_PATH = REPO_ROOT / "artifacts/odds_api_closing_consensus.json"

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

ODDS_API_EXTRA_ALIASES = {
    "texas_a_m_corpus_christi": [
        "Texas A&M-CC",
        "Texas A&M-CC Islanders",
        "Texas A&M Corpus Christi",
        "Texas A&M Corpus Christi Islanders",
    ],
    "miami__fl": ["Miami Hurricanes"],
    # Odds API uses "Loyola (Chi) Ramblers", not "Loyola Chicago Ramblers".
    "loyola__il": ["Loyola Chicago Ramblers", "Loyola (Chi) Ramblers"],
    "mount_st__mary_s": ["Mount St. Mary's Mountaineers"],
    "saint_mary_s__ca": ["Saint Mary's Gaels"],
    "st__john_s__ny": ["St. John's Red Storm"],
    "saint_francis": ["Saint Francis Red Flash"],
}


def _parse_snapshot_filename(path: Path) -> datetime:
    return datetime.strptime(path.stem, "%Y-%m-%d_%H%MZ").replace(tzinfo=timezone.utc)


def _season_for_date(d: date) -> int | None:
    for season, start_date, end_date in SEASONS:
        if start_date <= d <= end_date:
            return season
    return None


def _is_tournament_window(d: date) -> bool:
    start = date(d.year, *TOURNEY_START)
    end = date(d.year, *TOURNEY_END)
    return start <= d <= end


def _resolve_team_id(raw_name: str, resolver: TeamNameResolver) -> tuple[str, float, str]:
    result = resolver.resolve(raw_name)
    if result.method != "unresolved" and result.canonical_id:
        return result.canonical_id, float(result.confidence), result.method

    parts = str(raw_name).replace(".", "").split()
    for stop in range(len(parts) - 1, 0, -1):
        candidate_name = " ".join(parts[:stop])
        candidate = resolver.resolve(candidate_name)
        if candidate.method != "unresolved" and candidate.canonical_id:
            return candidate.canonical_id, float(candidate.confidence), f"suffix_strip:{candidate.method}"

    canonical_id = normalize_team_id(raw_name)
    return canonical_id, float(result.confidence), result.method


def _extract_h2h_probabilities(event: dict, bookmaker: dict) -> tuple[float, float] | None:
    home_name = str(event.get("home_team") or "")
    away_name = str(event.get("away_team") or "")
    for market in bookmaker.get("markets", []) or []:
        if market.get("key") != "h2h":
            continue
        outcomes = market.get("outcomes", []) or []
        if len(outcomes) < 2:
            continue
        prices = {}
        for outcome in outcomes:
            name = str(outcome.get("name") or "")
            price = outcome.get("price")
            if not name or price in (None, 0):
                continue
            prices[name] = float(price)
        if home_name not in prices or away_name not in prices:
            continue
        home_prob = decimal_to_probability(prices[home_name])
        away_prob = decimal_to_probability(prices[away_name])
        total = home_prob + away_prob
        if total <= 0:
            return None
        return home_prob / total, away_prob / total
    return None


def build_closing_consensus() -> dict[str, object]:
    resolver = TeamNameResolver(extra_aliases=ODDS_API_EXTRA_ALIASES)
    closing_by_event: dict[str, dict] = {}

    for path in sorted(RAW_DIR.glob("*.json")):
        snapshot_ts = _parse_snapshot_filename(path)
        payload = json.loads(path.read_text())
        for event in payload.get("data", []) or []:
            event_id = str(event.get("id") or "")
            if not event_id:
                continue
            commence_raw = event.get("commence_time")
            try:
                commence_dt = datetime.fromisoformat(str(commence_raw).replace("Z", "+00:00"))
            except ValueError:
                continue
            if snapshot_ts > commence_dt:
                continue

            season = _season_for_date(commence_dt.date())
            if season is None:
                continue

            home_name = str(event.get("home_team") or "")
            away_name = str(event.get("away_team") or "")
            if not home_name or not away_name:
                continue

            home_id, home_conf, home_method = _resolve_team_id(home_name, resolver)
            away_id, away_conf, away_method = _resolve_team_id(away_name, resolver)

            book_probs = []
            book_keys = []
            spread_books = 0
            total_books = 0
            for bookmaker in event.get("bookmakers", []) or []:
                probs = _extract_h2h_probabilities(event, bookmaker)
                if probs is not None:
                    book_probs.append(probs)
                    book_keys.append(str(bookmaker.get("key") or ""))
                market_keys = {m.get("key") for m in bookmaker.get("markets", []) or []}
                if "spreads" in market_keys:
                    spread_books += 1
                if "totals" in market_keys:
                    total_books += 1

            if not book_probs:
                continue

            home_consensus = sum(home for home, _ in book_probs) / len(book_probs)
            away_consensus = sum(away for _, away in book_probs) / len(book_probs)

            prior = closing_by_event.get(event_id)
            if prior is not None and prior["closing_snapshot"] >= snapshot_ts.isoformat():
                continue

            closing_by_event[event_id] = {
                "season": season,
                "event_id": event_id,
                "commence_time": commence_dt.isoformat(),
                "closing_snapshot": snapshot_ts.isoformat(),
                "home_team_name": home_name,
                "away_team_name": away_name,
                "home_team_id": home_id,
                "away_team_id": away_id,
                "home_resolution_confidence": home_conf,
                "away_resolution_confidence": away_conf,
                "home_resolution_method": home_method,
                "away_resolution_method": away_method,
                "home_win_prob": round(home_consensus, 6),
                "away_win_prob": round(away_consensus, 6),
                "n_books_h2h": len(book_probs),
                "n_books_spreads": spread_books,
                "n_books_totals": total_books,
                "bookmakers": sorted(set(book_keys)),
                "tournament_window": _is_tournament_window(commence_dt.date()),
            }

    games = sorted(
        closing_by_event.values(),
        key=lambda row: (row["season"], row["commence_time"], row["event_id"]),
    )
    season_counts = Counter(row["season"] for row in games)
    tournament_counts = Counter(row["season"] for row in games if row["tournament_window"])

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "The Odds API historical snapshots",
        "snapshot_dir": str(RAW_DIR),
        "n_games": len(games),
        "season_counts": dict(sorted(season_counts.items())),
        "tournament_window_counts": dict(sorted(tournament_counts.items())),
        "games": games,
    }


def main() -> None:
    artifact = build_closing_consensus()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True))
    print(f"Games written: {artifact['n_games']}")
    print(f"Season counts: {artifact['season_counts']}")
    print(f"Tournament-window counts: {artifact['tournament_window_counts']}")
    print(f"Wrote artifact: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
