#!/usr/bin/env python3
"""Build tournament opening line probabilities from Odds API historical snapshots.

For each NCAA tournament game in ncaa_tournament_market.json, finds the
day-before pre-game snapshot to compute the "opening" line probability.
Movement = closing_prob - opening_prob (from ncaa_tournament_market.json).

Opening line definition: best snapshot from the calendar day before game day.
  - Prefer day-before 2300Z (18-hour pre-tip window)
  - Fall back to day-before 1700Z when 2300Z absent
  - Primary match by event_id (stable across snapshots), secondary by team names

Output: artifacts/ncaa_tournament_opening.json
Schema per record: {year, team1, team2, team1_open_prob, n_books, snapshot_used}
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.normalize import normalize_team_id
from src.data.scrapers.betting_markets import decimal_to_probability

CONSENSUS_PATH = REPO_ROOT / "artifacts" / "odds_api_closing_consensus.json"
CLOSING_PATH = REPO_ROOT / "artifacts" / "ncaa_tournament_market.json"
RAW_DIR = REPO_ROOT / "data/raw/odds_api/ncaab"
OUTPUT_PATH = REPO_ROOT / "artifacts" / "ncaa_tournament_opening.json"

SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]

# Same alias table as build_tournament_market_artifact.py — keep in sync.
_ALIAS: dict[str, str] = {
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
    "loyola_chi_ramblers": "loyola__il",
    "loyola_chicago_ramblers": "loyola__il",
    "providence_friars": "providence",
    "southern_california_trojans": "southern_california",
    "georgia_tech_yellow_jackets": "georgia_tech",
    "oregon_state_beavers": "oregon_state",
    "cal_state_fullerton_titans": "cal_state_fullerton",
    "saint_francis_red_flash": "saint_francis",
}


def _resolve(raw_name: str) -> str:
    norm = normalize_team_id(raw_name)
    return _ALIAS.get(norm, norm)


def _extract_h2h_prob(event: dict, home_name: str, away_name: str) -> tuple[float, float, int] | None:
    """Extract consensus no-vig home/away win probabilities from a raw event.

    Returns (home_prob, away_prob, n_books) or None if no usable h2h data.
    """
    book_probs: list[tuple[float, float]] = []
    for bookmaker in event.get("bookmakers", []) or []:
        for market in bookmaker.get("markets", []) or []:
            if market.get("key") != "h2h":
                continue
            outcomes = market.get("outcomes", []) or []
            prices: dict[str, float] = {}
            for outcome in outcomes:
                name = str(outcome.get("name") or "")
                price = outcome.get("price")
                if name and price not in (None, 0):
                    prices[name] = float(price)
            if home_name not in prices or away_name not in prices:
                continue
            hp = decimal_to_probability(prices[home_name])
            ap = decimal_to_probability(prices[away_name])
            total = hp + ap
            if total <= 0:
                continue
            book_probs.append((hp / total, ap / total))

    if not book_probs:
        return None
    n = len(book_probs)
    return (
        sum(h for h, _ in book_probs) / n,
        sum(a for _, a in book_probs) / n,
        n,
    )


def _build_consensus_lookup() -> dict[tuple[int, str, str], dict]:
    """Build {(year, t1_id, t2_id): consensus_record} from closing consensus.

    Key uses frozenset so lookup works in either direction — caller handles
    orientation. Actually returns keyed by (season, home_id, away_id) for
    direct match, but also indexes by (season, frozenset) for pair lookup.
    """
    with open(CONSENSUS_PATH) as f:
        data = json.load(f)

    by_pair: dict[tuple[int, frozenset], dict] = {}
    for g in data["games"]:
        if not g.get("tournament_window"):
            continue
        season = int(g["season"])
        pair = frozenset({g["home_team_id"], g["away_team_id"]})
        key = (season, pair)
        # Keep record with most books if duplicated.
        if key not in by_pair or g["n_books_h2h"] > by_pair[key]["n_books_h2h"]:
            by_pair[key] = g
    return by_pair


def _load_raw_snapshot(path: Path) -> list[dict]:
    """Load events list from a raw snapshot file."""
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return payload.get("data", [])


def _get_opening_prob(
    team1: str,
    team2: str,
    event_id: str,
    game_date_str: str,  # YYYY-MM-DD of game day
    commence_dt: datetime,
) -> tuple[float, float, int, str] | None:
    """Find opening probability from day-before snapshot.

    Returns (home_prob, away_prob, n_books, snapshot_filename) or None.
    home_prob is P(event home team wins) — caller handles orientation.
    """
    game_date = datetime.strptime(game_date_str, "%Y-%m-%d").date()
    prev_date = (game_date - timedelta(days=1)).isoformat()  # YYYY-MM-DD

    # Prefer 2300Z (later line, more books), fall back to 1700Z.
    candidates = [
        RAW_DIR / f"{prev_date}_2300Z.json",
        RAW_DIR / f"{prev_date}_1700Z.json",
    ]

    for snapshot_path in candidates:
        events = _load_raw_snapshot(snapshot_path)
        if not events:
            continue

        snapshot_ts = datetime.strptime(snapshot_path.stem, "%Y-%m-%d_%H%MZ").replace(tzinfo=timezone.utc)

        # Skip snapshot taken after game started (shouldn't happen for day-before, but guard).
        if snapshot_ts >= commence_dt:
            continue

        for event in events:
            # Primary: match by event_id (stable UUID across snapshots).
            if str(event.get("id") or "") != event_id:
                # Secondary fallback: match by normalized team names.
                h_id = _resolve(str(event.get("home_team") or ""))
                a_id = _resolve(str(event.get("away_team") or ""))
                if frozenset({h_id, a_id}) != frozenset({team1, team2}):
                    continue

            result = _extract_h2h_prob(
                event,
                str(event.get("home_team") or ""),
                str(event.get("away_team") or ""),
            )
            if result is None:
                continue
            home_prob, away_prob, n_books = result
            return home_prob, away_prob, n_books, snapshot_path.name

    return None


def build_opening_market(year: int, consensus_lookup: dict) -> list[dict]:
    """Build per-game opening line records for one season."""
    # Load closing market to get the canonical team pairs and orientation.
    with open(CLOSING_PATH) as f:
        closing_data = json.load(f)

    year_closing = [g for g in closing_data["games"] if int(g["year"]) == year]
    if not year_closing:
        print(f"  {year}: no closing market data, skipping")
        return []

    records: list[dict] = []
    n_found = 0
    n_total = len(year_closing)

    for closing_game in year_closing:
        t1 = closing_game["team1"]
        t2 = closing_game["team2"]
        pair = frozenset({t1, t2})
        key = (year, pair)

        consensus = consensus_lookup.get(key)
        if consensus is None:
            continue  # No consensus record — can't find event_id.

        event_id = str(consensus.get("event_id") or "")
        commence_raw = consensus.get("commence_time", "")
        try:
            commence_dt = datetime.fromisoformat(str(commence_raw).replace("Z", "+00:00"))
        except ValueError:
            continue

        game_date_str = commence_dt.date().isoformat()
        home_id = consensus["home_team_id"]

        result = _get_opening_prob(t1, t2, event_id, game_date_str, commence_dt)
        if result is None:
            continue

        home_open_prob, away_open_prob, n_books, snapshot_used = result

        # Orient probability to team1.
        t1_open_prob = home_open_prob if home_id == t1 else away_open_prob

        records.append(
            {
                "year": year,
                "team1": t1,
                "team2": t2,
                "team1_open_prob": round(t1_open_prob, 6),
                "n_books": n_books,
                "snapshot_used": snapshot_used,
            }
        )
        n_found += 1

    coverage_pct = n_found / max(n_total, 1) * 100
    print(f"  {year}: {n_found}/{n_total} games with opening line ({coverage_pct:.1f}%)")
    return records


def main() -> None:
    consensus_lookup = _build_consensus_lookup()

    all_records: list[dict] = []
    for year in SEASONS:
        records = build_opening_market(year, consensus_lookup)
        all_records.extend(records)

    all_records.sort(key=lambda r: (r["year"], r["team1"], r["team2"]))

    artifact = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "The Odds API day-before snapshots (opening line proxy)",
        "seasons": SEASONS,
        "n_games": len(all_records),
        "games": all_records,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True))
    print(f"\nWrote {len(all_records)} opening line records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
