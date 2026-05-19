#!/usr/bin/env python3
"""Build tournament-only closing market probabilities for Kaggle model use.

Derives a per-game market probability for every NCAA tournament game that has
closing odds in the Odds API archive, covering 2021-2025.

Primary source: artifacts/odds_api_closing_consensus.json (pre-computed).
Gap-fill: raw snapshot files for games missed by the consensus due to:
  - Loyola (Chi) Ramblers alias bug (fixed in consensus as of 2026-05-13)
  - Championship games with UTC commence_time on April 9 (outside prior window)

Output: artifacts/ncaa_tournament_market.json
Schema per game record:
  {year, team1, team2, team1_win_prob, n_books, source}
  where team1 matches the tournament_results orientation (team1 = winner).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.normalize import normalize_team_id
from src.data.scrapers.betting_markets import decimal_to_probability

CONSENSUS_PATH = REPO_ROOT / "artifacts" / "odds_api_closing_consensus.json"
RAW_DIR = REPO_ROOT / "data/raw/odds_api/ncaab"
HIST_DIR = REPO_ROOT / "data/raw/historical"
OUTPUT_PATH = REPO_ROOT / "artifacts" / "ncaa_tournament_market.json"

SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]

# Extended window matches the fixed scripts: championship games tip late ET.
TOURNEY_START_MMDD = "03-15"
TOURNEY_END_MMDD = "04-09"

# Same alias table as audit script (post-fix).
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


def _is_tourney_window(d: date) -> bool:
    year = d.year
    start = date(year, 3, 15)
    end = date(year, 4, 9)
    return start <= d <= end


def _load_expected_pairs(year: int) -> dict[frozenset[str], dict]:
    """Return {frozenset({t1, t2}): game_dict} for all main-bracket games."""
    path = HIST_DIR / f"tournament_results_{year}.json"
    payload = json.loads(path.read_text())
    games = payload.get("games", payload) if isinstance(payload, dict) else payload
    result: dict[frozenset[str], dict] = {}
    for g in games:
        # Skip First Four games (they're not in the 63-game main bracket).
        if g.get("round_name", "") in ("FF", "First Four"):
            continue
        t1 = normalize_team_id(str(g["team1_id"]))
        t2 = normalize_team_id(str(g["team2_id"]))
        key = frozenset({t1, t2})
        result[key] = g
    return result


def _build_consensus_index(year: int) -> dict[frozenset[str], dict]:
    """Index consensus games by canonical team pair for one season."""
    with open(CONSENSUS_PATH) as f:
        data = json.load(f)
    index: dict[frozenset[str], dict] = {}
    for game in data["games"]:
        if game["season"] != year:
            continue
        if not game.get("tournament_window"):
            continue
        t1 = game["home_team_id"]
        t2 = game["away_team_id"]
        key = frozenset({t1, t2})
        # Keep the record with more books if duplicated.
        if key not in index or game["n_books_h2h"] > index[key]["n_books_h2h"]:
            index[key] = game
    return index


def _gap_fill_from_raw(year: int, missing_pairs: set[frozenset[str]]) -> dict[frozenset[str], dict]:
    """Scan raw snapshots for games absent from the consensus index.

    This handles:
    - Loyola (Chi) Ramblers alias (fixed in consensus scripts but consensus
      artifact may not have been rebuilt yet)
    - Championship games with UTC date April 9 (outside old April 8 window)
    """
    found: dict[frozenset[str], dict] = {}
    if not missing_pairs:
        return found

    month_prefixes = [f"{year}-03-", f"{year}-04-"]
    best_by_pair: dict[frozenset[str], dict] = {}

    for prefix in month_prefixes:
        for path in sorted(RAW_DIR.glob(f"{prefix}*.json")):
            snapshot_ts_str = path.stem  # e.g. "2024-04-08_1700Z"
            snapshot_ts = datetime.strptime(snapshot_ts_str, "%Y-%m-%d_%H%MZ").replace(tzinfo=timezone.utc)
            payload = json.loads(path.read_text())
            for event in payload.get("data", []):
                event_id = str(event.get("id") or "")
                commence_raw = event.get("commence_time")
                try:
                    commence_dt = datetime.fromisoformat(str(commence_raw).replace("Z", "+00:00"))
                except ValueError:
                    continue
                # Skip post-tip snapshots.
                if snapshot_ts > commence_dt:
                    continue
                if not _is_tourney_window(commence_dt.date()):
                    continue

                home_id = _resolve(str(event.get("home_team") or ""))
                away_id = _resolve(str(event.get("away_team") or ""))
                key = frozenset({home_id, away_id})
                if key not in missing_pairs:
                    continue

                # Compute consensus probability from available books.
                book_probs: list[tuple[float, float]] = []
                book_keys: list[str] = []
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
                        home_name = str(event.get("home_team") or "")
                        away_name = str(event.get("away_team") or "")
                        if home_name not in prices or away_name not in prices:
                            continue
                        hp = decimal_to_probability(prices[home_name])
                        ap = decimal_to_probability(prices[away_name])
                        total = hp + ap
                        if total <= 0:
                            continue
                        book_probs.append((hp / total, ap / total))
                        book_keys.append(str(bookmaker.get("key") or ""))

                if not book_probs:
                    continue

                n_books = len(book_probs)
                prior = best_by_pair.get(key)
                if prior is not None and prior["n_books_h2h"] >= n_books:
                    continue  # Keep the record with more books.

                home_win_prob = sum(h for h, _ in book_probs) / n_books
                away_win_prob = sum(a for _, a in book_probs) / n_books
                best_by_pair[key] = {
                    "home_team_id": home_id,
                    "away_team_id": away_id,
                    "home_win_prob": round(home_win_prob, 6),
                    "away_win_prob": round(away_win_prob, 6),
                    "n_books_h2h": n_books,
                }

    found.update(best_by_pair)
    return found


def build_tournament_market(year: int) -> list[dict]:
    """Build per-game market records for one season."""
    expected = _load_expected_pairs(year)
    consensus_idx = _build_consensus_index(year)

    records: list[dict] = []
    missing: set[frozenset[str]] = set()

    for pair_key, game in expected.items():
        consensus = consensus_idx.get(pair_key)
        if consensus is not None:
            home_id = consensus["home_team_id"]
            home_prob = consensus["home_win_prob"]
            away_prob = consensus["away_win_prob"]
            n_books = consensus["n_books_h2h"]
            source = "consensus"
        else:
            missing.add(pair_key)
            continue

        t1 = normalize_team_id(str(game["team1_id"]))
        t2 = normalize_team_id(str(game["team2_id"]))
        # Assign probability from team1's perspective.
        if home_id == t1:
            t1_win_prob = home_prob
        else:
            t1_win_prob = away_prob

        records.append(
            {
                "year": year,
                "team1": t1,
                "team2": t2,
                "team1_win_prob": round(t1_win_prob, 6),
                "n_books": n_books,
                "source": source,
            }
        )

    # Gap-fill from raw archive for missed pairs.
    if missing:
        gap_fills = _gap_fill_from_raw(year, missing)
        for pair_key in list(missing):
            game = expected[pair_key]
            fill = gap_fills.get(pair_key)
            if fill is None:
                continue  # Genuinely absent from archive.
            t1 = normalize_team_id(str(game["team1_id"]))
            t2 = normalize_team_id(str(game["team2_id"]))
            home_id = fill["home_team_id"]
            home_prob = fill["home_win_prob"]
            away_prob = fill["away_win_prob"]
            t1_win_prob = home_prob if home_id == t1 else away_prob
            records.append(
                {
                    "year": year,
                    "team1": t1,
                    "team2": t2,
                    "team1_win_prob": round(t1_win_prob, 6),
                    "n_books": fill["n_books_h2h"],
                    "source": "raw_gap_fill",
                }
            )
            missing.discard(pair_key)

    n_total = len(expected)
    n_found = len(records)
    print(f"  {year}: {n_found}/{n_total} games with odds ({n_found / max(n_total, 1) * 100:.1f}%)")
    if missing:
        absent = [sorted(p) for p in sorted(missing)]
        print(f"    genuinely absent: {absent}")

    return records


def main() -> None:
    all_records: list[dict] = []
    coverage: dict[str, dict] = {}

    for year in SEASONS:
        records = build_tournament_market(year)
        all_records.extend(records)
        n_expected = len(_load_expected_pairs(year))
        coverage[str(year)] = {
            "n_games": n_expected,
            "n_with_odds": len(records),
            "coverage_pct": round(len(records) / max(n_expected, 1) * 100, 2),
        }

    # Sort for determinism.
    all_records.sort(key=lambda r: (r["year"], r["team1"], r["team2"]))

    artifact = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "The Odds API historical snapshots via odds_api_closing_consensus.json",
        "seasons": SEASONS,
        "coverage": coverage,
        "n_games": len(all_records),
        "games": all_records,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True))
    print(f"\nWrote {len(all_records)} game records to {OUTPUT_PATH}")
    for yr, cov in coverage.items():
        print(f"  {yr}: {cov['n_with_odds']}/{cov['n_games']} ({cov['coverage_pct']}%)")


if __name__ == "__main__":
    main()
