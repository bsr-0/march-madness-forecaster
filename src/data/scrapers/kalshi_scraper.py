"""Kalshi prediction market scraper for NCAA championship futures.

Fetches championship probability implied by Kalshi binary option prices.
No authentication required for market data endpoints.

The Kaggle 3rd-place solution (2026, Brier 0.1160) used Kalshi futures
as the third leg of their triple-market blend, converted to pairwise
probabilities via Bradley-Terry.

Endpoint: api.elections.kalshi.com/trade-api/v2/markets
Series: KXMARMAD (Men's NCAA Basketball Champion)

Coverage: 2025+ (Kalshi launched NCAA markets in 2025).
Prices are cents-on-the-dollar: 0.13 = 13% implied championship probability.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from src.data.normalize import normalize_team_id

logger = logging.getLogger(__name__)

_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
_SERIES_TICKER = "KXMARMAD"
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)

# Kalshi abbreviation → canonical team ID overrides.
# Only needed where Kalshi's abbreviation diverges from normalize_team_id output.
_KALSHI_OVERRIDES: Dict[str, str] = {
    "CONN": "connecticut",
    "KU": "kansas",
    "MICH": "michigan",
    "MSU": "michigan_state",
    "UGA": "georgia",
    "OSU": "ohio_state",
    "USC": "southern_california",
    "UK": "kentucky",
    "UNC": "north_carolina",
    "CUSE": "syracuse",
    "SJU": "st_john_s",
    "VT": "virginia_tech",
    "ISU": "iowa_state",
    "WSU": "wichita_state",
    "WAKE": "wake_forest",
    "NEB": "nebraska",
    "ORE": "oregon",
    "MISS": "mississippi",
    "PITT": "pittsburgh",
    "MARQ": "marquette",
    "PV": "prairie_view",
    "HAW": "hawaii",
    "AKR": "akron",
    "XAV": "xavier",
    "CBU": "california_baptist",
    "KENN": "kennesaw_state",
    "USF": "south_florida",
    "WICH": "wichita_state",
}


@dataclass
class KalshiTeamFutures:
    """Championship futures price for one team from Kalshi."""

    team_id: str  # canonical
    kalshi_abbrev: str
    ticker: str  # e.g. KXMARMAD-26-MICH
    championship_prob: float  # last traded price (0-1)
    status: str  # "active" or "settled"
    result: str  # "yes", "no", or ""
    close_time: str  # market close datetime


@dataclass
class KalshiSnapshot:
    """All championship futures for a single season."""

    season: int
    timestamp: str
    teams: Dict[str, KalshiTeamFutures]  # canonical_id → futures

    def bradley_terry_prob(self, team1_id: str, team2_id: str) -> float:
        """P(team1 beats team2) via Bradley-Terry from championship odds.

        BT: P(A>B) = strength_A / (strength_A + strength_B)
        where strength = championship_prob (or small floor for zero-prob teams).
        """
        p1 = self.teams.get(team1_id)
        p2 = self.teams.get(team2_id)

        s1 = max(p1.championship_prob, 0.001) if p1 else 0.001
        s2 = max(p2.championship_prob, 0.001) if p2 else 0.001

        return s1 / (s1 + s2)


class KalshiScraper:
    """Fetches NCAA championship futures from Kalshi."""

    def __init__(self, cache_dir: str = "data/raw/kalshi"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": _USER_AGENT})

    def fetch_championship_futures(self, season: int) -> KalshiSnapshot:
        """Fetch all championship futures for a season.

        Args:
            season: Tournament year (e.g. 2026, 2027).

        Returns:
            KalshiSnapshot with per-team championship probabilities.
        """
        from datetime import datetime

        logger.info("Fetching Kalshi championship futures for %d...", season)

        all_markets: List[Dict[str, Any]] = []
        cursor: Optional[str] = None

        while True:
            url = f"{_API_BASE}/markets?series_ticker={_SERIES_TICKER}&limit=200"
            if cursor:
                url += f"&cursor={cursor}"

            try:
                resp = self.session.get(url, timeout=15)
                resp.raise_for_status()
                data = resp.json()
            except (requests.RequestException, json.JSONDecodeError) as exc:
                logger.warning("Kalshi API error: %s", exc)
                break

            markets = data.get("markets", [])
            all_markets.extend(markets)
            cursor = data.get("cursor")
            if not cursor or not markets:
                break
            time.sleep(1.0)

        # Filter to requested season (Kalshi uses 2-digit year: -26-, -27-)
        season_tag = f"-{season % 100}-"
        season_markets = [m for m in all_markets if season_tag in m.get("ticker", "")]

        teams: Dict[str, KalshiTeamFutures] = {}
        for m in season_markets:
            parsed = self._parse_market(m, season)
            if parsed:
                teams[parsed.team_id] = parsed

        snapshot = KalshiSnapshot(
            season=season,
            timestamp=datetime.utcnow().isoformat() + "Z",
            teams=teams,
        )

        # Save raw
        raw_path = self.cache_dir / f"kalshi_futures_{season}.json"
        with open(raw_path, "w") as f:
            json.dump(
                {
                    "season": season,
                    "timestamp": snapshot.timestamp,
                    "n_teams": len(teams),
                    "teams": {k: asdict(v) for k, v in teams.items()},
                },
                f,
                indent=2,
            )

        # Save processed
        proc_dir = Path("data/processed/kalshi")
        proc_dir.mkdir(parents=True, exist_ok=True)
        proc_path = proc_dir / f"kalshi_{season}.json"
        with open(proc_path, "w") as f:
            json.dump(
                {
                    "season": season,
                    "timestamp": snapshot.timestamp,
                    "n_teams": len(teams),
                    "teams": {k: asdict(v) for k, v in teams.items()},
                },
                f,
                indent=2,
            )

        logger.info(
            "Got Kalshi futures for %d teams in %d (sum P=%.2f)",
            len(teams),
            season,
            sum(t.championship_prob for t in teams.values()),
        )
        return snapshot

    def _parse_market(self, market: Dict[str, Any], season: int) -> Optional[KalshiTeamFutures]:
        """Parse a single Kalshi market into KalshiTeamFutures."""
        ticker = market.get("ticker", "")
        season_tag = f"-{season % 100}-"
        if season_tag not in ticker:
            return None

        abbrev = ticker.split(season_tag)[1]
        canonical = _KALSHI_OVERRIDES.get(abbrev, normalize_team_id(abbrev))

        price = float(market.get("last_price_dollars") or "0")

        return KalshiTeamFutures(
            team_id=canonical,
            kalshi_abbrev=abbrev,
            ticker=ticker,
            championship_prob=price,
            status=market.get("status", ""),
            result=market.get("result", ""),
            close_time=market.get("close_time", ""),
        )

    @staticmethod
    def load_processed(season: int) -> Optional[KalshiSnapshot]:
        """Load processed Kalshi data from disk."""
        path = Path(f"data/processed/kalshi/kalshi_{season}.json")
        if not path.exists():
            return None

        with open(path) as f:
            raw = json.load(f)

        teams = {}
        for tid, tdata in raw.get("teams", {}).items():
            teams[tid] = KalshiTeamFutures(**tdata)

        return KalshiSnapshot(
            season=raw["season"],
            timestamp=raw.get("timestamp", ""),
            teams=teams,
        )


def main():
    parser = argparse.ArgumentParser(description="Kalshi NCAA Futures Scraper")
    parser.add_argument("--season", type=int, default=2027, help="Season year")
    parser.add_argument("--all", action="store_true", help="Scrape 2026 + 2027")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    scraper = KalshiScraper()
    seasons = [2026, 2027] if args.all else [args.season]

    for season in seasons:
        snapshot = scraper.fetch_championship_futures(season)
        logger.info("Season %d: %d teams", season, len(snapshot.teams))

        # Show top teams
        top = sorted(
            snapshot.teams.values(),
            key=lambda t: t.championship_prob,
            reverse=True,
        )[:10]
        for t in top:
            print(f"  {t.team_id:25s}  P(champ)={t.championship_prob:.2f}  [{t.kalshi_abbrev}]  {t.status}")

        # Demo Bradley-Terry
        if len(top) >= 2:
            t1, t2 = top[0], top[1]
            bt = snapshot.bradley_terry_prob(t1.team_id, t2.team_id)
            print(f"\n  BT: P({t1.team_id} > {t2.team_id}) = {bt:.3f}")


if __name__ == "__main__":
    main()
