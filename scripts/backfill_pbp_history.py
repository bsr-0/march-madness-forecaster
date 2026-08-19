"""Historical play-by-play backfill driver.

Fetches pbp_{year}.json for a range of seasons via CBBpyPbpScraper, then
immediately builds clutch_features_{year}.json for each season as it
finishes, so results become usable incrementally rather than only after
every season completes.

This is a long-running, deliberately slow job (see cbbpy_pbp.py's
_DEFAULT_SCOREBOARD_DELAY / _DEFAULT_PBP_DELAY) -- tens of thousands of
requests across a full season range at a conservative pace. Safe to
interrupt and re-run: fetch_season_pbp checkpoints after every date, so a
restart resumes mid-season rather than starting over.

Resuming is automatic -- just re-run the same command; each season's
checkpoint (data/raw/historical/pbp_{year}.json) picks up where it left off.

Usage:
    python3 scripts/backfill_pbp_history.py --start-year 2008 --end-year 2026
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.features.clutch_metrics import build_season_clutch_features  # noqa: E402
from src.data.features.pbp_box_scores import build_season_shooting_features  # noqa: E402
from src.data.features.pbp_player_minutes import build_season_minutes_features  # noqa: E402
from src.data.scrapers.cbbpy_pbp import CBBpyPbpScraper  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/raw/historical/_pbp_backfill.log"),
    ],
)
logger = logging.getLogger("pbp_backfill")

DATA_ROOT = Path("data")
CACHE_DIR = DATA_ROOT / "raw" / "historical"

# Kept in sync with scripts/build_pbp_derived_features.py, which rebuilds
# these from already-fetched PBP without re-scraping.
BUILDERS = [
    ("clutch_features", build_season_clutch_features, "teams"),
    ("shooting_features", build_season_shooting_features, "teams"),
    ("player_minutes", build_season_minutes_features, "players"),
]


def run(start_year: int, end_year: int) -> None:
    scraper = CBBpyPbpScraper(cache_dir=str(CACHE_DIR))

    # Most recent seasons first -- most immediately useful for current
    # bracket-pool construction, and confirms the pipeline stays healthy
    # before sinking days into the oldest, least certain years.
    years = sorted(range(start_year, end_year + 1), reverse=True)

    for year in years:
        t0 = time.time()
        logger.info("=== Season %d: starting/resuming PBP fetch ===", year)
        try:
            pbp_payload = scraper.fetch_season_pbp(year)
        except Exception:
            logger.exception("Season %d: PBP fetch failed, moving on", year)
            continue

        n_games = len(pbp_payload.get("games", []))
        elapsed = time.time() - t0
        logger.info("Season %d: %d games, %.0fs", year, n_games, elapsed)

        if n_games == 0:
            logger.warning("Season %d: 0 games -- skipping derived feature builds", year)
            continue

        # All three derived feature sets come from the same payload. Failures
        # are per-builder so one bad derivation doesn't cost the others (and
        # never costs the fetched PBP, which is the expensive part).
        for name, builder, collection_key in BUILDERS:
            try:
                result = builder(year, DATA_ROOT, pbp_payload=pbp_payload)
            except Exception:
                logger.exception("Season %d: %s build failed", year, name)
                continue

            if not result:
                logger.warning("Season %d: %s produced nothing", year, name)
                continue

            out_path = CACHE_DIR / f"{name}_{year}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(
                "Season %d: wrote %s (%d %s)",
                year,
                out_path.name,
                len(result.get(collection_key, [])),
                collection_key,
            )

    logger.info("=== Backfill run finished (years %d-%d) ===", start_year, end_year)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2008)
    parser.add_argument("--end-year", type=int, default=2026)
    args = parser.parse_args()
    run(args.start_year, args.end_year)
