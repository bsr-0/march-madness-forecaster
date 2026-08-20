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
# (name, builder, collection_key, min_expected) -- min_expected is the smallest
# collection size that can plausibly come from a real season. Anything under it
# is silent data loss, not a quiet year, and is escalated rather than warned.
#
# player_minutes is 0 here on purpose: ESPN publishes no substitution events
# before 2025-02-11, so this builder legitimately yields nothing for every
# earlier season and there is no threshold that separates "broken" from
# "unsupported". Use the boxscore route instead --
# src/data/scrapers/espn_boxscore.py. The run-end summary reports its coverage
# either way so the gap stays visible.
BUILDERS = [
    ("clutch_features", build_season_clutch_features, "teams", 300),
    ("shooting_features", build_season_shooting_features, "teams", 300),
    ("player_minutes", build_season_minutes_features, "players", 0),
]


def run(start_year: int, end_year: int) -> None:
    scraper = CBBpyPbpScraper(cache_dir=str(CACHE_DIR))

    # Most recent seasons first -- most immediately useful for current
    # bracket-pool construction, and confirms the pipeline stays healthy
    # before sinking days into the oldest, least certain years.
    years = sorted(range(start_year, end_year + 1), reverse=True)

    # (name, year) -> collection size, or None when the build produced nothing.
    coverage: dict = {}

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
        for name, builder, collection_key, min_expected in BUILDERS:
            try:
                result = builder(year, DATA_ROOT, pbp_payload=pbp_payload)
            except Exception:
                logger.exception("Season %d: %s build failed", year, name)
                coverage[(name, year)] = None
                continue

            if not result:
                coverage[(name, year)] = None
                log = logger.error if min_expected else logger.warning
                log(
                    "Season %d: %s produced NOTHING from %d games -- no file written",
                    year,
                    name,
                    n_games,
                )
                continue

            n_items = len(result.get(collection_key, []))
            coverage[(name, year)] = n_items
            if min_expected and n_items < min_expected:
                # Loud on purpose: a thin-but-nonempty result still writes a
                # file, so without this it looks like a successful season.
                logger.error(
                    "Season %d: %s produced only %d %s from %d games (expected >= %d). "
                    "This is the signature of a parse failure or an upstream schema "
                    "change, not a real season.",
                    year,
                    name,
                    n_items,
                    collection_key,
                    n_games,
                    min_expected,
                )

            out_path = CACHE_DIR / f"{name}_{year}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(
                "Season %d: wrote %s (%d %s)",
                year,
                out_path.name,
                n_items,
                collection_key,
            )

    _log_coverage_summary(coverage, start_year, end_year)
    logger.info("=== Backfill run finished (years %d-%d) ===", start_year, end_year)


def _log_coverage_summary(coverage: dict, start_year: int, end_year: int) -> None:
    """Report per-builder coverage so gaps cannot hide in hours of scroll.

    The 2026-08-19 run silently produced zero player_minutes for 2024 and 26
    for 2023 while logging a steady stream of successful clutch/shooting
    writes. Anyone reading the tail of that log would reasonably conclude the
    backfill was healthy.
    """
    if not coverage:
        return
    logger.info("=== Coverage summary (years %d-%d) ===", start_year, end_year)
    for name, _builder, _collection_key, min_expected in BUILDERS:
        years = sorted((y for (n, y) in coverage if n == name), reverse=True)
        if not years:
            continue
        empty = [y for y in years if not coverage[(name, y)]]
        # A thin season still writes a file, so it looks like a success in the
        # log stream. Surface it here or it hides among the healthy seasons.
        thin = [
            y
            for y in years
            if coverage[(name, y)] and min_expected and coverage[(name, y)] < min_expected
        ]
        healthy = [y for y in years if y not in empty and y not in thin]
        notes = []
        if empty:
            notes.append(f"empty: {empty}")
        if thin:
            notes.append(f"THIN: {[(y, coverage[(name, y)]) for y in thin]}")
        logger.info(
            "  %-18s %2d/%2d seasons healthy (%s)",
            name,
            len(healthy),
            len(years),
            "; ".join(notes) if notes else "no gaps",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2008)
    parser.add_argument("--end-year", type=int, default=2026)
    args = parser.parse_args()
    run(args.start_year, args.end_year)
