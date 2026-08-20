"""Historical ESPN backfill driver (play-by-play + box scores).

Two scrapes per season, run sequentially in one process:

  1. play-by-play  -> pbp_{year}.json         -> clutch_features, shooting_features
  2. box scores    -> boxscores_{year}.json   -> player_minutes

Derived features are built as each season finishes, so results become usable
incrementally rather than only after every season completes.

**Why player_minutes comes from box scores, not play-by-play.** ESPN publishes
no substitution events before 2025-02-11, and the PBP minutes reconstruction
depends entirely on them: the 2026-08-19 run produced 9,581 players for 2026,
4,386 for 2025 (only from 2025-02-12 on), *zero* for 2024 and 26 for 2023. The
boxscore endpoint carries ESPN's own published per-player line, minutes first,
starters labelled, and works back to at least 2009. See FINDINGS.md 6d and
src/data/scrapers/espn_boxscore.py.

**Why one process rather than two.** Both scrapes share a ~2.0s per-request
politeness budget. Running them concurrently would halve the effective gap
between requests from this IP; running them sequentially keeps the footprint
identical to a single scrape and just takes longer. That matters because two
other ESPN access paths are already behind bot management (see cbbpy_pbp.py).

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
from src.data.features.boxscore_player_minutes import (  # noqa: E402
    MinutesCoverageError,
    build_season_minutes_features,
)
from src.data.scrapers.cbbpy_pbp import CBBpyPbpScraper  # noqa: E402
from src.data.scrapers.espn_boxscore import EspnBoxscoreScraper  # noqa: E402

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
# These two builders emit one row per TOURNAMENT-FIELD team (68), not per D-I
# team (~360). An earlier threshold of 300 was set from the wrong denominator
# and fired on every healthy season -- a guard that cries wolf every run is
# worse than no guard, because it trains you to skim past it. 60 leaves room
# for a field with an unresolved First Four slot without tolerating a real
# collapse.
#
# player_minutes is deliberately absent: it now comes from the boxscore stage
# below, which works for every season rather than only post-2025-02-11.
BUILDERS = [
    ("clutch_features", build_season_clutch_features, "teams", 60),
    ("shooting_features", build_season_shooting_features, "teams", 60),
]

# A real season yields several thousand distinct players; anything far below
# that is a parse failure rather than a quiet year.
MIN_EXPECTED_PLAYERS = 500


def run(
    start_year: int,
    end_year: int,
    *,
    skip_pbp: bool = False,
    skip_boxscore: bool = False,
) -> None:
    scraper = CBBpyPbpScraper(cache_dir=str(CACHE_DIR))

    # Most recent seasons first -- most immediately useful for current
    # bracket-pool construction, and confirms the pipeline stays healthy
    # before sinking days into the oldest, least certain years.
    years = sorted(range(start_year, end_year + 1), reverse=True)

    # (name, year) -> collection size, or None when the build produced nothing.
    coverage: dict = {}

    for year in years:
        if skip_pbp:
            if not skip_boxscore:
                _run_boxscore_stage(year, coverage)
            continue

        t0 = time.time()
        logger.info("=== Season %d: starting/resuming PBP fetch ===", year)
        try:
            pbp_payload = scraper.fetch_season_pbp(year)
        except Exception:
            logger.exception("Season %d: PBP fetch failed, moving on", year)
            if not skip_boxscore:
                _run_boxscore_stage(year, coverage)
            continue

        n_games = len(pbp_payload.get("games", []))
        elapsed = time.time() - t0
        logger.info("Season %d: %d games, %.0fs", year, n_games, elapsed)

        if n_games == 0:
            logger.warning("Season %d: 0 PBP games -- skipping PBP-derived builds", year)
            if not skip_boxscore:
                _run_boxscore_stage(year, coverage)
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

        # --- Box scores -> player_minutes ------------------------------
        # Runs after the PBP pass for this season so the two scrapes never
        # overlap; total wall time roughly doubles but the request rate seen
        # by ESPN is unchanged.
        if not skip_boxscore:
            _run_boxscore_stage(year, coverage)

    _log_coverage_summary(coverage, start_year, end_year)
    logger.info("=== Backfill run finished (years %d-%d) ===", start_year, end_year)


def _run_boxscore_stage(year: int, coverage: dict) -> None:
    """Scrape one season of box scores and build player_minutes from them."""
    t0 = time.time()
    logger.info("=== Season %d: starting/resuming boxscore fetch ===", year)
    scraper = EspnBoxscoreScraper(cache_dir=str(CACHE_DIR))
    try:
        payload = scraper.fetch_season(year)
    except Exception:
        logger.exception("Season %d: boxscore fetch failed, moving on", year)
        coverage[("player_minutes", year)] = None
        return

    n_games = len(payload.get("games", []))
    missing = payload.get("metadata", {}).get("games_missing_boxscore", 0)
    logger.info(
        "Season %d: %d boxscore games (%d pages had none), %.0fs",
        year,
        n_games,
        missing,
        time.time() - t0,
    )
    if n_games == 0:
        logger.error("Season %d: 0 boxscore games -- skipping player_minutes", year)
        coverage[("player_minutes", year)] = None
        return

    try:
        result = build_season_minutes_features(year, DATA_ROOT, boxscore_payload=payload)
    except MinutesCoverageError as exc:
        # Raised rather than logged: a thin artifact looks like success on disk.
        logger.error("Season %d: player_minutes rejected -- %s", year, exc)
        coverage[("player_minutes", year)] = None
        return
    except Exception:
        logger.exception("Season %d: player_minutes build failed", year)
        coverage[("player_minutes", year)] = None
        return

    n_players = len(result.get("players", []))
    coverage[("player_minutes", year)] = n_players
    out_path = CACHE_DIR / f"player_minutes_{year}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Season %d: wrote %s (%d players)", year, out_path.name, n_players)


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
    tracked = [(name, min_expected) for name, _b, _k, min_expected in BUILDERS]
    tracked.append(("player_minutes", MIN_EXPECTED_PLAYERS))
    for name, min_expected in tracked:
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
    parser.add_argument(
        "--skip-pbp",
        action="store_true",
        help="Skip the play-by-play scrape; only fetch box scores / player_minutes.",
    )
    parser.add_argument(
        "--skip-boxscore",
        action="store_true",
        help="Skip the boxscore scrape; only fetch play-by-play / clutch+shooting.",
    )
    args = parser.parse_args()
    run(
        args.start_year,
        args.end_year,
        skip_pbp=args.skip_pbp,
        skip_boxscore=args.skip_boxscore,
    )
