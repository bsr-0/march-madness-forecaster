"""Build all PBP-derived feature files from already-fetched pbp_{year}.json.

Decoupled from the scrape on purpose: pbp_{year}.json holds the full play
payload, so clutch / box-score / player-minutes features can be rebuilt at
any time (after a rule change, a bug fix, or a new derivation) without
re-hitting ESPN. Safe to run while backfill_pbp_history.py is still going --
it only reads PBP files, and skips seasons whose fetch is incomplete unless
--include-incomplete is passed.

Writes, per season:
    clutch_features_{year}.json   blown-lead / late-game splits
    shooting_features_{year}.json 3PT%, FT%, eFG%, per-game counting stats
    player_minutes_{year}.json    per-player minutes (for returning-minutes work)

Usage:
    python3 scripts/build_pbp_derived_features.py
    python3 scripts/build_pbp_derived_features.py --start-year 2024 --end-year 2026
    python3 scripts/build_pbp_derived_features.py --include-incomplete
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.features.clutch_metrics import build_season_clutch_features  # noqa: E402
from src.data.features.pbp_box_scores import build_season_shooting_features  # noqa: E402
from src.data.features.pbp_player_minutes import build_season_minutes_features  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pbp_derive")

DATA_ROOT = Path("data")
CACHE_DIR = DATA_ROOT / "raw" / "historical"

# (name, builder, collection_key, min_expected) -- kept in sync with
# scripts/backfill_pbp_history.py, including the escalation thresholds. See
# that module for why player_minutes carries a threshold of 0 (ESPN publishes
# no substitution events before 2025-02-11, so this builder legitimately
# yields nothing for earlier seasons -- use src/data/scrapers/espn_boxscore.py).
BUILDERS = [
    ("clutch_features", build_season_clutch_features, "teams", 300),
    ("shooting_features", build_season_shooting_features, "teams", 300),
    ("player_minutes", build_season_minutes_features, "players", 0),
]


def build_year(year: int, include_incomplete: bool) -> None:
    pbp_path = CACHE_DIR / f"pbp_{year}.json"
    if not pbp_path.exists():
        return

    with open(pbp_path) as f:
        payload = json.load(f)

    meta = payload.get("metadata", {})
    if not meta.get("complete") and not include_incomplete:
        logger.info(
            "Season %d: PBP fetch incomplete (through %s) — skipping, pass --include-incomplete to build anyway",
            year,
            meta.get("last_completed_date"),
        )
        return

    n_games = len(payload.get("games", []))
    logger.info("Season %d: building derived features from %d games", year, n_games)

    for name, builder, collection_key, min_expected in BUILDERS:
        try:
            result = builder(year, DATA_ROOT, pbp_payload=payload)
        except Exception:
            logger.exception("Season %d: %s build failed", year, name)
            continue

        if not result:
            log = logger.error if min_expected else logger.warning
            log("Season %d: %s produced NOTHING from %d games -- no file written", year, name, n_games)
            continue

        n_items = len(result.get(collection_key, []))
        if min_expected and n_items < min_expected:
            logger.error(
                "Season %d: %s produced only %d %s from %d games (expected >= %d). "
                "This is the signature of a parse failure or an upstream schema change.",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2008)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Build features from seasons whose PBP fetch is still in progress.",
    )
    args = parser.parse_args()

    for year in range(args.end_year, args.start_year - 1, -1):
        build_year(year, args.include_incomplete)


if __name__ == "__main__":
    main()
