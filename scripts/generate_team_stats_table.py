"""Generate a multi-year pre-tournament team stats table for the web app.

Produces:
  docs/data/team_stats_by_year.json - Torvik pre-tournament stats for every
    tournament-qualified team, one row per team per year, 2010-2026.

Pulls straight from data/raw/historical/torvik_{year}.json (already
validated pre-tournament, see data_type check below) via the shared
scripts._common loaders — no new data-loading logic. Restricted to
tournament-qualified teams (matches docs/data/team_profiles.json's existing
convention) rather than the full ~350-team D1 field. 2020 has no
tournament (COVID) so load_seeds_and_regions(2020) returns empty and that
year is skipped naturally.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._common import HIST_DIR, load_seeds_and_regions, load_torvik_and_ff

OUT = PROJECT_ROOT / "docs" / "data"
OUT.mkdir(parents=True, exist_ok=True)

YEARS = range(2010, 2027)
VALID_PRETOURNAMENT_TYPES = {"pre_tournament", "pre_tournament_computed"}

STAT_FIELDS = (
    "conference",
    "t_rank",
    "barthag",
    "adj_offensive_efficiency",
    "adj_defensive_efficiency",
    "adj_tempo",
    "effective_fg_pct",
    "turnover_rate",
    "offensive_reb_rate",
    "free_throw_rate",
    "opp_effective_fg_pct",
    "opp_turnover_rate",
    "defensive_reb_rate",
    "opp_free_throw_rate",
)


def _data_type_for(year: int) -> str | None:
    path = HIST_DIR / f"torvik_{year}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f).get("data_type")


def build_year_rows(year: int) -> list[dict]:
    seeds, regions = load_seeds_and_regions(year)
    if not seeds:
        return []  # no tournament that year (2020)

    data_type = _data_type_for(year)
    if data_type not in VALID_PRETOURNAMENT_TYPES:
        print(f"  {year}: SKIP — data_type={data_type!r}, not pre-tournament")
        return []

    torvik, _ff = load_torvik_and_ff(year)
    rows = []
    for team_id, seed in seeds.items():
        t = torvik.get(team_id)
        if not t:
            continue
        row = {
            "team_id": team_id,
            "team_name": t.get("team_name"),
            "seed": seed,
            "region": regions.get(team_id),
        }
        for field in STAT_FIELDS:
            row[field] = t.get(field)
        rows.append(row)

    rows.sort(key=lambda r: r["t_rank"] if r["t_rank"] is not None else 999)
    return rows


def main() -> None:
    stats_by_year = {}
    for year in YEARS:
        rows = build_year_rows(year)
        if rows:
            stats_by_year[str(year)] = rows
            print(f"  {year}: {len(rows)} teams")

    payload = {
        "years": sorted(int(y) for y in stats_by_year),
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "stats_by_year": stats_by_year,
    }

    out_path = OUT / "team_stats_by_year.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"\nWrote {out_path} ({len(payload['years'])} years)")


if __name__ == "__main__":
    main()
