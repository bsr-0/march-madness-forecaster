"""Point-in-time Torvik Four Factors lookup for training.

Loads monthly trank.php snapshots (e.g. torvik_four_factors_2024_20241231.json)
and returns the most recent snapshot before a given game date. This eliminates
the train/predict FF mismatch: training now uses Torvik FF (from trank.php)
instead of box-score-computed FF from ProprietaryMetricsEngine._four_factors().
"""

from __future__ import annotations

import json
import logging
import re
from bisect import bisect_right
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

HIST_DIR = Path("data/raw/historical")

# 8 FF fields matching ProprietaryTeamMetrics attribute names
FF_FIELDS = (
    "effective_fg_pct",
    "turnover_rate",
    "offensive_reb_rate",
    "free_throw_rate",
    "opp_effective_fg_pct",
    "opp_turnover_rate",
    "defensive_reb_rate",
    "opp_free_throw_rate",
)

# Pattern: torvik_four_factors_{year}_{YYYYMMDD}.json
_SNAPSHOT_RE = re.compile(r"torvik_four_factors_(\d{4})_(\d{8})\.json$")


class TorVikFFLookup:
    """Point-in-time Four Factors lookup from monthly trank.php snapshots.

    Scans data/raw/historical/ for dated snapshot files, sorts by cutoff date,
    and provides binary-search lookup for the most recent snapshot before any
    given game date.
    """

    def __init__(self, year: int, data_dir: Optional[Path] = None):
        self._year = year
        self._data_dir = data_dir or HIST_DIR
        # Sorted list of (date_str, snapshot_data) pairs
        # date_str is "YYYY-MM-DD" for bisect compatibility with game_date strings
        self._snapshots: list[tuple[str, dict]] = []
        self._load_snapshots()

    def _filter_team_data(self, data: dict) -> dict[str, dict]:
        """Index a snapshot payload by team_id, dropping metadata keys."""
        team_data: dict[str, dict] = {}
        for key, val in data.items():
            if isinstance(val, dict) and any(k in val for k in FF_FIELDS[:2]):
                team_data[key] = val
        return team_data

    def _load_snapshots(self) -> None:
        """Load dated snapshots sorted by cutoff date.

        Prefers the merged `torvik_{year}.json`'s "four_factors_snapshots"
        array when present, falling back to globbing the standalone dated
        snapshot files."""
        merged_path = self._data_dir / f"torvik_{self._year}.json"
        merged_snapshots = None
        if merged_path.exists():
            try:
                with open(merged_path) as f:
                    merged = json.load(f)
                if "four_factors_snapshots" in merged:
                    merged_snapshots = merged["four_factors_snapshots"]
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load %s: %s", merged_path, e)

        if merged_snapshots is not None:
            for entry in merged_snapshots:
                team_data = self._filter_team_data(entry["data"])
                if team_data:
                    self._snapshots.append((entry["date"], team_data))
        else:
            pattern = f"torvik_four_factors_{self._year}_*.json"
            files = sorted(self._data_dir.glob(pattern))

            for fpath in files:
                m = _SNAPSHOT_RE.search(fpath.name)
                if not m:
                    continue
                date_str_raw = m.group(2)  # YYYYMMDD
                # Convert to YYYY-MM-DD for string comparison with game_date
                date_str = f"{date_str_raw[:4]}-{date_str_raw[4:6]}-{date_str_raw[6:8]}"
                try:
                    with open(fpath) as f:
                        data = json.load(f)
                    team_data = self._filter_team_data(data)
                    if team_data:
                        self._snapshots.append((date_str, team_data))
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning("Failed to load snapshot %s: %s", fpath, e)

        self._snapshots.sort(key=lambda x: x[0])
        if self._snapshots:
            logger.info(
                "TorVikFFLookup: loaded %d snapshots for %d (dates: %s)",
                len(self._snapshots),
                self._year,
                [s[0] for s in self._snapshots],
            )

    @property
    def has_snapshots(self) -> bool:
        return len(self._snapshots) > 0

    @property
    def n_snapshots(self) -> int:
        return len(self._snapshots)

    def get_ff(self, team_id: str, game_date: str) -> Optional[Dict[str, float]]:
        """Get most recent FF snapshot before game_date for a team.

        Args:
            team_id: Normalized team ID.
            game_date: ISO date string (YYYY-MM-DD).

        Returns:
            Dict of 8 FF fields, or None if no snapshot available before game_date.
        """
        if not self._snapshots:
            return None

        # Binary search: find rightmost snapshot with date < game_date
        dates = [s[0] for s in self._snapshots]
        idx = bisect_right(dates, game_date) - 1
        if idx < 0:
            return None

        _, team_data = self._snapshots[idx]
        ff = team_data.get(team_id)
        if ff is None:
            return None

        return {k: float(ff.get(k, 0.0)) for k in FF_FIELDS}

    def overlay_metrics(self, metrics: dict, game_date: str) -> int:
        """Overlay Torvik FF onto ProprietaryTeamMetrics objects in-place.

        Args:
            metrics: Dict[str, ProprietaryTeamMetrics] from compute_as_of().
            game_date: ISO date string for point-in-time lookup.

        Returns:
            Number of teams successfully overlaid.
        """
        if not self._snapshots:
            return 0

        count = 0
        for team_id, m in metrics.items():
            ff = self.get_ff(team_id, game_date)
            if ff is None:
                continue
            for field in FF_FIELDS:
                val = ff.get(field, 0.0)
                if val > 0:
                    setattr(m, field, val)
            count += 1
        return count
