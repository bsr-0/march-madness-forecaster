"""Kaggle competition CSV data loader.

Loads and normalizes all standard Kaggle March Machine Learning Mania
competition CSV files into the pipeline's canonical format. This provides
an authoritative, offline-friendly data source that complements (and can
serve as a fallback for) the live scrapers.

Supported datasets:
  - MRegularSeasonCompactResults / MRegularSeasonDetailedResults
  - MNCAATourneyCompactResults / MNCAATourneyDetailedResults
  - MNCAATourneySeeds
  - MMasseyOrdinals          (→ ExternalRatingsLoader integration)
  - MTeamCoaches
  - MTeamConferences / MConferences
  - MSeasons
  - MGameCities / Cities
  - MSecondaryTourneyCompactResults / MSecondaryTourneyTeams
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datetime import date

from .normalize import normalize_team_id

logger = logging.getLogger(__name__)


# Selection Sunday dates — duplicated from pit_validation.py to avoid
# cross-layer import (data module should not import from pipeline).
# These are fixed historical dates that change only when new seasons are added.
_SELECTION_SUNDAY_DATES: Dict[int, date] = {
    2016: date(2016, 3, 13), 2017: date(2017, 3, 12), 2018: date(2018, 3, 11),
    2019: date(2019, 3, 17), 2021: date(2021, 3, 14), 2022: date(2022, 3, 13),
    2023: date(2023, 3, 12), 2024: date(2024, 3, 17), 2025: date(2025, 3, 16),
    2026: date(2026, 3, 15),
}


def _compute_max_ranking_day(season: int, day_zero_str: Optional[str] = None) -> int:
    """Compute the maximum safe ranking day number for a season.

    Returns the day number corresponding to Selection Sunday (the last day
    before the NCAA tournament bracket is finalized), or 133 as a conservative
    fallback when dates are unavailable.

    This prevents accidentally loading Massey Ordinal rankings from post-
    tournament dates, which would leak tournament outcomes into features.
    """
    sel_sun = _SELECTION_SUNDAY_DATES.get(season)
    if sel_sun is None or not day_zero_str:
        return 133  # Conservative fallback (~Feb 24 from Oct 14 DayZero)
    try:
        day_zero = date.fromisoformat(str(day_zero_str))
        return (sel_sun - day_zero).days
    except (ValueError, TypeError):
        return 133


class KaggleDataLoader:
    """Load Kaggle competition CSV files from a local directory.

    Typical usage::

        loader = KaggleDataLoader("data/kaggle")
        games = loader.load_regular_season_detailed(2025)
        ordinals = loader.load_massey_ordinals(2025)
    """

    def __init__(self, kaggle_dir: str):
        self.kaggle_dir = Path(kaggle_dir)
        self._teams_cache: Dict[int, str] = {}

    # ------------------------------------------------------------------
    # Team ID helpers
    # ------------------------------------------------------------------

    def load_teams(self, prefix: str = "M") -> Dict[int, str]:
        """Load {prefix}Teams.csv → {TeamID: TeamName}."""
        if self._teams_cache:
            return self._teams_cache
        path = self._find_csv(f"{prefix}Teams")
        if path is None:
            return {}
        result: Dict[int, str] = {}
        for row in self._read_csv(path):
            try:
                team_id = int(row.get("TeamID", 0))
                team_name = row.get("TeamName", "").strip()
                if team_id and team_name:
                    result[team_id] = team_name
            except (ValueError, TypeError):
                continue
        self._teams_cache = result
        return result

    def _team_name(self, team_id: int) -> str:
        """Resolve a Kaggle TeamID to a team name."""
        if not self._teams_cache:
            self.load_teams()
        return self._teams_cache.get(team_id, str(team_id))

    def _canonical_id(self, team_id: int) -> str:
        """Convert a Kaggle TeamID to the pipeline's canonical team_id."""
        name = self._team_name(team_id)
        return normalize_team_id(name) if name != str(team_id) else str(team_id)

    # ------------------------------------------------------------------
    # Regular season results
    # ------------------------------------------------------------------

    def load_regular_season_compact(self, season: int) -> List[Dict]:
        """Load MRegularSeasonCompactResults.csv for a given season.

        Returns a list of game dicts in the pipeline's canonical format.
        """
        return self._load_results("MRegularSeasonCompactResults", season, detailed=False)

    def load_regular_season_detailed(self, season: int) -> List[Dict]:
        """Load MRegularSeasonDetailedResults.csv for a given season.

        Returns game dicts with Four Factors box score data.
        """
        return self._load_results("MRegularSeasonDetailedResults", season, detailed=True)

    # ------------------------------------------------------------------
    # Tournament results
    # ------------------------------------------------------------------

    def load_tourney_compact(self, season: int) -> List[Dict]:
        """Load MNCAATourneyCompactResults.csv for a given season."""
        return self._load_results("MNCAATourneyCompactResults", season, detailed=False)

    def load_tourney_detailed(self, season: int) -> List[Dict]:
        """Load MNCAATourneyDetailedResults.csv for a given season."""
        return self._load_results("MNCAATourneyDetailedResults", season, detailed=True)

    # ------------------------------------------------------------------
    # Tournament seeds
    # ------------------------------------------------------------------

    def load_tourney_seeds(self, season: int) -> List[Dict]:
        """Load MNCAATourneySeeds.csv for a given season.

        Returns list of {team_id, team_name, seed, region}.
        """
        path = self._find_csv("MNCAATourneySeeds")
        if path is None:
            return []
        result: List[Dict] = []
        for row in self._read_csv(path):
            if self._int(row.get("Season")) != season:
                continue
            kaggle_id = self._int(row.get("TeamID"))
            seed_str = row.get("Seed", "").strip()
            if not kaggle_id or not seed_str:
                continue
            region = seed_str[0]
            seed_num = int("".join(c for c in seed_str[1:] if c.isdigit()) or "16")
            result.append({
                "team_id": self._canonical_id(kaggle_id),
                "team_name": self._team_name(kaggle_id),
                "kaggle_team_id": kaggle_id,
                "seed": seed_num,
                "seed_str": seed_str,
                "region": region,
            })
        logger.info("Loaded %d tournament seeds for %d from Kaggle CSV", len(result), season)
        return result

    # ------------------------------------------------------------------
    # Massey Ordinals (key integration)
    # ------------------------------------------------------------------

    def load_massey_ordinals(
        self,
        season: int,
        *,
        ranking_day_num: Optional[int] = None,
        max_day: Optional[int] = None,
    ) -> Dict[str, Dict[str, "MasseyOrdinalEntry"]]:
        """Load MMasseyOrdinals.csv for a given season.

        Returns a dict of {system_name: {canonical_team_id: MasseyOrdinalEntry}}.
        When ``ranking_day_num`` is None, uses the latest available day per system,
        capped at ``max_day`` to prevent loading post-Selection-Sunday rankings
        that could leak tournament outcomes.

        Args:
            season: Season year (e.g. 2026 for 2025-26).
            ranking_day_num: Specific day number (overrides auto-selection).
            max_day: Maximum day number to consider when auto-selecting.
                If None, dynamically computed from Selection Sunday + DayZero.
        """
        path = self._find_csv("MMasseyOrdinals")
        if path is None:
            # Try alternative filename patterns
            path = self._find_csv("MMasseyOrdinals_thruDay128")
            if path is None:
                path = self._find_csv("MMasseyOrdinals_thru_Season2026_Day128")
        if path is None:
            logger.warning("MMasseyOrdinals CSV not found in %s", self.kaggle_dir)
            return {}

        # Pass 1: collect all entries for the season, grouped by system
        by_system: Dict[str, Dict[int, List[Dict]]] = defaultdict(lambda: defaultdict(list))
        for row in self._read_csv(path):
            if self._int(row.get("Season")) != season:
                continue
            system = row.get("SystemName", "").strip()
            day_num = self._int(row.get("RankingDayNum"))
            kaggle_id = self._int(row.get("TeamID"))
            ordinal_rank = self._int(row.get("OrdinalRank"))
            if not system or not kaggle_id:
                continue
            if ordinal_rank is not None and ordinal_rank <= 0:
                continue
            if ordinal_rank is not None and ordinal_rank > 500:
                logger.debug(
                    "Massey Ordinals: unusually high rank %d for team %d "
                    "in system '%s' day %d (season %d)",
                    ordinal_rank, kaggle_id, system, day_num, season,
                )
            by_system[system][day_num].append({
                "kaggle_id": kaggle_id,
                "rank": ordinal_rank,
            })

        # Auto-compute max_day from Selection Sunday + DayZero if not provided
        if max_day is None and ranking_day_num is None:
            seasons = self.load_seasons()
            dz = next((s["day_zero"] for s in seasons if s["season"] == season), None)
            max_day = _compute_max_ranking_day(season, dz)
            logger.debug(
                "Massey Ordinals for %d: auto-computed max_day=%d from Selection Sunday",
                season, max_day,
            )

        # Pass 2: for each system, pick the target day (latest or specified)
        result: Dict[str, Dict[str, MasseyOrdinalEntry]] = {}
        for system, days in by_system.items():
            if ranking_day_num is not None:
                target_day = ranking_day_num
            else:
                # Cap at max_day to prevent loading post-Selection-Sunday rankings
                available_days = sorted(days.keys())
                if max_day is not None:
                    safe_days = [d for d in available_days if d <= max_day]
                    excluded = [d for d in available_days if d > max_day]
                    if excluded and system == next(iter(by_system)):  # Log once
                        logger.info(
                            "Massey Ordinals for %d: excluded %d post-ceiling days "
                            "(max_day=%d, excluded max=%d)",
                            season, len(excluded), max_day, max(excluded),
                        )
                    if safe_days:
                        available_days = safe_days
                    else:
                        logger.warning(
                            "Massey Ordinals for %d system '%s': no days <= max_day=%d; "
                            "skipping system to prevent leakage (available days: %s)",
                            season, system, max_day,
                            sorted(available_days)[:5],
                        )
                        continue
                target_day = max(available_days)
            entries = days.get(target_day, [])
            if not entries:
                # Fall back to closest earlier day
                ceiling = ranking_day_num or max_day or 999
                earlier = [d for d in days.keys() if d <= ceiling]
                if earlier:
                    target_day = max(earlier)
                    entries = days[target_day]
            if not entries:
                continue

            team_map: Dict[str, MasseyOrdinalEntry] = {}
            n_duplicates = 0
            for e in entries:
                cid = self._canonical_id(e["kaggle_id"])
                if cid in team_map:
                    n_duplicates += 1
                team_map[cid] = MasseyOrdinalEntry(
                    system_name=system,
                    team_id=cid,
                    team_name=self._team_name(e["kaggle_id"]),
                    kaggle_team_id=e["kaggle_id"],
                    ordinal_rank=e["rank"],
                    ranking_day_num=target_day,
                )
            if n_duplicates > 0:
                logger.warning(
                    "Massey Ordinals for %d system '%s' day %d: "
                    "%d duplicate team entries (last entry wins)",
                    season, system, target_day, n_duplicates,
                )
            result[system] = team_map

        n_systems = len(result)
        n_teams = max((len(v) for v in result.values()), default=0)
        logger.info(
            "Loaded Massey Ordinals for %d: %d systems, up to %d teams each",
            season, n_systems, n_teams,
        )
        return result

    def load_massey_ordinals_as_external_ratings(
        self,
        season: int,
        *,
        ranking_day_num: Optional[int] = None,
        max_day: Optional[int] = None,
    ) -> Dict[str, List[Dict]]:
        """Load Massey Ordinals and convert to ExternalRatingsLoader cache format.

        Returns {system_name: [{"team_name", "team_id", "rating", "ranking", "normalized"}]}
        suitable for saving via ExternalRatingsLoader.save_system().
        """
        ordinals = self.load_massey_ordinals(
            season, ranking_day_num=ranking_day_num, max_day=max_day,
        )
        if not ordinals:
            return {}

        result: Dict[str, List[Dict]] = {}
        for system, teams in ordinals.items():
            if not teams:
                continue
            n_teams = len(teams)
            entries: List[Dict] = []
            for entry in teams.values():
                # Invert rank to a rating (higher = better)
                rating = max(n_teams - entry.ordinal_rank + 1, 0)
                normalized = 1.0 - (entry.ordinal_rank - 1) / max(n_teams - 1, 1)
                entries.append({
                    "team_name": entry.team_name,
                    "team_id": entry.team_id,
                    "rating": float(rating),
                    "ranking": entry.ordinal_rank,
                    "normalized": round(normalized, 6),
                })
            result[system] = entries

        return result

    # ------------------------------------------------------------------
    # Coaches
    # ------------------------------------------------------------------

    def load_team_coaches(self, season: int) -> List[Dict]:
        """Load MTeamCoaches.csv for a given season.

        Returns list of {team_id, team_name, coach_name, first_day, last_day}.
        """
        path = self._find_csv("MTeamCoaches")
        if path is None:
            return []
        result: List[Dict] = []
        for row in self._read_csv(path):
            if self._int(row.get("Season")) != season:
                continue
            kaggle_id = self._int(row.get("TeamID"))
            coach = row.get("CoachName", "").strip()
            if not kaggle_id or not coach:
                continue
            result.append({
                "team_id": self._canonical_id(kaggle_id),
                "team_name": self._team_name(kaggle_id),
                "kaggle_team_id": kaggle_id,
                "coach_name": coach,
                "first_day_num": self._int(row.get("FirstDayNum")),
                "last_day_num": self._int(row.get("LastDayNum")),
            })
        return result

    # ------------------------------------------------------------------
    # Conferences
    # ------------------------------------------------------------------

    def load_team_conferences(self, season: int) -> Dict[str, str]:
        """Load MTeamConferences.csv for a given season.

        Returns {canonical_team_id: conference_abbrev}.
        """
        path = self._find_csv("MTeamConferences")
        if path is None:
            return {}
        result: Dict[str, str] = {}
        for row in self._read_csv(path):
            if self._int(row.get("Season")) != season:
                continue
            kaggle_id = self._int(row.get("TeamID"))
            conf = row.get("ConfAbbrev", "").strip()
            if kaggle_id and conf:
                result[self._canonical_id(kaggle_id)] = conf
        return result

    def load_conferences(self) -> Dict[str, str]:
        """Load MConferences.csv → {ConfAbbrev: Description}."""
        path = self._find_csv("MConferences") or self._find_csv("Conferences")
        if path is None:
            return {}
        result: Dict[str, str] = {}
        for row in self._read_csv(path):
            abbrev = row.get("ConfAbbrev", "").strip()
            desc = row.get("Description", "").strip()
            if abbrev:
                result[abbrev] = desc or abbrev
        return result

    # ------------------------------------------------------------------
    # Seasons
    # ------------------------------------------------------------------

    def load_seasons(self) -> List[Dict]:
        """Load MSeasons.csv → list of season metadata dicts."""
        path = self._find_csv("MSeasons")
        if path is None:
            return []
        result: List[Dict] = []
        for row in self._read_csv(path):
            result.append({
                "season": self._int(row.get("Season")),
                "day_zero": row.get("DayZero", ""),
                "region_w": row.get("RegionW", ""),
                "region_x": row.get("RegionX", ""),
                "region_y": row.get("RegionY", ""),
                "region_z": row.get("RegionZ", ""),
            })
        return result

    # ------------------------------------------------------------------
    # Game cities
    # ------------------------------------------------------------------

    def load_game_cities(self, season: int) -> Dict[str, Dict]:
        """Load MGameCities.csv for a season.

        Returns {game_id_str: {city, state, host_name}}.
        """
        path = self._find_csv("MGameCities")
        if path is None:
            return {}
        result: Dict[str, Dict] = {}
        for row in self._read_csv(path):
            if self._int(row.get("Season")) != season:
                continue
            day = self._int(row.get("DayNum"))
            wid = self._int(row.get("WTeamID"))
            lid = self._int(row.get("LTeamID"))
            key = f"{season}_{day}_{wid}_{lid}"
            city_id = self._int(row.get("CRType", row.get("CityID", 0)))
            result[key] = {
                "city_id": city_id,
                "host_name": row.get("HostName", ""),
            }
        return result

    # ------------------------------------------------------------------
    # Summary / inventory
    # ------------------------------------------------------------------

    def list_available_files(self) -> List[str]:
        """Return CSV basenames found in the kaggle directory."""
        if not self.kaggle_dir.is_dir():
            return []
        return sorted(p.stem for p in self.kaggle_dir.glob("*.csv"))

    def summary(self, season: int) -> Dict[str, Any]:
        """Return a quick summary of available Kaggle data for a season."""
        info: Dict[str, Any] = {"season": season, "kaggle_dir": str(self.kaggle_dir)}
        info["available_csvs"] = self.list_available_files()
        info["teams_count"] = len(self.load_teams())

        rs = self.load_regular_season_compact(season)
        info["regular_season_games"] = len(rs)

        ts = self.load_tourney_seeds(season)
        info["tourney_seeds"] = len(ts)

        tc = self.load_tourney_compact(season)
        info["tourney_games"] = len(tc)

        ordinals = self.load_massey_ordinals(season)
        info["massey_ordinal_systems"] = len(ordinals)

        coaches = self.load_team_coaches(season)
        info["coaches"] = len(coaches)

        confs = self.load_team_conferences(season)
        info["team_conferences"] = len(confs)

        return info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_results(
        self,
        csv_name: str,
        season: int,
        *,
        detailed: bool = False,
    ) -> List[Dict]:
        """Generic loader for Compact/Detailed results CSVs."""
        path = self._find_csv(csv_name)
        if path is None:
            return []
        result: List[Dict] = []
        for row in self._read_csv(path):
            if self._int(row.get("Season")) != season:
                continue
            w_id = self._int(row.get("WTeamID"))
            l_id = self._int(row.get("LTeamID"))
            w_score = self._int(row.get("WScore"))
            l_score = self._int(row.get("LScore"))
            day_num = self._int(row.get("DayNum"))
            num_ot = self._int(row.get("NumOT"))
            w_loc = row.get("WLoc", "N").strip()

            game_id = f"kaggle_{season}_{day_num}_{w_id}_{l_id}"

            # Compute possessions from detailed box score if available
            possessions = 0.0
            game = {
                "game_id": game_id,
                "season": season,
                "day_num": day_num,
                "team_id": self._canonical_id(w_id),
                "team_name": self._team_name(w_id),
                "opponent_id": self._canonical_id(l_id),
                "opponent_name": self._team_name(l_id),
                "team_score": w_score,
                "opponent_score": l_score,
                "kaggle_w_team_id": w_id,
                "kaggle_l_team_id": l_id,
                "location": w_loc,
                "num_ot": num_ot,
            }

            if detailed:
                # Winner stats
                w_fgm = self._float(row.get("WFGM"))
                w_fga = self._float(row.get("WFGA"))
                w_fgm3 = self._float(row.get("WFGM3"))
                w_fga3 = self._float(row.get("WFGA3"))
                w_ftm = self._float(row.get("WFTM"))
                w_fta = self._float(row.get("WFTA"))
                w_or = self._float(row.get("WOR"))
                w_dr = self._float(row.get("WDR"))
                w_ast = self._float(row.get("WAst"))
                w_to = self._float(row.get("WTO"))
                w_stl = self._float(row.get("WStl"))
                w_blk = self._float(row.get("WBlk"))
                w_pf = self._float(row.get("WPF"))
                # Loser stats
                l_fgm = self._float(row.get("LFGM"))
                l_fga = self._float(row.get("LFGA"))
                l_fgm3 = self._float(row.get("LFGM3"))
                l_fga3 = self._float(row.get("LFGA3"))
                l_ftm = self._float(row.get("LFTM"))
                l_fta = self._float(row.get("LFTA"))
                l_or = self._float(row.get("LOR"))
                l_dr = self._float(row.get("LDR"))
                l_ast = self._float(row.get("LAst"))
                l_to = self._float(row.get("LTO"))
                l_stl = self._float(row.get("LStl"))
                l_blk = self._float(row.get("LBlk"))
                l_pf = self._float(row.get("LPF"))

                # Estimate possessions (Dean Oliver formula)
                w_poss = w_fga - w_or + w_to + 0.475 * w_fta
                l_poss = l_fga - l_or + l_to + 0.475 * l_fta
                possessions = (w_poss + l_poss) / 2.0

                game.update({
                    "possessions": possessions,
                    "fgm": w_fgm, "fga": w_fga,
                    "fg3m": w_fgm3, "fg3a": w_fga3,
                    "ftm": w_ftm, "fta": w_fta,
                    "orb": w_or, "drb": w_dr,
                    "ast": w_ast, "turnovers": w_to,
                    "stl": w_stl, "blk": w_blk, "pf": w_pf,
                    # Opponent box score
                    "opp_fgm": l_fgm, "opp_fga": l_fga,
                    "opp_fg3m": l_fgm3, "opp_fg3a": l_fga3,
                    "opp_ftm": l_ftm, "opp_fta": l_fta,
                    "opp_orb": l_or, "opp_drb": l_dr,
                    "opp_ast": l_ast, "opp_turnovers": l_to,
                    "opp_stl": l_stl, "opp_blk": l_blk, "opp_pf": l_pf,
                })

            result.append(game)

            # Also add the mirror game (from loser's perspective)
            mirror = {
                "game_id": game_id,
                "season": season,
                "day_num": day_num,
                "team_id": self._canonical_id(l_id),
                "team_name": self._team_name(l_id),
                "opponent_id": self._canonical_id(w_id),
                "opponent_name": self._team_name(w_id),
                "team_score": l_score,
                "opponent_score": w_score,
                "kaggle_w_team_id": w_id,
                "kaggle_l_team_id": l_id,
                "location": {"H": "A", "A": "H"}.get(w_loc, "N"),
                "num_ot": num_ot,
            }
            if detailed:
                mirror.update({
                    "possessions": possessions,
                    "fgm": l_fgm, "fga": l_fga,
                    "fg3m": l_fgm3, "fg3a": l_fga3,
                    "ftm": l_ftm, "fta": l_fta,
                    "orb": l_or, "drb": l_dr,
                    "ast": l_ast, "turnovers": l_to,
                    "stl": l_stl, "blk": l_blk, "pf": l_pf,
                    "opp_fgm": w_fgm, "opp_fga": w_fga,
                    "opp_fg3m": w_fgm3, "opp_fg3a": w_fga3,
                    "opp_ftm": w_ftm, "opp_fta": w_fta,
                    "opp_orb": w_or, "opp_drb": w_dr,
                    "opp_ast": w_ast, "opp_turnovers": w_to,
                    "opp_stl": w_stl, "opp_blk": w_blk, "opp_pf": w_pf,
                })
            result.append(mirror)

        logger.info("Loaded %d game rows from %s for %d", len(result), csv_name, season)
        return result

    def _find_csv(self, basename: str) -> Optional[Path]:
        """Find a CSV file by basename, trying common patterns."""
        candidates = [
            self.kaggle_dir / f"{basename}.csv",
            self.kaggle_dir / f"{basename.lower()}.csv",
            # Kaggle sometimes uses Stage1/Stage2 variants
            self.kaggle_dir / f"{basename}_Stage1.csv",
            self.kaggle_dir / f"{basename}_Stage2.csv",
        ]
        # Also try glob for partial matches (e.g., MMasseyOrdinals_thruDay128)
        for c in candidates:
            if c.exists():
                return c
        # Glob fallback
        matches = sorted(self.kaggle_dir.glob(f"{basename}*.csv"))
        if matches:
            return matches[-1]  # Latest / most complete
        return None

    def _read_csv(self, path: Path) -> List[Dict[str, str]]:
        """Read a CSV file into a list of dicts."""
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                return list(reader)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", path, exc)
            return []

    @staticmethod
    def _int(value) -> int:
        if value is None:
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _float(value) -> float:
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0


class MasseyOrdinalEntry:
    """A single Massey Ordinal entry for a team in a given system."""

    __slots__ = (
        "system_name", "team_id", "team_name",
        "kaggle_team_id", "ordinal_rank", "ranking_day_num",
    )

    def __init__(
        self,
        system_name: str,
        team_id: str,
        team_name: str,
        kaggle_team_id: int,
        ordinal_rank: int,
        ranking_day_num: int,
    ):
        self.system_name = system_name
        self.team_id = team_id
        self.team_name = team_name
        self.kaggle_team_id = kaggle_team_id
        self.ordinal_rank = ordinal_rank
        self.ranking_day_num = ranking_day_num

    def __repr__(self) -> str:
        return (
            f"MasseyOrdinalEntry({self.system_name!r}, {self.team_name!r}, "
            f"rank={self.ordinal_rank}, day={self.ranking_day_num})"
        )
