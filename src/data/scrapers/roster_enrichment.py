"""Cross-reference cbbpy roster data across years to populate eligibility_year and is_transfer."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Suffixes to strip from player names for matching
_SUFFIX_PATTERN = re.compile(
    r"\s+(jr\.?|sr\.?|ii|iii|iv|v|vi|vii|viii)$", re.IGNORECASE
)
# Non-alphanumeric to underscore
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


class RosterEnrichment:
    """Cross-references roster data across years to populate eligibility_year and is_transfer."""

    def __init__(self, roster_dir: str, output_dir: Optional[str] = None):
        self.roster_dir = Path(roster_dir)
        self.output_dir = Path(output_dir) if output_dir else self.roster_dir

    @staticmethod
    def normalize_player_name(name: str) -> str:
        """Normalize a player name for cross-year matching.

        Lowercases, strips suffixes (Jr., III, etc.), replaces non-alnum with _.
        """
        raw = (name or "").strip().lower()
        # Remove periods from initials (D.J. -> dj)
        raw = raw.replace(".", "")
        # Strip suffixes
        raw = _SUFFIX_PATTERN.sub("", raw).strip()
        # Replace non-alnum with underscore, collapse multiples
        raw = _NON_ALNUM.sub("_", raw).strip("_")
        return raw

    @staticmethod
    def player_key(player: dict) -> str:
        """Return primary identity key for a player.

        Uses ESPN numeric player_id if available, otherwise normalized name.
        """
        pid = str(player.get("player_id") or "").strip()
        if pid and pid.isdigit():
            return f"espn:{pid}"
        return RosterEnrichment.normalize_player_name(player.get("name", ""))

    def load_all_rosters(self, start_year: int, end_year: int) -> Dict[int, Dict]:
        """Load all cbbpy_rosters_{year}.json files for the year range."""
        result: Dict[int, Dict] = {}
        for year in range(start_year, end_year + 1):
            path = self.roster_dir / f"cbbpy_rosters_{year}.json"
            if not path.exists():
                continue
            try:
                with open(path, "r") as f:
                    payload = json.load(f)
                if isinstance(payload.get("teams"), list):
                    result[year] = payload
            except (json.JSONDecodeError, OSError):
                continue
        return result

    def build_player_history(
        self, all_rosters: Dict[int, Dict]
    ) -> Dict[str, List[Tuple[int, str, float]]]:
        """Build global player history: {player_key: [(year, team_id, minutes_per_game), ...]}.

        Sorted chronologically per player.
        """
        history: Dict[str, List[Tuple[int, str, float]]] = defaultdict(list)
        for year in sorted(all_rosters.keys()):
            payload = all_rosters[year]
            for team in payload.get("teams", []):
                team_id = str(team.get("team_id", "")).strip()
                if not team_id:
                    continue
                for player in team.get("players", []):
                    key = self.player_key(player)
                    if not key:
                        continue
                    mpg = float(player.get("minutes_per_game", 0))
                    history[key].append((year, team_id, mpg))
        # Sort each player's entries chronologically
        for key in history:
            history[key].sort(key=lambda entry: entry[0])
        return history

    @staticmethod
    def compute_eligibility_year(
        player_key: str,
        target_year: int,
        history: Dict[str, List[Tuple[int, str, float]]],
    ) -> int:
        """Compute eligibility year for a player in a given season.

        = (number of distinct prior years this player appeared in any roster) + 1.
        Capped at 5 (graduate student).
        """
        entries = history.get(player_key, [])
        prior_years = set(y for (y, _, _) in entries if y < target_year)
        return min(len(prior_years) + 1, 5)

    @staticmethod
    def compute_is_transfer(
        player_key: str,
        target_year: int,
        target_team: str,
        history: Dict[str, List[Tuple[int, str, float]]],
    ) -> bool:
        """Determine if player is a transfer in target_year.

        True if player appeared on a DIFFERENT team in the most recent prior year.
        False if same team or no prior year data (true freshman).
        """
        entries = history.get(player_key, [])
        prior_entries = [(y, tid) for (y, tid, _) in entries if y < target_year]
        if not prior_entries:
            return False  # true freshman / new to D1
        most_recent_year = max(y for (y, _) in prior_entries)
        teams_in_most_recent = set(tid for (y, tid) in prior_entries if y == most_recent_year)
        return target_team not in teams_in_most_recent

    def enrich_all(
        self, start_year: int = 2005, end_year: int = 2026
    ) -> Dict[str, Any]:
        """Main entry point. Loads, cross-references, writes enriched files.

        Returns summary statistics for verification.
        """
        all_rosters = self.load_all_rosters(start_year, end_year)
        if not all_rosters:
            return {
                "years_processed": 0,
                "total_players_enriched": 0,
                "total_transfers": 0,
                "eligibility_distribution": {},
            }

        history = self.build_player_history(all_rosters)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        total_players = 0
        total_transfers = 0
        global_elig: Dict[int, int] = defaultdict(int)
        years_processed = 0

        for year in sorted(all_rosters.keys()):
            payload = all_rosters[year]
            year_players = 0
            year_transfers = 0
            year_elig: Dict[int, int] = defaultdict(int)

            for team in payload.get("teams", []):
                team_id = str(team.get("team_id", "")).strip()
                if not team_id:
                    continue
                for player in team.get("players", []):
                    key = self.player_key(player)
                    if not key:
                        continue

                    elig = self.compute_eligibility_year(key, year, history)
                    is_xfer = self.compute_is_transfer(key, year, team_id, history)

                    player["eligibility_year"] = elig
                    player["is_transfer"] = is_xfer

                    year_elig[elig] += 1
                    if is_xfer:
                        year_transfers += 1
                    year_players += 1

            # Add enrichment metadata
            payload["enrichment_metadata"] = {
                "enriched_at": datetime.now(timezone.utc).isoformat(),
                "total_players": year_players,
                "players_with_prior_history": sum(
                    v for k, v in year_elig.items() if k > 1
                ),
                "transfers_detected": year_transfers,
                "eligibility_distribution": dict(sorted(year_elig.items())),
                "years_cross_referenced": sorted(all_rosters.keys()),
            }

            # Write enriched file
            out_path = self.output_dir / f"cbbpy_rosters_{year}.json"
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)

            total_players += year_players
            total_transfers += year_transfers
            for k, v in year_elig.items():
                global_elig[k] += v
            years_processed += 1

            print(
                f"[{year}] {year_players} players, "
                f"{year_transfers} transfers, "
                f"elig: {dict(sorted(year_elig.items()))}"
            )

        return {
            "years_processed": years_processed,
            "total_players_enriched": total_players,
            "total_transfers": total_transfers,
            "eligibility_distribution": dict(sorted(global_elig.items())),
        }
