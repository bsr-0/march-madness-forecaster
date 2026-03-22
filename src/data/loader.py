"""Data loader for tournament data."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..models.team import Team
from ..models.bracket import Bracket

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads tournament data from JSON files."""
    
    @staticmethod
    def load_teams_from_json(file_path: str) -> List[Team]:
        """
        Load teams from a JSON file.

        Handles First Four play-in pairs: when multiple teams share the
        same (seed, region) and are marked ``first_four: true``, only the
        highest-rated team is kept, reducing 68 tournament teams to 64.

        Args:
            file_path: Path to JSON file

        Returns:
            List of Team objects (64 for a standard bracket)
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        raw_teams = data.get('teams', [])

        # Collapse First Four play-in pairs: for each (seed, region) slot
        # with multiple first_four entries, keep the highest-rated team.
        play_in = [t for t in raw_teams if t.get("first_four", False)]
        non_play_in = [t for t in raw_teams if not t.get("first_four", False)]

        if play_in:
            seen_slots: Dict[Tuple[int, str], dict] = {}
            for t in play_in:
                slot = (t["seed"], t["region"])
                prev = seen_slots.get(slot)
                if prev is None or t.get("rating", 0) > prev.get("rating", 0):
                    seen_slots[slot] = t
            resolved = non_play_in + list(seen_slots.values())
            n_removed = len(raw_teams) - len(resolved)
            if n_removed:
                logger.info(
                    "First Four resolution: collapsed %d play-in teams "
                    "into %d slots (%d teams removed)",
                    len(play_in), len(seen_slots), n_removed,
                )
        else:
            resolved = raw_teams

        teams = []
        for team_data in resolved:
            team = Team.from_dict(team_data)
            teams.append(team)

        return teams
    
    @staticmethod
    def load_bracket_from_json(file_path: str) -> Bracket:
        """
        Load complete bracket from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Bracket object
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return Bracket.from_dict(data)
    
    @staticmethod
    def save_bracket_to_json(bracket: Bracket, file_path: str) -> None:
        """
        Save bracket to JSON file.
        
        Args:
            bracket: Bracket to save
            file_path: Output file path
        """
        with open(file_path, 'w') as f:
            json.dump(bracket.to_dict(), f, indent=2)
    
    @staticmethod
    def load_matchups(file_path: str) -> List[Tuple[str, str, int, int]]:
        """
        Load matchups from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List of (team1_name, team2_name, round, game_id) tuples
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        matchups = []
        for matchup in data.get('matchups', []):
            matchups.append((
                matchup['team1'],
                matchup['team2'],
                matchup['round'],
                matchup['game_id']
            ))
        
        return matchups

    @staticmethod
    def hash_file(file_path: str) -> str:
        """Compute SHA-256 hash of a file for reproducibility tracking.

        Args:
            file_path: Path to file to hash.

        Returns:
            Hex-encoded SHA-256 digest.
        """
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def hash_dataset(
        file_paths: List[str],
        labels: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """Compute SHA-256 hashes for a set of training data files.

        Args:
            file_paths: Paths to data files.
            labels: Optional human-readable labels for each file.
                If not provided, file basenames are used.

        Returns:
            Dict mapping label → SHA-256 hex digest.  Also includes a
            ``combined`` key with a hash of all individual hashes
            (order-independent via sorting).
        """
        hashes: Dict[str, str] = {}
        for i, fp in enumerate(file_paths):
            label = labels[i] if labels and i < len(labels) else Path(fp).name
            try:
                hashes[label] = DataLoader.hash_file(fp)
            except FileNotFoundError:
                logger.warning("Dataset file not found for hashing: %s", fp)
                hashes[label] = "MISSING"

        # Combined hash (order-independent)
        combined = hashlib.sha256()
        for key in sorted(hashes):
            combined.update(hashes[key].encode())
        hashes["combined"] = combined.hexdigest()
        return hashes

