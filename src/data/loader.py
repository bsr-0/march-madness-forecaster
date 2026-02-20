"""Data loader for tournament data."""

import json
from typing import List, Tuple
from ..models.team import Team
from ..models.bracket import Bracket


class DataLoader:
    """Loads tournament data from JSON files."""
    
    @staticmethod
    def load_teams_from_json(file_path: str) -> List[Team]:
        """
        Load teams from a JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List of Team objects
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        teams = []
        for team_data in data.get('teams', []):
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
    
