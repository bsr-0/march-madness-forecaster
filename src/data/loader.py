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
    
    @staticmethod
    def create_sample_data(output_path: str) -> None:
        """
        Create sample tournament data for testing.
        
        Args:
            output_path: Path to save sample data
        """
        # Create sample teams for one region
        sample_data = {
            "teams": [
                {
                    "name": "Duke",
                    "seed": 1,
                    "region": "East",
                    "elo_rating": 1800,
                    "stats": {
                        "offensive_efficiency": 95,
                        "defensive_efficiency": 90,
                        "strength_of_schedule": 85,
                        "recent_performance": 92,
                        "tempo": 70,
                        "experience": 80
                    }
                },
                {
                    "name": "FGCU",
                    "seed": 16,
                    "region": "East",
                    "elo_rating": 1400,
                    "stats": {
                        "offensive_efficiency": 60,
                        "defensive_efficiency": 55,
                        "strength_of_schedule": 40,
                        "recent_performance": 65,
                        "tempo": 75,
                        "experience": 45
                    }
                },
                {
                    "name": "Kentucky",
                    "seed": 2,
                    "region": "East",
                    "elo_rating": 1750,
                    "stats": {
                        "offensive_efficiency": 90,
                        "defensive_efficiency": 88,
                        "strength_of_schedule": 80,
                        "recent_performance": 85,
                        "tempo": 68,
                        "experience": 75
                    }
                },
                {
                    "name": "Oakland",
                    "seed": 15,
                    "region": "East",
                    "elo_rating": 1420,
                    "stats": {
                        "offensive_efficiency": 62,
                        "defensive_efficiency": 58,
                        "strength_of_schedule": 42,
                        "recent_performance": 68,
                        "tempo": 72,
                        "experience": 48
                    }
                }
            ],
            "matchups": [
                {
                    "game_id": 1,
                    "round": 1,
                    "team1": "Duke",
                    "team2": "FGCU"
                },
                {
                    "game_id": 2,
                    "round": 1,
                    "team1": "Kentucky",
                    "team2": "Oakland"
                }
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(sample_data, f, indent=2)
