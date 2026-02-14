"""Bracket model for March Madness tournament structure."""

from typing import List, Dict, Optional
from .team import Team
from .game import Game


class Bracket:
    """Represents the complete March Madness tournament bracket."""
    
    # Tournament structure constants
    ROUNDS = ["First Four", "Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship"]
    REGIONS = ["East", "West", "South", "Midwest"]
    
    def __init__(self, teams: List[Team]):
        """
        Initialize bracket with teams.
        
        Args:
            teams: List of all tournament teams
        """
        self.teams = {team.name: team for team in teams}
        self.games: List[Game] = []
        self.champion: Optional[Team] = None
        self.final_four: List[Team] = []
        self._validate_teams()
    
    def _validate_teams(self):
        """Validate that we have correct number of teams and seeds."""
        if len(self.teams) not in [64, 68]:  # 64 main bracket or 68 with First Four
            raise ValueError(f"Expected 64 or 68 teams, got {len(self.teams)}")
        
        # Validate each region has seeds 1-16
        for region in self.REGIONS:
            region_teams = [t for t in self.teams.values() if t.region == region]
            seeds = sorted([t.seed for t in region_teams])
            
            if len(region_teams) >= 16:  # Only validate if region is complete
                expected_seeds = list(range(1, 17))
                if seeds[:16] != expected_seeds:
                    raise ValueError(f"Region {region} missing seeds: {set(expected_seeds) - set(seeds)}")
    
    def add_game(self, game: Game):
        """Add a game to the bracket."""
        self.games.append(game)
    
    def get_games_by_round(self, round_num: int) -> List[Game]:
        """Get all games in a specific round."""
        return [g for g in self.games if g.round == round_num]
    
    def get_team(self, name: str) -> Optional[Team]:
        """Get a team by name."""
        return self.teams.get(name)
    
    def set_champion(self, team: Team):
        """Set the tournament champion."""
        self.champion = team
    
    def set_final_four(self, teams: List[Team]):
        """Set the Final Four teams."""
        if len(teams) != 4:
            raise ValueError("Final Four must have exactly 4 teams")
        self.final_four = teams
    
    def to_dict(self) -> dict:
        """Convert bracket to dictionary."""
        return {
            "teams": [team.to_dict() for team in self.teams.values()],
            "games": [game.to_dict() for game in self.games],
            "champion": self.champion.name if self.champion else None,
            "final_four": [team.name for team in self.final_four],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Bracket":
        """Create bracket from dictionary."""
        teams = [Team.from_dict(t) for t in data["teams"]]
        bracket = cls(teams)
        
        # Reconstruct games
        for game_data in data.get("games", []):
            team1 = bracket.get_team(game_data["team1"])
            team2 = bracket.get_team(game_data["team2"])
            
            if team1 and team2:
                game = Game(
                    game_id=game_data["game_id"],
                    round=game_data["round"],
                    team1=team1,
                    team2=team2,
                )
                
                if game_data.get("predicted_winner"):
                    winner = bracket.get_team(game_data["predicted_winner"])
                    if winner:
                        game.set_prediction(
                            winner,
                            game_data.get("win_probability", 0.5),
                            game_data.get("model_scores", {})
                        )
                
                bracket.add_game(game)
        
        return bracket
    
    def get_matchup_path(self, team1: Team, team2: Team) -> Optional[int]:
        """
        Get the round where two teams would meet.
        
        Returns:
            Round number (1-6) or None if teams can't meet
        """
        # Teams from same region
        if team1.region != team2.region:
            # Different regions can only meet in Final Four or Championship
            return None
        
        # Calculate based on seed positions in standard bracket
        seed_diff = abs(team1.seed - team2.seed)
        
        # Standard bracket pairings
        if seed_diff == 15:  # 1 vs 16, 8 vs 9
            return 1
        elif seed_diff == 7:  # 4 vs 5, 12 vs 13
            return 1
        elif seed_diff in [9, 6]:  # Winners from round 1
            return 2
        else:
            # Approximate for other matchups
            return 3
