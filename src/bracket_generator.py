"""Bracket generator for March Madness predictions."""

from typing import Dict, List, Optional
from .models.team import Team
from .models.game import Game
from .models.bracket import Bracket
from .predictors.base import BasePredictor


class BracketGenerator:
    """Generates tournament bracket predictions."""
    
    # Standard tournament bracket structure (seed matchups by round)
    FIRST_ROUND_MATCHUPS = [
        (1, 16), (8, 9), (5, 12), (4, 13),
        (6, 11), (3, 14), (7, 10), (2, 15)
    ]
    
    def __init__(self, predictor: BasePredictor, upset_threshold: float = 0.4):
        """
        Initialize bracket generator.
        
        Args:
            predictor: Prediction model to use
            upset_threshold: Minimum probability to pick upset (0 to 1)
        """
        self.predictor = predictor
        self.upset_threshold = upset_threshold
    
    def generate_bracket(self, teams: List[Team]) -> Bracket:
        """
        Generate complete bracket predictions.
        
        Args:
            teams: List of all tournament teams
            
        Returns:
            Bracket with predictions filled in
        """
        bracket = Bracket(teams)
        
        # Generate first round games for each region
        game_id = 1
        for region in Bracket.REGIONS:
            region_teams = {t.seed: t for t in teams if t.region == region}
            
            for higher_seed, lower_seed in self.FIRST_ROUND_MATCHUPS:
                if higher_seed in region_teams and lower_seed in region_teams:
                    team1 = region_teams[higher_seed]
                    team2 = region_teams[lower_seed]
                    
                    game = Game(game_id=game_id, round=1, team1=team1, team2=team2)
                    self._predict_game(game)
                    bracket.add_game(game)
                    game_id += 1
        
        # Simulate remaining rounds
        current_round = 1
        max_rounds = 6  # Through championship
        
        while current_round < max_rounds:
            winners = self._get_round_winners(bracket, current_round)
            if len(winners) < 2:
                break
            
            # Generate next round matchups
            next_round = current_round + 1
            next_round_games = self._generate_next_round(winners, next_round, game_id, bracket)
            
            for game in next_round_games:
                self._predict_game(game)
                bracket.add_game(game)
                game_id += 1
            
            current_round = next_round
        
        # Set Final Four and Champion
        final_four = self._get_round_winners(bracket, 4)  # Elite 8 winners
        bracket.set_final_four(final_four)
        
        championship_winners = self._get_round_winners(bracket, 6)
        if championship_winners:
            bracket.set_champion(championship_winners[0])
        
        return bracket
    
    def fill_existing_bracket(self, bracket: Bracket) -> Bracket:
        """
        Fill predictions for an existing bracket structure.
        
        Args:
            bracket: Bracket with games already defined
            
        Returns:
            Bracket with predictions added
        """
        for game in bracket.games:
            if not game.predicted_winner:
                self._predict_game(game)
        
        return bracket
    
    def _predict_game(self, game: Game) -> None:
        """
        Predict the outcome of a single game.
        
        Args:
            game: Game to predict
        """
        winner, probability = self.predictor.predict(game.team1, game.team2)
        
        # Apply upset threshold - don't pick upsets unless probability is strong enough
        if winner.seed > min(game.team1.seed, game.team2.seed):
            # This is an upset prediction
            if probability < self.upset_threshold + 0.5:
                # Not confident enough, pick the favorite instead
                winner = game.team1 if game.team1.seed < game.team2.seed else game.team2
                probability = 0.5 + (0.5 - probability)
        
        # Get model scores if using ensemble
        model_scores = {}
        if hasattr(self.predictor, 'get_model_scores'):
            model_scores = self.predictor.get_model_scores()
        
        game.set_prediction(winner, probability, model_scores)
    
    def _get_round_winners(self, bracket: Bracket, round_num: int) -> List[Team]:
        """
        Get all winners from a specific round.
        
        Args:
            bracket: Tournament bracket
            round_num: Round number
            
        Returns:
            List of winning teams
        """
        games = bracket.get_games_by_round(round_num)
        winners = [g.predicted_winner for g in games if g.predicted_winner]
        return winners
    
    def _generate_next_round(
        self, 
        winners: List[Team], 
        round_num: int, 
        starting_game_id: int,
        bracket: Bracket
    ) -> List[Game]:
        """
        Generate matchups for the next round.
        
        Args:
            winners: Winners from previous round
            round_num: Next round number
            starting_game_id: Starting game ID
            bracket: Tournament bracket
            
        Returns:
            List of games for next round
        """
        games = []
        
        # Group winners by region for regional rounds
        if round_num <= 4:  # Regional rounds
            region_winners = {}
            for team in winners:
                if team.region not in region_winners:
                    region_winners[team.region] = []
                region_winners[team.region].append(team)
            
            game_id = starting_game_id
            for region, teams in region_winners.items():
                # Pair teams within region
                for i in range(0, len(teams), 2):
                    if i + 1 < len(teams):
                        game = Game(
                            game_id=game_id,
                            round=round_num,
                            team1=teams[i],
                            team2=teams[i + 1]
                        )
                        games.append(game)
                        game_id += 1
        else:
            # Final Four and Championship - pair sequentially
            game_id = starting_game_id
            for i in range(0, len(winners), 2):
                if i + 1 < len(winners):
                    game = Game(
                        game_id=game_id,
                        round=round_num,
                        team1=winners[i],
                        team2=winners[i + 1]
                    )
                    games.append(game)
                    game_id += 1
        
        return games
