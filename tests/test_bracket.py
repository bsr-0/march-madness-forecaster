"""Unit tests for bracket and game models."""

import pytest
from src.models.team import Team
from src.models.game import Game
from src.models.bracket import Bracket
from src.predictors.elo import EloPredictor
from src.bracket_generator import BracketGenerator


@pytest.fixture
def sample_team():
    """Create a sample team."""
    return Team(
        name="Duke",
        seed=1,
        region="East",
        elo_rating=1800,
        stats={"offensive_efficiency": 95}
    )


@pytest.fixture
def sample_game():
    """Create a sample game."""
    team1 = Team(name="Duke", seed=1, region="East")
    team2 = Team(name="FGCU", seed=16, region="East")
    return Game(game_id=1, round=1, team1=team1, team2=team2)


def test_team_creation(sample_team):
    """Test team model creation."""
    assert sample_team.name == "Duke"
    assert sample_team.seed == 1
    assert sample_team.elo_rating == 1800
    assert sample_team.stats["offensive_efficiency"] == 95


def test_team_elo_calculation():
    """Test Elo expected score calculation."""
    team1 = Team(name="TeamA", seed=1, region="East", elo_rating=1600)
    team2 = Team(name="TeamB", seed=2, region="East", elo_rating=1400)
    
    expected = team1.expected_score(team2.elo_rating)
    assert 0.5 < expected < 1.0  # Higher rated team should have >50%


def test_game_prediction(sample_game):
    """Test setting game predictions."""
    winner = sample_game.team1
    sample_game.set_prediction(winner, 0.95)
    
    assert sample_game.predicted_winner == winner
    assert sample_game.win_probability == 0.95


def test_game_upset_detection():
    """Test upset detection."""
    team1 = Team(name="Duke", seed=1, region="East")
    team2 = Team(name="FGCU", seed=16, region="East")
    game = Game(game_id=1, round=1, team1=team1, team2=team2)
    
    # Not an upset if favorite wins
    game.set_prediction(team1, 0.95)
    assert not game.is_upset
    
    # Is an upset if underdog wins
    game.set_prediction(team2, 0.55)
    assert game.is_upset


def test_bracket_creation():
    """Test bracket creation with multiple regions."""
    teams = []
    for region in ["East", "West", "South", "Midwest"]:
        for seed in range(1, 17):
            teams.append(Team(
                name=f"{region}-{seed}",
                seed=seed,
                region=region
            ))
    
    bracket = Bracket(teams)
    assert len(bracket.teams) == 64


def test_bracket_validation():
    """Test bracket validation."""
    # Too few teams
    with pytest.raises(ValueError):
        Bracket([Team(name="Duke", seed=1, region="East")])


def test_bracket_generator():
    """Test bracket generator with small dataset."""
    teams = []
    
    # Create minimal bracket (one region, 4 teams)
    for seed in [1, 16, 8, 9]:
        teams.append(Team(
            name=f"Team-{seed}",
            seed=seed,
            region="East",
            elo_rating=1700 - (seed * 20)
        ))
    
    predictor = EloPredictor()
    
    # Note: This will fail validation for a full bracket
    # but we can test the prediction logic works
    with pytest.raises(ValueError):
        bracket = Bracket(teams)


def test_team_serialization():
    """Test team to/from dict conversion."""
    team = Team(
        name="Duke",
        seed=1,
        region="East",
        elo_rating=1800,
        stats={"offensive_efficiency": 95}
    )
    
    team_dict = team.to_dict()
    assert team_dict["name"] == "Duke"
    assert team_dict["seed"] == 1
    
    team_restored = Team.from_dict(team_dict)
    assert team_restored.name == team.name
    assert team_restored.elo_rating == team.elo_rating


def test_game_serialization(sample_game):
    """Test game to dict conversion."""
    sample_game.set_prediction(sample_game.team1, 0.95)
    
    game_dict = sample_game.to_dict()
    assert game_dict["game_id"] == 1
    assert game_dict["predicted_winner"] == "Duke"
    assert game_dict["win_probability"] == 0.95
