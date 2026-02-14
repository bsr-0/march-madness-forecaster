"""Unit tests for prediction models."""

import pytest
from src.models.team import Team
from src.predictors.elo import EloPredictor
from src.predictors.seed_baseline import SeedBaselinePredictor
from src.predictors.statistical import StatisticalPredictor
from src.predictors.ensemble import EnsemblePredictor


@pytest.fixture
def sample_teams():
    """Create sample teams for testing."""
    team1 = Team(name="Duke", seed=1, region="East", elo_rating=1800)
    team1.stats = {
        "offensive_efficiency": 95,
        "defensive_efficiency": 90,
        "strength_of_schedule": 85,
        "recent_performance": 92,
    }
    
    team2 = Team(name="FGCU", seed=16, region="East", elo_rating=1400)
    team2.stats = {
        "offensive_efficiency": 60,
        "defensive_efficiency": 55,
        "strength_of_schedule": 40,
        "recent_performance": 65,
    }
    
    return team1, team2


def test_elo_predictor(sample_teams):
    """Test Elo rating predictor."""
    team1, team2 = sample_teams
    predictor = EloPredictor()
    
    winner, prob = predictor.predict(team1, team2)
    
    assert winner == team1  # Higher Elo should win
    assert 0.5 < prob <= 1.0  # Probability should favor higher Elo
    assert prob > 0.85  # Should be high probability for 400 Elo difference


def test_seed_baseline_predictor(sample_teams):
    """Test seed-based predictor."""
    team1, team2 = sample_teams
    predictor = SeedBaselinePredictor()
    
    winner, prob = predictor.predict(team1, team2)
    
    assert winner == team1  # 1 seed should beat 16
    assert prob > 0.99  # Historical 1 vs 16 is very high


def test_statistical_predictor(sample_teams):
    """Test statistical features predictor."""
    team1, team2 = sample_teams
    predictor = StatisticalPredictor()
    
    winner, prob = predictor.predict(team1, team2)
    
    assert winner == team1  # Better stats should win
    assert 0.5 < prob <= 1.0  # Probability should favor better stats


def test_ensemble_predictor(sample_teams):
    """Test ensemble predictor."""
    team1, team2 = sample_teams
    
    predictors = [
        EloPredictor(),
        SeedBaselinePredictor(),
        StatisticalPredictor()
    ]
    
    ensemble = EnsemblePredictor(predictors)
    winner, prob = ensemble.predict(team1, team2)
    
    assert winner == team1  # All models agree
    assert 0.5 < prob < 1.0
    
    # Check model scores available
    scores = ensemble.get_model_scores()
    assert len(scores) == 3


def test_elo_rating_updates(sample_teams):
    """Test Elo rating updates."""
    team1, team2 = sample_teams
    initial_rating1 = team1.elo_rating
    initial_rating2 = team2.elo_rating
    
    predictor = EloPredictor()
    predictor.update_ratings(team1, team2, team1_won=True)
    
    # Winner should gain rating, loser should lose rating
    assert team1.elo_rating > initial_rating1
    assert team2.elo_rating < initial_rating2


def test_equal_teams():
    """Test prediction with equal teams."""
    team1 = Team(name="TeamA", seed=5, region="East", elo_rating=1600)
    team2 = Team(name="TeamB", seed=5, region="East", elo_rating=1600)
    
    predictor = EloPredictor()
    winner, prob = predictor.predict(team1, team2)
    
    # Equal teams should have ~50% probability
    assert 0.49 < prob < 0.51


def test_ensemble_weights():
    """Test ensemble with custom weights."""
    team1 = Team(name="Duke", seed=1, region="East", elo_rating=1800)
    team2 = Team(name="FGCU", seed=16, region="East", elo_rating=1400)
    
    predictors = [
        EloPredictor(),
        SeedBaselinePredictor(),
    ]
    
    # Weight heavily toward Elo
    weights = {"elo": 0.9, "seed_baseline": 0.1}
    ensemble = EnsemblePredictor(predictors, weights)
    
    winner, prob = ensemble.predict(team1, team2)
    assert winner == team1
