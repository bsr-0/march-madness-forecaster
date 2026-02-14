<<<<<<< HEAD
# march-madness-forecaster
=======
# March Madness Forecaster

A mathematically robust prediction system for NCAA March Madness basketball tournaments. This tool uses multiple statistical models to automatically fill out tournament brackets based on team data and historical performance.

## Features

- **Multiple Prediction Models**:
  - **Elo Rating System**: Dynamic ratings that update based on game outcomes
  - **Seed-Based Baseline**: Historical win probabilities by tournament seed matchup
  - **Statistical Model**: Predictions based on team efficiency metrics and performance
  - **Ensemble Method**: Combines all models with weighted averaging for improved accuracy

- **Intelligent Upset Detection**: Probability-based predictions that don't always pick favorites
- **Configurable Parameters**: Tune model weights and upset thresholds
- **JSON Input/Output**: Easy integration with external data sources
- **Command-Line Interface**: Simple CLI for quick predictions

## Installation

### Requirements

- Python 3.8 or higher

### Setup

1. Clone or navigate to the repository:
```bash
cd march-madness-forecaster
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install the package:
```bash
pip install -e .
```

## Quick Start

### 1. Generate Sample Data

Create sample tournament data to test the system:

```bash
python -m src.main sample --output sample_tournament.json
```

### 2. Run Predictions

Generate bracket predictions using the ensemble model (recommended):

```bash
python -m src.main predict --input sample_tournament.json --output predictions.json
```

The system will:
- Load tournament teams and their statistics
- Run predictions using the selected model
- Display predicted champion, Final Four, and notable upsets
- Save complete bracket predictions to JSON

## Usage

### Command-Line Interface

#### Generate Predictions

```bash
python -m src.main predict --input TEAMS.json --output PREDICTIONS.json [OPTIONS]
```

**Options:**
- `--input, -i`: Input JSON file with tournament teams (required)
- `--output, -o`: Output JSON file for predictions (default: `bracket_predictions.json`)
- `--model, -m`: Prediction model to use (default: `ensemble`)
  - `elo`: Elo rating system only
  - `seed`: Historical seed performance only
  - `statistical`: Team statistics only
  - `ensemble`: Combine all models (recommended)
- `--upset-threshold`: Minimum probability to pick upsets, 0-1 (default: `0.4`)

**Examples:**

```bash
# Use ensemble model with default settings
python -m src.main predict -i teams.json -o my_bracket.json

# Use only Elo ratings
python -m src.main predict -i teams.json -o bracket.json --model elo

# More conservative upset picks (higher threshold)
python -m src.main predict -i teams.json --upset-threshold 0.6
```

#### Create Sample Data

```bash
python -m src.main sample --output FILENAME.json
```

Creates a sample tournament JSON file for testing.

## Input Data Format

The system expects tournament data in JSON format:

```json
{
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
    }
  ]
}
```

### Required Fields

- `name`: Team name (string)
- `seed`: Tournament seed 1-16 (integer)
- `region`: One of "East", "West", "South", "Midwest" (string)

### Optional Fields

- `elo_rating`: Elo rating (default: 1500.0)
- `stats`: Dictionary of team statistics
  - `offensive_efficiency`: Offensive rating
  - `defensive_efficiency`: Defensive rating
  - `strength_of_schedule`: Schedule difficulty
  - `recent_performance`: Recent form
  - `tempo`: Pace of play
  - `experience`: Team experience level

## Output Format

Predictions are saved as JSON with complete bracket information:

```json
{
  "teams": [...],
  "games": [
    {
      "game_id": 1,
      "round": 1,
      "team1": "Duke",
      "team2": "FGCU",
      "predicted_winner": "Duke",
      "win_probability": 0.952,
      "model_scores": {
        "elo": 0.95,
        "seed_baseline": 0.995,
        "statistical": 0.91
      }
    }
  ],
  "champion": "Duke",
  "final_four": ["Duke", "Kentucky", "Kansas", "Gonzaga"]
}
```

## Mathematical Approach

### Elo Rating System

The Elo system dynamically adjusts team ratings based on game outcomes:

- **Win Probability**: `P = 1 / (1 + 10^((R_opponent - R_team) / 400))`
- **Rating Update**: `R_new = R_old + K × (actual - expected)`
- Default K-factor: 32

### Seed-Based Baseline

Uses historical NCAA tournament data to estimate win probabilities by seed matchup. For example:
- 1 seed vs 16 seed: 99.5% win rate for 1 seed
- 5 seed vs 12 seed: 64.7% win rate for 5 seed
- 8 seed vs 9 seed: 48.8% win rate for 8 seed (nearly even)

### Statistical Model

Calculates team scores using weighted features:
- Offensive efficiency: 30%
- Defensive efficiency: 30%
- Strength of schedule: 15%
- Recent performance: 15%
- Tempo: 5%
- Experience: 5%

Converts score differences to win probabilities using a logistic function.

### Ensemble Method

Combines predictions from all models using weighted averaging:
- Each model provides a win probability
- Weights can be customized or set to equal (default)
- Final probability is the weighted sum of individual model predictions

## Testing

Run the test suite:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=src tests/
```

## Project Structure

```
march-madness-forecaster/
├── src/
│   ├── models/
│   │   ├── team.py          # Team data model
│   │   ├── game.py          # Game representation
│   │   └── bracket.py       # Bracket structure
│   ├── predictors/
│   │   ├── base.py          # Base predictor interface
│   │   ├── elo.py           # Elo rating system
│   │   ├── statistical.py   # Statistical model
│   │   ├── seed_baseline.py # Seed-based predictions
│   │   └── ensemble.py      # Ensemble predictor
│   ├── data/
│   │   └── loader.py        # Data loading/saving
│   ├── bracket_generator.py # Bracket generation
│   └── main.py              # CLI entry point
├── tests/
│   ├── test_predictors.py   # Predictor tests
│   └── test_bracket.py      # Model tests
├── requirements.txt
├── setup.py
└── README.md
```

## Example Workflow

1. **Prepare your tournament data** in JSON format with team information
2. **Run predictions** using the CLI
3. **Review the output** including champion, Final Four, and upsets
4. **Use the predictions** to fill out your bracket

```bash
# Create sample data
python -m src.main sample --output my_teams.json

# Edit my_teams.json with actual tournament data

# Generate predictions
python -m src.main predict --input my_teams.json --output my_bracket.json

# Review predictions
cat my_bracket.json
```

## Tips for Best Results

- **Use ensemble model**: Combines strengths of all models for most robust predictions
- **Provide quality data**: Better input statistics lead to better predictions
- **Tune upset threshold**: Lower values (0.3-0.4) pick more upsets, higher values (0.5+) are more conservative
- **Consider context**: The model doesn't account for injuries, coaching changes, or other real-time factors

## Future Enhancements

Potential improvements for the system:
- Historical game data integration for training
- Machine learning model optimization
- Real-time data fetching from sports APIs
- Injury and roster change adjustments
- Monte Carlo simulation for probability ranges
- Web interface for easier interaction

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## Acknowledgments

- Historical seed matchup data based on NCAA tournament history (1985-present)
- Elo rating system adapted from chess ratings
- Statistical methodology inspired by KenPom and other advanced basketball metrics

