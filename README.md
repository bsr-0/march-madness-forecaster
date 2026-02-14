
# March Madness Forecaster

**State-of-the-Art NCAA Tournament Prediction System**

A professional-grade prediction system using Graph Neural Networks (GNN), Transformers, Monte Carlo simulation, and game-theoretic bracket optimization. Designed to compete with the top 5% of sophisticated bracket pools.

## Architecture Overview

This system implements cutting-edge ML techniques:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PREDICTION PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│  Data Layer          │  ML Models           │  Optimization     │
│  ─────────────       │  ─────────           │  ────────────     │
│  • KenPom Scraper    │  • GNN (Schedule)    │  • Monte Carlo    │
│  • Torvik Scraper    │  • Transformer       │  • Calibration    │
│  • ESPN Picks        │  • LightGBM          │  • Leverage Calc  │
│  • Player RAPM/WARP  │  • CFA Ensemble      │  • Pareto Optim   │
│  • xP Shot Quality   │  • Elo/Seed Base     │  • Pool Analysis  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

### Phase 1: High-Resolution Data Engineering
- **Possession-level xP** (Expected Points) calculation
- **Player-level RAPM/WARP** tracking with transfer portal impact
- **Lead volatility & entropy** metrics for upset detection
- **Four Factors** analysis (eFG%, TO%, ORB%, FTR)

### Phase 2: GNN-Transformer Hybrid Architecture
- **Graph Convolutional Network (GCN)** for schedule-based strength propagation
- **Multi-hop SOS analysis** (opponent's opponent quality)
- **Temporal Transformer** for detecting "breakout windows"
- **Attention mechanisms** for game importance weighting

### Phase 3: Uncertainty Quantification
- **50,000+ Monte Carlo simulations** with injury noise injection
- **Brier Score optimization** for probability calibration
- **Isotonic regression** post-processing
- **Reliability diagrams** for model validation

### Phase 4: Game Theory Optimization
- **Public consensus scraping** from ESPN/Yahoo/CBS
- **Leverage ratio calculation** (Win Prob / Pick %)
- **Pareto-optimal bracket generation** (chalk to contrarian)
- **Pool size-specific strategy** recommendations

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

#### Run Full SOTA Pipeline (2026 Rubric)

```bash
python -m src.main sota --output sota_report.json --simulations 50000 --pool-size 150
```

With real data and custom scoring:

```bash
python -m src.main sota \
  --input teams_2026.json \
  --kenpom kenpom_2026.json \
  --torvik torvik_2026.json \
  --shotquality-teams shotquality_teams_2026.json \
  --public-picks public_picks_2026.json \
  --scoring-rules pool_scoring.json \
  --simulations 50000 \
  --output selection_sunday_report.json
```

This command executes the full rubric pipeline:
- Possession-level xP + player-level RAPM feature engineering
- Schedule adjacency + GCN SOS refinement
- Temporal transformer embeddings
- CFA model fusion + isotonic calibration
- Injury-noise Monte Carlo bracket simulation
- EV-max bracket selection with leverage and Pareto risk profiles

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
