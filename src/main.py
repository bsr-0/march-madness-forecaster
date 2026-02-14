"""Main CLI interface for March Madness forecaster."""

import argparse
import sys
from pathlib import Path

from .data.loader import DataLoader
from .bracket_generator import BracketGenerator
from .predictors.elo import EloPredictor
from .predictors.seed_baseline import SeedBaselinePredictor
from .predictors.statistical import StatisticalPredictor
from .predictors.ensemble import EnsemblePredictor


def create_predictor(model_type: str, ensemble_weights: dict = None):
    """
    Create predictor based on model type.
    
    Args:
        model_type: Type of model ('elo', 'seed', 'statistical', 'ensemble')
        ensemble_weights: Weights for ensemble model
        
    Returns:
        Predictor instance
    """
    if model_type == "elo":
        return EloPredictor()
    elif model_type == "seed":
        return SeedBaselinePredictor()
    elif model_type == "statistical":
        return StatisticalPredictor()
    elif model_type == "ensemble":
        predictors = [
            EloPredictor(),
            SeedBaselinePredictor(),
            StatisticalPredictor()
        ]
        return EnsemblePredictor(predictors, ensemble_weights)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def predict_bracket(args):
    """Generate bracket predictions."""
    print(f"Loading tournament data from {args.input}...")
    
    try:
        teams = DataLoader.load_teams_from_json(args.input)
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1
    
    print(f"Loaded {len(teams)} teams")
    
    # Normalize statistics if using statistical model
    if args.model in ["statistical", "ensemble"]:
        print("Normalizing team statistics...")
        StatisticalPredictor.normalize_stats(teams)
    
    print(f"Creating {args.model} predictor...")
    predictor = create_predictor(args.model)
    
    print(f"Generating bracket (upset threshold: {args.upset_threshold})...")
    generator = BracketGenerator(predictor, upset_threshold=args.upset_threshold)
    bracket = generator.generate_bracket(teams)
    
    # Display results
    print(f"\n{'='*60}")
    print(f"BRACKET PREDICTIONS - {args.model.upper()} MODEL")
    print(f"{'='*60}\n")
    
    if bracket.champion:
        print(f"🏆 PREDICTED CHAMPION: {bracket.champion.name} (Seed {bracket.champion.seed})")
    
    if bracket.final_four:
        print(f"\n📊 FINAL FOUR:")
        for team in bracket.final_four:
            print(f"   - {team.name} (Seed {team.seed}, {team.region})")
    
    # Count upsets
    upsets = [g for g in bracket.games if g.is_upset]
    print(f"\n🎯 PREDICTED UPSETS: {len(upsets)}")
    
    # Show some notable upsets
    notable_upsets = [g for g in upsets if abs(g.team1.seed - g.team2.seed) >= 3][:5]
    if notable_upsets:
        print("\n   Notable upsets:")
        for game in notable_upsets:
            loser = game.team1 if game.team1 != game.predicted_winner else game.team2
            print(f"   - Round {game.round}: {game.predicted_winner.name} ({game.predicted_winner.seed}) over {loser.name} ({loser.seed}) - {game.win_probability:.1%}")
    
    # Save to file
    print(f"\nSaving predictions to {args.output}...")
    DataLoader.save_bracket_to_json(bracket, args.output)
    print("✓ Done!")
    
    return 0


def create_sample(args):
    """Create sample data file."""
    print(f"Creating sample data at {args.output}...")
    DataLoader.create_sample_data(args.output)
    print("✓ Sample data created!")
    print(f"\nYou can now run predictions with:")
    print(f"  python -m src.main predict --input {args.output} --output predictions.json")
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="March Madness Bracket Forecaster - Mathematically robust tournament predictions"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Generate bracket predictions")
    predict_parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSON file with tournament teams and data"
    )
    predict_parser.add_argument(
        "--output", "-o",
        default="bracket_predictions.json",
        help="Output JSON file for predictions (default: bracket_predictions.json)"
    )
    predict_parser.add_argument(
        "--model", "-m",
        choices=["elo", "seed", "statistical", "ensemble"],
        default="ensemble",
        help="Prediction model to use (default: ensemble)"
    )
    predict_parser.add_argument(
        "--upset-threshold",
        type=float,
        default=0.4,
        help="Minimum probability to pick upsets (0-1, default: 0.4)"
    )
    
    # Sample command
    sample_parser = subparsers.add_parser("sample", help="Create sample tournament data")
    sample_parser.add_argument(
        "--output", "-o",
        default="sample_tournament.json",
        help="Output file for sample data (default: sample_tournament.json)"
    )
    
    args = parser.parse_args()
    
    if args.command == "predict":
        return predict_bracket(args)
    elif args.command == "sample":
        return create_sample(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
