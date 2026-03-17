#!/usr/bin/env python3
"""Generate placeholder tournament_results files for missing years."""

import json
from pathlib import Path


def create_placeholder_results(year: int, output_file: Path) -> None:
    """Create a placeholder tournament_results file for a missing year.
    
    Args:
        year: Tournament year
        output_file: Path to write results file
    """
    # Create empty tournament results structure
    results_data = {
        "year": year,
        "games": [],
        "note": f"Placeholder file - actual tournament results for {year} not available"
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)


def extract_2026_tournament_results(games_file: Path, output_file: Path) -> int:
    """Extract 2026 tournament games from historical games file.
    
    Args:
        games_file: Path to historical games file
        output_file: Path to write tournament results file
        
    Returns:
        Number of tournament games found
    """
    with open(games_file) as f:
        data = json.load(f)
    
    all_games = data.get("games", [])
    
    # Filter for March/April 2026 games (likely tournament games)
    tournament_games = []
    for game in all_games:
        date = game.get("date", "")
        if "2026-03" in date or "2026-04" in date:
            # Skip TBD games
            if game.get("team1_name") == "TBD" or game.get("team2_name") == "TBD":
                continue
            
            # Convert to tournament results format
            result_game = {
                "year": 2026,
                "round_name": "UNKNOWN",  # Would need bracket data to determine
                "region": "UNKNOWN",      # Would need bracket data to determine
                "team1_id": game["team1_id"],
                "team1_seed": None,       # Would need seeds data
                "team1_score": game["team1_score"],
                "team2_id": game["team2_id"],
                "team2_seed": None,       # Would need seeds data
                "team2_score": game["team2_score"],
                "team1_won": game["team1_score"] > game["team2_score"],
                "date": game["date"]
            }
            tournament_games.append(result_game)
    
    results_data = {
        "year": 2026,
        "games": tournament_games,
        "note": "Extracted from historical games - round/region/seed data missing"
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    return len(tournament_games)


def main():
    """Generate missing tournament results files."""
    hist_dir = Path('data/raw/historical')
    
    # Create placeholders for 2005-2017
    missing_years = list(range(2005, 2018))
    
    for year in missing_years:
        output_file = hist_dir / f'tournament_results_{year}.json'
        if not output_file.exists():
            create_placeholder_results(year, output_file)
            print(f'Created placeholder: {output_file}')
        else:
            print(f'Already exists: {output_file}')
    
    # Extract 2026 from historical games
    games_file = hist_dir / 'historical_games_2026.json'
    results_file = hist_dir / 'tournament_results_2026.json'
    
    if games_file.exists() and not results_file.exists():
        game_count = extract_2026_tournament_results(games_file, results_file)
        print(f'Extracted {game_count} tournament games to {results_file}')
        
        if game_count > 0:
            # Show sample
            with open(results_file) as f:
                data = json.load(f)
            sample_game = data["games"][0] if data["games"] else None
            if sample_game:
                print(f'  Sample game: {json.dumps(sample_game, indent=2)}')
    else:
        print(f'2026 results already exist or games file missing')


if __name__ == "__main__":
    main()
