#!/usr/bin/env python3
"""Transform game schema from team_id/team_name to team1_id/team1_name format."""

import json
import os
import shutil
from pathlib import Path


def transform_game_schema(input_file: Path, output_file: Path) -> int:
    """Transform game schema from team_id/team_name to team1_id/team1_name format.
    
    Args:
        input_file: Path to original game file
        output_file: Path to write transformed file
        
    Returns:
        Number of games transformed
    """
    with open(input_file) as f:
        data = json.load(f)
    
    games = data.get("games", data) if isinstance(data, dict) else data
    
    # Track seen game_ids to avoid duplicates (raw/ has both sides)
    seen_game_ids = set()
    transformed_games = []
    
    for game in games:
        if not isinstance(game, dict):
            continue
            
        game_id = game.get("game_id")
        if not game_id or game_id in seen_game_ids:
            continue
        seen_game_ids.add(game_id)
            
        # Transform field names - match historical/ schema exactly
        transformed = {
            "game_id": game_id,
            "season": game.get("season"),
            "date": game.get("date"),
            "team1_id": game.get("team_id"),
            "team1_name": game.get("team_name"),
            "team2_id": game.get("opponent_id"),
            "team2_name": game.get("opponent_name"),
            "team1_score": game.get("team_score"),
            "team2_score": game.get("opponent_score"),
        }
        
        transformed_games.append(transformed)
    
    # Save with same structure as historical/ files
    output_data = {
        "season": data.get("season") if isinstance(data, dict) else None,
        "provider": data.get("provider") if isinstance(data, dict) else None,
        "games": transformed_games,
        "team_games": data.get("team_games", []) if isinstance(data, dict) else [],
        "failed_game_ids": data.get("failed_game_ids", []) if isinstance(data, dict) else []
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    return len(transformed_games)


def main():
    """Transform raw game files to standardized schema."""
    raw_dir = Path('data/raw')
    backup_dir = Path('data/raw/backups')
    backup_dir.mkdir(exist_ok=True)
    
    for year in [2025, 2026]:
        input_file = raw_dir / f'historical_games_{year}.json'
        backup_file = backup_dir / f'historical_games_{year}_original.json'
        output_file = raw_dir / f'historical_games_{year}_transformed.json'
        
        print(f'Processing {year}...')
        
        if not input_file.exists():
            print(f'  File {input_file} not found, skipping')
            continue
        
        # Backup original
        shutil.copy2(input_file, backup_file)
        print(f'  Backed up to {backup_file}')
        
        # Transform
        game_count = transform_game_schema(input_file, output_file)
        print(f'  Transformed {game_count} games to {output_file}')
        
        # Verify transformation
        with open(output_file) as f:
            data = json.load(f)
        games = data.get("games", [])
        if games:
            sample = games[0]
            print(f'  Sample transformed game keys: {list(sample.keys())}')
            
            # Check for TBD games
            tbd_count = sum(1 for g in games if g.get("team1_name") == "TBD")
            if tbd_count > 0:
                print(f'  WARNING: {tbd_count} TBD games found')
        else:
            print(f'  WARNING: No games found in output')
    
    print('\nSchema transformation complete!')


if __name__ == "__main__":
    main()
