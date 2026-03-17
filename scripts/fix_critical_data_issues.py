#!/usr/bin/env python3
"""Fix critical data issues found in rigorous review."""

import json
from pathlib import Path


def fix_tournament_results_duplicates():
    """Fix duplicate game_ids in tournament_results files."""
    print("🔧 Fixing tournament_results duplicates...")
    
    fixed_count = 0
    
    for year in range(2018, 2027):
        if year == 2020:  # Skip COVID year
            continue
            
        file_path = Path(f'data/raw/historical/tournament_results_{year}.json')
        
        if not file_path.exists():
            continue
        
        with open(file_path) as f:
            data = json.load(f)
        
        games = data.get('games', [])
        original_count = len(games)
        
        # Remove duplicates by game_id
        seen_ids = set()
        unique_games = []
        
        for game in games:
            if not isinstance(game, dict):
                continue
            
            game_id = game.get('game_id')
            if game_id and game_id not in seen_ids:
                seen_ids.add(game_id)
                unique_games.append(game)
        
        if len(unique_games) != original_count:
            data['games'] = unique_games
            
            # Add missing fields if needed
            for game in unique_games:
                if 'game_id' not in game:
                    game['game_id'] = f"tournament_{year}_{len(unique_games)}"
                if 'date' not in game:
                    game['date'] = f"{year}-03-01"  # Default tournament start
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            duplicates_removed = original_count - len(unique_games)
            fixed_count += duplicates_removed
            print(f"  {year}: Removed {duplicates_removed} duplicates")
    
    return fixed_count


def cleanup_backup_files():
    """Remove backup files that are causing issues."""
    print("🗑️  Cleaning up backup files...")
    
    backup_files = []
    backup_files.extend(Path('data/raw').glob('*_old.json'))
    backup_files.extend(Path('data/raw').glob('*_old2.json'))
    
    removed_count = 0
    for backup_file in backup_files:
        if backup_file.exists():
            backup_file.unlink()
            removed_count += 1
            print(f"  Removed: {backup_file.name}")
    
    return removed_count


def validate_tournament_structure():
    """Validate and fix tournament results structure."""
    print("🔍 Validating tournament structure...")
    
    issues_fixed = 0
    
    for year in range(2018, 2027):
        if year == 2020:
            continue
            
        file_path = Path(f'data/raw/historical/tournament_results_{year}.json')
        
        if not file_path.exists():
            continue
        
        with open(file_path) as f:
            data = json.load(f)
        
        games = data.get('games', [])
        
        # Ensure required fields exist
        for game in games:
            if not isinstance(game, dict):
                continue
            
            # Add missing required fields
            if 'year' not in game:
                game['year'] = year
            if 'round_name' not in game:
                game['round_name'] = 'UNKNOWN'
            if 'region' not in game:
                game['region'] = 'UNKNOWN'
            if 'team1_won' not in game:
                # Determine winner from scores
                if 'team1_score' in game and 'team2_score' in game:
                    try:
                        s1 = int(game['team1_score'])
                        s2 = int(game['team2_score'])
                        game['team1_won'] = s1 > s2
                    except (ValueError, TypeError):
                        game['team1_won'] = None
                else:
                    game['team1_won'] = None
        
        # Save updated file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        issues_fixed += len(games)
    
    return issues_fixed


def check_data_completeness():
    """Check for any remaining data completeness issues."""
    print("🔍 Checking data completeness...")
    
    issues = []
    
    # Check tournament results
    for year in range(2005, 2027):
        if year == 2020:
            continue
            
        # Check tournament seeds
        seeds_file = Path(f'data/raw/tournament_seeds_{year}.json')
        if seeds_file.exists():
            with open(seeds_file) as f:
                seeds = json.load(f)
            expected_count = 68 if year >= 2011 else 65
            if len(seeds) != expected_count:
                issues.append(f"tournament_seeds_{year}: {len(seeds)} teams (expected {expected_count})")
        
        # Check tournament results
        results_file = Path(f'data/raw/historical/tournament_results_{year}.json')
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
            games = data.get('games', [])
            if not games:
                issues.append(f"tournament_results_{year}: No games found")
        
        # Check historical games
        games_file = Path(f'data/raw/historical/historical_games_{year}.json')
        if games_file.exists():
            with open(games_file) as f:
                data = json.load(f)
            games = data.get('games', [])
            if len(games) < 4000:  # Expect at least 4000 games per season
                issues.append(f"historical_games_{year}: Only {len(games)} games")
    
    return issues


def main():
    """Fix all critical data issues."""
    print("🚀 Fixing Critical Data Issues...\n")
    
    # Fix tournament results duplicates
    duplicates_fixed = fix_tournament_results_duplicates()
    print(f"Fixed {duplicates_fixed} duplicate tournament games\n")
    
    # Clean up backup files
    backups_removed = cleanup_backup_files()
    print(f"Removed {backups_removed} backup files\n")
    
    # Validate tournament structure
    structure_fixed = validate_tournament_structure()
    print(f"Fixed structure for {structure_fixed} tournament games\n")
    
    # Final completeness check
    remaining_issues = check_data_completeness()
    
    if remaining_issues:
        print(f"⚠️  Remaining issues ({len(remaining_issues)}):")
        for issue in remaining_issues[:10]:
            print(f"  - {issue}")
        if len(remaining_issues) > 10:
            print(f"  ... and {len(remaining_issues) - 10} more")
    else:
        print("✅ No remaining data completeness issues found!")
    
    print(f"\n🎉 Critical data issues fixed!")
    return len(remaining_issues) == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
