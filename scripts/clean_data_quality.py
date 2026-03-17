#!/usr/bin/env python3
"""Clean up data quality issues: TBD games, failed IDs, t_rank zeros."""

import json
from pathlib import Path
from collections import defaultdict


def remove_tbd_games():
    """Remove TBD placeholder games from game files."""
    raw_dir = Path('data/raw')
    cleaned_count = 0
    
    for year in [2025, 2026]:
        # Check both raw/ and raw/historical/ versions
        for file_path in [raw_dir / f'historical_games_{year}.json',
                         raw_dir / 'historical' / f'historical_games_{year}.json']:
            if not file_path.exists():
                continue
            
            with open(file_path) as f:
                data = json.load(f)
            
            games = data.get('games', [])
            original_count = len(games)
            
            # Filter out TBD games
            clean_games = [
                game for game in games
                if game.get('team1_name') != 'TBD' and game.get('team2_name') != 'TBD'
            ]
            
            if len(clean_games) != original_count:
                # Update the file
                data['games'] = clean_games
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                removed = original_count - len(clean_games)
                cleaned_count += removed
                print(f'Removed {removed} TBD games from {file_path}')
    
    return cleaned_count


def investigate_failed_game_ids():
    """Investigate failed game IDs in 2026 data."""
    file_path = Path('data/raw/historical/historical_games_2026.json')
    
    if not file_path.exists():
        print(f'File {file_path} not found')
        return
    
    with open(file_path) as f:
        data = json.load(f)
    
    failed_ids = data.get('failed_game_ids', [])
    games = data.get('games', [])
    
    print(f'Found {len(failed_ids)} failed game IDs in 2026')
    
    # Check if any failed IDs appear in the games list
    game_ids = set(game.get('game_id') for game in games)
    overlap = set(failed_ids) & game_ids
    
    if overlap:
        print(f'⚠️  {len(overlap)} failed IDs still appear in games list')
        print(f'  Sample: {list(overlap)[:5]}')
    else:
        print('✅ No failed IDs appear in games list (good)')
    
    # Try to find patterns in failed IDs
    if failed_ids:
        print(f'Sample failed IDs: {failed_ids[:10]}')
        
        # Check for date patterns
        date_prefixes = defaultdict(int)
        for game_id in failed_ids[:50]:
            if len(game_id) >= 8:
                date_prefix = game_id[:8]
                date_prefixes[date_prefix] += 1
        
        if date_prefixes:
            print('Date patterns in failed IDs:')
            for prefix, count in sorted(date_prefixes.items()):
                print(f'  {prefix}: {count} IDs')
    
    return failed_ids


def investigate_trank_zeros():
    """Investigate t_rank zeros in advanced_metrics_2026.json."""
    file_path = Path('data/raw/advanced_metrics_2026.json')
    
    if not file_path.exists():
        print(f'File {file_path} not found')
        return
    
    with open(file_path) as f:
        data = json.load(f)
    
    teams = data if isinstance(data, list) else data.get('teams', [])
    
    # Check t_rank values
    trank_values = [team.get('t_rank') for team in teams if isinstance(team, dict)]
    
    print(f't_rank analysis for {len(teams)} teams:')
    print(f'  Zero values: {sum(1 for v in trank_values if v == 0)}')
    print(f'  Null values: {sum(1 for v in trank_values if v is None)}')
    print(f'  Non-zero values: {sum(1 for v in trank_values if v not in [0, None])}')
    
    # If all zeros, check if we can compute from other rankings
    if all(v in [0, None] for v in trank_values):
        print('All t_rank values are zero/null')
        
        # Check for other ranking fields
        sample_team = teams[0] if teams else {}
        ranking_fields = [k for k in sample_team.keys() if 'rank' in k.lower() or 'rating' in k.lower()]
        print(f'Other ranking fields found: {ranking_fields}')
        
        # Show sample values for other fields
        for field in ranking_fields[:3]:
            values = [team.get(field) for team in teams[:10] if isinstance(team, dict)]
            print(f'  {field} sample: {values[:5]}')
    
    return teams


def generate_quality_report():
    """Generate a comprehensive data quality report."""
    report = {
        'generated_at': str(Path.cwd()),
        'cleaning_performed': {},
        'issues_found': {},
        'recommendations': []
    }
    
    # Remove TBD games
    tbd_removed = remove_tbd_games()
    report['cleaning_performed']['tbd_games_removed'] = tbd_removed
    
    # Investigate failed IDs
    failed_ids = investigate_failed_game_ids()
    report['issues_found']['failed_game_ids_2026'] = len(failed_ids) if failed_ids else 0
    
    # Investigate t_rank
    teams = investigate_trank_zeros()
    if teams:
        trank_zeros = sum(1 for team in teams if isinstance(team, dict) and team.get('t_rank') == 0)
        report['issues_found']['trank_zeros_2026'] = trank_zeros
    
    # Add recommendations
    if failed_ids:
        report['recommendations'].append({
            'issue': 'Failed game IDs',
            'action': 'Investigate scrape failures and attempt to re-fetch missing games',
            'count': len(failed_ids)
        })
    
    if teams and all(team.get('t_rank') in [0, None] for team in teams if isinstance(team, dict)):
        report['recommendations'].append({
            'issue': 't_rank zeros',
            'action': 'Compute t_rank from other ranking metrics or investigate data source',
            'count': len(teams)
        })
    
    # Save report
    report_file = Path('data/processed/data_quality_report.json')
    report_file.parent.mkdir(exist_ok=True)
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f'\n📄 Quality report saved: {report_file}')
    return report


def main():
    """Perform data quality cleanup."""
    print('🧹 Starting data quality cleanup...\n')
    
    report = generate_quality_report()
    
    print(f'\n✅ Cleanup complete!')
    print(f'TBD games removed: {report["cleaning_performed"].get("tbd_games_removed", 0)}')
    print(f'Issues investigated: {len(report["issues_found"])}')
    print(f'Recommendations: {len(report["recommendations"])}')


if __name__ == "__main__":
    main()
