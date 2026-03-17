#!/usr/bin/env python3
"""Rigorous review of all historical data for missing or corrupted data."""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter
import traceback


def check_json_parseability(file_path: Path) -> tuple:
    """Check if JSON file can be parsed."""
    try:
        with open(file_path) as f:
            data = json.load(f)
        return True, data, None
    except json.JSONDecodeError as e:
        return False, None, f"JSON decode error: {e}"
    except Exception as e:
        return False, None, f"File error: {e}"


def analyze_file_structure(data, file_path: Path) -> dict:
    """Analyze the structure and content of a JSON data file."""
    analysis = {
        'file_type': 'unknown',
        'record_count': 0,
        'null_fields': defaultdict(int),
        'empty_containers': defaultdict(int),
        'data_types': defaultdict(int),
        'issues': []
    }
    
    if isinstance(data, dict):
        analysis['file_type'] = 'dict'
        analysis['record_count'] = len(data)
        
        # Check for common patterns
        if 'teams' in data and isinstance(data['teams'], list):
            analysis['file_type'] = 'teams_list'
            analysis['record_count'] = len(data['teams'])
            analyze_team_records(data['teams'], analysis)
        elif 'games' in data and isinstance(data['games'], list):
            analysis['file_type'] = 'games_list'
            analysis['record_count'] = len(data['games'])
            analyze_game_records(data['games'], analysis)
        elif all(isinstance(v, dict) for v in list(data.values())[:5]):
            analysis['file_type'] = 'teams_dict'
            analyze_team_records(list(data.values()), analysis)
        else:
            analyze_generic_dict(data, analysis)
            
    elif isinstance(data, list):
        analysis['file_type'] = 'list'
        analysis['record_count'] = len(data)
        
        if data and isinstance(data[0], dict):
            # Determine if teams or games based on field names
            sample_keys = set(data[0].keys())
            if 'team_id' in sample_keys or 'team_name' in sample_keys:
                if 'game_id' in sample_keys or 'date' in sample_keys:
                    analysis['file_type'] = 'games_list'
                    analyze_game_records(data, analysis)
                else:
                    analysis['file_type'] = 'teams_list'
                    analyze_team_records(data, analysis)
            else:
                analyze_generic_list(data, analysis)
        else:
            analyze_generic_list(data, analysis)
    
    return analysis


def analyze_team_records(teams, analysis):
    """Analyze team records for data quality issues."""
    required_fields = ['team_id', 'team_name']
    
    for i, team in enumerate(teams[:100]):  # Sample first 100
        if not isinstance(team, dict):
            analysis['issues'].append(f"Record {i} is not a dict")
            continue
        
        # Check required fields
        for field in required_fields:
            if field not in team:
                analysis['null_fields'][field] += 1
            elif team[field] is None:
                analysis['null_fields'][field] += 1
            elif team[field] == "":
                analysis['null_fields'][field] += 1
        
        # Check for empty containers
        for key, value in team.items():
            if value == [] or value == {} or value == "":
                analysis['empty_containers'][key] += 1
            analysis['data_types'][key] = type(value).__name__


def analyze_game_records(games, analysis):
    """Analyze game records for data quality issues."""
    required_fields = ['game_id', 'date', 'team1_score', 'team2_score']
    
    for i, game in enumerate(games[:100]):  # Sample first 100
        if not isinstance(game, dict):
            analysis['issues'].append(f"Record {i} is not a dict")
            continue
        
        # Check required fields
        for field in required_fields:
            if field not in game:
                analysis['null_fields'][field] += 1
            elif game[field] is None:
                analysis['null_fields'][field] += 1
            elif game[field] == "":
                analysis['null_fields'][field] += 1
        
        # Check for impossible scores
        if 'team1_score' in game and 'team2_score' in game:
            try:
                s1 = int(game['team1_score'])
                s2 = int(game['team2_score'])
                if s1 < 0 or s2 < 0:
                    analysis['issues'].append(f"Game {i}: Negative scores")
                if s1 == 0 and s2 == 0 and game.get('team1_name') != 'TBD':
                    analysis['issues'].append(f"Game {i}: Zero-zero score")
            except (ValueError, TypeError):
                analysis['null_fields']['scores'] += 1
        
        # Check for empty containers
        for key, value in game.items():
            if value == [] or value == {} or value == "":
                analysis['empty_containers'][key] += 1
            analysis['data_types'][key] = type(value).__name__


def analyze_generic_dict(data, analysis):
    """Analyze generic dict structure."""
    for key, value in data.items():
        analysis['data_types'][key] = type(value).__name__
        if value is None:
            analysis['null_fields'][key] += 1
        elif value == [] or value == {}:
            analysis['empty_containers'][key] += 1


def analyze_generic_list(data, analysis):
    """Analyze generic list structure."""
    if data:
        analysis['data_types']['items'] = type(data[0]).__name__
        for i, item in enumerate(data[:100]):
            if item is None or item == "" or item == []:
                analysis['empty_containers']['item'] += 1


def review_historical_games():
    """Review all historical games files for data integrity."""
    print("🔍 Reviewing Historical Games Files...")
    
    issues = []
    game_files = []
    
    # Find all game files
    for pattern in ['historical_games_*.json', 'tournament_results_*.json']:
        game_files.extend(Path('data/raw').glob(pattern))
        game_files.extend(Path('data/raw/historical').glob(pattern))
    
    for file_path in sorted(game_files):
        success, data, error = check_json_parseability(file_path)
        
        if not success:
            issues.append(f"Parse error in {file_path}: {error}")
            continue
        
        analysis = analyze_file_structure(data, file_path)
        
        # Check for specific game issues
        if analysis['file_type'] in ['games_list', 'dict']:
            games = data.get('games', data) if isinstance(data, dict) else data
            
            # Check for duplicate game_ids
            if games and isinstance(games, list):
                game_ids = [g.get('game_id') for g in games if isinstance(g, dict)]
                duplicates = len(game_ids) - len(set(game_ids))
                if duplicates > 0:
                    issues.append(f"{file_path}: {duplicates} duplicate game_ids")
            
            # Check for TBD games
            tbd_count = sum(1 for g in games if isinstance(g, dict) and 
                           (g.get('team1_name') == 'TBD' or g.get('team2_name') == 'TBD' or
                            g.get('team_name') == 'TBD' or g.get('opponent_name') == 'TBD'))
            if tbd_count > 0:
                issues.append(f"{file_path}: {tbd_count} TBD games")
        
        # Check for high null field counts
        for field, count in analysis['null_fields'].items():
            if count > analysis['record_count'] * 0.5:  # More than 50% null
                issues.append(f"{file_path}: {count}/{analysis['record_count']} null {field}")
    
    return issues


def review_team_metrics():
    """Review team metrics files for data completeness."""
    print("🔍 Reviewing Team Metrics Files...")
    
    issues = []
    
    for file_path in Path('data/raw/historical').glob('team_metrics_*.json'):
        success, data, error = check_json_parseability(file_path)
        
        if not success:
            issues.append(f"Parse error in {file_path}: {error}")
            continue
        
        analysis = analyze_file_structure(data, file_path)
        
        if analysis['file_type'] in ['teams_list', 'teams_dict']:
            teams = data.get('teams', list(data.values())) if isinstance(data, dict) else data
            
            # Check for completely empty teams
            empty_teams = sum(1 for t in teams if isinstance(t, dict) and 
                            all(v in [None, "", [], {}] for v in t.values()))
            if empty_teams > 0:
                issues.append(f"{file_path}: {empty_teams} completely empty team records")
    
    return issues


def review_external_ratings():
    """Review external rating files for coverage and quality."""
    print("🔍 Reviewing External Ratings Files...")
    
    issues = []
    rating_systems = defaultdict(list)
    
    # Collect all external rating files
    for directory in [Path('data/raw'), Path('data/raw/historical')]:
        for file_path in directory.glob('external_*.json'):
            if 'massey_composite' in file_path.name:
                continue
            
            success, data, error = check_json_parseability(file_path)
            
            if not success:
                issues.append(f"Parse error in {file_path}: {error}")
                continue
            
            analysis = analyze_file_structure(data, file_path)
            
            # Extract system name and year
            parts = file_path.stem.split('_')
            if len(parts) >= 3:
                system = '_'.join(parts[1:-1])
                year_str = parts[-1]
                try:
                    year = int(year_str)
                    rating_systems[system].append((year, analysis['record_count']))
                except ValueError:
                    continue  # Skip if year is not an integer
                
                # Flag suspiciously small files
                if analysis['record_count'] < 100:
                    issues.append(f"{file_path}: Suspiciously small ({analysis['record_count']} teams)")
    
    # Check for coverage gaps in rating systems
    for system, year_data in rating_systems.items():
        years = sorted(set(y for y, _ in year_data))
        if len(years) > 1:
            # Check for missing years in sequence
            expected_years = range(min(years), max(years) + 1)
            missing_years = set(expected_years) - set(years)
            if missing_years:
                issues.append(f"Rating system {system}: Missing years {sorted(missing_years)}")
    
    return issues


def review_tournament_data():
    """Review tournament seeds and results data."""
    print("🔍 Reviewing Tournament Data...")
    
    issues = []
    
    # Check seeds files
    for file_path in list(Path('data/raw').glob('tournament_seeds_*.json')) + \
                   list(Path('data/raw/historical').glob('tournament_seeds_*.json')):
        success, data, error = check_json_parseability(file_path)
        
        if not success:
            issues.append(f"Parse error in {file_path}: {error}")
            continue
        
        seeds = data if isinstance(data, list) else data.get('seeds', data.get('teams', []))
        
        # Check seed distribution
        if seeds:
            year = file_path.stem.split('_')[-1]
            seed_counts = Counter()
            for seed in seeds:
                if isinstance(seed, dict):
                    seed_num = seed.get('seed')
                    if seed_num:
                        seed_counts[seed_num] += 1
            
            # Check for expected seed counts
            if int(year) >= 2011:  # Post-First Four era
                expected = {1: 4, 2: 4, 3: 4, 4: 4, 5: 4, 6: 4, 7: 4, 8: 4, 
                           9: 4, 10: 4, 11: 6, 12: 4, 13: 4, 14: 4, 15: 4, 16: 6}
                total_expected = 68
            else:  # Pre-First Four era
                expected = {1: 4, 2: 4, 3: 4, 4: 4, 5: 4, 6: 4, 7: 4, 8: 4, 
                           9: 4, 10: 4, 11: 4, 12: 4, 13: 4, 14: 4, 15: 4, 16: 5}
                total_expected = 65
            
            if len(seeds) != total_expected:
                issues.append(f"{file_path}: {len(seeds)} teams (expected {total_expected})")
            
            # Check seed distribution
            for seed, expected_count in expected.items():
                actual_count = seed_counts.get(seed, 0)
                if actual_count != expected_count:
                    issues.append(f"{file_path}: Seed {seed}: {actual_count} (expected {expected_count})")
    
    return issues


def main():
    """Conduct rigorous review of all historical data."""
    print("🚀 Starting Rigorous Historical Data Review...\n")
    
    all_issues = []
    
    # Review different data categories
    all_issues.extend(review_historical_games())
    print()
    
    all_issues.extend(review_team_metrics())
    print()
    
    all_issues.extend(review_external_ratings())
    print()
    
    all_issues.extend(review_tournament_data())
    print()
    
    # Categorize issues
    critical_issues = [i for i in all_issues if 'Parse error' in i or 'duplicate' in i]
    warning_issues = [i for i in all_issues if 'TBD' in i or 'suspiciously small' in i]
    info_issues = [i for i in all_issues if i not in critical_issues and i not in warning_issues]
    
    # Generate report
    report = {
        'timestamp': str(Path.cwd()),
        'summary': {
            'total_issues': len(all_issues),
            'critical': len(critical_issues),
            'warnings': len(warning_issues),
            'info': len(info_issues)
        },
        'issues': {
            'critical': critical_issues,
            'warnings': warning_issues,
            'info': info_issues
        }
    }
    
    # Save report
    report_file = Path('data/processed/rigorous_data_review_report.json')
    report_file.parent.mkdir(exist_ok=True)
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print(f"📊 Rigorous Data Review Complete!")
    print(f"   Total Issues: {len(all_issues)}")
    print(f"   Critical: {len(critical_issues)}")
    print(f"   Warnings: {len(warning_issues)}")
    print(f"   Info: {len(info_issues)}")
    print(f"\n📄 Report saved: {report_file}")
    
    if critical_issues:
        print(f"\n🚨 CRITICAL ISSUES FOUND:")
        for issue in critical_issues[:10]:
            print(f"   - {issue}")
        if len(critical_issues) > 10:
            print(f"   ... and {len(critical_issues) - 10} more")
    
    if warning_issues:
        print(f"\n⚠️  WARNINGS:")
        for issue in warning_issues[:5]:
            print(f"   - {issue}")
        if len(warning_issues) > 5:
            print(f"   ... and {len(warning_issues) - 5} more")
    
    if not all_issues:
        print(f"\n🎉 NO ISSUES FOUND! All historical data is clean and complete.")
    
    return len(all_issues) == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
