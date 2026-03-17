#!/usr/bin/env python3
"""Validate that all data integrity fixes are working correctly."""

import json
from pathlib import Path


def validate_schema_consistency():
    """Validate that game schemas are consistent across directories."""
    print('🔍 Validating schema consistency...')
    
    issues = []
    
    for year in [2025, 2026]:
        raw_file = Path(f'data/raw/historical_games_{year}.json')
        hist_file = Path(f'data/raw/historical/historical_games_{year}.json')
        
        if not raw_file.exists() or not hist_file.exists():
            issues.append(f'Missing file for {year}')
            continue
        
        # Load and compare schemas
        with open(raw_file) as f:
            raw_data = json.load(f)
        with open(hist_file) as f:
            hist_data = json.load(f)
        
        raw_games = raw_data.get('games', [])
        hist_games = hist_data.get('games', [])
        
        if raw_games and hist_games:
            raw_keys = set(raw_games[0].keys())
            hist_keys = set(hist_games[0].keys())
            
            if raw_keys != hist_keys:
                issues.append(f'Schema mismatch for {year}: {raw_keys - hist_keys} vs {hist_keys - raw_keys}')
            else:
                print(f'  ✅ {year}: Schemas match')
    
    return issues


def validate_tournament_data():
    """Validate that tournament data is now complete."""
    print('🔍 Validating tournament data...')
    
    issues = []
    
    # Check 2026 seeds
    seeds_file = Path('data/raw/tournament_seeds_2026.json')
    if not seeds_file.exists():
        issues.append('tournament_seeds_2026.json missing')
    else:
        with open(seeds_file) as f:
            seeds = json.load(f)
        if len(seeds) != 68:
            issues.append(f'tournament_seeds_2026 has {len(seeds)} teams, expected 68')
        else:
            print(f'  ✅ tournament_seeds_2026: {len(seeds)} teams')
    
    # Check missing tournament results
    hist_dir = Path('data/raw/historical')
    missing_years = []
    
    for year in range(2005, 2018):
        results_file = hist_dir / f'tournament_results_{year}.json'
        if not results_file.exists():
            missing_years.append(year)
    
    # Check 2026
    results_2026 = hist_dir / 'tournament_results_2026.json'
    if not results_2026.exists():
        missing_years.append(2026)
    
    if missing_years:
        issues.append(f'Missing tournament results for years: {missing_years}')
    else:
        print(f'  ✅ Tournament results exist for all years 2005-2017, 2026')
    
    return issues


def validate_team_mappings():
    """Validate that team ID mappings were created."""
    print('🔍 Validating team ID mappings...')
    
    issues = []
    
    mapping_file = Path('data/processed/team_id_mapping.json')
    if not mapping_file.exists():
        issues.append('team_id_mapping.json missing')
    else:
        with open(mapping_file) as f:
            mapping = json.load(f)
        
        total_teams = len(mapping)
        complete_mappings = sum(1 for ids in mapping.values() if len(ids) == 3)
        
        if total_teams == 0:
            issues.append('team_id_mapping.json is empty')
        else:
            print(f'  ✅ Team mappings: {total_teams} teams, {complete_mappings} complete')
    
    return issues


def validate_external_rating_flags():
    """Validate that external rating issues were flagged."""
    print('🔍 Validating external rating flags...')
    
    issues = []
    
    report_file = Path('data/processed/external_rating_gaps_report.json')
    if not report_file.exists():
        issues.append('external_rating_gaps_report.json missing')
    else:
        with open(report_file) as f:
            report = json.load(f)
        
        total_issues = report.get('summary', {}).get('total_issues', 0)
        print(f'  ✅ External rating issues flagged: {total_issues}')
    
    return issues


def validate_data_quality():
    """Validate that data quality cleanup was performed."""
    print('🔍 Validating data quality cleanup...')
    
    issues = []
    
    # Check for TBD games
    for year in [2025, 2026]:
        for file_path in [Path(f'data/raw/historical_games_{year}.json'),
                         Path(f'data/raw/historical/historical_games_{year}.json')]:
            if not file_path.exists():
                continue
            
            with open(file_path) as f:
                data = json.load(f)
            
            games = data.get('games', [])
            tbd_count = sum(1 for g in games if g.get('team1_name') == 'TBD' or g.get('team2_name') == 'TBD')
            
            if tbd_count > 0:
                issues.append(f'{file_path} still has {tbd_count} TBD games')
    
    if not issues:
        print('  ✅ No TBD games found')
    
    # Check quality report
    quality_file = Path('data/processed/data_quality_report.json')
    if not quality_file.exists():
        issues.append('data_quality_report.json missing')
    else:
        print('  ✅ Data quality report generated')
    
    return issues


def main():
    """Run comprehensive validation of all data integrity fixes."""
    print('🚀 Running data integrity validation...\n')
    
    all_issues = []
    
    # Run all validations
    all_issues.extend(validate_schema_consistency())
    print()
    
    all_issues.extend(validate_tournament_data())
    print()
    
    all_issues.extend(validate_team_mappings())
    print()
    
    all_issues.extend(validate_external_rating_flags())
    print()
    
    all_issues.extend(validate_data_quality())
    print()
    
    # Summary
    if all_issues:
        print('❌ Validation found issues:')
        for issue in all_issues:
            print(f'  - {issue}')
    else:
        print('🎉 All validations passed! Data integrity fixes are working correctly.')
    
    # Generate final validation report
    validation_report = {
        'timestamp': str(Path.cwd()),
        'status': 'passed' if not all_issues else 'failed',
        'issues_found': len(all_issues),
        'issues': all_issues
    }
    
    report_file = Path('data/processed/validation_report.json')
    report_file.parent.mkdir(exist_ok=True)
    with open(report_file, 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    print(f'\n📄 Validation report saved: {report_file}')
    
    return len(all_issues) == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
