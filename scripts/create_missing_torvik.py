#!/usr/bin/env python3
"""Create synthetic torvik_*.json files for 2005-2007 from four_factors and shooting data."""

import json
from pathlib import Path


def create_torvik_from_components(year: int):
    """Create torvik_YYYY.json by combining four_factors and shooting data."""
    
    # Load component files
    ff_file = Path(f'data/raw/torvik_four_factors_{year}.json')
    sh_file = Path(f'data/raw/torvik_shooting_{year}.json')
    
    if not ff_file.exists() or not sh_file.exists():
        print(f'Missing component files for {year}')
        return False
    
    with open(ff_file) as f:
        ff_data = json.load(f)
    with open(sh_file) as f:
        sh_data = json.load(f)
    
    # Both files are dicts keyed by team name
    print(f'Four factors teams: {len(ff_data)}')
    print(f'Shooting teams: {len(sh_data)}')
    
    # Combine data
    combined_teams = {}
    
    for team_name in ff_data:
        if team_name in sh_data:
            # Merge the data
            ff_stats = ff_data[team_name]
            sh_stats = sh_data[team_name]
            
            # Create torvik-style record
            team_record = {
                'team_id': team_name.lower().replace(' ', '_').replace('.', '').replace(',', '').replace("'", ''),
                'team_name': team_name,
                'season': year,
                # Four factors stats
                'effective_fg_pct': ff_stats.get('effective_fg_pct', 0.0),
                'turnover_rate': ff_stats.get('turnover_rate', 0.0),
                'offensive_reb_rate': ff_stats.get('offensive_reb_rate', 0.0),
                'free_throw_rate': ff_stats.get('free_throw_rate', 0.0),
                'opp_effective_fg_pct': ff_stats.get('opp_effective_fg_pct', 0.0),
                'opp_turnover_rate': ff_stats.get('opp_turnover_rate', 0.0),
                'defensive_reb_rate': ff_stats.get('defensive_reb_rate', 0.0),
                'opp_free_throw_rate': ff_stats.get('opp_free_throw_rate', 0.0),
                # Shooting stats
                'fg_pct': sh_stats.get('fg_pct', 0.0),
                'fg3_pct': sh_stats.get('fg3_pct', 0.0),
                'ft_pct': sh_stats.get('ft_pct', 0.0),
                'fg3_rate': sh_stats.get('fg3_rate', 0.0),
                'ft_rate': sh_stats.get('ft_rate', 0.0),
            }
            
            combined_teams[team_name] = team_record
    
    print(f'Combined teams: {len(combined_teams)}')
    
    # Create torvik-style structure
    torvik_data = {
        'season': year,
        'teams': list(combined_teams.values()),
        'source': 'synthetic_from_components',
        'note': f'Generated from torvik_four_factors_{year}.json and torvik_shooting_{year}.json'
    }
    
    # Save to both locations for consistency
    output_files = [
        Path(f'data/raw/torvik_{year}.json'),
        Path(f'data/raw/historical/torvik_{year}.json')
    ]
    
    for output_file in output_files:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(torvik_data, f, indent=2)
        print(f'Created: {output_file}')
    
    return True


def validate_synthetic_torvik(year: int):
    """Validate the created torvik file."""
    file_path = Path(f'data/raw/torvik_{year}.json')
    
    if not file_path.exists():
        return False
    
    with open(file_path) as f:
        data = json.load(f)
    
    teams = data.get('teams', [])
    
    # Check required fields
    required_fields = ['team_id', 'team_name', 'effective_fg_pct', 'opp_effective_fg_pct']
    missing_fields = []
    
    for team in teams[:5]:  # Check first 5 teams
        for field in required_fields:
            if field not in team:
                missing_fields.append(field)
    
    if missing_fields:
        print(f'  ❌ Missing fields: {set(missing_fields)}')
        return False
    
    print(f'  ✅ Validation passed: {len(teams)} teams')
    return True


def main():
    """Create missing torvik files for 2005-2007."""
    print('🔧 Creating missing torvik files for 2005-2007...\n')
    
    for year in range(2005, 2008):
        print(f'=== {year} ===')
        
        success = create_torvik_from_components(year)
        
        if success:
            validate_synthetic_torvik(year)
        else:
            print(f'  ❌ Failed to create torvik_{year}.json')
        
        print()
    
    print('✅ Synthetic torvik file creation complete!')


if __name__ == "__main__":
    main()
