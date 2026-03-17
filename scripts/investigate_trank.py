#!/usr/bin/env python3
"""Investigate and potentially fix t_rank zeros in advanced_metrics_2026.json."""

import json
from pathlib import Path


def investigate_trank_calculation():
    """Investigate how t_rank should be calculated from other metrics."""
    file_path = Path('data/raw/advanced_metrics_2026.json')
    
    with open(file_path) as f:
        data = json.load(f)
    
    teams = data if isinstance(data, list) else data.get('teams', [])
    
    print(f'Analyzing t_rank for {len(teams)} teams...')
    
    # Check available ranking fields
    sample_team = teams[0] if teams else {}
    ranking_fields = [k for k in sample_team.keys() if 'rank' in k.lower()]
    print(f'Available ranking fields: {ranking_fields}')
    
    # Check if t_rank might be calculated from overall_rank
    if 'overall_rank' in ranking_fields:
        overall_ranks = [team.get('overall_rank') for team in teams if isinstance(team, dict)]
        valid_overall = [r for r in overall_ranks if r is not None and r > 0]
        print(f'overall_rank: {len(valid_overall)} valid values, range {min(valid_overall)}-{max(valid_overall)}')
    
    # Check correlation between t_rank and other metrics
    if 'barthag' in sample_team:
        barthag_values = [team.get('barthag') for team in teams if isinstance(team, dict)]
        valid_barthag = [b for b in barthag_values if b is not None]
        print(f'barthag: {len(valid_barthag)} valid values, range {min(valid_barthag):.3f}-{max(valid_barthag):.3f}')
    
    # Show sample teams with their rankings
    print('\nSample teams (top 10 by barthag):')
    sorted_teams = sorted(teams, key=lambda x: x.get('barthag', 0), reverse=True)[:10]
    for i, team in enumerate(sorted_teams):
        team_name = str(team.get("team_name", "Unknown"))
        barthag = team.get("barthag", 0)
        overall_rank = str(team.get("overall_rank", "N/A"))
        trank = str(team.get("t_rank", "N/A"))
        print(f"  {i+1:2d}. {team_name[:20]:20s} "
              f"barthag: {barthag:.3f}, "
              f"overall_rank: {overall_rank:>4s}, "
              f"t_rank: {trank:>4s}")
    
    return teams


def compute_trank_from_barthag(teams):
    """Compute t_rank from barthag values (common approach)."""
    # Sort by barthag descending, assign ranks
    sorted_teams = sorted(teams, key=lambda x: x.get('barthag', 0), reverse=True)
    
    for i, team in enumerate(sorted_teams):
        team['computed_t_rank'] = i + 1
    
    return sorted_teams


def fix_trank_values():
    """Fix t_rank values by computing from barthag."""
    file_path = Path('data/raw/advanced_metrics_2026.json')
    backup_path = Path('data/raw/advanced_metrics_2026_original.json')
    
    # Create backup
    if not backup_path.exists():
        with open(file_path) as f:
            original_data = json.load(f)
        with open(backup_path, 'w') as f:
            json.dump(original_data, f, indent=2)
        print(f'Created backup: {backup_path}')
    
    # Load and compute t_rank
    with open(file_path) as f:
        data = json.load(f)
    
    teams = data if isinstance(data, list) else data.get('teams', [])
    
    # Compute t_rank from barthag
    teams_with_trank = compute_trank_from_barthag(teams)
    
    # Update t_rank values
    for team in teams_with_trank:
        if isinstance(team, dict):
            team['t_rank'] = team['computed_t_rank']
            del team['computed_t_rank']  # Remove temporary field
    
    # Save updated data
    updated_data = teams_with_trank if isinstance(data, list) else {**data, 'teams': teams_with_trank}
    
    with open(file_path, 'w') as f:
        json.dump(updated_data, f, indent=2)
    
    print(f'Updated t_rank values for {len(teams)} teams')
    
    # Verify fix
    with open(file_path) as f:
        check_data = json.load(f)
    
    check_teams = check_data if isinstance(check_data, list) else check_data.get('teams', [])
    zero_count = sum(1 for t in check_teams if t.get('t_rank') == 0)
    print(f'Verification: {zero_count} teams still have t_rank=0')
    
    return len(teams)


def main():
    """Investigate and fix t_rank zeros."""
    print('🔍 Investigating t_rank zeros...\n')
    
    teams = investigate_trank_calculation()
    
    print('\n🔧 Attempting to fix t_rank values...')
    fixed_count = fix_trank_values()
    
    print(f'\n✅ Fixed t_rank for {fixed_count} teams')


if __name__ == "__main__":
    main()
