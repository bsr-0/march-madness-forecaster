#!/usr/bin/env python3
"""Create team ID mapping table across different data sources."""

import json
from pathlib import Path
from collections import defaultdict


def load_team_ids_from_file(file_path: Path, id_key: str, name_key: str) -> dict:
    """Load team IDs from a JSON file.
    
    Args:
        file_path: Path to JSON file
        id_key: Key for team ID
        name_key: Key for team name
        
    Returns:
        Dict mapping team_id -> team_name
    """
    try:
        with open(file_path) as f:
            data = json.load(f)
        
        # Handle different file structures
        teams = []
        if isinstance(data, dict):
            if "teams" in data:
                teams = data["teams"]
            elif all(isinstance(v, dict) for v in list(data.values())[:3]):
                # Dict keyed by team_id
                teams = [{"team_id": k, "team_name": v.get("team_name", k)} for k, v in data.items()]
        elif isinstance(data, list):
            teams = data
        
        result = {}
        for team in teams:
            if isinstance(team, dict):
                team_id = team.get(id_key)
                team_name = team.get(name_key)
                if team_id and team_name:
                    result[team_id] = team_name
        
        return result
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}


def build_team_mapping() -> dict:
    """Build comprehensive team ID mapping across sources."""
    data_dir = Path('data/raw')
    hist_dir = Path('data/raw/historical')
    
    mapping = defaultdict(dict)
    
    # Load from team_metrics (canonical source - uses underscores)
    for year in [2025, 2026]:
        file_path = hist_dir / f'team_metrics_{year}.json'
        teams = load_team_ids_from_file(file_path, 'team_id', 'team_name')
        for team_id, team_name in teams.items():
            mapping[team_name]['team_metrics'] = team_id
    
    # Load from torvik (uses abbreviations)
    for year in [2025, 2026]:
        file_path = hist_dir / f'torvik_{year}.json'
        if not file_path.exists():
            file_path = data_dir / f'torvik_{year}.json'
        teams = load_team_ids_from_file(file_path, 'team_id', 'team_name')
        for team_id, team_name in teams.items():
            mapping[team_name]['torvik'] = team_id
    
    # Load from massey composite (mixed format)
    for year in [2025, 2026]:
        file_path = hist_dir / f'external_massey_composite_{year}.json'
        teams = load_team_ids_from_file(file_path, 'team_id', 'team_name')
        for team_id, team_name in teams.items():
            mapping[team_name]['massey'] = team_id
    
    return dict(mapping)


def normalize_team_name(name: str) -> str:
    """Normalize team name for matching."""
    if not name:
        return name
    
    # Remove common suffixes and normalize
    name = name.lower()
    suffixes = [' university', ' college', ' state', ' st', ' st.', ' university', ' institut', ' institute']
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    
    # Replace common abbreviations
    replacements = {
        'st': 'state',
        'cal': 'california',
        'usc': 'southern california',
        'ucla': 'california los angeles',
        'unc': 'north carolina',
        'ucf': 'central florida',
        'usf': 'south florida',
        'smu': 'southern methodist',
        'umass': 'massachusetts',
        'uconn': 'connecticut',
        'bc': 'boston college',
        'vt': 'virginia tech',
        'gt': 'georgia tech',
        'fsu': 'florida state',
        'lsu': 'louisiana state',
        'asu': 'arizona state',
        'osu': 'ohio state',
        'msu': 'michigan state',
        'isu': 'iowa state',
        'ksu': 'kansas state',
        'wsu': 'washington state',
        'psu': 'penn state',
        'niu': 'northern illinois',
        'siu': 'southern illinois',
        'ui': 'illinois',
        'ut': 'texas',
        'ua': 'arizona',
        'cu': 'colorado',
        'mu': 'missouri',
        'ku': 'kansas',
        'nu': 'nebraska',
        'ou': 'oklahoma',
        'tu': 'tulane',
        'bu': 'boston university',
    }
    
    for old, new in replacements.items():
        if name == old:
            name = new
    
    # Clean up spaces and punctuation
    name = ' '.join(name.split())
    name = name.replace(' & ', ' and ')
    
    return name


def main():
    """Create team ID mapping table."""
    mapping = build_team_mapping()
    
    # Create normalized mapping for fuzzy matching
    normalized_mapping = defaultdict(dict)
    
    for team_name, ids in mapping.items():
        norm_name = normalize_team_name(team_name)
        for source, team_id in ids.items():
            normalized_mapping[norm_name][source] = team_id
            # Also store original name
            normalized_mapping[norm_name]['original_name'] = team_name
    
    # Save both mappings
    output_dir = Path('data/processed')
    output_dir.mkdir(exist_ok=True)
    
    # Direct mapping (exact name matches)
    with open(output_dir / 'team_id_mapping.json', 'w') as f:
        json.dump(mapping, f, indent=2, sort_keys=True)
    
    # Normalized mapping (for fuzzy matching)
    with open(output_dir / 'team_id_mapping_normalized.json', 'w') as f:
        json.dump(normalized_mapping, f, indent=2, sort_keys=True)
    
    print(f'Created team ID mappings:')
    print(f'  Direct mapping: {output_dir / "team_id_mapping.json"}')
    print(f'  Normalized mapping: {output_dir / "team_id_mapping_normalized.json"}')
    
    # Statistics
    total_teams = len(mapping)
    complete_mappings = sum(1 for ids in mapping.values() if len(ids) == 3)
    partial_mappings = total_teams - complete_mappings
    
    print(f'  Total teams: {total_teams}')
    print(f'  Complete mappings (all 3 sources): {complete_mappings}')
    print(f'  Partial mappings: {partial_mappings}')
    
    # Show examples of different naming conventions
    print(f'\nSample naming differences:')
    examples = []
    for team_name, ids in list(mapping.items())[:10]:
        if len(ids) > 1:
            examples.append((team_name, ids))
    
    for team_name, ids in examples[:5]:
        print(f'  {team_name}:')
        for source, team_id in ids.items():
            print(f'    {source}: {team_id}')


if __name__ == "__main__":
    main()
