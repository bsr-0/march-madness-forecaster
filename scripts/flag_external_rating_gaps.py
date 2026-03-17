#!/usr/bin/env python3
"""Flag and attempt to repair external rating files with suspicious coverage gaps."""

import json
import os
from pathlib import Path
from collections import defaultdict


def analyze_external_ratings():
    """Analyze external rating files for coverage gaps."""
    issues = []
    expected_min_teams = 300  # Minimum expected teams for D-I ratings
    
    # Check both directories
    for directory in [Path('data/raw'), Path('data/raw/historical')]:
        if not directory.exists():
            continue
            
        for file_path in directory.glob('external_*.json'):
            if 'massey_composite' in file_path.name:
                continue  # Skip composite files
                
            try:
                with open(file_path) as f:
                    data = json.load(f)
                
                # Count teams
                if isinstance(data, list):
                    team_count = len(data)
                elif isinstance(data, dict):
                    if 'teams' in data:
                        team_count = len(data['teams'])
                    elif all(isinstance(v, dict) for v in list(data.values())[:3]):
                        team_count = len(data)
                    else:
                        team_count = 0
                else:
                    team_count = 0
                
                # Check for undersized files
                if team_count < expected_min_teams:
                    file_size = file_path.stat().st_size
                    issues.append({
                        'file': str(file_path),
                        'team_count': team_count,
                        'file_size': file_size,
                        'severity': 'critical' if team_count < 100 else 'warning'
                    })
                    
            except Exception as e:
                issues.append({
                    'file': str(file_path),
                    'error': str(e),
                    'severity': 'error'
                })
    
    return issues


def create_quality_flags(issues):
    """Create quality flags for problematic files."""
    flags = defaultdict(dict)
    
    for issue in issues:
        file_path = Path(issue['file'])
        year = None
        
        # Extract year from filename
        for part in file_path.stem.split('_'):
            if part.isdigit() and len(part) == 4:
                year = int(part)
                break
        
        if year:
            system = '_'.join(file_path.stem.split('_')[1:-1])  # Extract system name
            flags[year][system] = {
                'team_count': issue.get('team_count', 0),
                'file_size': issue.get('file_size', 0),
                'severity': issue['severity'],
                'note': 'Undersized external rating file'
            }
    
    return dict(flags)


def update_manifests(flags):
    """Update manifest files with quality flags."""
    manifest_dir = Path('data/raw')
    
    for year, systems in flags.items():
        manifest_file = manifest_dir / f'manifest_{year}.json'
        
        if not manifest_file.exists():
            continue
            
        try:
            with open(manifest_file) as f:
                manifest = json.load(f)
            
            # Add quality flags section
            if 'quality_flags' not in manifest:
                manifest['quality_flags'] = {}
            
            manifest['quality_flags']['external_ratings'] = systems
            
            # Save updated manifest
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            print(f'Updated {manifest_file} with quality flags')
            
        except Exception as e:
            print(f'Error updating {manifest_file}: {e}')


def generate_repair_report(issues):
    """Generate a report with repair recommendations."""
    report_file = Path('data/processed/external_rating_gaps_report.json')
    
    report = {
        'generated_at': str(Path.cwd()),
        'summary': {
            'total_issues': len(issues),
            'critical': len([i for i in issues if i.get('severity') == 'critical']),
            'warning': len([i for i in issues if i.get('severity') == 'warning']),
            'error': len([i for i in issues if i.get('severity') == 'error'])
        },
        'issues': issues,
        'recommendations': []
    }
    
    # Add recommendations
    critical_files = [i for i in issues if i.get('severity') == 'critical']
    if critical_files:
        report['recommendations'].append({
            'priority': 'high',
            'action': 'Re-download or investigate critical files',
            'files': [i['file'] for i in critical_files]
        })
    
    warning_files = [i for i in issues if i.get('severity') == 'warning']
    if warning_files:
        report['recommendations'].append({
            'priority': 'medium',
            'action': 'Verify data sources for warning files',
            'files': [i['file'] for i in warning_files]
        })
    
    # Save report
    report_file.parent.mkdir(exist_ok=True)
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report


def main():
    """Analyze and flag external rating coverage gaps."""
    print('Analyzing external rating coverage...')
    
    issues = analyze_external_ratings()
    
    if not issues:
        print('✅ No coverage issues found!')
        return
    
    print(f'Found {len(issues)} issues:')
    for issue in issues[:10]:  # Show first 10
        if 'team_count' in issue:
            print(f"  {issue['file']}: {issue['team_count']} teams ({issue['severity']})")
        else:
            print(f"  {issue['file']}: {issue.get('error', 'Unknown error')} ({issue['severity']})")
    
    if len(issues) > 10:
        print(f'  ... and {len(issues) - 10} more')
    
    # Create quality flags
    flags = create_quality_flags(issues)
    update_manifests(flags)
    
    # Generate repair report
    report = generate_repair_report(issues)
    
    print(f'\n📊 Summary:')
    print(f"  Critical: {report['summary']['critical']}")
    print(f"  Warning: {report['summary']['warning']}")
    print(f"  Error: {report['summary']['error']}")
    
    print(f'\n📄 Reports generated:')
    print(f'  {Path("data/processed/external_rating_gaps_report.json")}')
    print(f'  Manifest files updated with quality flags')


if __name__ == "__main__":
    main()
