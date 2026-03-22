#!/usr/bin/env python3
"""Validate generated web data JSON files for completeness and correctness.

Extracted from the inline heredoc in run-pipeline.yml to provide better
diagnostics and maintainability.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

REQUIRED_FILES = [
    "docs/data/bracket_2026.json",
    "docs/data/validation_2025.json",
    "docs/data/model_metrics.json",
    "docs/data/team_profiles.json",
    "docs/data/dashboard.json",
]


def validate():
    errors = []
    for f in REQUIRED_FILES:
        p = Path(f)
        if not p.exists():
            errors.append(f"MISSING: {f}")
            continue

        size = p.stat().st_size
        mtime = datetime.fromtimestamp(p.stat().st_mtime).isoformat()

        if size == 0:
            errors.append(f"EMPTY (0 bytes): {f}")
            continue

        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            errors.append(f"INVALID JSON: {f} — {e}")
            continue

        if not data:
            errors.append(f"EMPTY DATA: {f} (parses to empty {type(data).__name__})")
            continue

        print(f"  OK: {f} ({size:,} bytes, modified {mtime})")

    # Extra validation: check validation data has per-year entries
    val_path = Path("docs/data/validation_2025.json")
    if val_path.exists():
        try:
            val = json.loads(val_path.read_text())
            years = [e["year"] for e in val.get("per_year", [])]
            print(f"  Validation years: {years}")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            errors.append(f"VALIDATION STRUCTURE ERROR: {val_path} — {e}")

    if errors:
        print("\nValidation FAILED:")
        for err in errors:
            print(f"  ::error::{err}")
        return 1

    print("\nAll files validated successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(validate())
