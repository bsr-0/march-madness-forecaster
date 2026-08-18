"""Validate the generated docs/data/*.json payloads before deploy.

`.github/workflows/generate-web-data.yml` has always ended with
`python scripts/validate_web_data.py`, but the script never existed in this
repo's history — the step would have failed the job (the workflow runs with
`set -euo pipefail`) after doing all the generation work. It went unnoticed
because that workflow has never actually been triggered.

What this checks:

1. **Required files present.** The four payloads `docs/app.js` fetches in its
   blocking `Promise.all` — without any one of them the page renders an error
   banner instead of a bracket.
2. **Strict JSON.** Every payload is parsed with `NaN`/`Infinity` rejected.
   Python writes bare `NaN` for non-finite floats and reads it straight back,
   so a Python-only check passes while every browser refuses the file with
   "Unexpected token 'N'". This is not hypothetical: a NaN minutes value
   silently broke the whole team-stats table in the browser while the Python
   test suite stayed green.
3. **Shape spot-checks** on the payloads with known structure, so a file that
   is valid JSON but empty or truncated still fails.

Exits non-zero on any problem, with GitHub Actions error annotations.
"""

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "docs" / "data"

# Fetched in app.js's blocking Promise.all — the page cannot render without these.
REQUIRED = (
    "bracket_2026.json",
    "bracket_2026_exhaustive.json",
    "bracket_2026_region.json",
    "team_profiles.json",
)

# Fetched in their own try/catch; a missing one degrades to "no panel".
OPTIONAL = (
    "actual_2026.json",
    "loyo_points.json",
    "loyo_window_3yr_recency_fit.json",
    "team_stats_by_year.json",
    "matchups_by_year.json",
)

errors: list[str] = []
warnings: list[str] = []


def _err(msg: str) -> None:
    errors.append(msg)
    print(f"::error::{msg}")


def _warn(msg: str) -> None:
    warnings.append(msg)
    print(f"::warning::{msg}")


def load_strict(path: Path):
    """Parse JSON, rejecting the non-standard constants browsers refuse."""

    def reject(const):
        raise ValueError(f"non-standard JSON constant {const!r} (browsers reject this file)")

    with open(path) as f:
        return json.load(f, parse_constant=reject)


def check_no_nan(obj, path: Path, trail: str = "") -> None:
    """Belt and braces: catch non-finite floats the parser hook can't see.

    parse_constant only fires for bare NaN/Infinity literals. A file could in
    principle carry them another way, and this is cheap.
    """
    if isinstance(obj, float) and not math.isfinite(obj):
        _err(f"{path.name}: non-finite value at {trail or '<root>'}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            check_no_nan(v, path, f"{trail}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:200]):  # cap: these lists are long and homogeneous
            check_no_nan(v, path, f"{trail}[{i}]")


def check_bracket(name: str, data) -> None:
    rounds = data.get("rounds") if isinstance(data, dict) else None
    if not rounds:
        _err(f"{name}: no 'rounds' array")
        return
    if len(rounds) != 6:
        _err(f"{name}: {len(rounds)} rounds, expected 6 (R64..Championship)")
    total = sum(len(r.get("games", [])) for r in rounds)
    if total != 63:
        _err(f"{name}: {total} games, expected 63")


def check_team_profiles(data) -> None:
    teams = data.get("teams") if isinstance(data, dict) else None
    if not teams:
        _err("team_profiles.json: no 'teams' array")
        return
    if len(teams) != 68:
        _warn(f"team_profiles.json: {len(teams)} teams, expected 68")
    missing = [t.get("team_id") for t in teams if t.get("barthag") is None]
    if missing:
        _warn(f"team_profiles.json: {len(missing)} team(s) missing barthag, e.g. {missing[:3]}")


def check_by_year(name: str, data, key: str) -> None:
    years = data.get("years") if isinstance(data, dict) else None
    payload = data.get(key) if isinstance(data, dict) else None
    if not years or not payload:
        _err(f"{name}: missing 'years' or {key!r}")
        return
    for y in years:
        if str(y) not in payload:
            _err(f"{name}: year {y} listed but absent from {key!r}")
    if 2020 in years:
        _err(f"{name}: 2020 has no tournament and must not appear")


def main() -> int:
    if not DATA.is_dir():
        _err(f"{DATA} does not exist — did the generators run?")
        return 1

    for name in REQUIRED:
        if not (DATA / name).exists():
            _err(f"missing required payload: docs/data/{name}")

    for name in OPTIONAL:
        if not (DATA / name).exists():
            _warn(f"optional payload absent: docs/data/{name}")

    for path in sorted(DATA.glob("*.json")):
        try:
            data = load_strict(path)
        except ValueError as exc:
            _err(f"{path.name}: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001 - report whatever went wrong
            _err(f"{path.name}: unreadable ({exc})")
            continue

        check_no_nan(data, path)

        if path.name in ("bracket_2026.json", "bracket_2026_exhaustive.json", "bracket_2026_region.json"):
            check_bracket(path.name, data)
        elif path.name == "team_profiles.json":
            check_team_profiles(data)
        elif path.name == "team_stats_by_year.json":
            check_by_year(path.name, data, "stats_by_year")
        elif path.name == "matchups_by_year.json":
            check_by_year(path.name, data, "matchups_by_year")

    print()
    n = len(list(DATA.glob("*.json")))
    if errors:
        print(f"FAIL — {len(errors)} error(s), {len(warnings)} warning(s) across {n} payload(s).")
        return 1
    print(f"OK — {n} payload(s) valid ({len(warnings)} warning(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
