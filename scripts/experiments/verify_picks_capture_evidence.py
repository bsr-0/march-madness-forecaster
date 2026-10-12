"""Re-check the capture evidence recorded in archived public-picks files.

A picks archive may carry ``captured_at`` only when ``capture_evidence`` points
at a committed Internet Archive capture of ESPN's Who Picked Whom page. This
script re-derives the claim from those bytes, so the timestamp is never taken
on trust:

* the evidence file exists and matches its recorded SHA-256;
* ``captured_at`` equals the archive timestamp, which is before the season's
  R64 tip (``season_calendar.get_round_of_64_tip``);
* the capture shows all 64 teams x 6 rounds, and every share in the archive
  equals the page at the page's 0.1-point display precision (or, for ESPN's
  2024+ propositions API, at the archive's 4-decimal precision).

The acceptance rule was pre-registered in docs/PROJECT_ROADMAP_2027.md
(decision log, 2026-10-11). Exit code 0 means every evidenced archive passes.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
import re
import sys
import zlib
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.data.historical_picks import _PICKS_TEAM_ALIAS  # noqa: E402
from src.data.normalize import normalize_team_id  # noqa: E402
from src.data.season_calendar import get_round_of_64_tip  # noqa: E402

PICKS_DIR = REPO / "data" / "raw" / "historical_public_picks"
ROUNDS = ["R64", "R32", "S16", "E8", "F4", "CHAMP"]
_CELL = re.compile(r'<td class="seed(\d+)_round\s*(\d)"\s*>(.*?)</td>', re.S)


def _decode(raw: bytes) -> str:
    """Archive ``id_`` captures keep the original transfer encoding."""
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    elif not raw.lstrip()[:1] in (b"<", b"[", b"{"):
        try:
            import brotli

            raw = brotli.decompress(raw)
        except Exception:
            raw = zlib.decompress(raw)
    return raw.decode("utf-8", errors="replace")


def parse_who_picked_whom(raw: bytes) -> dict[tuple[int, str], dict[str, int]]:
    """Map (seed, page name) -> {round: share in tenths of a percent}."""
    teams: dict[tuple[int, str], dict[str, int]] = {}
    for seed, rnd, inner in _CELL.findall(_decode(raw)):
        name = re.search(r'class="teamName"[^>]*>(.*?)<', inner, re.S) or re.search(r"<b>(.*?)</b>", inner, re.S)
        pct = re.search(r"(\d+(?:\.\d)?)%", inner)
        if name and pct:
            key = (int(seed), html.unescape(re.sub(r"\s+", " ", name.group(1)).strip()))
            teams.setdefault(key, {})[ROUNDS[int(rnd)]] = round(float(pct.group(1)) * 10)
    return teams


def parse_gambit_propositions(raw: bytes) -> dict[tuple[int, str], dict[str, float]]:
    """ESPN's 2024+ propositions API -> (seed, name) -> {round: share}."""
    teams: dict[tuple[int, str], dict[str, float]] = {}
    for prop in json.loads(_decode(raw)):
        period = prop.get("scoringPeriodId")
        for outcome in prop.get("possibleOutcomes", []):
            counters = [c for c in outcome.get("choiceCounters", []) if c.get("scoringFormatId", 5) == 5]
            if period and counters:
                key = (outcome["regionSeed"], outcome["name"])
                teams.setdefault(key, {})[ROUNDS[period - 1]] = counters[0]["percentage"]
    return teams


def _context_seeds(year: int) -> dict[str, int]:
    ctx = json.loads((REPO / "data" / "raw" / "historical" / f"tournament_context_{year}.json").read_text())
    return {t["team_id"]: t["seed"] for t in ctx["seeds"]["teams"]}


def _resolve(seed: int, name: str, stored_ids: set[str], seeds: dict[str, int], notes: dict[str, str]) -> str | None:
    """Identity from the page name and seed only; never from share values."""
    candidates = {k for k in stored_ids if seeds.get(k, seeds.get(_PICKS_TEAM_ALIAS.get(k, ""))) == seed}
    if name in notes:
        noted = notes[name].split(" ")[0]
        return noted if noted in candidates else None
    for raw in (name, normalize_team_id(name)):
        if raw in candidates:
            return raw
        if _PICKS_TEAM_ALIAS.get(raw) in candidates:
            return _PICKS_TEAM_ALIAS[raw]
    return None


def verify_archive(path: Path) -> list[str]:
    """Return the problems found in one evidenced archive (empty means it passes)."""
    data = json.loads(path.read_text())
    year = data["year"]
    evidence = data.get("capture_evidence")
    if "captured_at" not in data:
        return []
    if not isinstance(evidence, dict):
        return [f"{path.name}: captured_at without capture_evidence"]
    capture = evidence["capture"]
    evidence_path = REPO / capture["evidence_file"]
    problems = []
    if not evidence_path.is_file():
        return [f"{path.name}: evidence file missing: {capture['evidence_file']}"]
    raw = evidence_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != capture["evidence_sha256"]:
        problems.append(f"{path.name}: evidence SHA-256 differs")
    captured = datetime.fromisoformat(data["captured_at"])
    archived = datetime.fromisoformat(capture["archive_timestamp_utc"])
    if captured != archived:
        problems.append(f"{path.name}: captured_at {captured} differs from archive timestamp {archived}")
    if not capture["raw_url"].startswith(f"https://web.archive.org/web/{archived:%Y%m%d%H%M%S}id_/"):
        problems.append(f"{path.name}: archive URL does not carry the recorded timestamp")
    tip = get_round_of_64_tip(year)
    if not captured < tip:
        problems.append(f"{path.name}: capture {captured} is not before the R64 tip {tip}")

    api = evidence.get("format") == "gambit_api"
    page = parse_gambit_propositions(raw) if api else parse_who_picked_whom(raw)
    stored = data["teams"]
    seeds = _context_seeds(year)
    notes = evidence.get("name_mapping_notes", {})
    if len(page) != 64 or any(set(v) != set(ROUNDS) for v in page.values()):
        problems.append(f"{path.name}: capture does not show 64 teams x 6 rounds")
    matched: dict[str, tuple[int, str]] = {}
    for seed, name in page:
        team_id = _resolve(seed, name, set(stored), seeds, notes)
        if team_id is None or team_id in matched:
            problems.append(f"{path.name}: cannot uniquely map page team {name!r} (seed {seed})")
            continue
        matched[team_id] = (seed, name)
    if set(matched) != set(stored):
        problems.append(f"{path.name}: archive teams differ from the capture: {sorted(set(stored) ^ set(matched))}")
    differing = [
        (team_id, rnd)
        for team_id, key in matched.items()
        for rnd in ROUNDS
        if (
            round(stored[team_id][rnd], 4) != round(page[key].get(rnd, -1.0), 4)
            if api
            else round(stored[team_id][rnd] * 1000) != page[key].get(rnd)
        )
    ]
    if differing:
        problems.append(f"{path.name}: {len(differing)} share(s) differ from the capture, e.g. {differing[:3]}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--picks-dir", type=Path, default=PICKS_DIR)
    args = parser.parse_args()
    checked, failures = [], []
    for path in sorted(args.picks_dir.glob("espn_picks_*.json")):
        if "captured_at" in json.loads(path.read_text()):
            checked.append(path.name)
            failures.extend(verify_archive(path))
    print(json.dumps({"evidenced_archives": checked, "problems": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
