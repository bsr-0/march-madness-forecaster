#!/usr/bin/env python3
"""Capture ESPN public pick percentages at a stated instant, once.

WHY THIS EXISTS. PROSPECTIVE_2027 CHECKPOINT 2 fixes a prediction-time cutoff
for public pick shares -- 2027-03-18 12:00 ET, one capture, no re-capture --
and the provenance gate in ``build_candidate_artifact`` refuses to build
without a checkable capture time. Nothing in the repo could produce one.

The two existing ingestion paths are both archival and neither can serve a live
deadline:

  Kaggle import   writes no capture time at all (which is why
                  espn_picks_2026.json has none), and the dataset is published
                  after the tournament.
  Wayback scrape  writes Wayback's own timestamp format (20230315120000),
                  which is not ISO-8601, and needs a snapshot to already exist.

``ESPNPicksScraper`` could always fetch live; nothing wired it to the archive
directory with a timestamp. That is all this script is.

    python scripts/capture_public_picks.py --year 2027 --dry-run   # rehearse
    python scripts/capture_public_picks.py --year 2027             # the capture

REHEARSE IT. The ESPN endpoints are undocumented and have changed domains more
than once. ``--dry-run`` performs the whole fetch and every validation, and
writes nothing -- run it well before March, not fifteen minutes before tip.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.historical_picks import _DEFAULT_PICKS_DIR, archive_candidates  # noqa: E402
from src.data.season_calendar import (  # noqa: E402
    EASTERN,
    get_public_picks_cutoff,
    get_round_of_64_tip,
)

# The loader reads only these keys; a payload missing them is useless even if
# it parses.
ROUND_KEYS = ("R64", "R32", "S16", "E8", "F4", "CHAMP")

# ESPN fields out of the scraper are percentages (0-100); the archive format
# every consumer reads is a fraction (0-1). Converting in the wrong direction
# is a silent 100x error that would make every team look certain, so the
# conversion is asserted rather than assumed.
PCT_TO_FRACTION = 100.0

MIN_TEAMS = 32


class CaptureError(RuntimeError):
    """Raised when a capture cannot be performed or trusted."""


def _now_eastern() -> datetime:
    return datetime.now(tz=EASTERN)


def _assert_within_cutoff(year: int, now: datetime) -> None:
    """Refuse a capture that is already too late to be the declared one."""
    # The declared cutoff is checked FIRST. For 2027 it lands exactly on the R64
    # tip, so a late capture trips both -- and the operator needs the specific
    # message ("you missed the stated instant, record it") rather than the
    # generic one about locked brackets.
    declared = get_public_picks_cutoff(year)
    if declared is not None and now > declared:
        raise CaptureError(
            f"it is {now.isoformat()}, past the declared cutoff {declared.isoformat()} "
            f"for {year} (PROSPECTIVE_2027 CHECKPOINT 2).\n"
            f"There is no --force for this. Missing the deadline is an event to record "
            f"in the checkpoint document, not one to override: a capture taken after "
            f"the stated instant is not the capture that was promised, and the "
            f"provenance gate will refuse it on the same grounds."
        )

    tip = get_round_of_64_tip(year)
    if now > tip:
        raise CaptureError(
            f"it is {now.isoformat()}, past the {year} R64 tip ({tip.isoformat()}). "
            f"Brackets are locked; these shares would describe games already played."
        )


def _assert_not_already_captured(year: int, picks_dir: Path) -> None:
    """One capture, no re-capture -- for a season that declared a cutoff."""
    existing = [p for p in archive_candidates(year, picks_dir) if p.exists()]
    if not existing:
        return
    if get_public_picks_cutoff(year) is None:
        raise CaptureError(
            f"{existing[0]} already exists. Delete it if you mean to replace it "
            f"({year} has no declared cutoff, so this is ordinary development data)."
        )
    raise CaptureError(
        f"{existing[0]} already exists and {year} declared a cutoff.\n"
        f"The capture has already happened. Re-capturing and presenting the result "
        f"as the original observation is the failure mode CHECKPOINT 2 calls fatal."
    )


def to_archive_teams(consensus) -> Dict[str, Dict[str, float]]:
    """Convert ``ConsensusData`` to the on-disk archive shape, 0-1 per round."""
    teams: Dict[str, Dict[str, float]] = {}
    for team_id, picks in consensus.teams.items():
        as_pct = picks.as_dict  # {"R64": 0-100, ...}
        row: Dict[str, float] = {}
        for key in ROUND_KEYS:
            value = float(as_pct.get(key, 0.0)) / PCT_TO_FRACTION
            if not 0.0 <= value <= 1.0:
                raise CaptureError(
                    f"{team_id} {key}={as_pct.get(key)!r} is outside 0-100 after "
                    f"conversion ({value}); the scraper's scale has changed and the "
                    f"archive would be silently wrong."
                )
            row[key] = round(value, 6)
        if picks.seed:
            row["seed"] = picks.seed
        teams[team_id] = row
    return teams


def _assert_usable(teams: Dict[str, Dict[str, float]], year: int) -> None:
    if len(teams) < MIN_TEAMS:
        raise CaptureError(
            f"only {len(teams)} team(s) returned for {year}; expected at least "
            f"{MIN_TEAMS}. The endpoint probably changed shape -- capturing a "
            f"near-empty archive is worse than capturing nothing, because the "
            f"loader would fill the gaps with seed-based rates and say so only "
            f"in a log line."
        )
    if all(row.get("CHAMP", 0.0) == 0.0 for row in teams.values()):
        raise CaptureError(
            "every team has CHAMP=0. Championship shares are the field's single "
            "most concentrated belief and the term the pool edge leans on hardest; "
            "an all-zero column means the parse failed, not that nobody picked."
        )


def capture(year: int, picks_dir: Path, *, dry_run: bool) -> Dict:
    from src.data.scrapers.espn_picks import ESPNPicksScraper

    now = _now_eastern()
    _assert_within_cutoff(year, now)
    if not dry_run:
        _assert_not_already_captured(year, picks_dir)

    consensus = ESPNPicksScraper().fetch_picks(year)
    if not consensus.teams:
        raise CaptureError(
            f"no pick data returned for {year}. The scraper tries "
            f"ESPN_PUBLIC_PICKS_URL, then the Gambit API, then its cache; all "
            f"failed. Set ESPN_PUBLIC_PICKS_URL to a JSON endpoint and retry -- "
            f"and note the clock, because the cutoff does not move for this."
        )

    teams = to_archive_teams(consensus)
    _assert_usable(teams, year)

    payload = {
        "year": year,
        "source": "ESPN Tournament Challenge (live capture)",
        "source_chain": consensus.sources,
        # ISO-8601 with an offset. The gate rejects naive timestamps: the cutoff
        # is stated in Eastern time and a naive reading is worth four hours.
        "captured_at": now.isoformat(),
        "declared_cutoff": (
            get_public_picks_cutoff(year).isoformat() if get_public_picks_cutoff(year) else None
        ),
        "teams": teams,
    }
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, default=_DEFAULT_PICKS_DIR)
    ap.add_argument("--dry-run", action="store_true", help="fetch and validate, write nothing")
    args = ap.parse_args()

    try:
        payload = capture(args.year, args.out_dir, dry_run=args.dry_run)
    except CaptureError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1

    teams = payload["teams"]
    top = sorted(teams.items(), key=lambda kv: kv[1].get("CHAMP", 0.0), reverse=True)[:5]
    print(f"captured {len(teams)} teams for {args.year} at {payload['captured_at']}")
    print(f"  source chain: {payload['source_chain']}")
    print("  most-picked champions:")
    for team_id, row in top:
        print(f"    {team_id:<28} {row.get('CHAMP', 0.0):.1%}")

    if args.dry_run:
        print("\n--dry-run: nothing written. Re-run without it to record the capture.")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"espn_picks_{args.year}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
