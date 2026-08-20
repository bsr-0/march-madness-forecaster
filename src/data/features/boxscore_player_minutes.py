"""Season player-minutes features from ESPN box scores.

Drop-in replacement for ``pbp_player_minutes.build_season_minutes_features``
that reads ``boxscores_{year}.json`` instead of ``pbp_{year}.json``. The output
schema is byte-compatible with the PBP version so downstream consumers of
``player_minutes_{year}.json`` need no change:

    {"season": int,
     "players": [{team_id, athlete_id, athlete_name,
                  games_played, games_started, total_minutes, minutes_per_game}],
     "generated_at": str, "source": str,
     "metadata": {games_used, games_rejected}}

Why this exists rather than the PBP path: ESPN publishes no substitution events
before 2025-02-11, so the PBP reconstruction yields nothing for any earlier
season (2024 produced zero, 2023 produced 26 players out of ~6000 games). The
boxscore route works for every season back to at least 2009, is ESPN's own
published figure rather than a reconstruction, and labels starters explicitly
instead of inferring them. See ``src/data/scrapers/espn_boxscore.py``.

``team_id`` is the slugified ESPN display name (``"Cornell Big Red"`` ->
``"cornell_big_red"``), matching the ``athlete_team`` values the PBP scrape
already wrote, so the two sources join without a bridge. Team-ID bridging to
canonical tournament ids stays with the consumer, as it does for the PBP
version — that join is player-level.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

from ..scrapers.espn_boxscore import validate_boxscore_minutes


class MinutesCoverageError(RuntimeError):
    """Raised when a season yields implausibly little minutes data.

    The PBP pipeline logged ``player_minutes produced nothing`` for 2024 and
    wrote a 26-player file for 2023, and the run continued past both without
    failing — so a silent, total data loss looked like a successful backfill
    for hours. This makes that failure mode loud.
    """


# A season has ~360 D-I teams and ~5000-6000 pre-tournament games; anything
# below this many distinct players means the parse broke, not that basketball
# was quiet.
_MIN_PLAUSIBLE_PLAYERS = 500


def build_season_minutes_features(
    year: int,
    data_root,
    *,
    boxscore_payload: Optional[Dict] = None,
    strict: bool = True,
) -> Dict:
    """Aggregate per-player season minutes from a season's box scores.

    Args:
        strict: when True (default), raise :class:`MinutesCoverageError` if the
            season produces implausibly few players. Pass False for smoke runs
            over deliberately truncated payloads.

    Games whose minutes fail ``validate_boxscore_minutes`` are excluded per
    team and counted in ``metadata`` so coverage stays auditable.
    """
    from .clutch_metrics import _enforce_pre_tournament_cutoff

    data_root = Path(data_root)
    if boxscore_payload is None:
        path = data_root / "raw" / "historical" / f"boxscores_{year}.json"
        if not path.exists():
            if strict:
                raise MinutesCoverageError(
                    f"No boxscores_{year}.json under {path.parent}. Run the "
                    f"EspnBoxscoreScraper for {year} before building minutes features."
                )
            return {}
        with open(path) as f:
            boxscore_payload = json.load(f)

    games = boxscore_payload.get("games", [])
    if not games:
        if strict:
            raise MinutesCoverageError(f"boxscores_{year}.json contains no games.")
        return {}

    _enforce_pre_tournament_cutoff(year, games)

    totals: Dict[Tuple[str, str], Dict] = {}
    games_ok = 0
    games_rejected = 0

    for game in games:
        ok_by_team = validate_boxscore_minutes(game)
        if not any(ok_by_team.values()):
            games_rejected += 1
            continue
        games_ok += 1

        for team in game.get("teams") or []:
            tid = team["team_id"]
            if not ok_by_team.get(tid):
                continue
            for p in team.get("players") or []:
                minutes = p.get("minutes")
                if minutes is None:
                    continue  # DNP — counts as neither a game played nor minutes
                key = (tid, p["athlete_id"])
                entry = totals.setdefault(
                    key,
                    {
                        "team_id": tid,
                        "athlete_id": p["athlete_id"],
                        "athlete_name": p.get("athlete_name"),
                        "games_played": 0,
                        "games_started": 0,
                        "total_minutes": 0.0,
                    },
                )
                entry["games_played"] += 1
                entry["games_started"] += 1 if p.get("started") else 0
                entry["total_minutes"] += float(minutes)

    if strict and len(totals) < _MIN_PLAUSIBLE_PLAYERS:
        raise MinutesCoverageError(
            f"Season {year}: only {len(totals)} distinct players across "
            f"{games_ok} accepted games ({games_rejected} rejected). Expected "
            f">= {_MIN_PLAUSIBLE_PLAYERS}. This is the signature of a parse "
            f"failure or an empty scrape, not a real season — refusing to "
            f"write a misleading artifact."
        )

    players_out = []
    for entry in totals.values():
        gp = entry["games_played"]
        entry["total_minutes"] = round(entry["total_minutes"], 2)
        entry["minutes_per_game"] = round(entry["total_minutes"] / gp, 2) if gp else 0.0
        players_out.append(entry)

    players_out.sort(key=lambda e: (-e["total_minutes"], e["athlete_id"]))

    return {
        "season": year,
        "players": players_out,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "espn_boxscore_html",
        "metadata": {"games_used": games_ok, "games_rejected": games_rejected},
    }
