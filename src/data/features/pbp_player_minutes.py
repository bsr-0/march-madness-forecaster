"""Player minutes derived from play-by-play substitution events.

Closes the ``returning_minutes_pct`` / ``freshman_minutes_pct`` provenance
caveat documented in FINDINGS.md and
``memory/next_steps_pretournament_player_data.md``: every
``cbbpy_rosters_{year}.json`` for 2010-2025 was built from cbbpy's
season-endpoint fast path, which returns the whole season in one call and so
carries **post-tournament** minutes. Those files all share one scrape date
(2026-02-21) and cannot be retroactively bounded. PBP can be, because
``cbbpy_pbp.py`` only walks pre-tournament dates.

ESPN's PBP carries explicit substitution events -- ``Substitution`` is in
fact the single most common play type (~7,600 across 50 games) -- with text
of the form ``"{Player} subbing in for {Team}"`` / ``"subbing out for"``,
each with ``athlete_id``, ``period`` and ``clock``. That's enough to
reconstruct on-court intervals per player.

**Starters are inferred, not stated.** No event marks the opening five, so a
player is treated as a starter if their first substitution event is a
*sub-out*, or if they record any non-substitution play before their first
*sub-in*. This is the standard reconstruction and is exact whenever the feed
is complete.

**Known limitation**: minutes are only as good as the substitution feed. If
a game is missing sub events (older seasons are likelier to be sparse), the
reconstruction silently under- or over-counts rather than failing loudly, so
``validate_game_minutes`` is provided to compare a game's reconstructed
totals against the expected team-minutes budget (5 players x game length).
Callers doing a historical backfill should check that before trusting a
season, and ``build_season_minutes_features`` drops games that fail it.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_REGULATION_PERIOD_SECONDS = 1200  # 20-minute halves
_OT_PERIOD_SECONDS = 300  # 5-minute overtimes
_REGULATION_PERIODS = 2

_SUB_IN_RE = re.compile(r"subbing in", re.I)
_SUB_OUT_RE = re.compile(r"subbing out", re.I)

# A game's reconstructed team minutes should land near 5 * game_minutes.
# Tolerance is deliberately loose -- it's a sanity gate against a broken or
# absent substitution feed, not a precision check.
_MINUTES_TOLERANCE_FRAC = 0.10


def period_length_seconds(period: int) -> int:
    return _REGULATION_PERIOD_SECONDS if period <= _REGULATION_PERIODS else _OT_PERIOD_SECONDS


def elapsed_seconds(period: int, seconds_remaining: float) -> float:
    """Absolute seconds elapsed since tip-off."""
    prior = sum(period_length_seconds(p) for p in range(1, period))
    return prior + (period_length_seconds(period) - seconds_remaining)


@dataclass
class PlayerGameMinutes:
    game_id: str
    team_id: str
    athlete_id: str
    athlete_name: Optional[str]
    seconds: float = 0.0
    started: bool = False

    @property
    def minutes(self) -> float:
        return self.seconds / 60.0


def _game_length_seconds(plays: List[Dict]) -> float:
    """Total game length, accounting for overtime periods."""
    if not plays:
        return 0.0
    max_period = max(int(p["period"]) for p in plays)
    return float(sum(period_length_seconds(p) for p in range(1, max_period + 1)))


def derive_game_player_minutes(game_payload: Dict) -> List[PlayerGameMinutes]:
    """Reconstruct per-player minutes for one game from substitution events."""
    plays = game_payload.get("plays") or []
    game_id = game_payload.get("game_id", "")
    if not plays:
        return []

    game_end = _game_length_seconds(plays)

    ordered = sorted(plays, key=lambda p: (p["period"], -p["seconds_remaining"]))

    # (athlete_id) -> events [(elapsed, kind)] where kind in {"in","out","play"}
    events: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
    identity: Dict[str, Tuple[Optional[str], Optional[str]]] = {}

    for p in ordered:
        aid = p.get("athlete_id")
        if not aid:
            continue
        t = elapsed_seconds(int(p["period"]), float(p["seconds_remaining"]))
        text = p.get("text") or ""
        team = p.get("athlete_team")
        if aid not in identity or (identity[aid][1] is None and team):
            identity[aid] = (p.get("athlete_name"), team)

        if p.get("play_type") == "Substitution":
            if _SUB_IN_RE.search(text):
                events[aid].append((t, "in"))
            elif _SUB_OUT_RE.search(text):
                events[aid].append((t, "out"))
        else:
            events[aid].append((t, "play"))

    results: List[PlayerGameMinutes] = []
    for aid, evs in events.items():
        evs.sort(key=lambda e: e[0])
        name, team = identity.get(aid, (None, None))
        if not team:
            continue

        # Starter inference: first sub event is an "out", or a real play
        # occurs before any "in".
        started = False
        for t, kind in evs:
            if kind == "in":
                break
            if kind == "out" or kind == "play":
                started = True
                break

        seconds = 0.0
        on_court = started
        last_in = 0.0 if started else None
        for t, kind in evs:
            if kind == "in":
                if not on_court:
                    on_court = True
                    last_in = t
            elif kind == "out":
                if on_court and last_in is not None:
                    seconds += max(0.0, t - last_in)
                on_court = False
                last_in = None
        if on_court and last_in is not None:
            seconds += max(0.0, game_end - last_in)

        results.append(
            PlayerGameMinutes(
                game_id=game_id,
                team_id=team,
                athlete_id=str(aid),
                athlete_name=name,
                seconds=seconds,
                started=started,
            )
        )
    return results


def validate_game_minutes(game_payload: Dict, players: List[PlayerGameMinutes]) -> Dict[str, bool]:
    """Per-team check that reconstructed minutes land near the 5-on-court budget.

    Returns ``{team_id: is_plausible}``. A team whose substitution feed is
    missing or partial will land well below ``5 * game_minutes`` and flag
    False here rather than silently producing wrong roster shares.
    """
    plays = game_payload.get("plays") or []
    expected = 5.0 * (_game_length_seconds(plays) / 60.0)
    if expected <= 0:
        return {}

    by_team: Dict[str, float] = defaultdict(float)
    for p in players:
        by_team[p.team_id] += p.minutes

    return {team: abs(total - expected) <= expected * _MINUTES_TOLERANCE_FRAC for team, total in by_team.items()}


def build_season_minutes_features(
    year: int,
    data_root,
    *,
    pbp_payload: Optional[Dict] = None,
) -> Dict:
    """Aggregate per-player season minutes from a season's PBP.

    Output is keyed by *raw* (cbbpy-style) team id and keeps athlete ids, so
    a downstream consumer can join it to roster/eligibility data to rebuild
    ``returning_minutes_pct`` / ``freshman_minutes_pct`` on a clean
    pre-tournament basis. Team-ID bridging to canonical ids is deliberately
    left to that consumer, since the join it needs is player-level.

    Games whose substitution feed fails ``validate_game_minutes`` are
    excluded, and counted in ``metadata`` so coverage is auditable.
    """
    from .clutch_metrics import _enforce_pre_tournament_cutoff

    data_root = Path(data_root)
    if pbp_payload is None:
        pbp_path = data_root / "raw" / "historical" / f"pbp_{year}.json"
        if not pbp_path.exists():
            return {}
        with open(pbp_path) as f:
            pbp_payload = json.load(f)

    games = pbp_payload.get("games", [])
    if not games:
        return {}

    _enforce_pre_tournament_cutoff(year, games)

    totals: Dict[Tuple[str, str], Dict] = {}
    games_ok = 0
    games_rejected = 0

    for game_payload in games:
        players = derive_game_player_minutes(game_payload)
        if not players:
            games_rejected += 1
            continue
        ok_by_team = validate_game_minutes(game_payload, players)
        if not any(ok_by_team.values()):
            games_rejected += 1
            continue
        games_ok += 1

        for p in players:
            if not ok_by_team.get(p.team_id):
                continue
            key = (p.team_id, p.athlete_id)
            entry = totals.setdefault(
                key,
                {
                    "team_id": p.team_id,
                    "athlete_id": p.athlete_id,
                    "athlete_name": p.athlete_name,
                    "games_played": 0,
                    "games_started": 0,
                    "total_minutes": 0.0,
                },
            )
            entry["games_played"] += 1
            entry["games_started"] += 1 if p.started else 0
            entry["total_minutes"] += p.minutes

    if not totals:
        return {}

    players_out = []
    for entry in totals.values():
        gp = entry["games_played"]
        players_out.append(
            {
                **entry,
                "total_minutes": round(entry["total_minutes"], 2),
                "minutes_per_game": round(entry["total_minutes"] / gp, 2) if gp else 0.0,
            }
        )

    return {
        "season": year,
        "players": players_out,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "espn_pbp_substitutions",
        "metadata": {
            "games_used": games_ok,
            "games_rejected": games_rejected,
        },
    }
