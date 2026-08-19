"""Blown-lead / clutch team-season features built from raw play-by-play.

Consumes ``pbp_{season}.json`` (written by
``src/data/scrapers/cbbpy_pbp.py``) and produces
``clutch_features_{season}.json`` in the same ``{"season": ..., "teams": [...]}``
shape as ``torvik_{year}.json``, so ``src/cli/pool_cmds.py`` can merge it
alongside the existing Torvik four-factors merge with minimal new code.

Team-ID handling mirrors ``src/prediction/elo_probabilities.py``: raw PBP
rows carry cbbpy-style team references, not canonical team IDs, so this
module bridges them with ``resolve_cbbpy_bridge`` against the full D1
universe rather than a naive prefix match (see FINDINGS.md's "cbbpy team-ID
bridge prefix collisions" defect — a bespoke bridge here would reproduce it).

Leakage guard: ``cbbpy_pbp.py`` already bounds its scrape window to
pre-tournament dates by default, but this module re-checks every game's date
against ``TOURNAMENT_START_DATES`` before aggregating, rather than trusting
that the input file was produced correctly — belt-and-suspenders, consistent
with how ``torvik.py`` validates both the live-scrape guard and the
cached-data guard separately.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from ..normalize import load_d1_team_ids, resolve_cbbpy_bridge

logger = logging.getLogger(__name__)

# Margin snapshots taken at these many seconds remaining in regulation
# (2nd half only — first-half snapshots aren't "late game" in any sense).
_LATE_GAME_MARKS = {"10min": 600, "5min": 300, "2min": 120, "1min": 60}

_BLOWN_LEAD_THRESHOLDS = (5, 10, 15, 20)

_REGULATION_PERIODS = 2  # cbbpy period numbering: 1, 2 = halves; 3+ = OT


@dataclass
class GameClutchRecord:
    """One team-side row summarizing a single game's margin trajectory."""

    game_id: str
    team_id: str
    opponent_id: str
    won: bool
    final_margin: float
    max_lead: float
    max_deficit: float
    largest_lead_blown: float
    margin_at: Dict[str, Optional[float]] = field(default_factory=dict)


def compute_game_margin_trajectory(plays: List[Dict], home_team_id: str, away_team_id: str) -> Optional[tuple]:
    """Reconstruct home/away margin trajectory from normalized play rows.

    ``plays`` rows must already be in the canonical shape produced by
    ``cbbpy_pbp.CBBpyPbpScraper._normalize_play_row``: period,
    seconds_remaining, home_score, away_score.

    Returns ``(home_record_fields, away_record_fields)`` dicts with
    final_margin/max_lead/max_deficit/largest_lead_blown/margin_at (all from
    the home team's perspective; the away side is the mirror), or None if
    there are too few plays to compute anything.
    """
    if not plays:
        return None

    ordered = sorted(
        plays,
        key=lambda p: (p["period"], -p["seconds_remaining"]),
    )

    max_lead = float("-inf")
    max_deficit = float("-inf")
    margin_at: Dict[str, Optional[float]] = dict.fromkeys(_LATE_GAME_MARKS, None)

    for play in ordered:
        margin = play["home_score"] - play["away_score"]
        max_lead = max(max_lead, margin)
        max_deficit = max(max_deficit, -margin)

        if play["period"] == _REGULATION_PERIODS:
            for label, secs in _LATE_GAME_MARKS.items():
                if margin_at[label] is None and play["seconds_remaining"] <= secs:
                    margin_at[label] = margin

    final_margin = ordered[-1]["home_score"] - ordered[-1]["away_score"]
    if max_lead == float("-inf"):
        return None

    # "Blown lead" from the home team's perspective: held a lead of at least
    # `max_lead` at some point but the game didn't end at least that far
    # ahead (loosely: they gave back the peak advantage).
    home_largest_lead_blown = max(0.0, max_lead - max(final_margin, 0.0)) if max_lead > 0 else 0.0
    away_largest_lead_blown = max(0.0, max_deficit - max(-final_margin, 0.0)) if max_deficit > 0 else 0.0

    home_fields = {
        "final_margin": final_margin,
        "max_lead": max_lead,
        "max_deficit": max_deficit,
        "largest_lead_blown": home_largest_lead_blown,
        "margin_at": dict(margin_at),
    }
    away_fields = {
        "final_margin": -final_margin,
        "max_lead": max_deficit,
        "max_deficit": max_lead,
        "largest_lead_blown": away_largest_lead_blown,
        "margin_at": {k: (None if v is None else -v) for k, v in margin_at.items()},
    }
    return home_fields, away_fields


def build_game_clutch_records(game_payload: Dict) -> List[GameClutchRecord]:
    """Build both team-side ``GameClutchRecord``s for one game payload.

    ``game_payload`` is one entry from ``pbp_{season}.json``'s ``games`` list
    (``game_id``, ``home_team_raw``, ``away_team_raw``, ``plays``).
    """
    game_id = game_payload.get("game_id", "")
    home_raw = game_payload.get("home_team_raw")
    away_raw = game_payload.get("away_team_raw")
    if not home_raw or not away_raw:
        return []

    trajectory = compute_game_margin_trajectory(game_payload.get("plays", []), home_raw, away_raw)
    if trajectory is None:
        return []
    home_fields, away_fields = trajectory

    records = []
    for team_raw, opp_raw, fields in (
        (home_raw, away_raw, home_fields),
        (away_raw, home_raw, away_fields),
    ):
        records.append(
            GameClutchRecord(
                game_id=game_id,
                team_id=team_raw,
                opponent_id=opp_raw,
                won=fields["final_margin"] > 0,
                final_margin=fields["final_margin"],
                max_lead=fields["max_lead"],
                max_deficit=fields["max_deficit"],
                largest_lead_blown=fields["largest_lead_blown"],
                margin_at=fields["margin_at"],
            )
        )
    return records


def aggregate_team_season_clutch(records: List[GameClutchRecord]) -> Dict[str, Dict]:
    """Group per-game records by (raw) team_id and compute season clutch rates."""
    by_team: Dict[str, List[GameClutchRecord]] = {}
    for r in records:
        by_team.setdefault(r.team_id, []).append(r)

    result: Dict[str, Dict] = {}
    for team_id, games in by_team.items():
        n = len(games)
        blown_rates = {}
        for threshold in _BLOWN_LEAD_THRESHOLDS:
            had_lead = [g for g in games if g.max_lead >= threshold]
            blown = [g for g in had_lead if g.largest_lead_blown >= threshold]
            blown_rates[f"blown_{threshold}pt_lead_rate"] = len(blown) / len(had_lead) if had_lead else None

        close_games = [g for g in games if abs(g.final_margin) <= 5]
        close_wins = [g for g in close_games if g.won]

        win_rates_when_leading = {}
        for label in _LATE_GAME_MARKS:
            leading = [g for g in games if (g.margin_at.get(label) or 0) > 0]
            wins = [g for g in leading if g.won]
            win_rates_when_leading[f"win_rate_when_leading_at_{label}"] = len(wins) / len(leading) if leading else None

        deltas = [g.final_margin - g.margin_at["5min"] for g in games if g.margin_at.get("5min") is not None]

        result[team_id] = {
            "games_with_clutch_data": n,
            "close_game_win_rate": (len(close_wins) / len(close_games) if close_games else None),
            "late_game_margin_delta_mean": (sum(deltas) / len(deltas) if deltas else None),
            **blown_rates,
            **win_rates_when_leading,
        }
    return result


def build_season_clutch_features(
    year: int,
    data_root,
    *,
    pbp_payload: Optional[Dict] = None,
) -> Dict:
    """End-to-end: load pbp_{year}.json (or use pbp_payload), bridge team IDs,
    aggregate, and return the ``clutch_features_{year}.json`` payload.

    Does not write to disk — callers write the returned dict, mirroring how
    ``cbbpy_pbp.CBBpyPbpScraper.fetch_season_pbp`` owns its own write.
    """
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

    all_records: List[GameClutchRecord] = []
    for game_payload in games:
        all_records.extend(build_game_clutch_records(game_payload))

    if not all_records:
        return {}

    raw_team_stats = aggregate_team_season_clutch(all_records)

    # Bridge raw cbbpy-style team IDs to canonical IDs, weighting collisions
    # by games played — same idiom as elo_probabilities.load_elo_barthag.
    canonical_ids = _load_canonical_ids(year, data_root)
    weighted_ids = {tid: stats["games_with_clutch_data"] for tid, stats in raw_team_stats.items()}
    bridge = resolve_cbbpy_bridge(weighted_ids, canonical_ids, universe=load_d1_team_ids(year, data_root))

    teams = [{"team_id": canonical, **raw_team_stats[raw]} for raw, canonical in bridge.items()]

    return {
        "season": year,
        "teams": teams,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "cbbpy_pbp",
    }


def _load_canonical_ids(year: int, data_root: Path):
    """Canonical ID universe to bridge onto: this season's tournament seeds if
    available, else the full D1 set (bridging onto D1 is always safe, just
    wasteful for teams that never made the tournament).
    """
    seeds_path = data_root / "raw" / "historical" / f"tournament_context_{year}.json"
    if seeds_path.exists():
        try:
            with open(seeds_path) as f:
                ctx = json.load(f)
            # ctx["seeds"] is {"season": ..., "teams": [{"team_id": ..., "seed": ...}, ...]},
            # not a flat {team_id: seed} dict -- confirmed against a real
            # tournament_context_2025.json during the live pilot run.
            seed_teams = (ctx.get("seeds") or {}).get("teams")
            if seed_teams:
                ids = frozenset(t["team_id"] for t in seed_teams if t.get("team_id"))
                if ids:
                    return ids
        except (json.JSONDecodeError, OSError):
            pass
    return load_d1_team_ids(year, data_root)


def _enforce_pre_tournament_cutoff(year: int, games: List[Dict]) -> None:
    """Raise if any game in the payload falls on/after the tournament start.

    Re-checks what ``cbbpy_pbp.py`` should have already bounded at scrape
    time — see this module's docstring for why that's not redundant.
    """
    try:
        from ...pipeline.config import TOURNAMENT_START_DATES
    except ImportError:
        return

    cutoff = TOURNAMENT_START_DATES.get(year)
    if cutoff is None:
        return

    for g in games:
        game_date = g.get("game_date")
        if not game_date:
            continue
        try:
            gd = date.fromisoformat(game_date[:10])
        except ValueError:
            continue
        if gd >= cutoff:
            from ...exceptions import LeakageError

            raise LeakageError(
                f"pbp_{year}.json contains a game on {game_date}, on/after "
                f"tournament start {cutoff.isoformat()}. Aggregating it into "
                f"pre-tournament clutch features would leak tournament "
                f"results — re-scrape with the default (pre-tournament-only) "
                f"window."
            )
