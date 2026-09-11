"""Rebuild per-season roster features from pre-cutoff, per-game box scores.

Why this exists. ``cbbpy_rosters_{year}.json`` was scraped once, on
2026-02-21, for every season back to 2003. For any season already played, a
player's ``games_played`` therefore includes that team's tournament run, and
``warp = bpm * minute_share * games_played / 300`` turns that directly into a
measure of how far the team advanced (audit H5: r = +0.49 to +0.83 with rounds
won in every season 2011-2025). The per-game box scores in
``boxscores_{year}.json`` are dated, and every one of them precedes its season's
tournament, so the same statistics can be re-aggregated over exactly the games
a forecaster could have seen.

What this deliberately reuses. The per-player derivations -- BPM, WARP, box
RAPM, usage, TS%, eFG%, per-game rates -- are produced by
``CBBpyRosterScraper._build_payload``, the same function that produced the
contaminated files, fed with rows shaped the way cbbpy shaped them. The only
thing that changes is which games go in. That is the point: a rebuilt feature
that differs from the old one *only* by the tournament games is directly
comparable, and any remaining difference is the leak.

What this deliberately does not do. It does not invent ``is_transfer`` or
``eligibility_year``: cbbpy never populated them either (they are the constants
False and 1 in every shipped file), so the overlay indices built from them were
already inert. They stay as those constants.

Team identity. Box-score team ids are slugified ESPN display names
(``alabama_crimson_tide``); the pipeline joins on canonical ids (``alabama``).
The bridge is ``resolve_cbbpy_bridge`` against the season's full D1 universe,
which is the only way to keep ``alabama_state_hornets`` from claiming
``alabama`` -- see its docstring. Unbridged teams are dropped and counted, never
guessed.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ..normalize import load_d1_team_ids, resolve_cbbpy_bridge
from ..scrapers.cbbpy_rosters import CBBpyRosterScraper
from ..scrapers.espn_boxscore import validate_boxscore_minutes

logger = logging.getLogger(__name__)

SOURCE = "espn_boxscore_html"
DATA_TYPE = "pre_tournament_rosters"


def _split_made_attempted(value: object) -> Tuple[float, float]:
    """``"3-9"`` -> (3.0, 9.0). Anything unparseable -> (0, 0)."""
    if value is None:
        return 0.0, 0.0
    raw = str(value).strip()
    if "-" not in raw:
        return 0.0, 0.0
    made, attempted = raw.split("-", 1)
    try:
        return float(made), float(attempted)
    except ValueError:
        return 0.0, 0.0


def _to_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _tournament_start(year: int) -> Optional[date]:
    from ..season_calendar import TOURNAMENT_START_DATES

    return TOURNAMENT_START_DATES.get(year)


def boxscore_rows_for_season(
    payload: Dict,
    cutoff: date,
) -> Tuple[List[Dict], Dict[str, int], Dict[str, object]]:
    """Flatten a boxscores payload into cbbpy-shaped rows, pre-cutoff only.

    Returns ``(rows, games_by_raw_team, stats)`` where ``games_by_raw_team``
    counts distinct games per raw box-score team id (the bridge weight) and
    ``stats`` records what was kept and dropped.
    """
    rows: List[Dict] = []
    games_by_team: Dict[str, set] = defaultdict(set)
    stats: Dict[str, object] = {
        "games_total": 0,
        "games_on_or_after_cutoff": 0,
        "games_used": 0,
        "team_games_rejected_minutes": 0,
        "max_game_date": None,
    }
    max_date: Optional[date] = None

    for game in payload.get("games") or []:
        stats["games_total"] += 1  # type: ignore[operator]
        game_date_raw = game.get("game_date")
        try:
            game_date = date.fromisoformat(str(game_date_raw)[:10])
        except (TypeError, ValueError):
            continue
        if game_date >= cutoff:
            stats["games_on_or_after_cutoff"] += 1  # type: ignore[operator]
            continue
        game_id = str(game.get("game_id") or "")
        if not game_id:
            continue
        minutes_ok = validate_boxscore_minutes(game)
        used_any = False
        for team in game.get("teams") or []:
            raw_tid = str(team.get("team_id") or "")
            if not raw_tid:
                continue
            if not minutes_ok.get(raw_tid, False):
                stats["team_games_rejected_minutes"] += 1  # type: ignore[operator]
                continue
            display = str(team.get("team_display") or raw_tid)
            for p in team.get("players") or []:
                st = p.get("stats") or {}
                fgm, fga = _split_made_attempted(st.get("fieldGoalsMade-fieldGoalsAttempted"))
                tpm, _tpa = _split_made_attempted(
                    st.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted")
                )
                ftm, fta = _split_made_attempted(st.get("freeThrowsMade-freeThrowsAttempted"))
                del ftm
                rows.append(
                    {
                        # `team` is what _build_payload normalises into a team id;
                        # the raw slug is kept alongside so the bridge can rewrite it.
                        "team": raw_tid,
                        "team_display": display,
                        "player": p.get("athlete_name") or "",
                        "player_id": p.get("athlete_id") or "",
                        "game_id": game_id,
                        "position": p.get("position") or "G",
                        "starter": bool(p.get("started")),
                        "min": p.get("minutes") if p.get("minutes") is not None else st.get("minutes"),
                        "pts": _to_float(st.get("points")),
                        "reb": _to_float(st.get("rebounds")),
                        "ast": _to_float(st.get("assists")),
                        "stl": _to_float(st.get("steals")),
                        "blk": _to_float(st.get("blocks")),
                        "to": _to_float(st.get("turnovers")),
                        "fga": fga,
                        "fgm": fgm,
                        "3pm": tpm,
                        "fta": fta,
                        "oreb": _to_float(st.get("offensiveRebounds")),
                        "dreb": _to_float(st.get("defensiveRebounds")),
                        "pf": _to_float(st.get("fouls")),
                    }
                )
            games_by_team[raw_tid].add(game_id)
            used_any = True
        if used_any:
            stats["games_used"] += 1  # type: ignore[operator]
            max_date = game_date if max_date is None or game_date > max_date else max_date

    stats["max_game_date"] = max_date.isoformat() if max_date else None
    return rows, {t: len(g) for t, g in games_by_team.items()}, stats


def build_season_roster_payload(year: int, data_root: Path | str = "data") -> Dict:
    """Rebuild ``cbbpy_rosters``-schema roster features for ``year`` from box scores.

    Raises ``FileNotFoundError`` if the box-score file is absent and
    ``ValueError`` if the season has no tournament start date to cut off at.
    """
    data_root = Path(data_root)
    path = data_root / "raw" / "historical" / f"boxscores_{year}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run scripts/backfill_pbp_history.py boxscore stage first")
    cutoff = _tournament_start(year)
    if cutoff is None:
        raise ValueError(f"no TOURNAMENT_START_DATES entry for {year}; cannot define a pre-tournament cutoff")

    with open(path) as f:
        payload = json.load(f)

    rows, games_by_raw, stats = boxscore_rows_for_season(payload, cutoff)

    # Bridge raw ESPN slugs onto canonical ids against the full D1 universe.
    universe = load_d1_team_ids(year, data_root)
    bridge = resolve_cbbpy_bridge(games_by_raw, canonical_ids=universe, universe=universe) if universe else {}
    bridged_rows: List[Dict] = []
    dropped_raw: set = set()
    for r in rows:
        canonical = bridge.get(r["team"])
        if canonical is None:
            dropped_raw.add(r["team"])
            continue
        r = dict(r)
        r["team"] = canonical
        bridged_rows.append(r)

    scraper = CBBpyRosterScraper()
    built = scraper._build_payload(year, bridged_rows)

    out = {
        "year": year,
        "source": SOURCE,
        "data_type": DATA_TYPE,
        "cutoff_date": cutoff.isoformat(),
        "max_game_date": stats["max_game_date"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "teams": built.get("teams", []),
        "metadata": {
            **stats,
            "d1_universe_size": len(universe),
            "raw_team_ids": len(games_by_raw),
            "teams_bridged": len(set(bridge.values())),
            "raw_team_ids_dropped_unbridged": len(dropped_raw),
            "note": (
                "Per-player statistics aggregated over games dated strictly before cutoff_date, "
                "via CBBpyRosterScraper._build_payload -- identical formulas to cbbpy_rosters_{year}.json, "
                "different game window. is_transfer/eligibility_year are the same constants cbbpy shipped."
            ),
        },
    }
    logger.info(
        "%d: %d games used (%d on/after cutoff excluded), %d/%d raw team ids bridged to %d canonical, "
        "%d raw ids dropped",
        year,
        stats["games_used"],
        stats["games_on_or_after_cutoff"],
        len(bridge),
        len(games_by_raw),
        len(set(bridge.values())),
        len(dropped_raw),
    )
    return out


def write_season_roster_payload(year: int, data_root: Path | str = "data") -> Path:
    payload = build_season_roster_payload(year, data_root)
    out_path = Path(data_root) / "raw" / "historical" / f"rosters_boxscore_{year}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f)
    return out_path
