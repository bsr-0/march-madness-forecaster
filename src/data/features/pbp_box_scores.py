"""Team box-score stats derived from raw play-by-play.

Closes the ``three_pt_pct`` / ``ft_pct`` gap documented in FINDINGS.md and
``memory/next_steps_pretournament_player_data.md``: Torvik's player CSV
endpoint silently ignores its ``begin``/``end`` date params (confirmed
byte-identical output), so it can never supply *pre-tournament* shooting
splits for a past season. PBP can, because ``cbbpy_pbp.py`` bounds its scrape
window to pre-tournament dates by construction and every play carries its
own game date.

Consumes ``pbp_{season}.json`` and produces per-team-per-game box lines,
which aggregate to season shooting rates in the same
``{"season": ..., "teams": [...]}`` shape as ``torvik_{year}.json`` /
``clutch_features_{year}.json``.

**Derivation rules are empirical, not assumed** — each was checked against
real ESPN payloads (2026 season, game 401808890) before being written here,
because several are counterintuitive:

* ``play_type == "MadeFreeThrow"`` is ESPN's label for **missed** free
  throws too. Only ``scoring_play`` distinguishes made from missed. Trusting
  the type name would inflate FT% to 100%.
* ``shooting_play`` is ``True`` for free throws as well as field goals, so
  FGA must be filtered on ``points_attempted in (2, 3)`` — otherwise every
  free throw double-counts as a field-goal attempt.
* ``athlete_team`` is ``null`` on some plays (notably every ``Steal``), so
  team attribution falls back to ``home_away`` — which *is* populated there,
  and refers to the acting player's team (verified: the stealing team, not
  the team that lost the ball).
* ``Dead Ball Rebound`` rows are team rebounds with no athlete; they are
  counted separately from player rebounds (``team_reb``) since ESPN's own
  box score reports those apart from ORB/DRB.

**Validation** (game 401808890, TCU vs LSU New Orleans, 2025-11-04): 23 of
24 team totals reproduce ESPN's published box score *exactly* — PTS, FG,
3PT, FT, REB, OREB, DREB, TO, STL, BLK, PF for both teams, with derived
points (74/78) matching the final score. The lone exception is TCU assists:
ESPN's box score says 13, the PBP feed contains 12 ``Assisted by`` events
(no alternate phrasing, no unattributed play — checked). That is an
inconsistency between ESPN's own two feeds, not a parsing bug: box scores
come from the official scorer, PBP text is entered separately, and the two
don't always reconcile. **Treat PBP-derived assists as approximate**;
shooting splits (the reason this module exists) reconcile exactly.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from ..normalize import load_d1_team_ids, resolve_cbbpy_bridge

logger = logging.getLogger(__name__)

_ASSIST_MARKER = "Assisted by"


@dataclass
class TeamGameBox:
    """One team's box-score line for a single game, derived from PBP."""

    game_id: str
    team_id: str
    opponent_id: str
    is_home: bool

    fgm: int = 0
    fga: int = 0
    fg3m: int = 0
    fg3a: int = 0
    ftm: int = 0
    fta: int = 0
    orb: int = 0
    drb: int = 0
    team_reb: int = 0
    ast: int = 0
    stl: int = 0
    blk: int = 0
    tov: int = 0
    pf: int = 0

    @property
    def pts(self) -> int:
        """Points implied by the shot counts (2s + 3s + FTs)."""
        return 2 * (self.fgm - self.fg3m) + 3 * self.fg3m + self.ftm

    def as_dict(self) -> Dict:
        d = {
            "game_id": self.game_id,
            "team_id": self.team_id,
            "opponent_id": self.opponent_id,
            "is_home": self.is_home,
            "pts": self.pts,
        }
        for f in ("fgm fga fg3m fg3a ftm fta orb drb team_reb ast stl blk tov pf").split():
            d[f] = getattr(self, f)
        return d


def _play_team(play: Dict, home_team: str, away_team: str) -> Optional[str]:
    """Which team a play belongs to.

    Prefers the explicit ``athlete_team``; falls back to ``home_away``, which
    stays populated on plays where the athlete block is missing (Steal, some
    rebounds). Returns None for plays belonging to neither side (timeouts,
    period boundaries).
    """
    team = play.get("athlete_team")
    if team:
        return team
    side = play.get("home_away")
    if side == "home":
        return home_team
    if side == "away":
        return away_team
    return None


def derive_game_box_scores(game_payload: Dict) -> Dict[str, TeamGameBox]:
    """Derive both teams' box-score lines from one game's PBP payload.

    Returns ``{team_raw_id: TeamGameBox}``. Empty if the game lacks the team
    attribution needed to split plays between sides.
    """
    home = game_payload.get("home_team_raw")
    away = game_payload.get("away_team_raw")
    game_id = game_payload.get("game_id", "")
    if not home or not away:
        return {}

    boxes = {
        home: TeamGameBox(game_id=game_id, team_id=home, opponent_id=away, is_home=True),
        away: TeamGameBox(game_id=game_id, team_id=away, opponent_id=home, is_home=False),
    }

    for play in game_payload.get("plays", []):
        team = _play_team(play, home, away)
        if team not in boxes:
            continue
        box = boxes[team]

        play_type = play.get("play_type") or ""
        scoring = bool(play.get("scoring_play"))
        shooting = bool(play.get("shooting_play"))
        pts_att = play.get("points_attempted")
        text = play.get("text") or ""

        # --- Shooting. Free throws are shooting_play=True too, hence the
        # explicit points_attempted split (see module docstring).
        if shooting and pts_att == 1:
            box.fta += 1
            if scoring:
                box.ftm += 1
        elif shooting and pts_att in (2, 3):
            box.fga += 1
            if scoring:
                box.fgm += 1
            if pts_att == 3:
                box.fg3a += 1
                if scoring:
                    box.fg3m += 1

        # --- Rebounds. Dead-ball/team rebounds tracked apart from player ones.
        if play_type == "Offensive Rebound":
            box.orb += 1
        elif play_type == "Defensive Rebound":
            box.drb += 1
        elif play_type == "Dead Ball Rebound":
            box.team_reb += 1

        # --- Everything else.
        if _ASSIST_MARKER in text:
            box.ast += 1
        if play_type == "Steal":
            box.stl += 1
        if play_type == "Block Shot":
            box.blk += 1
        if "Turnover" in play_type:
            box.tov += 1
        if "Foul" in play_type:
            box.pf += 1

    return boxes


def aggregate_team_season_box(boxes: List[TeamGameBox]) -> Dict[str, Dict]:
    """Group per-game box lines by raw team id into season shooting rates."""
    by_team: Dict[str, List[TeamGameBox]] = {}
    for b in boxes:
        by_team.setdefault(b.team_id, []).append(b)

    out: Dict[str, Dict] = {}
    for team_id, games in by_team.items():
        tot = {
            f: sum(getattr(g, f) for g in games) for f in "fgm fga fg3m fg3a ftm fta orb drb ast stl blk tov pf".split()
        }
        n = len(games)
        out[team_id] = {
            "games_with_box_data": n,
            "three_pt_pct": (tot["fg3m"] / tot["fg3a"]) if tot["fg3a"] else None,
            "ft_pct": (tot["ftm"] / tot["fta"]) if tot["fta"] else None,
            "fg_pct": (tot["fgm"] / tot["fga"]) if tot["fga"] else None,
            "three_pt_rate": (tot["fg3a"] / tot["fga"]) if tot["fga"] else None,
            "effective_fg_pct": (((tot["fgm"] + 0.5 * tot["fg3m"]) / tot["fga"]) if tot["fga"] else None),
            "ast_per_game": tot["ast"] / n,
            "stl_per_game": tot["stl"] / n,
            "blk_per_game": tot["blk"] / n,
            "tov_per_game": tot["tov"] / n,
            "pf_per_game": tot["pf"] / n,
            "orb_per_game": tot["orb"] / n,
            "drb_per_game": tot["drb"] / n,
        }
    return out


def build_season_shooting_features(
    year: int,
    data_root,
    *,
    pbp_payload: Optional[Dict] = None,
) -> Dict:
    """End-to-end: load pbp_{year}.json, derive box scores, bridge team IDs, aggregate.

    Mirrors ``clutch_metrics.build_season_clutch_features`` exactly -- same
    leakage re-check, same ``resolve_cbbpy_bridge`` against the full D1
    universe, same output shape. Does not write to disk; callers write the
    returned dict.
    """
    from .clutch_metrics import _enforce_pre_tournament_cutoff, _load_canonical_ids

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

    all_boxes: List[TeamGameBox] = []
    for game_payload in games:
        all_boxes.extend(derive_game_box_scores(game_payload).values())

    if not all_boxes:
        return {}

    raw_stats = aggregate_team_season_box(all_boxes)

    canonical_ids = _load_canonical_ids(year, data_root)
    weighted_ids = {tid: s["games_with_box_data"] for tid, s in raw_stats.items()}
    bridge = resolve_cbbpy_bridge(weighted_ids, canonical_ids, universe=load_d1_team_ids(year, data_root))

    teams = [{"team_id": canonical, **raw_stats[raw]} for raw, canonical in bridge.items()]

    return {
        "season": year,
        "teams": teams,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "espn_pbp_derived_box",
    }
