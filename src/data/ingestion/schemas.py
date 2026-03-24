"""Typed data contracts for the Historical Game Pipeline.

Defines ``IngestionGameRecord`` — the canonical game-level record produced by
every game fetcher.  All providers must populate this exact schema before data
reaches the rest of the pipeline.

``IngestionGameRecord`` is intentionally separate from
``proprietary_metrics.GameRecord``, which is a *team-perspective* record used
inside the ML engine.  Use ``IngestionGameRecord.to_team_game_rows()`` to
convert to the two team-perspective dicts that ``team_games_to_game_records()``
expects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


def _to_float(v) -> float:
    if v is None:
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


@dataclass
class IngestionGameRecord:
    """Canonical single-game record at the ingestion layer boundary.

    Stores data from the *home/away* perspective.  Fields prefixed with
    ``home_`` / ``away_`` hold per-team box-score stats.  Box-score fields
    default to 0.0 when the source provider does not supply them (e.g. the
    ESPN scoreboard for in-progress games, or score-only fallback paths).

    Invariants enforced by ``validate()``:
    - ``game_id`` is non-empty
    - ``date`` is ISO format YYYY-MM-DD
    - Scores are non-negative and at least one is non-zero
    """

    game_id: str
    date: str       # YYYY-MM-DD
    season: int

    home_team_id: str
    home_team_name: str
    away_team_id: str
    away_team_name: str

    home_score: int
    away_score: int

    # ── Box-score fields — 0.0 when not available ──────────────────────────
    home_fgm: float = 0.0
    home_fga: float = 0.0
    home_fg3m: float = 0.0
    home_fg3a: float = 0.0
    home_ftm: float = 0.0
    home_fta: float = 0.0
    home_orb: float = 0.0
    home_drb: float = 0.0
    home_ast: float = 0.0
    home_stl: float = 0.0
    home_blk: float = 0.0
    home_tov: float = 0.0
    home_pf: float = 0.0

    away_fgm: float = 0.0
    away_fga: float = 0.0
    away_fg3m: float = 0.0
    away_fg3a: float = 0.0
    away_ftm: float = 0.0
    away_fta: float = 0.0
    away_orb: float = 0.0
    away_drb: float = 0.0
    away_ast: float = 0.0
    away_stl: float = 0.0
    away_blk: float = 0.0
    away_tov: float = 0.0
    away_pf: float = 0.0

    # ── Game context ───────────────────────────────────────────────────────
    overtime: bool = False
    neutral_site: bool = False
    conference_game: Optional[bool] = None

    # Which provider produced this record (for audit / fallback logging)
    provider: str = ""

    # ── Validation ─────────────────────────────────────────────────────────

    def validate(self) -> List[str]:
        """Return a list of error strings; empty list means the record is valid."""
        errors: List[str] = []
        if not self.game_id:
            errors.append("game_id is empty")
        if not self.date or len(self.date) < 10 or self.date[4:5] != "-":
            errors.append(f"date '{self.date}' is not ISO format YYYY-MM-DD")
        if self.home_score < 0 or self.away_score < 0:
            errors.append(f"negative score: home={self.home_score}, away={self.away_score}")
        if self.home_score == 0 and self.away_score == 0:
            errors.append("both scores are 0 — game is likely incomplete or a forfeit")
        if not self.home_team_id:
            errors.append("home_team_id is empty")
        if not self.away_team_id:
            errors.append("away_team_id is empty")
        return errors

    # ── Conversion ─────────────────────────────────────────────────────────

    def to_team_game_rows(self) -> List[Dict]:
        """Convert to two team-perspective dicts.

        Output schema is compatible with ``team_games_to_game_records()``::

            {
                "game_id", "date", "season",
                "team_id", "team_name", "opponent_id", "opponent_name",
                "team_score", "opponent_score", "possessions",
                "fgm", "fga", "fg3m", "fg3a", "fta",
                "turnovers", "orb", "drb",
            }

        Possessions are estimated via the standard Dean Oliver formula:
        ``poss ≈ FGA - ORB + TOV + 0.475 × FTA``.
        If that yields zero (box scores unavailable), a score-average fallback
        is used so downstream calculations are never starved of a possession
        estimate.
        """
        def _poss(fga: float, orb: float, tov: float, fta: float) -> float:
            p = fga - orb + tov + 0.475 * fta
            return max(p, 0.0)

        home_poss = _poss(self.home_fga, self.home_orb, self.home_tov, self.home_fta)
        away_poss = _poss(self.away_fga, self.away_orb, self.away_tov, self.away_fta)
        if home_poss <= 0 and away_poss <= 0:
            fallback = max((self.home_score + self.away_score) / 2.0, 30.0)
            home_poss = away_poss = fallback

        base = {"game_id": self.game_id, "date": self.date, "season": self.season, "overtime": self.overtime}

        home_row = {
            **base,
            "team_id": self.home_team_id,
            "team_name": self.home_team_name,
            "opponent_id": self.away_team_id,
            "opponent_name": self.away_team_name,
            "team_score": self.home_score,
            "opponent_score": self.away_score,
            "possessions": home_poss,
            "fgm": self.home_fgm,
            "fga": self.home_fga,
            "fg3m": self.home_fg3m,
            "fg3a": self.home_fg3a,
            "fta": self.home_fta,
            "turnovers": self.home_tov,
            "orb": self.home_orb,
            "drb": self.home_drb,
        }
        away_row = {
            **base,
            "team_id": self.away_team_id,
            "team_name": self.away_team_name,
            "opponent_id": self.home_team_id,
            "opponent_name": self.home_team_name,
            "team_score": self.away_score,
            "opponent_score": self.home_score,
            "possessions": away_poss,
            "fgm": self.away_fgm,
            "fga": self.away_fga,
            "fg3m": self.away_fg3m,
            "fg3a": self.away_fg3a,
            "fta": self.away_fta,
            "turnovers": self.away_tov,
            "orb": self.away_orb,
            "drb": self.away_drb,
        }
        return [home_row, away_row]

    def to_game_row(self) -> Dict:
        """Convert to a single game-level dict (team1/team2 perspective).

        Used by ``HistoricalDataPipeline.run()`` to build the ``games`` list
        inside the season JSON artifact.
        """
        return {
            "game_id": self.game_id,
            "season": self.season,
            "date": self.date,
            "team1_id": self.home_team_id,
            "team1_name": self.home_team_name,
            "team2_id": self.away_team_id,
            "team2_name": self.away_team_name,
            "team1_score": self.home_score,
            "team2_score": self.away_score,
            "overtime": self.overtime,
        }
