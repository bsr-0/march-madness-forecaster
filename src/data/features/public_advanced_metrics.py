"""Derive KenPom-like and Torvik-like team metrics from public data."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class TeamGameRow:
    team_id: str
    team_name: str
    opponent_id: str
    points: float
    opp_points: float
    possessions: float
    fga: float = 0.0
    fgm: float = 0.0
    fg3a: float = 0.0
    fg3m: float = 0.0
    fta: float = 0.0
    tov: float = 0.0
    orb: float = 0.0
    opp_drb: float = 0.0
    opp_fga: float = 0.0
    opp_fgm: float = 0.0
    opp_fg3a: float = 0.0
    opp_fg3m: float = 0.0
    opp_fta: float = 0.0
    opp_tov: float = 0.0
    opp_orb: float = 0.0
    drb: float = 0.0


class PublicAdvancedMetricsBuilder:
    """Build adjusted team metrics from public game/team records."""

    def build(self, team_metric_records: List[Dict], teams: Optional[List[Dict]] = None) -> Dict:
        rows = self._to_game_rows(team_metric_records)
        if not rows:
            return {"teams": []}

        by_team: Dict[str, List[TeamGameRow]] = defaultdict(list)
        for row in rows:
            by_team[row.team_id].append(row)

        raw_off = {}
        raw_def = {}
        tempo = {}
        names = {}
        for tid, games in by_team.items():
            poss = sum(g.possessions for g in games)
            pts = sum(g.points for g in games)
            opp_pts = sum(g.opp_points for g in games)
            raw_off[tid] = 100.0 * pts / max(poss, 1.0)
            raw_def[tid] = 100.0 * opp_pts / max(poss, 1.0)
            tempo[tid] = poss / max(len(games), 1)
            names[tid] = games[0].team_name or tid

        league_off = sum(raw_off.values()) / max(len(raw_off), 1)
        league_def = sum(raw_def.values()) / max(len(raw_def), 1)
        adj_off = dict(raw_off)
        adj_def = dict(raw_def)

        for _ in range(12):
            next_off = {}
            next_def = {}
            for tid, games in by_team.items():
                off_samples = []
                def_samples = []
                for g in games:
                    opp_def = adj_def.get(g.opponent_id, league_def)
                    opp_off = adj_off.get(g.opponent_id, league_off)
                    game_off = 100.0 * g.points / max(g.possessions, 1.0)
                    game_def = 100.0 * g.opp_points / max(g.possessions, 1.0)
                    off_samples.append(game_off * league_def / max(opp_def, 1e-6))
                    def_samples.append(game_def * league_off / max(opp_off, 1e-6))
                next_off[tid] = sum(off_samples) / max(len(off_samples), 1)
                next_def[tid] = sum(def_samples) / max(len(def_samples), 1)
            adj_off = next_off
            adj_def = next_def

        teams_out = []
        for tid, games in by_team.items():
            ff = self._four_factors(games)
            wins = sum(1 for g in games if g.points > g.opp_points)
            losses = len(games) - wins
            adj_em = adj_off[tid] - adj_def[tid]
            sos = sum((adj_off.get(g.opponent_id, league_off) - adj_def.get(g.opponent_id, league_def)) for g in games) / max(len(games), 1)

            teams_out.append(
                {
                    "team_id": tid,
                    "name": names.get(tid, tid),
                    "conference": self._infer_conference(tid, teams),
                    "adj_efficiency_margin": adj_em,
                    "adj_offensive_efficiency": adj_off[tid],
                    "adj_defensive_efficiency": adj_def[tid],
                    "adj_tempo": tempo[tid],
                    "overall_rank": 0,
                    "offensive_rank": 0,
                    "defensive_rank": 0,
                    "luck": self._luck(games, adj_em),
                    "sos_adj_em": sos,
                    "sos_opp_o": sum(adj_off.get(g.opponent_id, league_off) for g in games) / max(len(games), 1),
                    "sos_opp_d": sum(adj_def.get(g.opponent_id, league_def) for g in games) / max(len(games), 1),
                    "ncsos_adj_em": sos,
                    "wins": wins,
                    "losses": losses,
                    "t_rank": 0,
                    "barthag": self._barthag(adj_em),
                    "effective_fg_pct": ff["effective_fg_pct"],
                    "turnover_rate": ff["turnover_rate"],
                    "offensive_reb_rate": ff["offensive_reb_rate"],
                    "free_throw_rate": ff["free_throw_rate"],
                    "opp_effective_fg_pct": ff["opp_effective_fg_pct"],
                    "opp_turnover_rate": ff["opp_turnover_rate"],
                    "defensive_reb_rate": ff["defensive_reb_rate"],
                    "opp_free_throw_rate": ff["opp_free_throw_rate"],
                }
            )

        ranked = sorted(teams_out, key=lambda r: r["adj_efficiency_margin"], reverse=True)
        for i, row in enumerate(ranked, start=1):
            row["overall_rank"] = i
        ranked_off = sorted(teams_out, key=lambda r: r["adj_offensive_efficiency"], reverse=True)
        for i, row in enumerate(ranked_off, start=1):
            row["offensive_rank"] = i
        ranked_def = sorted(teams_out, key=lambda r: r["adj_defensive_efficiency"])
        for i, row in enumerate(ranked_def, start=1):
            row["defensive_rank"] = i

        return {"teams": teams_out}

    def rows_from_records(self, team_metric_records: List[Dict]) -> List["TeamGameRow"]:
        return self._to_game_rows(team_metric_records)

    def _to_game_rows(self, records: List[Dict]) -> List[TeamGameRow]:
        rows: List[TeamGameRow] = []
        for rec in records:
            if not isinstance(rec, dict):
                continue

            team_id = self._team_id(self._pick(rec, ["team_id", "team", "team_slug", "school", "team_name"]))
            team_name = str(self._pick(rec, ["team_name", "team", "school", "team_display_name"]) or team_id)
            opp_id = self._team_id(self._pick(rec, ["opponent_id", "opponent", "opp", "opponent_name", "away_team", "team2", "home_team"]))
            if not team_id or not opp_id:
                continue

            points = self._to_float(self._pick(rec, ["team_score", "points", "pts", "score", "home_score", "team1_score"]))
            opp_points = self._to_float(self._pick(rec, ["opponent_score", "opp_points", "opp_score", "away_score", "team2_score"]))

            poss = self._to_float(self._pick(rec, ["possessions", "team_possessions", "poss"]))
            fga = self._to_float(self._pick(rec, ["fga", "field_goals_attempted", "team_fga"]))
            fgm = self._to_float(self._pick(rec, ["fgm", "field_goals_made", "team_fgm"]))
            fg3a = self._to_float(self._pick(rec, ["fg3a", "three_point_attempts", "team_fg3a", "x3pa"]))
            fg3m = self._to_float(self._pick(rec, ["fg3m", "three_point_field_goals_made", "team_fg3m", "x3pm"]))
            fta = self._to_float(self._pick(rec, ["fta", "free_throws_attempted", "team_fta"]))
            tov = self._to_float(self._pick(rec, ["turnovers", "tov", "team_tov"]))
            orb = self._to_float(self._pick(rec, ["orb", "offensive_rebounds", "team_orb"]))
            drb = self._to_float(self._pick(rec, ["drb", "defensive_rebounds", "team_drb"]))

            opp_fga = self._to_float(self._pick(rec, ["opp_fga", "opponent_fga"]))
            opp_fgm = self._to_float(self._pick(rec, ["opp_fgm", "opponent_fgm"]))
            opp_fg3a = self._to_float(self._pick(rec, ["opp_fg3a", "opponent_fg3a"]))
            opp_fg3m = self._to_float(self._pick(rec, ["opp_fg3m", "opponent_fg3m"]))
            opp_fta = self._to_float(self._pick(rec, ["opp_fta", "opponent_fta"]))
            opp_tov = self._to_float(self._pick(rec, ["opp_tov", "opponent_tov"]))
            opp_orb = self._to_float(self._pick(rec, ["opp_orb", "opponent_orb"]))
            opp_drb = self._to_float(self._pick(rec, ["opp_drb", "opponent_drb"]))

            if poss <= 0:
                poss = fga - orb + tov + 0.475 * fta
                opp_poss = opp_fga - opp_orb + opp_tov + 0.475 * opp_fta
                if opp_poss > 0:
                    poss = (poss + opp_poss) / 2.0

            if poss <= 0:
                continue

            rows.append(
                TeamGameRow(
                    team_id=team_id,
                    team_name=team_name,
                    opponent_id=opp_id,
                    points=points,
                    opp_points=opp_points,
                    possessions=poss,
                    fga=fga,
                    fgm=fgm,
                    fg3a=fg3a,
                    fg3m=fg3m,
                    fta=fta,
                    tov=tov,
                    orb=orb,
                    opp_drb=opp_drb,
                    opp_fga=opp_fga,
                    opp_fgm=opp_fgm,
                    opp_fg3a=opp_fg3a,
                    opp_fg3m=opp_fg3m,
                    opp_fta=opp_fta,
                    opp_tov=opp_tov,
                    opp_orb=opp_orb,
                    drb=drb,
                )
            )
        return rows

    def _four_factors(self, games: List[TeamGameRow]) -> Dict[str, float]:
        fgm = sum(g.fgm for g in games)
        fg3m = sum(g.fg3m for g in games)
        fga = sum(g.fga for g in games)
        tov = sum(g.tov for g in games)
        poss = sum(g.possessions for g in games)
        orb = sum(g.orb for g in games)
        opp_drb = sum(g.opp_drb for g in games)
        fta = sum(g.fta for g in games)

        opp_fgm = sum(g.opp_fgm for g in games)
        opp_fg3m = sum(g.opp_fg3m for g in games)
        opp_fga = sum(g.opp_fga for g in games)
        opp_tov = sum(g.opp_tov for g in games)
        opp_orb = sum(g.opp_orb for g in games)
        drb = sum(g.drb for g in games)
        opp_fta = sum(g.opp_fta for g in games)

        return {
            "effective_fg_pct": (fgm + 0.5 * fg3m) / max(fga, 1.0),
            "turnover_rate": tov / max(poss, 1.0),
            "offensive_reb_rate": orb / max((orb + opp_drb), 1.0),
            "free_throw_rate": fta / max(fga, 1.0),
            "opp_effective_fg_pct": (opp_fgm + 0.5 * opp_fg3m) / max(opp_fga, 1.0),
            "opp_turnover_rate": opp_tov / max(poss, 1.0),
            "defensive_reb_rate": drb / max((drb + opp_orb), 1.0),
            "opp_free_throw_rate": opp_fta / max(opp_fga, 1.0),
        }

    @staticmethod
    def _luck(games: List[TeamGameRow], adj_em: float) -> float:
        if not games:
            return 0.0
        actual_win = sum(1 for g in games if g.points > g.opp_points) / len(games)
        exp = 1.0 / (1.0 + 10 ** (-adj_em / 10.0))
        return actual_win - exp

    @staticmethod
    def _barthag(adj_em: float) -> float:
        return 1.0 / (1.0 + 10 ** (-adj_em / 10.0))

    @staticmethod
    def _infer_conference(team_id: str, teams: Optional[List[Dict]]) -> str:
        if not teams:
            return ""
        for row in teams:
            name = str(row.get("name", ""))
            rid = "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")
            if rid == team_id:
                return str(row.get("conference", ""))
        return ""

    @staticmethod
    def _team_id(value) -> str:
        if value is None:
            return ""
        return "".join(c.lower() if c.isalnum() else "_" for c in str(value)).strip("_")

    @staticmethod
    def _pick(row: Dict, keys: List[str]):
        for key in keys:
            if key in row and row[key] is not None:
                return row[key]
        return None

    @staticmethod
    def _to_float(value) -> float:
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
