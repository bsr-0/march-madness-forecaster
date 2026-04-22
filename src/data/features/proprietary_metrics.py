"""Proprietary advanced metrics engine.

Computes KenPom-equivalent and ShotQuality-equivalent metrics from
public box-score data (cbbpy / Torvik / Sports Reference).
"""

from __future__ import annotations

import bisect
import csv
import logging
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# Data containers

@dataclass
class GameRecord:
    """One team-side row from a single game."""

    game_id: str
    game_date: str  # YYYY-MM-DD
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
    ftm: float = 0.0
    tov: float = 0.0
    orb: float = 0.0
    drb: float = 0.0

    opp_fga: float = 0.0
    opp_fgm: float = 0.0
    opp_fg3a: float = 0.0
    opp_fg3m: float = 0.0
    opp_fta: float = 0.0
    opp_ftm: float = 0.0
    opp_tov: float = 0.0
    opp_orb: float = 0.0
    opp_drb: float = 0.0

    ast: float = 0.0
    stl: float = 0.0
    blk: float = 0.0
    pf: float = 0.0
    opp_ast: float = 0.0
    opp_stl: float = 0.0
    opp_blk: float = 0.0
    opp_pf: float = 0.0

    is_home: bool = False
    is_neutral: bool = True
    has_box_score: Optional[bool] = None

    def __post_init__(self):
        if self.has_box_score is None:
            self.has_box_score = self.fga > 0


@dataclass
class ProprietaryTeamMetrics:
    """Complete output for a single team -- replaces KenPom + ShotQuality."""

    team_id: str
    team_name: str
    conference: str = ""

    adj_offensive_efficiency: float = 100.0
    adj_defensive_efficiency: float = 100.0
    adj_efficiency_margin: float = 0.0
    adj_tempo: float = 68.0

    # Four Factors -- offense
    effective_fg_pct: float = 0.50
    turnover_rate: float = 0.18
    offensive_reb_rate: float = 0.30
    free_throw_rate: float = 0.30

    # Four Factors -- defense
    opp_effective_fg_pct: float = 0.50
    opp_turnover_rate: float = 0.18
    defensive_reb_rate: float = 0.70
    opp_free_throw_rate: float = 0.30

    # Supplementary shooting
    two_pt_pct: float = 0.48
    three_pt_pct: float = 0.34
    three_pt_rate: float = 0.35
    ft_pct: float = 0.72
    opp_two_pt_pct: float = 0.48
    opp_three_pt_pct: float = 0.34
    opp_three_pt_rate: float = 0.35

    # SOS
    sos_adj_em: float = 0.0
    sos_opp_o: float = 100.0
    sos_opp_d: float = 100.0
    ncsos_adj_em: float = 0.0

    luck: float = 0.0
    wab: float = 0.0

    # Poisson Binomial resume metrics
    sor: float = 0.5
    wab_poisson: float = 0.0

    offensive_xp_per_possession: float = 1.0
    defensive_xp_per_possession: float = 1.0
    shot_distribution_score: float = 0.0
    three_pt_variance: float = 0.095
    momentum: float = 0.0
    recent_adj_em: float = 0.0
    pace_adjusted_variance: float = 0.0
    consistency: float = 0.5
    sos_adjusted_consistency: float = 0.5
    barthag: float = 0.5

    # Extended metrics
    elo_rating: float = 1500.0
    free_throw_pct: float = 0.72
    opp_free_throw_pct: float = 0.72
    assist_to_turnover_ratio: float = 1.0
    assist_rate: float = 0.50
    steal_rate: float = 0.08
    block_rate: float = 0.05
    defensive_disruption_rate: float = 0.13
    opp_two_pt_pct_allowed: float = 0.48
    opp_three_pt_attempt_rate: float = 0.35
    conference_adj_em: float = 0.0
    seed_efficiency_residual: float = 0.0

    win_pct: float = 0.5
    elite_sos: float = 0.0
    q1_wins: int = 0
    q1_losses: int = 0
    q1_win_pct: float = 0.0
    efficiency_ratio: float = 1.0
    foul_rate: float = 0.18
    three_pt_regression_signal: float = 0.0

    # Schedule/context features
    rest_days: float = 5.0
    top5_minutes_share: float = 0.70
    preseason_ap_rank: int = 0
    coach_tournament_appearances: int = 0
    conf_tourney_champion: bool = False
    conf_tourney_games: int = 0
    conf_tourney_margin: float = 0.0
    late_season_games: int = 0
    late_season_margin: float = 0.0
    late_season_win_pct: float = 0.0
    pace_variance: float = 0.0
    coach_tournament_win_rate: float = 0.0

    # Advanced metrics
    true_shooting_pct: float = 0.54
    opp_true_shooting_pct: float = 0.54
    neutral_site_win_pct: float = 0.5
    neutral_site_games: int = 0
    home_adj_em: float = 0.0
    away_adj_em: float = 0.0
    home_court_dependence: float = 0.0
    momentum_5g: float = 0.0
    transition_efficiency: float = 0.0
    defensive_transition_vulnerability: float = 0.0
    road_neutral_wins: int = 0
    road_neutral_losses: int = 0
    road_neutral_games: int = 0
    road_neutral_win_pct: float = 0.5
    wins: int = 0
    losses: int = 0

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


# Engine

class ProprietaryMetricsEngine:
    """Compute all proprietary advanced metrics from game-level box scores."""

    HCA_POINTS: float = 3.75
    MARGIN_CAP: float = 16.0
    BUBBLE_RANK: int = 45
    BUBBLE_EM_PRIOR: float = 5.0
    SOS_ITERATIONS: int = 15

    def __init__(self, require_cutoff_date: bool = True) -> None:
        self._elo_prior: Optional[Dict[str, float]] = None
        self._end_of_season_elo: Optional[Dict[str, float]] = None
        self._require_cutoff_date = require_cutoff_date

    def compute(
        self,
        game_records: List[GameRecord],
        conference_map: Optional[Dict[str, str]] = None,
        cutoff_date: Optional[str] = None,
    ) -> Dict[str, ProprietaryTeamMetrics]:
        """Run the full engine. Returns team_id -> ProprietaryTeamMetrics."""
        if cutoff_date is None and self._require_cutoff_date:
            raise ValueError(
                "cutoff_date is required to prevent temporal leakage. "
                "Pass cutoff_date='YYYY-MM-DD' or set require_cutoff_date=False."
            )

        if cutoff_date and self._require_cutoff_date:
            try:
                from ...pipeline.config import TOURNAMENT_START_DATES

                cutoff_year = int(cutoff_date[:4])
                tournament_start = TOURNAMENT_START_DATES.get(cutoff_year)
                if tournament_start and cutoff_date > tournament_start.isoformat():
                    from ...exceptions import LeakageError

                    raise LeakageError(
                        f"cutoff_date={cutoff_date} is after tournament start "
                        f"{tournament_start} for {cutoff_year}."
                    )
            except ImportError:
                pass

        if not game_records:
            return {}

        if cutoff_date:
            game_records = [g for g in game_records if g.game_date < cutoff_date]
            if not game_records:
                return {}

        by_team: Dict[str, List[GameRecord]] = defaultdict(list)
        for rec in game_records:
            by_team[rec.team_id].append(rec)
        for tid in by_team:
            by_team[tid].sort(key=lambda g: g.game_date)

        raw_off, raw_def, tempo, names = self._raw_efficiency(by_team)
        adj_off, adj_def = self._iterative_sos_adjust(by_team, raw_off, raw_def)

        all_team_ids = sorted(by_team.keys())
        league_avg_em = float(np.mean([adj_off[t] - adj_def[t] for t in all_team_ids]))

        results: Dict[str, ProprietaryTeamMetrics] = {}
        for tid in all_team_ids:
            games = by_team[tid]
            box_games = [g for g in games if g.has_box_score]
            adj_em = adj_off[tid] - adj_def[tid]

            ff = self._four_factors(box_games)
            shooting = self._supplementary_shooting(box_games)
            sos = self._strength_of_schedule(games, adj_off, adj_def, conference_map or {})
            luck = self._correlated_gaussian_luck(games)
            barthag = self._pythagorean_win_pct(adj_off[tid], adj_def[tid])
            extended = self._extended_box_score_metrics(box_games)

            off_xp = self._box_score_xp(ff, side="offense", ft_pct=extended["free_throw_pct"])
            def_xp = self._box_score_xp(
                {
                    "effective_fg_pct": ff["opp_effective_fg_pct"],
                    "turnover_rate": ff["opp_turnover_rate"],
                    "offensive_reb_rate": 1.0 - ff["defensive_reb_rate"],
                    "free_throw_rate": ff["opp_free_throw_rate"],
                },
                side="offense",
                ft_pct=extended["opp_free_throw_pct"],
            )

            shot_dist = self._shot_distribution_score(box_games)
            three_var = self._three_point_variance(box_games)
            momentum, recent_em = self._momentum(games, adj_off, adj_def)
            pace_var = self._pace_adjusted_variance(games)
            consistency = self._consistency(games)
            sos_consistency = self._sos_adjusted_consistency(games, adj_off, adj_def)
            opp_shot_selection = self._opponent_shot_selection(box_games)
            foul_rate = self._foul_rate(box_games)

            wins = sum(1 for g in games if g.points > g.opp_points)
            losses = len(games) - wins
            n_games = max(wins + losses, 1)
            eff_ratio = adj_off[tid] / max(adj_def[tid], 1e-6)

            actual_3p = shooting.get("three_pt_pct", 0.345)
            n_3pa = sum(g.fg3a for g in games)
            prior_weight = 100.0
            shrunk_3p = (n_3pa * actual_3p + prior_weight * 0.345) / (n_3pa + prior_weight)
            three_pt_regression = shrunk_3p - 0.345

            ts_pct, opp_ts_pct = self._true_shooting_pct(games)
            neutral_win_pct, neutral_games = self._neutral_site_record(games)
            home_em, away_em, hc_dependence = self._home_away_splits(games, adj_off, adj_def)
            mom_5g = self._momentum_5g(games, adj_off, adj_def)

            results[tid] = ProprietaryTeamMetrics(
                team_id=tid,
                team_name=names.get(tid, tid),
                conference=(conference_map or {}).get(tid, ""),
                adj_offensive_efficiency=adj_off[tid],
                adj_defensive_efficiency=adj_def[tid],
                adj_efficiency_margin=adj_em,
                adj_tempo=tempo.get(tid, 68.0),
                **ff,
                **shooting,
                sos_adj_em=sos["sos_adj_em"],
                sos_opp_o=sos["sos_opp_o"],
                sos_opp_d=sos["sos_opp_d"],
                ncsos_adj_em=sos["ncsos_adj_em"],
                luck=luck,
                barthag=barthag,
                offensive_xp_per_possession=off_xp,
                defensive_xp_per_possession=def_xp,
                shot_distribution_score=shot_dist,
                three_pt_variance=three_var,
                momentum=momentum,
                recent_adj_em=recent_em,
                pace_adjusted_variance=pace_var,
                consistency=consistency,
                sos_adjusted_consistency=sos_consistency,
                free_throw_pct=extended["free_throw_pct"],
                opp_free_throw_pct=extended["opp_free_throw_pct"],
                assist_to_turnover_ratio=extended["assist_to_turnover_ratio"],
                assist_rate=extended["assist_rate"],
                steal_rate=extended["steal_rate"],
                block_rate=extended["block_rate"],
                defensive_disruption_rate=extended["defensive_disruption_rate"],
                opp_two_pt_pct_allowed=opp_shot_selection["opp_two_pt_pct_allowed"],
                opp_three_pt_attempt_rate=opp_shot_selection["opp_three_pt_attempt_rate"],
                win_pct=wins / n_games,
                efficiency_ratio=eff_ratio,
                foul_rate=foul_rate,
                three_pt_regression_signal=three_pt_regression,
                pace_variance=self._pace_variance(games),
                true_shooting_pct=ts_pct,
                opp_true_shooting_pct=opp_ts_pct,
                neutral_site_win_pct=neutral_win_pct,
                neutral_site_games=neutral_games,
                home_adj_em=home_em,
                away_adj_em=away_em,
                home_court_dependence=hc_dependence,
                momentum_5g=mom_5g,
                wins=wins,
                losses=losses,
            )

        self._compute_wab(results, by_team)
        self._compute_sor_and_wab_poisson(results, by_team)

        # Elo ratings (inline, MOV-adjusted, K=38)
        prior = getattr(self, "_elo_prior", None)
        self._end_of_season_elo = self._compute_elo_inline(results, by_team, prior)

        self._compute_conference_strength(results, by_team, conference_map or {})
        self._compute_elite_sos_and_quadrants(results, by_team, adj_off, adj_def)
        self._compute_rest_days(results, by_team, reference_date=cutoff_date)

        self._by_team = by_team
        self._adj_off = adj_off
        self._adj_def = adj_def

        return results

    # Internal computation helpers

    def _raw_efficiency(
        self, by_team: Dict[str, List[GameRecord]]
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, str]]:
        """Compute raw offensive/defensive efficiency and tempo per team."""
        raw_off: Dict[str, float] = {}
        raw_def: Dict[str, float] = {}
        tempo: Dict[str, float] = {}
        names: Dict[str, str] = {}

        for tid, games in by_team.items():
            total_poss = sum(g.possessions for g in games)
            total_pts = sum(g.points for g in games)
            total_opp = sum(g.opp_points for g in games)
            n = len(games)
            raw_off[tid] = 100.0 * total_pts / max(total_poss, 1.0)
            raw_def[tid] = 100.0 * total_opp / max(total_poss, 1.0)
            tempo[tid] = total_poss / max(n, 1)
            names[tid] = games[0].team_name or tid

        return raw_off, raw_def, tempo, names

    def _iterative_sos_adjust(
        self,
        by_team: Dict[str, List[GameRecord]],
        raw_off: Dict[str, float],
        raw_def: Dict[str, float],
        initial_adj_off: Optional[Dict[str, float]] = None,
        initial_adj_def: Optional[Dict[str, float]] = None,
        n_iterations: Optional[int] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Additive iterative SOS adjustment (KenPom-style)."""
        league_off = float(np.mean(list(raw_off.values()))) if raw_off else 100.0
        league_def = float(np.mean(list(raw_def.values()))) if raw_def else 100.0

        adj_off = dict(initial_adj_off) if initial_adj_off is not None else dict(raw_off)
        adj_def = dict(initial_adj_def) if initial_adj_def is not None else dict(raw_def)
        if initial_adj_off is not None:
            for tid in raw_off:
                if tid not in adj_off:
                    adj_off[tid] = raw_off[tid]
            for tid in raw_def:
                if tid not in adj_def:
                    adj_def[tid] = raw_def[tid]

        iters = n_iterations if n_iterations is not None else self.SOS_ITERATIONS
        DAMPING = 0.7

        for _iteration in range(iters):
            next_off: Dict[str, float] = {}
            next_def: Dict[str, float] = {}

            for tid, games in by_team.items():
                off_adjustments: List[float] = []
                def_adjustments: List[float] = []
                weights: List[float] = []

                n_games = len(games)
                for idx, g in enumerate(games):
                    hca = 0.0
                    if not g.is_neutral:
                        hca = self.HCA_POINTS if g.is_home else -self.HCA_POINTS

                    adj_pts = g.points - hca
                    adj_opp = g.opp_points + hca

                    margin = adj_pts - adj_opp
                    if abs(margin) > self.MARGIN_CAP:
                        excess = abs(margin) - self.MARGIN_CAP
                        if margin > 0:
                            adj_pts -= excess / 2
                            adj_opp += excess / 2
                        else:
                            adj_pts += excess / 2
                            adj_opp -= excess / 2

                    poss = max(g.possessions, 1.0)
                    game_off = 100.0 * adj_pts / poss
                    game_def = 100.0 * adj_opp / poss

                    opp_def = adj_def.get(g.opponent_id, league_def)
                    opp_off = adj_off.get(g.opponent_id, league_off)

                    off_adjustments.append(game_off + (opp_def - league_def))
                    def_adjustments.append(game_def + (opp_off - league_off))

                    recency = math.exp(-0.693 * (n_games - 1 - idx) / 30.0)
                    weights.append(recency)

                total_w = sum(weights) or 1.0
                computed_off = sum(o * w for o, w in zip(off_adjustments, weights)) / total_w
                computed_def = sum(d * w for d, w in zip(def_adjustments, weights)) / total_w

                next_off[tid] = DAMPING * computed_off + (1.0 - DAMPING) * adj_off[tid]
                next_def[tid] = DAMPING * computed_def + (1.0 - DAMPING) * adj_def[tid]

            off_mean = float(np.mean(list(next_off.values())))
            def_mean = float(np.mean(list(next_def.values())))
            off_shift = league_off - off_mean
            def_shift = league_def - def_mean
            for tid in next_off:
                next_off[tid] += off_shift
                next_def[tid] += def_shift

            adj_off = next_off
            adj_def = next_def

        return adj_off, adj_def

    def _four_factors(self, games: List[GameRecord]) -> Dict[str, float]:
        """Dean Oliver's Four Factors -- offense + defense."""
        fgm = sum(g.fgm for g in games)
        fg3m = sum(g.fg3m for g in games)
        fga = sum(g.fga for g in games)
        tov = sum(g.tov for g in games)
        fta = sum(g.fta for g in games)
        orb = sum(g.orb for g in games)
        opp_drb = sum(g.opp_drb for g in games)

        opp_fgm = sum(g.opp_fgm for g in games)
        opp_fg3m = sum(g.opp_fg3m for g in games)
        opp_fga = sum(g.opp_fga for g in games)
        opp_tov = sum(g.opp_tov for g in games)
        opp_fta = sum(g.opp_fta for g in games)
        opp_orb = sum(g.opp_orb for g in games)
        drb = sum(g.drb for g in games)

        off_tov_denom = fga + 0.44 * fta + tov
        def_tov_denom = opp_fga + 0.44 * opp_fta + opp_tov

        return {
            "effective_fg_pct": (fgm + 0.5 * fg3m) / max(fga, 1.0),
            "turnover_rate": tov / max(off_tov_denom, 1.0),
            "offensive_reb_rate": orb / max(orb + opp_drb, 1.0),
            "free_throw_rate": fta / max(fga, 1.0),
            "opp_effective_fg_pct": (opp_fgm + 0.5 * opp_fg3m) / max(opp_fga, 1.0),
            "opp_turnover_rate": opp_tov / max(def_tov_denom, 1.0),
            "defensive_reb_rate": drb / max(drb + opp_orb, 1.0),
            "opp_free_throw_rate": opp_fta / max(opp_fga, 1.0),
        }

    def _supplementary_shooting(self, games: List[GameRecord]) -> Dict[str, float]:
        """Compute 2P%, 3P%, 3P rate, FT%, and opponent equivalents."""
        fga = sum(g.fga for g in games)
        fgm = sum(g.fgm for g in games)
        fg3a = sum(g.fg3a for g in games)
        fg3m = sum(g.fg3m for g in games)
        fta = sum(g.fta for g in games)
        ftm = sum(g.ftm for g in games)

        opp_fga = sum(g.opp_fga for g in games)
        opp_fgm = sum(g.opp_fgm for g in games)
        opp_fg3a = sum(g.opp_fg3a for g in games)
        opp_fg3m = sum(g.opp_fg3m for g in games)

        fg2a = max(fga - fg3a, 1.0)
        fg2m = fgm - fg3m
        opp_fg2a = max(opp_fga - opp_fg3a, 1.0)
        opp_fg2m = opp_fgm - opp_fg3m

        return {
            "two_pt_pct": fg2m / max(fg2a, 1.0),
            "three_pt_pct": fg3m / max(fg3a, 1.0),
            "three_pt_rate": fg3a / max(fga, 1.0),
            "ft_pct": ftm / max(fta, 1.0),
            "opp_two_pt_pct": opp_fg2m / max(opp_fg2a, 1.0),
            "opp_three_pt_pct": opp_fg3m / max(opp_fg3a, 1.0),
            "opp_three_pt_rate": opp_fg3a / max(opp_fga, 1.0),
        }

    def _strength_of_schedule(
        self,
        games: List[GameRecord],
        adj_off: Dict[str, float],
        adj_def: Dict[str, float],
        conference_map: Dict[str, str] = None,
    ) -> Dict[str, float]:
        """Compute SOS and non-conference SOS."""
        league_off = float(np.mean(list(adj_off.values()))) if adj_off else 100.0
        league_def = float(np.mean(list(adj_def.values()))) if adj_def else 100.0
        conference_map = conference_map or {}

        n = max(len(games), 1)
        opp_o = [adj_off.get(g.opponent_id, league_off) for g in games]
        opp_d = [adj_def.get(g.opponent_id, league_def) for g in games]
        opp_em = [o - d for o, d in zip(opp_o, opp_d)]

        team_conf = conference_map.get(games[0].team_id, "") if games else ""
        if team_conf and conference_map:
            nc_ems = [
                adj_off.get(g.opponent_id, league_off) - adj_def.get(g.opponent_id, league_def)
                for g in games
                if conference_map.get(g.opponent_id, "") != team_conf
            ]
        else:
            nc_ems = [
                adj_off.get(g.opponent_id, league_off) - adj_def.get(g.opponent_id, league_def)
                for g in games
                if g.is_neutral or not g.is_home
            ]

        return {
            "sos_adj_em": sum(opp_em) / n,
            "sos_opp_o": sum(opp_o) / n,
            "sos_opp_d": sum(opp_d) / n,
            "ncsos_adj_em": sum(nc_ems) / max(len(nc_ems), 1) if nc_ems else 0.0,
        }

    def _correlated_gaussian_luck(self, games: List[GameRecord]) -> float:
        """KenPom-style luck: actual_win% - expected_win%."""
        if len(games) < 12:
            return 0.0

        margins = [g.points - g.opp_points for g in games]
        mean_m = float(np.mean(margins))
        std_m = float(np.std(margins, ddof=1))

        if std_m < 0.1:
            return 0.0

        z = mean_m / std_m
        expected_win_pct = float(scipy_stats.norm.cdf(z))
        actual_win_pct = sum(1 for m in margins if m > 0) / len(margins)

        raw_luck = actual_win_pct - expected_win_pct
        MIN_GAMES = 12
        FULL_WEIGHT_GAMES = 32
        shrinkage = min(1.0, (len(games) - MIN_GAMES) / (FULL_WEIGHT_GAMES - MIN_GAMES))
        return raw_luck * shrinkage

    def _pythagorean_win_pct(self, adj_o: float, adj_d: float) -> float:
        """Pythagorean win% via logistic on efficiency margin."""
        margin = adj_o - adj_d
        return 1.0 / (1.0 + math.exp(-0.1735 * margin))

    def _box_score_xp(self, ff: Dict[str, float], side: str = "offense", ft_pct: float = 0.72) -> float:
        """Expected points per possession from Four Factors decomposition."""
        efg = ff.get("effective_fg_pct", 0.50)
        tov = ff.get("turnover_rate", 0.18)
        orb = ff.get("offensive_reb_rate", 0.30)
        ftr = ff.get("free_throw_rate", 0.30)

        composite = 0.40 * efg + 0.25 * (1.0 - tov) + 0.20 * orb + 0.15 * ftr * ft_pct
        xp = composite * 2.0
        return float(np.clip(xp, 0.5, 1.8))

    def _shot_distribution_score(self, games: List[GameRecord]) -> float:
        """Proxy for shot quality distribution (rim + 3pt vs midrange)."""
        fga = sum(g.fga for g in games)
        fg3a = sum(g.fg3a for g in games)
        fta = sum(g.fta for g in games)

        if fga < 1:
            return 0.0

        three_rate = fg3a / fga
        ft_rate = fta / fga
        estimated_rim_rate = float(np.clip(0.18 + 0.7 * ft_rate, 0.15, 0.50))
        midrange_rate = max(1.0 - three_rate - estimated_rim_rate, 0.05)
        return float((estimated_rim_rate + three_rate) - 0.75 * midrange_rate)

    def _three_point_variance(self, games: List[GameRecord]) -> float:
        """Game-to-game 3P% stdev with Bayesian shrinkage."""
        THREE_PT_VAR_PRIOR_STD = 0.095
        THREE_PT_VAR_PRIOR_WEIGHT = 8.0

        per_game_3p = []
        for g in games:
            if g.fg3a >= 5:
                per_game_3p.append(g.fg3m / g.fg3a)

        if len(per_game_3p) < 5:
            return THREE_PT_VAR_PRIOR_STD

        sample_var = float(np.var(per_game_3p, ddof=1))
        prior_var = THREE_PT_VAR_PRIOR_STD**2
        n = len(per_game_3p)
        shrunk_var = (n * sample_var + THREE_PT_VAR_PRIOR_WEIGHT * prior_var) / (n + THREE_PT_VAR_PRIOR_WEIGHT)
        return float(np.sqrt(shrunk_var))

    def _momentum(
        self,
        games: List[GameRecord],
        adj_off: Dict[str, float],
        adj_def: Dict[str, float],
    ) -> Tuple[float, float]:
        """Last-10-game rolling form with Normal-Normal Bayesian shrinkage."""
        MARGIN_SIGMA_SQ = 11.0**2
        MOMENTUM_SIGNAL_SIGMA_SQ = 3.0**2

        league_off = float(np.mean(list(adj_off.values()))) if adj_off else 100.0
        league_def = float(np.mean(list(adj_def.values()))) if adj_def else 100.0

        if len(games) < 12:
            return 0.0, adj_off.get(games[0].team_id, 100.0) - adj_def.get(games[0].team_id, 100.0) if games else 0.0

        recent = games[-10:]
        recent_margins = []
        for g in recent:
            opp_em = adj_off.get(g.opponent_id, league_off) - adj_def.get(g.opponent_id, league_def)
            recent_margins.append((g.points - g.opp_points) - opp_em)

        season_margins = []
        for g in games:
            opp_em = adj_off.get(g.opponent_id, league_off) - adj_def.get(g.opponent_id, league_def)
            season_margins.append((g.points - g.opp_points) - opp_em)

        recent_em = float(np.mean(recent_margins))
        season_em = float(np.mean(season_margins))
        raw_delta = recent_em - season_em

        n_recent = len(recent_margins)
        lam = (MARGIN_SIGMA_SQ / n_recent) / (MARGIN_SIGMA_SQ / n_recent + MOMENTUM_SIGNAL_SIGMA_SQ)
        shrunk_momentum = raw_delta * (1.0 - lam)

        return float(shrunk_momentum), float(recent_em)

    def _pace_adjusted_variance(self, games: List[GameRecord]) -> float:
        """Scoring-margin variance adjusted for pace with Bayesian shrinkage."""
        PACE_VAR_PRIOR_STD = 11.0
        PACE_VAR_PRIOR_WEIGHT = 8.0

        adjusted_margins = []
        for g in games:
            margin = g.points - g.opp_points
            pace_factor = max(g.possessions, 40.0) / 70.0
            adjusted_margins.append(margin / pace_factor)

        n = len(adjusted_margins)
        if n < 2:
            return PACE_VAR_PRIOR_STD

        sample_var = float(np.var(adjusted_margins, ddof=1))
        prior_var = PACE_VAR_PRIOR_STD**2
        shrunk_var = (n * sample_var + PACE_VAR_PRIOR_WEIGHT * prior_var) / (n + PACE_VAR_PRIOR_WEIGHT)
        return float(np.sqrt(shrunk_var))

    CONSISTENCY_PRIOR_STD = 8.0
    CONSISTENCY_PRIOR_WEIGHT = 8.0

    def _consistency(self, games: List[GameRecord]) -> float:
        """Inverse of scoring-margin stdev with Bayesian shrinkage."""
        if len(games) < 5:
            return 0.5

        margins = [g.points - g.opp_points for g in games]
        n = len(margins)
        sample_var = float(np.var(margins, ddof=1))
        prior_var = self.CONSISTENCY_PRIOR_STD**2
        shrunk_var = (n * sample_var + self.CONSISTENCY_PRIOR_WEIGHT * prior_var) / (n + self.CONSISTENCY_PRIOR_WEIGHT)
        return 1.0 / (1.0 + float(np.sqrt(shrunk_var)))

    def _sos_adjusted_consistency(
        self,
        games: List[GameRecord],
        adj_off: Dict[str, float],
        adj_def: Dict[str, float],
    ) -> float:
        """Consistency on SOS-adjusted residuals with Bayesian shrinkage."""
        if len(games) < 5:
            return 0.5

        league_off = float(np.mean(list(adj_off.values()))) if adj_off else 100.0
        league_def = float(np.mean(list(adj_def.values()))) if adj_def else 100.0

        residuals = []
        for g in games:
            opp_em = adj_off.get(g.opponent_id, league_off) - adj_def.get(g.opponent_id, league_def)
            residuals.append((g.points - g.opp_points) - opp_em)

        n = len(residuals)
        sample_var = float(np.var(residuals, ddof=1))
        prior_var = self.CONSISTENCY_PRIOR_STD**2
        shrunk_var = (n * sample_var + self.CONSISTENCY_PRIOR_WEIGHT * prior_var) / (n + self.CONSISTENCY_PRIOR_WEIGHT)
        return 1.0 / (1.0 + float(np.sqrt(shrunk_var)))

    def _compute_wab(
        self,
        results: Dict[str, ProprietaryTeamMetrics],
        by_team: Dict[str, List[GameRecord]],
    ) -> None:
        """Wins Above Bubble (NCAA selection committee metric)."""
        bubble_em = self.BUBBLE_EM_PRIOR

        for tid, games in by_team.items():
            if tid not in results:
                continue

            total_wab = 0.0
            for g in games:
                opp = results.get(g.opponent_id)
                opp_em = opp.adj_efficiency_margin if opp else 0.0

                bubble_wp = self._log5_win_prob(bubble_em, opp_em)
                if not g.is_neutral:
                    if g.is_home:
                        bubble_wp = self._log5_win_prob(bubble_em, opp_em + self.HCA_POINTS)
                    else:
                        bubble_wp = self._log5_win_prob(bubble_em + self.HCA_POINTS, opp_em)

                bubble_wp = float(np.clip(bubble_wp, 0.01, 0.99))

                is_win = g.points > g.opp_points
                if is_win:
                    total_wab += 1.0 - bubble_wp
                else:
                    total_wab += 0.0 - bubble_wp

            results[tid].wab = round(total_wab, 2)

    # Poisson Binomial SOR & WAB

    @staticmethod
    def _poisson_binomial_cdf(k: int, probs: List[float]) -> float:
        """P(X <= k) for Poisson Binomial distribution via recursive DP."""
        n = len(probs)
        if n == 0:
            return 1.0
        k = max(0, min(k, n))

        pmf = np.zeros(n + 1, dtype=np.float64)
        pmf[0] = 1.0

        for p_i in probs:
            p_i = float(np.clip(p_i, 1e-6, 1.0 - 1e-6))
            q_i = 1.0 - p_i
            new_pmf = np.zeros_like(pmf)
            new_pmf[0] = pmf[0] * q_i
            for j in range(1, n + 1):
                new_pmf[j] = pmf[j] * q_i + pmf[j - 1] * p_i
            pmf = new_pmf

        return float(np.sum(pmf[: k + 1]))

    @staticmethod
    def _poisson_binomial_mean(probs: List[float]) -> float:
        """Expected value of Poisson Binomial = sum of probabilities."""
        return float(sum(probs))

    def _compute_sor_and_wab_poisson(
        self,
        results: Dict[str, ProprietaryTeamMetrics],
        by_team: Dict[str, List[GameRecord]],
    ) -> None:
        """Compute SOR and WAB via Poisson Binomial distributions."""
        REFERENCE_EM = 10.0
        bubble_em = self.BUBBLE_EM_PRIOR

        for tid, games in by_team.items():
            if tid not in results:
                continue

            ref_win_probs = []
            bubble_win_probs = []

            for g in games:
                opp = results.get(g.opponent_id)
                opp_em = opp.adj_efficiency_margin if opp else 0.0

                if g.is_neutral:
                    ref_p = self._log5_win_prob(REFERENCE_EM, opp_em)
                elif g.is_home:
                    ref_p = self._log5_win_prob(REFERENCE_EM + self.HCA_POINTS, opp_em)
                else:
                    ref_p = self._log5_win_prob(REFERENCE_EM, opp_em + self.HCA_POINTS)

                ref_p = float(np.clip(ref_p, 0.005, 0.995))
                ref_win_probs.append(ref_p)

                if g.is_neutral:
                    bub_p = self._log5_win_prob(bubble_em, opp_em)
                elif g.is_home:
                    bub_p = self._log5_win_prob(bubble_em, opp_em + self.HCA_POINTS)
                else:
                    bub_p = self._log5_win_prob(bubble_em + self.HCA_POINTS, opp_em)

                bub_p = float(np.clip(bub_p, 0.005, 0.995))
                bubble_win_probs.append(bub_p)

            actual_wins = sum(1 for g in games if g.points > g.opp_points)

            if ref_win_probs:
                sor = self._poisson_binomial_cdf(actual_wins, ref_win_probs)
            else:
                sor = 0.5

            expected_bubble_wins = self._poisson_binomial_mean(bubble_win_probs)
            wab_pb = actual_wins - expected_bubble_wins

            results[tid].sor = round(sor, 4)
            results[tid].wab_poisson = round(wab_pb, 2)

    def _extended_box_score_metrics(self, games: List[GameRecord]) -> Dict[str, float]:
        """FT%, A/TO, assist rate, steal rate, block rate, defensive disruption."""
        fta = sum(g.fta for g in games)
        ftm = sum(g.ftm for g in games)
        opp_fta = sum(g.opp_fta for g in games)
        opp_ftm = sum(g.opp_ftm for g in games)
        ast = sum(g.ast for g in games)
        tov = sum(g.tov for g in games)
        fgm = sum(g.fgm for g in games)
        stl = sum(g.stl for g in games)
        blk = sum(g.blk for g in games)
        total_poss = sum(max(g.possessions, 1.0) for g in games)

        return {
            "free_throw_pct": ftm / max(fta, 1.0),
            "opp_free_throw_pct": opp_ftm / max(opp_fta, 1.0),
            "assist_to_turnover_ratio": ast / max(tov, 1.0),
            "assist_rate": ast / max(fgm, 1.0),
            "steal_rate": stl / max(total_poss, 1.0),
            "block_rate": blk / max(total_poss, 1.0),
            "defensive_disruption_rate": (stl + blk) / max(total_poss, 1.0),
        }

    def _opponent_shot_selection(self, games: List[GameRecord]) -> Dict[str, float]:
        """Opponent 2P% allowed and 3PA rate (controllable metrics)."""
        opp_fga = sum(g.opp_fga for g in games)
        opp_fgm = sum(g.opp_fgm for g in games)
        opp_fg3a = sum(g.opp_fg3a for g in games)
        opp_fg3m = sum(g.opp_fg3m for g in games)

        opp_fg2a = max(opp_fga - opp_fg3a, 1.0)
        opp_fg2m = opp_fgm - opp_fg3m

        return {
            "opp_two_pt_pct_allowed": opp_fg2m / max(opp_fg2a, 1.0),
            "opp_three_pt_attempt_rate": opp_fg3a / max(opp_fga, 1.0),
        }

    def _compute_elo_inline(
        self,
        results: Dict[str, ProprietaryTeamMetrics],
        by_team: Dict[str, List[GameRecord]],
        prior_elo: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """MOV-adjusted Elo ratings (K=38, cross-season carryover)."""
        _ELO_REGRESSION = 0.25
        _ELO_MEAN = 1500.0

        all_games: List[GameRecord] = []
        seen_game_ids: Dict[str, set] = defaultdict(set)
        for tid, games in by_team.items():
            for g in games:
                pair_key = tuple(sorted([g.team_id, g.opponent_id]))
                if g.game_id not in seen_game_ids.get(pair_key, set()):
                    all_games.append(g)
                    seen_game_ids.setdefault(pair_key, set()).add(g.game_id)
        all_games.sort(key=lambda g: g.game_date)

        elo: Dict[str, float] = defaultdict(lambda: _ELO_MEAN)
        if prior_elo:
            for t in by_team:
                elo[t] = (1.0 - _ELO_REGRESSION) * prior_elo.get(t, _ELO_MEAN) + _ELO_REGRESSION * _ELO_MEAN
        K_BASE = 38.0
        ELO_HCA = self.HCA_POINTS * 13.3

        for g in all_games:
            t1, t2 = g.team_id, g.opponent_id
            _ = elo[t1]
            _ = elo[t2]

            hca = 0.0
            if not g.is_neutral:
                hca = ELO_HCA if g.is_home else -ELO_HCA

            e1 = 1.0 / (1.0 + 10 ** (-(elo[t1] + hca - elo[t2]) / 400.0))
            margin = g.points - g.opp_points
            s1 = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)
            mov_mult = np.log1p(abs(margin))
            elo_diff = abs(elo[t1] - elo[t2])
            elo_dampening = 2.2 / (elo_diff * 0.001 + 2.2)
            k = K_BASE * mov_mult * elo_dampening

            delta = k * (s1 - e1)
            elo[t1] += delta
            elo[t2] -= delta

        for tid in results:
            results[tid].elo_rating = round(elo.get(tid, 1500.0), 1)

        return dict(elo)

    def _compute_conference_strength(
        self,
        results: Dict[str, ProprietaryTeamMetrics],
        by_team: Dict[str, List[GameRecord]],
        conference_map: Dict[str, str],
    ) -> None:
        """Average AdjEM of conference peers (or frequent-opponent cluster)."""
        if not conference_map:
            for tid in results:
                games = by_team.get(tid, [])
                if not games:
                    results[tid].conference_adj_em = results[tid].sos_adj_em
                    continue

                opp_counts: Dict[str, int] = defaultdict(int)
                for g in games:
                    opp_counts[g.opponent_id] += 1

                conf_peer_ems = []
                for opp_id, count in opp_counts.items():
                    if count >= 2 and opp_id in results:
                        conf_peer_ems.append(results[opp_id].adj_efficiency_margin)

                if conf_peer_ems:
                    results[tid].conference_adj_em = float(np.mean(conf_peer_ems))
                else:
                    results[tid].conference_adj_em = results[tid].sos_adj_em
            return

        conf_teams: Dict[str, List[str]] = defaultdict(list)
        for tid, conf in conference_map.items():
            if conf:
                conf_teams[conf].append(tid)

        conf_avg: Dict[str, float] = {}
        for conf, tids in conf_teams.items():
            ems = [results[t].adj_efficiency_margin for t in tids if t in results]
            conf_avg[conf] = float(np.mean(ems)) if ems else 0.0

        for tid in results:
            conf = conference_map.get(tid, "")
            results[tid].conference_adj_em = conf_avg.get(conf, results[tid].sos_adj_em)

    def _foul_rate(self, games: List[GameRecord]) -> float:
        """Team personal fouls per possession."""
        total_pf = sum(g.pf for g in games)
        total_poss = sum(max(g.possessions, 1.0) for g in games)
        if total_poss < 1.0:
            return 0.18
        return total_pf / total_poss

    def _compute_elite_sos_and_quadrants(
        self,
        results: Dict[str, ProprietaryTeamMetrics],
        by_team: Dict[str, List[GameRecord]],
        adj_off: Dict[str, float],
        adj_def: Dict[str, float],
    ) -> None:
        """Compute elite SOS and Quadrant record using AdjEM thresholds."""
        ELITE_EM_THRESHOLD = 15.0

        for tid, games in by_team.items():
            if tid not in results:
                continue

            elite_ems: List[float] = []
            q1_w, q1_l = 0, 0
            rn_w, rn_l = 0, 0

            for g in games:
                opp_em = adj_off.get(g.opponent_id, 100.0) - adj_def.get(g.opponent_id, 100.0)

                if opp_em >= ELITE_EM_THRESHOLD:
                    elite_ems.append(opp_em)

                q = self._classify_quadrant_by_em(opp_em, g.is_home, g.is_neutral)
                is_win = g.points > g.opp_points
                if q == 1:
                    if is_win:
                        q1_w += 1
                    else:
                        q1_l += 1

                if g.is_neutral or not g.is_home:
                    if is_win:
                        rn_w += 1
                    else:
                        rn_l += 1

            results[tid].elite_sos = float(np.mean(elite_ems)) if elite_ems else 0.0
            results[tid].q1_wins = q1_w
            results[tid].q1_losses = q1_l
            results[tid].q1_win_pct = q1_w / max(q1_w + q1_l, 1)
            results[tid].road_neutral_wins = rn_w
            results[tid].road_neutral_losses = rn_l
            results[tid].road_neutral_games = rn_w + rn_l
            results[tid].road_neutral_win_pct = rn_w / max(rn_w + rn_l, 1)

    @staticmethod
    def _classify_quadrant_by_em(opp_em: float, is_home: bool, is_neutral: bool) -> int:
        """Classify game into NCAA quadrant (1-4) using AdjEM thresholds."""
        if is_neutral:
            if opp_em >= 10.0:
                return 1
            elif opp_em >= 3.0:
                return 2
            elif opp_em >= -8.0:
                return 3
            return 4
        elif is_home:
            if opp_em >= 15.0:
                return 1
            elif opp_em >= 8.0:
                return 2
            elif opp_em >= -3.0:
                return 3
            return 4
        else:
            if opp_em >= 5.0:
                return 1
            elif opp_em >= -2.0:
                return 2
            elif opp_em >= -12.0:
                return 3
            return 4

    def _true_shooting_pct(self, games: List[GameRecord]) -> Tuple[float, float]:
        """True Shooting %: PTS / (2 * (FGA + 0.44 * FTA))."""
        total_pts = sum(g.points for g in games)
        total_fga = sum(g.fga for g in games)
        total_fta = sum(g.fta for g in games)

        opp_pts = sum(g.opp_points for g in games)
        opp_fga = sum(g.opp_fga for g in games)
        opp_fta = sum(g.opp_fta for g in games)

        tsa = 2.0 * (total_fga + 0.44 * total_fta)
        opp_tsa = 2.0 * (opp_fga + 0.44 * opp_fta)

        return float(total_pts / max(tsa, 1.0)), float(opp_pts / max(opp_tsa, 1.0))

    def _neutral_site_record(self, games: List[GameRecord]) -> Tuple[float, int]:
        """Neutral-site win% with Beta-Binomial shrinkage."""
        NEUTRAL_PRIOR_ALPHA = 2.0

        neutral_games = [g for g in games if g.is_neutral]
        n = len(neutral_games)
        wins = sum(1 for g in neutral_games if g.points > g.opp_points)
        shrunk_pct = (wins + NEUTRAL_PRIOR_ALPHA) / (n + 2 * NEUTRAL_PRIOR_ALPHA)
        return float(shrunk_pct), n

    def _home_away_splits(
        self,
        games: List[GameRecord],
        adj_off: Dict[str, float],
        adj_def: Dict[str, float],
    ) -> Tuple[float, float, float]:
        """Home/away AdjEM splits with Normal-Normal Bayesian shrinkage."""
        MARGIN_SIGMA_SQ = 11.0**2
        HCA_SIGNAL_SIGMA_SQ = 4.0**2

        league_off = float(np.mean(list(adj_off.values()))) if adj_off else 100.0
        league_def = float(np.mean(list(adj_def.values()))) if adj_def else 100.0

        home_margins = []
        away_margins = []
        for g in games:
            opp_em = adj_off.get(g.opponent_id, league_off) - adj_def.get(g.opponent_id, league_def)
            quality_margin = (g.points - g.opp_points) - opp_em
            if g.is_home:
                home_margins.append(quality_margin)
            elif not g.is_neutral:
                away_margins.append(quality_margin)

        def _shrink_split(margins: list) -> float:
            n = len(margins)
            if n == 0:
                return 0.0
            observed = float(np.mean(margins))
            lam = (MARGIN_SIGMA_SQ / n) / (MARGIN_SIGMA_SQ / n + HCA_SIGNAL_SIGMA_SQ)
            return observed * (1.0 - lam)

        home_em = _shrink_split(home_margins)
        away_em = _shrink_split(away_margins)
        return home_em, away_em, home_em - away_em

    def _momentum_5g(
        self,
        games: List[GameRecord],
        adj_off: Dict[str, float],
        adj_def: Dict[str, float],
    ) -> float:
        """5-game rolling form with Normal-Normal Bayesian shrinkage."""
        MARGIN_SIGMA_SQ = 11.0**2
        MOMENTUM_SIGNAL_SIGMA_SQ = 3.0**2

        league_off = float(np.mean(list(adj_off.values()))) if adj_off else 100.0
        league_def = float(np.mean(list(adj_def.values()))) if adj_def else 100.0

        if len(games) < 8:
            return 0.0

        recent = games[-5:]
        recent_margins = []
        for g in recent:
            opp_em = adj_off.get(g.opponent_id, league_off) - adj_def.get(g.opponent_id, league_def)
            recent_margins.append((g.points - g.opp_points) - opp_em)

        season_margins = []
        for g in games:
            opp_em = adj_off.get(g.opponent_id, league_off) - adj_def.get(g.opponent_id, league_def)
            season_margins.append((g.points - g.opp_points) - opp_em)

        raw_delta = float(np.mean(recent_margins)) - float(np.mean(season_margins))
        n_recent = len(recent_margins)
        lam = (MARGIN_SIGMA_SQ / n_recent) / (MARGIN_SIGMA_SQ / n_recent + MOMENTUM_SIGNAL_SIGMA_SQ)
        return raw_delta * (1.0 - lam)

    def _pace_variance(self, games: List[GameRecord]) -> float:
        """Game-to-game pace stdev."""
        if len(games) < 5:
            return 0.0

        per_game_pace = [g.possessions for g in games if g.possessions > 20]
        if len(per_game_pace) < 5:
            return 0.0

        return float(np.std(per_game_pace, ddof=1))

    def _compute_rest_days(
        self,
        results: Dict[str, ProprietaryTeamMetrics],
        by_team: Dict[str, List[GameRecord]],
        reference_date: Optional[str] = None,
    ) -> None:
        """Compute days since last game for each team."""
        from datetime import datetime as _dt

        ref_dt = None
        if reference_date:
            try:
                ref_dt = _dt.strptime(reference_date, "%Y-%m-%d")
            except (ValueError, TypeError):
                pass

        for tid, games in by_team.items():
            if tid not in results or not games:
                continue
            try:
                if ref_dt is not None:
                    last_date = _dt.strptime(games[-1].game_date, "%Y-%m-%d")
                    delta = (ref_dt - last_date).days
                    results[tid].rest_days = float(max(delta, 0))
                elif len(games) >= 2:
                    last = _dt.strptime(games[-1].game_date, "%Y-%m-%d")
                    prev = _dt.strptime(games[-2].game_date, "%Y-%m-%d")
                    delta = (last - prev).days
                    results[tid].rest_days = float(max(delta, 0))
                else:
                    results[tid].rest_days = 5.0
            except (ValueError, TypeError):
                results[tid].rest_days = 5.0

    def compute_h2h_record(self, team1_id: str, team2_id: str) -> float:
        """Team1 win% in H2H games vs team2 (0.5 if no meetings)."""
        if not hasattr(self, "_by_team"):
            return 0.5

        games = self._by_team.get(team1_id, [])
        h2h_wins = 0
        h2h_total = 0
        for g in games:
            if g.opponent_id == team2_id:
                h2h_total += 1
                if g.points > g.opp_points:
                    h2h_wins += 1

        if h2h_total == 0:
            return 0.5

        weight = min(1.0, h2h_total / 4.0)
        raw_rate = h2h_wins / h2h_total
        return weight * raw_rate + (1.0 - weight) * 0.5

    def compute_common_opponent_margin(self, team1_id: str, team2_id: str) -> float:
        """Normalized margin differential through common opponents."""
        if not hasattr(self, "_by_team"):
            return 0.0

        def _opp_margins(team_id: str) -> Dict[str, List[float]]:
            margins: Dict[str, List[float]] = defaultdict(list)
            for g in self._by_team.get(team_id, []):
                margin = g.points - g.opp_points
                margin = float(np.clip(margin, -self.MARGIN_CAP, self.MARGIN_CAP))
                margins[g.opponent_id].append(margin)
            return margins

        m1 = _opp_margins(team1_id)
        m2 = _opp_margins(team2_id)

        common_opps = set(m1.keys()) & set(m2.keys())
        if not common_opps:
            return 0.0

        diffs = []
        for opp in common_opps:
            avg1 = float(np.mean(m1[opp]))
            avg2 = float(np.mean(m2[opp]))
            diffs.append(avg1 - avg2)

        raw = float(np.mean(diffs))
        return float(np.clip(raw / 20.0, -1.5, 1.5))

    @staticmethod
    def _log5_win_prob(team_a_em: float, team_b_em: float) -> float:
        """Win probability from efficiency margin differential (k=11.5)."""
        diff = team_a_em - team_b_em
        diff = float(np.clip(diff, -40.0, 40.0))
        return 1.0 / (1.0 + 10 ** (-diff / 11.5))


# Converter: team_games JSON -> GameRecord

def team_games_to_game_records(
    team_games: List[Dict],
    season_year: int,
) -> List[GameRecord]:
    """Convert team_games arrays from historical JSON to GameRecord objects."""
    row_index: Dict[Tuple[str, str], Dict] = {}
    for row in team_games:
        gid = str(row.get("game_id", ""))
        tid = _team_id(str(row.get("team_id", "")))
        if gid and tid:
            row_index[(gid, tid)] = row

    records: List[GameRecord] = []
    seen: set = set()

    for row in team_games:
        gid = str(row.get("game_id", ""))
        raw_tid = _team_id(str(row.get("team_id", "")))
        raw_oid = _team_id(str(row.get("opponent_id", "")))
        if not gid or not raw_tid or not raw_oid:
            continue

        dedup_key = (gid, raw_tid)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        points = _to_float(row.get("team_score", 0))
        opp_points = _to_float(row.get("opponent_score", 0))
        if points == 0 and opp_points == 0:
            continue

        fgm = _to_float(row.get("fgm", 0))
        fga = _to_float(row.get("fga", 0))
        fg3m = _to_float(row.get("fg3m", 0))
        fg3a = _to_float(row.get("fg3a", 0))
        fta = _to_float(row.get("fta", 0))
        tov = _to_float(row.get("turnovers") or row.get("tov", 0))
        orb = _to_float(row.get("orb", 0))
        drb = _to_float(row.get("drb", 0))

        row_has_box = row.get("has_box_score")
        if row_has_box is None:
            row_has_box = fga > 0
        else:
            row_has_box = bool(row_has_box)

        ftm = max(points - 2.0 * fgm - fg3m, 0.0) if fgm > 0 else 0.0

        poss = _to_float(row.get("possessions", 0))
        if poss <= 0 and fga > 0:
            poss = fga - orb + tov + 0.475 * fta
        if poss <= 0:
            poss = max((points + opp_points) / 2.0, 30.0)

        mirror = row_index.get((gid, raw_oid))
        if mirror:
            opp_fgm = _to_float(mirror.get("fgm", 0))
            opp_fga = _to_float(mirror.get("fga", 0))
            opp_fg3m = _to_float(mirror.get("fg3m", 0))
            opp_fg3a = _to_float(mirror.get("fg3a", 0))
            opp_fta = _to_float(mirror.get("fta", 0))
            opp_tov = _to_float(mirror.get("turnovers") or mirror.get("tov", 0))
            opp_orb = _to_float(mirror.get("orb", 0))
            opp_drb = _to_float(mirror.get("drb", 0))
            opp_ftm = max(opp_points - 2.0 * opp_fgm - opp_fg3m, 0.0) if opp_fgm > 0 else 0.0
        else:
            opp_fgm = opp_fga = opp_fg3m = opp_fg3a = 0.0
            opp_fta = opp_ftm = opp_tov = opp_orb = opp_drb = 0.0

        raw_date = row.get("date") or row.get("game_date")
        game_date = str(raw_date or f"{season_year}-01-01")
        team_name = str(row.get("team_name", raw_tid))

        records.append(
            GameRecord(
                game_id=gid,
                game_date=game_date,
                team_id=raw_tid,
                team_name=team_name,
                opponent_id=raw_oid,
                points=points,
                opp_points=opp_points,
                possessions=poss,
                fga=fga,
                fgm=fgm,
                fg3a=fg3a,
                fg3m=fg3m,
                fta=fta,
                ftm=ftm,
                tov=tov,
                orb=orb,
                drb=drb,
                opp_fga=opp_fga,
                opp_fgm=opp_fgm,
                opp_fg3a=opp_fg3a,
                opp_fg3m=opp_fg3m,
                opp_fta=opp_fta,
                opp_ftm=opp_ftm,
                opp_tov=opp_tov,
                opp_orb=opp_orb,
                opp_drb=opp_drb,
                is_home=False,
                is_neutral=True,
                has_box_score=row_has_box,
            )
        )

    # Date inference: when all games share a single placeholder date,
    # infer chronological dates from game_id ordering.
    unique_dates = set(r.game_date for r in records)
    if len(unique_dates) <= 1 and len(records) > 50:
        records.sort(key=lambda r: (r.game_id, r.team_id))
        from datetime import date as _date, timedelta as _td

        season_start = _date(season_year - 1, 11, 1)
        season_end = _date(season_year, 4, 10)
        total_days = (season_end - season_start).days
        gid_order: Dict[str, int] = {}
        idx = 0
        for r in records:
            if r.game_id not in gid_order:
                gid_order[r.game_id] = idx
                idx += 1
        n_unique_games = max(idx, 1)
        for r in records:
            rank = gid_order[r.game_id]
            frac = rank / max(n_unique_games - 1, 1)
            day_offset = (int(frac * total_days) // 30) * 30
            inferred = season_start + _td(days=day_offset)
            r.game_date = inferred.isoformat()
        logger.info(
            "Year %d: inferred dates for %d games -> %d monthly buckets.",
            season_year,
            n_unique_games,
            len(set(r.game_date for r in records)),
        )

    logger.info(
        "Year %d: converted %d team_games rows -> %d GameRecords.",
        season_year,
        len(team_games),
        len(records),
    )
    return records


# IncrementalMetricsEngine

class IncrementalMetricsEngine:
    """Compute team metrics incrementally using only games before a given date."""

    def __init__(
        self,
        game_records: List[GameRecord],
        conference_map: Optional[Dict[str, str]] = None,
        prior_elo: Optional[Dict[str, float]] = None,
    ):
        self._conference_map = conference_map
        self._prior_elo = prior_elo
        self._all_records = sorted(game_records, key=lambda g: g.game_date)

        date_set: set = set()
        for g in self._all_records:
            date_set.add(g.game_date)
        self._unique_dates = sorted(date_set)

        self._elo_snapshots: Dict[str, Dict[str, float]] = {}
        self._compute_all_elo_snapshots()

        self._cache: Dict[str, Dict[str, ProprietaryTeamMetrics]] = {}
        self._by_team_cache: Dict[str, Dict[str, List[GameRecord]]] = {}

    def _compute_all_elo_snapshots(self) -> None:
        """Process all games chronologically and snapshot Elo at each date."""
        _ELO_REGRESSION = 0.25
        _ELO_MEAN = 1500.0
        K_BASE = 38.0
        ELO_HCA = 3.75 * 13.3

        elo: Dict[str, float] = defaultdict(lambda: _ELO_MEAN)
        if self._prior_elo:
            for tid, val in self._prior_elo.items():
                elo[tid] = (1.0 - _ELO_REGRESSION) * val + _ELO_REGRESSION * _ELO_MEAN

        seen_ids: set = set()
        deduped: List[GameRecord] = []
        for g in self._all_records:
            pair = (g.game_id, min(g.team_id, g.opponent_id))
            if pair not in seen_ids:
                seen_ids.add(pair)
                deduped.append(g)

        games_by_date: Dict[str, List[GameRecord]] = defaultdict(list)
        for g in deduped:
            games_by_date[g.game_date].append(g)

        for date in self._unique_dates:
            self._elo_snapshots[date] = dict(elo)

            for g in games_by_date.get(date, []):
                t1, t2 = g.team_id, g.opponent_id
                _ = elo[t1]
                _ = elo[t2]

                hca = 0.0
                if not g.is_neutral:
                    hca = ELO_HCA if g.is_home else -ELO_HCA

                e1 = 1.0 / (1.0 + 10 ** (-(elo[t1] + hca - elo[t2]) / 400.0))
                margin = g.points - g.opp_points
                s1 = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)
                mov_mult = np.log1p(abs(margin))
                elo_diff = abs(elo[t1] - elo[t2])
                elo_dampening = 2.2 / (elo_diff * 0.001 + 2.2)
                k = K_BASE * mov_mult * elo_dampening
                delta = k * (s1 - e1)
                elo[t1] += delta
                elo[t2] -= delta

        self._end_of_season_elo = dict(elo)

    def get_end_of_season_elo(self) -> Dict[str, float]:
        """Return end-of-season Elo for cross-season carryover."""
        return dict(self._end_of_season_elo)

    def games_played_before(self, team_id: str, date: str) -> int:
        """Return count of games played by team_id strictly before date."""
        if not hasattr(self, "_games_before_cache"):
            from collections import defaultdict
            import bisect

            team_dates: Dict[str, list] = defaultdict(list)
            for g in self._all_records:
                team_dates[g.team_id].append(g.game_date)
            self._games_before_cache = {tid: sorted(dates) for tid, dates in team_dates.items()}
        dates = self._games_before_cache.get(team_id)
        if not dates:
            return 0
        import bisect

        return bisect.bisect_left(dates, date)

    def compute_as_of(self, as_of_date: str) -> Dict[str, ProprietaryTeamMetrics]:
        """Compute metrics for all teams using only games before as_of_date."""
        if as_of_date in self._cache:
            self._by_team = self._by_team_cache.get(as_of_date, {})
            return self._cache[as_of_date]

        prefix = [g for g in self._all_records if g.game_date < as_of_date]
        if len(prefix) < 50:
            self._by_team_cache[as_of_date] = {}
            self._cache[as_of_date] = {}
            return {}

        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        engine._elo_prior = self._prior_elo

        by_team: Dict[str, List[GameRecord]] = defaultdict(list)
        for rec in prefix:
            by_team[rec.team_id].append(rec)
        for tid in by_team:
            by_team[tid].sort(key=lambda g: g.game_date)
        self._by_team = by_team
        self._by_team_cache[as_of_date] = by_team

        raw_off, raw_def, tempo, names = engine._raw_efficiency(by_team)
        adj_off, adj_def = engine._iterative_sos_adjust(by_team, raw_off, raw_def)

        engine._by_team = dict(by_team)
        engine._adj_off = adj_off
        engine._adj_def = adj_def

        all_team_ids = sorted(by_team.keys())

        results: Dict[str, ProprietaryTeamMetrics] = {}
        for tid in all_team_ids:
            games = by_team[tid]
            box_games = [g for g in games if g.has_box_score]
            adj_o = adj_off[tid]
            adj_d = adj_def[tid]
            n_games = len(games)

            ff = engine._four_factors(box_games)
            shooting = engine._supplementary_shooting(box_games)
            sos = engine._strength_of_schedule(games, adj_off, adj_def, self._conference_map)
            luck = engine._correlated_gaussian_luck(games)
            barthag = engine._pythagorean_win_pct(adj_o, adj_d)
            xp_o = engine._box_score_xp(ff, side="offense", ft_pct=shooting.get("free_throw_pct", 0.72))
            xp_d = engine._box_score_xp(ff, side="defense", ft_pct=shooting.get("opp_free_throw_pct", 0.72))
            shot_dist = engine._shot_distribution_score(box_games)
            tpv = engine._three_point_variance(box_games)
            pav = engine._pace_adjusted_variance(games)
            consistency = engine._consistency(games)
            sos_consistency = engine._sos_adjusted_consistency(games, adj_off, adj_def)
            mom_delta, recent_em = engine._momentum(games, adj_off, adj_def)
            mom5_delta = engine._momentum_5g(games, adj_off, adj_def)
            ext = engine._extended_box_score_metrics(box_games)
            opp_shots = engine._opponent_shot_selection(box_games)
            ts_pct, opp_ts_pct = engine._true_shooting_pct(box_games)
            home_em, away_em, hc_dep = engine._home_away_splits(games, adj_off, adj_def)
            nsw, nsg = engine._neutral_site_record(games)

            wins = sum(1 for g in games if g.points > g.opp_points)
            losses = n_games - wins

            results[tid] = ProprietaryTeamMetrics(
                team_id=tid,
                team_name=names.get(tid, tid),
                adj_offensive_efficiency=adj_o,
                adj_defensive_efficiency=adj_d,
                adj_efficiency_margin=adj_o - adj_d,
                adj_tempo=tempo.get(tid, 67.0),
                effective_fg_pct=ff.get("effective_fg_pct", 0.0),
                turnover_rate=ff.get("turnover_rate", 0.0),
                offensive_reb_rate=ff.get("offensive_reb_rate", 0.0),
                free_throw_rate=ff.get("free_throw_rate", 0.0),
                opp_effective_fg_pct=ff.get("opp_effective_fg_pct", 0.0),
                opp_turnover_rate=ff.get("opp_turnover_rate", 0.0),
                defensive_reb_rate=ff.get("defensive_reb_rate", 0.0),
                opp_free_throw_rate=ff.get("opp_free_throw_rate", 0.0),
                sos_adj_em=sos.get("sos_adj_em", 0.0),
                sos_opp_o=sos.get("sos_opp_o", 0.0),
                sos_opp_d=sos.get("sos_opp_d", 0.0),
                ncsos_adj_em=sos.get("ncsos_adj_em", 0.0),
                luck=luck,
                barthag=barthag,
                offensive_xp_per_possession=xp_o,
                defensive_xp_per_possession=xp_d,
                shot_distribution_score=shot_dist,
                three_pt_variance=tpv,
                pace_adjusted_variance=pav,
                consistency=consistency,
                sos_adjusted_consistency=sos_consistency,
                momentum=mom_delta,
                recent_adj_em=recent_em,
                momentum_5g=mom5_delta,
                wab=0.0,
                wins=wins,
                losses=losses,
                win_pct=wins / max(n_games, 1),
                two_pt_pct=shooting.get("two_pt_pct", 0.0),
                three_pt_pct=shooting.get("three_pt_pct", 0.0),
                three_pt_rate=shooting.get("three_pt_rate", 0.0),
                free_throw_pct=shooting.get("free_throw_pct", 0.0),
                opp_free_throw_pct=shooting.get("opp_free_throw_pct", 0.0),
                opp_true_shooting_pct=opp_ts_pct,
                true_shooting_pct=ts_pct,
                efficiency_ratio=adj_o / max(adj_d, 1.0),
                elo_rating=0.0,
                assist_to_turnover_ratio=ext.get("assist_to_turnover_ratio", 0.0),
                assist_rate=ext.get("assist_rate", 0.0),
                steal_rate=ext.get("steal_rate", 0.0),
                block_rate=ext.get("block_rate", 0.0),
                defensive_disruption_rate=ext.get("defensive_disruption_rate", 0.0),
                opp_two_pt_pct_allowed=opp_shots.get("opp_two_pt_pct_allowed", 0.0),
                opp_three_pt_attempt_rate=opp_shots.get("opp_three_pt_attempt_rate", 0.0),
                neutral_site_win_pct=nsw,
                neutral_site_games=nsg,
                home_adj_em=home_em,
                away_adj_em=away_em,
                home_court_dependence=hc_dep,
            )

        engine._compute_wab(results, by_team)
        engine._compute_sor_and_wab_poisson(results, by_team)
        engine._compute_elite_sos_and_quadrants(results, by_team, adj_off, adj_def)
        engine._compute_conference_strength(results, by_team, self._conference_map or {})
        engine._compute_rest_days(results, by_team, reference_date=as_of_date)

        for tid in all_team_ids:
            games = by_team[tid]
            total_pf = sum(g.pf for g in games)
            total_poss = sum(max(g.possessions, 1.0) for g in games)
            results[tid].foul_rate = total_pf / max(total_poss, 1.0)

        for tid in all_team_ids:
            raw_3p = results[tid].three_pt_pct
            n = len(by_team[tid])
            league_mean_3p = 0.345
            k_prior = 50.0
            shrunk_3p = (n * raw_3p + k_prior * league_mean_3p) / (n + k_prior)
            results[tid].three_pt_regression_signal = shrunk_3p - league_mean_3p

        for tid in all_team_ids:
            games = by_team[tid]
            if len(games) >= 5:
                poss_list = [g.possessions for g in games]
                results[tid].pace_variance = float(np.std(poss_list, ddof=1))
            else:
                results[tid].pace_variance = 5.0

        # Override Elo with incremental snapshots (bisect for non-game dates).
        _idx = bisect.bisect_left(self._unique_dates, as_of_date)
        if _idx < len(self._unique_dates):
            elo_snap = self._elo_snapshots[self._unique_dates[_idx]]
        else:
            elo_snap = self.get_end_of_season_elo()
        for tid in all_team_ids:
            if tid in elo_snap:
                results[tid].elo_rating = elo_snap[tid]

        if self._conference_map:
            self._compute_conf_tourney_features(results, by_team, self._conference_map, as_of_date)

        self._cache[as_of_date] = results
        return results

    @staticmethod
    def _compute_conf_tourney_features(
        results: Dict[str, "ProprietaryTeamMetrics"],
        by_team: Dict[str, List["GameRecord"]],
        conference_map: Dict[str, str],
        as_of_date: str,
    ) -> None:
        """Populate conference tournament + late-season recency features."""
        from datetime import timedelta

        try:
            from ...pipeline.config import TOURNAMENT_START_DATES
        except ImportError:
            return

        year = int(as_of_date[:4])
        tourney_start = TOURNAMENT_START_DATES.get(year)
        if tourney_start is None:
            return

        window_start = (tourney_start - timedelta(days=12)).isoformat()
        window_end = (tourney_start - timedelta(days=1)).isoformat()

        if as_of_date <= window_start:
            return

        for tid, metrics in results.items():
            conf = conference_map.get(tid)
            conf_games = []
            all_window_games = []

            for g in by_team.get(tid, []):
                if g.game_date < window_start or g.game_date > window_end:
                    continue
                if g.game_date >= as_of_date:
                    continue

                all_window_games.append(g)
                if conf and conference_map.get(g.opponent_id) == conf:
                    conf_games.append(g)

            n_conf = min(len(conf_games), 5)
            metrics.conf_tourney_games = n_conf
            if n_conf > 0:
                metrics.conf_tourney_margin = sum(
                    g.points - g.opp_points for g in conf_games
                ) / n_conf

            n_all = len(all_window_games)
            metrics.late_season_games = n_all
            if n_all > 0:
                metrics.late_season_margin = sum(
                    g.points - g.opp_points for g in all_window_games
                ) / n_all
                metrics.late_season_win_pct = sum(
                    1 for g in all_window_games if g.points > g.opp_points
                ) / n_all

    def compute_h2h_record(self, team1_id: str, team2_id: str) -> float:
        """Team1 win rate vs team2 from current point-in-time games."""
        if not hasattr(self, "_by_team"):
            return 0.5

        games = self._by_team.get(team1_id, [])
        h2h_wins = 0
        h2h_total = 0
        for g in games:
            if g.opponent_id == team2_id:
                h2h_total += 1
                if g.points > g.opp_points:
                    h2h_wins += 1

        if h2h_total == 0:
            return 0.5

        weight = min(1.0, h2h_total / 4.0)
        raw_rate = h2h_wins / h2h_total
        return weight * raw_rate + (1.0 - weight) * 0.5

    def compute_common_opponent_margin(self, team1_id: str, team2_id: str) -> float:
        """Normalized margin differential through common opponents."""
        if not hasattr(self, "_by_team"):
            return 0.0

        margin_cap = float(getattr(self, "MARGIN_CAP", 30.0))

        def _opp_margins(team_id: str) -> Dict[str, List[float]]:
            margins: Dict[str, List[float]] = defaultdict(list)
            for g in self._by_team.get(team_id, []):
                margin = g.points - g.opp_points
                margin = float(np.clip(margin, -margin_cap, margin_cap))
                margins[g.opponent_id].append(margin)
            return margins

        m1 = _opp_margins(team1_id)
        m2 = _opp_margins(team2_id)
        common_opps = set(m1.keys()) & set(m2.keys())
        if not common_opps:
            return 0.0

        diffs = []
        for opp in common_opps:
            avg1 = float(np.mean(m1[opp]))
            avg2 = float(np.mean(m2[opp]))
            diffs.append(avg1 - avg2)

        raw = float(np.mean(diffs))
        return float(np.clip(raw / 20.0, -1.5, 1.5))

    @staticmethod
    def metrics_to_team_vector(
        m: ProprietaryTeamMetrics,
        seed: int = 0,
        external_rating_composite: float = float("nan"),
        external_rating_spread: float = float("nan"),
        massey_features=None,
    ) -> np.ndarray:
        """Convert ProprietaryTeamMetrics to the canonical team feature vector."""
        from .feature_engineering import TEAM_FEATURE_DIM

        v = np.zeros(TEAM_FEATURE_DIM, dtype=np.float64)
        v[0] = m.adj_offensive_efficiency
        v[1] = m.adj_defensive_efficiency
        v[2] = m.adj_tempo
        v[3] = m.effective_fg_pct
        v[4] = m.turnover_rate
        v[5] = m.offensive_reb_rate
        v[6] = m.free_throw_rate
        v[7] = m.opp_effective_fg_pct
        v[8] = m.opp_turnover_rate
        v[9] = m.defensive_reb_rate
        v[10] = m.opp_free_throw_rate
        v[18] = m.offensive_xp_per_possession
        v[19] = m.shot_distribution_score
        v[20] = m.sos_adj_em
        v[21] = m.sos_opp_o
        v[22] = m.sos_opp_d
        v[23] = m.ncsos_adj_em
        v[24] = m.luck
        v[25] = m.wab_poisson
        v[26] = m.momentum
        v[27] = m.three_pt_variance
        v[28] = m.pace_adjusted_variance
        v[29] = m.elo_rating
        v[30] = m.opp_two_pt_pct_allowed
        v[31] = m.opp_three_pt_attempt_rate
        v[32] = m.conference_adj_em
        v[33] = m.three_pt_pct
        v[34] = m.three_pt_rate
        v[35] = m.defensive_xp_per_possession
        v[36] = m.win_pct
        v[37] = m.three_pt_regression_signal
        v[38] = min(m.rest_days, 14.0)
        v[39] = 0.0
        v[40] = m.pace_variance
        v[41] = m.neutral_site_win_pct
        from .tournament_features import compute_tournament_resume_composite

        v[42] = compute_tournament_resume_composite(
            q1_win_pct=m.q1_win_pct,
            q1_games=m.q1_wins + m.q1_losses,
            road_neutral_win_pct=m.road_neutral_win_pct,
            road_neutral_games=m.road_neutral_games,
            elite_sos=m.elite_sos,
            sor=m.sor,
        )
        v[43] = 0.0
        v[44] = 0.0
        if seed > 0:
            v[45] = float(np.log1p(17 - seed) / np.log1p(16))
        else:
            v[45] = 0.0
        v[46] = float(m.conf_tourney_champion)
        v[47] = float(m.conf_tourney_games)
        v[48] = m.conf_tourney_margin
        v[49] = float(m.late_season_games)
        v[50] = m.late_season_margin
        v[51] = m.late_season_win_pct
        v[52] = 0.0
        v[53] = 0.0

        inf_mask = np.isinf(v)
        if inf_mask.any():
            v[inf_mask] = np.nan
        return v

    @staticmethod
    def build_matchup_vector(
        v1: np.ndarray,
        v2: np.ndarray,
        seed1: int = 0,
        seed2: int = 0,
        engine: Optional["IncrementalMetricsEngine"] = None,
        team1_id: str = "",
        team2_id: str = "",
    ) -> np.ndarray:
        """Build matchup vector from two team feature vectors."""
        diff = v1 - v2

        from .feature_engineering import ABSOLUTE_LEVEL_INDICES

        _ABS_IDX = ABSOLUTE_LEVEL_INDICES if ABSOLUTE_LEVEL_INDICES else [0, 1, 26, 37, 49]
        absolute = np.array([(v1[i] + v2[i]) / 2.0 for i in _ABS_IDX])

        tempo_interaction = (v1[2] * v2[2]) / 4624.0
        tempo_diff = v1[2] - v2[2]
        eff_diff = (v1[0] - v1[1]) - (v2[0] - v2[1])
        style_mismatch = (tempo_diff * eff_diff) / 600.0

        _SEED_EXPECTED_EM = {
            1: 28, 2: 21, 3: 16, 4: 12, 5: 9, 6: 6, 7: 4, 8: 2,
            9: 0, 10: -2, 11: -4, 12: -6, 13: -9, 14: -12, 15: -16, 16: -21,
        }
        residual1 = (v1[0] - v1[1]) - _SEED_EXPECTED_EM.get(seed1, 0)
        residual2 = (v2[0] - v2[1]) - _SEED_EXPECTED_EM.get(seed2, 0)
        seed_em_residual_diff = (residual1 - residual2) / 20.0

        sos_seed_interaction = ((v1[26] - v2[26]) * (seed1 - seed2)) / 200.0

        var_diff = v1[35] - v2[35]
        three_pt_var_seed_interaction = var_diff * (seed1 - seed2) / 15.0

        if seed1 > 0 and seed2 > 0:
            seed_interaction = (seed1 * seed2) / 128.0 - 1.0
            seed_diff = (seed1 - seed2) / 15.0
        else:
            seed_interaction = 0.0
            seed_diff = 0.0

        interactions = np.array([
            tempo_interaction, style_mismatch, seed_em_residual_diff,
            sos_seed_interaction, three_pt_var_seed_interaction,
            seed_interaction, seed_diff,
        ])

        result = np.concatenate([diff, absolute, interactions])
        from .feature_engineering import MATCHUP_DIM

        assert result.shape[0] == MATCHUP_DIM, (
            f"build_matchup_vector produced {result.shape[0]}-dim vector, "
            f"expected MATCHUP_DIM={MATCHUP_DIM}."
        )
        return result


# CBBpy team-map CSV loader

_CBBPY_TEAM_MAP_CACHE: Optional[Dict[str, str]] = None


def _load_cbbpy_team_map(csv_path: Optional[str] = None) -> Dict[str, str]:
    """Load CBBpy team map CSV: {display_name -> location}."""
    global _CBBPY_TEAM_MAP_CACHE
    if _CBBPY_TEAM_MAP_CACHE is not None:
        return _CBBPY_TEAM_MAP_CACHE

    if csv_path is None:
        _here = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(_here, "..", "..", "..", "data", "raw", "cbbpy_team_map.csv")

    result: Dict[str, str] = {}
    if not os.path.exists(csv_path):
        _CBBPY_TEAM_MAP_CACHE = result
        return result

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            team = row.get("team", "").strip()
            location = row.get("location", "").strip()
            if team and location:
                result[team] = location

    _CBBPY_TEAM_MAP_CACHE = result
    return result


# Converter: Torvik/public data -> GameRecord

def torvik_to_game_records(
    torvik_teams: List[Dict],
    historical_games: List[Dict],
    season_year: int,
) -> List[GameRecord]:
    """Convert Torvik team stats + historical game rows into GameRecord objects."""
    _torvik_name_to_id: Dict[str, str] = {}
    for t in torvik_teams:
        if isinstance(t, dict):
            tid = t.get("team_id", "")
            tname = t.get("name", "")
        else:
            tid = getattr(t, "team_id", "")
            tname = getattr(t, "name", "")
        if tid and tname:
            canon = _team_id(tid)
            _torvik_name_to_id[_team_id(tname)] = canon
            _torvik_name_to_id[canon] = canon
            cleaned = tname.replace("&amp;", "&")
            if cleaned != tname:
                _torvik_name_to_id[_team_id(cleaned)] = canon
            stripped = re.sub(r"\s*\([^)]*\)\s*", "", tname).strip()
            if stripped != tname:
                _torvik_name_to_id[_team_id(stripped)] = canon
                stripped_clean = re.sub(r"\s*\([^)]*\)\s*", "", cleaned).strip()
                if stripped_clean != stripped:
                    _torvik_name_to_id[_team_id(stripped_clean)] = canon

    from src.data.normalize import _QUICK_ALIAS as _shared_aliases

    for alias, target in _shared_aliases.items():
        if target in _torvik_name_to_id and alias not in _torvik_name_to_id:
            _torvik_name_to_id[alias] = _torvik_name_to_id[target]

    _cbbpy_display_to_location = _load_cbbpy_team_map()

    _name_by_raw_id: Dict[str, str] = {}
    for game in historical_games:
        if not isinstance(game, dict):
            continue
        raw = _team_id(
            str(game.get("team_id") or game.get("team1_id") or game.get("team1") or game.get("home_team") or "")
        )
        name = str(game.get("team_name") or game.get("team1_name") or "")
        if raw and name and raw not in _name_by_raw_id:
            _name_by_raw_id[raw] = name
        raw2 = _team_id(
            str(game.get("opponent_id") or game.get("team2_id") or game.get("team2") or game.get("away_team") or "")
        )
        name2 = str(game.get("opponent_name") or game.get("team2_name") or "")
        if raw2 and name2 and raw2 not in _name_by_raw_id:
            _name_by_raw_id[raw2] = name2

    _canonical_cache: Dict[str, str] = {}
    for raw_id, display_name in _name_by_raw_id.items():
        location = _cbbpy_display_to_location.get(display_name)
        if location:
            norm_location = _team_id(location)
            canon = _torvik_name_to_id.get(norm_location)
            if canon:
                _canonical_cache[raw_id] = canon

    _torvik_id_set = set(_torvik_name_to_id.values())

    def _resolve_canonical(raw_id: str) -> str:
        if raw_id in _canonical_cache:
            return _canonical_cache[raw_id]
        if raw_id in _torvik_id_set:
            _canonical_cache[raw_id] = raw_id
            return raw_id
        _canonical_cache[raw_id] = raw_id
        return raw_id

    records: List[GameRecord] = []
    for game in historical_games:
        if not isinstance(game, dict):
            continue

        game_id = str(game.get("game_id") or game.get("id") or "")
        raw_team = _team_id(
            str(game.get("team_id") or game.get("team1_id") or game.get("team1") or game.get("home_team") or "")
        )
        raw_opp = _team_id(
            str(game.get("opponent_id") or game.get("team2_id") or game.get("team2") or game.get("away_team") or "")
        )
        team_id = _resolve_canonical(raw_team)
        opp_id = _resolve_canonical(raw_opp)
        if not game_id or not team_id or not opp_id:
            continue

        points = _to_float(
            game.get("team_score") or game.get("team1_score") or game.get("home_score") or game.get("points") or 0
        )
        opp_points = _to_float(
            game.get("opponent_score")
            or game.get("team2_score")
            or game.get("away_score")
            or game.get("opp_points")
            or 0
        )

        fga = _to_float(game.get("fga", 0))
        fgm = _to_float(game.get("fgm", 0))
        fg3a = _to_float(game.get("fg3a") or game.get("x3pa", 0))
        fg3m = _to_float(game.get("fg3m") or game.get("x3pm", 0))
        fta = _to_float(game.get("fta", 0))
        ftm = _to_float(game.get("ftm", 0))
        tov = _to_float(game.get("turnovers") or game.get("tov", 0))
        orb = _to_float(game.get("orb") or game.get("offensive_rebounds", 0))
        drb = _to_float(game.get("drb") or game.get("defensive_rebounds", 0))

        game_has_box = game.get("has_box_score")
        if game_has_box is None:
            game_has_box = fga > 0
        else:
            game_has_box = bool(game_has_box)

        opp_fga = _to_float(game.get("opp_fga", 0))
        opp_fgm = _to_float(game.get("opp_fgm", 0))
        opp_fg3a = _to_float(game.get("opp_fg3a", 0))
        opp_fg3m = _to_float(game.get("opp_fg3m", 0))
        opp_fta = _to_float(game.get("opp_fta", 0))
        opp_ftm = _to_float(game.get("opp_ftm", 0))
        opp_tov = _to_float(game.get("opp_tov") or game.get("opp_turnovers", 0))
        opp_orb = _to_float(game.get("opp_orb", 0))
        opp_drb = _to_float(game.get("opp_drb", 0))

        ast = _to_float(game.get("ast") or game.get("assists", 0))
        stl = _to_float(game.get("stl") or game.get("steals", 0))
        blk = _to_float(game.get("blk") or game.get("blocks", 0))
        pf = _to_float(game.get("pf") or game.get("personal_fouls") or game.get("fouls", 0))
        opp_ast = _to_float(game.get("opp_ast") or game.get("opp_assists", 0))
        opp_stl = _to_float(game.get("opp_stl") or game.get("opp_steals", 0))
        opp_blk = _to_float(game.get("opp_blk") or game.get("opp_blocks", 0))
        opp_pf = _to_float(game.get("opp_pf") or game.get("opp_personal_fouls") or game.get("opp_fouls", 0))

        poss = _to_float(game.get("possessions", 0))
        if poss <= 0 and fga > 0:
            poss = fga - orb + tov + 0.475 * fta
            opp_poss = opp_fga - opp_orb + opp_tov + 0.475 * opp_fta
            if opp_poss > 0:
                poss = (poss + opp_poss) / 2.0
        if poss <= 0:
            poss = max((points + opp_points) / 2.0, 30.0)

        is_neutral = bool(game.get("neutral_site", game.get("is_neutral", False)))
        is_home = bool(game.get("is_home", not is_neutral))

        raw_date = game.get("game_date") or game.get("date") or game.get("start_date")
        if not raw_date:
            logger.warning("Game %s missing date; using %d-01-01.", game_id or "unknown", season_year)
        game_date = str(raw_date or f"{season_year}-01-01")
        team_name = str(game.get("team_name") or game.get("team1_name") or game.get("team1") or team_id)

        records.append(
            GameRecord(
                game_id=game_id,
                game_date=game_date,
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
                ftm=ftm,
                tov=tov,
                orb=orb,
                drb=drb,
                ast=ast,
                stl=stl,
                blk=blk,
                pf=pf,
                opp_fga=opp_fga,
                opp_fgm=opp_fgm,
                opp_fg3a=opp_fg3a,
                opp_fg3m=opp_fg3m,
                opp_fta=opp_fta,
                opp_ftm=opp_ftm,
                opp_tov=opp_tov,
                opp_orb=opp_orb,
                opp_drb=opp_drb,
                opp_ast=opp_ast,
                opp_stl=opp_stl,
                opp_blk=opp_blk,
                opp_pf=opp_pf,
                is_home=is_home,
                is_neutral=is_neutral,
                has_box_score=game_has_box,
            )
        )

    return records


def _team_id(name: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
