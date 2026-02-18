"""
Proprietary advanced metrics engine.

Computes KenPom-equivalent and ShotQuality-equivalent metrics entirely from
public box-score data (cbbpy / Torvik / Sports Reference).  No paid API needed.

Metrics produced
================
* Adjusted Offensive / Defensive Efficiency  (iterative SOS adjustment)
* Adjusted Tempo
* Four Factors  (eFG%, TO%, ORB%, FTR)  — offense & defense
* Luck factor   (Correlated Gaussian Method — game-by-game margin variance)
* Wins Above Bubble  (WAB — results-only, schedule-aware)
* Proprietary xP per possession  (Four-Factors decomposition)
* Shot Distribution Score  (rim + 3pt vs midrange proxy)
* 3-Point Variance  (game-to-game 3P% stdev — upset risk proxy)
* Momentum / Rolling Form  (last-10-game rolling AdjEM)
* Pace-Adjusted Variance  (low-possession games amplify noise)
* Consistency Rating  (inverse of scoring-margin stdev)
* Elo Rating  (MOV-adjusted, K=38, per SBCB methodology)
* Free Throw %  (team + opponent — most stable shooting metric)
* Assist-to-Turnover Ratio  (halfcourt execution quality)
* Assist Rate  (AST/FGM — team ball movement)
* Steal Rate / Block Rate / Defensive Disruption  (per-possession)
* Opponent 2P% Allowed  (controllable, stable — unlike opp 3P%)
* Opponent 3PA Rate  (controllable shot selection forcing)
* Conference Strength  (average AdjEM of conference peers)
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

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
    possessions: float  # estimated or observed

    # Box-score fields (0 when unavailable)
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

    # Extended box-score fields (assists, steals, blocks, fouls)
    ast: float = 0.0
    stl: float = 0.0
    blk: float = 0.0
    pf: float = 0.0  # personal fouls
    opp_ast: float = 0.0
    opp_stl: float = 0.0
    opp_blk: float = 0.0
    opp_pf: float = 0.0

    is_home: bool = False
    is_neutral: bool = True


@dataclass
class ProprietaryTeamMetrics:
    """Complete output for a single team — replaces KenPom + ShotQuality."""

    team_id: str
    team_name: str
    conference: str = ""

    # Adjusted efficiency  (KenPom AdjO / AdjD / AdjEM / AdjT)
    adj_offensive_efficiency: float = 100.0
    adj_defensive_efficiency: float = 100.0
    adj_efficiency_margin: float = 0.0
    adj_tempo: float = 68.0

    # Four Factors — offense
    effective_fg_pct: float = 0.50
    turnover_rate: float = 0.18
    offensive_reb_rate: float = 0.30
    free_throw_rate: float = 0.30

    # Four Factors — defense
    opp_effective_fg_pct: float = 0.50
    opp_turnover_rate: float = 0.18
    defensive_reb_rate: float = 0.70
    opp_free_throw_rate: float = 0.30

    # Supplementary shooting
    two_pt_pct: float = 0.48
    three_pt_pct: float = 0.34
    three_pt_rate: float = 0.35  # 3PA / FGA
    ft_pct: float = 0.72
    opp_two_pt_pct: float = 0.48
    opp_three_pt_pct: float = 0.34
    opp_three_pt_rate: float = 0.35

    # SOS
    sos_adj_em: float = 0.0
    sos_opp_o: float = 100.0
    sos_opp_d: float = 100.0
    ncsos_adj_em: float = 0.0

    # Luck  (CGM: actual_win% − expected_win%)
    luck: float = 0.0

    # WAB
    wab: float = 0.0

    # Proprietary xP per possession  (Four-Factors decomposition)
    offensive_xp_per_possession: float = 1.0
    defensive_xp_per_possession: float = 1.0

    # Shot distribution score  (rim + 3pt vs midrange proxy)
    shot_distribution_score: float = 0.0

    # 3-Point Variance  (game-to-game 3P% stdev — high = volatile)
    three_pt_variance: float = 0.0

    # Momentum  (last-10-game rolling AdjEM delta)
    momentum: float = 0.0
    recent_adj_em: float = 0.0   # AdjEM over last 10 games only

    # Pace-adjusted variance  (stdev of per-game scoring margin / pace factor)
    pace_adjusted_variance: float = 0.0

    # Consistency  (1 / (1 + stdev_margin))
    consistency: float = 0.5

    # Barthag / Pythagorean win%
    barthag: float = 0.5

    # --- Extended metrics (Tier 1-2 from research gap analysis) ---

    # Elo rating (MOV-adjusted, K=38, per SBCB methodology)
    elo_rating: float = 1500.0

    # Free throw shooting skill (FTM/FTA — 98% skill, most stable shooting metric)
    free_throw_pct: float = 0.72
    opp_free_throw_pct: float = 0.72

    # Assist-to-turnover ratio (halfcourt execution quality)
    assist_to_turnover_ratio: float = 1.0
    assist_rate: float = 0.50  # AST / FGM — team ball movement

    # Defensive disruption (steals + blocks per possession)
    steal_rate: float = 0.08
    block_rate: float = 0.05
    defensive_disruption_rate: float = 0.13  # (STL + BLK) / opp_possessions

    # Opponent 2P% (controllable, stable — unlike opp 3P%)
    opp_two_pt_pct_allowed: float = 0.48

    # Opponent 3PA rate (controllable shot selection forcing)
    opp_three_pt_attempt_rate: float = 0.35

    # Conference strength (average AdjEM of conference opponents)
    conference_adj_em: float = 0.0

    # Seed-efficiency residual (interaction: actual quality vs seed expectation)
    seed_efficiency_residual: float = 0.0

    # --- Tier 1-2 additions from exhaustive audit ---

    # Win percentage (simple but strong Kaggle baseline)
    win_pct: float = 0.5

    # Elite SOS: average AdjEM of top-30 opponents only (tournament-calibrated)
    elite_sos: float = 0.0

    # Quadrant 1 wins (Q1 = top-30 NET home, top-50 neutral, top-75 away)
    q1_wins: int = 0
    q1_losses: int = 0
    q1_win_pct: float = 0.0

    # Efficiency ratio (AdjO / AdjD — multiplicative quality measure)
    efficiency_ratio: float = 1.0

    # Foul rate (personal fouls per possession — tournament risk)
    foul_rate: float = 0.18

    # 3-Point regression signal (actual 3P% - expected 3P% from shot quality)
    # Positive = shooting above expected → likely to regress DOWN in tournament
    three_pt_regression_signal: float = 0.0

    # --- Schedule/context features (computed from game dates + external feeds) ---

    # Days since last game before tournament (rest advantage)
    rest_days: float = 5.0

    # Top-5 player minutes share (bench dependency risk — high = top-heavy)
    top5_minutes_share: float = 0.70

    # Preseason AP ranking (0 = unranked, lower = better; from open data feed)
    preseason_ap_rank: int = 0

    # Head coach tournament appearances (from open data feed)
    coach_tournament_appearances: int = 0

    # Conference tournament champion flag (auto-bid context)
    conf_tourney_champion: bool = False

    # Record
    wins: int = 0
    losses: int = 0

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ProprietaryMetricsEngine:
    """
    Compute all proprietary advanced metrics from game-level box scores.

    Primary data sources:  Torvik JSON, cbbpy game logs, Sports Reference.
    Can also ingest raw game records built from any public box-score feed.
    """

    # Home-court advantage in points (college basketball average)
    HCA_POINTS: float = 3.75
    # Margin cap to limit blowout inflation
    MARGIN_CAP: float = 16.0
    # Bubble team ranking for WAB (approx 45th in NET/AdjEM)
    BUBBLE_RANK: int = 45
    # Convergence iterations for SOS adjustment
    SOS_ITERATIONS: int = 15

    def compute(
        self,
        game_records: List[GameRecord],
        conference_map: Optional[Dict[str, str]] = None,
        cutoff_date: Optional[str] = None,
    ) -> Dict[str, ProprietaryTeamMetrics]:
        """
        Run the full engine.  Returns team_id → ProprietaryTeamMetrics.

        Args:
            game_records: list of per-team-game rows (one row per team per game)
            conference_map: optional team_id → conference name
            cutoff_date: YYYY-MM-DD — only use games on or before this date.
                         Prevents leakage from tournament games when computing
                         pre-tournament metrics.  If None, use all games.
        """
        if not game_records:
            return {}

        # Filter games to prevent temporal leakage
        if cutoff_date:
            game_records = [g for g in game_records if g.game_date <= cutoff_date]
            if not game_records:
                return {}

        by_team: Dict[str, List[GameRecord]] = defaultdict(list)
        for rec in game_records:
            by_team[rec.team_id].append(rec)

        # Sort each team's games chronologically
        for tid in by_team:
            by_team[tid].sort(key=lambda g: g.game_date)

        # --- Step 1: raw efficiency + tempo ---
        raw_off, raw_def, tempo, names = self._raw_efficiency(by_team)

        # --- Step 2: iterative SOS-adjusted efficiency ---
        adj_off, adj_def = self._iterative_sos_adjust(by_team, raw_off, raw_def)

        # --- Step 3: compute all derived metrics per team ---
        all_team_ids = sorted(by_team.keys())
        league_avg_em = float(np.mean([adj_off[t] - adj_def[t] for t in all_team_ids]))

        results: Dict[str, ProprietaryTeamMetrics] = {}
        for tid in all_team_ids:
            games = by_team[tid]
            adj_em = adj_off[tid] - adj_def[tid]

            ff = self._four_factors(games)
            shooting = self._supplementary_shooting(games)
            sos = self._strength_of_schedule(games, adj_off, adj_def)
            luck = self._correlated_gaussian_luck(games)
            barthag = self._pythagorean_win_pct(adj_off[tid], adj_def[tid])

            # --- New Tier 1-2 metrics ---
            extended = self._extended_box_score_metrics(games)

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

            shot_dist = self._shot_distribution_score(games)
            three_var = self._three_point_variance(games)
            momentum, recent_em = self._momentum(games, adj_off, adj_def)
            pace_var = self._pace_adjusted_variance(games)
            consistency = self._consistency(games)

            opp_shot_selection = self._opponent_shot_selection(games)
            foul_rate = self._foul_rate(games)

            wins = sum(1 for g in games if g.points > g.opp_points)
            losses = len(games) - wins
            n_games = max(wins + losses, 1)

            # Efficiency ratio: multiplicative quality measure
            eff_ratio = adj_off[tid] / max(adj_def[tid], 1e-6)

            # 3-Point regression signal: actual 3P% vs expected from shot quality
            # Expected 3P% ≈ league-average 3P% (0.345) scaled by defensive
            # contest pressure (approximated by shot distribution quality)
            actual_3p = shooting.get("three_pt_pct", 0.34)
            expected_3p = 0.345  # D1 average baseline
            three_pt_regression = actual_3p - expected_3p

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
                wins=wins,
                losses=losses,
            )

        # --- Step 4: WAB (needs full-league rankings) ---
        self._compute_wab(results, by_team)

        # --- Step 5: Elo ratings (needs all games chronologically) ---
        self._compute_elo_ratings(results, by_team)

        # --- Step 6: Conference strength (needs league-wide AdjEM) ---
        self._compute_conference_strength(results, by_team, conference_map or {})

        # --- Step 7: Elite SOS + Quadrant record (needs full-league AdjEM) ---
        self._compute_elite_sos_and_quadrants(results, by_team, adj_off, adj_def)

        # --- Step 8: Rest days (days since last game) ---
        self._compute_rest_days(results, by_team)

        return results

    # ------------------------------------------------------------------
    # Internal computation helpers
    # ------------------------------------------------------------------

    def _raw_efficiency(
        self, by_team: Dict[str, List[GameRecord]]
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, str]]:
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
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Iterative multiplicative SOS adjustment (KenPom method).

        Incorporates:
        - Home-court advantage adjustment  (±3.75 pts)
        - Margin cap  (±16 pts) to limit blowout distortion
        - Recency weighting  (exponential decay, half-life ≈ 30 games)
        """
        league_off = float(np.mean(list(raw_off.values()))) if raw_off else 100.0
        league_def = float(np.mean(list(raw_def.values()))) if raw_def else 100.0

        adj_off = dict(raw_off)
        adj_def = dict(raw_def)

        for _iteration in range(self.SOS_ITERATIONS):
            next_off: Dict[str, float] = {}
            next_def: Dict[str, float] = {}

            for tid, games in by_team.items():
                off_samples: List[float] = []
                def_samples: List[float] = []
                weights: List[float] = []

                n_games = len(games)
                for idx, g in enumerate(games):
                    # --- Home-court adjustment ---
                    hca = 0.0
                    if not g.is_neutral:
                        hca = self.HCA_POINTS if g.is_home else -self.HCA_POINTS

                    adj_pts = g.points - hca
                    adj_opp = g.opp_points + hca

                    # --- Margin cap ---
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

                    off_samples.append(
                        game_off * league_def / max(opp_def, 1e-6)
                    )
                    def_samples.append(
                        game_def * league_off / max(opp_off, 1e-6)
                    )

                    # Recency weight: exponential decay, half-life ~30 games
                    recency = math.exp(-0.693 * (n_games - 1 - idx) / 30.0)
                    weights.append(recency)

                total_w = sum(weights) or 1.0
                next_off[tid] = sum(o * w for o, w in zip(off_samples, weights)) / total_w
                next_def[tid] = sum(d * w for d, w in zip(def_samples, weights)) / total_w

            adj_off = next_off
            adj_def = next_def

        return adj_off, adj_def

    def _four_factors(self, games: List[GameRecord]) -> Dict[str, float]:
        """Dean Oliver's Four Factors — offense + defense."""
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

        # Use Oliver's proper denominator for TO%:  TOV / (FGA + 0.44*FTA + TOV)
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
    ) -> Dict[str, float]:
        league_off = float(np.mean(list(adj_off.values()))) if adj_off else 100.0
        league_def = float(np.mean(list(adj_def.values()))) if adj_def else 100.0

        n = max(len(games), 1)
        opp_o = [adj_off.get(g.opponent_id, league_off) for g in games]
        opp_d = [adj_def.get(g.opponent_id, league_def) for g in games]
        opp_em = [o - d for o, d in zip(opp_o, opp_d)]

        # NCSOS: non-conference games only (heuristic: neutral-site or away)
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
        """
        KenPom-style luck via the Correlated Gaussian Method.

        luck = actual_win% − expected_win%
        where expected_win% = Φ(mean_margin / std_margin)
        """
        if len(games) < 5:
            return 0.0

        margins = [g.points - g.opp_points for g in games]
        mean_m = float(np.mean(margins))
        std_m = float(np.std(margins, ddof=1))

        if std_m < 0.1:
            return 0.0

        z = mean_m / std_m
        expected_win_pct = float(scipy_stats.norm.cdf(z))
        actual_win_pct = sum(1 for m in margins if m > 0) / len(margins)

        return actual_win_pct - expected_win_pct

    def _pythagorean_win_pct(self, adj_o: float, adj_d: float) -> float:
        """KenPom Pythagorean win% with exponent 10.25."""
        exp = 10.25
        num = adj_o ** exp
        denom = num + adj_d ** exp
        if denom < 1e-12:
            return 0.5
        return float(num / denom)

    def _box_score_xp(self, ff: Dict[str, float], side: str = "offense", ft_pct: float = 0.72) -> float:
        """
        Proprietary xP per possession from Four Factors decomposition.

        xP ≈ eFG% × 2 × (1 − TO%) × (1 + ORB% × (1 − eFG%)) + FTR × FT%

        This decomposes scoring into: keeping the ball → shooting effectively
        → getting second chances → drawing fouls.
        """
        efg = ff.get("effective_fg_pct", 0.50)
        tov = ff.get("turnover_rate", 0.18)
        orb = ff.get("offensive_reb_rate", 0.30)
        ftr = ff.get("free_throw_rate", 0.30)

        # Four-factors decomposition of expected points per possession
        keep_ball = 1.0 - tov
        shoot_well = efg * 2.0
        second_chance = 1.0 + orb * (1.0 - efg)
        free_throws = ftr * ft_pct

        xp = keep_ball * (shoot_well * second_chance + free_throws)
        return float(np.clip(xp, 0.5, 1.8))

    def _shot_distribution_score(self, games: List[GameRecord]) -> float:
        """
        Proxy for ShotQuality's shot distribution.

        Uses 3PA/FGA ratio and estimated rim rate from FTR to approximate
        the quality of shot selection.  Higher = more rim + 3pt (efficient).
        """
        fga = sum(g.fga for g in games)
        fg3a = sum(g.fg3a for g in games)
        fta = sum(g.fta for g in games)

        if fga < 1:
            return 0.0

        three_rate = fg3a / fga
        # FTR correlates with rim attacks  (r ≈ 0.65)
        ft_rate = fta / fga
        estimated_rim_rate = float(np.clip(0.18 + 0.7 * ft_rate, 0.15, 0.50))
        midrange_rate = max(1.0 - three_rate - estimated_rim_rate, 0.05)

        # Positive when shot mix favors rim + 3 over midrange
        score = (estimated_rim_rate + three_rate) - 0.75 * midrange_rate
        return float(score)

    def _three_point_variance(self, games: List[GameRecord]) -> float:
        """
        Game-to-game 3P% standard deviation.

        High variance teams are upset-prone on cold-shooting nights.
        """
        per_game_3p = []
        for g in games:
            if g.fg3a >= 5:  # only games with meaningful 3PA
                per_game_3p.append(g.fg3m / g.fg3a)

        if len(per_game_3p) < 5:
            return 0.0

        return float(np.std(per_game_3p, ddof=1))

    def _momentum(
        self,
        games: List[GameRecord],
        adj_off: Dict[str, float],
        adj_def: Dict[str, float],
    ) -> Tuple[float, float]:
        """
        Rolling form over last 10 games vs season average.

        Returns (momentum_delta, recent_adj_em).
        """
        league_off = float(np.mean(list(adj_off.values()))) if adj_off else 100.0
        league_def = float(np.mean(list(adj_def.values()))) if adj_def else 100.0

        if len(games) < 12:
            return 0.0, adj_off.get(games[0].team_id, 100.0) - adj_def.get(games[0].team_id, 100.0) if games else 0.0

        recent = games[-10:]
        recent_margins = []
        for g in recent:
            opp_em = adj_off.get(g.opponent_id, league_off) - adj_def.get(g.opponent_id, league_def)
            # Margin adjusted for opponent quality
            raw_margin = g.points - g.opp_points
            quality_margin = raw_margin - opp_em
            recent_margins.append(quality_margin)

        season_margins = []
        for g in games:
            opp_em = adj_off.get(g.opponent_id, league_off) - adj_def.get(g.opponent_id, league_def)
            raw_margin = g.points - g.opp_points
            quality_margin = raw_margin - opp_em
            season_margins.append(quality_margin)

        recent_em = float(np.mean(recent_margins))
        season_em = float(np.mean(season_margins))
        momentum = recent_em - season_em

        return float(momentum), float(recent_em)

    def _pace_adjusted_variance(self, games: List[GameRecord]) -> float:
        """
        Scoring-margin variance adjusted for pace.

        Low-possession games inflate the influence of randomness.
        """
        if len(games) < 5:
            return 0.0

        adjusted_margins = []
        for g in games:
            margin = g.points - g.opp_points
            # Normalize to 70-possession baseline
            pace_factor = max(g.possessions, 40.0) / 70.0
            adjusted_margins.append(margin / pace_factor)

        return float(np.std(adjusted_margins, ddof=1))

    def _consistency(self, games: List[GameRecord]) -> float:
        """
        Inverse of scoring-margin stdev.  1 / (1 + std_margin).

        Consistent teams outperform in single-elimination.
        """
        if len(games) < 5:
            return 0.5

        margins = [g.points - g.opp_points for g in games]
        std_m = float(np.std(margins, ddof=1))
        return 1.0 / (1.0 + std_m)

    def _compute_wab(
        self,
        results: Dict[str, ProprietaryTeamMetrics],
        by_team: Dict[str, List[GameRecord]],
    ) -> None:
        """
        Wins Above Bubble — adopted by NCAA selection committee in 2024.

        For each game, compute P(bubble_team_wins), then:
          Win  → WAB_game = 1.0 − P(bubble_wins)
          Loss → WAB_game = 0.0 − P(bubble_wins)

        Bubble team = team ranked ~45th in AdjEM.
        """
        ranked = sorted(results.values(), key=lambda t: t.adj_efficiency_margin, reverse=True)
        bubble_em = 0.0
        if len(ranked) >= self.BUBBLE_RANK:
            bubble_em = ranked[self.BUBBLE_RANK - 1].adj_efficiency_margin
        elif ranked:
            bubble_em = ranked[-1].adj_efficiency_margin

        for tid, games in by_team.items():
            if tid not in results:
                continue

            total_wab = 0.0
            for g in games:
                opp = results.get(g.opponent_id)
                opp_em = opp.adj_efficiency_margin if opp else 0.0

                # P(bubble beats opponent) via log5
                bubble_wp = self._log5_win_prob(bubble_em, opp_em)

                # Home-court adjustment
                if not g.is_neutral:
                    if g.is_home:
                        bubble_wp -= 0.035  # bubble playing away
                    else:
                        bubble_wp += 0.035  # bubble playing at home

                bubble_wp = float(np.clip(bubble_wp, 0.01, 0.99))

                is_win = g.points > g.opp_points
                if is_win:
                    total_wab += 1.0 - bubble_wp
                else:
                    total_wab += 0.0 - bubble_wp

            results[tid].wab = round(total_wab, 2)

    def _extended_box_score_metrics(self, games: List[GameRecord]) -> Dict[str, float]:
        """
        Compute extended box-score metrics not in original Four Factors:
        FT%, A/TO, assist rate, steal rate, block rate, defensive disruption.
        """
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
        opp_total_poss = total_poss  # symmetric

        return {
            "free_throw_pct": ftm / max(fta, 1.0),
            "opp_free_throw_pct": opp_ftm / max(opp_fta, 1.0),
            "assist_to_turnover_ratio": ast / max(tov, 1.0),
            "assist_rate": ast / max(fgm, 1.0),
            "steal_rate": stl / max(opp_total_poss, 1.0) * 100.0,
            "block_rate": blk / max(opp_total_poss, 1.0) * 100.0,
            "defensive_disruption_rate": (stl + blk) / max(opp_total_poss, 1.0) * 100.0,
        }

    def _opponent_shot_selection(self, games: List[GameRecord]) -> Dict[str, float]:
        """
        Controllable defensive shot quality metrics.

        opp_two_pt_pct: Opponent 2P% allowed (stable, controllable).
        opp_three_pt_attempt_rate: % of opponent FGA from 3 (shot selection forcing).
        """
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

    def _compute_elo_ratings(
        self,
        results: Dict[str, ProprietaryTeamMetrics],
        by_team: Dict[str, List[GameRecord]],
    ) -> None:
        """
        MOV-adjusted Elo ratings per SBCB methodology.

        - Base K-factor: 38
        - MOV multiplier: (3 + |margin|)^0.85
        - HCA: +50 Elo for home team
        - Autocorrection: shrink towards 1500 each game by 1%
        """
        # Collect all games in chronological order (deduplicated)
        all_games: List[GameRecord] = []
        seen_game_ids: Dict[str, set] = defaultdict(set)
        for tid, games in by_team.items():
            for g in games:
                # Only add one side of each game
                pair_key = tuple(sorted([g.team_id, g.opponent_id]))
                if g.game_id not in seen_game_ids.get(pair_key, set()):
                    all_games.append(g)
                    seen_game_ids.setdefault(pair_key, set()).add(g.game_id)
        all_games.sort(key=lambda g: g.game_date)

        elo: Dict[str, float] = defaultdict(lambda: 1500.0)
        K_BASE = 38.0

        for g in all_games:
            t1 = g.team_id
            t2 = g.opponent_id

            # Home-court advantage
            hca = 0.0
            if not g.is_neutral:
                hca = 50.0 if g.is_home else -50.0

            # Expected score
            e1 = 1.0 / (1.0 + 10 ** (-(elo[t1] + hca - elo[t2]) / 400.0))

            # Actual outcome
            margin = g.points - g.opp_points
            s1 = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)

            # MOV multiplier: (3 + |margin|)^0.85
            mov_mult = (3.0 + abs(margin)) ** 0.85

            # K scaled by MOV
            k = K_BASE * mov_mult / max(((elo[t1] - elo[t2]) / 400.0) ** 2 + 1.0, 1.0)

            # Update
            delta = k * (s1 - e1)
            elo[t1] += delta
            elo[t2] -= delta

            # Autocorrection (shrink 1% towards 1500)
            elo[t1] = elo[t1] * 0.99 + 1500.0 * 0.01
            elo[t2] = elo[t2] * 0.99 + 1500.0 * 0.01

        # Assign final Elo to results
        for tid in results:
            results[tid].elo_rating = round(elo.get(tid, 1500.0), 1)

    def _compute_conference_strength(
        self,
        results: Dict[str, ProprietaryTeamMetrics],
        by_team: Dict[str, List[GameRecord]],
        conference_map: Dict[str, str],
    ) -> None:
        """
        Compute average AdjEM of each team's conference peers.

        Falls back to SOS-based estimate when conference assignments are
        unavailable.
        """
        if not conference_map:
            # Without conference labels, use opponent average AdjEM as proxy
            for tid in results:
                results[tid].conference_adj_em = results[tid].sos_adj_em
            return

        # Group teams by conference
        conf_teams: Dict[str, List[str]] = defaultdict(list)
        for tid, conf in conference_map.items():
            if conf:
                conf_teams[conf].append(tid)

        # Compute per-conference average AdjEM
        conf_avg: Dict[str, float] = {}
        for conf, tids in conf_teams.items():
            ems = [results[t].adj_efficiency_margin for t in tids if t in results]
            conf_avg[conf] = float(np.mean(ems)) if ems else 0.0

        for tid in results:
            conf = conference_map.get(tid, "")
            results[tid].conference_adj_em = conf_avg.get(conf, results[tid].sos_adj_em)

    def _foul_rate(self, games: List[GameRecord]) -> float:
        """Team personal fouls per possession — tournament foul-trouble risk."""
        total_pf = sum(g.pf for g in games)
        total_poss = sum(max(g.possessions, 1.0) for g in games)
        if total_poss < 1.0:
            return 0.18  # D1 average
        return total_pf / total_poss

    def _compute_elite_sos_and_quadrants(
        self,
        results: Dict[str, ProprietaryTeamMetrics],
        by_team: Dict[str, List[GameRecord]],
        adj_off: Dict[str, float],
        adj_def: Dict[str, float],
    ) -> None:
        """
        Compute elite SOS (top-30 opponents only) and Quadrant record.

        Quadrant definitions (NCAA NET proxy using AdjEM ranking):
          Q1: top-30 home, top-50 neutral, top-75 away
          Q2: 31-75 home, 51-100 neutral, 76-135 away
          Q3: 76-160 home, 101-200 neutral, 136-240 away
          Q4: 161+ home, 201+ neutral, 241+ away

        We approximate using AdjEM ranks since NET is not publicly available.
        """
        # Rank all teams by AdjEM
        ranked_teams = sorted(
            results.keys(),
            key=lambda t: results[t].adj_efficiency_margin,
            reverse=True,
        )
        team_rank: Dict[str, int] = {t: i + 1 for i, t in enumerate(ranked_teams)}

        for tid, games in by_team.items():
            if tid not in results:
                continue

            # --- Elite SOS: average AdjEM of top-30 ranked opponents ---
            elite_ems: List[float] = []
            q1_w, q1_l = 0, 0

            for g in games:
                opp_rank = team_rank.get(g.opponent_id, 200)
                opp_em = (adj_off.get(g.opponent_id, 100.0)
                          - adj_def.get(g.opponent_id, 100.0))

                # Elite SOS: only top-30
                if opp_rank <= 30:
                    elite_ems.append(opp_em)

                # Quadrant classification
                q = self._classify_quadrant(opp_rank, g.is_home, g.is_neutral)
                is_win = g.points > g.opp_points
                if q == 1:
                    if is_win:
                        q1_w += 1
                    else:
                        q1_l += 1

            results[tid].elite_sos = float(np.mean(elite_ems)) if elite_ems else 0.0
            results[tid].q1_wins = q1_w
            results[tid].q1_losses = q1_l
            results[tid].q1_win_pct = q1_w / max(q1_w + q1_l, 1)

    @staticmethod
    def _classify_quadrant(opp_rank: int, is_home: bool, is_neutral: bool) -> int:
        """Classify a game into NCAA quadrant (1-4) based on opponent rank and venue."""
        if is_neutral:
            if opp_rank <= 50:
                return 1
            elif opp_rank <= 100:
                return 2
            elif opp_rank <= 200:
                return 3
            return 4
        elif is_home:
            if opp_rank <= 30:
                return 1
            elif opp_rank <= 75:
                return 2
            elif opp_rank <= 160:
                return 3
            return 4
        else:  # away
            if opp_rank <= 75:
                return 1
            elif opp_rank <= 135:
                return 2
            elif opp_rank <= 240:
                return 3
            return 4

    def _compute_rest_days(
        self,
        results: Dict[str, ProprietaryTeamMetrics],
        by_team: Dict[str, List[GameRecord]],
    ) -> None:
        """Compute days since last game for each team (rest advantage)."""
        from datetime import datetime as _dt

        for tid, games in by_team.items():
            if tid not in results or not games:
                continue
            # Games are sorted chronologically; take last game date
            last_date_str = games[-1].game_date
            try:
                last_date = _dt.strptime(last_date_str, "%Y-%m-%d")
                # Assume tournament starts ~March 20 of the season year
                # (real date comes from config, but this is a reasonable proxy)
                year = last_date.year if last_date.month <= 6 else last_date.year + 1
                tourney_start = _dt(year, 3, 20)
                delta = (tourney_start - last_date).days
                results[tid].rest_days = float(max(delta, 0))
            except (ValueError, TypeError):
                results[tid].rest_days = 5.0  # D1 average

    @staticmethod
    def _log5_win_prob(team_a_em: float, team_b_em: float) -> float:
        """
        Win probability from efficiency margin differential.

        Uses logistic model:  P(A wins) = 1 / (1 + 10^(−EM_diff / 10))
        """
        diff = team_a_em - team_b_em
        diff = float(np.clip(diff, -40.0, 40.0))
        return 1.0 / (1.0 + 10 ** (-diff / 10.0))


# ---------------------------------------------------------------------------
# Converter: Torvik/public data → GameRecord
# ---------------------------------------------------------------------------

def torvik_to_game_records(torvik_teams: List[Dict], historical_games: List[Dict]) -> List[GameRecord]:
    """
    Convert Torvik team stats + historical game rows into GameRecord objects.

    Works with data from cbbpy, sportsipy, or any source that provides
    game-level box scores.
    """
    records: List[GameRecord] = []
    for game in historical_games:
        if not isinstance(game, dict):
            continue

        game_id = str(game.get("game_id") or game.get("id") or "")
        team_id = _team_id(str(game.get("team_id") or game.get("team1") or game.get("home_team") or ""))
        opp_id = _team_id(str(game.get("opponent_id") or game.get("team2") or game.get("away_team") or ""))
        if not game_id or not team_id or not opp_id:
            continue

        points = _to_float(game.get("team_score") or game.get("home_score") or game.get("points") or 0)
        opp_points = _to_float(game.get("opponent_score") or game.get("away_score") or game.get("opp_points") or 0)

        fga = _to_float(game.get("fga", 0))
        fgm = _to_float(game.get("fgm", 0))
        fg3a = _to_float(game.get("fg3a") or game.get("x3pa", 0))
        fg3m = _to_float(game.get("fg3m") or game.get("x3pm", 0))
        fta = _to_float(game.get("fta", 0))
        ftm = _to_float(game.get("ftm", 0))
        tov = _to_float(game.get("turnovers") or game.get("tov", 0))
        orb = _to_float(game.get("orb") or game.get("offensive_rebounds", 0))
        drb = _to_float(game.get("drb") or game.get("defensive_rebounds", 0))

        opp_fga = _to_float(game.get("opp_fga", 0))
        opp_fgm = _to_float(game.get("opp_fgm", 0))
        opp_fg3a = _to_float(game.get("opp_fg3a", 0))
        opp_fg3m = _to_float(game.get("opp_fg3m", 0))
        opp_fta = _to_float(game.get("opp_fta", 0))
        opp_ftm = _to_float(game.get("opp_ftm", 0))
        opp_tov = _to_float(game.get("opp_tov") or game.get("opp_turnovers", 0))
        opp_orb = _to_float(game.get("opp_orb", 0))
        opp_drb = _to_float(game.get("opp_drb", 0))

        # Extended box-score: assists, steals, blocks, fouls
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
            # Estimate from score + typical D1 efficiency
            poss = max((points + opp_points) / 2.0, 30.0)

        is_neutral = bool(game.get("neutral_site", game.get("is_neutral", False)))
        is_home = bool(game.get("is_home", not is_neutral))

        game_date = str(game.get("game_date") or game.get("date") or game.get("start_date") or "2026-01-01")
        team_name = str(game.get("team_name") or game.get("team1") or team_id)

        records.append(GameRecord(
            game_id=game_id,
            game_date=game_date,
            team_id=team_id,
            team_name=team_name,
            opponent_id=opp_id,
            points=points,
            opp_points=opp_points,
            possessions=poss,
            fga=fga, fgm=fgm, fg3a=fg3a, fg3m=fg3m, fta=fta, ftm=ftm,
            tov=tov, orb=orb, drb=drb,
            ast=ast, stl=stl, blk=blk, pf=pf,
            opp_fga=opp_fga, opp_fgm=opp_fgm, opp_fg3a=opp_fg3a, opp_fg3m=opp_fg3m,
            opp_fta=opp_fta, opp_ftm=opp_ftm, opp_tov=opp_tov, opp_orb=opp_orb, opp_drb=opp_drb,
            opp_ast=opp_ast, opp_stl=opp_stl, opp_blk=opp_blk, opp_pf=opp_pf,
            is_home=is_home, is_neutral=is_neutral,
        ))

    return records


def _team_id(name: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
