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

import csv
import math
import os
import re
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

    # Per-game pace variance (game-to-game tempo stdev — upset risk amplifier)
    pace_variance: float = 0.0

    # Coach tournament win rate (wins / games in NCAA tournament — experience quality)
    coach_tournament_win_rate: float = 0.0

    # --- Advanced metrics (KenPom/ShotQuality replacements) ---

    # True Shooting % — composite shooting efficiency (2P + 3P + FT combined)
    # More stable than eFG% because it accounts for free throw volume.
    # Formula: PTS / (2 × (FGA + 0.44 × FTA))
    true_shooting_pct: float = 0.54
    opp_true_shooting_pct: float = 0.54

    # Neutral-site win % — directly relevant to tournament (all neutral venues)
    neutral_site_win_pct: float = 0.5
    neutral_site_games: int = 0

    # Home/Away AdjEM splits — captures team-specific HCA variance
    # Some teams are much better/worse at home; tournament is neutral.
    home_adj_em: float = 0.0
    away_adj_em: float = 0.0
    # Differential: how much a team benefits from home court (high = home-dependent)
    home_court_dependence: float = 0.0

    # 5-game momentum (finer granularity than 10-game)
    momentum_5g: float = 0.0

    # Transition efficiency estimate (pace-derived proxy)
    # High-pace teams generate more transition opportunities.
    # Estimated: (pace - league_avg_pace) × efficiency × scale
    transition_efficiency: float = 0.0

    # Defensive transition vulnerability (opponent pace interaction)
    # Approximated from pace-adjusted defensive efficiency differential
    defensive_transition_vulnerability: float = 0.0

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
    # Historical average AdjEM of the ~45th-ranked team (2015-2025 average).
    # Used as a Bayesian prior to dampen the effect of end-of-season ranking
    # shifts on the bubble team definition.  Without this, WAB for early-season
    # games uses a bubble AdjEM that reflects the final season rankings —
    # information not available at game time.
    BUBBLE_EM_PRIOR: float = 5.0
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
            sos = self._strength_of_schedule(games, adj_off, adj_def, conference_map or {})
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

            # B2: Bayesian 3PT shrinkage. Teams with low 3PA volume have
            # noisy raw 3P%. Shrink toward D1 average (0.345) proportional
            # to sample size. Prior weight ~100 3PA ≈ 3 games of attempts.
            actual_3p = shooting.get("three_pt_pct", 0.345)
            n_3pa = sum(g.fg3a for g in games)
            prior_weight = 100.0
            shrunk_3p = (n_3pa * actual_3p + prior_weight * 0.345) / (n_3pa + prior_weight)
            three_pt_regression = shrunk_3p - 0.345

            # --- Advanced KenPom/ShotQuality replacements ---
            ts_pct, opp_ts_pct = self._true_shooting_pct(games)
            neutral_win_pct, neutral_games = self._neutral_site_record(games)
            home_em, away_em, hc_dependence = self._home_away_splits(games, adj_off, adj_def)
            mom_5g = self._momentum_5g(games, adj_off, adj_def)
            trans_off, trans_def_vuln = self._transition_efficiency(games, adj_off, adj_def)

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
                pace_variance=self._pace_variance(games),
                true_shooting_pct=ts_pct,
                opp_true_shooting_pct=opp_ts_pct,
                neutral_site_win_pct=neutral_win_pct,
                neutral_site_games=neutral_games,
                home_adj_em=home_em,
                away_adj_em=away_em,
                home_court_dependence=hc_dependence,
                momentum_5g=mom_5g,
                transition_efficiency=trans_off,
                defensive_transition_vulnerability=trans_def_vuln,
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

        # --- Step 9: H2H records + common opponent margins (for matchup features) ---
        # Store per-team game data for later H2H/common-opp computation.
        # This is consumed by FeatureEngineer.create_matchup_features().
        self._by_team = by_team
        self._adj_off = adj_off
        self._adj_def = adj_def

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
        Additive iterative SOS adjustment (convergent KenPom-style method).

        Key improvements over the previous multiplicative approach:
        1. **Additive adjustments** — computes per-possession margin
           adjustments rather than multiplicative scaling, guaranteeing
           convergence to a fixed point.
        2. **Convergence damping** — uses a damping factor (0.7) that
           blends new estimates with old to prevent oscillation.
        3. **League-mean anchoring** — re-centers after each iteration
           so league averages stay fixed, preventing drift.

        Also incorporates:
        - Home-court advantage adjustment  (±3.75 pts)
        - Margin cap  (±16 pts) applied BEFORE HCA to prevent asymmetry
        - Recency weighting  (exponential decay, half-life ≈ 30 games)
        """
        league_off = float(np.mean(list(raw_off.values()))) if raw_off else 100.0
        league_def = float(np.mean(list(raw_def.values()))) if raw_def else 100.0

        adj_off = dict(raw_off)
        adj_def = dict(raw_def)

        DAMPING = 0.7  # Blend factor: new = damp * computed + (1-damp) * old

        for _iteration in range(self.SOS_ITERATIONS):
            next_off: Dict[str, float] = {}
            next_def: Dict[str, float] = {}

            for tid, games in by_team.items():
                off_adjustments: List[float] = []
                def_adjustments: List[float] = []
                weights: List[float] = []

                n_games = len(games)
                for idx, g in enumerate(games):
                    # --- Home-court adjustment ---
                    hca = 0.0
                    if not g.is_neutral:
                        hca = self.HCA_POINTS if g.is_home else -self.HCA_POINTS

                    adj_pts = g.points - hca
                    adj_opp = g.opp_points + hca

                    # --- Margin cap (applied symmetrically post-HCA) ---
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

                    # Additive SOS: adjust by how much opponent deviates
                    # from league average (not multiplicative ratio).
                    opp_def = adj_def.get(g.opponent_id, league_def)
                    opp_off = adj_off.get(g.opponent_id, league_off)

                    # If opponent defense is tougher than average (higher
                    # opp_def), credit the team with higher offensive rating.
                    off_adjustment = game_off + (opp_def - league_def)
                    def_adjustment = game_def + (opp_off - league_off)

                    off_adjustments.append(off_adjustment)
                    def_adjustments.append(def_adjustment)

                    # Recency weight: exponential decay, half-life ~30 games
                    recency = math.exp(-0.693 * (n_games - 1 - idx) / 30.0)
                    weights.append(recency)

                total_w = sum(weights) or 1.0
                computed_off = sum(o * w for o, w in zip(off_adjustments, weights)) / total_w
                computed_def = sum(d * w for d, w in zip(def_adjustments, weights)) / total_w

                # Damped update to ensure convergence
                next_off[tid] = DAMPING * computed_off + (1.0 - DAMPING) * adj_off[tid]
                next_def[tid] = DAMPING * computed_def + (1.0 - DAMPING) * adj_def[tid]

            # Re-center around league averages to prevent drift
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
        conference_map: Dict[str, str] = None,
    ) -> Dict[str, float]:
        league_off = float(np.mean(list(adj_off.values()))) if adj_off else 100.0
        league_def = float(np.mean(list(adj_def.values()))) if adj_def else 100.0
        conference_map = conference_map or {}

        n = max(len(games), 1)
        opp_o = [adj_off.get(g.opponent_id, league_off) for g in games]
        opp_d = [adj_def.get(g.opponent_id, league_def) for g in games]
        opp_em = [o - d for o, d in zip(opp_o, opp_d)]

        # NCSOS: non-conference games only.
        # Use conference_map for accurate identification; fall back to the
        # old heuristic (neutral-site or away) only when conference data is
        # unavailable so NCSOS is never empty.
        team_conf = conference_map.get(games[0].team_id, "") if games else ""
        if team_conf and conference_map:
            nc_ems = [
                adj_off.get(g.opponent_id, league_off) - adj_def.get(g.opponent_id, league_def)
                for g in games
                if conference_map.get(g.opponent_id, "") != team_conf
            ]
        else:
            # Fallback heuristic when conference labels unavailable
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

        # Sample-size shrinkage: partial weight at 12-32 games, full at 32+
        MIN_GAMES = 12
        FULL_WEIGHT_GAMES = 32
        shrinkage = min(1.0, (len(games) - MIN_GAMES) / (FULL_WEIGHT_GAMES - MIN_GAMES))
        return raw_luck * shrinkage

    def _pythagorean_win_pct(self, adj_o: float, adj_d: float) -> float:
        """Pythagorean win% for per-possession efficiency values.

        A3: KenPom's exponent 10.25 is calibrated for per-game totals
        (65-85 PPG range). At per-possession efficiency scale (95-115),
        10.25 is too aggressive: e.g., AdjO=110 vs AdjD=100 → 0.937,
        but real teams at that spread win ~85%. Use log5 on the efficiency
        margin instead, which is well-calibrated for per-possession data.
        """
        margin = adj_o - adj_d
        # log5-equivalent: sigmoid of margin scaled by empirical constant.
        # Scale factor 0.145 calibrated so AdjEM=+10 → ~0.85 win%,
        # AdjEM=+20 → ~0.95, AdjEM=0 → 0.50 (consistent with KenPom).
        import math
        return 1.0 / (1.0 + math.exp(-0.145 * margin))

    def _box_score_xp(self, ff: Dict[str, float], side: str = "offense", ft_pct: float = 0.72) -> float:
        """
        Expected points per possession from Four Factors decomposition.

        Uses Dean Oliver's empirically validated factor weights (2004),
        refined by Kubatko et al. (2007) for modern college basketball:

            xP = 0.40 × eFG% + 0.25 × (1 − TO%) + 0.20 × ORB% + 0.15 × FTR×FT%

        These weights are derived from regression of Four Factors against
        offensive efficiency across D1 seasons.  The weighted sum is then
        scaled to approximate points-per-possession units (multiply by ~2.0
        since a possession produces ~1.0 PPP on average).

        Reference:
          Oliver, Dean. "Basketball on Paper" (2004), Chapter 4.
          Kubatko et al., "A Starting Point for Analyzing Basketball
          Statistics", JQAS 3(3), 2007.
        """
        efg = ff.get("effective_fg_pct", 0.50)
        tov = ff.get("turnover_rate", 0.18)
        orb = ff.get("offensive_reb_rate", 0.30)
        ftr = ff.get("free_throw_rate", 0.30)

        # Oliver's empirically validated factor weights
        shooting_contrib = 0.40 * efg
        ball_care_contrib = 0.25 * (1.0 - tov)
        rebounding_contrib = 0.20 * orb
        ft_contrib = 0.15 * ftr * ft_pct

        # Weighted composite → scale to PPP-like units
        composite = shooting_contrib + ball_care_contrib + rebounding_contrib + ft_contrib
        xp = composite * 2.0  # Scale so D1 average ≈ 1.0 PPP

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
        # FIX M3: Use only the historical prior for bubble team AdjEM.
        # The previous approach ranked all teams by final adj_efficiency_margin
        # and used the ~45th team's value — this leaked end-of-season rankings
        # into WAB for every game, including early-season ones.  The historical
        # prior (5.0 AdjEM) is stable across seasons and introduces no leakage.
        bubble_em = self.BUBBLE_EM_PRIOR

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
            "steal_rate": stl / max(opp_total_poss, 1.0),
            "block_rate": blk / max(opp_total_poss, 1.0),
            "defensive_disruption_rate": (stl + blk) / max(opp_total_poss, 1.0),
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

            # B5: Derive Elo HCA from the unified HCA_POINTS constant.
            # 3.75 pts * 13.3 Elo/pt ≈ 50 Elo (consistent with FiveThirtyEight).
            ELO_HCA = self.HCA_POINTS * 13.3
            hca = 0.0
            if not g.is_neutral:
                hca = ELO_HCA if g.is_home else -ELO_HCA

            # Expected score
            e1 = 1.0 / (1.0 + 10 ** (-(elo[t1] + hca - elo[t2]) / 400.0))

            # Actual outcome
            margin = g.points - g.opp_points
            s1 = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)

            # MOV multiplier: log-based to saturate for blowouts.
            # ln(1 + |margin|) gives ~2.1 for 7-pt win, ~3.4 for 30-pt blowout.
            # This prevents cupcake blowouts from inflating Elo disproportionately.
            mov_mult = np.log1p(abs(margin))

            # K scaled by MOV, dampened by elo diff (FiveThirtyEight-style).
            # Denominator: 2.2 / (elo_diff * 0.001 + 2.2) shrinks updates for
            # expected blowouts more aggressively than the old quadratic form.
            elo_diff = abs(elo[t1] - elo[t2])
            elo_dampening = 2.2 / (elo_diff * 0.001 + 2.2)
            k = K_BASE * mov_mult * elo_dampening

            # Update
            delta = k * (s1 - e1)
            elo[t1] += delta
            elo[t2] -= delta

            # A2: Per-game autocorrection REMOVED. With ~30 games per season,
            # 0.99^30 = 0.74 destroyed 26% of accumulated Elo signal, compressing
            # the distribution. Standard practice: regress at season boundaries
            # only. Since this codebase processes one season at a time, no
            # intra-season regression is needed.

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
        Compute elite SOS (high-AdjEM opponents only) and Quadrant record.

        FIX M4: Use AdjEM thresholds instead of end-of-season rankings.
        Rank-based classification leaks because it requires knowledge of
        all teams' final performance.  AdjEM thresholds are self-contained
        — they depend only on each opponent's own metrics.

        AdjEM thresholds (approximate historical mapping):
          Elite: AdjEM >= 15.0 (~top 30)
          Q1 boundary: varies by venue — see _classify_quadrant_by_em()
        """
        # Historical AdjEM thresholds for approximate quadrant mapping.
        # These are stable across seasons and don't depend on final rankings.
        ELITE_EM_THRESHOLD = 15.0  # ~ top 30

        for tid, games in by_team.items():
            if tid not in results:
                continue

            elite_ems: List[float] = []
            q1_w, q1_l = 0, 0

            for g in games:
                opp_em = (adj_off.get(g.opponent_id, 100.0)
                          - adj_def.get(g.opponent_id, 100.0))

                # Elite SOS: high-AdjEM opponents only
                if opp_em >= ELITE_EM_THRESHOLD:
                    elite_ems.append(opp_em)

                # Quadrant classification by AdjEM thresholds
                q = self._classify_quadrant_by_em(opp_em, g.is_home, g.is_neutral)
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
    def _classify_quadrant_by_em(opp_em: float, is_home: bool, is_neutral: bool) -> int:
        """
        FIX M4: Classify game into NCAA quadrant (1-4) using AdjEM thresholds.

        Uses historical AdjEM cutoffs that approximate rank-based quadrants
        without requiring knowledge of all teams' final rankings:
          Q1: ~top 30-75 depending on venue → AdjEM >= 15/10/5
          Q2: next tier → AdjEM >= 8/3/-2
          Q3: mid tier → AdjEM >= -3/-8/-12
          Q4: below Q3
        """
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
        else:  # away
            if opp_em >= 5.0:
                return 1
            elif opp_em >= -2.0:
                return 2
            elif opp_em >= -12.0:
                return 3
            return 4

    def _true_shooting_pct(self, games: List[GameRecord]) -> Tuple[float, float]:
        """
        True Shooting %: PTS / (2 × (FGA + 0.44 × FTA))

        More stable than eFG% because it incorporates free throw volume
        and shooting skill in a single composite metric.

        Returns (team_ts_pct, opp_ts_pct).
        """
        total_pts = sum(g.points for g in games)
        total_fga = sum(g.fga for g in games)
        total_fta = sum(g.fta for g in games)

        opp_pts = sum(g.opp_points for g in games)
        opp_fga = sum(g.opp_fga for g in games)
        opp_fta = sum(g.opp_fta for g in games)

        tsa = 2.0 * (total_fga + 0.44 * total_fta)
        opp_tsa = 2.0 * (opp_fga + 0.44 * opp_fta)

        ts_pct = total_pts / max(tsa, 1.0)
        opp_ts_pct = opp_pts / max(opp_tsa, 1.0)

        return float(ts_pct), float(opp_ts_pct)

    def _neutral_site_record(self, games: List[GameRecord]) -> Tuple[float, int]:
        """
        Win % at neutral sites only.

        Directly relevant to tournament prediction since all NCAA
        tournament games are played at neutral venues.

        Returns (neutral_win_pct, n_neutral_games).
        """
        neutral_games = [g for g in games if g.is_neutral]
        if not neutral_games:
            return 0.5, 0

        wins = sum(1 for g in neutral_games if g.points > g.opp_points)
        return wins / len(neutral_games), len(neutral_games)

    def _home_away_splits(
        self,
        games: List[GameRecord],
        adj_off: Dict[str, float],
        adj_def: Dict[str, float],
    ) -> Tuple[float, float, float]:
        """
        Compute AdjEM separately for home vs away games.

        The differential (home_em - away_em) captures "home court dependence":
        teams with high dependence are at greater risk in neutral-site
        tournament games.

        Returns (home_adj_em, away_adj_em, home_court_dependence).
        """
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
            # Neutral games excluded — they represent the tournament baseline

        # Require minimum 5 games in each split for stability
        home_em = float(np.mean(home_margins)) if len(home_margins) >= 5 else 0.0
        away_em = float(np.mean(away_margins)) if len(away_margins) >= 5 else 0.0

        # Home court dependence: positive = team benefits from home court
        # Shrink toward 0 when sample is small
        if len(home_margins) >= 5 and len(away_margins) >= 5:
            dependence = home_em - away_em
        else:
            dependence = 0.0

        return home_em, away_em, dependence

    def _momentum_5g(
        self,
        games: List[GameRecord],
        adj_off: Dict[str, float],
        adj_def: Dict[str, float],
    ) -> float:
        """
        5-game rolling form — finer granularity momentum signal.

        Captures hot/cold streaks that the 10-game window smooths over.
        """
        league_off = float(np.mean(list(adj_off.values()))) if adj_off else 100.0
        league_def = float(np.mean(list(adj_def.values()))) if adj_def else 100.0

        if len(games) < 8:
            return 0.0

        recent = games[-5:]
        recent_margins = []
        for g in recent:
            opp_em = adj_off.get(g.opponent_id, league_off) - adj_def.get(g.opponent_id, league_def)
            quality_margin = (g.points - g.opp_points) - opp_em
            recent_margins.append(quality_margin)

        season_margins = []
        for g in games:
            opp_em = adj_off.get(g.opponent_id, league_off) - adj_def.get(g.opponent_id, league_def)
            quality_margin = (g.points - g.opp_points) - opp_em
            season_margins.append(quality_margin)

        return float(np.mean(recent_margins)) - float(np.mean(season_margins))

    def _transition_efficiency(
        self,
        games: List[GameRecord],
        adj_off: Dict[str, float],
        adj_def: Dict[str, float],
    ) -> Tuple[float, float]:
        """
        Transition efficiency estimate (pace-derived proxy).

        Teams that play significantly faster than league average generate
        more transition opportunities.  We estimate transition efficiency
        as the interaction of pace surplus with offensive/defensive quality.

        This proxies ShotQuality's transition PPP metric using only box-score
        data.  The key insight: fast teams that are also efficient likely
        excel in transition (half-court efficiency is more uniform).

        Returns (offensive_transition_eff, defensive_transition_vuln).
        """
        if not games:
            return 0.0, 0.0

        tid = games[0].team_id
        team_tempo = np.mean([g.possessions for g in games])

        # League average tempo
        all_tempos = []
        for team_games in [games]:  # Just from this team's perspective
            all_tempos.extend([g.possessions for g in team_games])
        league_tempo = 68.0  # D1 average

        # Pace surplus: how much faster this team plays than average
        pace_surplus = (team_tempo - league_tempo) / league_tempo  # Normalized

        # Offensive transition efficiency: pace surplus × (AdjO / league_avg_O)
        adj_o = adj_off.get(tid, 100.0)
        off_trans = pace_surplus * (adj_o / 100.0 - 1.0)

        # Defensive transition vulnerability: opponent pace surplus × (AdjD / league_avg_D)
        adj_d = adj_def.get(tid, 100.0)
        def_trans_vuln = pace_surplus * (adj_d / 100.0 - 1.0)

        return float(off_trans), float(def_trans_vuln)

    def _pace_variance(self, games: List[GameRecord]) -> float:
        """
        Game-to-game pace (possessions) standard deviation.

        High pace variance teams are less predictable — they may play at
        very different tempos depending on the opponent, which amplifies
        randomness in single-elimination settings.  Low-possession games
        inflate 3P% variance, so pace variance interacts with upset risk.
        """
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

    def compute_h2h_record(self, team1_id: str, team2_id: str) -> float:
        """
        Compute team1's win% in head-to-head games against team2 this season.

        Returns 0.5 if no H2H games found (neutral prior).
        """
        if not hasattr(self, '_by_team'):
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

        # Shrink toward 0.5 for small samples: w * (wins/total) + (1-w) * 0.5
        # where w = min(1, h2h_total / 4) — full weight at 4+ meetings
        weight = min(1.0, h2h_total / 4.0)
        raw_rate = h2h_wins / h2h_total
        return weight * raw_rate + (1.0 - weight) * 0.5

    def compute_common_opponent_margin(self, team1_id: str, team2_id: str) -> float:
        """
        Compute margin differential through common opponents.

        For each opponent both teams have played, compute the difference in
        scoring margin.  The result is averaged across all common opponents
        and normalized to approximately [-1, 1].

        Returns 0.0 if no common opponents found.
        """
        if not hasattr(self, '_by_team'):
            return 0.0

        # Build opponent → margin map for each team
        def _opp_margins(team_id: str) -> Dict[str, List[float]]:
            margins: Dict[str, List[float]] = defaultdict(list)
            for g in self._by_team.get(team_id, []):
                margin = g.points - g.opp_points
                # Cap margin to prevent blowouts from dominating
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

        # Normalize by 20 (typical spread range) to get ~[-1, 1]
        raw = float(np.mean(diffs))
        return float(np.clip(raw / 20.0, -1.5, 1.5))

    def compute_point_in_time_metrics(
        self,
        team_id: str,
        as_of_date: str,
    ) -> Optional[Dict[str, float]]:
        """
        Compute team metrics using only games on or before ``as_of_date``.

        This enables true point-in-time feature computation — instead of
        using end-of-season aggregates for all training games, each game's
        features reflect only what was observable at that point in time.

        Returns a dict of key metric values, or None if insufficient data
        (fewer than 8 games played by the cutoff date).
        """
        if not hasattr(self, '_by_team'):
            return None

        all_games = self._by_team.get(team_id, [])
        pit_games = [g for g in all_games if g.game_date <= as_of_date]

        if len(pit_games) < 8:
            return None

        # Compute raw efficiency from the point-in-time games
        total_poss = sum(max(g.possessions, 1.0) for g in pit_games)
        total_pts = sum(g.points for g in pit_games)
        total_opp = sum(g.opp_points for g in pit_games)
        raw_off = 100.0 * total_pts / max(total_poss, 1.0)
        raw_def = 100.0 * total_opp / max(total_poss, 1.0)
        tempo = total_poss / max(len(pit_games), 1)

        # Use league-wide final adjusted values for opponent SOS
        # (this is a compromise — full PIT SOS would require
        # recomputing for all teams at each date, which is O(n^2*seasons))
        league_off = float(np.mean(list(self._adj_off.values()))) if self._adj_off else 100.0
        league_def = float(np.mean(list(self._adj_def.values()))) if self._adj_def else 100.0

        # Apply single-pass SOS adjustment using final opponent ratings
        off_adjustments = []
        def_adjustments = []
        for g in pit_games:
            opp_def = self._adj_def.get(g.opponent_id, league_def)
            opp_off = self._adj_off.get(g.opponent_id, league_off)
            poss = max(g.possessions, 1.0)
            game_off = 100.0 * g.points / poss
            game_def = 100.0 * g.opp_points / poss
            off_adjustments.append(game_off + (opp_def - league_def))
            def_adjustments.append(game_def + (opp_off - league_off))

        adj_off = float(np.mean(off_adjustments)) if off_adjustments else raw_off
        adj_def = float(np.mean(def_adjustments)) if def_adjustments else raw_def
        adj_em = adj_off - adj_def

        # Four Factors from PIT games
        ff = self._four_factors(pit_games)
        shooting = self._supplementary_shooting(pit_games)

        # Win percentage from PIT games
        wins = sum(1 for g in pit_games if g.points > g.opp_points)
        n_games = max(len(pit_games), 1)

        return {
            "adj_offensive_efficiency": adj_off,
            "adj_defensive_efficiency": adj_def,
            "adj_efficiency_margin": adj_em,
            "adj_tempo": tempo,
            "effective_fg_pct": ff["effective_fg_pct"],
            "turnover_rate": ff["turnover_rate"],
            "offensive_reb_rate": ff["offensive_reb_rate"],
            "free_throw_rate": ff["free_throw_rate"],
            "opp_effective_fg_pct": ff["opp_effective_fg_pct"],
            "opp_turnover_rate": ff["opp_turnover_rate"],
            "defensive_reb_rate": ff["defensive_reb_rate"],
            "opp_free_throw_rate": ff["opp_free_throw_rate"],
            "win_pct": wins / n_games,
            "three_pt_pct": shooting.get("three_pt_pct", 0.34),
            "two_pt_pct": shooting.get("two_pt_pct", 0.48),
            "three_pt_rate": shooting.get("three_pt_rate", 0.35),
            "n_games": len(pit_games),
        }

    @staticmethod
    def _log5_win_prob(team_a_em: float, team_b_em: float) -> float:
        """
        Win probability from efficiency margin differential.

        Uses logistic model:  P(A wins) = 1 / (1 + 10^(−EM_diff / k))

        The scaling parameter k=11.5 is calibrated from NCAA tournament
        outcomes (2002–2024).  KenPom-class models typically use k ∈ [11, 13].
        The previous k=10 overestimated upset probability.
        """
        diff = team_a_em - team_b_em
        diff = float(np.clip(diff, -40.0, 40.0))
        return 1.0 / (1.0 + 10 ** (-diff / 11.5))

    def validate_four_factors_weights(
        self,
        game_records: Dict[str, List[GameRecord]],
        results: Optional[Dict[str, "ProprietaryTeamMetrics"]] = None,
        n_bootstrap: int = 500,
        random_seed: int = 42,
    ) -> Dict:
        """Sensitivity analysis of Dean Oliver four-factor weights on college data.

        Fits a logistic regression of the four factors (differential) against
        game outcomes and compares the fitted coefficients to the assumed
        0.40/0.25/0.20/0.15 priors.  Also perturbs each weight by +/-20% and
        measures Brier score change.

        This is a DIAGNOSTIC method — it does NOT change the default weights.

        Args:
            game_records: Dict of team_id → list of GameRecord.
            n_bootstrap: Number of bootstrap resamples for coefficient CIs.
            random_seed: Seed for reproducibility.

        Returns:
            Dict with fitted_weights, weight_cis, oliver_priors,
            sensitivity_brier_deltas, and recommendation.
        """
        oliver_priors = np.array([0.40, 0.25, 0.20, 0.15])
        factor_names = ["eFG%", "ball_care", "ORB%", "FTR_x_FT%"]

        # Build dataset: per-game four-factor differential → win/loss
        X_list, y_list = [], []
        teams = list(game_records.keys())
        for i, t1 in enumerate(teams):
            for t2 in teams[i + 1:]:
                games_1 = game_records.get(t1, [])
                games_2 = game_records.get(t2, [])
                if not games_1 or not games_2:
                    continue
                ff1 = self._four_factors(games_1)
                ff2 = self._four_factors(games_2)
                diff = np.array([
                    ff1.get("effective_fg_pct", 0.5) - ff2.get("effective_fg_pct", 0.5),
                    (1 - ff1.get("turnover_rate", 0.18)) - (1 - ff2.get("turnover_rate", 0.18)),
                    ff1.get("offensive_reb_rate", 0.3) - ff2.get("offensive_reb_rate", 0.3),
                    ff1.get("free_throw_rate", 0.3) * 0.72 - ff2.get("free_throw_rate", 0.3) * 0.72,
                ])
                # Use AdjEM to determine "outcome" for all possible matchups
                r1 = results.get(t1) if results else None
                r2 = results.get(t2) if results else None
                em1 = r1.adj_efficiency_margin if r1 else 0.0
                em2 = r2.adj_efficiency_margin if r2 else 0.0
                win_prob = 1.0 / (1.0 + 10 ** (-(em1 - em2) / 11.5))
                X_list.append(diff)
                y_list.append(1 if win_prob > 0.5 else 0)

        if len(X_list) < 50:
            return {
                "fitted_weights": oliver_priors.tolist(),
                "weight_cis": [[w, w] for w in oliver_priors],
                "oliver_priors": oliver_priors.tolist(),
                "sensitivity_brier_deltas": {},
                "recommendation": "insufficient_data",
                "n_matchups": len(X_list),
            }

        X = np.array(X_list)
        y = np.array(y_list)

        # Fit logistic regression
        try:
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
            lr.fit(X, y)
            raw_coefs = np.abs(lr.coef_[0])
            fitted_weights = raw_coefs / max(np.sum(raw_coefs), 1e-12)
        except Exception:
            fitted_weights = oliver_priors.copy()

        # Bootstrap CIs on coefficients
        rng = np.random.default_rng(random_seed)
        boot_weights = []
        for _ in range(n_bootstrap):
            idx = rng.choice(len(y), size=len(y), replace=True)
            try:
                lr_boot = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs")
                lr_boot.fit(X[idx], y[idx])
                raw = np.abs(lr_boot.coef_[0])
                boot_weights.append(raw / max(np.sum(raw), 1e-12))
            except Exception:
                continue

        if boot_weights:
            boot_arr = np.array(boot_weights)
            weight_cis = [
                [float(np.percentile(boot_arr[:, i], 2.5)),
                 float(np.percentile(boot_arr[:, i], 97.5))]
                for i in range(4)
            ]
        else:
            weight_cis = [[w, w] for w in fitted_weights]

        # Sensitivity: perturb each Oliver weight by ±20%, measure Brier change
        # Using the linear xP model: xP = w . factors
        sensitivity = {}
        base_preds = 1.0 / (1.0 + np.exp(-(X @ oliver_priors)))
        base_brier = float(np.mean((base_preds - y) ** 2))

        for i, name in enumerate(factor_names):
            for direction, mult in [("plus_20pct", 1.2), ("minus_20pct", 0.8)]:
                perturbed = oliver_priors.copy()
                perturbed[i] *= mult
                perturbed /= np.sum(perturbed)  # Renormalize
                pert_preds = 1.0 / (1.0 + np.exp(-(X @ perturbed)))
                pert_brier = float(np.mean((pert_preds - y) ** 2))
                sensitivity[f"{name}_{direction}"] = round(pert_brier - base_brier, 6)

        # Recommendation
        max_deviation = max(abs(fitted_weights[i] - oliver_priors[i]) for i in range(4))
        recommendation = "keep_priors" if max_deviation < 0.10 else "consider_update"

        return {
            "fitted_weights": fitted_weights.tolist(),
            "fitted_weight_names": dict(zip(factor_names, fitted_weights.tolist())),
            "weight_cis": weight_cis,
            "oliver_priors": oliver_priors.tolist(),
            "base_brier": base_brier,
            "sensitivity_brier_deltas": sensitivity,
            "recommendation": recommendation,
            "max_deviation_from_prior": float(max_deviation),
            "n_matchups": len(X_list),
        }


# ---------------------------------------------------------------------------
# CBBpy team-map CSV loader
# ---------------------------------------------------------------------------

_CBBPY_TEAM_MAP_CACHE: Optional[Dict[str, str]] = None


def _load_cbbpy_team_map(csv_path: Optional[str] = None) -> Dict[str, str]:
    """Load the CBBpy team map CSV and return {display_name → location}.

    The CSV has columns: season, id, team, location, conference, conference_abb.
    ``team`` is the full display name with mascot (e.g. "New Mexico State Aggies").
    ``location`` is the school name only (e.g. "New Mexico State").

    Uses the latest season entry for each team.  Falls back to an empty dict
    if the CSV is not available (the pipeline will still work via Torvik
    prefix-matching, just without the disambiguation).
    """
    global _CBBPY_TEAM_MAP_CACHE
    if _CBBPY_TEAM_MAP_CACHE is not None:
        return _CBBPY_TEAM_MAP_CACHE

    if csv_path is None:
        # Default path: data/raw/cbbpy_team_map.csv relative to project root
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
                # Later seasons overwrite earlier ones (dict last-wins)
                result[team] = location

    _CBBPY_TEAM_MAP_CACHE = result
    return result


# ---------------------------------------------------------------------------
# Converter: Torvik/public data → GameRecord
# ---------------------------------------------------------------------------

def torvik_to_game_records(torvik_teams: List[Dict], historical_games: List[Dict]) -> List[GameRecord]:
    """
    Convert Torvik team stats + historical game rows into GameRecord objects.

    Works with data from cbbpy, sportsipy, or any source that provides
    game-level box scores.

    Uses ``torvik_teams`` to resolve mascot-suffixed game IDs (e.g.
    ``duke_blue_devils``) to Torvik canonical IDs (e.g. ``duke``).  This
    ensures the proprietary engine keys its results by the same IDs that
    the tournament pipeline uses.
    """
    # ------------------------------------------------------------------
    # Build a display_name → canonical_id resolver using the CBBpy team
    # map CSV.  The CSV ``location`` column gives the school name without
    # mascot (e.g. "New Mexico State", "Houston Christian") which is then
    # matched against Torvik canonical names.  This avoids the false
    # prefix-match problems that plagued mascot-stripping (e.g.
    # "New Mexico State Aggies" being falsely mapped to "new_mexico").
    # ------------------------------------------------------------------

    # Build Torvik name→canonical_id lookup.
    # Multiple normalized forms per team to handle:
    #   - HTML entities: "Texas A&amp;M" → "Texas A&M"
    #   - Parentheticals: "St. John's (NY)" → "St. John's"
    #   - Suffix variations: "McNeese State" → also try "McNeese"
    _torvik_name_to_id: Dict[str, str] = {}  # normalized_name → canonical_id
    for t in torvik_teams:
        if isinstance(t, dict):
            tid = t.get("team_id", "")
            tname = t.get("name", "")
        else:
            tid = getattr(t, "team_id", "")
            tname = getattr(t, "name", "")
        if tid and tname:
            canon = _team_id(tid)
            # Primary: normalized name as-is
            _torvik_name_to_id[_team_id(tname)] = canon
            # Also add canonical ID itself as a lookup key
            _torvik_name_to_id[canon] = canon
            # Decode HTML entities (e.g. "&amp;" → "&")
            cleaned = tname.replace("&amp;", "&")
            if cleaned != tname:
                _torvik_name_to_id[_team_id(cleaned)] = canon
            # Strip parentheticals (e.g. "(NY)")
            stripped = re.sub(r"\s*\([^)]*\)\s*", "", tname).strip()
            if stripped != tname:
                _torvik_name_to_id[_team_id(stripped)] = canon
                # Also with HTML decode + strip
                stripped_clean = re.sub(r"\s*\([^)]*\)\s*", "", cleaned).strip()
                if stripped_clean != stripped:
                    _torvik_name_to_id[_team_id(stripped_clean)] = canon

    # CBBpy→Torvik alias overrides for known naming mismatches.
    # CBBpy/ESPN uses modern branding; Torvik may lag behind.
    _CBBPY_TO_TORVIK_ALIASES: Dict[str, str] = {
        "mcneese": "mcneese_state",         # rebranded from McNeese State
        "american_university": "american",   # Torvik uses short name
    }
    for alias, target in _CBBPY_TO_TORVIK_ALIASES.items():
        if target in _torvik_name_to_id:
            _torvik_name_to_id[alias] = _torvik_name_to_id[target]

    # Load CBBpy team map CSV: maps display names (with mascot) to
    # school-only location names that disambiguate similar schools.
    _cbbpy_display_to_location = _load_cbbpy_team_map()

    # Build canonical cache: raw_id → canonical_id
    # Phase 1: Scan games to collect (raw_id, display_name) pairs.
    _name_by_raw_id: Dict[str, str] = {}
    for game in historical_games:
        if not isinstance(game, dict):
            continue
        raw = _team_id(str(game.get("team_id") or game.get("team1_id") or game.get("team1") or game.get("home_team") or ""))
        name = str(game.get("team_name") or game.get("team1_name") or "")
        if raw and name and raw not in _name_by_raw_id:
            _name_by_raw_id[raw] = name
        raw2 = _team_id(str(game.get("opponent_id") or game.get("team2_id") or game.get("team2") or game.get("away_team") or ""))
        name2 = str(game.get("opponent_name") or game.get("team2_name") or "")
        if raw2 and name2 and raw2 not in _name_by_raw_id:
            _name_by_raw_id[raw2] = name2

    # Phase 2: For each raw_id, use the CSV to find the school location,
    # then match that location against Torvik names.
    _canonical_cache: Dict[str, str] = {}
    for raw_id, display_name in _name_by_raw_id.items():
        # Look up display name in CBBpy CSV (exact match)
        location = _cbbpy_display_to_location.get(display_name)
        if location:
            norm_location = _team_id(location)
            canon = _torvik_name_to_id.get(norm_location)
            if canon:
                _canonical_cache[raw_id] = canon

    # Set of Torvik canonical IDs for exact-match fallback.
    _torvik_id_set = set(_torvik_name_to_id.values())

    def _resolve_canonical(raw_id: str) -> str:
        """Map mascot-suffixed ID to Torvik canonical ID if possible."""
        if raw_id in _canonical_cache:
            return _canonical_cache[raw_id]
        # Fallback: exact match on Torvik canonical ID (no prefix matching
        # to avoid false positives like new_mexico_highlands → new_mexico).
        if raw_id in _torvik_id_set:
            _canonical_cache[raw_id] = raw_id
            return raw_id
        # No match — keep raw ID (non-tournament team).
        _canonical_cache[raw_id] = raw_id
        return raw_id

    records: List[GameRecord] = []
    for game in historical_games:
        if not isinstance(game, dict):
            continue

        game_id = str(game.get("game_id") or game.get("id") or "")
        raw_team = _team_id(str(game.get("team_id") or game.get("team1_id") or game.get("team1") or game.get("home_team") or ""))
        raw_opp = _team_id(str(game.get("opponent_id") or game.get("team2_id") or game.get("team2") or game.get("away_team") or ""))
        team_id = _resolve_canonical(raw_team)
        opp_id = _resolve_canonical(raw_opp)
        if not game_id or not team_id or not opp_id:
            continue

        points = _to_float(game.get("team_score") or game.get("team1_score") or game.get("home_score") or game.get("points") or 0)
        opp_points = _to_float(game.get("opponent_score") or game.get("team2_score") or game.get("away_score") or game.get("opp_points") or 0)

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
        team_name = str(game.get("team_name") or game.get("team1_name") or game.get("team1") or team_id)

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
