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

    # --- Poisson Binomial resume metrics (NCAA selection committee) ---
    # SOR: P(avg_top25_team achieves ≤ this team's wins against this schedule)
    # Computed via Poisson Binomial CDF over per-game win probabilities.
    sor: float = 0.5

    # WAB via Poisson Binomial: actual_wins - E[wins_for_bubble_team]
    # More principled than game-by-game WAB: uses the full distribution
    # of expected wins rather than summing marginal per-game contributions.
    wab_poisson: float = 0.0

    # Proprietary xP per possession  (Four-Factors decomposition)
    offensive_xp_per_possession: float = 1.0
    defensive_xp_per_possession: float = 1.0

    # Shot distribution score  (rim + 3pt vs midrange proxy)
    shot_distribution_score: float = 0.0

    # 3-Point Variance  (game-to-game 3P% stdev — high = volatile)
    # C1: Default is D1 population prior (0.095), consistent with
    # Bayesian shrinkage in _three_point_variance().
    three_pt_variance: float = 0.095

    # Momentum  (last-10-game rolling AdjEM delta)
    momentum: float = 0.0
    recent_adj_em: float = 0.0   # AdjEM over last 10 games only

    # Pace-adjusted variance  (stdev of per-game scoring margin / pace factor)
    pace_adjusted_variance: float = 0.0

    # Consistency  (1 / (1 + stdev_margin))
    consistency: float = 0.5

    # C3: SOS-adjusted consistency — uses opponent-quality-adjusted residuals,
    # eliminating the schedule-variance confound in raw consistency.
    # Formula: 1 / (1 + stdev(actual_margin - opp_em))
    # Default 0.5 = neutral prior (same as raw consistency).
    sos_adjusted_consistency: float = 0.5

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

    # Road + neutral record (tournament-critical — all tourney games are neutral)
    road_neutral_wins: int = 0
    road_neutral_losses: int = 0
    road_neutral_games: int = 0
    road_neutral_win_pct: float = 0.5

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

    def __init__(self, require_cutoff_date: bool = True) -> None:
        # D2: Cross-season Elo state.
        # Set _elo_prior before calling compute() to initialize teams from
        # prior-season end-of-season ratings.  After compute(), _end_of_season_elo
        # contains this season's final ratings for passing to the next season.
        # Standard regression: start_elo = 0.75 * prior + 0.25 * 1500.
        self._elo_prior: Optional[Dict[str, float]] = None
        self._end_of_season_elo: Optional[Dict[str, float]] = None
        self._require_cutoff_date = require_cutoff_date

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
                         pre-tournament metrics.  **Always required** unless
                         require_cutoff_date=False was passed at construction
                         (testing only).

        Raises:
            ValueError: If cutoff_date is None and require_cutoff_date is True.
        """
        if cutoff_date is None and self._require_cutoff_date:
            raise ValueError(
                "cutoff_date is required to prevent temporal leakage. "
                "Pass cutoff_date='YYYY-MM-DD' (e.g. the day before the "
                "tournament starts) or set require_cutoff_date=False when "
                "constructing ProprietaryMetricsEngine for testing only."
            )

        if not game_records:
            return {}

        # Filter games to prevent temporal leakage
        if cutoff_date:
            logger.info("Filtering games to cutoff_date=%s", cutoff_date)
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
            sos_consistency = self._sos_adjusted_consistency(games, adj_off, adj_def)

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
            # C4: transition_efficiency REMOVED — speculative formula with no empirical
            # basis. pace_surplus * (AdjO/100 - 1) assumes fast teams generate extra
            # efficiency from transition without any play-by-play evidence. Fields
            # remain in dataclass at default 0.0 for backward compatibility.

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

        # --- Step 4: WAB (needs full-league rankings) ---
        self._compute_wab(results, by_team)

        # --- Step 4b: Poisson Binomial SOR & WAB (schedule-aware resume metrics) ---
        self._compute_sor_and_wab_poisson(results, by_team)

        # --- Step 5: Elo ratings (needs all games chronologically) ---
        # D2: Pass prior_elo if set (cross-season carryover); save end-of-season result.
        prior = getattr(self, '_elo_prior', None)
        self._end_of_season_elo = self._compute_elo_ratings(results, by_team, prior_elo=prior)

        # --- Step 6: Conference strength (needs league-wide AdjEM) ---
        self._compute_conference_strength(results, by_team, conference_map or {})

        # --- Step 7: Elite SOS + Quadrant record (needs full-league AdjEM) ---
        self._compute_elite_sos_and_quadrants(results, by_team, adj_off, adj_def)

        # --- Step 8: Rest days (days since last game) ---
        self._compute_rest_days(results, by_team, reference_date=cutoff_date)

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
        initial_adj_off: Optional[Dict[str, float]] = None,
        initial_adj_def: Optional[Dict[str, float]] = None,
        n_iterations: Optional[int] = None,
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

        Args:
            initial_adj_off/def: warm-start from a previous computation's
                converged values (e.g., prior date's result in incremental
                mode).  Reduces iterations needed for convergence.
            n_iterations: override SOS_ITERATIONS (default 15).  Use ~5
                when warm-starting from a nearby prior.
        """
        league_off = float(np.mean(list(raw_off.values()))) if raw_off else 100.0
        league_def = float(np.mean(list(raw_def.values()))) if raw_def else 100.0

        # Warm-start: use provided initial values if available (for
        # incremental computation where prior date's converged values are close).
        adj_off = dict(initial_adj_off) if initial_adj_off is not None else dict(raw_off)
        adj_def = dict(initial_adj_def) if initial_adj_def is not None else dict(raw_def)
        # Ensure all teams in raw_off/raw_def are represented (new teams
        # since the warm-start snapshot get initialized from raw values).
        if initial_adj_off is not None:
            for tid in raw_off:
                if tid not in adj_off:
                    adj_off[tid] = raw_off[tid]
            for tid in raw_def:
                if tid not in adj_def:
                    adj_def[tid] = raw_def[tid]

        iters = n_iterations if n_iterations is not None else self.SOS_ITERATIONS
        DAMPING = 0.7  # Blend factor: new = damp * computed + (1-damp) * old

        for _iteration in range(iters):
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
        # log5-equivalent: sigmoid of margin scaled by analytically derived constant.
        # D1 fix: 0.145 was documented as giving AdjEM=+10 → ~0.85 but actually
        # gives 0.810. Corrected to 0.1735 (= ln(17/3)/10), derived by solving
        # 0.85 = 1/(1+exp(-k*10)). Verified: AdjEM=+10 → 0.850, AdjEM=+20 → 0.970,
        # AdjEM=0 → 0.500 (consistent with KenPom calibration).
        return 1.0 / (1.0 + math.exp(-0.1735 * margin))

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
        Game-to-game 3P% standard deviation with Bayesian shrinkage.

        High variance teams are upset-prone on cold-shooting nights.

        C1 fix: Raw sample stdev from n=5 games has a chi-squared 95% CI
        spanning 0.6x–2.9x the true value.  We shrink toward the D1
        population stdev (~0.095) using a conjugate-prior approach on
        variance, then take sqrt.  This mirrors the 3PT% shrinkage
        (Pattern A) but is appropriate for a variance estimator.

        Formula (on variance):  shrunk_var = (n*s² + k*σ₀²) / (n + k)
        where σ₀ = 0.095 (D1 population stdev → σ₀² ≈ 0.009025),
        k = 8 (prior pseudo-observations, ~1/3 of a season).
        """
        THREE_PT_VAR_PRIOR_STD = 0.095   # D1 population game-to-game 3P% stdev
        THREE_PT_VAR_PRIOR_WEIGHT = 8.0  # pseudo-observations (~1/3 season)

        per_game_3p = []
        for g in games:
            if g.fg3a >= 5:  # only games with meaningful 3PA
                per_game_3p.append(g.fg3m / g.fg3a)

        if len(per_game_3p) < 5:
            return THREE_PT_VAR_PRIOR_STD

        sample_var = float(np.var(per_game_3p, ddof=1))
        prior_var = THREE_PT_VAR_PRIOR_STD ** 2
        n = len(per_game_3p)

        shrunk_var = (n * sample_var + THREE_PT_VAR_PRIOR_WEIGHT * prior_var) / (
            n + THREE_PT_VAR_PRIOR_WEIGHT
        )
        return float(np.sqrt(shrunk_var))

    def _momentum(
        self,
        games: List[GameRecord],
        adj_off: Dict[str, float],
        adj_def: Dict[str, float],
    ) -> Tuple[float, float]:
        """
        Rolling form over last 10 games vs season average, with Normal-Normal
        Bayesian shrinkage on the momentum delta.

        Returns (shrunk_momentum_delta, recent_adj_em).

        C1 fix: With n=10 recent games, SE(mean quality-adjusted margin) ≈
        11/√10 ≈ 3.5 pts.  True momentum signals are ~2-3 pts (SNR ≈ 0.7).
        Without shrinkage, estimated momentum is still substantially noisy.

        We apply Normal-Normal shrinkage to the momentum delta toward 0
        (prior: no momentum effect):
          λ = (σ²_margin / n_recent) / (σ²_margin / n_recent + σ²_signal)
          shrunk_momentum = raw_delta * (1 - λ)

        The recent_adj_em (absolute level) is returned unshrunk — it is used
        downstream to compare teams' current form levels, not deltas, and
        benefits from the implicit regularization of opponent-quality adjustment.

        Constants:
          MARGIN_SIGMA = 11.0 pts  (D1 game-to-game margin stdev)
          MOMENTUM_SIGNAL_SIGMA = 3.0 pts  (population SD of true momentum)

        At n=10: λ ≈ 12.1 / 21.1 ≈ 0.57 → retains 43% of raw delta
        At n=15: λ ≈ 8.1 / 17.1 ≈ 0.47 → retains 53% of raw delta
        At n=25: λ ≈ 4.8 / 13.8 ≈ 0.35 → retains 65% of raw delta
        """
        MARGIN_SIGMA_SQ = 11.0 ** 2         # D1 game margin variance
        MOMENTUM_SIGNAL_SIGMA_SQ = 3.0 ** 2  # Population SD of true momentum

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
        raw_delta = recent_em - season_em

        # Normal-Normal shrinkage: λ is noise fraction of total variance
        n_recent = len(recent_margins)
        lam = (MARGIN_SIGMA_SQ / n_recent) / (MARGIN_SIGMA_SQ / n_recent + MOMENTUM_SIGNAL_SIGMA_SQ)
        shrunk_momentum = raw_delta * (1.0 - lam)

        return float(shrunk_momentum), float(recent_em)

    def _pace_adjusted_variance(self, games: List[GameRecord]) -> float:
        """
        Scoring-margin variance adjusted for pace, with Bayesian shrinkage
        toward the D1 population margin stdev.

        Low-possession games inflate the influence of randomness.

        C1 fix: Raw sample stdev from n=5 games has a chi-squared 95% CI
        spanning 0.6x–2.9x the true value.  We shrink toward the D1
        population margin stdev using a conjugate-prior approach on variance,
        identical to _three_point_variance().

        Formula (on variance):  shrunk_var = (n*s² + k*σ₀²) / (n + k)
        where σ₀ = 11.0 pts (D1 population margin stdev → σ₀² = 121),
        k = 8 (pseudo-observations, ~1/3 season, matching 3PT variance fix).

        Returns prior σ₀ when n < 2 (can't estimate variance from 0–1 games).
        """
        PACE_VAR_PRIOR_STD = 11.0    # D1 population game margin stdev (points)
        PACE_VAR_PRIOR_WEIGHT = 8.0  # pseudo-observations (~1/3 season)

        adjusted_margins = []
        for g in games:
            margin = g.points - g.opp_points
            # Normalize to 70-possession baseline
            pace_factor = max(g.possessions, 40.0) / 70.0
            adjusted_margins.append(margin / pace_factor)

        n = len(adjusted_margins)
        if n < 2:
            return PACE_VAR_PRIOR_STD

        sample_var = float(np.var(adjusted_margins, ddof=1))
        prior_var = PACE_VAR_PRIOR_STD ** 2
        shrunk_var = (n * sample_var + PACE_VAR_PRIOR_WEIGHT * prior_var) / (
            n + PACE_VAR_PRIOR_WEIGHT
        )
        return float(np.sqrt(shrunk_var))

    # C8: Bayesian shrinkage constants for consistency metrics.
    # Same conjugate-prior pattern as _pace_adjusted_variance / _three_point_variance.
    CONSISTENCY_PRIOR_STD = 8.0    # D1 population scoring-margin stdev (Tier 2)
    CONSISTENCY_PRIOR_WEIGHT = 8.0  # pseudo-observations (~1/3 season, Tier 2)

    def _consistency(self, games: List[GameRecord]) -> float:
        """
        Inverse of scoring-margin stdev with Bayesian shrinkage.
        1 / (1 + shrunk_std).

        Consistent teams outperform in single-elimination.
        Shrinkage prevents extreme values from small samples (n=5-10).
        """
        if len(games) < 5:
            return 0.5

        margins = [g.points - g.opp_points for g in games]
        n = len(margins)
        sample_var = float(np.var(margins, ddof=1))
        prior_var = self.CONSISTENCY_PRIOR_STD ** 2
        shrunk_var = (n * sample_var + self.CONSISTENCY_PRIOR_WEIGHT * prior_var) / (
            n + self.CONSISTENCY_PRIOR_WEIGHT
        )
        return 1.0 / (1.0 + float(np.sqrt(shrunk_var)))

    def _sos_adjusted_consistency(
        self,
        games: List[GameRecord],
        adj_off: Dict[str, float],
        adj_def: Dict[str, float],
    ) -> float:
        """
        Consistency on SOS-adjusted residuals with Bayesian shrinkage.  C3 + C8 fix.

        Raw margin stdev conflates true inconsistency with schedule variance:
        a team playing both cupcakes and elite opponents shows high raw stdev
        even when performing exactly as expected in each game.

        Fix: compute residual_i = actual_margin_i - opp_em_i for each game,
        where opp_em = adj_off[opp] - adj_def[opp].  The stdev of these
        residuals isolates genuine game-to-game inconsistency from schedule
        heterogeneity.  Bayesian shrinkage toward population stdev prevents
        extreme values from small samples.

        Returns 0.5 (neutral prior) for n < 5.
        """
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
        prior_var = self.CONSISTENCY_PRIOR_STD ** 2
        shrunk_var = (n * sample_var + self.CONSISTENCY_PRIOR_WEIGHT * prior_var) / (
            n + self.CONSISTENCY_PRIOR_WEIGHT
        )
        return 1.0 / (1.0 + float(np.sqrt(shrunk_var)))

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

                # D4: Home-court adjustment via analytical log5 recomputation.
                # The old ±0.035 flat offset was ~3-7x too small.  HCA of
                # 3.75 points shifts win probability by ~0.10-0.25 at
                # competitive margins.  Recompute using the log5 formula
                # with HCA-adjusted efficiency margins.
                if not g.is_neutral:
                    if g.is_home:
                        # Team is home → bubble is away → opponent gets HCA
                        bubble_wp = self._log5_win_prob(bubble_em, opp_em + self.HCA_POINTS)
                    else:
                        # Team is away → bubble is home → bubble gets HCA
                        bubble_wp = self._log5_win_prob(bubble_em + self.HCA_POINTS, opp_em)

                bubble_wp = float(np.clip(bubble_wp, 0.01, 0.99))

                is_win = g.points > g.opp_points
                if is_win:
                    total_wab += 1.0 - bubble_wp
                else:
                    total_wab += 0.0 - bubble_wp

            results[tid].wab = round(total_wab, 2)

    # ------------------------------------------------------------------
    # Poisson Binomial SOR & WAB  (NCAA Selection Committee metrics)
    # ------------------------------------------------------------------

    @staticmethod
    def _poisson_binomial_cdf(k: int, probs: List[float]) -> float:
        """
        Compute P(X ≤ k) for a Poisson Binomial distribution.

        The Poisson Binomial is the distribution of the sum of n independent
        (but NOT identically distributed) Bernoulli random variables.  Each
        game has a different win probability p_i for the reference team.

        Uses the recursive DP method (O(n²) time, O(n) space) which is
        exact and numerically stable for n ≤ 40 (typical season length).
        For very long schedules, falls back to the DFT-CF method.

        Args:
            k: number of successes (wins) to evaluate CDF at
            probs: list of per-game win probabilities [p_1, ..., p_n]

        Returns:
            P(X ≤ k) — cumulative probability of k or fewer wins
        """
        n = len(probs)
        if n == 0:
            return 1.0
        k = max(0, min(k, n))

        # DP approach: pmf[j] = P(exactly j wins after processing games so far)
        # Initialize with "0 games processed" → P(0 wins) = 1.0
        pmf = np.zeros(n + 1, dtype=np.float64)
        pmf[0] = 1.0

        for p_i in probs:
            # Clamp to avoid degenerate probabilities
            p_i = float(np.clip(p_i, 1e-6, 1.0 - 1e-6))
            q_i = 1.0 - p_i
            # Process in reverse to avoid overwriting values we still need
            new_pmf = np.zeros_like(pmf)
            new_pmf[0] = pmf[0] * q_i
            for j in range(1, n + 1):
                new_pmf[j] = pmf[j] * q_i + pmf[j - 1] * p_i
            pmf = new_pmf

        # CDF = P(X ≤ k) = sum of pmf[0..k]
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
        """
        Compute Strength of Record (SOR) and WAB via Poisson Binomial distributions.

        SOR — "How impressive is this record given this schedule?"
        =====================================================================
        For each team, compute the probability that a reference team
        (AdjEM ≈ top-25, i.e., AdjEM = 10.0) would achieve *at most* the
        same number of wins against this team's exact schedule.

            SOR = P(X ≤ actual_wins)  where X ~ PoissonBinomial(p_1, ..., p_n)

        Each p_i is the probability that the reference team beats opponent_i,
        accounting for game location (home/away/neutral).

        High SOR (→1.0) means the team's record is very impressive: even a
        strong reference team would rarely do as well against this schedule.
        Low SOR (→0.0) means the record is unimpressive: a top-25 team would
        almost always match or exceed this win total.

        The NCAA selection committee adopted SOR in 2018 as a primary
        at-large selection and seeding metric.

        WAB (Poisson Binomial) — "How many more wins than a bubble team?"
        =====================================================================
        WAB_pb = actual_wins - E[wins for bubble team against same schedule]

        where E[wins] = sum of per-game bubble win probabilities (the mean
        of the Poisson Binomial distribution for the bubble team).

        This is mathematically equivalent to the per-game WAB sum but
        conceptually framed as the difference between actual wins and the
        expected wins from the Poisson Binomial distribution.  The Poisson
        Binomial framing is preferred because it makes clear that WAB is
        fundamentally about the *expected number of wins* against a specific
        schedule, not just a sum of per-game credits/debits.

        Constants:
            - Reference team AdjEM = 10.0 (top-25 caliber, ~NCAA average at-large)
            - Bubble team AdjEM = 5.0 (BUBBLE_EM_PRIOR, ~45th team)
            - HCA = 3.75 points (HCA_POINTS)
        """
        REFERENCE_EM = 10.0  # Top-25 team for SOR (NCAA committee standard)
        bubble_em = self.BUBBLE_EM_PRIOR  # ~45th team for WAB

        for tid, games in by_team.items():
            if tid not in results:
                continue

            ref_win_probs = []    # P(reference team wins game_i) — for SOR
            bubble_win_probs = [] # P(bubble team wins game_i) — for WAB

            for g in games:
                opp = results.get(g.opponent_id)
                opp_em = opp.adj_efficiency_margin if opp else 0.0

                # --- Reference team (SOR) ---
                if g.is_neutral:
                    ref_p = self._log5_win_prob(REFERENCE_EM, opp_em)
                elif g.is_home:
                    # Team was home → reference plays in team's home venue → ref gets HCA
                    ref_p = self._log5_win_prob(REFERENCE_EM + self.HCA_POINTS, opp_em)
                else:
                    # Team was away → reference plays away → opponent gets HCA
                    ref_p = self._log5_win_prob(REFERENCE_EM, opp_em + self.HCA_POINTS)

                ref_p = float(np.clip(ref_p, 0.005, 0.995))
                ref_win_probs.append(ref_p)

                # --- Bubble team (WAB Poisson) ---
                if g.is_neutral:
                    bub_p = self._log5_win_prob(bubble_em, opp_em)
                elif g.is_home:
                    # Team was home → bubble is away → opponent gets HCA
                    bub_p = self._log5_win_prob(bubble_em, opp_em + self.HCA_POINTS)
                else:
                    # Team was away → bubble is home → bubble gets HCA
                    bub_p = self._log5_win_prob(bubble_em + self.HCA_POINTS, opp_em)

                bub_p = float(np.clip(bub_p, 0.005, 0.995))
                bubble_win_probs.append(bub_p)

            actual_wins = sum(1 for g in games if g.points > g.opp_points)

            # --- SOR via Poisson Binomial CDF ---
            if ref_win_probs:
                sor = self._poisson_binomial_cdf(actual_wins, ref_win_probs)
            else:
                sor = 0.5

            # --- WAB via Poisson Binomial expected value ---
            expected_bubble_wins = self._poisson_binomial_mean(bubble_win_probs)
            wab_pb = actual_wins - expected_bubble_wins

            results[tid].sor = round(sor, 4)
            results[tid].wab_poisson = round(wab_pb, 2)

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
        prior_elo: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        MOV-adjusted Elo ratings per SBCB methodology.

        - Base K-factor: 38
        - MOV multiplier: ln(1 + |margin|) — saturating, ~2.1x at 7pt, ~3.4x at 30pt
        - HCA: +50 Elo for home team (B5: derived from HCA_POINTS * 13.3)
        - No intra-season autocorrection (A2: regress at season boundaries only)

        D2: Accept optional prior_elo for cross-season carryover.
        Standard regression: start_elo = 0.75 * prior + 0.25 * 1500.
        Matches FiveThirtyEight NFL Elo methodology (Silver, 2014).
        Teams absent from prior_elo start at mean (1500.0).

        Returns:
            Dict of {team_id: end_of_season_elo} for cross-season carryover.
        """
        # D2: Regression constants (Tier 2 — structurally constrained by 538 methodology)
        _ELO_REGRESSION = 0.25   # 25% toward mean per season boundary
        _ELO_MEAN = 1500.0

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

        # D2: Initialize from regressed prior if provided; else flat start at mean.
        if prior_elo:
            elo: Dict[str, float] = {
                t: (1.0 - _ELO_REGRESSION) * prior_elo.get(t, _ELO_MEAN) + _ELO_REGRESSION * _ELO_MEAN
                for t in by_team
            }
            # Teams not in prior (new programs, transfers) start at mean
            for t in by_team:
                if t not in elo:
                    elo[t] = _ELO_MEAN
        else:
            elo = defaultdict(lambda: _ELO_MEAN)
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

        # D2: Return end-of-season ratings for cross-season carryover
        return dict(elo)

    def _compute_conference_strength(
        self,
        results: Dict[str, ProprietaryTeamMetrics],
        by_team: Dict[str, List[GameRecord]],
        conference_map: Dict[str, str],
    ) -> None:
        """
        Compute average AdjEM of each team's conference peers.

        FIX-CONF: Enhanced to provide useful signal even without conference
        labels.  When conference_map is available, uses true conference
        membership.  When unavailable, computes "frequent opponent cluster
        strength" — the AdjEM of teams a team plays most often (typically
        conference opponents play 2x per season while non-conference play 1x).
        This adds unique signal beyond SOS because it focuses on the subset
        of opponents a team plays repeatedly.
        """
        if not conference_map:
            # FIX-CONF: Without conference labels, infer conference peers
            # from schedule frequency.  In NCAA basketball, teams play each
            # conference opponent 2x (home + away).  Non-conference opponents
            # are played 0-1 times.  Opponents played >= 2 times are likely
            # in the same conference.
            for tid in results:
                games = by_team.get(tid, [])
                if not games:
                    results[tid].conference_adj_em = results[tid].sos_adj_em
                    continue

                # Count opponent frequency
                opp_counts: Dict[str, int] = defaultdict(int)
                for g in games:
                    opp_counts[g.opponent_id] += 1

                # Opponents played >= 2 times are likely conference peers
                conf_peer_ems = []
                for opp_id, count in opp_counts.items():
                    if count >= 2 and opp_id in results:
                        conf_peer_ems.append(results[opp_id].adj_efficiency_margin)

                if conf_peer_ems:
                    results[tid].conference_adj_em = float(np.mean(conf_peer_ems))
                else:
                    # Not enough repeat opponents yet — use SOS fallback
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
            rn_w, rn_l = 0, 0

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

                # Road + neutral record (tournament-critical: all tourney
                # games are on neutral courts, so road/neutral performance
                # is more predictive than overall record)
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
        Win % at neutral sites only, with Beta-Binomial shrinkage.

        Directly relevant to tournament prediction since all NCAA
        tournament games are played at neutral venues.

        C1 fix: Raw wins/n gives 0.0 or 1.0 with n=1 — no signal.
        We use a Beta(α₀, α₀) conjugate prior centered at 0.5.
        Posterior mean = (wins + α₀) / (n + 2*α₀).

        α₀ = 2.0 → 4 pseudo-games (2W+2L prior) at 0.5.
        At n=0: returns 0.5 (full prior).
        At n=4: ~40% shrinkage toward 0.5.
        At n=10: ~29% shrinkage — elite teams escape the prior.

        Returns (shrunk_neutral_win_pct, n_neutral_games).
        """
        NEUTRAL_PRIOR_ALPHA = 2.0  # Beta(2, 2) → weak prior at 0.5

        neutral_games = [g for g in games if g.is_neutral]
        n = len(neutral_games)
        wins = sum(1 for g in neutral_games if g.points > g.opp_points)

        # Beta-Binomial posterior mean — handles n=0 cleanly
        shrunk_pct = (wins + NEUTRAL_PRIOR_ALPHA) / (n + 2 * NEUTRAL_PRIOR_ALPHA)
        return float(shrunk_pct), n

    def _home_away_splits(
        self,
        games: List[GameRecord],
        adj_off: Dict[str, float],
        adj_def: Dict[str, float],
    ) -> Tuple[float, float, float]:
        """
        Compute AdjEM separately for home vs away games, with Normal-Normal
        Bayesian shrinkage on each split.

        The differential (home_em - away_em) captures "home court dependence":
        teams with high dependence are at greater risk in neutral-site
        tournament games.

        C1 fix: The hard n<5 gate (returning 0.0) is a step function that
        creates discontinuities and ignores evidence from 1-4 games entirely.
        With n=5 home games, SE(mean_margin) ≈ 11/√5 ≈ 4.9 pts while the
        home-court signal is ~3-5 pts (SNR ≈ 1.0).

        We use Normal-Normal shrinkage independently per split:
          shrunk_mean = observed_mean * (1 - λ)
          λ = σ²_likelihood / (σ²_likelihood + σ²_signal)
          σ²_likelihood = MARGIN_SIGMA² / n   (sampling uncertainty)
          σ²_signal = HCA_SIGNAL_SIGMA²        (population SD of true split EM)

        This is equivalent to a Bayesian update with prior N(0, σ²_signal)
        — i.e., prior belief that a team has zero venue-specific advantage.

        Constants:
          MARGIN_SIGMA = 11.0 pts  (D1 game-to-game margin stdev)
          HCA_SIGNAL_SIGMA = 4.0 pts  (population SD of true home-EM across D1)

        At n=5 home games:  λ ≈ (121/5)/(121/5+16) ≈ 0.60 → 40% of raw kept
        At n=10:            λ ≈ (121/10)/(121/10+16) ≈ 0.43 → 57% kept
        At n=16:            λ ≈ (121/16)/(121/16+16) ≈ 0.32 → 68% kept

        Returns (shrunk_home_adj_em, shrunk_away_adj_em, shrunk_dependence).
        """
        MARGIN_SIGMA_SQ = 11.0 ** 2       # D1 game margin variance
        HCA_SIGNAL_SIGMA_SQ = 4.0 ** 2   # Population SD of true split EM

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

        def _shrink_split(margins: list) -> float:
            """Normal-Normal posterior mean; prior is N(0, σ²_signal)."""
            n = len(margins)
            if n == 0:
                return 0.0
            observed = float(np.mean(margins))
            lam = (MARGIN_SIGMA_SQ / n) / (MARGIN_SIGMA_SQ / n + HCA_SIGNAL_SIGMA_SQ)
            return observed * (1.0 - lam)

        home_em = _shrink_split(home_margins)
        away_em = _shrink_split(away_margins)

        # Home court dependence: shrunk difference — positive = benefits from home
        dependence = home_em - away_em

        return home_em, away_em, dependence

    def _momentum_5g(
        self,
        games: List[GameRecord],
        adj_off: Dict[str, float],
        adj_def: Dict[str, float],
    ) -> float:
        """
        5-game rolling form — finer granularity momentum signal, with
        Normal-Normal Bayesian shrinkage on the momentum delta.

        Captures hot/cold streaks that the 10-game window smooths over.

        C1 fix: With n=5 recent games, SE(mean quality-adjusted margin) ≈
        11/√5 ≈ 4.9 pts.  Typical real momentum signals are 2-3 pts above
        season baseline (SNR ≈ 0.5).  Without shrinkage the feature is
        more noise than signal.

        We apply Normal-Normal shrinkage to the momentum delta toward 0
        (prior belief: no momentum effect):
          λ = (σ²_margin / n_recent) / (σ²_margin / n_recent + σ²_signal)
          shrunk_momentum = raw_delta * (1 - λ)

        Constants:
          MARGIN_SIGMA = 11.0 pts  (D1 game-to-game margin stdev)
          MOMENTUM_SIGNAL_SIGMA = 3.0 pts  (population SD of true momentum)

        At n=5:  λ ≈ 24.2 / 33.2 ≈ 0.73 → retains 27% of raw delta
        At n=8:  λ ≈ 15.1 / 24.1 ≈ 0.63 → retains 37% of raw delta
        """
        MARGIN_SIGMA_SQ = 11.0 ** 2         # D1 game margin variance
        MOMENTUM_SIGNAL_SIGMA_SQ = 3.0 ** 2  # Population SD of true momentum

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

        raw_delta = float(np.mean(recent_margins)) - float(np.mean(season_margins))

        # Normal-Normal shrinkage: λ is the fraction of variance explained by noise
        n_recent = len(recent_margins)
        lam = (MARGIN_SIGMA_SQ / n_recent) / (MARGIN_SIGMA_SQ / n_recent + MOMENTUM_SIGNAL_SIGMA_SQ)
        return raw_delta * (1.0 - lam)

    # C4: _transition_efficiency() REMOVED.
    # The formula `pace_surplus * (AdjO/100 - 1)` assumed fast teams generate
    # extra offensive efficiency from transition opportunities.  Without play-by-play
    # data (ShotQuality transition PPP), this interaction term is unmotivated
    # correlation speculation.  Fields transition_efficiency and
    # defensive_transition_vulnerability remain in ProprietaryTeamMetrics at
    # default 0.0 for backward compatibility.

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
        reference_date: Optional[str] = None,
    ) -> None:
        """Compute days since last game for each team (rest advantage).

        Args:
            reference_date: YYYY-MM-DD date to measure rest from (e.g.,
                as_of_date or tournament cutoff).  If None, computes the
                gap between the team's last two games as a proxy.
        """
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
                    # Days between reference date and team's last game.
                    last_date = _dt.strptime(games[-1].game_date, "%Y-%m-%d")
                    delta = (ref_dt - last_date).days
                    results[tid].rest_days = float(max(delta, 0))
                elif len(games) >= 2:
                    # No reference date: gap between last two games.
                    last = _dt.strptime(games[-1].game_date, "%Y-%m-%d")
                    prev = _dt.strptime(games[-2].game_date, "%Y-%m-%d")
                    delta = (last - prev).days
                    results[tid].rest_days = float(max(delta, 0))
                else:
                    results[tid].rest_days = 5.0  # D1 average
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

        # D5: Build dataset from actual per-game four-factor differentials
        # and real game outcomes (not AdjEM-inferred).  This eliminates the
        # circular validation where AdjEM was used to determine outcomes
        # despite being derived from the same box-score data.
        X_list, y_list = [], []
        seen_game_ids: set = set()

        for team_id, games in game_records.items():
            for g in games:
                # Deduplicate: each game appears in both teams' records
                if g.game_id in seen_game_ids:
                    continue
                seen_game_ids.add(g.game_id)

                opp_games = game_records.get(g.opponent_id, [])
                if not opp_games:
                    continue

                # Find opponent's record for this same game
                opp_game = None
                for og in opp_games:
                    if og.game_id == g.game_id:
                        opp_game = og
                        break
                if opp_game is None:
                    continue

                # Per-game four-factor differentials
                ff_team = self._four_factors([g])
                ff_opp = self._four_factors([opp_game])
                diff = np.array([
                    ff_team.get("effective_fg_pct", 0.5) - ff_opp.get("effective_fg_pct", 0.5),
                    (1 - ff_team.get("turnover_rate", 0.18)) - (1 - ff_opp.get("turnover_rate", 0.18)),
                    ff_team.get("offensive_reb_rate", 0.3) - ff_opp.get("offensive_reb_rate", 0.3),
                    ff_team.get("free_throw_rate", 0.3) * 0.72 - ff_opp.get("free_throw_rate", 0.3) * 0.72,
                ])
                # Actual game outcome (not AdjEM-inferred)
                y = 1 if g.points > g.opp_points else 0
                X_list.append(diff)
                y_list.append(y)

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
# Converter: team_games JSON → GameRecord  (for incremental engine)
# ---------------------------------------------------------------------------


def team_games_to_game_records(
    team_games: List[Dict],
    season_year: int,
) -> List[GameRecord]:
    """Convert ``team_games`` arrays from historical JSON to GameRecord objects.

    The ``team_games`` format has one row per team per game.  Each game
    appears twice (one row for each team).  We use the mirror row to
    populate opponent box-score fields.

    Fields available: game_id, team_id, opponent_id, team_score,
    opponent_score, possessions, fgm, fga, fg3m, fg3a, fta, turnovers,
    orb, drb, date.

    FTM is derived: ``ftm = points - 2*fgm - fg3m``.
    All games are treated as neutral (historical data lacks venue info).
    """
    logger = logging.getLogger(__name__)

    # Build (game_id, team_id) → row index for opponent stat lookup.
    row_index: Dict[Tuple[str, str], Dict] = {}
    for row in team_games:
        gid = str(row.get("game_id", ""))
        tid = _team_id(str(row.get("team_id", "")))
        if gid and tid:
            row_index[(gid, tid)] = row

    records: List[GameRecord] = []
    seen: set = set()  # avoid duplicates when same game_id appears twice

    for row in team_games:
        gid = str(row.get("game_id", ""))
        raw_tid = _team_id(str(row.get("team_id", "")))
        raw_oid = _team_id(str(row.get("opponent_id", "")))
        if not gid or not raw_tid or not raw_oid:
            continue

        # Deduplicate: only process each (game_id, team_id) once.
        dedup_key = (gid, raw_tid)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        points = _to_float(row.get("team_score", 0))
        opp_points = _to_float(row.get("opponent_score", 0))
        if points == 0 and opp_points == 0:
            continue  # skip forfeits

        fgm = _to_float(row.get("fgm", 0))
        fga = _to_float(row.get("fga", 0))
        fg3m = _to_float(row.get("fg3m", 0))
        fg3a = _to_float(row.get("fg3a", 0))
        fta = _to_float(row.get("fta", 0))
        tov = _to_float(row.get("turnovers") or row.get("tov", 0))
        orb = _to_float(row.get("orb", 0))
        drb = _to_float(row.get("drb", 0))

        # Derive FTM: points = 2*(fgm - fg3m) + 3*fg3m + ftm
        #           → ftm = points - 2*fgm - fg3m
        ftm = max(points - 2.0 * fgm - fg3m, 0.0) if fgm > 0 else 0.0

        poss = _to_float(row.get("possessions", 0))
        if poss <= 0 and fga > 0:
            poss = fga - orb + tov + 0.475 * fta
        if poss <= 0:
            poss = max((points + opp_points) / 2.0, 30.0)

        # Opponent box scores from the mirror row.
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

        records.append(GameRecord(
            game_id=gid,
            game_date=game_date,
            team_id=raw_tid,
            team_name=team_name,
            opponent_id=raw_oid,
            points=points,
            opp_points=opp_points,
            possessions=poss,
            fga=fga, fgm=fgm, fg3a=fg3a, fg3m=fg3m, fta=fta, ftm=ftm,
            tov=tov, orb=orb, drb=drb,
            opp_fga=opp_fga, opp_fgm=opp_fgm, opp_fg3a=opp_fg3a, opp_fg3m=opp_fg3m,
            opp_fta=opp_fta, opp_ftm=opp_ftm, opp_tov=opp_tov, opp_orb=opp_orb, opp_drb=opp_drb,
            is_home=False, is_neutral=True,
        ))

    # ── Date inference ──────────────────────────────────────────────────
    # Many historical years have a single placeholder date for all games.
    # When this happens, infer chronological dates from game_id ordering
    # (game_ids are monotonically increasing within a season).
    unique_dates = set(r.game_date for r in records)
    if len(unique_dates) <= 1 and len(records) > 50:
        # Sort records by game_id to establish chronological order.
        records.sort(key=lambda r: (r.game_id, r.team_id))
        # Assign dates spread from season start (Nov 1) to season end (Apr 10).
        from datetime import date as _date, timedelta as _td
        season_start = _date(season_year - 1, 11, 1)
        season_end = _date(season_year, 4, 10)
        total_days = (season_end - season_start).days
        # Group by game_id to assign same date to both sides of a game.
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
            # Bucket to monthly granularity (~30-day intervals): reduces
            # unique dates to ~5 per season, making
            # IncrementalMetricsEngine.compute_as_of() cache-effective
            # (each unique date is computed once via SOS).  Weekly (7-day)
            # buckets would still produce 23 unique dates × 639 teams ×
            # 15 SOS iterations = too slow for holdout evaluation loops.
            day_offset = (int(frac * total_days) // 30) * 30
            inferred = season_start + _td(days=day_offset)
            r.game_date = inferred.isoformat()
        n_inferred_dates = len(set(r.game_date for r in records))
        logger.info(
            "Year %d: inferred dates for %d games → %d monthly buckets from game_id ordering.",
            season_year, n_unique_games, n_inferred_dates,
        )
        logger.warning(
            "Year %d: synthetic date inference active — rest_days and "
            "back_to_back features will be degenerate (coarse monthly "
            "buckets make same-day/next-day detection unreliable). "
            "Consider excluding these features for this season or "
            "treating them as missing.",
            season_year,
        )

    logger.info(
        "Year %d: converted %d team_games rows → %d GameRecords.",
        season_year, len(team_games), len(records),
    )
    return records


# ---------------------------------------------------------------------------
# IncrementalMetricsEngine — temporal-leakage-free feature computation
# ---------------------------------------------------------------------------


class IncrementalMetricsEngine:
    """Compute team metrics incrementally from box-score data.

    For a given date, produces ProprietaryTeamMetrics for each team using
    ONLY games played strictly before that date.  Designed for training
    sample construction where each sample must use only data available
    at the time of the game.

    Key design:
    - SOS is recomputed per unique date via ProprietaryMetricsEngine.compute()
      on the game prefix, warm-started from the previous date's converged values.
    - Elo is maintained as running state (inherently incremental), snapshotted
      at each date boundary and used to override the batch engine's Elo.
    - Results are cached per date.

    Usage::

        engine = IncrementalMetricsEngine(records, conf_map, prior_elo)
        metrics = engine.compute_as_of("2025-01-15")
        # metrics[team_id] → ProprietaryTeamMetrics using only pre-Jan-15 games
    """

    def __init__(
        self,
        game_records: List[GameRecord],
        conference_map: Optional[Dict[str, str]] = None,
        prior_elo: Optional[Dict[str, float]] = None,
    ):
        self._conference_map = conference_map
        self._prior_elo = prior_elo

        # Sort all records chronologically.
        self._all_records = sorted(game_records, key=lambda g: g.game_date)

        # Extract unique dates (sorted).
        date_set: set = set()
        for g in self._all_records:
            date_set.add(g.game_date)
        self._unique_dates = sorted(date_set)

        # Pre-compute Elo snapshots for all date boundaries.
        self._elo_snapshots: Dict[str, Dict[str, float]] = {}
        self._compute_all_elo_snapshots()

        # Cache: date → Dict[team_id, ProprietaryTeamMetrics]
        self._cache: Dict[str, Dict[str, ProprietaryTeamMetrics]] = {}

        # SOS warm-start state REMOVED.  Warm-starting from the previous
        # date's converged SOS values created a train/test distribution shift:
        # training samples used 5-iteration warm-start convergence while
        # prediction used 15-iteration cold-start.  Even though the per-call
        # error was small (~0.3%), it was systematically biased.  Cold-starting
        # every date ensures identical convergence behavior in training and
        # prediction.  The extra computation (10 iterations × ~160 dates ×
        # 350 teams) is negligible.

    def _compute_all_elo_snapshots(self) -> None:
        """Process all games chronologically and snapshot Elo at each date."""
        _ELO_REGRESSION = 0.25
        _ELO_MEAN = 1500.0
        K_BASE = 38.0
        HCA_POINTS = 3.75
        ELO_HCA = HCA_POINTS * 13.3

        # Initialize Elo from prior.
        elo: Dict[str, float] = defaultdict(lambda: _ELO_MEAN)
        if self._prior_elo:
            for tid, val in self._prior_elo.items():
                elo[tid] = (1.0 - _ELO_REGRESSION) * val + _ELO_REGRESSION * _ELO_MEAN

        # Deduplicate games (each game has two rows).
        seen_ids: set = set()
        deduped: List[GameRecord] = []
        for g in self._all_records:
            pair = (g.game_id, min(g.team_id, g.opponent_id))
            if pair not in seen_ids:
                seen_ids.add(pair)
                deduped.append(g)

        # Group deduped games by date.
        games_by_date: Dict[str, List[GameRecord]] = defaultdict(list)
        for g in deduped:
            games_by_date[g.game_date].append(g)

        # Snapshot Elo BEFORE each date's games, then process.
        for date in self._unique_dates:
            # Snapshot current Elo state (before today's games).
            self._elo_snapshots[date] = dict(elo)

            # Process today's games.
            for g in games_by_date.get(date, []):
                t1, t2 = g.team_id, g.opponent_id
                # Ensure both teams exist.
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

        # Store final Elo for cross-season carryover.
        self._end_of_season_elo = dict(elo)

    def get_end_of_season_elo(self) -> Dict[str, float]:
        """Return end-of-season Elo for cross-season carryover."""
        return dict(self._end_of_season_elo)

    def games_played_before(self, team_id: str, date: str) -> int:
        """Return count of games played by *team_id* strictly before *date*.

        Uses a precomputed lookup for O(1) amortized performance.
        """
        if not hasattr(self, "_games_before_cache"):
            # Build sorted per-team date lists once, then binary-search.
            from collections import defaultdict
            import bisect
            team_dates: Dict[str, list] = defaultdict(list)
            for g in self._all_records:
                team_dates[g.team_id].append(g.game_date)
            self._games_before_cache = {
                tid: sorted(dates) for tid, dates in team_dates.items()
            }
        dates = self._games_before_cache.get(team_id)
        if not dates:
            return 0
        import bisect
        return bisect.bisect_left(dates, date)

    def compute_as_of(self, as_of_date: str) -> Dict[str, ProprietaryTeamMetrics]:
        """Compute metrics for all teams using only games before *as_of_date*.

        Returns empty dict if fewer than 50 game records exist before that date
        (insufficient data for meaningful metrics).
        """
        if as_of_date in self._cache:
            return self._cache[as_of_date]

        # Filter to games strictly before as_of_date.
        prefix = [g for g in self._all_records if g.game_date < as_of_date]
        if len(prefix) < 50:
            self._cache[as_of_date] = {}
            return {}

        # Create a temporary engine for this prefix (no cutoff_date needed —
        # the prefix is already filtered to games before as_of_date).
        engine = ProprietaryMetricsEngine(require_cutoff_date=False)
        engine._elo_prior = self._prior_elo

        # Always cold-start SOS with full iterations to match the batch
        # engine's convergence behavior (used at prediction time).  See
        # warm-start removal note in __init__.

        # Build by_team dict from prefix games.
        by_team: Dict[str, List[GameRecord]] = defaultdict(list)
        for rec in prefix:
            by_team[rec.team_id].append(rec)
        for tid in by_team:
            by_team[tid].sort(key=lambda g: g.game_date)

        # Step 1: Raw efficiency.
        raw_off, raw_def, tempo, names = engine._raw_efficiency(by_team)

        # Step 2: SOS adjustment (cold-start, full iterations).
        adj_off, adj_def = engine._iterative_sos_adjust(
            by_team, raw_off, raw_def,
        )

        # Step 3: Compute all derived metrics (reuse engine internals).
        # Store adj values on the engine so its methods can access them.
        engine._by_team = dict(by_team)
        engine._adj_off = adj_off
        engine._adj_def = adj_def

        all_team_ids = sorted(by_team.keys())
        league_avg_em = float(np.mean([adj_off[t] - adj_def[t] for t in all_team_ids]))

        results: Dict[str, ProprietaryTeamMetrics] = {}
        for tid in all_team_ids:
            games = by_team[tid]
            adj_o = adj_off[tid]
            adj_d = adj_def[tid]
            em = adj_o - adj_d
            tmpo = tempo.get(tid, 67.0)
            n_games = len(games)

            # Four Factors
            ff = engine._four_factors(games)

            # Shooting
            shooting = engine._supplementary_shooting(games)

            # SOS
            sos = engine._strength_of_schedule(games, adj_off, adj_def, self._conference_map)

            # Luck
            luck = engine._correlated_gaussian_luck(games)

            # Pythagorean (barthag)
            barthag = engine._pythagorean_win_pct(adj_o, adj_d)

            # xP / shot distribution
            xp_o = engine._box_score_xp(ff, side="offense", ft_pct=shooting.get("free_throw_pct", 0.72))
            xp_d = engine._box_score_xp(ff, side="defense", ft_pct=shooting.get("opp_free_throw_pct", 0.72))
            shot_dist = engine._shot_distribution_score(games)

            # Variance metrics
            tpv = engine._three_point_variance(games)
            pav = engine._pace_adjusted_variance(games)

            # Consistency
            consistency = engine._consistency(games)
            sos_consistency = engine._sos_adjusted_consistency(games, adj_off, adj_def)

            # Momentum
            mom_delta, recent_em = engine._momentum(games, adj_off, adj_def)
            mom5_delta = engine._momentum_5g(games, adj_off, adj_def)

            # Extended box score
            ext = engine._extended_box_score_metrics(games)

            # Opponent shot selection
            opp_shots = engine._opponent_shot_selection(games)

            # True shooting (returns tuple: team_ts, opp_ts)
            ts_pct, opp_ts_pct = engine._true_shooting_pct(games)

            # Home/away splits
            home_em, away_em, hc_dep = engine._home_away_splits(games, adj_off, adj_def)

            # Neutral site
            nsw, nsg = engine._neutral_site_record(games)

            wins = sum(1 for g in games if g.points > g.opp_points)
            losses = n_games - wins
            adj_em = adj_o - adj_d

            results[tid] = ProprietaryTeamMetrics(
                team_id=tid,
                team_name=names.get(tid, tid),
                adj_offensive_efficiency=adj_o,
                adj_defensive_efficiency=adj_d,
                adj_efficiency_margin=adj_em,
                adj_tempo=tmpo,
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
                wab=0.0,  # computed below after all teams processed
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
                elo_rating=0.0,  # overridden below
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

        # WAB: needs all teams' results to assess opponent quality.
        engine._compute_wab(results, by_team)

        # Poisson Binomial SOR & WAB (schedule-aware resume metrics).
        engine._compute_sor_and_wab_poisson(results, by_team)

        # Elite SOS & Quadrants.
        engine._compute_elite_sos_and_quadrants(results, by_team, adj_off, adj_def)

        # Conference strength.
        engine._compute_conference_strength(results, by_team, self._conference_map or {})

        # Rest days (from most recent game to as_of_date).
        engine._compute_rest_days(results, by_team, reference_date=as_of_date)

        # Foul rate.
        for tid in all_team_ids:
            games = by_team[tid]
            total_pf = sum(g.pf for g in games)
            total_poss = sum(max(g.possessions, 1.0) for g in games)
            results[tid].foul_rate = total_pf / max(total_poss, 1.0)

        # Three-point regression signal.
        for tid in all_team_ids:
            raw_3p = results[tid].three_pt_pct
            n = len(by_team[tid])
            # Bayesian shrinkage toward D1 mean (0.345).
            league_mean_3p = 0.345
            k_prior = 50.0  # ~50 game pseudo-observations
            shrunk_3p = (n * raw_3p + k_prior * league_mean_3p) / (n + k_prior)
            results[tid].three_pt_regression_signal = shrunk_3p - league_mean_3p

        # Pace variance.
        for tid in all_team_ids:
            games = by_team[tid]
            if len(games) >= 5:
                poss_list = [g.possessions for g in games]
                results[tid].pace_variance = float(np.std(poss_list, ddof=1))
            else:
                results[tid].pace_variance = 5.0  # D1 prior

        # Override Elo with incremental snapshots.
        # _elo_snapshots[D] stores the Elo state *before* D's games are
        # processed — i.e., Elo after all games strictly before D.  Use
        # bisect to handle as_of_date values that fall on non-game days
        # (e.g., "YYYY-03-14" for tournament prediction), which have no
        # exact key in the snapshots dict.  bisect_left finds the first
        # unique_date >= as_of_date, whose snapshot equals "Elo after all
        # games before as_of_date" because there are no games between
        # as_of_date and that first game day.
        _idx = bisect.bisect_left(self._unique_dates, as_of_date)
        if _idx < len(self._unique_dates):
            elo_snap = self._elo_snapshots[self._unique_dates[_idx]]
        else:
            # as_of_date is after all game dates — use final season Elo.
            elo_snap = self.get_end_of_season_elo()
        for tid in all_team_ids:
            if tid in elo_snap:
                results[tid].elo_rating = elo_snap[tid]

        self._cache[as_of_date] = results
        return results

    @staticmethod
    def metrics_to_team_vector(
        m: ProprietaryTeamMetrics,
        seed: int = 0,
        external_rating_composite: float = 0.0,
        external_rating_spread: float = 0.0,
    ) -> np.ndarray:
        """Convert ProprietaryTeamMetrics to a 66-dim team feature vector.

        Matches the exact ordering in TeamFeatures.to_vector() so that
        matchup vectors built from incremental metrics are compatible with
        models trained on FeatureEngineer-produced vectors.

        Features requiring roster/PBP data (RAPM, volatility, etc.) are
        set to zero.  Features requiring external data (AP rank, coach
        data) use neutral defaults.

        Gap #1: Extended from 64 to 66 dims to include external_rating_composite
        and external_rating_spread (indices 66-67), matching TEAM_FEATURE_DIM=68.
        This ensures feature name alignment between training and inference.
        """
        from .feature_engineering import TEAM_FEATURE_DIM
        v = np.zeros(TEAM_FEATURE_DIM, dtype=np.float64)
        # Core efficiency (3)
        v[0] = m.adj_offensive_efficiency
        v[1] = m.adj_defensive_efficiency
        v[2] = m.adj_tempo
        # Four Factors offense (4)
        v[3] = m.effective_fg_pct
        v[4] = m.turnover_rate
        v[5] = m.offensive_reb_rate
        v[6] = m.free_throw_rate
        # Four Factors defense (4)
        v[7] = m.opp_effective_fg_pct
        v[8] = m.opp_turnover_rate
        v[9] = m.defensive_reb_rate
        v[10] = m.opp_free_throw_rate
        # Player metrics (6): indices 11-16 → zero (no roster data)
        # Experience (3): indices 17-19 → zero (no roster data)
        # Volatility (4): indices 20-23 → zero (no PBP data)
        # Shot quality (2)
        v[24] = m.offensive_xp_per_possession
        v[25] = m.shot_distribution_score
        # Schedule (4)
        v[26] = m.sos_adj_em
        v[27] = m.sos_opp_o
        v[28] = m.sos_opp_d
        v[29] = m.ncsos_adj_em
        # Luck (1)
        v[30] = m.luck
        # WAB (1)
        v[31] = m.wab
        # Poisson Binomial resume metrics (2) — SOR + WAB_poisson
        v[32] = m.sor
        v[33] = m.wab_poisson
        # Momentum (1)
        v[34] = m.momentum
        # Variance (2)
        v[35] = m.three_pt_variance
        v[36] = m.pace_adjusted_variance
        # Elo (1)
        v[37] = m.elo_rating
        # Free throw % (1)
        v[38] = m.free_throw_pct
        # Ball movement (2)
        v[39] = m.assist_to_turnover_ratio
        v[40] = m.assist_rate
        # Defensive disruption (2)
        v[41] = m.steal_rate
        v[42] = m.block_rate
        # Opponent shot selection (2)
        v[43] = m.opp_two_pt_pct_allowed
        v[44] = m.opp_three_pt_attempt_rate
        # Conference quality (1)
        v[45] = m.conference_adj_em
        # Shooting splits (2)
        v[46] = m.three_pt_pct
        v[47] = m.three_pt_rate
        # Defensive xP (1)
        v[48] = m.defensive_xp_per_possession
        # Win % (1)
        v[49] = m.win_pct
        # Elite SOS (1)
        v[50] = m.elite_sos
        # Q1 win % (1)
        v[51] = m.q1_win_pct
        # Foul rate (1)
        v[52] = m.foul_rate
        # 3PT regression (1)
        v[53] = m.three_pt_regression_signal
        # Rest days (1) — capped at 14
        v[54] = min(m.rest_days, 14.0)
        # Top5 minutes share (1) — zero (no roster)
        v[55] = 0.0
        # Preseason AP rank (1) — default unranked: 0.25
        v[56] = 0.25
        # Coach tournament exp (1) — default 0
        v[57] = 0.0
        # Coach tournament win rate (1) — default 0
        v[58] = 0.0
        # Coach deep run rate (1) — default 0
        v[59] = 0.0
        # Coach stage consistency (1) — default 0
        v[60] = 0.0
        # Per-stage coaching (3) — default 0 (no coach data incrementally)
        v[61] = 0.0  # coach_f4_appearances
        v[62] = 0.0  # coach_e8_appearances
        v[63] = 0.0  # coach_s16_appearances
        # Graph-theoretic SOS (2) — default 0 (no graph data incrementally)
        v[64] = 0.0  # pagerank_sos
        v[65] = 0.0  # multi_hop_sos
        # Win quality metrics (3) — default 0 (no graph data incrementally)
        v[66] = 0.0  # best_win_percentile
        v[67] = 0.0  # paper_tiger_score
        v[68] = 0.0  # dominance_ratio
        # Pace variance (1)
        v[69] = m.pace_variance
        # Conf tourney champion (1) — 0 (not known incrementally)
        v[70] = 0.0
        # Neutral-site win % (1)
        v[71] = m.neutral_site_win_pct
        # Home court dependence (1)
        v[72] = m.home_court_dependence
        # Tournament resume composite (1) — Bayesian-shrunk opponent quality
        from .tournament_features import compute_tournament_resume_composite
        v[73] = compute_tournament_resume_composite(
            q1_win_pct=m.q1_win_pct,
            q1_games=m.q1_wins + m.q1_losses,
            road_neutral_win_pct=m.road_neutral_win_pct,
            road_neutral_games=m.road_neutral_games,
            elite_sos=m.elite_sos,
            sor=m.sor,
        )
        # Position RAPM (2) — zero (no roster)
        v[74] = 0.0
        v[75] = 0.0

        # External rating composite + spread (2)
        v[76] = external_rating_composite
        v[77] = external_rating_spread

        # Seed strength (1)
        if seed > 0:
            v[78] = float(np.log1p(17 - seed) / np.log1p(16))
        else:
            v[78] = 0.0

        # NaN/inf guard
        bad = np.isnan(v) | np.isinf(v)
        if bad.any():
            v[bad] = 0.0
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
        """Build matchup vector from two TEAM_FEATURE_DIM team vectors.

        Layout: [0:N] diff, [N:N+5] absolute, [N+5:N+12] interactions.
        Matches MatchupFeatures.to_vector() from feature_engineering.py.
        """
        # Differential — includes all team features
        diff = v1 - v2

        # Absolute-level features — use resolved indices from feature_engineering
        from .feature_engineering import ABSOLUTE_LEVEL_INDICES
        _ABS_IDX = ABSOLUTE_LEVEL_INDICES if ABSOLUTE_LEVEL_INDICES else [0, 1, 26, 37, 49]
        absolute = np.array([(v1[i] + v2[i]) / 2.0 for i in _ABS_IDX])

        # Interaction features (7)
        tempo_interaction = (v1[2] * v2[2]) / 4624.0
        tempo_diff = v1[2] - v2[2]
        eff_diff = (v1[0] - v1[1]) - (v2[0] - v2[1])
        style_mismatch = (tempo_diff * eff_diff) / 600.0
        h2h_record = 0.5
        common_opp_margin = 0.0
        travel_advantage = 0.0
        if engine is not None and team1_id and team2_id:
            h2h_record = engine.compute_h2h_record(team1_id, team2_id)
            common_opp_margin = engine.compute_common_opponent_margin(team1_id, team2_id)

        if seed1 > 0 and seed2 > 0:
            seed_interaction = (seed1 * seed2) / 128.0 - 1.0
            # Gap #3: Raw seed difference — strongest single predictor
            seed_diff = (seed1 - seed2) / 15.0
        else:
            seed_interaction = 0.0
            seed_diff = 0.0

        interactions = np.array([
            tempo_interaction, style_mismatch, h2h_record,
            common_opp_margin, travel_advantage, seed_interaction,
            seed_diff,
        ])

        return np.concatenate([diff, absolute, interactions])


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

def torvik_to_game_records(
    torvik_teams: List[Dict],
    historical_games: List[Dict],
    season_year: int,
) -> List[GameRecord]:
    """
    Convert Torvik team stats + historical game rows into GameRecord objects.

    Works with data from cbbpy, sportsipy, or any source that provides
    game-level box scores.

    Uses ``torvik_teams`` to resolve mascot-suffixed game IDs (e.g.
    ``duke_blue_devils``) to Torvik canonical IDs (e.g. ``duke``).  This
    ensures the proprietary engine keys its results by the same IDs that
    the tournament pipeline uses.
    """
    logger = logging.getLogger(__name__)
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

        raw_date = game.get("game_date") or game.get("date") or game.get("start_date")
        if not raw_date:
            logger.warning(
                "Game %s missing date; using %d-01-01 fallback.",
                game_id or "unknown",
                season_year,
            )
        game_date = str(raw_date or f"{season_year}-01-01")
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
