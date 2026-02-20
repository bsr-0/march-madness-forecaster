"""
Game theory optimization for bracket pools.

Maximizes Expected Value (EV) relative to competitors by finding
high-leverage picks with favorable Win Probability / Pick Percentage ratios.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import namedtuple
import numpy as np


_BranchPick = namedtuple('_BranchPick', ['p_win', 'survival', 'pts'])


@dataclass
class LeveragePick:
    """A pick with leverage analysis."""
    
    team_id: str
    team_name: str
    seed: int
    region: str
    
    # Probabilities
    model_probability: float  # Our model's probability
    public_pick_percentage: float  # % of public picking this team
    
    # Round
    round_name: str  # "Champion", "Final Four", etc.
    points_value: int  # Points for picking correctly
    
    @property
    def leverage_ratio(self) -> float:
        """Ratio of model prob to public percentage."""
        if self.public_pick_percentage <= 0:
            return float('inf')
        return self.model_probability / self.public_pick_percentage
    
    @property
    def expected_value(self) -> float:
        """Expected points from this pick."""
        return self.model_probability * self.points_value
    
    @property
    def expected_value_differential(self) -> float:
        """
        Expected points gained vs public.
        
        Positive = outperform public, Negative = underperform.
        """
        public_ev = self.public_pick_percentage * self.points_value
        return self.expected_value - public_ev
    
    def __str__(self) -> str:
        return (
            f"{self.team_name} ({self.seed}) - {self.round_name}: "
            f"Model {self.model_probability:.1%} vs Public {self.public_pick_percentage:.1%} "
            f"(Leverage: {self.leverage_ratio:.2f}x)"
        )


@dataclass
class BracketConfiguration:
    """A complete bracket configuration."""
    
    picks: Dict[str, str]  # game_id -> winner team_id
    champion: str
    final_four: List[str]
    
    # Metadata
    strategy: str = "balanced"  # "balanced", "chalk", "contrarian"
    expected_points: float = 0.0
    variance: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "picks": self.picks,
            "champion": self.champion,
            "final_four": self.final_four,
            "strategy": self.strategy,
            "expected_points": self.expected_points,
        }


@dataclass
class TeamMetadata:
    """Metadata used for richer game-theory recommendations."""

    team_name: str
    seed: int
    region: str


class LeverageCalculator:
    """
    Calculates leverage ratios for bracket optimization.
    
    Leverage = Win Probability / Pick Percentage
    
    High leverage picks offer better value because:
    - If correct: You beat more competitors
    - If wrong: Everyone else is also wrong
    """
    
    def __init__(
        self,
        model_probs: Dict[str, Dict[str, float]],
        public_picks: Dict[str, Dict[str, float]],
        scoring_system: Dict[str, int] = None,
        team_metadata: Optional[Dict[str, TeamMetadata]] = None,
    ):
        """
        Initialize calculator.
        
        Args:
            model_probs: team_id -> {round: probability}
            public_picks: team_id -> {round: percentage}
            scoring_system: round -> points (e.g., {"R64": 1, "R32": 2, ...})
        """
        self.model_probs = model_probs
        self.public_picks = public_picks
        self.team_metadata = team_metadata or {}
        
        # Standard NCAA bracket scoring
        self.scoring_system = scoring_system or {
            "R64": 10,
            "R32": 20,
            "S16": 40,
            "E8": 80,
            "F4": 160,
            "CHAMP": 320,
        }

    def _team_meta(self, team_id: str) -> TeamMetadata:
        return self.team_metadata.get(
            team_id,
            TeamMetadata(team_name=team_id, seed=0, region=""),
        )
    
    def find_leverage_picks(
        self,
        min_leverage: float = 1.5,
        min_probability: float = 0.05
    ) -> List[LeveragePick]:
        """
        Find all high-leverage picks.
        
        Args:
            min_leverage: Minimum leverage ratio
            min_probability: Minimum model probability (filter noise)
            
        Returns:
            List of LeveragePick sorted by leverage
        """
        leverage_picks = []
        
        for team_id, probs in self.model_probs.items():
            public = self.public_picks.get(team_id, {})
            
            for round_name, model_prob in probs.items():
                if model_prob < min_probability:
                    continue
                
                public_pct = public.get(round_name, 0.001)  # Avoid div by 0
                leverage = model_prob / public_pct
                
                if leverage >= min_leverage:
                    meta = self._team_meta(team_id)
                    leverage_picks.append(LeveragePick(
                        team_id=team_id,
                        team_name=meta.team_name,
                        seed=meta.seed,
                        region=meta.region,
                        model_probability=model_prob,
                        public_pick_percentage=public_pct,
                        round_name=round_name,
                        points_value=self.scoring_system.get(round_name, 0),
                    ))
        
        # Sort by leverage ratio
        leverage_picks.sort(key=lambda x: x.leverage_ratio, reverse=True)
        
        return leverage_picks
    
    def find_fade_picks(
        self,
        max_leverage: float = 0.7
    ) -> List[LeveragePick]:
        """
        Find over-picked teams to fade.
        
        Teams with leverage < 1 are over-valued by public.
        
        Args:
            max_leverage: Maximum leverage to include
            
        Returns:
            List of teams to avoid
        """
        fade_picks = []
        
        for team_id, probs in self.model_probs.items():
            public = self.public_picks.get(team_id, {})
            
            for round_name, model_prob in probs.items():
                public_pct = public.get(round_name, 0.001)
                leverage = model_prob / public_pct
                
                if leverage <= max_leverage and public_pct > 0.1:
                    meta = self._team_meta(team_id)
                    fade_picks.append(LeveragePick(
                        team_id=team_id,
                        team_name=meta.team_name,
                        seed=meta.seed,
                        region=meta.region,
                        model_probability=model_prob,
                        public_pick_percentage=public_pct,
                        round_name=round_name,
                        points_value=self.scoring_system.get(round_name, 0),
                    ))
        
        # Sort by leverage (lowest first = most over-picked)
        fade_picks.sort(key=lambda x: x.leverage_ratio)
        
        return fade_picks


class ParetoOptimizer:
    """
    Generates Pareto-optimal brackets along risk/reward frontier.
    
    - Conservative brackets: Maximize expected points
    - Aggressive brackets: Maximize upside potential
    - The Pareto frontier offers best risk/reward tradeoffs
    """
    
    def __init__(
        self,
        leverage_calculator: LeverageCalculator,
        pool_size: int = 100
    ):
        """
        Initialize optimizer.
        
        Args:
            leverage_calculator: LeverageCalculator instance
            pool_size: Number of entries in bracket pool
        """
        self.calculator = leverage_calculator
        self.pool_size = pool_size
    
    def generate_pareto_brackets(
        self,
        num_brackets: int = 5
    ) -> List[BracketConfiguration]:
        """
        Generate brackets along Pareto frontier.
        
        Args:
            num_brackets: Number of brackets to generate
            
        Returns:
            List of bracket configurations from conservative to aggressive
        """
        brackets = []
        
        # Risk levels from 0 (chalk) to 1 (max contrarian)
        risk_levels = np.linspace(0, 1, num_brackets)
        
        for risk in risk_levels:
            if risk < 0.2:
                strategy = "chalk"
            elif risk < 0.6:
                strategy = "balanced"
            else:
                strategy = "contrarian"
            
            bracket = self._generate_bracket(risk, strategy)
            brackets.append(bracket)
        
        return brackets
    
    def _generate_bracket(
        self,
        risk_level: float,
        strategy: str
    ) -> BracketConfiguration:
        """
        Generate single bracket with given risk level.
        
        Args:
            risk_level: 0 = chalk, 1 = max contrarian
            strategy: Strategy name
            
        Returns:
            BracketConfiguration
        """
        if self._can_build_full_bracket():
            picks, champion, final_four, expected_points, variance = self._generate_full_bracket(risk_level)
        else:
            picks, champion, final_four, expected_points, variance = self._generate_summary_bracket(risk_level)

        return BracketConfiguration(
            picks=picks,
            champion=champion,
            final_four=final_four,
            strategy=strategy,
            expected_points=expected_points,
            variance=variance,
        )

    def _expected_value_contribution(self, team_id: str, round_name: str) -> Tuple[float, float]:
        """EV and variance for a single pick (legacy, round-independent)."""
        p = float(self.calculator.model_probs.get(team_id, {}).get(round_name, 0.0))
        pts = float(self.calculator.scoring_system.get(round_name, 0))
        ev = p * pts
        var = p * (1.0 - p) * (pts ** 2)
        return ev, var

    def _path_ev_var(self, team_id: str, round_name: str, survival_prob: float) -> Tuple[float, float]:
        """EV and variance conditional on team surviving to this round.

        The team must have survived all prior rounds (probability = survival_prob)
        AND win this round.  This properly models the path dependence in bracket
        scoring where later-round points require earlier-round wins.
        """
        p_win = float(self.calculator.model_probs.get(team_id, {}).get(round_name, 0.0))
        pts = float(self.calculator.scoring_system.get(round_name, 0))
        # Joint probability of surviving AND winning
        joint = survival_prob * p_win
        ev = joint * pts
        # Var(X) = E[X^2] - E[X]^2 where X = pts if joint event, else 0
        ex2 = joint * pts ** 2
        var = ex2 - ev ** 2
        return ev, var

    def _risk_adjusted_score(self, team_id: str, round_name: str, risk_level: float) -> float:
        model_prob = float(self.calculator.model_probs.get(team_id, {}).get(round_name, 0.0))
        public_prob = float(self.calculator.public_picks.get(team_id, {}).get(round_name, 0.01))
        leverage = model_prob / max(public_prob, 0.001)
        seed = max(1, self.calculator._team_meta(team_id).seed or 16)

        # Low-risk brackets favor strong seeds; high-risk brackets reward leverage.
        seed_strength = (17 - seed) / 16.0
        return model_prob * (leverage ** risk_level) + (1.0 - risk_level) * 0.02 * seed_strength

    def _pick_winner(self, team1_id: str, team2_id: str, round_name: str, risk_level: float) -> str:
        score1 = self._risk_adjusted_score(team1_id, round_name, risk_level)
        score2 = self._risk_adjusted_score(team2_id, round_name, risk_level)
        if abs(score1 - score2) > 1e-9:
            return team1_id if score1 > score2 else team2_id

        # Deterministic tie-break: better seed, then lexical order for stability.
        seed1 = max(1, self.calculator._team_meta(team1_id).seed or 16)
        seed2 = max(1, self.calculator._team_meta(team2_id).seed or 16)
        if seed1 != seed2:
            return team1_id if seed1 < seed2 else team2_id
        return team1_id if team1_id < team2_id else team2_id

    def _build_seed_map(self) -> Dict[str, Dict[int, str]]:
        by_region: Dict[str, Dict[int, str]] = {r: {} for r in ("East", "West", "South", "Midwest")}
        for team_id in self.calculator.model_probs.keys():
            meta = self.calculator._team_meta(team_id)
            if meta.region not in by_region:
                continue
            if meta.seed <= 0:
                continue
            by_region[meta.region][meta.seed] = team_id
        return by_region

    def _can_build_full_bracket(self) -> bool:
        by_region = self._build_seed_map()
        for region in ("East", "West", "South", "Midwest"):
            seeds = by_region.get(region, {})
            if len(seeds) < 16:
                return False
            if any(seed not in seeds for seed in range(1, 17)):
                return False
        return True

    def _generate_full_bracket(self, risk_level: float) -> Tuple[Dict[str, str], str, List[str], float, float]:
        by_region = self._build_seed_map()
        seed_order = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]

        picks: Dict[str, str] = {}
        region_champs: Dict[str, str] = {}
        expected_points = 0.0
        variance = 0.0

        # Track survival probability for each picked team and branch picks
        # for covariance computation.
        team_survival: Dict[str, float] = {}
        # all_branch_picks maps a branch key to a list of _BranchPick entries.
        # A branch key is the tuple of game indices along one path through the
        # bracket tree.  For simplicity, we key by (region, branch_idx) where
        # branch_idx groups the 8 R64 games into increasingly merged branches.
        all_branch_picks: List[List[_BranchPick]] = []

        for region in ("East", "West", "South", "Midwest"):
            region_teams = by_region[region]

            # --- R64 ----------------------------------------------------------
            r64_winners: List[str] = []
            # Each R64 game starts a branch; 8 branches per region.
            region_branches: List[List[_BranchPick]] = [[] for _ in range(8)]

            for game_idx, (high_seed, low_seed) in enumerate(seed_order):
                team1 = region_teams[high_seed]
                team2 = region_teams[low_seed]
                winner = self._pick_winner(team1, team2, "R64", risk_level)
                picks[f"R64_{region}_{high_seed}v{low_seed}"] = winner

                survival = 1.0  # First round: no prior survival requirement
                p_win = float(self.calculator.model_probs.get(winner, {}).get("R64", 0.0))
                pts = float(self.calculator.scoring_system.get("R64", 0))

                ev, var = self._path_ev_var(winner, "R64", survival)
                expected_points += ev
                variance += var

                team_survival[winner] = p_win  # After R64, survival = p_win_R64
                region_branches[game_idx].append(_BranchPick(p_win=p_win, survival=survival, pts=pts))
                r64_winners.append(winner)

            # --- R32 ----------------------------------------------------------
            r32_winners: List[str] = []
            # Merge adjacent branches pairwise: 8 -> 4
            r32_branches: List[List[_BranchPick]] = []

            for idx in range(0, len(r64_winners), 2):
                winner = self._pick_winner(r64_winners[idx], r64_winners[idx + 1], "R32", risk_level)
                picks[f"R32_{region}_{idx // 2 + 1}"] = winner

                survival = team_survival.get(winner, 1.0)
                p_win = float(self.calculator.model_probs.get(winner, {}).get("R32", 0.0))
                pts = float(self.calculator.scoring_system.get("R32", 0))

                ev, var = self._path_ev_var(winner, "R32", survival)
                expected_points += ev
                variance += var

                team_survival[winner] = survival * p_win
                # Merge the two R64 branches + this R32 pick
                merged = region_branches[idx] + region_branches[idx + 1]
                merged.append(_BranchPick(p_win=p_win, survival=survival, pts=pts))
                r32_branches.append(merged)
                r32_winners.append(winner)

            # --- S16 ----------------------------------------------------------
            s16_winners: List[str] = []
            s16_branches: List[List[_BranchPick]] = []

            for idx in range(0, len(r32_winners), 2):
                winner = self._pick_winner(r32_winners[idx], r32_winners[idx + 1], "S16", risk_level)
                picks[f"S16_{region}_{idx // 2 + 1}"] = winner

                survival = team_survival.get(winner, 1.0)
                p_win = float(self.calculator.model_probs.get(winner, {}).get("S16", 0.0))
                pts = float(self.calculator.scoring_system.get("S16", 0))

                ev, var = self._path_ev_var(winner, "S16", survival)
                expected_points += ev
                variance += var

                team_survival[winner] = survival * p_win
                merged = r32_branches[idx] + r32_branches[idx + 1]
                merged.append(_BranchPick(p_win=p_win, survival=survival, pts=pts))
                s16_branches.append(merged)
                s16_winners.append(winner)

            # --- E8 -----------------------------------------------------------
            e8_winner = self._pick_winner(s16_winners[0], s16_winners[1], "E8", risk_level)
            picks[f"E8_{region}"] = e8_winner

            survival = team_survival.get(e8_winner, 1.0)
            p_win = float(self.calculator.model_probs.get(e8_winner, {}).get("E8", 0.0))
            pts = float(self.calculator.scoring_system.get("E8", 0))

            ev, var = self._path_ev_var(e8_winner, "E8", survival)
            expected_points += ev
            variance += var

            team_survival[e8_winner] = survival * p_win
            region_branch = s16_branches[0] + s16_branches[1]
            region_branch.append(_BranchPick(p_win=p_win, survival=survival, pts=pts))
            all_branch_picks.append(region_branch)
            region_champs[region] = e8_winner

        # --- Final Four -------------------------------------------------------
        final_four = [
            region_champs["East"],
            region_champs["West"],
            region_champs["South"],
            region_champs["Midwest"],
        ]

        semi1 = self._pick_winner(region_champs["East"], region_champs["West"], "F4", risk_level)
        semi2 = self._pick_winner(region_champs["South"], region_champs["Midwest"], "F4", risk_level)
        picks["F4_East_West"] = semi1
        picks["F4_South_Midwest"] = semi2

        for semi_winner, branch_indices in [(semi1, (0, 1)), (semi2, (2, 3))]:
            survival = team_survival.get(semi_winner, 1.0)
            p_win = float(self.calculator.model_probs.get(semi_winner, {}).get("F4", 0.0))
            pts = float(self.calculator.scoring_system.get("F4", 0))

            ev, var = self._path_ev_var(semi_winner, "F4", survival)
            expected_points += ev
            variance += var

            team_survival[semi_winner] = survival * p_win

        # --- Championship -----------------------------------------------------
        champion = self._pick_winner(semi1, semi2, "CHAMP", risk_level)
        picks["CHAMP"] = champion

        survival = team_survival.get(champion, 1.0)
        p_win_champ = float(self.calculator.model_probs.get(champion, {}).get("CHAMP", 0.0))
        pts_champ = float(self.calculator.scoring_system.get("CHAMP", 0))

        ev, var = self._path_ev_var(champion, "CHAMP", survival)
        expected_points += ev
        variance += var

        # --- Covariance from path dependence ----------------------------------
        # For each branch, picks i (earlier) and j (later) are correlated
        # because j cannot score unless i also won.  For pick i at index a and
        # pick j at index b > a in the same branch:
        #   cov(X_i, X_j) = survival_j * p_win_i * (1 - p_win_i) * pts_i * pts_j
        # We add 2 * cov to the total variance for each pair.
        for branch in all_branch_picks:
            n = len(branch)
            for i in range(n):
                for j in range(i + 1, n):
                    bp_i = branch[i]
                    bp_j = branch[j]
                    cov = bp_j.survival * bp_i.p_win * (1.0 - bp_i.p_win) * bp_i.pts * bp_j.pts
                    variance += 2.0 * cov

        return picks, champion, final_four, expected_points, variance

    def _generate_summary_bracket(self, risk_level: float) -> Tuple[Dict[str, str], str, List[str], float, float]:
        champion_scores: Dict[str, float] = {}
        for team_id, probs in self.calculator.model_probs.items():
            champ_prob = probs.get("CHAMP", 0.0)
            public_prob = self.calculator.public_picks.get(team_id, {}).get("CHAMP", 0.01)
            leverage = champ_prob / max(public_prob, 0.001)
            champion_scores[team_id] = champ_prob * (leverage ** risk_level)

        champion = max(champion_scores, key=champion_scores.get) if champion_scores else ""

        f4_candidates = []
        for team_id, probs in self.calculator.model_probs.items():
            f4_prob = probs.get("F4", 0.0)
            public_prob = self.calculator.public_picks.get(team_id, {}).get("F4", 0.01)
            leverage = f4_prob / max(public_prob, 0.001)
            f4_candidates.append((team_id, f4_prob * (leverage ** risk_level), f4_prob))
        f4_candidates.sort(key=lambda x: x[1], reverse=True)

        final_four: List[str] = []
        used_regions = set()
        for team_id, _score, _f4_prob in f4_candidates:
            if len(final_four) == 4:
                break
            region = self.calculator._team_meta(team_id).region
            if region and region in used_regions:
                continue
            final_four.append(team_id)
            if region:
                used_regions.add(region)
        if len(final_four) < 4:
            for team_id, _score, _f4_prob in f4_candidates:
                if team_id in final_four:
                    continue
                final_four.append(team_id)
                if len(final_four) == 4:
                    break

        picks = {"CHAMP": champion}
        for idx, team_id in enumerate(final_four, start=1):
            picks[f"F4_{idx}"] = team_id

        expected_points = 0.0
        variance = 0.0
        if champion:
            ev, var = self._expected_value_contribution(champion, "CHAMP")
            expected_points += ev
            variance += var
        for team_id in final_four:
            ev, var = self._expected_value_contribution(team_id, "F4")
            expected_points += ev
            variance += var
        return picks, champion, final_four, expected_points, variance
    
    def recommend_for_pool_size(self) -> str:
        """
        Recommend bracket strategy based on pool size.
        
        Returns:
            Strategy recommendation
        """
        if self.pool_size <= 10:
            return (
                "Small pool (<10): Use CHALK bracket. "
                "Pick favorites and win with accuracy."
            )
        elif self.pool_size <= 50:
            return (
                "Medium pool (10-50): Use BALANCED bracket. "
                "Mix chalk with 1-2 leverage picks."
            )
        elif self.pool_size <= 200:
            return (
                "Large pool (50-200): Use MODERATE LEVERAGE bracket. "
                "Need differentiation - pick 2-3 contrarian plays."
            )
        else:
            return (
                "Very large pool (200+): Use HIGH LEVERAGE bracket. "
                "Must be different to win. Target undervalued teams."
            )


def calculate_pool_dynamics(
    pool_size: int,
    model_probs: Dict[str, float],
    public_picks: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate expected performance dynamics for different strategies.
    
    Args:
        pool_size: Number of competitors
        model_probs: Our championship probabilities
        public_picks: Public championship picks
        
    Returns:
        Strategy performance estimates
    """
    results = {}
    
    # Chalk strategy - pick by probability
    chalk_champion = max(model_probs, key=model_probs.get)
    chalk_prob = model_probs[chalk_champion]
    chalk_public = public_picks.get(chalk_champion, 0.3)
    
    # Expected competitors also picking chalk champion
    expected_chalk_competition = pool_size * chalk_public
    
    # Probability of winning pool with chalk
    # If chalk wins, we split with ~N*public_pct people
    chalk_win_share = chalk_prob / max(expected_chalk_competition, 1)
    results["chalk_ev"] = chalk_win_share
    
    # Contrarian strategy - pick high leverage
    contrarian_options = [
        (tid, model_probs[tid] / max(public_picks.get(tid, 0.01), 0.01))
        for tid in model_probs
    ]
    contrarian_options.sort(key=lambda x: x[1], reverse=True)
    
    if contrarian_options:
        contrarian_champion = contrarian_options[0][0]
        contrarian_prob = model_probs[contrarian_champion]
        contrarian_public = public_picks.get(contrarian_champion, 0.01)
        
        expected_contrarian_competition = pool_size * contrarian_public
        contrarian_win_share = contrarian_prob / max(expected_contrarian_competition, 1)
        results["contrarian_ev"] = contrarian_win_share
        results["leverage_ratio"] = contrarian_prob / contrarian_public
    
    # Recommendation
    if results.get("contrarian_ev", 0) > results.get("chalk_ev", 0):
        results["recommendation"] = "contrarian"
    else:
        results["recommendation"] = "chalk"
    
    return results


@dataclass
class PoolAnalysis:
    """Complete pool analysis with bracket recommendations."""
    
    pool_size: int
    strategy_evs: Dict[str, float]
    recommended_strategy: str
    leverage_picks: List[LeveragePick]
    fade_picks: List[LeveragePick]
    pareto_brackets: List[BracketConfiguration]
    
    def print_summary(self) -> None:
        """Print analysis summary."""
        print(f"\n{'='*60}")
        print(f"POOL ANALYSIS - {self.pool_size} entries")
        print(f"{'='*60}")
        
        print(f"\nRecommended Strategy: {self.recommended_strategy.upper()}")
        
        print("\n📈 TOP LEVERAGE PICKS:")
        for pick in self.leverage_picks[:5]:
            print(f"  {pick}")
        
        print("\n📉 TEAMS TO FADE (Overvalued by public):")
        for pick in self.fade_picks[:5]:
            print(f"  {pick}")
        
        print(f"\n{'='*60}")


def analyze_pool(
    pool_size: int,
    model_probs: Dict[str, Dict[str, float]],
    public_picks: Dict[str, Dict[str, float]],
    scoring_system: Optional[Dict[str, int]] = None,
    team_metadata: Optional[Dict[str, TeamMetadata]] = None,
) -> PoolAnalysis:
    """
    Complete pool analysis.
    
    Args:
        pool_size: Number of entries
        model_probs: Model probabilities by team and round
        public_picks: Public pick percentages
        
    Returns:
        PoolAnalysis with recommendations
    """
    calculator = LeverageCalculator(
        model_probs,
        public_picks,
        scoring_system=scoring_system,
        team_metadata=team_metadata,
    )
    optimizer = ParetoOptimizer(calculator, pool_size)
    
    leverage_picks = calculator.find_leverage_picks()
    fade_picks = calculator.find_fade_picks()
    
    pareto_brackets = optimizer.generate_pareto_brackets()
    
    # Calculate strategy EVs
    championship_model = {
        tid: probs.get("CHAMP", 0) 
        for tid, probs in model_probs.items()
    }
    championship_public = {
        tid: probs.get("CHAMP", 0.01) 
        for tid, probs in public_picks.items()
    }
    
    dynamics = calculate_pool_dynamics(
        pool_size, championship_model, championship_public
    )
    
    return PoolAnalysis(
        pool_size=pool_size,
        strategy_evs=dynamics,
        recommended_strategy=dynamics.get("recommendation", "balanced"),
        leverage_picks=leverage_picks,
        fade_picks=fade_picks,
        pareto_brackets=pareto_brackets,
    )
