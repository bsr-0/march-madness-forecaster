"""
Game theory optimization for bracket pools.

Maximizes Expected Value (EV) relative to competitors by finding
high-leverage picks with favorable Win Probability / Pick Percentage ratios.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from copy import deepcopy


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
        scoring_system: Dict[str, int] = None
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
        
        # Standard NCAA bracket scoring
        self.scoring_system = scoring_system or {
            "R64": 10,
            "R32": 20,
            "S16": 40,
            "E8": 80,
            "F4": 160,
            "CHAMP": 320,
        }
    
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
                    leverage_picks.append(LeveragePick(
                        team_id=team_id,
                        team_name=team_id,  # Would be resolved from data
                        seed=0,  # Would be resolved from data
                        region="",  # Would be resolved from data
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
                    fade_picks.append(LeveragePick(
                        team_id=team_id,
                        team_name=team_id,
                        seed=0,
                        region="",
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
        # Get leverage picks
        leverage_picks = self.calculator.find_leverage_picks(
            min_leverage=1.0 + risk_level  # Higher risk = higher leverage threshold
        )
        
        picks = {}
        champion = None
        final_four = []
        
        # Select champion based on risk level
        champion_picks = [p for p in leverage_picks if p.round_name == "CHAMP"]
        
        if champion_picks and risk_level > 0.3:
            # Pick contrarian champion
            champion = champion_picks[0].team_id
        else:
            # Pick by model probability
            champion_probs = {
                tid: probs.get("CHAMP", 0) 
                for tid, probs in self.calculator.model_probs.items()
            }
            champion = max(champion_probs, key=champion_probs.get)
        
        # Expected points calculation would be more sophisticated
        expected_points = sum(
            p.expected_value 
            for p in leverage_picks[:10]  # Top 10 leverage picks
        )
        
        return BracketConfiguration(
            picks=picks,
            champion=champion,
            final_four=final_four,
            strategy=strategy,
            expected_points=expected_points,
        )
    
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
    calculator = LeverageCalculator(model_probs, public_picks, scoring_system=scoring_system)
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
