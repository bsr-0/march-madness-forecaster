"""
Monte Carlo simulation engine for bracket prediction.

Runs 50,000+ simulations of the full tournament with noise injection
to model uncertainty (injuries, variance, etc.).
"""

from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import multiprocessing
from tqdm import tqdm


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulation."""
    
    num_simulations: int = 50000
    noise_std: float = 0.05  # Standard deviation of probability noise
    injury_probability: float = 0.02  # Per-game injury probability
    random_seed: Optional[int] = None
    parallel_workers: int = None  # None = use all CPUs
    batch_size: int = 1000  # Simulations per batch
    
    def __post_init__(self):
        if self.parallel_workers is None:
            self.parallel_workers = max(1, multiprocessing.cpu_count() - 1)


@dataclass
class TeamSimState:
    """Team state during simulation."""
    
    team_id: str
    seed: int
    region: str
    base_strength: float
    injury_impact: float = 0.0
    
    @property
    def effective_strength(self) -> float:
        return self.base_strength * (1.0 - self.injury_impact)


@dataclass
class SimulationResult:
    """Result of a single bracket simulation."""
    
    champion: str
    final_four: List[str]
    elite_eight: List[str]
    sweet_sixteen: List[str]
    round_of_32: List[str]
    round_results: List[List[str]]  # Winners by round
    
    def to_dict(self) -> dict:
        return {
            "champion": self.champion,
            "final_four": self.final_four,
            "elite_eight": self.elite_eight,
            "sweet_sixteen": self.sweet_sixteen,
        }


@dataclass
class AggregatedResults:
    """Aggregated results from all simulations."""
    
    championship_odds: Dict[str, float] = field(default_factory=dict)
    final_four_odds: Dict[str, float] = field(default_factory=dict)
    elite_eight_odds: Dict[str, float] = field(default_factory=dict)
    sweet_sixteen_odds: Dict[str, float] = field(default_factory=dict)
    round_of_32_odds: Dict[str, float] = field(default_factory=dict)
    
    # Matchup specific
    matchup_odds: Dict[Tuple[str, str], float] = field(default_factory=dict)
    
    # Upset metrics
    upset_counts: Dict[Tuple[int, int], int] = field(default_factory=dict)
    
    num_simulations: int = 0
    
    def get_leverage_picks(
        self, 
        public_odds: Dict[str, float],
        min_leverage: float = 1.5
    ) -> List[Tuple[str, float, float, float]]:
        """
        Find high-leverage picks.
        
        Args:
            public_odds: Public championship pick percentages
            min_leverage: Minimum leverage ratio
            
        Returns:
            List of (team_id, model_odds, public_odds, leverage)
        """
        leverage_picks = []
        
        for team_id, model_odds in self.championship_odds.items():
            public = public_odds.get(team_id, 0.001)  # Avoid division by zero
            leverage = model_odds / public
            
            if leverage >= min_leverage:
                leverage_picks.append((team_id, model_odds, public, leverage))
        
        return sorted(leverage_picks, key=lambda x: x[3], reverse=True)


class MonteCarloEngine:
    """
    Monte Carlo simulation engine for tournament brackets.
    
    Features:
    - Parallel simulation execution
    - Noise injection for uncertainty modeling
    - Injury scenario simulation
    - Progress tracking
    """
    
    def __init__(
        self,
        predict_fn: Callable[[str, str], float],
        config: SimulationConfig = None
    ):
        """
        Initialize Monte Carlo engine.
        
        Args:
            predict_fn: Function that takes (team1_id, team2_id) and returns
                       probability that team1 wins
            config: Simulation configuration
        """
        self.predict_fn = predict_fn
        self.config = config or SimulationConfig()
        
        if self.config.random_seed:
            np.random.seed(self.config.random_seed)
    
    def simulate_tournament(
        self,
        bracket: "TournamentBracket",
        show_progress: bool = True
    ) -> AggregatedResults:
        """
        Run full Monte Carlo simulation.
        
        Args:
            bracket: Tournament bracket structure
            show_progress: Show progress bar
            
        Returns:
            Aggregated simulation results
        """
        results = AggregatedResults()
        all_results: List[SimulationResult] = []
        
        # Run simulations in parallel
        num_batches = (self.config.num_simulations + self.config.batch_size - 1) // self.config.batch_size
        
        if show_progress:
            pbar = tqdm(total=self.config.num_simulations, desc="Simulating")
        
        # For parallelization (simplified - actual would use ProcessPoolExecutor)
        for batch_idx in range(num_batches):
            batch_start = batch_idx * self.config.batch_size
            batch_end = min(batch_start + self.config.batch_size, self.config.num_simulations)
            batch_count = batch_end - batch_start
            
            for _ in range(batch_count):
                result = self._simulate_single(bracket)
                all_results.append(result)
            
            if show_progress:
                pbar.update(batch_count)
        
        if show_progress:
            pbar.close()
        
        # Aggregate results
        results = self._aggregate_results(all_results)
        
        return results
    
    def _simulate_single(self, bracket: "TournamentBracket") -> SimulationResult:
        """
        Simulate single tournament.
        
        Args:
            bracket: Tournament bracket
            
        Returns:
            Single simulation result
        """
        # Initialize team states with noise
        team_states = {}
        for team in bracket.teams:
            noise = np.random.normal(0, self.config.noise_std)
            injury = np.random.random() < self.config.injury_probability
            
            team_states[team.team_id] = TeamSimState(
                team_id=team.team_id,
                seed=team.seed,
                region=team.region,
                base_strength=team.strength + noise,
                injury_impact=0.15 if injury else 0.0
            )
        
        round_results = []
        current_teams = bracket.first_round_matchups.copy()
        
        # Simulate each round from 64 -> 1.
        while len(current_teams) > 1:
            round_winners = []
            
            for i in range(0, len(current_teams), 2):
                if i + 1 < len(current_teams):
                    team1 = current_teams[i]
                    team2 = current_teams[i + 1]
                    
                    winner = self._simulate_game(
                        team_states[team1],
                        team_states[team2]
                    )
                    round_winners.append(winner)
                else:
                    round_winners.append(current_teams[i])
            
            round_results.append(round_winners)
            current_teams = round_winners
        
        champion = current_teams[0] if current_teams else None
        
        # Extract milestone teams by canonical round index.
        # round_results[0] = winners of R64 (teams in R32), round_results[3] = Final Four teams.
        round_of_32 = round_results[0] if len(round_results) >= 1 else []
        sweet_sixteen = round_results[1] if len(round_results) >= 2 else []
        elite_eight = round_results[2] if len(round_results) >= 3 else []
        final_four = round_results[3] if len(round_results) >= 4 else []
        
        return SimulationResult(
            champion=champion,
            final_four=final_four,
            elite_eight=elite_eight,
            sweet_sixteen=sweet_sixteen,
            round_of_32=round_of_32,
            round_results=round_results
        )
    
    def _simulate_game(
        self,
        team1: TeamSimState,
        team2: TeamSimState
    ) -> str:
        """
        Simulate single game with noise.
        
        Args:
            team1: Team 1 state
            team2: Team 2 state
            
        Returns:
            Winner team_id
        """
        # Get base probability
        base_prob = self.predict_fn(team1.team_id, team2.team_id)
        
        # Adjust for injuries
        strength_ratio = team1.effective_strength / max(team2.effective_strength, 0.01)
        injury_adjustment = (strength_ratio - 1.0) * 0.1  # Subtle adjustment
        
        # Add game-level noise
        game_noise = np.random.normal(0, self.config.noise_std)
        
        # Final probability
        final_prob = np.clip(base_prob + injury_adjustment + game_noise, 0.01, 0.99)
        
        # Simulate outcome
        if np.random.random() < final_prob:
            return team1.team_id
        else:
            return team2.team_id
    
    def _aggregate_results(
        self,
        results: List[SimulationResult]
    ) -> AggregatedResults:
        """
        Aggregate individual simulation results.
        
        Args:
            results: List of simulation results
            
        Returns:
            Aggregated statistics
        """
        n = len(results)
        
        championship_counts: Dict[str, int] = {}
        final_four_counts: Dict[str, int] = {}
        elite_eight_counts: Dict[str, int] = {}
        sweet_sixteen_counts: Dict[str, int] = {}
        round_of_32_counts: Dict[str, int] = {}
        
        for result in results:
            # Championship
            if result.champion:
                championship_counts[result.champion] = championship_counts.get(result.champion, 0) + 1
            
            # Final Four
            for team in result.final_four:
                final_four_counts[team] = final_four_counts.get(team, 0) + 1
            
            # Elite Eight
            for team in result.elite_eight:
                elite_eight_counts[team] = elite_eight_counts.get(team, 0) + 1
            
            # Sweet Sixteen
            for team in result.sweet_sixteen:
                sweet_sixteen_counts[team] = sweet_sixteen_counts.get(team, 0) + 1
            
            # Round of 32
            for team in result.round_of_32:
                round_of_32_counts[team] = round_of_32_counts.get(team, 0) + 1
        
        return AggregatedResults(
            championship_odds={k: v/n for k, v in championship_counts.items()},
            final_four_odds={k: v/n for k, v in final_four_counts.items()},
            elite_eight_odds={k: v/n for k, v in elite_eight_counts.items()},
            sweet_sixteen_odds={k: v/n for k, v in sweet_sixteen_counts.items()},
            round_of_32_odds={k: v/n for k, v in round_of_32_counts.items()},
            num_simulations=n,
        )


@dataclass
class TournamentTeam:
    """Team in tournament bracket."""
    team_id: str
    seed: int
    region: str
    strength: float = 0.5


@dataclass
class TournamentBracket:
    """Tournament bracket structure."""
    
    teams: List[TournamentTeam]
    first_round_matchups: List[str]  # Ordered team_ids for first round
    
    @classmethod
    def create_standard_bracket(cls, teams_by_region: Dict[str, List[TournamentTeam]]) -> "TournamentBracket":
        """
        Create standard NCAA tournament bracket.
        
        Args:
            teams_by_region: Dict mapping region -> list of teams ordered by seed
            
        Returns:
            TournamentBracket
        """
        all_teams = []
        first_round = []
        
        # Standard seed matchup order for a region
        seed_order = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
        
        for region in ["East", "West", "South", "Midwest"]:
            region_teams = teams_by_region.get(region, [])
            seed_to_team = {t.seed: t for t in region_teams}
            
            all_teams.extend(region_teams)
            
            for high_seed, low_seed in seed_order:
                if high_seed in seed_to_team:
                    first_round.append(seed_to_team[high_seed].team_id)
                if low_seed in seed_to_team:
                    first_round.append(seed_to_team[low_seed].team_id)
        
        return cls(teams=all_teams, first_round_matchups=first_round)


def run_simulation(
    predict_fn: Callable[[str, str], float],
    teams_by_region: Dict[str, List[TournamentTeam]],
    num_simulations: int = 50000,
    show_progress: bool = True
) -> AggregatedResults:
    """
    Convenience function to run Monte Carlo simulation.
    
    Args:
        predict_fn: Prediction function
        teams_by_region: Teams organized by region
        num_simulations: Number of simulations
        show_progress: Show progress bar
        
    Returns:
        Aggregated results
    """
    bracket = TournamentBracket.create_standard_bracket(teams_by_region)
    
    config = SimulationConfig(num_simulations=num_simulations)
    engine = MonteCarloEngine(predict_fn, config)
    
    return engine.simulate_tournament(bracket, show_progress)
