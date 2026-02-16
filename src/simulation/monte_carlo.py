"""
Monte Carlo simulation engine for bracket prediction.

Runs 50,000+ simulations of the full tournament with noise injection
to model uncertainty (injuries, variance, etc.).
"""

from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import multiprocessing


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
            public = public_odds.get(team_id, 0.001)
            leverage = model_odds / public

            if leverage >= min_leverage:
                leverage_picks.append((team_id, model_odds, public, leverage))

        return sorted(leverage_picks, key=lambda x: x[3], reverse=True)


def _run_batch(
    batch_size: int,
    seed: int,
    first_round_matchups: List[str],
    team_data: Dict[str, Tuple[int, str, float]],
    matchup_probs: Dict[Tuple[str, str], float],
    noise_std: float,
    injury_probability: float,
) -> List[Dict]:
    """
    Run a batch of simulations in a subprocess.

    This is a module-level function so it can be pickled for
    ProcessPoolExecutor.

    Args:
        batch_size: Number of simulations to run
        seed: Random seed for this batch
        first_round_matchups: Ordered team IDs for first round
        team_data: team_id -> (seed, region, base_strength)
        matchup_probs: Pre-computed base matchup probabilities
        noise_std: Noise standard deviation
        injury_probability: Per-game injury probability

    Returns:
        List of result dictionaries
    """
    rng = np.random.RandomState(seed)
    results = []

    for _ in range(batch_size):
        # Initialize team states with noise
        team_strengths = {}
        for team_id, (tseed, region, strength) in team_data.items():
            noise = rng.normal(0, noise_std)
            injury = rng.random() < injury_probability
            injury_impact = 0.15 if injury else 0.0
            team_strengths[team_id] = strength + noise - (strength * injury_impact)

        current_teams = list(first_round_matchups)
        round_results = []

        while len(current_teams) > 1:
            round_winners = []
            for i in range(0, len(current_teams), 2):
                if i + 1 < len(current_teams):
                    team1 = current_teams[i]
                    team2 = current_teams[i + 1]

                    key = (team1, team2)
                    base_prob = matchup_probs.get(key, 0.5)

                    # Adjust for noise-modified strengths
                    s1 = team_strengths.get(team1, 0.5)
                    s2 = team_strengths.get(team2, 0.5)
                    strength_ratio = s1 / max(abs(s2), 0.01)
                    injury_adj = (strength_ratio - 1.0) * 0.1

                    game_noise = rng.normal(0, noise_std)
                    final_prob = np.clip(base_prob + injury_adj + game_noise, 0.01, 0.99)

                    if rng.random() < final_prob:
                        round_winners.append(team1)
                    else:
                        round_winners.append(team2)
                else:
                    round_winners.append(current_teams[i])

            round_results.append(round_winners)
            current_teams = round_winners

        champion = current_teams[0] if current_teams else None
        round_of_32 = round_results[0] if len(round_results) >= 1 else []
        sweet_sixteen = round_results[1] if len(round_results) >= 2 else []
        elite_eight = round_results[2] if len(round_results) >= 3 else []
        final_four = round_results[3] if len(round_results) >= 4 else []

        results.append({
            "champion": champion,
            "final_four": final_four,
            "elite_eight": elite_eight,
            "sweet_sixteen": sweet_sixteen,
            "round_of_32": round_of_32,
        })

    return results


class MonteCarloEngine:
    """
    Monte Carlo simulation engine for tournament brackets.

    Features:
    - True parallel simulation via ProcessPoolExecutor
    - Noise injection for uncertainty modeling
    - Injury scenario simulation
    - Pre-computed matchup probability cache
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
        Run full Monte Carlo simulation with true parallelism.

        Pre-computes all matchup probabilities, then distributes batches
        across ProcessPoolExecutor workers for CPU-parallel execution.

        Args:
            bracket: Tournament bracket structure
            show_progress: Show progress bar

        Returns:
            Aggregated simulation results
        """
        # Pre-compute all possible matchup probabilities so the predict_fn
        # (which may not be picklable) doesn't need to cross process boundaries.
        team_ids = [t.team_id for t in bracket.teams]
        matchup_probs: Dict[Tuple[str, str], float] = {}
        for i, t1 in enumerate(team_ids):
            for j, t2 in enumerate(team_ids):
                if i < j:
                    p = self.predict_fn(t1, t2)
                    matchup_probs[(t1, t2)] = p
                    matchup_probs[(t2, t1)] = 1.0 - p

        team_data = {
            t.team_id: (t.seed, t.region, t.strength) for t in bracket.teams
        }

        num_sims = self.config.num_simulations
        batch_size = self.config.batch_size
        n_workers = self.config.parallel_workers
        base_seed = self.config.random_seed or 42

        # Split into batches with distinct seeds
        batches = []
        remaining = num_sims
        batch_idx = 0
        while remaining > 0:
            bs = min(batch_size, remaining)
            batches.append((bs, base_seed + batch_idx * 1000))
            remaining -= bs
            batch_idx += 1

        all_raw_results: List[Dict] = []

        if n_workers > 1 and len(batches) > 1:
            try:
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = []
                    for bs, seed in batches:
                        future = executor.submit(
                            _run_batch,
                            bs, seed,
                            bracket.first_round_matchups,
                            team_data, matchup_probs,
                            self.config.noise_std,
                            self.config.injury_probability,
                        )
                        futures.append(future)

                    for future in as_completed(futures):
                        all_raw_results.extend(future.result())
            except (RuntimeError, OSError):
                # Fallback to sequential if multiprocessing fails
                for bs, seed in batches:
                    batch_results = _run_batch(
                        bs, seed, bracket.first_round_matchups,
                        team_data, matchup_probs,
                        self.config.noise_std, self.config.injury_probability,
                    )
                    all_raw_results.extend(batch_results)
        else:
            for bs, seed in batches:
                batch_results = _run_batch(
                    bs, seed, bracket.first_round_matchups,
                    team_data, matchup_probs,
                    self.config.noise_std, self.config.injury_probability,
                )
                all_raw_results.extend(batch_results)

        return self._aggregate_raw_results(all_raw_results)

    def _aggregate_raw_results(
        self,
        results: List[Dict]
    ) -> AggregatedResults:
        """Aggregate raw simulation result dictionaries."""
        n = len(results)

        championship_counts: Dict[str, int] = {}
        final_four_counts: Dict[str, int] = {}
        elite_eight_counts: Dict[str, int] = {}
        sweet_sixteen_counts: Dict[str, int] = {}
        round_of_32_counts: Dict[str, int] = {}

        for result in results:
            champion = result.get("champion")
            if champion:
                championship_counts[champion] = championship_counts.get(champion, 0) + 1

            for team in result.get("final_four", []):
                final_four_counts[team] = final_four_counts.get(team, 0) + 1

            for team in result.get("elite_eight", []):
                elite_eight_counts[team] = elite_eight_counts.get(team, 0) + 1

            for team in result.get("sweet_sixteen", []):
                sweet_sixteen_counts[team] = sweet_sixteen_counts.get(team, 0) + 1

            for team in result.get("round_of_32", []):
                round_of_32_counts[team] = round_of_32_counts.get(team, 0) + 1

        return AggregatedResults(
            championship_odds={k: v / n for k, v in championship_counts.items()},
            final_four_odds={k: v / n for k, v in final_four_counts.items()},
            elite_eight_odds={k: v / n for k, v in elite_eight_counts.items()},
            sweet_sixteen_odds={k: v / n for k, v in sweet_sixteen_counts.items()},
            round_of_32_odds={k: v / n for k, v in round_of_32_counts.items()},
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
