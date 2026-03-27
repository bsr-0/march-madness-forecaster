"""
Unified backtesting framework across Kaggle and ESPN pool strategies.

Evaluates both calibration mode (Kaggle round-weighted Brier score) and
EV mode (pool rank percentile) on historical tournament data, providing
a single report that answers: "How would each strategy have performed
in 2018, 2019, 2021-2025?"

Architecture
------------
The backtester operates in three layers:

1. **Data layer** — :class:`TournamentHistory` loads actual tournament
   results for a given year (seed matchups, round labels, outcomes).
   It supports both Kaggle CSV directories and the pipeline's own JSON
   historical data.

2. **Evaluation layer** — Two mode-specific evaluators:
   - *Calibration*: Scores probability predictions via weighted Brier.
     Optionally evaluates the champion-boost hedge from
     :mod:`dual_submission` to measure how much the 0-1 trick improves
     P(top finish) against known leaderboard thresholds.
   - *EV (pool)*: Builds a model bracket and opponent brackets (via
     :mod:`competitor_archetypes`), scores all brackets against the
     actual 63-game outcome, and reports our rank in the pool.

3. **Aggregation layer** — :class:`UnifiedBacktestResult` collects all
   per-year, per-mode results and generates a summary report with
   multi-year statistics, trend analysis, and mode comparison.

Usage::

    backtester = UnifiedBacktester(predict_fn_factory, kaggle_dir="data/kaggle")
    config = UnifiedBacktestConfig(
        years=[2023, 2024, 2025],
        modes=["calibration", "ev"],
        pool_sizes=[100, 500],
    )
    result = backtester.run_backtest(config)
    print(result.summary())
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ...data.normalize import normalize_team_id
from .kaggle_backtest import (
    KaggleBacktester,
    YearBacktestResult,
    _get_kaggle_rank,
)

logger = logging.getLogger(__name__)

# Historical LOYO years (excludes 2020 — tournament cancelled due to COVID)
LOYO_YEARS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]

# Standard NCAA bracket seed pairings per region (R64)
_SEED_MATCHUP_ORDER = [
    (1, 16), (8, 9), (5, 12), (4, 13),
    (6, 11), (3, 14), (7, 10), (2, 15),
]

# Kaggle round-weight structure (2023+ Brier scoring)
_KAGGLE_ROUND_WEIGHTS = {
    "FF": 0.5, "R64": 1.0, "R32": 2.0, "S16": 4.0,
    "E8": 8.0, "F4": 16.0, "NCG": 32.0,
}

# ESPN standard scoring for bracket pools
_ESPN_SCORING = {
    "R64": 10, "R32": 20, "S16": 40,
    "E8": 80, "F4": 160, "CHAMP": 320,
}

_BACKTEST_TEAM_ID_ALIAS = {
    "alabama_st": "alabama_state",
    "st_francis_pa": "saint_francis_pa",
    # normalize_team_id maps "saint_francis_pa" -> "saint_francis" via _QUICK_ALIAS;
    # re-map here so seed lookups stay consistent with game-result canonicalization.
    "saint_francis": "saint_francis_pa",
    "san_diego_st": "san_diego_state",
    "mt_st_mary_s": "mount_st_mary_s",
    "american_univ": "american",
    "ne_omaha": "omaha",
    "siue": "siu_edwardsville",
    "st_john_s__ny": "st_john_s_ny",
    "mississippi_st": "mississippi_state",
    "colorado_st": "colorado_state",
    "iowa_st": "iowa_state",
}


def _canonical_backtest_team_id(team_id: str) -> str:
    norm = normalize_team_id(team_id)
    return _BACKTEST_TEAM_ID_ALIAS.get(norm, norm)


# ---------------------------------------------------------------------------
# Historical tournament data loader
# ---------------------------------------------------------------------------

@dataclass
class TournamentGame:
    """A single tournament game with actual outcome."""
    team1_id: str
    team2_id: str
    team1_seed: int
    team2_seed: int
    team1_won: bool
    round_name: str          # R64, R32, S16, E8, F4, NCG
    team1_score: int = 0
    team2_score: int = 0

    @property
    def winner_id(self) -> str:
        return self.team1_id if self.team1_won else self.team2_id

    @property
    def loser_id(self) -> str:
        return self.team2_id if self.team1_won else self.team1_id

    def to_eval_dict(self) -> Dict:
        """Convert to format expected by KaggleBacktester.evaluate_predictions."""
        return {
            "team1": self.team1_id,
            "team2": self.team2_id,
            "team1_won": self.team1_won,
            "team1_seed": self.team1_seed,
            "team2_seed": self.team2_seed,
            "round": self.round_name,
        }


@dataclass
class TournamentHistory:
    """Complete tournament outcome for a single year.

    Provides the ground-truth data needed for both Kaggle-mode and
    EV-mode backtesting:
    - ``games``: All 63 tournament games with seeds, round labels, outcomes.
    - ``seeds``: team_id -> seed mapping for all 64 teams.
    - ``regions``: team_id -> region (East, West, South, Midwest).
    - ``first_round_matchups``: 64 team IDs in standard bracket order.
    - ``champion_id``: The actual champion.
    """
    year: int
    games: List[TournamentGame]
    seeds: Dict[str, int]
    regions: Dict[str, str]
    first_round_matchups: List[str]  # 64 team IDs in bracket order
    champion_id: str = ""

    @property
    def n_games(self) -> int:
        return len(self.games)

    def get_actual_bracket_bool(self) -> np.ndarray:
        """Build the 63-element boolean array of actual outcomes.

        True = first-listed team in each game slot won.  Game ordering
        follows standard bracket traversal (32 R64, 16 R32, ..., 1 NCG).
        """
        outcome = np.zeros(63, dtype=bool)
        current_teams = list(self.first_round_matchups)
        game_idx = 0

        for round_idx in range(6):
            next_round_teams = []
            for g in range(0, len(current_teams), 2):
                if g + 1 >= len(current_teams):
                    next_round_teams.append(current_teams[g])
                    continue
                t1, t2 = current_teams[g], current_teams[g + 1]
                # Determine who actually won
                winner = self._find_winner(t1, t2)
                outcome[game_idx] = (winner == t1)
                next_round_teams.append(winner)
                game_idx += 1
            current_teams = next_round_teams

        return outcome

    def _find_winner(self, t1: str, t2: str) -> str:
        """Find the actual winner of a matchup."""
        for game in self.games:
            if (game.team1_id == t1 and game.team2_id == t2):
                return game.winner_id
            if (game.team1_id == t2 and game.team2_id == t1):
                return game.winner_id
        # Fallback: lower seed wins
        s1 = self.seeds.get(t1, 8)
        s2 = self.seeds.get(t2, 8)
        return t1 if s1 <= s2 else t2


def load_tournament_history_from_kaggle(
    kaggle_dir: str,
    year: int,
) -> Optional[TournamentHistory]:
    """Load tournament history from Kaggle competition CSV files.

    Reads MNCAATourneyCompactResults, MNCAATourneySeeds, and MTeams
    to reconstruct the full tournament bracket and outcomes.

    Args:
        kaggle_dir: Path to directory containing Kaggle CSV files.
        year: Tournament year to load.

    Returns:
        TournamentHistory or None if data is unavailable.
    """
    try:
        from ...data.kaggle_loader import KaggleDataLoader
    except ImportError:
        logger.warning("KaggleDataLoader not available")
        return None

    loader = KaggleDataLoader(kaggle_dir)

    # Load tournament seeds
    seed_entries = loader.load_tourney_seeds(year)
    if not seed_entries:
        logger.warning("No tournament seeds found for %d", year)
        return None

    seeds: Dict[str, int] = {}
    regions: Dict[str, str] = {}
    region_map = {"W": "West", "X": "East", "Y": "South", "Z": "Midwest"}
    for entry in seed_entries:
        tid = _canonical_backtest_team_id(entry["team_id"])
        seeds[tid] = entry["seed"]
        raw_region = entry.get("region", "")
        regions[tid] = region_map.get(raw_region, raw_region)

    # Load tournament results
    tourney_games = loader.load_tourney_compact(year)
    if not tourney_games:
        logger.warning("No tournament games found for %d", year)
        return None

    # Deduplicate (loader returns mirror rows)
    seen_game_ids = set()
    unique_games = []
    for g in tourney_games:
        gid = g["game_id"]
        if gid in seen_game_ids:
            continue
        seen_game_ids.add(gid)
        # Only keep games where team_score > opponent_score (winner row)
        if g["team_score"] > g["opponent_score"]:
            unique_games.append(g)

    # Assign round labels by day_num ordering
    day_nums = sorted(set(g["day_num"] for g in unique_games))
    round_labels = ["R64", "R64", "R32", "R32", "S16", "E8", "F4", "NCG"]
    day_to_round: Dict[int, str] = {}
    for i, day in enumerate(day_nums):
        if i < len(round_labels):
            day_to_round[day] = round_labels[i]
        else:
            day_to_round[day] = "NCG"

    games: List[TournamentGame] = []
    for g in unique_games:
        t1 = _canonical_backtest_team_id(g["team_id"])
        t2 = _canonical_backtest_team_id(g["opponent_id"])
        round_name = day_to_round.get(g["day_num"], "R64")
        games.append(TournamentGame(
            team1_id=t1,
            team2_id=t2,
            team1_seed=seeds.get(t1, 8),
            team2_seed=seeds.get(t2, 8),
            team1_won=True,  # Winner row always has team_score > opponent_score
            round_name=round_name,
            team1_score=g["team_score"],
            team2_score=g["opponent_score"],
        ))

    if len(games) < 32:
        logger.warning(
            "Only %d tournament games found for %d (expected 63+)",
            len(games), year,
        )

    # Build first_round_matchups in standard bracket order
    first_round_matchups = _build_first_round_matchups(seeds, regions)

    # Identify champion (winner of last game by day_num)
    champion_id = ""
    if unique_games:
        last_game = max(unique_games, key=lambda g: g["day_num"])
        champion_id = _canonical_backtest_team_id(last_game["team_id"])

    history = TournamentHistory(
        year=year,
        games=games,
        seeds=seeds,
        regions=regions,
        first_round_matchups=first_round_matchups,
        champion_id=champion_id,
    )

    logger.info(
        "Loaded tournament history for %d: %d games, %d seeded teams, champion=%s",
        year, len(games), len(seeds), champion_id,
    )
    return history


def load_tournament_history_from_json(
    history_dir: str,
    year: int,
) -> Optional[TournamentHistory]:
    """Load tournament history from pipeline JSON files.

    Falls back from Kaggle CSVs when they are not available but we have
    the pipeline's own historical JSON caches.

    Args:
        history_dir: Path to directory containing tournament JSON files.
        year: Tournament year to load.

    Returns:
        TournamentHistory or None if data is unavailable.
    """
    import json

    seeds_path = Path(history_dir) / f"tournament_seeds_{year}.json"
    if not seeds_path.exists():
        return None

    try:
        with open(seeds_path) as f:
            seeds_data = json.load(f)
    except Exception:
        return None

    teams = seeds_data.get("teams", [])
    if not teams:
        return None

    seeds: Dict[str, int] = {}
    regions: Dict[str, str] = {}
    for t in teams:
        tid = _canonical_backtest_team_id(t.get("team_id", ""))
        seeds[tid] = t.get("seed", 8)
        regions[tid] = t.get("region", "")

    first_round_matchups = _build_first_round_matchups(seeds, regions)

    # Load actual game results from tournament_results_{year}.json
    games: List[TournamentGame] = []
    champion_id = ""
    results_path = Path(history_dir) / f"tournament_results_{year}.json"
    if results_path.exists():
        try:
            with open(results_path) as f:
                results_data = json.load(f)
            raw_games = results_data if isinstance(results_data, list) else results_data.get("games", [])
            for g in raw_games:
                t1 = _canonical_backtest_team_id(g["team1_id"])
                t2 = _canonical_backtest_team_id(g["team2_id"])
                games.append(TournamentGame(
                    team1_id=t1,
                    team2_id=t2,
                    team1_seed=g.get("team1_seed", seeds.get(t1, 8)),
                    team2_seed=g.get("team2_seed", seeds.get(t2, 8)),
                    team1_won=g["team1_won"],
                    round_name=g.get("round_name", "R64"),
                    team1_score=g.get("team1_score", 0),
                    team2_score=g.get("team2_score", 0),
                ))
            # Identify champion from NCG
            ncg = [g for g in games if g.round_name == "NCG"]
            if ncg:
                champion_id = ncg[0].winner_id
            logger.info(
                "Loaded %d tournament games from %s (champion=%s)",
                len(games), results_path, champion_id,
            )
        except Exception as e:
            logger.warning("Failed to load tournament results for %d: %s", year, e)

    return TournamentHistory(
        year=year,
        games=games,
        seeds=seeds,
        regions=regions,
        first_round_matchups=first_round_matchups,
        champion_id=champion_id,
    )


def _build_first_round_matchups(
    seeds: Dict[str, int],
    regions: Dict[str, str],
) -> List[str]:
    """Build ordered 64-team first-round matchup list.

    Standard NCAA bracket order: East, West, South, Midwest,
    each region ordered by seed pairing:
    (1v16), (8v9), (5v12), (4v13), (6v11), (3v14), (7v10), (2v15).
    """
    matchups: List[str] = []
    region_order = ["East", "West", "South", "Midwest"]

    # Group teams by region
    teams_by_region: Dict[str, Dict[int, str]] = {}
    for tid, seed in seeds.items():
        region = regions.get(tid, "")
        if region not in teams_by_region:
            teams_by_region[region] = {}
        teams_by_region[region][seed] = tid

    for region in region_order:
        region_teams = teams_by_region.get(region, {})
        for high_seed, low_seed in _SEED_MATCHUP_ORDER:
            t_high = region_teams.get(high_seed, f"unknown_{region}_{high_seed}")
            t_low = region_teams.get(low_seed, f"unknown_{region}_{low_seed}")
            matchups.extend([t_high, t_low])

    return matchups


# ---------------------------------------------------------------------------
# Champion-boost backtest helper
# ---------------------------------------------------------------------------

@dataclass
class ChampionBoostBacktestResult:
    """Evaluation of the champion boost (0-1 trick) against actual outcome.

    For each year, we record which champion the strategy selected, whether
    that pick was correct, and the Brier delta between the boosted and
    baseline submissions.
    """
    year: int
    selected_champion: str
    selected_champion_seed: int
    actual_champion: str
    champion_correct: bool
    baseline_rw_brier: float      # Round-weighted Brier without boost
    boosted_rw_brier: float       # Round-weighted Brier with champion boost
    brier_delta: float            # boosted - baseline (negative = boost helped)
    n_games_boosted: int = 0


def _evaluate_champion_boost(
    predictions: Dict[Tuple[str, str], float],
    games: List[TournamentGame],
    seeds: Dict[str, int],
    year: int,
) -> Optional[ChampionBoostBacktestResult]:
    """Evaluate champion boost strategy against actual tournament outcome.

    This back-computes what would have happened if we applied the 0-1 trick
    on the top champion candidate, using the actual tournament results as
    ground truth.

    Args:
        predictions: Baseline (team1, team2) -> P(team1 wins).
        games: Actual tournament games.
        seeds: team_id -> seed.
        year: Tournament year.

    Returns:
        ChampionBoostBacktestResult or None if insufficient data.
    """
    try:
        from ...optimization.dual_submission import ChampionBoostStrategy
    except ImportError:
        return None

    if not predictions or not games:
        return None

    # Guard against API mismatches between the strategy class versions.
    import inspect
    init_params = inspect.signature(ChampionBoostStrategy.__init__).parameters
    if "predict_fn" not in init_params:
        logger.debug("ChampionBoostStrategy API does not accept predict_fn; skipping boost eval.")
        return None

    # Find actual champion
    actual_champion = ""
    for g in games:
        if g.round_name in ("NCG", "Championship", "Finals"):
            actual_champion = g.winner_id
            break
    if not actual_champion:
        ncg_round = [g for g in games if g.round_name == "NCG"]
        if ncg_round:
            actual_champion = ncg_round[0].winner_id

    # Build predict_fn from the predictions dict
    def predict_fn(t1: str, t2: str) -> float:
        p = predictions.get((t1, t2))
        if p is not None:
            return p
        p = predictions.get((t2, t1))
        if p is not None:
            return 1.0 - p
        return 0.5

    # Run champion selection
    strategy = ChampionBoostStrategy(
        predict_fn=predict_fn,
        team_seeds=seeds,
    )
    champion_id = strategy.select_champion()
    if not champion_id:
        return None

    champion_seed = seeds.get(champion_id, 8)
    champion_correct = (champion_id == actual_champion)

    # Generate boosted predictions
    # Build matchup list from predictions keys
    matchup_ids = []
    for (t1, t2) in predictions:
        matchup_ids.append((f"{year}_{t1}_{t2}", t1, t2))

    boost_probs = strategy.generate_champion_boost(
        matchup_ids=matchup_ids,
        champion_id=champion_id,
    )

    # Score both baseline and boosted against actuals
    baseline_rw = _compute_round_weighted_brier(predictions, games)
    boosted_preds = dict(predictions)
    n_boosted = 0
    for mid, boosted_p in boost_probs.items():
        # Parse matchup_id to get team pair
        parts = mid.split("_")
        if len(parts) >= 3:
            t1 = "_".join(parts[1:-1])  # Handle team IDs with underscores
            t2 = parts[-1]
            # Try simpler parsing
        for (t1, t2), orig_p in list(predictions.items()):
            mid_check = f"{year}_{t1}_{t2}"
            if mid_check in boost_probs:
                new_p = boost_probs[mid_check]
                if abs(new_p - orig_p) > 0.01:
                    boosted_preds[(t1, t2)] = new_p
                    n_boosted += 1

    boosted_rw = _compute_round_weighted_brier(boosted_preds, games)

    return ChampionBoostBacktestResult(
        year=year,
        selected_champion=champion_id,
        selected_champion_seed=champion_seed,
        actual_champion=actual_champion,
        champion_correct=champion_correct,
        baseline_rw_brier=baseline_rw,
        boosted_rw_brier=boosted_rw,
        brier_delta=boosted_rw - baseline_rw,
        n_games_boosted=n_boosted,
    )


def _compute_round_weighted_brier(
    predictions: Dict[Tuple[str, str], float],
    games: List[TournamentGame],
) -> float:
    """Compute Kaggle round-weighted Brier score."""
    weighted_se_sum = 0.0
    weight_sum = 0.0

    for game in games:
        t1, t2 = game.team1_id, game.team2_id
        outcome = 1.0 if game.team1_won else 0.0

        pred = predictions.get((t1, t2))
        if pred is None:
            p2 = predictions.get((t2, t1))
            pred = (1.0 - p2) if p2 is not None else 0.5

        pred = max(0.001, min(0.999, pred))
        se = (pred - outcome) ** 2
        rw = _KAGGLE_ROUND_WEIGHTS.get(game.round_name, 1.0)
        weighted_se_sum += rw * se
        weight_sum += rw

    return weighted_se_sum / weight_sum if weight_sum > 0 else 0.25


# ---------------------------------------------------------------------------
# Pool backtest helper
# ---------------------------------------------------------------------------

def _score_bracket_against_actual(
    bracket_winners: List[str],
    actual_outcome: np.ndarray,
    first_round_matchups: List[str],
    scoring_system: Dict[str, int],
    actual_history: TournamentHistory,
) -> float:
    """Score a bracket (list of 63 winners) against actual outcome.

    Converts the bracket's winner list to boolean representation, then
    computes the ESPN-style bracket score.

    Args:
        bracket_winners: 63 team IDs in standard game order.
        actual_outcome: 63-element boolean array of actual results.
        first_round_matchups: 64 team IDs in bracket order.
        scoring_system: Round -> points mapping.

    Returns:
        Total bracket score (higher = better).
    """
    from ...simulation.pool_competition import (
        ROUND_NAMES,
        GAMES_PER_ROUND,
        build_scoring_vector,
    )

    scoring_vector = build_scoring_vector(scoring_system)

    # Convert bracket winners to boolean array
    bracket_bool = np.zeros(63, dtype=bool)
    current_teams = list(first_round_matchups)
    game_idx = 0
    winner_idx = 0

    for round_idx in range(6):
        next_round_teams = []
        for g in range(0, len(current_teams), 2):
            if g + 1 >= len(current_teams):
                next_round_teams.append(current_teams[g])
                continue
            t1, t2 = current_teams[g], current_teams[g + 1]
            if winner_idx < len(bracket_winners):
                winner = bracket_winners[winner_idx]
                bracket_bool[game_idx] = (winner == t1)
                next_round_teams.append(winner)
            else:
                bracket_bool[game_idx] = True
                next_round_teams.append(t1)
            game_idx += 1
            winner_idx += 1
        current_teams = next_round_teams

    # Score: count correct picks × round points
    correct = bracket_bool == actual_outcome
    return float(correct.astype(np.float64) @ scoring_vector)


def _generate_seed_based_bracket(
    first_round_matchups: List[str],
    seeds: Dict[str, int],
    rng: np.random.Generator,
    upset_propensity: float = 0.0,
) -> List[str]:
    """Generate a bracket by picking based on seed with optional upset noise.

    Args:
        first_round_matchups: 64 team IDs in bracket order.
        seeds: team_id -> seed.
        rng: Random generator.
        upset_propensity: 0.0 = always pick lower seed, 1.0 = random.

    Returns:
        List of 63 winner team IDs in standard game order.
    """
    winners = []
    current_teams = list(first_round_matchups)

    for round_idx in range(6):
        next_round_teams = []
        for g in range(0, len(current_teams), 2):
            if g + 1 >= len(current_teams):
                next_round_teams.append(current_teams[g])
                continue
            t1, t2 = current_teams[g], current_teams[g + 1]
            s1 = seeds.get(t1, 8)
            s2 = seeds.get(t2, 8)

            # Base probability: lower seed wins
            if s1 == s2:
                p_t1 = 0.5
            else:
                # Historical seed win rates (approximate)
                seed_diff = s2 - s1  # Positive if t1 is favored
                p_t1 = 1.0 / (1.0 + math.exp(-0.15 * seed_diff))

            # Blend toward 0.5 based on upset_propensity
            p_t1 = (1.0 - upset_propensity) * p_t1 + upset_propensity * 0.5

            if rng.random() < p_t1:
                winner = t1
            else:
                winner = t2

            winners.append(winner)
            next_round_teams.append(winner)
        current_teams = next_round_teams

    return winners


def _generate_model_bracket(
    first_round_matchups: List[str],
    predict_fn: Callable[[str, str], float],
    rng: np.random.Generator,
) -> List[str]:
    """Generate a bracket using the model's pairwise probabilities.

    For each matchup, sample the winner from the model's predicted
    probability.  This gives the most likely bracket (in expectation).

    Args:
        first_round_matchups: 64 team IDs in bracket order.
        predict_fn: (team1, team2) -> P(team1 wins).
        rng: Random generator.

    Returns:
        List of 63 winner team IDs in standard game order.
    """
    winners = []
    current_teams = list(first_round_matchups)

    for round_idx in range(6):
        next_round_teams = []
        for g in range(0, len(current_teams), 2):
            if g + 1 >= len(current_teams):
                next_round_teams.append(current_teams[g])
                continue
            t1, t2 = current_teams[g], current_teams[g + 1]
            p = predict_fn(t1, t2)
            winner = t1 if rng.random() < p else t2
            winners.append(winner)
            next_round_teams.append(winner)
        current_teams = next_round_teams

    return winners


# ---------------------------------------------------------------------------
# Archetype-aware opponent generation
# ---------------------------------------------------------------------------

def _generate_archetype_opponent_brackets(
    n_opponents: int,
    first_round_matchups: List[str],
    seeds: Dict[str, int],
    predict_fn: Callable[[str, str], float],
    rng: np.random.Generator,
    random_seed: int = 42,
    public_picks: Optional[Dict[str, Dict[str, float]]] = None,
) -> List[List[str]]:
    """Generate opponent brackets using the behavioral archetype engine.

    Replaces the crude upset-propensity heuristic with the full archetype
    model: ChalkPicker, ExpertMimic, ChaosSeeker, Homer — each filling
    brackets according to their behavioral patterns.

    The archetype mix defaults to ESPN national pool distribution.  Each
    opponent is assigned an archetype sampled from the mix, then that
    archetype's behavioral model determines pick probabilities for each
    matchup in the bracket.

    Args:
        n_opponents: Number of opponent brackets to generate.
        first_round_matchups: 64 team IDs in standard bracket order.
        seeds: team_id -> seed.
        predict_fn: (team1, team2) -> P(team1 wins) for model probs.
        rng: NumPy random generator.
        random_seed: Base seed for per-bracket RNG.
        public_picks: Historical public pick distribution
            (team_id -> {round_name: pick_pct}).  When provided, replaces
            the seed-based proxy for opponent archetype modeling.

    Returns:
        List of 63-element winner lists (one per opponent).
    """
    from ...simulation.competitor_archetypes import (
        get_archetype_mix,
        create_archetypes,
    )

    # Build model probs from seeds for the archetypes
    model_probs: Dict[str, Dict[str, float]] = {}
    for tid in set(first_round_matchups):
        # Build approximate round probabilities from seed
        seed = seeds.get(tid, 8)
        seed_strength = max(0.0, (17 - seed) / 16.0)
        model_probs[tid] = {
            "R64": min(0.99, seed_strength * 0.8 + 0.1),
            "R32": min(0.95, seed_strength * 0.6 + 0.05),
            "S16": min(0.9, seed_strength * 0.4 + 0.02),
            "E8": min(0.8, seed_strength * 0.25 + 0.01),
            "F4": min(0.6, seed_strength * 0.15),
            "CHAMP": min(0.4, seed_strength * 0.08),
        }

    # Use real historical public picks when available (Fix 4); fall back to
    # seed-based proxy when not.  Real picks give more realistic opponent
    # behavior because public chalk bias is team-specific, not just seed-based.
    if public_picks is None:
        public_picks = model_probs

    mix = get_archetype_mix("espn_national")
    archetypes = create_archetypes(mix)
    archetype_names = [a.params.name for a in archetypes]
    archetype_weights = [a.params.prevalence for a in archetypes]

    # Normalize
    total_w = sum(archetype_weights)
    if total_w > 0:
        archetype_weights = [w / total_w for w in archetype_weights]

    # Cumulative distribution for sampling
    cum_weights = []
    running = 0.0
    for w in archetype_weights:
        running += w
        cum_weights.append(running)

    opponent_brackets = []
    for i in range(n_opponents):
        # Sample an archetype
        r = rng.random()
        arch_idx = 0
        for j, cw in enumerate(cum_weights):
            if r <= cw:
                arch_idx = j
                break
        archetype = archetypes[arch_idx]

        # Generate bracket using archetype's behavioral model
        bracket_rng = np.random.RandomState(random_seed + i)
        winners = []
        current_teams = list(first_round_matchups)

        for round_idx in range(6):
            round_names = ["R64", "R32", "S16", "E8", "F4", "CHAMP"]
            round_name = round_names[min(round_idx, 5)]
            next_round_teams = []

            for g in range(0, len(current_teams), 2):
                if g + 1 >= len(current_teams):
                    next_round_teams.append(current_teams[g])
                    continue

                t1, t2 = current_teams[g], current_teams[g + 1]
                seed_t1 = seeds.get(t1, 8)

                m_prob = model_probs.get(t1, {}).get(round_name, 0.5)
                p_pct = public_picks.get(t1, {}).get(round_name, 0.5)

                pick_prob = archetype.predict_pick_probability(
                    t1, round_name, m_prob, p_pct, seed_t1,
                )
                pick_prob = max(0.01, min(0.99, pick_prob))

                if bracket_rng.rand() < pick_prob:
                    winner = t1
                else:
                    winner = t2

                winners.append(winner)
                next_round_teams.append(winner)

            current_teams = next_round_teams

        opponent_brackets.append(winners)

    return opponent_brackets


# ---------------------------------------------------------------------------
# ESPN Protocol Section 4.7 metric helpers (Fix 2)
# ---------------------------------------------------------------------------


def _compute_path_protection_for_bracket(
    winners: List[str],
    first_round_matchups: List[str],
    predict_fn: Callable[[str, str], float],
    team_regions: Dict[str, str],
    scoring: Dict[str, int],
) -> float:
    """Compute path protection score for a bracket (list of 63 winners).

    Converts the flat winner list to BracketPick objects, then delegates
    to PathProtectionScorer to compute the ratio of conditional EV to
    independent EV.  Target: > 0.85 (Protocol Section 4.4).

    Args:
        winners: 63 winner team IDs in standard game order.
        first_round_matchups: 64 team IDs in bracket order.
        predict_fn: (team1, team2) -> P(team1 wins).
        team_regions: team_id -> region name.
        scoring: Round -> points mapping.

    Returns:
        Path protection score in [0, 1].
    """
    from ...optimization.bracket_search import BracketPick
    from ...optimization.path_protection import PathProtectionScorer

    picks: List = []
    current_teams = list(first_round_matchups)
    winner_idx = 0
    game_idx = 0

    for round_num in range(6):
        next_teams: List[str] = []
        for g in range(0, len(current_teams), 2):
            if g + 1 >= len(current_teams):
                next_teams.append(current_teams[g])
                continue
            t1, t2 = current_teams[g], current_teams[g + 1]
            if winner_idx < len(winners):
                winner = winners[winner_idx]
                loser = t2 if winner == t1 else t1
            else:
                p = predict_fn(t1, t2)
                winner, loser = (t1, t2) if p >= 0.5 else (t2, t1)
            p_val = predict_fn(winner, loser)
            picks.append(BracketPick(
                round_num=round_num,
                game_idx=game_idx,
                winner_id=winner,
                loser_id=loser,
                win_probability=p_val,
            ))
            next_teams.append(winner)
            winner_idx += 1
            game_idx += 1
        current_teams = next_teams

    if not picks:
        return 1.0

    return PathProtectionScorer().compute_path_protection_score(
        picks, predict_fn, team_regions, scoring,
    )


def _compute_leverage_accuracy_for_year(
    first_round_matchups: List[str],
    predict_fn: Callable[[str, str], float],
    public_picks: Dict[str, Dict[str, float]],
    games: List,
    scoring: Dict[str, int],
    rng: np.random.Generator,
    n_sims: int = 150,
) -> Tuple[float, int]:
    """Estimate leverage accuracy for a historical backtest year.

    Approximates per-round model probabilities via a Monte Carlo bracket
    simulation (n_sims runs), then builds leverage picks as teams where
    model_prob exceeds public_pick_pct by > 5 pp.  Returns the hit rate
    among those qualifying picks against the actual tournament outcomes.

    Args:
        first_round_matchups: 64 team IDs in bracket order.
        predict_fn: (team1, team2) -> P(team1 wins).
        public_picks: team_id -> {round_name: pick_pct} from historical data.
        games: List of TournamentGame objects (actual results).
        scoring: Round -> points mapping.
        rng: NumPy generator for reproducibility.
        n_sims: Monte Carlo simulations for round-prob estimation.

    Returns:
        Tuple of (leverage_accuracy, n_qualifying_picks).
    """
    from ...optimization.leverage import LeverageCalculator, evaluate_leverage_accuracy

    round_names = ["R64", "R32", "S16", "E8", "F4", "CHAMP"]

    # Estimate round probabilities via Monte Carlo simulation of the bracket.
    # For each simulation, replay the bracket using predict_fn and track which
    # team advances to each round.
    advance_counts: Dict[str, Dict[str, int]] = {
        tid: {rn: 0 for rn in round_names}
        for tid in set(first_round_matchups)
    }

    for _ in range(n_sims):
        current = list(first_round_matchups)
        for round_idx, round_name in enumerate(round_names):
            nxt: List[str] = []
            for g in range(0, len(current), 2):
                if g + 1 >= len(current):
                    nxt.append(current[g])
                    continue
                t1, t2 = current[g], current[g + 1]
                p = predict_fn(t1, t2)
                winner = t1 if rng.random() < p else t2
                advance_counts[winner][round_name] += 1
                nxt.append(winner)
            current = nxt

    model_probs: Dict[str, Dict[str, float]] = {
        tid: {rn: advance_counts[tid][rn] / n_sims for rn in round_names}
        for tid in advance_counts
    }

    # Build team metadata for LeverageCalculator (seed not strictly required)
    from ...optimization.leverage import LeverageCalculator, TeamMetadata, evaluate_leverage_accuracy
    team_meta = {}  # minimal — leverage calc works without it

    calculator = LeverageCalculator(
        model_probs=model_probs,
        public_picks=public_picks,
        scoring_system=scoring,
    )
    leverage_picks = calculator.find_leverage_picks()

    # Build actual_outcomes: team_id -> {round_name: True/False}
    # True = team actually reached/won this round.
    # Map TournamentGame.round_name "NCG" to "CHAMP" for consistency.
    _round_alias = {"NCG": "CHAMP"}
    actual_winners: Dict[str, set] = {}
    for game in games:
        rn = _round_alias.get(game.round_name, game.round_name)
        wid = game.winner_id
        actual_winners.setdefault(wid, set()).add(rn)

    actual_outcomes: Dict[str, Dict[str, bool]] = {}
    for tid in set(first_round_matchups):
        won_rounds = actual_winners.get(tid, set())
        actual_outcomes[tid] = {rn: (rn in won_rounds) for rn in round_names}

    accuracy = evaluate_leverage_accuracy(leverage_picks, actual_outcomes)
    return accuracy, len([
        p for p in leverage_picks
        if p.model_probability - p.public_pick_percentage >= 0.05
    ])


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class UnifiedBacktestConfig:
    """Configuration for the unified backtester."""

    years: List[int] = field(
        default_factory=lambda: list(LOYO_YEARS),
    )
    modes: List[str] = field(
        default_factory=lambda: ["calibration", "ev"],
    )
    pool_sizes: List[int] = field(
        default_factory=lambda: [100, 500, 1000],
    )
    payout_structures: List[str] = field(
        default_factory=lambda: ["winner_take_all", "top_10pct"],
    )
    n_pool_simulations: int = 1000  # Archetype brackets per pool simulation
    kaggle_effective_pool_size: int = 3000
    # Whether to evaluate champion boost (0-1 trick) within calibration mode
    evaluate_champion_boost: bool = True
    # Number of model brackets to sample for EV mode (diverse bracket portfolio)
    n_model_brackets: int = 5
    random_seed: int = 42
    # ESPN Protocol Section 4.7 metrics
    compute_espn_metrics: bool = True
    historical_picks_dir: str = "data/raw/historical_public_picks"

    def __post_init__(self):
        valid_modes = {"calibration", "ev"}
        invalid = set(self.modes) - valid_modes
        if invalid:
            raise ValueError(f"Invalid modes: {invalid}. Must be in {valid_modes}")


# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------

@dataclass
class YearModeResult:
    """Result for one year under one strategy configuration."""

    year: int
    mode: str  # "calibration" or "ev"
    brier_score: float = 0.0
    round_weighted_brier: float = 0.0
    kaggle_rank_estimate: str = ""
    accuracy: float = 0.0
    n_games: int = 0
    # Champion boost evaluation (calibration mode)
    champion_boost: Optional[ChampionBoostBacktestResult] = None
    # Pool-mode specific fields
    pool_size: int = 0
    payout_structure: str = ""
    pool_rank_percentile: float = 0.0  # Where our bracket ranked (0.0 = 1st, 1.0 = last)
    pool_rank_position: int = 0  # Absolute rank in simulated pool
    pool_score: float = 0.0  # Our bracket's score
    opponent_mean_score: float = 0.0  # Mean opponent score
    # ESPN Protocol Section 4.7 metrics
    path_protection_score: float = 0.0  # Target: > 0.85 of unconditional P
    leverage_accuracy: float = 0.0  # Target: > 0.55 among leverage > +5%
    p_top_10_pct: float = 0.0  # Target: > 0.30
    n_leverage_picks: int = 0


@dataclass
class UnifiedBacktestResult:
    """Aggregate results from unified backtesting."""

    config: UnifiedBacktestConfig
    year_mode_results: List[YearModeResult] = field(default_factory=list)
    summary_by_mode: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate human-readable report across all years and modes."""
        lines = ["=" * 70, "UNIFIED BACKTEST REPORT", "=" * 70]

        # Group by mode
        calibration_results = [r for r in self.year_mode_results if r.mode == "calibration"]
        ev_results = [r for r in self.year_mode_results if r.mode == "ev"]

        if calibration_results:
            lines.append("\n--- CALIBRATION MODE (Kaggle) ---")
            for r in sorted(calibration_results, key=lambda x: x.year):
                line = (
                    f"  {r.year}: Brier={r.brier_score:.4f}  "
                    f"RW-Brier={r.round_weighted_brier:.4f}  "
                    f"Acc={r.accuracy:.1%}  "
                    f"[{r.kaggle_rank_estimate}]"
                )
                if r.champion_boost is not None:
                    cb = r.champion_boost
                    marker = "CORRECT" if cb.champion_correct else "wrong"
                    line += (
                        f"  | Champ={cb.selected_champion}({cb.selected_champion_seed})"
                        f" [{marker}] delta={cb.brier_delta:+.4f}"
                    )
                lines.append(line)

            briers = [r.brier_score for r in calibration_results]
            rw_briers = [r.round_weighted_brier for r in calibration_results]
            lines.append(
                f"  Mean Brier: {np.mean(briers):.4f} +/- {np.std(briers):.4f}"
            )
            lines.append(
                f"  Mean RW-Brier: {np.mean(rw_briers):.4f} +/- {np.std(rw_briers):.4f}"
            )

            # Year-over-year trend
            if "calibration" in self.summary_by_mode:
                trend = self.summary_by_mode["calibration"].get("yoy_brier_trend")
                if trend is not None:
                    direction = "improving" if trend < 0 else "worsening"
                    lines.append(
                        f"  YoY Brier trend: {trend:+.5f}/year ({direction})"
                    )

            # Champion boost summary
            boost_results = [
                r.champion_boost for r in calibration_results
                if r.champion_boost is not None
            ]
            if boost_results:
                n_correct = sum(1 for b in boost_results if b.champion_correct)
                mean_delta = np.mean([b.brier_delta for b in boost_results])
                lines.append(
                    f"  Champion Boost: {n_correct}/{len(boost_results)} correct, "
                    f"mean delta={mean_delta:+.4f}"
                )

        if ev_results:
            lines.append("\n--- EV MODE (Pool Strategy) ---")
            # Group by pool_size and payout
            configs_seen = set()
            for r in ev_results:
                key = (r.pool_size, r.payout_structure)
                if key not in configs_seen:
                    configs_seen.add(key)
                    subset = [
                        x for x in ev_results
                        if x.pool_size == r.pool_size
                        and x.payout_structure == r.payout_structure
                    ]
                    lines.append(
                        f"\n  Pool={r.pool_size}, Payout={r.payout_structure}:"
                    )
                    for s in sorted(subset, key=lambda x: x.year):
                        lines.append(
                            f"    {s.year}: Rank={s.pool_rank_position}/"
                            f"{s.pool_size}  Score={s.pool_score:.0f}  "
                            f"({s.pool_rank_percentile:.1%})"
                        )
                    pcts = [s.pool_rank_percentile for s in subset]
                    wins = sum(1 for s in subset if s.pool_rank_position == 1)
                    lines.append(
                        f"    Mean percentile: {np.mean(pcts):.1%}  "
                        f"Pool wins: {wins}/{len(subset)}"
                    )

        lines.append("=" * 70)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main backtester
# ---------------------------------------------------------------------------

class UnifiedBacktester:
    """Orchestrates backtesting across both Kaggle and pool strategies.

    For calibration mode:
      - Generates predictions for all 63 tournament games
      - Scores via round-weighted Brier against actual outcomes
      - Evaluates champion-boost hedge submission
      - Estimates Kaggle leaderboard rank

    For EV mode:
      - Generates model bracket(s) using predict_fn
      - Generates archetype opponent brackets
      - Scores all brackets against the actual 63-game outcome
      - Reports rank in simulated pool
    """

    def __init__(
        self,
        predict_fn_factory: Optional[Callable] = None,
        kaggle_dir: Optional[str] = None,
        historical_results_dir: str = "data/raw/historical_full",
    ):
        """
        Args:
            predict_fn_factory: Factory that takes (year: int) and returns
                a predict_fn: (team1_id, team2_id) -> float.
                If None, uses seed-based probability estimates.
            kaggle_dir: Path to Kaggle CSV directory (for loading historical
                tournament results).  Preferred data source.
            historical_results_dir: Fallback directory with pipeline JSON
                historical data.
        """
        self.predict_fn_factory = predict_fn_factory
        self.kaggle_dir = kaggle_dir
        self.historical_results_dir = historical_results_dir
        self._history_cache: Dict[int, TournamentHistory] = {}

    def load_history(self, year: int) -> Optional[TournamentHistory]:
        """Load tournament history for a year (with caching)."""
        if year in self._history_cache:
            return self._history_cache[year]

        history = None

        # Try Kaggle CSVs first
        if self.kaggle_dir:
            history = load_tournament_history_from_kaggle(self.kaggle_dir, year)

        # Fall back to JSON
        if history is None or not history.games:
            json_history = load_tournament_history_from_json(
                self.historical_results_dir, year,
            )
            if json_history is not None:
                if history is None:
                    history = json_history
                elif not history.games and json_history.games:
                    history = json_history

        if history is not None:
            self._history_cache[year] = history

        return history

    def run_backtest(self, config: UnifiedBacktestConfig) -> UnifiedBacktestResult:
        """Run unified backtest across all configured years and modes.

        Args:
            config: Backtest configuration specifying years, modes,
                pool sizes, and payout structures.

        Returns:
            UnifiedBacktestResult with per-year, per-mode results and
            aggregate summary statistics.
        """
        results: List[YearModeResult] = []
        skipped_years: List[int] = []

        for year in config.years:
            history = self.load_history(year)
            if history is None:
                logger.warning("No historical data for %d, skipping", year)
                skipped_years.append(year)
                continue

            # Build predict_fn for this year
            predict_fn = self._get_predict_fn(year, history)

            if "calibration" in config.modes:
                cal_result = self._run_calibration_year(
                    year, history, predict_fn, config,
                )
                if cal_result is not None:
                    results.append(cal_result)

            if "ev" in config.modes:
                for pool_size in config.pool_sizes:
                    for payout in config.payout_structures:
                        ev_result = self._run_ev_year(
                            year, history, predict_fn,
                            pool_size, payout, config,
                        )
                        if ev_result is not None:
                            results.append(ev_result)

        if skipped_years:
            logger.warning(
                "Backtest skipped %d of %d requested years due to missing "
                "historical data: %s",
                len(skipped_years), len(config.years), skipped_years,
            )
        if not results:
            logger.error(
                "Backtest produced ZERO results across all %d requested years. "
                "Output artifact will be empty. Check historical data availability.",
                len(config.years),
            )

        # Compute summary statistics
        summary = self._compute_summary(results)

        return UnifiedBacktestResult(
            config=config,
            year_mode_results=results,
            summary_by_mode=summary,
        )

    def _get_predict_fn(
        self,
        year: int,
        history: TournamentHistory,
    ) -> Callable[[str, str], float]:
        """Get prediction function for a given year.

        If predict_fn_factory is provided, uses that.
        Otherwise, falls back to seed-based logistic probability.
        """
        if self.predict_fn_factory is not None:
            try:
                return self.predict_fn_factory(year)
            except Exception as e:
                logger.exception(
                    "predict_fn_factory failed for %d: %s. Falling back to seed model.",
                    year, e,
                )

        # Seed-based logistic model as fallback
        seeds = history.seeds

        def seed_predict(t1: str, t2: str) -> float:
            s1 = seeds.get(t1, 8)
            s2 = seeds.get(t2, 8)
            if s1 == s2:
                return 0.5
            diff = s2 - s1  # Positive means t1 is favored (lower seed)
            return 1.0 / (1.0 + math.exp(-0.175 * diff))

        return seed_predict

    def _run_calibration_year(
        self,
        year: int,
        history: TournamentHistory,
        predict_fn: Callable[[str, str], float],
        config: UnifiedBacktestConfig,
    ) -> Optional[YearModeResult]:
        """Run calibration-mode backtest for a single year.

        Scores all 63 tournament matchup predictions against actual
        outcomes using round-weighted Brier.  Optionally evaluates
        the champion boost strategy.
        """
        if not history.games:
            logger.warning("No games for calibration backtest year %d", year)
            return None

        # Generate predictions for all actual tournament matchups
        predictions: Dict[Tuple[str, str], float] = {}
        seeds = history.seeds
        per_game_seed_fallbacks = 0
        for game in history.games:
            try:
                pred = predict_fn(game.team1_id, game.team2_id)
            except Exception as e:
                s1 = seeds.get(game.team1_id, game.team1_seed or 8)
                s2 = seeds.get(game.team2_id, game.team2_seed or 8)
                if s1 == s2:
                    pred = 0.5
                else:
                    diff = s2 - s1  # Positive means team1 is favored (lower seed)
                    pred = 1.0 / (1.0 + math.exp(-0.175 * diff))
                per_game_seed_fallbacks += 1
                logger.warning(
                    "Calibration backtest %d game-level fallback for %s vs %s: %s",
                    year,
                    game.team1_id,
                    game.team2_id,
                    e,
                )
            predictions[(game.team1_id, game.team2_id)] = pred
        if per_game_seed_fallbacks:
            logger.warning(
                "Calibration backtest %d used game-level seed fallback for %d/%d games.",
                year,
                per_game_seed_fallbacks,
                len(history.games),
            )

        # Evaluate using KaggleBacktester
        eval_games = [g.to_eval_dict() for g in history.games]
        kaggle_bt = KaggleBacktester()
        year_result = kaggle_bt.evaluate_predictions(predictions, eval_games, year)

        # Champion boost evaluation
        champion_boost = None
        if config.evaluate_champion_boost:
            champion_boost = _evaluate_champion_boost(
                predictions, history.games, history.seeds, year,
            )

        result = YearModeResult(
            year=year,
            mode="calibration",
            brier_score=year_result.brier_score,
            round_weighted_brier=year_result.round_weighted_brier,
            kaggle_rank_estimate=year_result.estimated_kaggle_rank,
            accuracy=year_result.accuracy,
            n_games=year_result.n_games,
            champion_boost=champion_boost,
        )

        logger.info(
            "Calibration backtest %d: Brier=%.4f, RW-Brier=%.4f, Acc=%.1f%% [%s]",
            year, result.brier_score, result.round_weighted_brier,
            result.accuracy * 100, result.kaggle_rank_estimate,
        )
        return result

    def _run_ev_year(
        self,
        year: int,
        history: TournamentHistory,
        predict_fn: Callable[[str, str], float],
        pool_size: int,
        payout_structure: str,
        config: UnifiedBacktestConfig,
    ) -> Optional[YearModeResult]:
        """Simulate pool competition for a historical year.

        1. Get model's bracket using predict_fn
        2. Generate N opponent brackets using seed-based archetypes
        3. Score ALL brackets against the actual 63-game outcome
        4. Report our rank in the simulated pool
        """
        if not history.games or not history.first_round_matchups:
            logger.warning("Insufficient data for EV backtest year %d", year)
            return None

        if len(history.first_round_matchups) < 64:
            logger.warning(
                "Need 64 first-round teams for EV backtest, got %d for year %d",
                len(history.first_round_matchups), year,
            )
            return None

        rng = np.random.default_rng(config.random_seed + year)

        # Get actual tournament outcome
        actual_outcome = history.get_actual_bracket_bool()

        # Build scoring system
        scoring = dict(_ESPN_SCORING)
        if payout_structure == "upset_bonus":
            # Increase points for upset-heavy scoring
            scoring = {k: v for k, v in scoring.items()}

        from ...simulation.pool_competition import build_scoring_vector
        scoring_vector = build_scoring_vector(scoring)

        # Generate model bracket(s)
        model_brackets = []
        for i in range(config.n_model_brackets):
            bracket = _generate_model_bracket(
                history.first_round_matchups,
                predict_fn,
                np.random.default_rng(config.random_seed + year * 100 + i),
            )
            model_brackets.append(bracket)

        # Load historical public picks (Fix 4): use real ESPN "Who Picked Whom"
        # data when available; fall back to seed-based approximation seamlessly.
        from ...data.historical_picks import load_historical_public_picks
        from pathlib import Path as _Path
        hist_picks_dir = _Path(config.historical_picks_dir)
        historical_public_picks = load_historical_public_picks(
            year=year,
            bracket_teams=history.seeds,
            picks_dir=hist_picks_dir,
        )

        # Generate opponent brackets using the archetype engine for realistic
        # field modeling.  Each archetype produces brackets with distinct
        # behavioral patterns (chalk, expert, chaos, homer) — much more
        # realistic than uniform upset-propensity noise.
        # Pass historical public picks so opponent archetypes reflect real
        # year-specific chalk bias rather than generic seed-based priors.
        n_opponents = max(1, pool_size - config.n_model_brackets)
        opponent_brackets = _generate_archetype_opponent_brackets(
            n_opponents=n_opponents,
            first_round_matchups=history.first_round_matchups,
            seeds=history.seeds,
            predict_fn=predict_fn,
            rng=rng,
            random_seed=config.random_seed + year * 10000,
            public_picks=historical_public_picks,
        )

        # Score all brackets against actual outcome
        model_scores = []
        for bracket in model_brackets:
            score = _score_bracket_against_actual(
                bracket, actual_outcome,
                history.first_round_matchups,
                scoring, history,
            )
            model_scores.append(score)

        opponent_scores = []
        for bracket in opponent_brackets:
            score = _score_bracket_against_actual(
                bracket, actual_outcome,
                history.first_round_matchups,
                scoring, history,
            )
            opponent_scores.append(score)

        # Best model bracket
        best_model_idx = int(np.argmax(model_scores))
        best_model_score = model_scores[best_model_idx]

        # Compute rank
        all_scores = model_scores + opponent_scores
        rank = sum(1 for s in all_scores if s > best_model_score) + 1
        total = len(all_scores)
        percentile = rank / total

        # ESPN Protocol Section 4.7 metrics (Fix 2): compute path_protection_score
        # and leverage_accuracy which were previously always left at 0.0.
        path_protection_score = 0.0
        leverage_accuracy = 0.0
        n_leverage_picks = 0
        if config.compute_espn_metrics:
            try:
                best_bracket_winners = model_brackets[best_model_idx]
                path_protection_score = _compute_path_protection_for_bracket(
                    winners=best_bracket_winners,
                    first_round_matchups=history.first_round_matchups,
                    predict_fn=predict_fn,
                    team_regions=history.regions,
                    scoring=scoring,
                )
                leverage_accuracy, n_leverage_picks = _compute_leverage_accuracy_for_year(
                    first_round_matchups=history.first_round_matchups,
                    predict_fn=predict_fn,
                    public_picks=historical_public_picks,
                    games=history.games,
                    scoring=scoring,
                    rng=np.random.default_rng(config.random_seed + year * 999),
                )
            except Exception as exc:
                logger.warning(
                    "ESPN metrics computation failed for year %d: %s", year, exc,
                )

        result = YearModeResult(
            year=year,
            mode="ev",
            pool_size=total,
            payout_structure=payout_structure,
            pool_rank_position=rank,
            pool_rank_percentile=percentile,
            pool_score=best_model_score,
            opponent_mean_score=float(np.mean(opponent_scores)),
            path_protection_score=path_protection_score,
            leverage_accuracy=leverage_accuracy,
            n_leverage_picks=n_leverage_picks,
        )

        logger.info(
            "EV backtest %d (pool=%d, %s): rank=%d/%d (%.1f%%), "
            "score=%.0f, opp_mean=%.0f, path_prot=%.3f, lev_acc=%.3f",
            year, total, payout_structure, rank, total,
            percentile * 100, best_model_score, result.opponent_mean_score,
            path_protection_score, leverage_accuracy,
        )
        return result

    def _compute_summary(
        self,
        results: List[YearModeResult],
    ) -> Dict[str, Dict[str, float]]:  # noqa: E301
        """Compute aggregate summary statistics per mode."""
        summary: Dict[str, Dict[str, float]] = {}

        cal_results = [r for r in results if r.mode == "calibration"]
        if cal_results:
            briers = [r.brier_score for r in cal_results]
            rw_briers = [r.round_weighted_brier for r in cal_results]
            summary["calibration"] = {
                "mean_brier": float(np.mean(briers)),
                "std_brier": float(np.std(briers)),
                "mean_rw_brier": float(np.mean(rw_briers)),
                "std_rw_brier": float(np.std(rw_briers)),
                "mean_accuracy": float(np.mean([r.accuracy for r in cal_results])),
                "n_years": float(len(cal_results)),
            }

            # Trend analysis: compute year-over-year Brier improvement
            trend = self._compute_trend(cal_results)
            if trend is not None:
                summary["calibration"]["yoy_brier_trend"] = trend

        ev_results = [r for r in results if r.mode == "ev"]
        if ev_results:
            pcts = [r.pool_rank_percentile for r in ev_results]
            wins = sum(1 for r in ev_results if r.pool_rank_position == 1)
            # ESPN Protocol Section 4.7 metrics
            path_scores = [r.path_protection_score for r in ev_results if r.path_protection_score > 0]
            lev_accs = [r.leverage_accuracy for r in ev_results if r.n_leverage_picks > 0]
            top10_rates = [r.p_top_10_pct for r in ev_results if r.p_top_10_pct > 0]
            n_folds = len(ev_results)
            se_pct = float(np.std(pcts) / max(1, n_folds ** 0.5)) if n_folds > 1 else 0.0

            summary["ev"] = {
                "mean_percentile": float(np.mean(pcts)),
                "std_percentile": float(np.std(pcts)),
                "se_percentile": se_pct,
                "n_wins": float(wins),
                "n_configs": float(len(ev_results)),
                "win_rate": float(wins / len(ev_results)) if ev_results else 0.0,
                # ESPN Protocol Section 4.7 compliance
                "mean_path_protection": float(np.mean(path_scores)) if path_scores else 0.0,
                "mean_leverage_accuracy": float(np.mean(lev_accs)) if lev_accs else 0.0,
                "mean_p_top_10_pct": float(np.mean(top10_rates)) if top10_rates else 0.0,
                # Protocol compliance flags
                "protocol_pool_rank_pass": float(np.mean(pcts)) <= 0.20,  # 80th %ile = top 20%
                "protocol_path_protection_pass": (float(np.mean(path_scores)) >= 0.85) if path_scores else False,
                "protocol_leverage_accuracy_pass": (float(np.mean(lev_accs)) >= 0.55) if lev_accs else False,
                "protocol_p_top_10_pass": (float(np.mean(top10_rates)) >= 0.30) if top10_rates else False,
            }

            # Per-payout-structure breakdown
            payout_breakdown = self._compute_payout_breakdown(ev_results)
            if payout_breakdown:
                summary["ev_by_payout"] = payout_breakdown

        return summary

    @staticmethod
    def _compute_trend(
        cal_results: List[YearModeResult],
    ) -> Optional[float]:
        """Compute year-over-year Brier trend via simple linear regression.

        A negative trend means the model is improving (lower Brier) over time.

        Returns:
            Slope of Brier score vs year, or None if insufficient data.
        """
        sorted_results = sorted(cal_results, key=lambda r: r.year)
        if len(sorted_results) < 3:
            return None

        years = np.array([r.year for r in sorted_results], dtype=float)
        briers = np.array([r.brier_score for r in sorted_results], dtype=float)

        # Simple OLS: slope = cov(x,y) / var(x)
        x_mean = years.mean()
        y_mean = briers.mean()
        numerator = ((years - x_mean) * (briers - y_mean)).sum()
        denominator = ((years - x_mean) ** 2).sum()
        if abs(denominator) < 1e-12:
            return None
        return float(numerator / denominator)

    @staticmethod
    def _compute_payout_breakdown(
        ev_results: List[YearModeResult],
    ) -> Dict[str, Dict[str, float]]:
        """Break down EV results by payout structure for strategy comparison.

        Returns:
            Dict mapping payout_structure -> {mean_percentile, win_rate, ...}
        """
        from collections import defaultdict

        by_payout: Dict[str, List[YearModeResult]] = defaultdict(list)
        for r in ev_results:
            by_payout[r.payout_structure].append(r)

        breakdown: Dict[str, Dict[str, float]] = {}
        for payout, results in by_payout.items():
            pcts = [r.pool_rank_percentile for r in results]
            wins = sum(1 for r in results if r.pool_rank_position == 1)
            breakdown[payout] = {
                "mean_percentile": float(np.mean(pcts)),
                "n_configs": float(len(results)),
                "win_rate": float(wins / len(results)) if results else 0.0,
                "best_percentile": float(min(pcts)) if pcts else 1.0,
                "worst_percentile": float(max(pcts)) if pcts else 1.0,
            }

        return breakdown
