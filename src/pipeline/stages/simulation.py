"""Simulation and inference helpers — extracted from SOTAPipeline.

Contains Monte Carlo simulation, round-probability conversion, public pick
loading, betting market integration, injury noise modeling, and model
confidence interval estimation.

Each function takes a ``pipeline`` parameter (SOTAPipeline instance)
to access config and mutable state.  This is a pragmatic extraction
that reduces sota.py line count while maintaining exact behavioral
equivalence.

Implements Agent Directive V7 S2 (modular architecture decomposition).
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np

from ...data.models.game_flow import GameFlow
from ...data.models.player import Roster
from ...models.team import Team
from ...simulation.monte_carlo import (
    SimulationConfig,
    TournamentBracket,
    TournamentTeam,
)
from ..config import (
    DataRequirementError,
    TOURNAMENT_START_DATES,
)
from .game_utils import (
    compute_game_sort_key as _gu_compute_game_sort_key,
    detect_tournament_game as _gu_detect_tournament_game,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------


def _resolve_play_in_teams(
    teams_by_region: Dict[str, List[TournamentTeam]],
) -> Dict[str, List[TournamentTeam]]:
    """Resolve First Four play-in matchups within each region.

    The NCAA tournament starts with 68 teams: 64 in the main bracket plus
    4 First Four play-in games. Each play-in game features two teams sharing
    the same seed in the same region.  This function detects those duplicate
    seeds and keeps only the stronger team (higher AdjEM), reducing each
    region to exactly 16 teams for the 63-game bracket simulation.
    """
    resolved: Dict[str, List[TournamentTeam]] = {}
    for region, region_teams in teams_by_region.items():
        # Group teams by seed to find play-in duplicates
        by_seed: Dict[int, List[TournamentTeam]] = {}
        for t in region_teams:
            by_seed.setdefault(t.seed, []).append(t)

        kept: List[TournamentTeam] = []
        for seed, seed_teams in by_seed.items():
            if len(seed_teams) == 1:
                kept.append(seed_teams[0])
            else:
                # Play-in pair: keep the team with higher strength (AdjEM)
                winner = max(seed_teams, key=lambda t: t.strength)
                eliminated = [t for t in seed_teams if t.team_id != winner.team_id]
                logger.info(
                    "First Four resolved: %s (strength=%.2f) advances over %s in %s region (seed %d)",
                    winner.team_id,
                    winner.strength,
                    ", ".join(t.team_id for t in eliminated),
                    region,
                    seed,
                )
                kept.append(winner)
        resolved[region] = kept
    return resolved


def run_monte_carlo(
    pipeline,
    teams: List[Team],
    rosters: Dict[str, Roster],
):
    """Run full-bracket Monte Carlo simulation.

    Translated from ``SOTAPipeline._run_monte_carlo``.
    """
    teams_by_region: Dict[str, List[TournamentTeam]] = {r: [] for r in ["East", "West", "South", "Midwest"]}

    for team in teams:
        if team.region not in teams_by_region:
            raise DataRequirementError(f"Unknown region '{team.region}' for team '{team.name}'.")
        team_id = pipeline._team_id(team.name)
        # A4: team.strength is not used in game simulation (matchup_probs
        # determines outcomes). Set to AdjEM for display purposes only.
        feats = pipeline.feature_engineer.team_features[team_id]
        strength = float(feats.adj_efficiency_margin)
        teams_by_region[team.region].append(
            TournamentTeam(team_id=team_id, seed=team.seed, region=team.region, strength=strength)
        )

    # Resolve First Four play-in games (68 teams -> 64 teams)
    teams_by_region = _resolve_play_in_teams(teams_by_region)

    for region in teams_by_region:
        teams_by_region[region] = sorted(teams_by_region[region], key=lambda t: t.seed)
        if len(teams_by_region[region]) != 16:
            raise DataRequirementError(
                f"Region {region} has {len(teams_by_region[region])} teams. "
                "Full-bracket simulation requires 16 seeded teams per region."
            )
        seeds = {team.seed for team in teams_by_region[region]}
        if seeds != set(range(1, 17)):
            raise DataRequirementError(
                f"Region {region} must contain seeds 1-16 for a valid 63-game bracket."
            )

    # A4: Monte Carlo receives calibrated, tournament-adapted probabilities.
    # noise_std from config (default 0.12) controls bracket diversity.
    # injury_probability=0.0: injuries handled pre-simulation via
    # _injury_adjusted_probability().
    # Read mutable runtime state (MC calibration may override config defaults)
    _rs = getattr(pipeline, "_runtime_state", {})
    cfg = SimulationConfig(
        num_simulations=int(_rs.get("num_simulations", pipeline.config.num_simulations)),
        noise_std=float(_rs.get("mc_noise_std", pipeline.config.mc_noise_std)),
        injury_probability=0.0,
        random_seed=pipeline.config.random_seed,
        batch_size=500,
        regional_correlation=float(_rs.get("mc_regional_correlation", pipeline.config.mc_regional_correlation)),
    )

    bracket = TournamentBracket.create_standard_bracket(teams_by_region)
    injury_noise_table = build_injury_noise_table(pipeline, rosters, {
        pipeline._team_id(t.name): float(pipeline.feature_engineer.team_features[pipeline._team_id(t.name)].adj_efficiency_margin)
        for t in teams
    })
    matchup_cache: Dict[Tuple[str, str], float] = {}

    def predict_fn(team1_id: str, team2_id: str) -> float:
        key = (team1_id, team2_id)
        if key in matchup_cache:
            return matchup_cache[key]

        base_prob = pipeline.predict_probability(team1_id, team2_id)
        adjusted = injury_adjusted_probability(
            pipeline,
            base_prob,
            injury_noise_table.get(team1_id),
            injury_noise_table.get(team2_id),
        )
        matchup_cache[(team1_id, team2_id)] = adjusted
        matchup_cache[(team2_id, team1_id)] = float(np.clip(1.0 - adjusted, 0.01, 0.99))
        return adjusted

    from ...simulation.monte_carlo import MonteCarloEngine, validate_upset_rates

    engine = MonteCarloEngine(predict_fn, config=cfg)
    sim_results = engine.simulate_tournament(bracket, show_progress=False)

    # E1: Validate simulated upset rates against historical actuals.
    # Log-only diagnostic — does not block the pipeline.
    try:
        upset_validation = validate_upset_rates(sim_results, teams_by_region)
        if not upset_validation["passed"]:
            logger.warning(
                "MC upset rate validation FAILED — simulated rates deviate "
                "from historical. Consider adjusting mc_noise_std (currently %.3f).",
                float(_rs.get("mc_noise_std", pipeline.config.mc_noise_std)),
            )
        champion_check = upset_validation.get("champion_seed_validation", {})
        if champion_check and not champion_check.get("seed_1_passed", True):
            logger.warning(
                "MC champion-seed validation flagged: seed-1 champion share %.3f "
                "outside expected range [%.2f, %.2f].",
                champion_check.get("bucket_probabilities", {}).get("seed_1", 0.0),
                champion_check.get("seed_1_expected_range", [0.45, 0.70])[0],
                champion_check.get("seed_1_expected_range", [0.45, 0.70])[1],
            )
    except Exception as e:
        logger.debug("Upset rate validation skipped: %s", e)

    return sim_results


# ---------------------------------------------------------------------------
# Round probability conversion
# ---------------------------------------------------------------------------


def to_round_probabilities(pipeline, sim_results) -> Dict[str, Dict[str, float]]:
    """Convert simulation results to per-team round probabilities.

    Translated from ``SOTAPipeline._to_round_probabilities``.
    """
    model_probs: Dict[str, Dict[str, float]] = {}
    team_ids = set(pipeline.team_struct.keys())
    team_ids.update(sim_results.round_of_32_odds.keys())
    team_ids.update(sim_results.sweet_sixteen_odds.keys())
    team_ids.update(sim_results.elite_eight_odds.keys())
    team_ids.update(sim_results.final_four_odds.keys())
    team_ids.update(sim_results.championship_odds.keys())

    for team_id in team_ids:
        model_probs[team_id] = {
            "R32": sim_results.round_of_32_odds.get(team_id, 0.0),
            "S16": sim_results.sweet_sixteen_odds.get(team_id, 0.0),
            "E8": sim_results.elite_eight_odds.get(team_id, 0.0),
            "F4": sim_results.final_four_odds.get(team_id, 0.0),
            "CHAMP": sim_results.championship_odds.get(team_id, 0.0),
            "R64": 1.0,
        }

    return model_probs


def to_round_probabilities_from_sim(
    pipeline, sim_data: Dict
) -> Dict[str, Dict[str, float]]:
    """Build round probabilities from serialized simulation data.

    Like ``to_round_probabilities`` but operates on the dict stored in
    the report's ``artifacts.simulation`` rather than a live
    ``AggregatedResults`` object.  Used by ``_build_ev_analysis`` which
    receives the report dict, not the raw simulation object.

    Translated from ``SOTAPipeline._to_round_probabilities_from_sim``.
    """
    championship_odds = sim_data.get("championship_odds", {})
    final_four_odds = sim_data.get("final_four_odds", {})
    elite_eight_odds = sim_data.get("elite_eight_odds", {})
    sweet_sixteen_odds = sim_data.get("sweet_sixteen_odds", {})
    round_of_32_odds = sim_data.get("round_of_32_odds", {})

    team_ids = set(pipeline.team_struct.keys())
    for odds_dict in (championship_odds, final_four_odds, elite_eight_odds,
                      sweet_sixteen_odds, round_of_32_odds):
        team_ids.update(odds_dict.keys())

    model_probs: Dict[str, Dict[str, float]] = {}
    for team_id in team_ids:
        model_probs[team_id] = {
            "R64": 1.0,
            "R32": round_of_32_odds.get(team_id, 0.0),
            "S16": sweet_sixteen_odds.get(team_id, 0.0),
            "E8": elite_eight_odds.get(team_id, 0.0),
            "F4": final_four_odds.get(team_id, 0.0),
            "CHAMP": championship_odds.get(team_id, 0.0),
        }
    return model_probs


# ---------------------------------------------------------------------------
# Public picks loading
# ---------------------------------------------------------------------------


def load_public_picks(
    pipeline, model_probs: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """Load and aggregate public bracket pick percentages.

    Translated from ``SOTAPipeline._load_public_picks``.
    """
    if pipeline.config.public_picks_json:
        with open(pipeline.config.public_picks_json, "r") as f:
            payload = json.load(f)
        pipeline._validate_feed_freshness("Public picks", payload)
        public: Dict[str, Dict[str, float]] = {}
        pipeline.public_pick_sources = []

        # Format A: explicit per-source payload object {"espn": {...}, "yahoo": {...}, "cbs": {...}}
        source_weights = {"espn": 0.5, "yahoo": 0.3, "cbs": 0.2}
        source_rows: Dict[str, Dict[str, Dict[str, float]]] = {}
        for source in ("espn", "yahoo", "cbs"):
            block = payload.get(source)
            rows = extract_public_pick_rows(pipeline, block) if isinstance(block, dict) else {}
            if rows:
                source_rows[source] = rows
                pipeline.public_pick_sources.append(source)

        if source_rows:
            aggregate_rows: Dict[str, Dict[str, float]] = {}
            aggregate_weights: Dict[str, float] = {}
            for source, rows in source_rows.items():
                w = source_weights[source]
                for team_id, row in rows.items():
                    if team_id not in aggregate_rows:
                        aggregate_rows[team_id] = {"R64": 0.0, "R32": 0.0, "S16": 0.0, "E8": 0.0, "F4": 0.0, "CHAMP": 0.0}
                        aggregate_weights[team_id] = 0.0
                    aggregate_weights[team_id] += w
                    for round_name in ("R64", "R32", "S16", "E8", "F4", "CHAMP"):
                        aggregate_rows[team_id][round_name] += w * float(row.get(round_name, 0.0))
            public = {
                team_id: normalize_public_pick_row(
                    {
                        round_name: aggregate_rows[team_id][round_name] / max(aggregate_weights[team_id], 1e-9)
                        for round_name in ("R64", "R32", "S16", "E8", "F4", "CHAMP")
                    }
                )
                for team_id in aggregate_rows
            }
        else:
            # Format B: pre-aggregated payload {"teams": {...}, "sources": [...]}
            rows = extract_public_pick_rows(pipeline, payload)
            public = {team_id: normalize_public_pick_row(row) for team_id, row in rows.items()}
            if isinstance(payload.get("sources"), list):
                pipeline.public_pick_sources = [str(s).lower() for s in payload["sources"]]
            elif public:
                pipeline.public_pick_sources = ["espn"]

        if len(set(pipeline.public_pick_sources)) < pipeline.config.min_public_sources:
            raise DataRequirementError(
                f"Public pick source coverage too low ({len(set(pipeline.public_pick_sources))}). "
                f"Need at least {pipeline.config.min_public_sources} independent sources."
            )
        pipeline._validate_source_coverage(
            "Public picks",
            public,
            list(pipeline.team_struct.values()),
            min_ratio=0.75,
        )
        return public

    if not pipeline.config.scrape_live:
        import logging
        logging.getLogger(__name__).warning(
            "Public pick data unavailable; falling back to model probabilities (chalk bracket)."
        )
        pipeline.public_pick_sources = ["model_fallback"]
        return {team_id: dict(round_probs) for team_id, round_probs in model_probs.items()}

    # Use ScraperOrchestrator for parallel fetching with retries,
    # circuit breakers, and health tracking.  This replaces the
    # sequential scraper calls that had no retry logic.
    from ...data.scrapers.espn_picks import ScraperOrchestrator
    orchestrator = ScraperOrchestrator(cache_dir=pipeline.config.data_cache_dir)
    result = orchestrator.fetch_all_picks(
        year=pipeline.config.year,
        min_sources=pipeline.config.min_public_sources,
    )
    pipeline.public_pick_sources = result.successful_sources
    if len(set(pipeline.public_pick_sources)) < pipeline.config.min_public_sources:
        raise DataRequirementError(
            f"Public pick source coverage too low ({len(set(pipeline.public_pick_sources))}). "
            f"Need at least {pipeline.config.min_public_sources} independent sources. "
            f"Health: {result.health_summary}"
        )
    if result.is_degraded:
        logger.warning(
            "Scraper orchestrator degraded: %d/3 sources succeeded.\n%s",
            len(result.successful_sources), result.health_summary,
        )
    consensus = result.consensus
    public = {pipeline._team_id(team_id): normalize_public_pick_row(picks.as_dict) for team_id, picks in consensus.teams.items()}
    pipeline._validate_source_coverage("Public picks", public, list(pipeline.team_struct.values()), min_ratio=0.75)
    return public


def extract_public_pick_rows(
    pipeline, payload: Dict
) -> Dict[str, Dict[str, float]]:
    """Extract per-team pick rows from a public picks payload.

    Translated from ``SOTAPipeline._extract_public_pick_rows``.
    """
    if not isinstance(payload, dict):
        return {}
    teams = payload.get("teams")
    if not isinstance(teams, dict):
        return {}

    rows: Dict[str, Dict[str, float]] = {}
    _round_keys = [
        ("R64", "round_of_64_pct"),
        ("R32", "round_of_32_pct"),
        ("S16", "sweet_16_pct"),
        ("E8", "elite_8_pct"),
        ("F4", "final_four_pct"),
        ("CHAMP", "champion_pct"),
    ]
    for raw_team_id, row in teams.items():
        if not isinstance(row, dict):
            continue
        row_team_id = row.get("team_id") or raw_team_id
        team_id = pipeline._team_id(str(row_team_id))
        rows[team_id] = {
            rnd: normalize_pick_probability(row.get(rnd, row.get(alt)))
            for rnd, alt in _round_keys
        }
    return rows


def normalize_public_pick_row(row: Dict[str, float]) -> Dict[str, float]:
    """Normalize a single row of public pick percentages.

    Translated from ``SOTAPipeline._normalize_public_pick_row``.
    """
    return {
        "R64": normalize_pick_probability(row.get("R64", 0.0)),
        "R32": normalize_pick_probability(row.get("R32", 0.0)),
        "S16": normalize_pick_probability(row.get("S16", 0.0)),
        "E8": normalize_pick_probability(row.get("E8", 0.0)),
        "F4": normalize_pick_probability(row.get("F4", 0.0)),
        "CHAMP": normalize_pick_probability(row.get("CHAMP", 0.0)),
    }


def normalize_pick_probability(value) -> float:
    """Clamp and normalize a pick probability value.

    Handles various input formats:
    - float/int: 0.452 or 45.2
    - str: "45.2%", "45.2", ".452"

    Values > 1.0 are interpreted as percentages and divided by 100.
    Result is always in [0.0001, 0.9999].
    """
    if value is None:
        return 0.0001
    if isinstance(value, str):
        value = value.strip().rstrip("%")
        if not value:
            return 0.0001
    try:
        v = float(value)
    except (ValueError, TypeError):
        return 0.0001
    if v > 1.0:
        v = v / 100.0
    return float(np.clip(v, 0.0001, 0.9999))


# ---------------------------------------------------------------------------
# Game deduplication
# ---------------------------------------------------------------------------


def unique_games(
    pipeline, game_flows: Dict[str, List[GameFlow]]
) -> List[GameFlow]:
    """Return deduplicated list of games from game flow dict.

    Translated from ``SOTAPipeline._unique_games``.
    """
    if pipeline.all_game_flows:
        return list(pipeline.all_game_flows)
    unique: Dict[str, GameFlow] = {}
    for flows in game_flows.values():
        for g in flows:
            unique[g.game_id] = g
    return list(unique.values())


# ---------------------------------------------------------------------------
# Model confidence intervals
# ---------------------------------------------------------------------------


def estimate_model_confidence_intervals(
    pipeline, game_flows: Dict[str, List[GameFlow]]
) -> Dict[str, Dict[str, float]]:
    """DIAGNOSTIC ONLY: Estimate model confidence intervals on validation data.

    This method evaluates all three models on validation-era games and
    computes bootstrap Brier CIs.  It does NOT set pipeline.model_confidence
    to prevent leakage: confidence scores used by CFA must come from each
    model's training process (training loss / OOF Brier), not from
    validation-era evaluation.  If validation-era Brier were used for
    confidence, it would leak validation data into CFA base weights that
    are later optimized on a subset of the same validation era.

    Translated from ``SOTAPipeline._estimate_model_confidence_intervals``.
    """
    try:
        from .._optional_imports import (
            SIGNIFICANCE_TESTING_AVAILABLE,
            model_significance_report,
        )
    except ImportError:
        SIGNIFICANCE_TESTING_AVAILABLE = False
        model_significance_report = None  # type: ignore[assignment]

    all_games = sorted(
        [
            g for g in unique_games(pipeline, game_flows)
            if not _gu_detect_tournament_game(getattr(g, "game_date", f"{pipeline.config.year}-01-01"), fallback_year=pipeline.config.year)
            and g.team1_id in pipeline.feature_engineer.team_features
            and g.team2_id in pipeline.feature_engineer.team_features
        ],
        key=lambda g: (_gu_compute_game_sort_key(getattr(g, "game_date", f"{pipeline.config.year}-01-01"), fallback_year=pipeline.config.year), g.game_id),
    )

    # Only use validation-era games (after the baseline training split)
    if pipeline._validation_sort_key_boundary is not None:
        games = [
            g for g in all_games
            if _gu_compute_game_sort_key(getattr(g, "game_date", f"{pipeline.config.year}-01-01"), fallback_year=pipeline.config.year) >= pipeline._validation_sort_key_boundary
        ]
    else:
        # No validation split available — cannot estimate confidence
        # without risking leakage.  Keep conservative defaults.
        return {}

    # A1: Only track baseline model — GNN/Transformer removed from ensemble.
    model_preds = {"baseline": []}
    outcomes = []
    for g in games:
        outcome = pipeline._game_outcome(g)
        if outcome is None:
            continue
        outcomes.append(outcome)

        matchup = pipeline.feature_engineer.create_matchup_features(g.team1_id, g.team2_id, proprietary_engine=pipeline.proprietary_engine)
        feat_vec = matchup.to_vector()
        if pipeline.feature_selector is not None and pipeline.feature_selector.is_fitted:
            feat_vec = pipeline.feature_selector.transform(feat_vec.reshape(1, -1))[0]
        model_preds["baseline"].append(pipeline.baseline_model.predict_proba(feat_vec))

    y = np.array(outcomes, dtype=float)
    if len(y) < 12:
        return {}

    stats: Dict[str, Dict[str, float]] = {}
    for model_name, pred_list in model_preds.items():
        p = np.clip(np.array(pred_list, dtype=float), 0.01, 0.99)
        center, lo, hi = bootstrap_brier_interval(pipeline, p, y)
        width = max(0.0, hi - lo)
        confidence = float(np.clip(1.0 - (center + width), 0.1, 0.95))
        # NOTE: Do NOT set pipeline.model_confidence here — that would leak
        # validation-era data into CFA base weights.  Confidence is set
        # by each model's training process: GNN/transformer from training
        # loss, baseline from validation Brier at line 1574.
        stats[model_name] = {
            "brier": float(center),
            "brier_ci_low": float(lo),
            "brier_ci_high": float(hi),
            "ci_width": float(width),
            "confidence_diagnostic": confidence,
        }
    # Fix 3: Pairwise significance tests between models
    if SIGNIFICANCE_TESTING_AVAILABLE and len(y) >= 20:
        try:
            sig_report = model_significance_report(
                {name: np.clip(np.array(preds, dtype=float), 0.01, 0.99) for name, preds in model_preds.items()},
                y,
            )
            stats["pairwise_tests"] = sig_report
        except Exception as _sig_exc:
            logger.debug("Model significance testing failed: %s", _sig_exc)

    pipeline.model_uncertainty = stats
    return stats


def bootstrap_brier_interval(
    pipeline,
    predictions: np.ndarray,
    outcomes: np.ndarray,
    rounds: int = 400,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for Brier score.

    Translated from ``SOTAPipeline._bootstrap_brier_interval``.
    """
    n = len(predictions)
    if n == 0:
        return 0.25, 0.25, 0.25
    center = float(np.mean((predictions - outcomes) ** 2))
    if n < 10:
        return center, center, center
    samples = []
    for _ in range(rounds):
        idx = pipeline.rng.integers(0, n, size=n)
        p = predictions[idx]
        y = outcomes[idx]
        samples.append(float(np.mean((p - y) ** 2)))
    lo, hi = np.percentile(np.array(samples), [5, 95])
    return center, float(lo), float(hi)


# ---------------------------------------------------------------------------
# Injury noise modeling
# ---------------------------------------------------------------------------


def build_injury_noise_table(
    pipeline,
    rosters: Dict[str, Roster],
    base_strengths: Dict[str, float],
) -> Dict[str, np.ndarray]:
    """Precompute per-team player-level injury/availability noise tables.

    Each team gets ``injury_noise_samples`` draws that represent relative
    strength shift from Selection Sunday uncertainty.

    Returns empty dict when no injury data is provided, preventing
    uninformed N(0, 0.03) random perturbation of all probabilities.

    Translated from ``SOTAPipeline._build_injury_noise_table``.
    """
    # E3: Only generate injury noise when injury data is available.
    # Without real injury reports, random perturbation adds noise
    # without information — degrading prediction quality.
    if pipeline.config.injury_report_json is None:
        return {}

    samples = max(256, int(pipeline.config.injury_noise_samples))
    out: Dict[str, np.ndarray] = {}

    for team_id in base_strengths:
        roster = rosters.get(team_id)
        if roster is None or not roster.players:
            continue  # No roster data for this team; skip (no noise applied)

        contrib = np.array([max(0.0, p.contribution_score) for p in roster.players], dtype=float)
        if float(np.sum(contrib)) <= 0.0:
            continue  # No player contribution data; skip

        base_availability = np.array([p.availability_factor for p in roster.players], dtype=float)
        event_prob = np.clip(0.03 + 0.02 * (1.0 - np.mean(base_availability)), 0.01, 0.10)

        event_mask = pipeline.rng.random((samples, len(roster.players))) < event_prob
        severity = pipeline.rng.uniform(0.20, 0.80, size=(samples, len(roster.players)))
        avail_matrix = np.broadcast_to(base_availability, (samples, len(roster.players))).copy()
        avail_matrix[event_mask] = np.clip(avail_matrix[event_mask] * (1.0 - severity[event_mask]), 0.0, 1.0)

        team_talent = avail_matrix @ contrib
        baseline = float(np.sum(base_availability * contrib))
        relative_shift = (team_talent - baseline) / max(abs(baseline), 1.0)
        out[team_id] = np.clip(relative_shift.astype(np.float32), -0.6, 0.6)
    return out


def injury_adjusted_probability(
    pipeline,
    base_probability: float,
    team1_noise: Optional[np.ndarray],
    team2_noise: Optional[np.ndarray],
) -> float:
    """Adjust win probability for injury noise.

    Translated from ``SOTAPipeline._injury_adjusted_probability``.
    """
    if team1_noise is None or team2_noise is None:
        return float(np.clip(base_probability, 0.01, 0.99))
    n = min(len(team1_noise), len(team2_noise))
    if n == 0:
        return float(np.clip(base_probability, 0.01, 0.99))

    p0 = float(np.clip(base_probability, 0.01, 0.99))
    base_logit = math.log(p0 / (1.0 - p0))
    delta = 0.75 * (team1_noise[:n] - team2_noise[:n])
    probs = 1.0 / (1.0 + np.exp(-(base_logit + delta)))
    return float(np.clip(float(np.mean(probs)), 0.01, 0.99))


# ---------------------------------------------------------------------------
# Betting markets
# ---------------------------------------------------------------------------


def load_betting_markets(pipeline) -> Optional["MarketConsensus"]:  # noqa: F821
    """Load betting market odds and compute market consensus.

    Tries JSON cache first, then live scrapers.  Returns None if
    no betting data is available.

    Translated from ``SOTAPipeline._load_betting_markets``.
    """
    try:
        from ...data.scrapers.betting_markets import (
            TheOddsAPIScraper,
            FanDuelScraper,
            DraftKingsScraper,
            MarketConsensus,
            compute_market_consensus,
        )
    except ImportError:
        logger.debug("betting_markets module not available")
        return None

    odds_by_source = []
    cache_dir = getattr(pipeline.config, "data_cache_dir", "data/raw/betting_odds")

    # Try JSON cache path first
    if pipeline.config.betting_odds_json:
        try:
            import json as _json
            with open(pipeline.config.betting_odds_json) as f:
                raw = _json.load(f)
            from ...data.scrapers.betting_markets import BettingMarketOdds
            loaded = {}
            for tid, data in raw.items():
                loaded[tid] = BettingMarketOdds(
                    team_id=tid,
                    team_name=data.get("team_name", tid),
                    season=data.get("season", pipeline.config.year),
                    source=data.get("source", "cache"),
                    championship_odds=data.get("championship_odds", 0),
                    implied_probability=data.get("implied_probability", 0.0),
                )
            if loaded:
                odds_by_source.append(loaded)
                logger.info("Loaded %d teams from betting odds cache", len(loaded))
        except Exception as e:
            logger.warning("Failed to load betting odds cache: %s", e)

    # Try live scrapers
    for ScraperCls in [TheOddsAPIScraper, FanDuelScraper, DraftKingsScraper]:
        try:
            scraper = ScraperCls(cache_dir=cache_dir)
            odds = scraper.scrape(pipeline.config.year)
            if odds:
                odds_by_source.append(odds)
                logger.info(
                    "Loaded %d teams from %s",
                    len(odds), ScraperCls.__name__,
                )
        except Exception as e:
            logger.debug("%s scrape failed: %s", ScraperCls.__name__, e)

    if not odds_by_source:
        logger.info("No betting market data available")
        return None

    consensus = compute_market_consensus(odds_by_source, adjust_vig=True)
    logger.info(
        "Market consensus: %d teams, sources=%s",
        len(consensus.team_probabilities), consensus.sources,
    )
    return consensus


def apply_market_blend(
    pipeline,
    bracket_sim,
    market_consensus: "MarketConsensus",  # noqa: F821
) -> None:
    """Blend market implied probabilities into simulation championship odds.

    Modifies ``bracket_sim.championship_odds`` in-place, combining
    the model's MC-derived championship probabilities with sportsbook
    implied probabilities using the configured blend weight.

    Translated from ``SOTAPipeline._apply_market_blend``.
    """
    from ...data.scrapers.betting_markets import blend_with_model

    market_probs = market_consensus.team_probabilities
    model_champ = bracket_sim.championship_odds

    blended = blend_with_model(
        model_probs=model_champ,
        market_probs=market_probs,
        market_weight=pipeline.config.market_blend_weight,
    )

    # Update in-place
    for tid, prob in blended.items():
        bracket_sim.championship_odds[tid] = prob

    logger.info(
        "Applied market blend (weight=%.2f): %d teams adjusted",
        pipeline.config.market_blend_weight,
        len(blended),
    )


# ---------------------------------------------------------------------------
# Tournament game exclusion
# ---------------------------------------------------------------------------


def exclude_tournament_games(
    pipeline, games: List[GameFlow], year: Optional[int] = None
) -> List[GameFlow]:
    """Remove games on or after the NCAA tournament start date.

    This is a hard safety guard to ensure tournament results never
    leak into regular-season training features.

    Translated from ``SOTAPipeline._exclude_tournament_games``.
    """
    yr = year or pipeline.config.year
    t_start = TOURNAMENT_START_DATES.get(yr, date(yr, 3, 14))
    cutoff_key = _gu_compute_game_sort_key(t_start.isoformat(), fallback_year=yr)
    before = len(games)
    filtered = [
        g for g in games
        if _gu_compute_game_sort_key(
            getattr(g, "game_date", f"{yr}-01-01"),
            fallback_year=yr,
        ) < cutoff_key
    ]
    removed = before - len(filtered)
    if removed > 0:
        logger.info(
            "Hard tournament cutoff: excluded %d games on or after %s for year %d",
            removed, t_start.isoformat(), yr,
        )
    return filtered
