"""EV analysis methods — extracted from TournamentPipeline.

Contains: refresh_ev_analysis, _build_ev_analysis, _get_ev_scoring_system,
_run_pool_competition_simulation, _pareto_brackets_to_winner_lists,
_generate_chalk_winners, _select_ev_bracket, _load_scoring_rules.

Each function takes a ``pipeline`` parameter (TournamentPipeline instance)
to access config and mutable state. This is a pragmatic extraction
that reduces tournament_pipeline line count while maintaining exact behavioral
equivalence.

Implements Agent Directive V7 S2 (modular architecture decomposition).
"""

from __future__ import annotations

import json
import logging
from typing import Callable, Dict, List, Optional, Tuple

from ..config import (
    EVModeReport,
    DataRequirementError,
    ForecastConfig,
)
from ...optimization.leverage import TeamMetadata, analyze_pool, get_strategy_profile
from ...simulation.monte_carlo import SimulationConfig, TournamentBracket, TournamentTeam

logger = logging.getLogger(__name__)

_REGIONS = ("East", "West", "South", "Midwest")
_SEED_ORDER = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]


def refresh_ev_analysis(
    pipeline,
    new_picks_json: Optional[str] = None,
    force: bool = False,
    significance_threshold_pp: float = 1.0,
) -> Dict:
    """Re-optimize EV analysis with updated public picks without re-training.

    This is the fast path for the 48-hour pre-tournament window.  Models
    remain frozen; only leverage analysis, bracket portfolio, and strategy
    recommendations are recomputed against fresh public pick data.

    Args:
        new_picks_json: Optional path to updated picks JSON.  If None,
            re-fetches from live scrapers.
        force: Force re-optimization even if picks appear unchanged.
        significance_threshold_pp: Minimum championship pick shift (in
            percentage points) to trigger re-optimization.

    Returns:
        Dict with ``refresh_result`` key containing comparison metrics
        and ``ev_analysis`` with the updated EV report (if reoptimized).

    Raises:
        RuntimeError: If called before ``run()`` or on a non-EV pipeline.
    """
    if not hasattr(pipeline, "_last_ev_report") or pipeline._last_ev_report is None:
        raise RuntimeError(
            "refresh_ev_analysis() requires a prior run(). "
            "Call pipeline.run() first to train models and generate initial EV analysis."
        )
    if pipeline.config.mode != "ev":
        raise RuntimeError(
            "refresh_ev_analysis() is only available in EV mode. "
            f"Current mode: {pipeline.config.mode}"
        )

    from ...optimization.live_refresh import LivePicksRefreshWorkflow

    workflow = LivePicksRefreshWorkflow.from_pipeline_run(
        pipeline, pipeline._last_ev_report,
    )
    result = workflow.refresh(
        force=force,
        significance_threshold_pp=significance_threshold_pp,
        new_picks_json=new_picks_json,
    )

    output = {"refresh_result": result.to_dict()}
    if result.was_refreshed and result.updated_ev_report:
        pipeline._last_ev_report["ev_analysis"] = result.updated_ev_report
        output["ev_analysis"] = result.updated_ev_report

    return output


def _build_ev_analysis(pipeline, base_report: Dict) -> "EVModeReport":
    """Construct EV-mode analysis from shared pipeline results.

    Uses the model's calibrated probabilities and public pick data
    to identify leverage opportunities and generate pool-optimized
    bracket recommendations.
    """
    ev = EVModeReport(
        pool_size=pipeline.config.ev_pool_size,
        scoring_system=pipeline.config.ev_scoring_system,
        target_percentile=pipeline.config.ev_target_percentile,
    )

    # Extract artifacts from the shared pipeline
    artifacts = base_report.get("artifacts", {})
    sim_data = artifacts.get("simulation", {})

    # Build full round probabilities from MC simulation results.
    # The shared pipeline stores per-round odds; we extract all rounds
    # so EV analysis has the complete picture for leverage calculation
    # across every tournament round (not just championship/F4).
    model_round_probs = pipeline._to_round_probabilities_from_sim(sim_data)

    # Load public picks for leverage calculation
    public_picks = pipeline._load_public_picks(model_round_probs)

    # Build team metadata
    team_metadata = {
        team_id: TeamMetadata(
            team_name=team.name, seed=team.seed, region=team.region
        )
        for team_id, team in pipeline.team_struct.items()
    }

    # EV scoring system round weights
    ev_scoring = _get_ev_scoring_system(pipeline)

    # Build pool-size-adaptive strategy profile from config
    ev_scoring_name = pipeline.config.ev_scoring_system or "standard"
    strategy_profile = get_strategy_profile(
        pipeline.config.ev_pool_size,
        scoring_system=ev_scoring_name,
        contrarian_override=(
            pipeline.config.ev_contrarian_strength
            if pipeline.config.ev_contrarian_strength != 1.0
            else None
        ),
        payout_structure=pipeline.config.ev_payout_structure,
    )
    logger.info(
        "EV strategy profile: pool_size=%d, contrarian=%.1f, risk=%s — %s",
        strategy_profile.pool_size,
        strategy_profile.contrarian_strength,
        strategy_profile.champion_risk_level,
        strategy_profile.description,
    )

    # Behavioral archetype modeling for realistic opponent field
    archetype_pick_dist = None
    if pipeline.config.ev_enable_archetypes:
        try:
            from ...simulation.competitor_archetypes import (
                blend_archetype_picks,
                create_archetypes,
                get_archetype_mix,
            )
            archetype_mix = get_archetype_mix(
                pipeline.config.ev_pool_type,
                overrides=pipeline.config.ev_archetype_overrides,
            )
            archetypes = create_archetypes(archetype_mix)
            # Build seed lookup from team metadata
            seed_lookup = {
                tid: tm.seed for tid, tm in team_metadata.items()
            }
            archetype_pick_dist = blend_archetype_picks(
                archetypes, model_round_probs, public_picks,
                seeds=seed_lookup,
            )
            logger.info(
                "Archetype modeling enabled: pool_type=%s, archetypes=%s",
                pipeline.config.ev_pool_type,
                [a.params.name for a in archetypes],
            )
        except Exception as e:
            logger.warning("Archetype modeling failed: %s", e)

    # Run pool analysis with EV parameters and strategy profile
    pool_analysis = analyze_pool(
        pipeline.config.ev_pool_size,
        model_round_probs,
        public_picks,
        scoring_system=ev_scoring,
        team_metadata=team_metadata,
        ev_scoring_system=ev_scoring_name,
        strategy_profile=strategy_profile,
        archetype_picks=archetype_pick_dist,
    )

    # Wire bracket search optimizer as the primary bracket refinement strategy.
    # The Pareto brackets from analyze_pool() provide good starting seeds;
    # SimulatedAnnealingOptimizer navigates the 2^63 space from those seeds
    # to find higher-EV configurations that pure sampling misses.
    effective_public_for_search = (
        archetype_pick_dist if archetype_pick_dist is not None else public_picks
    )
    pool_analysis.pareto_brackets = _optimize_pareto_brackets_with_search(
        pipeline=pipeline,
        pool_analysis=pool_analysis,
        public_picks=effective_public_for_search,
        scoring_system=ev_scoring,
    )

    ev.recommended_strategy = pool_analysis.recommended_strategy

    # Leverage picks
    ev.leverage_picks = [
        {
            "team_id": p.team_id,
            "team_name": pipeline.team_id_to_name.get(p.team_id, p.team_name),
            "round": p.round_name,
            "model_probability": round(p.model_probability, 4),
            "public_pick_percentage": round(p.public_pick_percentage, 4),
            "leverage_ratio": round(p.leverage_ratio, 3),
            "ev_differential": round(p.expected_value_differential, 3),
        }
        for p in pool_analysis.leverage_picks[:20]
    ]

    # Fade picks (over-owned teams to avoid)
    ev.fade_picks = [
        {
            "team_id": p.team_id,
            "team_name": pipeline.team_id_to_name.get(p.team_id, p.team_name),
            "round": p.round_name,
            "model_probability": round(p.model_probability, 4),
            "public_pick_percentage": round(p.public_pick_percentage, 4),
            "leverage_ratio": round(p.leverage_ratio, 3),
        }
        for p in pool_analysis.fade_picks[:15]
    ]

    # Model vs public divergence for championship
    championship_odds = {
        team_id: probs.get("CHAMP", 0.0)
        for team_id, probs in model_round_probs.items()
    }
    for team_id, model_prob in championship_odds.items():
        pub_prob = public_picks.get(team_id, {}).get("CHAMP", 0.0)
        if pub_prob > 0.01 or model_prob > 0.01:
            ev.model_vs_public_divergence[team_id] = round(
                model_prob - pub_prob, 4
            )

    # Pareto brackets summary
    ev.pareto_brackets = [
        b.to_dict() if hasattr(b, "to_dict") else {}
        for b in pool_analysis.pareto_brackets[:5]
    ]

    # Pool EV analysis by strategy
    ev.pool_ev_analysis = pool_analysis.strategy_evs if hasattr(pool_analysis, "strategy_evs") else {}

    # --- ESPN multi-bracket portfolio (Protocol Section 4.5) ---
    # Generate risk-profiled brackets via ESPNPoolPortfolio when the pool
    # has Pareto seed brackets to optimize from.
    #
    # ESPNPoolPortfolio.generate() needs a SearchBracket (List[BracketPick]),
    # but pareto_brackets contain BracketConfiguration (Dict[str,str] picks).
    # We reuse _bracket_config_to_search_bracket for the conversion — the
    # same helper that _optimize_pareto_brackets_with_search uses.
    if pipeline.config.espn_n_brackets >= 1 and pool_analysis.pareto_brackets:
        try:
            from ...optimization.bracket_portfolio import ESPNPoolPortfolio

            # Build seed/region lookups from team struct (same as bracket search)
            _teams_by_region: Dict[str, Dict[int, str]] = {r: {} for r in _REGIONS}
            team_seeds: Dict[str, int] = {}
            team_regions: Dict[str, str] = {}
            for tid, team in pipeline.team_struct.items():
                if team.region not in _teams_by_region:
                    continue
                _teams_by_region[team.region][team.seed] = tid
                team_seeds[tid] = team.seed
                team_regions[tid] = team.region

            predict_fn = pipeline.predict_probability

            portfolio = ESPNPoolPortfolio(
                predict_fn=predict_fn,
                public_picks=public_picks,
                team_seeds=team_seeds,
                team_regions=team_regions,
                scoring_system=ev_scoring,
            )
            profile_names = [
                p.strip()
                for p in pipeline.config.espn_risk_profiles.split(",")
            ]

            # Convert the top Pareto bracket (BracketConfiguration) to a
            # SearchBracket so the SA optimizer inside generate() can work.
            seed_bracket = _bracket_config_to_search_bracket(
                pool_analysis.pareto_brackets[0], _teams_by_region, predict_fn,
            )

            espn_results = portfolio.generate(
                profile_names=profile_names,
                pool_size=pipeline.config.ev_pool_size,
                base_bracket=seed_bracket,
            )
            ev.espn_portfolio = [
                {
                    "profile": name,
                    "bracket": bracket.to_submission_dict()
                    if hasattr(bracket, "to_submission_dict")
                    else {},
                }
                for name, bracket in espn_results
            ]
            logger.info(
                "ESPN portfolio: generated %d brackets (%s)",
                len(espn_results),
                ", ".join(name for name, _ in espn_results),
            )
        except Exception as exc:
            logger.warning("ESPNPoolPortfolio generation failed: %s", exc)

    # Bracket portfolio summary from base report
    ev.bracket_portfolio_summary = artifacts.get("bracket_portfolio", {})

    # Win probability estimates via pool competition simulation
    # (Phase 4: double Monte Carlo — simulate tournament outcomes,
    # score all brackets, rank against synthetic opponent field.)
    effective_picks = archetype_pick_dist if archetype_pick_dist is not None else public_picks
    try:
        pool_sim_result = _run_pool_competition_simulation(
            pipeline,
            pareto_brackets=pool_analysis.pareto_brackets,
            pick_distribution=effective_picks,
            scoring_system=ev_scoring,
            target_percentiles=[
                pipeline.config.ev_target_percentile,
                0.01, 0.05, 0.10, 0.25,
            ],
        )
        ev.win_probabilities = pool_sim_result.get_best_win_probabilities()
        ev.competition_simulation = pool_sim_result.to_dict()
    except Exception as exc:
        logger.warning("Pool competition simulation failed: %s", exc)
        ev.win_probabilities = {
            "top_1pct": 0.0,
            "top_5pct": 0.0,
            "top_10pct": 0.0,
        }
        ev.competition_simulation = {"error": str(exc)}

    # Protocol v2 ESPN optimizer (path-dependent, public-pick-aware).
    # Keep behind ev_enable_search to control extra compute.
    if pipeline.config.ev_enable_search:
        try:
            from ...espn.bracket_optimizer import (
                ESPNBracketOptimizer,
                ESPNOptimizationConfig,
            )

            first_round_matchups = _build_first_round_matchups(pipeline)
            all_team_ids = list(dict.fromkeys(first_round_matchups))
            matchup_probs: Dict[Tuple[str, str], float] = {}
            for i, team1 in enumerate(all_team_ids):
                for j in range(i + 1, len(all_team_ids)):
                    team2 = all_team_ids[j]
                    p12 = float(pipeline.predict_probability(team1, team2))
                    matchup_probs[(team1, team2)] = p12
                    matchup_probs[(team2, team1)] = 1.0 - p12

            optimizer = ESPNBracketOptimizer(
                first_round_matchups=first_round_matchups,
                matchup_probs=matchup_probs,
                model_round_probs=model_round_probs,
                public_pick_distribution=effective_picks,
                scoring_system=ev_scoring,
            )
            opt_result = optimizer.optimize(
                ESPNOptimizationConfig(
                    num_simulations=max(10000, int(pipeline.config.portfolio_n_simulations)),
                    pool_size=max(2, int(pipeline.config.ev_pool_size)),
                    top_money_pct=max(0.01, float(pipeline.config.ev_target_percentile)),
                    n_candidates=max(6, int(pipeline.config.espn_n_brackets * 3)),
                    random_seed=int(pipeline.config.random_seed),
                    max_path_disruption_cost=float(pipeline.config.max_path_disruption_cost),
                )
            )
            ev.competition_simulation["protocol_v2_optimizer"] = {
                "rank_percentile": round(opt_result.simulated_rank_percentile, 4),
                "top_10_rate": round(opt_result.top_10_rate, 4),
                "top_money_rate": round(opt_result.top_money_rate, 4),
                "path_protection_score": round(opt_result.path_protection_score, 4),
                "selected_bracket": opt_result.selected_bracket,
            }
            ev.win_probabilities["top_10pct_protocol_v2"] = round(opt_result.top_10_rate, 6)
        except Exception as exc:
            logger.warning("Protocol v2 ESPN optimizer failed: %s", exc)

    logger.info(
        "EV Mode: pool_size=%d, scoring=%s, strategy=%s, "
        "leverage_picks=%d, fade_picks=%d, win_probs=%s",
        ev.pool_size,
        ev.scoring_system,
        ev.recommended_strategy,
        len(ev.leverage_picks),
        len(ev.fade_picks),
        {k: f"{v:.3f}" for k, v in ev.win_probabilities.items()},
    )

    return ev


def _get_ev_scoring_system(pipeline) -> Dict[str, int]:
    """Return round-point mapping for the configured EV scoring system."""
    if pipeline.config.ev_scoring_system == "flat":
        return {"R64": 1, "R32": 2, "S16": 3, "E8": 4, "F4": 5, "CHAMP": 6}
    elif pipeline.config.ev_scoring_system == "upset_bonus":
        # Base points + seed-difference bonus handled downstream
        return {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
    # Standard ESPN scoring
    return {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}


def _build_first_round_matchups(pipeline) -> List[str]:
    """Construct standard 64-team first-round ordering from team_struct."""
    teams_by_region: Dict[str, Dict[int, str]] = {r: {} for r in _REGIONS}
    for team_id, team in pipeline.team_struct.items():
        if team.region in teams_by_region:
            teams_by_region[team.region][int(team.seed)] = team_id

    first_round: List[str] = []
    for region in _REGIONS:
        seed_map = teams_by_region.get(region, {})
        for high_seed, low_seed in _SEED_ORDER:
            if high_seed in seed_map and low_seed in seed_map:
                first_round.append(seed_map[high_seed])
                first_round.append(seed_map[low_seed])

    if len(first_round) != 64:
        raise DataRequirementError(
            f"Unable to build first-round ESPN bracket; expected 64 teams, got {len(first_round)}."
        )
    return first_round


# ------------------------------------------------------------------
# Pool competition simulation (Phase 4)
# ------------------------------------------------------------------

def _run_pool_competition_simulation(
    pipeline,
    pareto_brackets: List,
    pick_distribution: Dict[str, Dict[str, float]],
    scoring_system: Dict[str, int],
    target_percentiles: Optional[List[float]] = None,
):
    """Run double-MC pool competition simulation for win probabilities.

    Generates synthetic opponent brackets from the pick distribution,
    simulates tournament outcomes, scores all brackets, and computes
    the probability of finishing in each target percentile.

    Args:
        pareto_brackets: Pareto-optimal BracketConfiguration objects
            from the leverage optimizer.
        pick_distribution: Archetype-blended or public pick probs,
            team_id -> {round_name: pick_probability}.
        scoring_system: Round -> points mapping.
        target_percentiles: Percentile thresholds to estimate.

    Returns:
        PoolSimulationResult with per-bracket performance summaries.
    """
    from ...simulation.pool_competition import (
        PoolCompetitionSimulator,
        PoolSimulationConfig,
    )
    from ...simulation.monte_carlo import TournamentBracket, TournamentTeam

    # Build TournamentBracket to get first_round_matchups
    teams_by_region: Dict[str, List[TournamentTeam]] = {
        r: [] for r in ["East", "West", "South", "Midwest"]
    }
    seeds: Dict[str, int] = {}
    for team_id, team in pipeline.team_struct.items():
        if team.region not in teams_by_region:
            continue
        strength = 0.5
        if hasattr(pipeline, "feature_engineer") and team_id in pipeline.feature_engineer.team_features:
            strength = float(pipeline.feature_engineer.team_features[team_id].adj_efficiency_margin)
        teams_by_region[team.region].append(
            TournamentTeam(
                team_id=team_id,
                seed=team.seed,
                region=team.region,
                strength=strength,
            )
        )
        seeds[team_id] = team.seed

    # Sort by seed for standard bracket construction
    for region in teams_by_region:
        teams_by_region[region] = sorted(
            teams_by_region[region], key=lambda t: t.seed
        )

    bracket = TournamentBracket.create_standard_bracket(teams_by_region)
    first_round = bracket.first_round_matchups

    # Pre-compute matchup probabilities
    matchup_probs: Dict = {}
    team_ids = [t.team_id for t in bracket.teams]
    for i, t1 in enumerate(team_ids):
        for j, t2 in enumerate(team_ids):
            if i < j:
                p = pipeline.predict_probability(t1, t2)
                matchup_probs[(t1, t2)] = p
                matchup_probs[(t2, t1)] = 1.0 - p

    # Convert Pareto brackets to winner-list format
    model_brackets, model_metadata = _pareto_brackets_to_winner_lists(
        pipeline, pareto_brackets, first_round,
    )

    if not model_brackets:
        logger.warning(
            "No valid model brackets for competition simulation; "
            "generating chalk bracket as fallback."
        )
        chalk_winners = _generate_chalk_winners(first_round, matchup_probs)
        model_brackets = [{"winners": chalk_winners}]
        model_metadata = [{"id": "chalk_fallback", "strategy": "chalk"}]

    # Deduplicate target percentiles and sort
    if target_percentiles:
        target_percentiles = sorted(set(target_percentiles))

    # Scale n_tournaments by pool size for adequate sampling.
    # Protocol v2 requires >=10,000 simulations for ESPN pathway.
    pool_size = pipeline.config.ev_pool_size
    base_n_tournaments = 10000
    if pool_size > 5000:
        base_n_tournaments = 20000
    elif pool_size > 1000:
        base_n_tournaments = 15000

    config = PoolSimulationConfig(
        n_tournaments=base_n_tournaments,
        n_opponents=max(1, pool_size - len(model_brackets)),
        noise_std=float(getattr(pipeline, "_runtime_state", {}).get("mc_noise_std", pipeline.config.mc_noise_std)),
        random_seed=pipeline.config.random_seed,
        scoring_system=scoring_system,
        upset_bonus_enabled=(pipeline.config.ev_scoring_system == "upset_bonus"),
    )

    simulator = PoolCompetitionSimulator(
        config=config,
        first_round_matchups=first_round,
        matchup_probs=matchup_probs,
        pick_distribution=pick_distribution,
        seeds=seeds,
    )

    return simulator.run(
        model_brackets,
        model_bracket_metadata=model_metadata,
        target_percentiles=target_percentiles,
    )


def _pareto_brackets_to_winner_lists(
    pipeline,
    pareto_brackets: List,
    first_round_matchups: List[str],
) -> tuple:
    """Convert Pareto BracketConfiguration objects to winner-list format.

    Replays each bracket's picks dict through the bracket structure
    to produce a flat list of 63 winners in standard game order.

    Returns:
        Tuple of (model_brackets, model_metadata) where model_brackets
        is a list of {"winners": [team_id, ...]} dicts.
    """
    SEED_ORDER = [
        (1, 16), (8, 9), (5, 12), (4, 13),
        (6, 11), (3, 14), (7, 10), (2, 15),
    ]

    # Build seed-to-team mapping by region
    by_region: Dict[str, Dict[int, str]] = {
        r: {} for r in ("East", "West", "South", "Midwest")
    }
    for team_id, team in pipeline.team_struct.items():
        if team.region in by_region:
            by_region[team.region][team.seed] = team_id

    model_brackets = []
    model_metadata = []

    for b_idx, bracket in enumerate(pareto_brackets):
        if not hasattr(bracket, "picks"):
            continue

        picks = bracket.picks
        winners: List[str] = []

        # Replay the bracket structure to extract winners in game order
        # R64
        region_winners: Dict[str, List[str]] = {}
        for region in ("East", "West", "South", "Midwest"):
            region_r64 = []
            for high_seed, low_seed in SEED_ORDER:
                game_key = f"R64_{region}_{high_seed}v{low_seed}"
                winner = picks.get(game_key)
                if winner is None:
                    # Fallback: pick higher seed
                    winner = by_region.get(region, {}).get(high_seed, "")
                winners.append(winner)
                region_r64.append(winner)
            region_winners[region] = region_r64

        # R32
        for region in ("East", "West", "South", "Midwest"):
            r32_winners = []
            prev = region_winners[region]
            for idx in range(0, len(prev), 2):
                game_key = f"R32_{region}_{idx // 2 + 1}"
                winner = picks.get(game_key)
                if winner is None:
                    winner = prev[idx] if idx < len(prev) else ""
                winners.append(winner)
                r32_winners.append(winner)
            region_winners[region] = r32_winners

        # S16
        for region in ("East", "West", "South", "Midwest"):
            s16_winners = []
            prev = region_winners[region]
            for idx in range(0, len(prev), 2):
                game_key = f"S16_{region}_{idx // 2 + 1}"
                winner = picks.get(game_key)
                if winner is None:
                    winner = prev[idx] if idx < len(prev) else ""
                winners.append(winner)
                s16_winners.append(winner)
            region_winners[region] = s16_winners

        # E8 (region finals)
        for region in ("East", "West", "South", "Midwest"):
            prev = region_winners[region]
            game_key = f"E8_{region}_1"
            winner = picks.get(game_key)
            if winner is None:
                winner = prev[0] if prev else ""
            winners.append(winner)
            region_winners[region] = [winner]

        # F4 (semi-finals: East vs West, South vs Midwest — standard pairing)
        f4_teams = [
            region_winners.get("East", [""])[0],
            region_winners.get("West", [""])[0],
            region_winners.get("South", [""])[0],
            region_winners.get("Midwest", [""])[0],
        ]
        f4_winners = []
        for idx in range(0, len(f4_teams), 2):
            game_key = f"F4_{idx // 2 + 1}"
            winner = picks.get(game_key)
            if winner is None:
                winner = f4_teams[idx] if idx < len(f4_teams) else ""
            winners.append(winner)
            f4_winners.append(winner)

        # Championship
        game_key = "CHAMP_1"
        winner = picks.get(game_key)
        if winner is None:
            winner = bracket.champion if hasattr(bracket, "champion") else (
                f4_winners[0] if f4_winners else ""
            )
        winners.append(winner)

        if len(winners) == 63:
            model_brackets.append({"winners": winners})
            model_metadata.append({
                "id": f"pareto_{b_idx}",
                "strategy": getattr(bracket, "strategy", "unknown"),
            })
        else:
            logger.warning(
                "Pareto bracket %d produced %d winners (expected 63), skipping",
                b_idx, len(winners),
            )

    return model_brackets, model_metadata


def _generate_chalk_winners(
    first_round_matchups: List[str],
    matchup_probs: Dict,
) -> List[str]:
    """Generate a probability-based fallback bracket.

    Uses stochastic sampling (single draw per game) rather than
    deterministic chalk (p >= 0.5), so the fallback bracket reflects
    model uncertainty instead of systematically favoring higher seeds.
    """
    import random

    rng = random.Random(42)  # deterministic for reproducibility
    current_teams = list(first_round_matchups)
    winners: List[str] = []

    while len(current_teams) > 1:
        next_round = []
        for g in range(0, len(current_teams), 2):
            if g + 1 >= len(current_teams):
                next_round.append(current_teams[g])
                continue
            t1, t2 = current_teams[g], current_teams[g + 1]
            p = matchup_probs.get((t1, t2), 0.5)
            winner = t1 if rng.random() < p else t2
            winners.append(winner)
            next_round.append(winner)
        current_teams = next_round

    return winners


def _select_ev_bracket(pipeline, pool_analysis):
    pareto = pool_analysis.pareto_brackets
    if not pareto:
        raise ValueError("Pareto optimizer returned no bracket configurations.")

    if pipeline.config.pool_size < 20:
        return pareto[0]
    if pipeline.config.pool_size > 500:
        return pareto[-1]
    return pareto[len(pareto) // 2]


def _load_scoring_rules(pipeline) -> Optional[Dict[str, int]]:
    if not pipeline.config.scoring_rules_json:
        return None
    with open(pipeline.config.scoring_rules_json, "r") as f:
        data = json.load(f)

    if "scoring_system" in data and isinstance(data["scoring_system"], dict):
        rules = data["scoring_system"]
    else:
        rules = data

    parsed = {
        "R64": int(rules.get("R64", 10)),
        "R32": int(rules.get("R32", 20)),
        "S16": int(rules.get("S16", 40)),
        "E8": int(rules.get("E8", 80)),
        "F4": int(rules.get("F4", 160)),
        "CHAMP": int(rules.get("CHAMP", 320)),
    }
    return parsed


# ------------------------------------------------------------------
# Bracket search optimizer wiring (Phase 3)
# ------------------------------------------------------------------


def _optimize_pareto_brackets_with_search(
    pipeline,
    pool_analysis,
    public_picks: Dict[str, Dict[str, float]],
    scoring_system: Dict[str, int],
) -> List:
    """Refine Pareto brackets using the SimulatedAnnealingOptimizer.

    Converts each BracketConfiguration to a SearchBracket, runs the
    SA optimizer in EV mode (maximising per-pick uniqueness-weighted
    expected value), then converts the result back.

    Args:
        pipeline: TournamentPipeline instance with predict_probability and team_struct.
        pool_analysis: PoolAnalysis whose pareto_brackets are used as seeds.
        public_picks: Per-round public pick rates (team_id -> {round: pct}).
        scoring_system: Round -> points mapping.

    Returns:
        List of refined BracketConfiguration objects (same length as input).
    """
    from ...optimization.bracket_search import SAConfig, SimulatedAnnealingOptimizer

    if not pool_analysis.pareto_brackets:
        return pool_analysis.pareto_brackets

    # Build seed map: region -> {seed: team_id}
    teams_by_region: Dict[str, Dict[int, str]] = {r: {} for r in _REGIONS}
    team_regions: Dict[str, str] = {}
    team_seeds: Dict[str, int] = {}
    for team_id, team in pipeline.team_struct.items():
        if team.region not in teams_by_region:
            continue
        teams_by_region[team.region][team.seed] = team_id
        team_regions[team_id] = team.region
        team_seeds[team_id] = team.seed

    # Championship-level public picks summary for optimizer fallback
    champ_public = {
        tid: probs.get("CHAMP", 0.01)
        for tid, probs in public_picks.items()
    }

    predict_fn = pipeline.predict_probability
    sa_config = SAConfig(
        max_iterations=500,
        max_no_improve=100,
        random_seed=getattr(pipeline.config, "random_seed", 42),
        ev_mode=True,
        pool_size=getattr(pipeline.config, "ev_pool_size", 100),
        enable_path_protection=bool(team_regions),
    )
    optimizer = SimulatedAnnealingOptimizer(
        predict_fn=predict_fn,
        config=sa_config,
        public_picks=champ_public,
        scoring_system=scoring_system,
        round_public_picks=public_picks,
        team_regions=team_regions,
        team_seeds=team_seeds,
    )

    refined: List = []
    for bracket_config in pool_analysis.pareto_brackets:
        try:
            search_bracket = _bracket_config_to_search_bracket(
                bracket_config, teams_by_region, predict_fn,
            )
            optimized = optimizer.optimize(search_bracket)
            new_config = _search_bracket_to_bracket_config(
                optimized, bracket_config, teams_by_region,
            )
            refined.append(new_config)
        except Exception as exc:
            logger.warning(
                "Bracket search optimizer failed for '%s' bracket: %s",
                getattr(bracket_config, "strategy", "unknown"), exc,
            )
            refined.append(bracket_config)

    logger.info(
        "Bracket search optimizer: refined %d/%d Pareto brackets via SA",
        sum(1 for b in refined if getattr(b, "strategy", "") == "search_optimized"),
        len(refined),
    )
    return refined


def _bracket_config_to_search_bracket(
    bracket_config,
    teams_by_region: Dict[str, Dict[int, str]],
    predict_fn: Callable[[str, str], float],
):
    """Convert a BracketConfiguration to a SearchBracket for SA optimization.

    Replays the bracket structure game-by-game, looking up each game's
    winner from the BracketConfiguration.picks dict.  Games missing from
    the dict fall back to the model's favourite (higher predict_fn result).

    The picks list is built in the canonical region-then-round order so
    that the position index in SearchBracket.picks can be mapped back to
    game IDs in _search_bracket_to_bracket_config.
    """
    from ...optimization.bracket_search import BracketPick, SearchBracket

    picks_dict = getattr(bracket_config, "picks", {}) or {}
    picks_list: List = []
    game_idx = 0

    # --- R64 ---
    region_r64: Dict[str, List[str]] = {}
    for region in _REGIONS:
        r64: List[str] = []
        seed_map = teams_by_region.get(region, {})
        for high_seed, low_seed in _SEED_ORDER:
            t1 = seed_map.get(high_seed, "")
            t2 = seed_map.get(low_seed, "")
            if not t1 or not t2:
                continue
            winner = picks_dict.get(f"R64_{region}_{high_seed}v{low_seed}")
            if winner not in (t1, t2):
                p = predict_fn(t1, t2)
                winner, loser = (t1, t2) if p >= 0.5 else (t2, t1)
            else:
                loser = t2 if winner == t1 else t1
            p_val = predict_fn(winner, loser)
            picks_list.append(BracketPick(
                round_num=0, game_idx=game_idx,
                winner_id=winner, loser_id=loser, win_probability=p_val,
            ))
            r64.append(winner)
            game_idx += 1
        region_r64[region] = r64

    # --- R32 ---
    region_r32: Dict[str, List[str]] = {}
    for region in _REGIONS:
        r32: List[str] = []
        prev = region_r64.get(region, [])
        for idx in range(0, len(prev), 2):
            if idx + 1 >= len(prev):
                break
            t1, t2 = prev[idx], prev[idx + 1]
            winner = picks_dict.get(f"R32_{region}_{idx // 2 + 1}")
            if winner not in (t1, t2):
                p = predict_fn(t1, t2)
                winner, loser = (t1, t2) if p >= 0.5 else (t2, t1)
            else:
                loser = t2 if winner == t1 else t1
            p_val = predict_fn(winner, loser)
            picks_list.append(BracketPick(
                round_num=1, game_idx=game_idx,
                winner_id=winner, loser_id=loser, win_probability=p_val,
            ))
            r32.append(winner)
            game_idx += 1
        region_r32[region] = r32

    # --- S16 ---
    region_s16: Dict[str, List[str]] = {}
    for region in _REGIONS:
        s16: List[str] = []
        prev = region_r32.get(region, [])
        for idx in range(0, len(prev), 2):
            if idx + 1 >= len(prev):
                break
            t1, t2 = prev[idx], prev[idx + 1]
            winner = picks_dict.get(f"S16_{region}_{idx // 2 + 1}")
            if winner not in (t1, t2):
                p = predict_fn(t1, t2)
                winner, loser = (t1, t2) if p >= 0.5 else (t2, t1)
            else:
                loser = t2 if winner == t1 else t1
            p_val = predict_fn(winner, loser)
            picks_list.append(BracketPick(
                round_num=2, game_idx=game_idx,
                winner_id=winner, loser_id=loser, win_probability=p_val,
            ))
            s16.append(winner)
            game_idx += 1
        region_s16[region] = s16

    # --- E8 ---
    region_e8: Dict[str, str] = {}
    for region in _REGIONS:
        prev = region_s16.get(region, [])
        if len(prev) < 2:
            continue
        t1, t2 = prev[0], prev[1]
        winner = picks_dict.get(f"E8_{region}") or picks_dict.get(f"E8_{region}_1")
        if winner not in (t1, t2):
            p = predict_fn(t1, t2)
            winner, loser = (t1, t2) if p >= 0.5 else (t2, t1)
        else:
            loser = t2 if winner == t1 else t1
        p_val = predict_fn(winner, loser)
        picks_list.append(BracketPick(
            round_num=3, game_idx=game_idx,
            winner_id=winner, loser_id=loser, win_probability=p_val,
        ))
        region_e8[region] = winner
        game_idx += 1

    # --- F4 (East vs West, South vs Midwest) ---
    f4_winners: List[str] = []
    for r1, r2, key1, key2 in [
        ("East", "West", "F4_East_West", "F4_1"),
        ("South", "Midwest", "F4_South_Midwest", "F4_2"),
    ]:
        t1 = region_e8.get(r1, "")
        t2 = region_e8.get(r2, "")
        if not t1 or not t2:
            continue
        winner = picks_dict.get(key1) or picks_dict.get(key2)
        if winner not in (t1, t2):
            p = predict_fn(t1, t2)
            winner, loser = (t1, t2) if p >= 0.5 else (t2, t1)
        else:
            loser = t2 if winner == t1 else t1
        p_val = predict_fn(winner, loser)
        picks_list.append(BracketPick(
            round_num=4, game_idx=game_idx,
            winner_id=winner, loser_id=loser, win_probability=p_val,
        ))
        f4_winners.append(winner)
        game_idx += 1

    # --- CHAMP ---
    champion = getattr(bracket_config, "champion", "")
    if len(f4_winners) >= 2:
        t1, t2 = f4_winners[0], f4_winners[1]
        winner = picks_dict.get("CHAMP") or picks_dict.get("CHAMP_1")
        if winner not in (t1, t2):
            p = predict_fn(t1, t2)
            winner, loser = (t1, t2) if p >= 0.5 else (t2, t1)
        else:
            loser = t2 if winner == t1 else t1
        p_val = predict_fn(winner, loser)
        picks_list.append(BracketPick(
            round_num=5, game_idx=game_idx,
            winner_id=winner, loser_id=loser, win_probability=p_val,
        ))
        champion = winner

    from ...optimization.bracket_search import SearchBracket
    return SearchBracket(picks=picks_list, champion=champion)


def _search_bracket_to_bracket_config(
    search_bracket,
    original_config,
    teams_by_region: Dict[str, Dict[int, str]],
):
    """Convert an optimized SearchBracket back to BracketConfiguration.

    Uses the same canonical ordering as _bracket_config_to_search_bracket
    (R64 East→West→South→Midwest, then R32, …) so that
    picks_list[pos] maps unambiguously to a game ID.
    """
    from ...optimization.leverage import BracketConfiguration

    picks_list = search_bracket.picks
    picks_dict: Dict[str, str] = {}
    pos = 0

    # R64
    region_r64: Dict[str, List[str]] = {}
    for region in _REGIONS:
        r64: List[str] = []
        for high_seed, low_seed in _SEED_ORDER:
            if pos >= len(picks_list):
                break
            pick = picks_list[pos]
            picks_dict[f"R64_{region}_{high_seed}v{low_seed}"] = pick.winner_id
            r64.append(pick.winner_id)
            pos += 1
        region_r64[region] = r64

    # R32
    region_r32: Dict[str, List[str]] = {}
    for region in _REGIONS:
        r32: List[str] = []
        prev = region_r64.get(region, [])
        for idx in range(0, len(prev), 2):
            if pos >= len(picks_list):
                break
            pick = picks_list[pos]
            picks_dict[f"R32_{region}_{idx // 2 + 1}"] = pick.winner_id
            r32.append(pick.winner_id)
            pos += 1
        region_r32[region] = r32

    # S16
    region_s16: Dict[str, List[str]] = {}
    for region in _REGIONS:
        s16: List[str] = []
        prev = region_r32.get(region, [])
        for idx in range(0, len(prev), 2):
            if pos >= len(picks_list):
                break
            pick = picks_list[pos]
            picks_dict[f"S16_{region}_{idx // 2 + 1}"] = pick.winner_id
            s16.append(pick.winner_id)
            pos += 1
        region_s16[region] = s16

    # E8
    region_e8: Dict[str, str] = {}
    for region in _REGIONS:
        if pos >= len(picks_list):
            break
        pick = picks_list[pos]
        picks_dict[f"E8_{region}"] = pick.winner_id
        region_e8[region] = pick.winner_id
        pos += 1

    # F4
    f4_winners: List[str] = []
    for key in ("F4_East_West", "F4_South_Midwest"):
        if pos >= len(picks_list):
            break
        pick = picks_list[pos]
        picks_dict[key] = pick.winner_id
        f4_winners.append(pick.winner_id)
        pos += 1

    # CHAMP
    champion = search_bracket.champion
    if pos < len(picks_list):
        pick = picks_list[pos]
        picks_dict["CHAMP"] = pick.winner_id
        champion = pick.winner_id
    else:
        picks_dict["CHAMP"] = champion

    final_four = [region_e8.get(r, "") for r in _REGIONS]
    return BracketConfiguration(
        picks=picks_dict,
        champion=champion,
        final_four=final_four,
        strategy="search_optimized",
        expected_points=getattr(search_bracket, "fitness", 0.0),
        variance=getattr(original_config, "variance", 0.0),
    )
