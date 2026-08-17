"""Recency-weighted P(1st) hyperparameter fitter (3-year window).

Diagnostic-only walk-forward fitter for ``blend_alpha`` (see
``PoolHyperparameters`` in ``scripts/mc_pool_backtest.py``). Tunes
``blend_alpha`` by grid search, scoring each candidate value with the SAME
binary win-fraction MC-pool-simulation estimator that
``meta_region_poolaware``'s production candidate selection uses — never
Brier score or any per-game accuracy proxy. CLAUDE.md: "Model accuracy is
not the pool's bottleneck — don't pursue it for P(1st)."

Deliberately duplicates (rather than extracts) ``meta_region_poolaware``'s
candidate-generation and scoring logic, restricted to the "blend"
probability base — the same "reimplement, don't refactor ``_run_one_year``"
precedent already used by ``scripts/generate_poolaware_bracket.py``.
``_run_one_year``'s candidate order is regression-locked by
``tests/test_parallel_run_backtest.py`` via a shared ``77777 + year`` RNG
stream; this fitter uses an unrelated ``88888 + ...`` stream so it can never
perturb that lock.

LOAD-BEARING CAVEAT: a ``RecencyAlphaFitter`` instance's baked-in
``n_opponents``/``opponent_source`` must match the outer
``mc_pool_backtest`` CLI invocation's own flags, or the fitter silently
tunes ``blend_alpha`` against a different opponent field than the one
actually used for selection. Nothing in the ``HparamFitter`` protocol
enforces this — it is pure documentation discipline.

Diagnostic only: this module is NOT wired into
``scripts/generate_poolaware_bracket.py`` (the live production script),
which uses only the "tv"/"mass_avg" probability bases — never "blend" — and
is structurally unaffected by ``blend_alpha`` regardless of anything here.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from scripts.mc_pool_backtest import (
    ALL_MODES,
    ESPN_SCORING,
    ROUND_NAMES,
    PoolHyperparameters,
    _load_team_stats,
    _picks_dict_to_bool_array,
    build_blend_round_probabilities,
    build_first_round_matchups,
    build_noseed_round_probabilities,
    build_seed_probabilities,
    build_seed_round_probabilities,
    derive_f4_region_pairing,
    generate_opponent_brackets,
    load_seeds_and_regions,
    load_tournament_results,
    resolve_first_four,
    resolve_opponent_pick_distribution,
    score_brackets_team_identity,
    simulate_tournament_outcomes,
    train_noseed_model,
)
from src.optimization.bracket_construction import construct_bracket

_REGION_RISK_LEVELS: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
_EXHAUSTIVE_RISK_LEVELS: Tuple[float, ...] = (0.3, 0.5, 0.7)
_DEFAULT_ALPHA_GRID: Tuple[float, ...] = tuple(round(x * 0.1, 1) for x in range(11))


class _YearContext:
    """Walk-forward-safe per-year data needed to score candidate blend_alphas.

    Built once per fit-year and reused across the alpha grid — mirrors what
    ``_run_one_year`` builds once per test year and reuses across modes.
    """

    __slots__ = (
        "year",
        "seeds",
        "regions",
        "first_round",
        "seed_rp",
        "noseed_rp",
        "seed_pw",
        "pick_dist",
        "n_opponents",
        "pool_chalk_noise_std",
    )

    def __init__(self, year, seeds, regions, first_round, seed_rp, noseed_rp,
                 seed_pw, pick_dist, n_opponents, pool_chalk_noise_std):
        self.year = year
        self.seeds = seeds
        self.regions = regions
        self.first_round = first_round
        self.seed_rp = seed_rp
        self.noseed_rp = noseed_rp
        self.seed_pw = seed_pw
        self.pick_dist = pick_dist
        self.n_opponents = n_opponents
        self.pool_chalk_noise_std = pool_chalk_noise_std


class RecencyAlphaFitter:
    """Walk-forward ``blend_alpha`` fitter using only the most recent years.

    Grid-searches ``alpha_grid`` on the ``window_years`` most recent entries
    of ``train_years``, scoring each alpha by the mean best-candidate P(1st)
    (binary win-fraction MC estimator) across those years, then returns
    ``PoolHyperparameters(blend_alpha=best_alpha, ...)``.

    ``n_opponents``/``opponent_source``/``pool_blend_weight``/
    ``team_identity``/``pa_trials_fit`` are baked in at construction time
    because the ``HparamFitter`` protocol only ever passes ``train_years`` —
    same pattern as ``StrategiesFitter`` in ``scripts/mc_pool_backtest.py``.
    They must match the outer CLI invocation's own flags (see module
    docstring's load-bearing caveat).
    """

    def __init__(
        self,
        window_years: int = 3,
        alpha_grid: Sequence[float] = _DEFAULT_ALPHA_GRID,
        n_opponents: int = 30,
        opponent_source: str = "pool",
        pool_blend_weight: float = 0.7,
        team_identity: bool = True,
        pa_trials_fit: int = 100,
        enabled_modes: Sequence[str] = ALL_MODES,
    ) -> None:
        self.window_years = window_years
        self.alpha_grid: Tuple[float, ...] = tuple(alpha_grid)
        self.n_opponents = n_opponents
        self.opponent_source = opponent_source
        self.pool_blend_weight = pool_blend_weight
        self.team_identity = team_identity
        self.pa_trials_fit = pa_trials_fit
        self.enabled_modes: Tuple[str, ...] = tuple(enabled_modes)

    def __call__(self, train_years: Sequence[int]) -> PoolHyperparameters:
        recent = sorted(train_years)[-self.window_years:]
        if len(recent) < self.window_years:
            # Not enough walk-forward history yet — fall back to baseline.
            return PoolHyperparameters(enabled_modes=self.enabled_modes)

        max_train_year = max(train_years)
        scores: Dict[float, List[float]] = {alpha: [] for alpha in self.alpha_grid}
        for yt in recent:
            ctx = self._build_year_context(yt, max_train_year)
            if ctx is None:
                continue
            for alpha in self.alpha_grid:
                blend_rp = build_blend_round_probabilities(ctx.seed_rp, ctx.noseed_rp, alpha=alpha)
                candidates = self._generate_blend_candidates(blend_rp, ctx)
                if not candidates:
                    continue
                scores[alpha].append(self._score_candidates(candidates, ctx, alpha))

        valid_alphas = [a for a in self.alpha_grid if scores[a]]
        if not valid_alphas:
            return PoolHyperparameters(enabled_modes=self.enabled_modes)

        best_alpha = max(valid_alphas, key=lambda a: sum(scores[a]) / len(scores[a]))
        return PoolHyperparameters(blend_alpha=best_alpha, enabled_modes=self.enabled_modes)

    def _build_year_context(self, yt: int, max_train_year: int) -> Optional[_YearContext]:
        """Build walk-forward-safe context for scoring fit-year ``yt``.

        ``yt <= max_train_year < test_year`` by the caller's contract
        (``walk_forward_train_years``), so everything built here — including
        the noseed model, itself trained only on years ``< yt`` — sits
        strictly inside the outer walk-forward window: a nested window, never
        touching the real test year.
        """
        assert yt <= max_train_year, (
            f"_build_year_context received yt={yt} > max_train_year={max_train_year}"
        )

        seeds, regions = load_seeds_and_regions(yt)
        if not seeds or not regions:
            return None
        games = load_tournament_results(yt)
        if not games:
            return None
        resolve_first_four(games, seeds, regions)
        try:
            region_order = derive_f4_region_pairing(games, regions)
        except ValueError:
            return None
        first_round = build_first_round_matchups(seeds, regions, region_order=region_order)
        if len(first_round) != 64:
            return None

        stats = _load_team_stats(yt)
        seed_pw = build_seed_probabilities(seeds)

        model = train_noseed_model(max_year=yt)
        assert all(y < yt for y in model.train_years), (
            f"walk-forward violation: noseed model for fit-year {yt} trained on {model.train_years}"
        )

        seed_rp = build_seed_round_probabilities(seeds)
        noseed_rp = build_noseed_round_probabilities(model, seeds, stats)

        try:
            pick_dist, year_n_opponents, pool_chalk_noise_std = resolve_opponent_pick_distribution(
                yt, seeds, self.n_opponents, self.opponent_source, self.pool_blend_weight
            )
        except Exception:
            return None

        return _YearContext(
            year=yt,
            seeds=seeds,
            regions=regions,
            first_round=first_round,
            seed_rp=seed_rp,
            noseed_rp=noseed_rp,
            seed_pw=seed_pw,
            pick_dist=pick_dist,
            n_opponents=year_n_opponents,
            pool_chalk_noise_std=pool_chalk_noise_std,
        )

    def _generate_blend_candidates(self, blend_rp, ctx: _YearContext) -> List[np.ndarray]:
        """Mirror ``meta_region_poolaware``'s candidate generation, blend base
        only: region_top_n x 5 risk levels + exhaustive_champion x 3 risk
        levels (production lines ~3252-3270 of ``scripts/mc_pool_backtest.py``).
        """
        candidates: List[np.ndarray] = []
        scoring = dict(ESPN_SCORING)
        pub = ctx.pick_dist if ctx.pick_dist else {}
        for risk in _REGION_RISK_LEVELS:
            try:
                picks, _champ, _, _, _ = construct_bracket(
                    mode="region_top_n",
                    seeds=ctx.seeds,
                    regions=ctx.regions,
                    round_probs=blend_rp,
                    public_picks=pub,
                    risk_level=risk,
                    pool_size=ctx.n_opponents,
                    scoring_system=scoring,
                )
                candidates.append(_picks_dict_to_bool_array(picks, ctx.first_round))
            except Exception:
                pass
        for risk in _EXHAUSTIVE_RISK_LEVELS:
            try:
                picks, _champ, _, _, _ = construct_bracket(
                    mode="exhaustive_champion",
                    seeds=ctx.seeds,
                    regions=ctx.regions,
                    round_probs=blend_rp,
                    public_picks=pub,
                    risk_level=risk,
                    pool_size=ctx.n_opponents,
                    scoring_system=scoring,
                )
                candidates.append(_picks_dict_to_bool_array(picks, ctx.first_round))
            except Exception:
                pass
        return candidates

    def _score_candidates(self, candidates: Sequence[np.ndarray], ctx: _YearContext, alpha: float) -> float:
        """Binary win-fraction P(1st) estimate for the best candidate at this
        alpha, via MC pool simulation — mirrors production's binary
        estimator exactly (never a rank-based proxy; a rank-based estimator
        was tested 2026-05-16 and caused a severe regression, see
        ``scripts/mc_pool_backtest.py``'s in-code explanation).

        Uses a dedicated RNG stream, deliberately distinct from production's
        ``77777 + year`` seed.
        """
        rng = np.random.default_rng(88888 + ctx.year * 1000 + round(alpha * 100))
        best_p1 = 0.0
        for bvec in candidates:
            wins = 0
            for _ in range(self.pa_trials_fit):
                opp = generate_opponent_brackets(
                    n_opponents=ctx.n_opponents,
                    first_round_matchups=ctx.first_round,
                    pick_distribution=ctx.pick_dist,
                    matchup_probs=ctx.seed_pw,
                    seeds=ctx.seeds,
                    rng=rng,
                    chalk_noise_std=ctx.pool_chalk_noise_std,
                )
                _out, br = simulate_tournament_outcomes(
                    n_tournaments=1,
                    first_round_matchups=ctx.first_round,
                    matchup_probs=ctx.seed_pw,
                    seeds=ctx.seeds,
                    noise_std=0.16,
                    rng=rng,
                )
                sim_winners = {rnd: set(br[0][ri]) for ri, rnd in enumerate(ROUND_NAMES)}
                c_score = score_brackets_team_identity(
                    bvec.reshape(1, 63), sim_winners, ctx.first_round, ESPN_SCORING
                )[0]
                opp_scores = score_brackets_team_identity(
                    opp, sim_winners, ctx.first_round, ESPN_SCORING
                )
                if c_score >= opp_scores.max():
                    wins += 1
            p1 = wins / self.pa_trials_fit
            if p1 > best_p1:
                best_p1 = p1
        return best_p1


# Pre-configured instance matching the contract baked into
# docs/data/loyo_window_3yr_recency_fit.json. Must be invoked with a
# matching outer CLI contract:
#
#   python -m scripts.mc_pool_backtest \
#     --team-identity --opponent pool --n-opponents 30 \
#     --modes meta_region_poolaware --years 2024 2025 2026 \
#     --hparam-fitter src.optimization.recency_hparam_fitter:recency_fitter_3yr_pool30 \
#     --pa-trials 500
recency_fitter_3yr_pool30 = RecencyAlphaFitter(
    window_years=3,
    n_opponents=30,
    opponent_source="pool",
    team_identity=True,
    pa_trials_fit=100,
)
