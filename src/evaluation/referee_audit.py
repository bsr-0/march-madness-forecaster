"""Referee robustness audit for ``meta_region_poolaware``.

THE QUESTION
------------
The production strategy is *selected* by simulating candidate brackets
against one tournament-outcome model (the seed head-to-head table, the
"referee") and its headline P(1st) is *measured* against that same referee.
A selector can score well against a referee by learning that referee's
quirks without making better pool decisions. This module scores the frozen
strategy, its frozen candidate set, and every baseline strategy against
several referees at once, on identical trials, and re-runs the selection
with each referee held out.

WHAT IS FROZEN
--------------
Everything about production. The candidate recipe, risk grids, opponent
model, payout, scoring and the selection rule are imported from, or
reproduced verbatim against, the production code, and
:func:`check_parity` refuses to proceed if the reproduced production choice
differs from what the canonical harness run logged for that season. The
pass/fail thresholds are in :data:`CRITERIA` and were fixed in
``artifacts/referee_audit/PREREGISTRATION.md`` before the first run.

DESIGN
------
Per season:

1. Build the season context exactly as ``mc_pool_backtest._run_one_year``
   does (play-in resolution, real Final Four pairing, the canonical
   opponent field).
2. Build the referees as pairwise tables over the 64-team field.
3. Rebuild the production candidate set and select under each referee on
   the production selection stream (``77777 + year``), so the seed-referee
   choice is bit-identical to production and every other referee's choice
   is a common-random-numbers counterpart.
4. Draw ``n_eval_trials`` evaluation trials on a DIFFERENT stream. Trial t
   has one opponent field shared by every referee, and one outcome per
   referee drawn from an identically seeded generator, so referees differ
   only in their probabilities.
5. Score every unique bracket once per (trial, referee) with a vectorised
   team-identity scorer that is pinned to the production scorer by test.

Aggregation across seasons, the paired bootstrap, the self-referee
premium, the LORO comparison and the criteria are pure functions of the
per-season records so they can be unit-tested on synthetic input.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ROUND_NAMES: Tuple[str, ...] = ("R64", "R32", "S16", "E8", "F4", "CHAMP")
GAMES_PER_ROUND: Tuple[int, ...] = (32, 16, 8, 4, 2, 1)

# --- Referees ---------------------------------------------------------------

#: Referees with full coverage of the evaluation seasons. Only these enter a
#: verdict. Order is the order columns are reported in.
CRITERION_REFEREES: Tuple[str, ...] = ("seed", "torvik", "blend", "pit", "market")

#: Reported alongside but never used for a decision: partial coverage (fte)
#: or n = one outcome per season (actual).
SUPPLEMENTARY_REFEREES: Tuple[str, ...] = ("fte", "actual")

#: The referee the production selector uses.
SELECTION_REFEREE = "seed"

#: The primary independent referee named by the pre-registration.
INDEPENDENT_REFEREE = "market"

#: Margin standard deviation (points) used to turn FiveThirtyEight power
#: ratings into win probabilities. Fixed before running; the ordinary
#: college-basketball figure.
FTE_MARGIN_SIGMA = 11.0

#: The season-level baseline every delta is taken against: the harness's
#: stochastic seed mode, the comparator behind the published headline.
BASELINE_STRATEGY = "seed"
PRODUCTION_STRATEGY = "meta_region_poolaware"

#: strategy -> the referee it was built from or selected against.
OWN_REFEREE: Dict[str, str] = {
    "seed": "seed",
    "seed_chalk": "seed",
    "torvik": "torvik",
    "torvik_argmax": "torvik",
    "fixed_tv_r50": "torvik",
    "blend": "blend",
    "blend_argmax": "blend",
    "fixed_blend_r10": "blend",
    "fixed_blend_r35": "blend",
    "fixed_blend_r50": "blend",
    "pit": "pit",
    "pit_argmax": "pit",
    "market": "market",
    "market_argmax": "market",
    "meta_region_poolaware": "seed",
    "meta_region_4champ": "seed",
}

#: Which strategies are non-independent of which referee, and why. The
#: audit report carries this table verbatim.
NON_INDEPENDENCE: Dict[str, Dict[str, str]] = {
    "seed": {
        "meta_region_poolaware": "selection referee; the published P(1st) is measured against it",
        "meta_region_4champ": "selection referee",
        "seed / seed_chalk": "built from the same head-to-head table",
        "blend / blend_argmax / fixed_blend_*": "blend pairwise is 0.5 * seed pairwise",
        "(all seasons)": "the 2010-2025 fit window contains every evaluation season",
    },
    "torvik": {
        "torvik / torvik_argmax / fixed_tv_r50": "built from barthag",
        "cand:tv_* / cand:tv_mass80_*": "candidate bases are barthag or 80% barthag",
        "cand:mass_avg_*": "the Massey composite includes Torvik as a member system",
        "blend, pit (referees and strategies)": "both fitted models take Torvik team stats as features",
    },
    "blend": {
        "blend / blend_argmax / fixed_blend_*": "built from it",
        "cand:blend_*": "candidate base",
        "seed": "half of blend's pairwise mass is the seed table",
    },
    "pit": {
        "pit / pit_argmax": "built from it",
        "torvik": "barthag is a model feature",
    },
    "market": {
        "market / market_argmax": "built from it",
        "(none of the production candidate bases)": "market is not a candidate base and was never selected against",
    },
    "fte": {"(none)": "external ratings; never used anywhere in the pipeline"},
    "actual": {"(none)": "reality"},
}

# --- Pre-registered criteria -------------------------------------------------

#: Machine-readable copy of artifacts/referee_audit/PREREGISTRATION.md.
#: tests/test_referee_audit.py pins the numbers here to the document.
CRITERIA: Dict[str, object] = {
    "criterion_referees": list(CRITERION_REFEREES),
    "baseline": BASELINE_STRATEGY,
    "production": PRODUCTION_STRATEGY,
    "independent_referee": INDEPENDENT_REFEREE,
    "bootstrap_resamples": 5000,
    "bootstrap_seed": 42,
    "ci_level": 0.95,
    "C2_material_relative_premium": 0.33,
    "C3_min_retained_fraction_of_self_edge": 0.5,
    "n_eval_trials": 300,
    "n_stochastic_samples": 50,
    "selection_trials": 500,
    "eval_seed": 20260913,
    "selection_seed_base": 77777,
    "fte_margin_sigma": FTE_MARGIN_SIGMA,
}


@dataclass
class AuditConfig:
    n_eval_trials: int = int(CRITERIA["n_eval_trials"])
    n_stochastic: int = int(CRITERIA["n_stochastic_samples"])
    pa_trials: int = int(CRITERIA["selection_trials"])
    eval_seed: int = int(CRITERIA["eval_seed"])
    stochastic_seed: int = 4242
    #: year -> label the canonical harness run selected; used for parity.
    expected_labels: Dict[int, str] = field(default_factory=dict)
    #: Directory to write the per-season frozen candidate set to (None = skip).
    candidates_dir: Optional[str] = None


# ---------------------------------------------------------------------------
# Vectorised team-identity scoring
# ---------------------------------------------------------------------------


def decode_membership(brackets: np.ndarray, first_round: Sequence[str], team_index: Mapping[str, int]) -> np.ndarray:
    """``(6, n, 64)`` bool: bracket b advances team j past round r.

    Vectorised equivalent of ``picks_by_round`` applied to every row; the
    scorer below is pinned to ``score_brackets_team_identity`` by test.
    """
    b = np.asarray(brackets, dtype=bool).reshape(-1, 63)
    n = b.shape[0]
    n_teams = len(team_index)
    idx = np.array([team_index[t] for t in first_round], dtype=np.int64)
    current = np.broadcast_to(idx, (n, len(first_round))).copy()
    membership = np.zeros((6, n, n_teams), dtype=bool)
    g0 = 0
    for r in range(6):
        ng = current.shape[1] // 2
        t1 = current[:, 0::2]
        t2 = current[:, 1::2]
        pick = b[:, g0 : g0 + ng]
        win = np.where(pick, t1, t2)
        rows = np.repeat(np.arange(n), ng)
        membership[r, rows, win.ravel()] = True
        current = win
        g0 += ng
    return membership


def winners_vector(winners_by_round: Mapping[str, set], team_index: Mapping[str, int]) -> np.ndarray:
    """``(6, 64)`` float: 1.0 where the team won that round."""
    w = np.zeros((6, len(team_index)), dtype=float)
    for r, rn in enumerate(ROUND_NAMES):
        for t in winners_by_round.get(rn, ()):
            j = team_index.get(t)
            if j is not None:
                w[r, j] = 1.0
    return w


def score_membership(membership: np.ndarray, winners: np.ndarray, points: Sequence[float]) -> np.ndarray:
    """Team-identity scores for every bracket in ``membership``."""
    total = np.zeros(membership.shape[1], dtype=float)
    for r in range(6):
        total += points[r] * (membership[r].astype(float) @ winners[r])
    return total


def ranks_against(opp_scores: np.ndarray, our_scores: np.ndarray) -> np.ndarray:
    """Harness rank convention: opponents strictly better + 1 + ties / 2."""
    srt = np.sort(opp_scores)
    n = srt.size
    right = np.searchsorted(srt, our_scores, side="right")
    left = np.searchsorted(srt, our_scores, side="left")
    better = n - right
    tied = right - left
    return better + 1 + tied / 2.0


# ---------------------------------------------------------------------------
# Season context
# ---------------------------------------------------------------------------


@dataclass
class SeasonContext:
    year: int
    seeds: Dict[str, int]
    regions: Dict[str, str]
    games: list
    first_round: List[str]
    team_index: Dict[str, int]
    pick_dist: dict
    n_opponents: int
    pool_size: int
    chalk_noise_std: float
    seed_pw: dict
    seed_rp: dict
    bases: Dict[str, object]  # name -> ProbabilityBase (with pairwise)
    referees: Dict[str, dict]  # name -> full ordered-pair table over the 64 teams
    actual_winners: Dict[str, set]
    torvik_rp: object
    massey_avg: Optional[object]
    massey_best: Optional[object]
    blend_rp: Optional[object]


def _full_pair_table(pairwise, teams: Sequence[str]) -> Dict[Tuple[str, str], float]:
    """Every ordered pair of the field, so the simulator never hits its 0.5 default."""
    out: Dict[Tuple[str, str], float] = {}
    for a in teams:
        for b in teams:
            if a != b:
                out[(a, b)] = float(pairwise.p(a, b))
    return out


def load_fte_pairwise(year: int, teams: Sequence[str], sigma: float = FTE_MARGIN_SIGMA):
    """FiveThirtyEight pre-tournament power ratings as a pairwise table, or None.

    P(A beats B) = Phi((r_A - r_B) / sigma). Returns None when the season is
    absent or any team of the field is unmapped -- partial coverage is
    reported, never patched.
    """
    from scipy.stats import norm

    from src.data.normalize import normalize_team_id
    from src.prediction.pairwise import PairwiseProbabilities

    path = PROJECT_ROOT / "data" / "kaggle" / "fivethirtyeight_ratings.json"
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    cols = doc["columns"]
    iy, it, ir = cols.index("year"), cols.index("team"), cols.index("power_rating")
    rating = {normalize_team_id(row[it]): float(row[ir]) for row in doc["data"] if row[iy] == year}
    if not rating or any(t not in rating for t in teams):
        return None
    probs = {}
    for a in teams:
        for b in teams:
            if a != b:
                probs[(a, b)] = float(norm.cdf((rating[a] - rating[b]) / sigma))
    return PairwiseProbabilities.from_dict(probs, f"fte_power({year}, sigma={sigma})")


def build_season_context(year: int, n_opponents_default: Optional[int] = None) -> SeasonContext:
    """Rebuild the season exactly as the production harness does."""
    from scripts._common import load_tournament_results
    from scripts.mc_pool_backtest import (
        N_OPPONENTS,
        _load_team_stats,
        _load_torvik_barthag,
        build_base_from_ratings,
        build_first_round_matchups,
        build_pit_base,
        derive_f4_region_pairing,
        load_seeds_and_regions,
        resolve_first_four,
        resolve_opponent_pick_distribution,
    )
    from src.optimization.payout import resolve_pool_size
    from src.prediction.massey_best_probabilities import build_massey_best_round_probabilities
    from src.prediction.massey_probabilities import load_massey_avg_barthag
    from src.prediction.market_probabilities import load_market_ratings
    from src.prediction.noseed_model import (
        build_blend_probabilities,
        build_blend_round_probabilities,
        build_noseed_probabilities,
        build_noseed_round_probabilities,
        train_noseed_model,
    )
    from src.prediction.pairwise import PairwiseProbabilities, ProbabilityBase
    from src.prediction.seed_probabilities import build_seed_probabilities, build_seed_round_probabilities
    from src.simulation.pool_competition import actual_winners_by_round

    n_opp_default = N_OPPONENTS if n_opponents_default is None else n_opponents_default

    seeds, regions = load_seeds_and_regions(year)
    games = load_tournament_results(year)
    if not seeds or not games:
        raise ValueError(f"{year}: missing seeds or games")
    resolve_first_four(games, seeds, regions)
    region_order = derive_f4_region_pairing(games, regions)
    first_round = build_first_round_matchups(seeds, regions, region_order=region_order)
    if len(first_round) != 64:
        raise ValueError(f"{year}: {len(first_round)} teams in first round")
    team_index = {t: i for i, t in enumerate(first_round)}

    pick_dist, n_opponents, chalk = resolve_opponent_pick_distribution(year, seeds, n_opp_default, "pool")
    pool_size = resolve_pool_size(n_opponents, 1)

    stats = _load_team_stats(year)
    seed_pw = build_seed_probabilities(seeds)
    seed_rp = build_seed_round_probabilities(seeds)
    model = train_noseed_model(max_year=year)
    assert all(y < year for y in model.train_years), f"walk-forward violation for {year}"
    noseed_rp = build_noseed_round_probabilities(model, seeds, stats)
    blend_rp = build_blend_round_probabilities(seed_rp, noseed_rp, alpha=0.5)
    noseed_pw = build_noseed_probabilities(model, seeds, stats)
    blend_pw = build_blend_probabilities(seed_pw, noseed_pw, alpha=0.5)

    seed_base = ProbabilityBase("seed", seed_rp, PairwiseProbabilities.from_dict(seed_pw, "historical_seed_h2h"))
    blend_base = ProbabilityBase("blend", blend_rp, PairwiseProbabilities.from_dict(blend_pw, "blend(seed,noseed,alpha=0.5)"))
    torvik_base = build_base_from_ratings("torvik", seeds, regions, _load_torvik_barthag(year, seeds))
    pit_base = build_pit_base(year, seeds, regions)
    market_base = build_base_from_ratings("market", seeds, regions, load_market_ratings(year, seeds))
    massey_avg = build_base_from_ratings("massey_avg", seeds, regions, load_massey_avg_barthag(year, seeds, Path("data")))
    massey_best = build_massey_best_round_probabilities(seeds, regions, test_year=year, data_root=Path("data"))

    bases = {"seed": seed_base, "torvik": torvik_base, "blend": blend_base}
    if pit_base is not None:
        bases["pit"] = pit_base
    if market_base is not None:
        bases["market"] = market_base

    referees = {name: _full_pair_table(base.pairwise, first_round) for name, base in bases.items()}
    fte = load_fte_pairwise(year, first_round)
    if fte is not None:
        referees["fte"] = _full_pair_table(fte, first_round)

    return SeasonContext(
        year=year,
        seeds=seeds,
        regions=regions,
        games=games,
        first_round=first_round,
        team_index=team_index,
        pick_dist=pick_dist if pick_dist else {},
        n_opponents=n_opponents,
        pool_size=pool_size,
        chalk_noise_std=float(chalk),
        seed_pw=seed_pw,
        seed_rp=seed_rp,
        bases=bases,
        referees=referees,
        actual_winners=actual_winners_by_round(games),
        torvik_rp=torvik_base,
        massey_avg=massey_avg,
        massey_best=massey_best,
        blend_rp=blend_base,
    )


# ---------------------------------------------------------------------------
# Frozen candidate set and selection
# ---------------------------------------------------------------------------


def build_production_candidates(ctx: SeasonContext) -> List[Tuple[np.ndarray, str]]:
    """The ``meta_region_poolaware`` candidate set, verbatim from the harness.

    Same construction calls, same labels, same order, same de-duplication.
    The harness builds these inline (``mc_pool_backtest.py``, the
    ``meta_region_poolaware`` branch); parity of the selected label against
    the canonical log is asserted by :func:`check_parity`.
    """
    from scripts.mc_pool_backtest import ESPN_SCORING, _picks_dict_to_bool_array
    from src.optimization.bracket_construction import construct_bracket
    from src.optimization.poolaware_recipe import (
        POOLAWARE_EXHAUSTIVE_RISKS,
        POOLAWARE_RISK_LEVELS,
        build_poolaware_prob_bases,
    )

    candidates: List[Tuple[np.ndarray, str]] = []
    scoring = dict(ESPN_SCORING)

    def try_add(label: str, **kwargs) -> None:
        try:
            p, _ch, _, _, _ = construct_bracket(
                seeds=ctx.seeds,
                regions=ctx.regions,
                public_picks=ctx.pick_dist,
                pool_size=ctx.pool_size,
                scoring_system=scoring,
                pool_factor_mode="threshold",
                **kwargs,
            )
            candidates.append((_picks_dict_to_bool_array(p, ctx.first_round), label))
        except Exception:
            pass

    prob_bases = build_poolaware_prob_bases(
        ctx.torvik_rp, massey_avg=ctx.massey_avg, massey_best=ctx.massey_best, blend=ctx.blend_rp
    )
    one_seeds = [tid for tid, s in ctx.seeds.items() if s == 1]
    for forced in one_seeds:
        try_add(f"tv_champ={forced}", mode="region_top_n", round_probs=ctx.torvik_rp, risk_level=0.5, forced_champion=forced)
    for risk in POOLAWARE_RISK_LEVELS:
        for name, rp in prob_bases:
            try_add(f"{name}_region_risk={risk}", mode="region_top_n", round_probs=rp, risk_level=risk)
    for risk in POOLAWARE_EXHAUSTIVE_RISKS:
        for name, rp in prob_bases:
            try_add(f"{name}_exhaust_risk={risk}", mode="exhaustive_champion", round_probs=rp, risk_level=risk)

    seen: set = set()
    unique: List[Tuple[np.ndarray, str]] = []
    for vec, label in candidates:
        key = vec.tobytes()
        if key not in seen:
            seen.add(key)
            unique.append((vec, label))
    return unique


def selection_p1_by_referee(
    ctx: SeasonContext,
    candidates: Sequence[Tuple[np.ndarray, str]],
    referee_names: Sequence[str],
    pa_trials: int,
    seed_base: int = int(CRITERIA["selection_seed_base"]),
) -> Dict[str, np.ndarray]:
    """Selection-trial P(1st) of every candidate under every referee.

    Uses the production functions (``draw_selection_trials``,
    ``precompute_trial_scores``, ``p_first_from_scores``) on the production
    stream ``default_rng(77777 + year)`` for EVERY referee, so the seed column
    is bit-identical to what production computes and the other columns are
    common-random-numbers counterparts of it.
    """
    from scripts.mc_pool_backtest import ESPN_SCORING, REFEREE_NOISE_STD, draw_selection_trials
    from src.optimization.pool_objectives import p_first_from_scores, precompute_trial_scores

    vecs = [v for v, _ in candidates]
    out: Dict[str, np.ndarray] = {}
    for name in referee_names:
        rng = np.random.default_rng(seed_base + ctx.year)
        trials = draw_selection_trials(
            pa_trials,
            n_opponents=ctx.n_opponents,
            first_round=ctx.first_round,
            pick_dist=ctx.pick_dist,
            matchup_probs=ctx.referees[name],
            seeds=ctx.seeds,
            rng=rng,
            chalk_noise_std=ctx.chalk_noise_std,
            noise_std=REFEREE_NOISE_STD,
        )
        scores = precompute_trial_scores(vecs, trials, ctx.first_round, ESPN_SCORING)
        out[name] = np.array([p_first_from_scores(scores, i) for i in range(scores.n_candidates)])
    return out


def argmax_first(values: Sequence[float]) -> int:
    """Production tie-break: the FIRST index reaching the maximum (strict ``>``)."""
    best, best_i = -np.inf, 0
    for i, v in enumerate(values):
        if v > best:
            best, best_i = v, i
    return best_i


def loro_choices(p1_sel: Mapping[str, np.ndarray], referees: Sequence[str]) -> Dict[str, Dict[str, int]]:
    """Candidate index chosen under each selection rule, per held-out referee.

    Returns ``{held_out: {"loro": i, "self": i, "production": i, "average_all": i}}``.
    """
    out: Dict[str, Dict[str, int]] = {}
    prod = argmax_first(p1_sel[SELECTION_REFEREE])
    avg_all = argmax_first(np.mean([p1_sel[r] for r in referees], axis=0))
    for held in referees:
        train = [r for r in referees if r != held]
        out[held] = {
            "loro": argmax_first(np.mean([p1_sel[r] for r in train], axis=0)),
            "self": argmax_first(p1_sel[held]),
            "production": prod,
            "average_all": avg_all,
        }
    return out


def build_4champ_choice(ctx: SeasonContext, candidates: Sequence[Tuple[np.ndarray, str]]) -> Optional[np.ndarray]:
    """``meta_region_4champ`` exactly as the harness selects it (seed referee, 300 trials, rng 99999 + year)."""
    from scripts.mc_pool_backtest import ESPN_SCORING, REFEREE_NOISE_STD, draw_selection_trials, score_candidate_p1

    champs = [(v, l) for v, l in candidates if l.startswith("tv_champ=")]
    if not champs:
        return None
    rng = np.random.default_rng(99999 + ctx.year)
    trials = draw_selection_trials(
        300,
        n_opponents=ctx.n_opponents,
        first_round=ctx.first_round,
        pick_dist=ctx.pick_dist,
        matchup_probs=ctx.seed_pw,
        seeds=ctx.seeds,
        rng=rng,
        noise_std=REFEREE_NOISE_STD,
    )
    p1s = [score_candidate_p1(v, trials, ctx.first_round, ESPN_SCORING) for v, _ in champs]
    return champs[argmax_first(p1s)][0]


# ---------------------------------------------------------------------------
# Strategy set
# ---------------------------------------------------------------------------


def build_strategies(
    ctx: SeasonContext,
    candidates: Sequence[Tuple[np.ndarray, str]],
    choices: Mapping[str, Dict[str, int]],
    cfg: AuditConfig,
) -> Dict[str, np.ndarray]:
    """name -> (k, 63) bool brackets. k = 1 for deterministic, n_stochastic otherwise."""
    from scripts.mc_pool_backtest import (
        ESPN_SCORING,
        _picks_dict_to_bool_array,
        build_model_bracket_argmax,
        sample_model_brackets,
    )
    from src.optimization.bracket_construction import construct_bracket

    strategies: Dict[str, np.ndarray] = {}

    def argmax_vec(base) -> np.ndarray:
        winners = build_model_bracket_argmax(ctx.first_round, base)
        vec = np.zeros(63, dtype=bool)
        current = list(ctx.first_round)
        gi = 0
        for _ in range(6):
            nxt = []
            for g in range(0, len(current), 2):
                w = winners[gi]
                vec[gi] = w == current[g]
                nxt.append(w)
                gi += 1
            current = nxt
        return vec.reshape(1, 63)

    for k, (name, base) in enumerate(ctx.bases.items()):
        strategies[f"{name}_argmax"] = argmax_vec(base)
        rng = np.random.default_rng(np.random.SeedSequence([cfg.stochastic_seed, ctx.year, k]))
        strategies[name] = sample_model_brackets(ctx.first_round, base, cfg.n_stochastic, rng)
    strategies["seed_chalk"] = strategies.pop("seed_argmax")

    def fixed_rule(name: str, base, risk: float) -> None:
        if base is None:
            return
        picks, _c, _, _, _ = construct_bracket(
            mode="region_top_n",
            seeds=ctx.seeds,
            regions=ctx.regions,
            round_probs=dict(base),
            public_picks=ctx.pick_dist,
            risk_level=risk,
            pool_size=ctx.pool_size,
            scoring_system=dict(ESPN_SCORING),
        )
        strategies[name] = _picks_dict_to_bool_array(picks, ctx.first_round).reshape(1, 63)

    fixed_rule("fixed_tv_r50", ctx.torvik_rp, 0.5)
    fixed_rule("fixed_blend_r10", ctx.blend_rp, 0.10)
    fixed_rule("fixed_blend_r35", ctx.blend_rp, 0.35)
    fixed_rule("fixed_blend_r50", ctx.blend_rp, 0.50)

    prod_idx = choices[SELECTION_REFEREE]["production"]
    strategies[PRODUCTION_STRATEGY] = candidates[prod_idx][0].reshape(1, 63)
    four = build_4champ_choice(ctx, candidates)
    if four is not None:
        strategies["meta_region_4champ"] = four.reshape(1, 63)
    for vec, label in candidates:
        strategies[f"cand:{label}"] = vec.reshape(1, 63)
    for held, rules in choices.items():
        for rule in ("loro", "self", "average_all"):
            strategies[f"sel:{rule}:{held}"] = candidates[rules[rule]][0].reshape(1, 63)
    return strategies


# ---------------------------------------------------------------------------
# Evaluation trials
# ---------------------------------------------------------------------------


def evaluate_season(ctx: SeasonContext, strategies: Mapping[str, np.ndarray], cfg: AuditConfig) -> Dict[str, Dict[str, Dict[str, float]]]:
    """metrics[referee][strategy] on common-random-number trials.

    Trial t: one opponent field (``SeedSequence([eval_seed, year, t, 0])``)
    shared by all referees; one outcome per referee from a generator seeded
    ``SeedSequence([eval_seed, year, t, 1])`` -- identical for every referee.
    ``actual`` uses the real result with the same opponent fields.
    """
    from scripts.mc_pool_backtest import ESPN_SCORING, REFEREE_NOISE_STD
    from src.simulation.pool_competition import generate_opponent_brackets, simulate_tournament_outcomes

    points = [ESPN_SCORING[r] for r in ROUND_NAMES]
    # Unique bracket matrix so identical brackets are scored once.
    rows: List[np.ndarray] = []
    index_of: Dict[bytes, int] = {}
    strat_rows: Dict[str, List[int]] = {}
    for name, mat in strategies.items():
        ids = []
        for row in np.asarray(mat, dtype=bool).reshape(-1, 63):
            key = row.tobytes()
            if key not in index_of:
                index_of[key] = len(rows)
                rows.append(row)
            ids.append(index_of[key])
        strat_rows[name] = ids
    U = np.stack(rows)
    membership = decode_membership(U, ctx.first_round, ctx.team_index)

    referee_names = list(ctx.referees) + ["actual"]
    T = cfg.n_eval_trials
    ranks = {r: np.zeros((U.shape[0], T)) for r in referee_names}
    scores = {r: np.zeros((U.shape[0], T)) for r in referee_names}
    w_actual = winners_vector(ctx.actual_winners, ctx.team_index)

    for t in range(T):
        rng_opp = np.random.default_rng(np.random.SeedSequence([cfg.eval_seed, ctx.year, t, 0]))
        # The harness's evaluation pass draws opponents WITHOUT chalk noise
        # (only its selection pass threads pool_chalk_noise_std); mirror it.
        opp = generate_opponent_brackets(ctx.n_opponents, ctx.first_round, ctx.seed_pw, ctx.pick_dist, ctx.seeds, rng_opp)
        opp_m = decode_membership(opp, ctx.first_round, ctx.team_index)
        for name in referee_names:
            if name == "actual":
                w = w_actual
            else:
                rng_out = np.random.default_rng(np.random.SeedSequence([cfg.eval_seed, ctx.year, t, 1]))
                _o, by_round = simulate_tournament_outcomes(
                    n_tournaments=1,
                    first_round_matchups=ctx.first_round,
                    matchup_probs=ctx.referees[name],
                    seeds=ctx.seeds,
                    noise_std=REFEREE_NOISE_STD,
                    rng=rng_out,
                )
                w = winners_vector({rn: set(by_round[0][ri]) for ri, rn in enumerate(ROUND_NAMES)}, ctx.team_index)
            s_u = score_membership(membership, w, points)
            s_o = score_membership(opp_m, w, points)
            ranks[name][:, t] = ranks_against(s_o, s_u)
            scores[name][:, t] = s_u

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name in referee_names:
        out[name] = {}
        for strat, ids in strat_rows.items():
            rk = ranks[name][ids]
            sc = scores[name][ids]
            out[name][strat] = {
                "p_first": float((rk == 1.0).mean()),
                "mean_rank": float(rk.mean()),
                "top3": float((rk <= 3.0).mean()),
                "top10": float((rk <= 10.0).mean()),
                "mean_score": float(sc.mean()),
            }
    return out


# ---------------------------------------------------------------------------
# Referee calibration on the real games
# ---------------------------------------------------------------------------


def referee_game_scores(referees: Mapping[str, Mapping[Tuple[str, str], float]], games: Sequence[dict], team_index: Mapping[str, int]) -> Dict[str, Dict[str, float]]:
    """Log loss, Brier and sharpness of each referee's RAW pairwise table on the season's real games.

    Play-in games are excluded (the field is the 64 that played the Round of
    64). Raw means before the simulator's logit noise, so this measures the
    table, not the draw. Which referee is credible is an empirical question,
    and this is the number that answers it.
    """
    out: Dict[str, Dict[str, float]] = {}
    for name, table in referees.items():
        ll = br = sh = 0.0
        n = 0
        for g in games:
            if g.get("round_name") == "FF":
                continue
            t1, t2 = g["team1_id"], g["team2_id"]
            if t1 not in team_index or t2 not in team_index:
                continue
            p = table.get((t1, t2))
            if p is None:
                continue
            p = min(max(float(p), 1e-6), 1 - 1e-6)
            y = 1.0 if g["team1_won"] else 0.0
            ll += -(y * math.log(p) + (1 - y) * math.log(1 - p))
            br += (p - y) ** 2
            sh += abs(p - 0.5)
            n += 1
        if n:
            out[name] = {"log_loss": ll / n, "brier": br / n, "sharpness": sh / n, "n_games": n}
    return out


# ---------------------------------------------------------------------------
# Per-season driver
# ---------------------------------------------------------------------------


def check_parity(year: int, chosen_label: str, expected: Mapping[int, str]) -> Dict[str, object]:
    exp = expected.get(year)
    return {"expected": exp, "reproduced": chosen_label, "ok": (exp is None) or (exp == chosen_label), "checked": exp is not None}


def run_season(year: int, cfg: AuditConfig) -> Dict[str, object]:
    """Everything for one season. Picklable entry point for a process pool."""
    import sys
    import time

    from src.simulation.pool_competition import picks_by_round

    def _t(msg: str, t0: float) -> float:
        print(f"    [{year}] {msg} {time.time() - t0:.0f}s", file=sys.__stdout__, flush=True)
        return time.time()

    t0 = time.time()
    ctx = build_season_context(year)
    t0 = _t("context", t0)
    referee_names = [r for r in CRITERION_REFEREES if r in ctx.referees]
    candidates = build_production_candidates(ctx)
    t0 = _t(f"{len(candidates)} candidates", t0)
    p1_sel = selection_p1_by_referee(ctx, candidates, referee_names, cfg.pa_trials)
    t0 = _t("selection", t0)
    choices = loro_choices(p1_sel, referee_names)
    prod_idx = choices[SELECTION_REFEREE]["production"]
    parity = check_parity(year, candidates[prod_idx][1], cfg.expected_labels)
    if not parity["ok"]:
        raise RuntimeError(
            f"{year}: parity failure -- reproduced production choice {parity['reproduced']!r}, "
            f"canonical log says {parity['expected']!r}. The audit does not describe production; stopping."
        )
    strategies = build_strategies(ctx, candidates, choices, cfg)
    t0 = _t("strategies", t0)
    metrics = evaluate_season(ctx, strategies, cfg)
    _t("evaluation", t0)

    labels = [l for _, l in candidates]
    if cfg.candidates_dir:
        d = Path(cfg.candidates_dir)
        d.mkdir(parents=True, exist_ok=True)
        rec = []
        for vec, label in candidates:
            picks = picks_by_round(vec, ctx.first_round)
            rec.append({"label": label, "champion": sorted(picks["CHAMP"])[0], "picks": {r: sorted(picks[r]) for r in ROUND_NAMES}})
        (d / f"candidates_{year}.json").write_text(json.dumps({"year": year, "production_label": labels[prod_idx], "candidates": rec}, indent=1))

    return {
        "year": year,
        "n_opponents": ctx.n_opponents,
        "pool_size": ctx.pool_size,
        "chalk_noise_std": ctx.chalk_noise_std,
        "referees_available": list(ctx.referees) + ["actual"],
        "candidate_labels": labels,
        "selection_p1": {r: [float(x) for x in v] for r, v in p1_sel.items()},
        "choices": {h: {k: {"index": i, "label": labels[i]} for k, i in rules.items()} for h, rules in choices.items()},
        "parity": parity,
        "metrics": metrics,
        "referee_calibration": referee_game_scores(ctx.referees, ctx.games, ctx.team_index),
    }


# ---------------------------------------------------------------------------
# Aggregation, statistics, criteria (pure functions of the season records)
# ---------------------------------------------------------------------------


def paired_bootstrap(diff: np.ndarray, n: int = int(CRITERIA["bootstrap_resamples"]), seed: int = int(CRITERIA["bootstrap_seed"])) -> Dict[str, float]:
    diff = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for k in range(n):
        idx = rng.integers(0, diff.size, diff.size)
        means[k] = diff[idx].mean()
    return {"mean": float(diff.mean()), "ci_lo": float(np.percentile(means, 2.5)), "ci_hi": float(np.percentile(means, 97.5)), "n": int(diff.size)}


def _series(seasons: Sequence[dict], referee: str, strategy: str, metric: str) -> Tuple[List[int], np.ndarray]:
    years, vals = [], []
    for s in seasons:
        m = s["metrics"].get(referee, {}).get(strategy)
        if m is not None:
            years.append(s["year"])
            vals.append(m[metric])
    return years, np.array(vals, dtype=float)


def aggregate(seasons: Sequence[dict], metric: str = "p_first") -> Dict[str, Dict[str, dict]]:
    """table[referee][strategy] = pooled mean, paired delta vs baseline, seasons."""
    order = list(CRITERION_REFEREES) + list(SUPPLEMENTARY_REFEREES)
    present = {r for s in seasons for r in s["metrics"]}
    referees = [r for r in order if r in present] + sorted(present - set(order))
    strategies = sorted({st for s in seasons for r in s["metrics"].values() for st in r})
    table: Dict[str, Dict[str, dict]] = {}
    for ref in referees:
        table[ref] = {}
        yb, base = _series(seasons, ref, BASELINE_STRATEGY, metric)
        for st in strategies:
            ys, vals = _series(seasons, ref, st, metric)
            if not ys:
                continue
            entry = {"mean": float(vals.mean()), "n_seasons": len(ys), "per_season": dict(zip(ys, vals.tolist()))}
            if ys == yb and st != BASELINE_STRATEGY:
                entry["delta_vs_baseline"] = paired_bootstrap(vals - base)
                entry["seasons_won_vs_baseline"] = int((vals > base).sum())
            table[ref][st] = entry
    return table


def self_referee_premium(seasons: Sequence[dict], strategies: Sequence[str], referees: Sequence[str] = CRITERION_REFEREES) -> Dict[str, dict]:
    """P1 under a strategy's own referee minus its mean under the other criterion referees."""
    out: Dict[str, dict] = {}
    rel_thresh = float(CRITERIA["C2_material_relative_premium"])
    for st in strategies:
        own = OWN_REFEREE.get(st)
        if own is None or own not in referees:
            continue
        others = [r for r in referees if r != own]
        _, v_own = _series(seasons, own, st, "p_first")
        other_series = [_series(seasons, r, st, "p_first")[1] for r in others]
        if not v_own.size or any(o.size != v_own.size for o in other_series):
            continue
        v_other = np.mean(other_series, axis=0)
        prem = paired_bootstrap(v_own - v_other)
        rel = prem["mean"] / v_own.mean() if v_own.mean() > 0 else float("nan")
        # Reversal: positive delta vs baseline under own referee, <= 0 under some other.
        _, b_own = _series(seasons, own, BASELINE_STRATEGY, "p_first")
        d_own = float((v_own - b_own).mean())
        reversed_under = []
        for r, o in zip(others, other_series):
            _, b_r = _series(seasons, r, BASELINE_STRATEGY, "p_first")
            if d_own > 0 and float((o - b_r).mean()) <= 0:
                reversed_under.append(r)
        material = bool(rel > rel_thresh and not (prem["ci_lo"] <= 0.0 <= prem["ci_hi"]))
        out[st] = {
            "own_referee": own,
            "p1_own": float(v_own.mean()),
            "p1_other_mean": float(v_other.mean()),
            "p1_by_referee": {r: float(o.mean()) for r, o in zip(others, other_series)},
            "premium": prem,
            "relative_premium": float(rel),
            "material": material,
            "delta_vs_baseline_own": d_own,
            "reversal_under": reversed_under,
        }
    return out


def loro_table(seasons: Sequence[dict], referees: Sequence[str] = CRITERION_REFEREES) -> Dict[str, dict]:
    """Under each held-out referee: P1 of the LORO, self, production and average-all choices vs baseline."""
    out: Dict[str, dict] = {}
    for held in referees:
        _, base = _series(seasons, held, BASELINE_STRATEGY, "p_first")
        row: Dict[str, object] = {"baseline_p1": float(base.mean())}
        for rule in ("loro", "self", "average_all"):
            _, v = _series(seasons, held, f"sel:{rule}:{held}", "p_first")
            if v.size != base.size:
                continue
            row[rule] = {"p1": float(v.mean()), "edge_vs_baseline": paired_bootstrap(v - base)}
        _, vp = _series(seasons, held, PRODUCTION_STRATEGY, "p_first")
        row["production"] = {"p1": float(vp.mean()), "edge_vs_baseline": paired_bootstrap(vp - base)}
        if "loro" in row and "self" in row:
            _, vl = _series(seasons, held, f"sel:loro:{held}", "p_first")
            _, vs = _series(seasons, held, f"sel:self:{held}", "p_first")
            row["loro_minus_self"] = paired_bootstrap(vl - vs)
            row["choice_agreement_loro_vs_production"] = float(
                np.mean([s["choices"][held]["loro"]["index"] == s["choices"][held]["production"]["index"] for s in seasons if held in s["choices"]])
            )
        out[held] = row
    return out


def pooled_calibration(seasons: Sequence[dict]) -> Dict[str, Dict[str, float]]:
    """Game-weighted pooled log loss / Brier / sharpness per referee over all seasons."""
    acc: Dict[str, Dict[str, float]] = {}
    for s in seasons:
        for r, m in s.get("referee_calibration", {}).items():
            a = acc.setdefault(r, {"log_loss": 0.0, "brier": 0.0, "sharpness": 0.0, "n_games": 0, "n_seasons": 0})
            k = m["n_games"]
            a["log_loss"] += m["log_loss"] * k
            a["brier"] += m["brier"] * k
            a["sharpness"] += m["sharpness"] * k
            a["n_games"] += k
            a["n_seasons"] += 1
    return {
        r: {"log_loss": a["log_loss"] / a["n_games"], "brier": a["brier"] / a["n_games"], "sharpness": a["sharpness"] / a["n_games"],
            "n_games": a["n_games"], "n_seasons": a["n_seasons"]}
        for r, a in acc.items() if a["n_games"]
    }


def evaluate_criteria(table_p1: Mapping[str, Mapping[str, dict]], premium: Mapping[str, dict], loro: Mapping[str, dict]) -> Dict[str, object]:
    """Apply the pre-registered C1/C2/C3 rules. Pure; tested on synthetic input."""
    refs = list(CRITERIA["criterion_referees"])
    prod = PRODUCTION_STRATEGY

    # C1
    deltas = {r: table_p1[r][prod]["delta_vs_baseline"] for r in refs if r in table_p1 and prod in table_p1[r]}
    all_positive = all(d["mean"] > 0 for d in deltas.values()) and len(deltas) == len(refs)
    ind = deltas.get(INDEPENDENT_REFEREE)
    ind_ci_excl = bool(ind and not (ind["ci_lo"] <= 0.0 <= ind["ci_hi"]) and ind["mean"] > 0)
    any_ci_below = any(d["ci_hi"] < 0 for d in deltas.values())
    if ind is None:
        c1 = "INDETERMINATE"
    elif ind["mean"] <= 0 or any_ci_below:
        c1 = "FAIL"
    elif all_positive and ind_ci_excl:
        c1 = "PASS"
    else:
        c1 = "INDETERMINATE"

    # C2
    p = premium.get(prod)
    c2 = "INDETERMINATE" if p is None else ("FAIL" if p["material"] else "PASS")

    # C3
    frac = float(CRITERIA["C3_min_retained_fraction_of_self_edge"])
    e = {}
    for h in refs:
        row = loro.get(h, {})
        if "loro" not in row or "self" not in row:
            continue
        e[h] = {"e_loro": row["loro"]["edge_vs_baseline"]["mean"], "e_self": row["self"]["edge_vs_baseline"]["mean"]}
    if len(e) != len(refs):
        c3 = "INDETERMINATE"
    elif any(v["e_loro"] <= 0 for v in e.values()):
        c3 = "FAIL"
    elif all(v["e_loro"] >= frac * v["e_self"] for v in e.values() if v["e_self"] > 0):
        c3 = "PASS"
    else:
        c3 = "INDETERMINATE"

    statuses = [c1, c2, c3]
    verdict = "ROBUST" if all(s == "PASS" for s in statuses) else ("NOT ROBUST" if "FAIL" in statuses else "INDETERMINATE")
    return {
        "C1_cross_referee_edge": {"status": c1, "deltas": deltas, "independent_ci_excludes_zero": ind_ci_excl},
        "C2_self_referee_premium": {"status": c2, "detail": p},
        "C3_leave_one_referee_out": {"status": c3, "edges": e, "min_retained_fraction": frac},
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Canonical-log parsing (for parity)
# ---------------------------------------------------------------------------

_SELECTED_RE = re.compile(r"^\s*(\d{4})\s+meta_region_poolaware\s+selected=(\S+)\s+\(best of", re.M)


def expected_labels_from_log(path: Path) -> Dict[int, str]:
    """``{year: label}`` from a canonical ``mc_pool_backtest`` run log."""
    text = Path(path).read_text()
    return {int(y): lab for y, lab in _SELECTED_RE.findall(text)}
