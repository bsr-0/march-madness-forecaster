"""Learned meta-selector for deterministic bracket construction.

Replaces stochastic coin-flip bracket generation with a model that combines
multiple probability bases as INPUT FEATURES to make per-game bracket picks.
One bracket per model per year — no randomness in our bracket.

Architecture:
    Per tournament game → feature vector (all base probs + ESPN picks + context)
    → meta-model (leverage formula or trained GBM)
    → binary pick (deterministic)
    → 63 picks = 1 bracket

Two modes:
    meta_leverage: Per-game pick maximizing leverage = P(win) × (1 - public_pick%).
                   No ML, no training. Deterministic given probs + ESPN picks.
    meta_gbm:      Shallow LightGBM trained on historical pool-leverage-weighted
                   outcomes. Walk-forward LOYO.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ROUND_NAMES = ["R64", "R32", "S16", "E8", "F4", "CHAMP"]
ESPN_SCORING = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}

# Probability bases used as features. Order determines column layout.
# Missing bases get NaN fill (LightGBM handles natively).
BASE_FEATURE_ORDER = (
    "seed",
    "torvik",
    "elo",
    "odds",
    "massey_avg",
    "massey_best",
    "spread_power",
    "ap_strength",
    "stacked",
    "noseed",
    "blend",
    "contrarian",
    "colley",
    "srs",
    "glm",
)

# Context features appended after base probability features.
CONTEXT_KEYS = (
    "coach_experience",
    "momentum",
    "talent",
    "volatility",
    "coach_pase",
    "conf_tourney_depth",
    "field_chalk_signal",
    "ncsos",
    "close_game_pct",
    "elo_slope",
    "injury_bpr_deduction",
)


def _pairwise_prob(
    team1: str,
    team2: str,
    round_name: str,
    round_probs: Dict[str, Dict[str, float]],
) -> float:
    """Normalize marginal advancement probs to a head-to-head-like score.

    KNOWN INVALID AS A PROBABILITY — DELIBERATELY UNCHANGED (2026-08-19).

    This is the ``p1 / (p1 + p2)`` reconstruction documented in
    ``src/prediction/pairwise.py``. It is exact only in R64; from R32 onward the
    reach probabilities do not cancel and the result is biased toward the
    favorite by roughly 7pp (R32) rising to 14pp (CHAMP). It is NOT
    ``P(team1 beats team2)`` and must not be used as one.

    It is retained here because this value is consumed as a **GBM feature**, not
    as a probability: ``_game_features`` computes it identically across every
    probability base and feeds the *disagreement between bases* to the model.
    A consistent monotone transform of the underlying signal is a legitimate
    feature even when it is not a calibrated probability, and the learner is
    free to recalibrate it. Replacing it would change every meta_gbm* mode's
    features and require retraining and re-backtesting the meta-selector, which
    is model work rather than a probability-contract fix.

    Worth revisiting: feeding genuine pairwise probabilities here would give the
    model a better-conditioned feature, and the deeper rounds are exactly where
    this transform is worst. Tracked in
    ARCHITECTURE_AUDIT_PREFERENCE_BRACKETS.md §2 (site 13).
    Allowlisted in tests/test_pairwise_contract.py::_ALLOWLIST.
    """
    p1 = round_probs.get(team1, {}).get(round_name, 0.0)
    p2 = round_probs.get(team2, {}).get(round_name, 0.0)
    total = p1 + p2
    if total > 1e-8:
        return p1 / total
    return 0.5


def _game_features(
    team1: str,
    team2: str,
    round_name: str,
    round_idx: int,
    base_round_probs: Dict[str, Dict[str, Dict[str, float]]],
    pick_distribution: Optional[Dict[str, Dict[str, float]]],
    seeds: Dict[str, int],
    context: Optional[Dict[str, Dict[str, float]]],
    vegas_r1: Optional[Dict[Tuple[str, str], float]] = None,
) -> np.ndarray:
    """Build feature vector for a single game.

    Features (per game):
        Per base: pairwise P(team1 wins) from that base     [len(BASE_FEATURE_ORDER)]
        seed_team1, seed_team2                               [2]
        public_pick_team1, public_pick_team2                 [2]
        leverage_diff = best_prob_diff - public_pct_diff     [1]
        source_agreement = fraction of bases favoring team1  [1]
        consensus_prob = mean P(team1) across available bases [1]
        max_disagreement = max - min P(team1) across bases   [1]
        seed_matchup_type = min(s1,s2) encoding matchup tier [1]
        round_index (0-5)                                    [1]
        Per context key: value_team1 - value_team2           [len(CONTEXT_KEYS)]
        prob_variance, prob_skewness                         [2]
        market_model_gap                                     [1]
        vegas_r1_prob: game-specific Vegas no-vig implied     [1] (NaN if not R1 or no data)

    Total: 26 (base) + 1 (vegas_r1) = 27 when vegas_r1 enabled, else 26.

    Args:
        vegas_r1: Optional mapping of (team1_id, team2_id) → P(team1 wins)
            from game-specific R1 moneylines. Key order: (min_id, max_id) by
            string sort. Only populated for R1 games with available data.
    """
    feats = []

    # Per-base pairwise probabilities
    agreements = 0
    n_valid = 0
    valid_probs = []
    for base_name in BASE_FEATURE_ORDER:
        rp = base_round_probs.get(base_name)
        if rp is not None:
            p = _pairwise_prob(team1, team2, round_name, rp)
            feats.append(p)
            valid_probs.append(p)
            if p > 0.5:
                agreements += 1
            n_valid += 1
        else:
            feats.append(np.nan)

    # Seed features
    s1 = seeds.get(team1, 8)
    s2 = seeds.get(team2, 8)
    feats.append(float(s1))
    feats.append(float(s2))

    # Public pick percentages
    if pick_distribution is not None:
        pp1 = pick_distribution.get(team1, {}).get(round_name, 0.5)
        pp2 = pick_distribution.get(team2, {}).get(round_name, 0.5)
    else:
        pp1 = 0.5
        pp2 = 0.5
    feats.append(pp1)
    feats.append(pp2)

    # Leverage diff: how much our best prob diverges from public ownership
    # Use torvik as primary, fall back to first available base
    best_p = None
    for base_name in ("torvik", "seed", "elo", "odds"):
        rp = base_round_probs.get(base_name)
        if rp is not None:
            best_p = _pairwise_prob(team1, team2, round_name, rp)
            break
    if best_p is None:
        best_p = 0.5
    lev_diff = (best_p - (1 - best_p)) - (pp1 - pp2)
    feats.append(lev_diff)

    # Source agreement: fraction of available bases favoring team1
    feats.append(agreements / max(n_valid, 1))

    # Consensus probability: mean P(team1) across all available bases
    feats.append(np.mean(valid_probs) if valid_probs else 0.5)

    # Max disagreement: spread across bases (uncertainty signal)
    feats.append((max(valid_probs) - min(valid_probs)) if len(valid_probs) >= 2 else 0.0)

    # Seed matchup type: min seed encodes the matchup tier
    # 1v16 → 1, 2v15 → 2, 3v14 → 3, ..., 8v9 → 8
    feats.append(float(min(s1, s2)))

    # Round index
    feats.append(float(round_idx))

    # Context diffs
    if context is not None:
        for key in CONTEXT_KEYS:
            ctx = context.get(key, {})
            v1 = ctx.get(team1, 0.0)
            v2 = ctx.get(team2, 0.0)
            feats.append(v1 - v2)
    else:
        for _ in CONTEXT_KEYS:
            feats.append(0.0)

    # S4: Cognitive diversity metrics (derived from existing base probs).
    if len(valid_probs) >= 2:
        vp = np.array(valid_probs)
        feats.append(float(np.var(vp)))  # prob_variance
        mean_p = np.mean(vp)
        std_p = np.std(vp)
        feats.append(float(((vp - mean_p) ** 3).mean() / (std_p**3)) if std_p > 1e-8 else 0.0)  # prob_skewness
    else:
        feats.append(0.0)  # prob_variance
        feats.append(0.0)  # prob_skewness

    # market_model_gap: odds_prob - torvik_prob (when sharp money disagrees)
    odds_rp = base_round_probs.get("odds")
    torvik_rp = base_round_probs.get("torvik")
    if odds_rp is not None and torvik_rp is not None:
        p_odds = _pairwise_prob(team1, team2, round_name, odds_rp)
        p_torvik = _pairwise_prob(team1, team2, round_name, torvik_rp)
        feats.append(p_odds - p_torvik)
    else:
        feats.append(0.0)

    # Vegas R1 game-specific probability (only for R1, NaN otherwise)
    if vegas_r1 is not None and round_name == "R64":
        key = (min(team1, team2), max(team1, team2))
        vr1 = vegas_r1.get(key)
        if vr1 is not None:
            # vr1 is P(min_id wins); convert to P(team1 wins)
            feats.append(vr1 if team1 == key[0] else 1.0 - vr1)
        else:
            feats.append(np.nan)
    elif vegas_r1 is not None:
        # Non-R1 round: NaN (model learns to ignore for later rounds)
        feats.append(np.nan)
    # If vegas_r1 is None, don't append anything (feature not enabled)

    return np.array(feats, dtype=np.float64)


def feature_names(include_vegas_r1: bool = False) -> List[str]:
    """Return ordered list of feature names matching _game_features output.

    Args:
        include_vegas_r1: If True, includes the vegas_r1_prob feature (27 total).
            Default False preserves backward compatibility (26 features).
    """
    names = [f"p_{base}" for base in BASE_FEATURE_ORDER]
    names.extend(["seed_t1", "seed_t2", "public_pick_t1", "public_pick_t2"])
    names.extend(
        [
            "leverage_diff",
            "source_agreement",
            "consensus_prob",
            "max_disagreement",
            "seed_matchup_type",
            "round_index",
        ]
    )
    names.extend([f"ctx_{key}_diff" for key in CONTEXT_KEYS])
    names.extend(["prob_variance", "prob_skewness", "market_model_gap"])
    if include_vegas_r1:
        names.append("vegas_r1_prob")
    return names


def n_features(include_vegas_r1: bool = False) -> int:
    """Number of features per game."""
    return len(feature_names(include_vegas_r1))


# ---------------------------------------------------------------------------
# Leverage baseline (no ML)
# ---------------------------------------------------------------------------


def leverage_pick(
    team1: str,
    team2: str,
    round_name: str,
    base_round_probs: Dict[str, Dict[str, Dict[str, float]]],
    pick_distribution: Optional[Dict[str, Dict[str, float]]],
    seeds: Dict[str, int],
    primary_base: str = "torvik",
) -> bool:
    """Pick the team with higher leverage = P(win) x (1 - public_pick%).

    Returns True to pick team1, False to pick team2.
    Falls back to probability-only if no pick distribution.
    """
    rp = base_round_probs.get(primary_base)
    if rp is None:
        # Fall back through bases
        for fallback in ("seed", "elo", "odds", "blend"):
            rp = base_round_probs.get(fallback)
            if rp is not None:
                break
    if rp is None:
        return True  # no data, pick team1 (arbitrary)

    p1 = _pairwise_prob(team1, team2, round_name, rp)
    p2 = 1.0 - p1

    if pick_distribution is not None:
        pp1 = pick_distribution.get(team1, {}).get(round_name, 0.5)
        pp2 = pick_distribution.get(team2, {}).get(round_name, 0.5)
    else:
        # No pick data — just use probability
        return p1 >= p2

    lev1 = p1 * (1.0 - pp1)
    lev2 = p2 * (1.0 - pp2)
    return lev1 >= lev2


def build_leverage_bracket(
    first_round_matchups: List[str],
    base_round_probs: Dict[str, Dict[str, Dict[str, float]]],
    pick_distribution: Optional[Dict[str, Dict[str, float]]],
    seeds: Dict[str, int],
    primary_base: str = "torvik",
) -> np.ndarray:
    """Build one deterministic bracket using leverage maximization.

    Walks R64→Championship. Path-consistent: winners advance.
    Returns (63,) boolean array (True = team1 picked).
    """
    bracket = np.zeros(63, dtype=bool)
    current_teams = list(first_round_matchups)
    game_idx = 0

    for round_idx in range(6):
        round_name = ROUND_NAMES[round_idx]
        next_round = []

        for g in range(0, len(current_teams), 2):
            if g + 1 >= len(current_teams):
                next_round.append(current_teams[g])
                continue

            t1, t2 = current_teams[g], current_teams[g + 1]
            pick_t1 = leverage_pick(t1, t2, round_name, base_round_probs, pick_distribution, seeds, primary_base)
            bracket[game_idx] = pick_t1
            winner = t1 if pick_t1 else t2
            next_round.append(winner)
            game_idx += 1

        current_teams = next_round

    return bracket


# ---------------------------------------------------------------------------
# Trained GBM meta-selector
# ---------------------------------------------------------------------------


def build_training_data(
    train_years: Sequence[int],
    data_root: Path = Path("data"),
    augment: bool = True,
    drop_chalk: bool = True,
    e_pap_weight: bool = False,
    pool_size: int = 30,
    use_vegas_r1: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y, weights) for meta-selector training.

    For each tournament game in each training year:
        X:       feature vector (all base probs + ESPN picks + context)
        y:       1 if team1 actually won, 0 if team2 won
        weight:  round_pts (default) or E[PAP] if e_pap_weight=True

    Args:
        augment: If True, also include the swapped (team2, team1) view of
            each game with flipped label. Doubles training data and removes
            positional bias (team1 is always the higher seed in R64).
        drop_chalk: If True, exclude R64 games where min seed <= 2
            (1v16 and 2v15 matchups). These are ~96% chalk and add noise.
        e_pap_weight: If True, weight = pts × (1 - pub_winner_rate × (N-1)/N)
            where pub_winner_rate is the public pick % for the actual winner.
            Label is STILL correctness (no contrarian forcing in the loss).
        pool_size: Pool size for E[PAP] denominator (default 30).

    Walk-forward safe: only loads data for years in train_years.
    """
    import json

    all_X, all_y, all_w = [], [], []

    for year in train_years:
        try:
            brp, pick_dist, seeds, context, first_round, games = _load_year_data(year, data_root)
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
            logger.debug("Skipping training year %d: %s", year, exc)
            continue

        # Load Vegas R1 data if enabled
        vr1 = (load_vegas_r1(year, data_root) or {}) if use_vegas_r1 else None

        # Walk the actual tournament to get real matchups per round
        game_features_and_labels = _extract_game_features_and_labels(
            games, first_round, brp, pick_dist, seeds, context, vegas_r1=vr1
        )

        for feat_vec, label, round_name, team1, team2 in game_features_and_labels:
            pts = ESPN_SCORING.get(round_name, 10)

            # Drop chalk: skip 1v16 and 2v15 R64 games
            if drop_chalk and round_name == "R64":
                s1 = seeds.get(team1, 8)
                s2 = seeds.get(team2, 8)
                if min(s1, s2) <= 2:
                    continue

            if e_pap_weight and pick_dist:
                # E[PAP]: weight by competitive edge of the correct pick.
                winner = team1 if label == 1 else team2
                winner_pub = pick_dist.get(winner, {}).get(round_name, 0.5)
                weight = float(pts) * (1.0 - winner_pub * (pool_size - 1) / pool_size)
                weight = max(weight, 1.0)  # floor at 1 to keep all games in training
            else:
                weight = float(pts)
            all_X.append(feat_vec)
            all_y.append(float(label))
            all_w.append(weight)

            # Augment: add swapped view (team2 as team1, label flipped)
            if augment:
                feat_swapped = _game_features(
                    team2,
                    team1,
                    round_name,
                    ROUND_NAMES.index(round_name),
                    brp,
                    pick_dist,
                    seeds,
                    context,
                    vegas_r1=vr1,
                )
                all_X.append(feat_swapped)
                all_y.append(1.0 - float(label))
                all_w.append(weight)

    X = np.array(all_X, dtype=np.float64)
    y = np.array(all_y, dtype=np.float64)
    w = np.array(all_w, dtype=np.float64)
    return X, y, w


def _load_year_data(
    year: int,
    data_root: Path,
) -> Tuple[
    Dict[str, Dict[str, Dict[str, float]]],  # base_round_probs
    Optional[Dict[str, Dict[str, float]]],  # pick_distribution
    Dict[str, int],  # seeds
    Dict[str, Dict[str, float]],  # context
    List[str],  # first_round_matchups
    List[dict],  # tournament games
]:
    """Load all data needed for one training year.

    Returns the same structures that _run_one_year builds, but lighter:
    only loads the bases + context, no opponent generation.
    """
    import json

    hist_dir = data_root / "raw" / "historical"
    ctx_path = hist_dir / f"tournament_context_{year}.json"
    ctx = None
    if ctx_path.exists():
        with open(ctx_path) as f:
            ctx = json.load(f)

    # Seeds
    seeds_file = ctx.get("seeds") if ctx else None
    if seeds_file is None:
        with open(hist_dir / f"tournament_seeds_{year}.json") as f:
            seeds_file = json.load(f)
    seeds_raw = seeds_file["teams"] if isinstance(seeds_file, dict) and "teams" in seeds_file else seeds_file
    seeds = {t["team_id"]: t["seed"] for t in seeds_raw}

    # Tournament results (ground truth)
    games_file = ctx.get("results") if ctx else None
    if games_file is None:
        with open(hist_dir / f"tournament_results_{year}.json") as f:
            games_file = json.load(f)
    games = games_file["games"] if isinstance(games_file, dict) and "games" in games_file else games_file

    # Regions for bracket structure
    regions = {}
    for t in seeds_raw:
        regions[t["team_id"]] = t.get("region", "Unknown")

    # First Four resolution: remove FF losers from seeds/regions
    ff_games = [g for g in games if g.get("round_name") == "FF"]
    for g in ff_games:
        loser = g["team2_id"] if g["team1_won"] else g["team1_id"]
        seeds.pop(loser, None)
        regions.pop(loser, None)

    # Derive F4 region pairing from actual games, then build matchup list
    from scripts.mc_pool_backtest import derive_f4_region_pairing, build_first_round_matchups

    region_order = derive_f4_region_pairing(games, regions)
    first_round = build_first_round_matchups(seeds, regions, region_order=region_order)

    # Build all available round_probs
    base_round_probs = _build_base_round_probs(year, seeds, regions, data_root)

    # ESPN pick distribution
    pick_dist = _load_pick_distribution(year, seeds, data_root)

    # Context features
    context = _load_context(year, seeds, data_root)

    return base_round_probs, pick_dist, seeds, context, first_round, games


def _build_base_round_probs(
    year: int,
    seeds: Dict[str, int],
    regions: Dict[str, str],
    data_root: Path,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Build round_probs from all available probability bases for a year."""
    import json

    from src.prediction.seed_probabilities import build_seed_round_probabilities

    hist_dir = data_root / "raw" / "historical"
    brp: Dict[str, Dict[str, Dict[str, float]]] = {}

    # Seed (always available)
    brp["seed"] = build_seed_round_probabilities(seeds)

    # Torvik
    try:
        torvik_path = hist_dir / f"torvik_{year}.json"
        with open(torvik_path) as f:
            torvik_file = json.load(f)
        torvik_teams = torvik_file.get("teams", torvik_file) if isinstance(torvik_file, dict) else torvik_file
        barthag = {}
        for t in torvik_teams:
            tid = t.get("team_id", t.get("canonical_id"))
            if tid and tid in seeds:
                barthag[tid] = t.get("barthag", 0.5)
        if barthag:
            brp["torvik"] = _build_mc_round_probs(seeds, regions, barthag)
    except FileNotFoundError:
        pass

    # Elo
    try:
        from src.prediction.elo_probabilities import load_elo_barthag

        elo_barthag = load_elo_barthag(year, seeds, data_root)
        if elo_barthag is not None:
            brp["elo"] = _build_mc_round_probs(seeds, regions, elo_barthag)
    except (FileNotFoundError, ImportError):
        pass

    # Market odds
    try:
        from src.prediction.market_probabilities import load_market_ratings

        market_barthag = load_market_ratings(year, seeds)
        if market_barthag is not None:
            brp["odds"] = _build_mc_round_probs(seeds, regions, market_barthag)
    except (FileNotFoundError, ImportError):
        pass

    # Spread power
    try:
        from src.prediction.market_probabilities import load_spread_power_ratings

        spread_barthag = load_spread_power_ratings(year, seeds)
        if spread_barthag is not None:
            brp["spread_power"] = _build_mc_round_probs(seeds, regions, spread_barthag)
    except (FileNotFoundError, ImportError):
        pass

    # Massey avg
    try:
        from src.prediction.massey_probabilities import load_massey_avg_barthag

        massey_barthag = load_massey_avg_barthag(year, seeds, data_root)
        if massey_barthag is not None:
            brp["massey_avg"] = _build_mc_round_probs(seeds, regions, massey_barthag)
    except (FileNotFoundError, ImportError):
        pass

    # Massey best (walk-forward)
    try:
        from src.prediction.massey_best_probabilities import build_massey_best_round_probabilities

        massey_best_rp = build_massey_best_round_probabilities(seeds, regions, test_year=year, data_root=data_root)
        if massey_best_rp is not None:
            brp["massey_best"] = massey_best_rp
    except (FileNotFoundError, ImportError):
        pass

    # AP strength
    try:
        from src.prediction.ap_probabilities import load_ap_strength_barthag

        ap_barthag = load_ap_strength_barthag(year, seeds, seeds.keys(), data_root)
        if ap_barthag is not None:
            brp["ap_strength"] = _build_mc_round_probs(seeds, regions, ap_barthag)
    except (FileNotFoundError, ImportError):
        pass

    # Stacked (walk-forward)
    try:
        from src.prediction.stacked_probabilities import build_stacked_round_probabilities

        stacked_rp = build_stacked_round_probabilities(seeds, regions, test_year=year, data_root=data_root)
        if stacked_rp is not None:
            brp["stacked"] = stacked_rp
    except (FileNotFoundError, ImportError):
        pass

    # Noseed + blend (requires training, expensive — skip for training data)
    # The noseed model uses only 12 Torvik features and was identified as a
    # toy model in the convergence test. Skip for training to keep it fast.
    # For test-year bracket construction, the backtest passes full brp.

    # Contrarian (requires pick_dist — added in build_training_data if available)

    # Custom ratings (Colley, SRS, GLM Quality) — computed from regular season
    # game results. No external data needed. Used by Kaggle 3rd + 10th place.
    try:
        from src.data.features.custom_ratings import (
            compute_all_custom_ratings,
            ratings_to_canonical,
        )

        all_custom = compute_all_custom_ratings(year, data_root)
        for rating_name, raw_ratings in all_custom.items():
            canonical = ratings_to_canonical(raw_ratings, data_root)
            # Filter to tournament teams and convert to barthag-like [0,1]
            tourney_ratings = {}
            vals = [v for k, v in canonical.items() if k in seeds]
            if not vals:
                continue
            vmin, vmax = min(vals), max(vals)
            span = vmax - vmin if vmax > vmin else 1.0
            for tid in seeds:
                raw = canonical.get(tid)
                if raw is not None:
                    tourney_ratings[tid] = 0.1 + 0.8 * (raw - vmin) / span
                else:
                    tourney_ratings[tid] = 0.5
            brp[rating_name] = _build_mc_round_probs(seeds, regions, tourney_ratings)
    except (ImportError, Exception) as exc:
        logger.debug("Custom ratings unavailable for %d: %s", year, exc)

    return brp


def load_vegas_r1(
    year: int,
    data_root: Path = Path("data"),
) -> Optional[Dict[Tuple[str, str], float]]:
    """Load game-specific Vegas R1 implied probabilities for a year.

    Returns dict of (min_team_id, max_team_id) → P(min_team_id wins),
    or None if no data. Uses no-vig implied probabilities from closing
    moneylines (SBRO 2008-2022) or the 2026 pre-tournament snapshot.

    These are pre-game probabilities (no leakage): closing lines are set
    before tip-off.
    """
    import csv

    # Try processed tournament-filtered data first
    for pattern in [
        data_root / "processed" / "vegas_r1" / f"vegas_r1_{year}.csv",
        data_root / "raw" / "bpi" / f"vegas_r1_moneylines_{year}.csv",
    ]:
        if not pattern.exists():
            continue
        try:
            games: Dict[Tuple[str, str], float] = {}
            with open(pattern) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Determine team IDs and probability
                    if "favored_team" in row:
                        # 2026 format from 3rd-place repo: (favored, underdog, prob)
                        from src.data.normalize import normalize_team_id

                        fav = normalize_team_id(row["favored_team"])
                        dog = normalize_team_id(row["underdog_team"])
                        fav_prob = float(row["favored_novig_prob"])
                        key = (min(fav, dog), max(fav, dog))
                        games[key] = fav_prob if fav == key[0] else 1.0 - fav_prob
                    elif "home_team" in row:
                        # SBRO format: (home, away, moneyline_home, implied_prob)
                        home = row["home_team"]
                        away = row["away_team"]
                        prob_h = float(row.get("implied_prob_home", 0.5))
                        key = (min(home, away), max(home, away))
                        games[key] = prob_h if home == key[0] else 1.0 - prob_h

            if games:
                logger.debug("Loaded %d Vegas R1 games for %d", len(games), year)
                return games
        except (FileNotFoundError, KeyError, ValueError) as exc:
            logger.debug("Failed to load Vegas R1 for %d: %s", year, exc)
            continue

    return None


def _build_mc_round_probs(
    seeds: Dict[str, int],
    regions: Dict[str, str],
    barthag: Dict[str, float],
    n_sims: int = 5000,
) -> Dict[str, Dict[str, float]]:
    """Run Log5 MC simulation to convert barthag → round advancement probs.

    Lighter than the backtest's 10K sims — 5K is sufficient for training features.
    """
    rng = np.random.default_rng(42)

    # Build bracket structure from seeds + regions
    region_teams: Dict[str, List[Tuple[int, str]]] = {}
    for tid, seed in seeds.items():
        reg = regions.get(tid, "Unknown")
        if reg not in region_teams:
            region_teams[reg] = []
        region_teams[reg].append((seed, tid))

    # Sort each region by seed matchup order
    matchup_order = {
        s: i for i, (s, _) in enumerate([(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)])
    }

    def _log5(ba: float, bb: float) -> float:
        num = ba * (1 - bb)
        den = ba * (1 - bb) + bb * (1 - ba)
        return num / den if den > 1e-12 else 0.5

    # Build first-round matchups per region
    all_matchups = []
    for reg in sorted(region_teams.keys()):
        teams = region_teams[reg]
        by_seed = {s: tid for s, tid in teams}
        for s1, s2 in [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]:
            t1 = by_seed.get(s1)
            t2 = by_seed.get(s2)
            if t1 and t2:
                all_matchups.append((t1, t2))

    # Count round advancements
    round_counts: Dict[str, Dict[str, int]] = {rn: {} for rn in ROUND_NAMES}

    for _ in range(n_sims):
        current = [t for pair in all_matchups for t in pair]
        for round_idx, round_name in enumerate(ROUND_NAMES):
            next_round = []
            for g in range(0, len(current), 2):
                if g + 1 >= len(current):
                    next_round.append(current[g])
                    continue
                t1, t2 = current[g], current[g + 1]
                ba = barthag.get(t1, 0.5)
                bb = barthag.get(t2, 0.5)
                p1 = _log5(ba, bb)
                winner = t1 if rng.random() < p1 else t2
                round_counts[round_name][winner] = round_counts[round_name].get(winner, 0) + 1
                next_round.append(winner)
            current = next_round

    # Convert counts to probabilities
    result: Dict[str, Dict[str, float]] = {}
    for tid in seeds:
        result[tid] = {}
        for rn in ROUND_NAMES:
            result[tid][rn] = round_counts[rn].get(tid, 0) / n_sims

    return result


def _load_pick_distribution(
    year: int,
    seeds: Dict[str, int],
    data_root: Path,
) -> Optional[Dict[str, Dict[str, float]]]:
    """Load ESPN public pick distribution for a year."""
    import json

    espn_path = data_root / "raw" / "historical" / f"espn_picks_{year}.json"
    if not espn_path.exists():
        # Try alternate location
        espn_path = data_root / "raw" / "historical_public_picks" / f"espn_picks_{year}.json"
    if not espn_path.exists():
        return None

    try:
        with open(espn_path) as f:
            raw = json.load(f)
        # ESPN picks format: dict with "teams" key containing {team_id: {round: pct}}
        # or a flat dict of {team_id: {round: pct}}
        if isinstance(raw, dict) and "teams" in raw:
            teams_data = raw["teams"]
        elif isinstance(raw, dict):
            # Check if this looks like a teams dict (values are dicts with round keys)
            # vs metadata dict (values are strings/ints)
            first_val = next(iter(raw.values()), None) if raw else None
            if isinstance(first_val, dict):
                teams_data = raw
            else:
                return None
        elif isinstance(raw, list):
            # List of {team_id, round_name, pick_pct} records
            dist: Dict[str, Dict[str, float]] = {}
            for entry in raw:
                tid = entry.get("team_id", entry.get("canonical_id", ""))
                rnd = entry.get("round_name", entry.get("round", ""))
                pct = entry.get("pick_pct", entry.get("pct", 0.5))
                if tid and rnd:
                    if tid not in dist:
                        dist[tid] = {}
                    dist[tid][rnd] = pct
            return dist if dist else None
        else:
            return None

        # Extract only round pick percentages, skip metadata fields like "seed"
        round_keys = set(ROUND_NAMES)
        dist = {}
        for tid, rounds in teams_data.items():
            if isinstance(rounds, dict):
                dist[tid] = {r: float(rounds.get(r, 0.0)) for r in round_keys if r in rounds}
        return dist if dist else None
    except (json.JSONDecodeError, KeyError):
        return None


def _load_context(
    year: int,
    seeds: Dict[str, int],
    data_root: Path,
) -> Dict[str, Dict[str, float]]:
    """Load context features (coach, momentum, talent, volatility)."""
    context: Dict[str, Dict[str, float]] = {}

    try:
        from src.prediction.coach_adj_probabilities import load_coach_experience

        context["coach_experience"] = load_coach_experience(year, seeds.keys(), data_root)
    except (FileNotFoundError, ImportError):
        context["coach_experience"] = {}

    try:
        from src.prediction.momentum_probabilities import load_team_momentum

        context["momentum"] = load_team_momentum(year, seeds.keys(), data_root)
    except (FileNotFoundError, ImportError):
        context["momentum"] = {}

    try:
        from src.prediction.roster_adj_probabilities import load_team_talent

        context["talent"] = load_team_talent(year, seeds.keys(), data_root)
    except (FileNotFoundError, ImportError):
        context["talent"] = {}

    try:
        from src.prediction.volatile_probabilities import load_team_volatility

        context["volatility"] = load_team_volatility(year, seeds.keys(), data_root)
    except (FileNotFoundError, ImportError):
        context["volatility"] = {}

    # S2: Tournament-specific box score features
    try:
        from src.prediction.tournament_features import load_four_factors

        ft_rate, to_margin = load_four_factors(year, seeds.keys(), data_root)
        context["four_factor_ft_rate"] = ft_rate
        context["four_factor_to_margin"] = to_margin
    except (FileNotFoundError, ImportError):
        context["four_factor_ft_rate"] = {}
        context["four_factor_to_margin"] = {}

    try:
        from src.prediction.tournament_features import load_conf_tourney_champ

        context["conf_tourney_champ"] = load_conf_tourney_champ(year, seeds.keys(), data_root)
    except (FileNotFoundError, ImportError):
        context["conf_tourney_champ"] = {}

    try:
        from src.prediction.tournament_features import load_ranking_momentum

        context["ranking_momentum"] = load_ranking_momentum(year, seeds.keys(), data_root)
    except (FileNotFoundError, ImportError):
        context["ranking_momentum"] = {}

    # Coach PASE (Performance Above Seed Expectation) from nishaanamin dataset.
    # Maps coach name → PASE score, then resolves to team_id via MTeamCoaches.csv.
    try:
        import csv as _csv
        import json as _json

        pase_path = data_root / "kaggle" / "coach_results.json"
        coaches_path = data_root / "kaggle" / "MTeamCoaches.csv"
        teams_path = data_root / "kaggle" / "MTeams.csv"

        if pase_path.exists() and coaches_path.exists() and teams_path.exists():
            with open(pase_path) as f:
                cr = _json.load(f)
            cols = cr["columns"]
            pase_idx = cols.index("pase")
            coach_name_idx = cols.index("coach")

            def _norm(s: str) -> str:
                return s.lower().replace(" ", "_").replace(".", "").replace("'", "")

            pase_by_norm = {_norm(row[coach_name_idx]): row[pase_idx] for row in cr["data"]}

            # Map TeamID → canonical via MTeams
            tid_to_name = {}
            with open(teams_path) as f:
                for row in _csv.DictReader(f):
                    tid_to_name[int(row["TeamID"])] = row["TeamName"]

            # Map team_id → coach PASE for this season
            from src.data.normalize import normalize_team_id

            coach_pase: Dict[str, float] = {}
            with open(coaches_path) as f:
                for row in _csv.DictReader(f):
                    if int(row["Season"]) != year:
                        continue
                    team_num = int(row["TeamID"])
                    coach_norm = _norm(row["CoachName"].replace("_", " "))
                    pase = pase_by_norm.get(coach_norm, 0.0)
                    team_name = tid_to_name.get(team_num, "")
                    if team_name:
                        canonical = normalize_team_id(team_name)
                        if canonical in seeds:
                            coach_pase[canonical] = float(pase) if pase else 0.0

            context["coach_pase"] = coach_pase
        else:
            context["coach_pase"] = {}
    except Exception:
        context["coach_pase"] = {}

    # ncsos, close_game_pct, elo_slope — computed from Kaggle regular season data
    try:
        from src.data.features.custom_ratings import (
            compute_close_game_pct,
            compute_elo_slope,
            compute_ncsos,
            ratings_to_canonical,
        )

        ncsos_raw = compute_ncsos(year, data_root)
        close_raw = compute_close_game_pct(year, data_root)
        slope_raw = compute_elo_slope(year, data_root)

        ncsos_can = ratings_to_canonical(ncsos_raw, data_root)
        close_can = ratings_to_canonical(close_raw, data_root)
        slope_can = ratings_to_canonical(slope_raw, data_root)

        context["ncsos"] = {t: v for t, v in ncsos_can.items() if t in seeds}
        context["close_game_pct"] = {t: v for t, v in close_can.items() if t in seeds}
        context["elo_slope"] = {t: v for t, v in slope_can.items() if t in seeds}
    except Exception:
        context["ncsos"] = {}
        context["close_game_pct"] = {}
        context["elo_slope"] = {}

    # Conference tournament depth
    try:
        from src.data.features.custom_ratings import (
            compute_conf_tourney_depth,
            ratings_to_canonical,
        )

        depth_raw = compute_conf_tourney_depth(year, data_root)
        depth_can = ratings_to_canonical(depth_raw, data_root)
        context["conf_tourney_depth"] = {t: v for t, v in depth_can.items() if t in seeds}
    except Exception:
        context["conf_tourney_depth"] = {}

    # Field chalk signal — same value for all teams (year-level feature).
    # The diff (t1 - t2) is always 0, making this a dead feature for the
    # GBM. Kept for backward compatibility; seed-scaled volatility variant
    # tested 2026-05-01 and regressed (4.4% vs 4.6% P(1st)), reverted.
    try:
        from src.data.features.custom_ratings import compute_field_chalk_signal

        chalk_sig = compute_field_chalk_signal(year, seeds, data_root)
        context["field_chalk_signal"] = {t: chalk_sig for t in seeds}
    except Exception:
        context["field_chalk_signal"] = {}

    # Injury BPR deduction (#13): sum of injured players' BPR × severity factor.
    # Deterministic, not stochastic noise. Requires roster JSON with player-level
    # box_plus_minus and injury_status. Falls back to empty dict when unavailable
    # (most historical years lack injury data → feature is 0 during LOYO training).
    try:
        import glob as _glob

        roster_paths = sorted(_glob.glob(str(data_root / "raw" / f"rosters_{year}.json")))
        if not roster_paths:
            roster_paths = sorted(_glob.glob(str(data_root / f"rosters_{year}.json")))
        if roster_paths:
            import json as _rjson

            with open(roster_paths[0]) as f:
                roster_data = _rjson.load(f)

            _INJURY_FACTORS = {
                "season_ending": 1.00,
                "out": 0.75,
                "doubtful": 0.60,
                "questionable": 0.50,
                "day-to-day": 0.25,
            }
            from src.data.normalize import normalize_team_id

            injury_deductions: Dict[str, float] = {}
            for team_entry in roster_data:
                team_name = team_entry.get("team_name", team_entry.get("team", ""))
                if not team_name:
                    continue
                canonical = normalize_team_id(team_name)
                if canonical not in seeds:
                    continue
                deduction = 0.0
                for player in team_entry.get("players", []):
                    status = str(player.get("injury_status", "healthy")).lower()
                    factor = _INJURY_FACTORS.get(status, 0.0)
                    if factor > 0:
                        bpr = float(player.get("box_plus_minus", 0.0))
                        deduction += abs(bpr) * factor
                injury_deductions[canonical] = deduction
            context["injury_bpr_deduction"] = injury_deductions
        else:
            context["injury_bpr_deduction"] = {}
    except Exception:
        context["injury_bpr_deduction"] = {}

    return context


def load_meta_context(
    year: int,
    seeds: Dict[str, int],
    data_root: Path = Path("data"),
) -> Dict[str, Dict[str, float]]:
    """Public wrapper for _load_context — use at inference time."""
    return _load_context(year, seeds, data_root)


def _extract_game_features_and_labels(
    games: List[dict],
    first_round_matchups: List[str],
    base_round_probs: Dict[str, Dict[str, Dict[str, float]]],
    pick_distribution: Optional[Dict[str, Dict[str, float]]],
    seeds: Dict[str, int],
    context: Dict[str, Dict[str, float]],
    vegas_r1: Optional[Dict[Tuple[str, str], float]] = None,
) -> List[Tuple[np.ndarray, int, str, str, str]]:
    """Extract features and labels from actual tournament games.

    Walks the actual bracket to determine real matchups at each round.
    Returns list of (feature_vector, label, round_name, team1, team2) tuples.
    label = 1 if team1 (top-listed) won, 0 if team2 won.
    """
    from src.simulation.pool_competition import actual_winners_by_round

    winners_by_rnd = actual_winners_by_round(games)
    results = []

    current_teams = list(first_round_matchups)
    game_idx = 0

    for round_idx, round_name in enumerate(ROUND_NAMES):
        round_winners = winners_by_rnd.get(round_name, set())
        next_round = []

        for g in range(0, len(current_teams), 2):
            if g + 1 >= len(current_teams):
                next_round.append(current_teams[g])
                continue

            t1, t2 = current_teams[g], current_teams[g + 1]

            # Determine actual winner
            if t1 in round_winners:
                label = 1
                winner = t1
            elif t2 in round_winners:
                label = 0
                winner = t2
            else:
                # Neither team advanced — skip (First Four casualty or data gap)
                next_round.append(t1)  # placeholder
                game_idx += 1
                continue

            feat = _game_features(
                t1,
                t2,
                round_name,
                round_idx,
                base_round_probs,
                pick_distribution,
                seeds,
                context,
                vegas_r1=vegas_r1,
            )
            results.append((feat, label, round_name, t1, t2))
            next_round.append(winner)
            game_idx += 1

        current_teams = next_round

    return results


def train_meta_selector(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    max_depth: int = 3,
    n_estimators: int = 50,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> Any:
    """Train a shallow LightGBM classifier for per-game bracket picks.

    Constrained to prevent overfitting on ~800 training samples:
    - max_depth=3 (8 leaves max)
    - n_estimators=50
    - min_child_samples=20
    - subsample=0.8
    """
    return _train_native_lightgbm_classifier(
        X,
        y,
        weights,
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
    )


class _NativeLightGBMClassifier:
    """Thin wrapper around ``lightgbm.train`` with a sklearn-like API.

    The local environment currently has a LightGBM/sklearn wrapper mismatch:
    newer sklearn removed ``force_all_finite`` while the installed LightGBM
    wrapper still passes it through. The native training API avoids that path
    while preserving deterministic tree behavior for the meta-selector.
    """

    def __init__(self, booster: Any):
        self.booster = booster
        self.feature_importances_ = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.booster.predict(X)
        return (np.asarray(probabilities) >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probabilities = np.clip(np.asarray(self.booster.predict(X), dtype=float), 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - probabilities, probabilities])


def _load_lightgbm_native() -> Any:
    import contextlib
    import importlib
    import io

    stderr_buffer = io.StringIO()
    with contextlib.redirect_stderr(stderr_buffer):
        return importlib.import_module("lightgbm")


def _train_native_lightgbm_classifier(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    max_depth: int,
    n_estimators: int,
    learning_rate: float,
    random_state: int,
) -> _NativeLightGBMClassifier:
    lgb = _load_lightgbm_native()

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "max_depth": max_depth,
        "num_leaves": min(2**max_depth, 16),
        "learning_rate": learning_rate,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "feature_fraction": 1.0,
        "seed": random_state,
        "feature_fraction_seed": random_state,
        "bagging_seed": random_state,
        "data_random_seed": random_state,
        "verbosity": -1,
    }
    dataset = lgb.Dataset(X, label=y, weight=weights, feature_name=feature_names(), free_raw_data=False)
    booster = lgb.train(params, dataset, num_boost_round=n_estimators)
    return _NativeLightGBMClassifier(booster)


def train_meta_selector_xgb(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    max_depth: int = 3,
    n_estimators: int = 50,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> Any:
    """Train XGBoost classifier with same constraints as LightGBM variant (S7)."""
    import xgboost as xgb

    model = xgb.XGBClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        min_child_weight=20,
        subsample=0.8,
        random_state=random_state,
        verbosity=0,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X, y, sample_weight=weights)
    return model


def tune_and_train_meta_selector(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    random_state: int = 42,
) -> Any:
    """Tune hyperparameters via weighted cross-validation, then train final model.

    Uses 5-fold CV on the training data to select best hyperparams from a
    focused grid. The final model is trained on ALL training data with the
    best params. This is safe because the training data itself is already
    walk-forward (years < test_year only).
    """
    from sklearn.model_selection import StratifiedKFold

    param_grid = [
        {"max_depth": 2, "n_estimators": 50, "learning_rate": 0.1},
        {"max_depth": 3, "n_estimators": 50, "learning_rate": 0.1},
        {"max_depth": 3, "n_estimators": 100, "learning_rate": 0.05},
        {"max_depth": 4, "n_estimators": 50, "learning_rate": 0.1},
        {"max_depth": 4, "n_estimators": 100, "learning_rate": 0.05},
        {"max_depth": 3, "n_estimators": 50, "learning_rate": 0.2},
    ]

    names = feature_names()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    best_score = -1.0
    best_params = param_grid[0]

    for params in param_grid:
        scores = []
        for train_idx, val_idx in skf.split(X, y):
            model = _train_native_lightgbm_classifier(
                X[train_idx],
                y[train_idx],
                weights[train_idx],
                max_depth=params["max_depth"],
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"],
                random_state=random_state,
            )
            # Score: weighted accuracy (same metric as training objective)
            preds = model.predict(X[val_idx])
            correct = (preds == y[val_idx]).astype(float)
            score = np.average(correct, weights=weights[val_idx])
            scores.append(score)
        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    logger.info("Best CV params: %s (score=%.4f)", best_params, best_score)

    # Train final model on all data with best params
    return train_meta_selector(
        X,
        y,
        weights,
        max_depth=best_params["max_depth"],
        n_estimators=best_params["n_estimators"],
        learning_rate=best_params["learning_rate"],
        random_state=random_state,
    )


def train_meta_selector_lr(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    C: float = 100.0,
    random_state: int = 42,
) -> Any:
    """Train Logistic Regression meta-selector with explicit feature interactions.

    Kaggle 3rd place (Brier 0.1160, #3/3485) found LR outperforms GBM on
    tournament-sized data (~650 rows): LR CV Brier 0.124 vs XGB 0.157.

    Adds pairwise interactions between high-signal features to give LR
    non-linear capacity without GBM's overfitting risk on small data.

    The trained model exposes .predict() compatible with build_trained_bracket().
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    # Feature interaction indices — these are the high-signal features where
    # pairwise interactions capture non-linear relationships LR can't learn.
    # Inspired by Kaggle 3rd: seed×massey, seed×prob, consensus×disagreement.
    names = feature_names()

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "interactions",
                PolynomialFeatures(
                    degree=2,
                    interaction_only=True,
                    include_bias=False,
                ),
            ),
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=C,
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )

    pipeline.fit(X, y, lr__sample_weight=weights)
    return pipeline


class _MarginToClassifier:
    """Wraps a margin regressor to expose .predict() returning 0/1 labels.

    Trains on point differential (team1_score - team2_score) and converts
    to binary picks via sign of predicted margin. Compatible with
    build_trained_bracket() which calls model.predict().

    Kaggle 10th place (2026) found that predicting margin instead of
    binary win/loss produces a richer training signal — a 20-point
    blowout teaches more than a 1-point squeaker.
    """

    def __init__(self, regressor: Any):
        self.regressor = regressor

    def predict(self, X) -> np.ndarray:
        """Predict binary labels (1 = team1 wins) from margin predictions."""
        margins = self.regressor.predict(X)
        return (np.asarray(margins) > 0).astype(int)

    def predict_margin(self, X) -> np.ndarray:
        """Return raw margin predictions (for calibration/analysis)."""
        return np.asarray(self.regressor.predict(X))


def build_margin_training_data(
    train_years: Sequence[int],
    data_root: Path = Path("data"),
    augment: bool = True,
    drop_chalk: bool = True,
    use_vegas_r1: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y_margin, weights) for margin regression training.

    Same as build_training_data but y is the point differential
    (team1_score - team2_score) instead of binary 0/1.

    Weight = round_pts (ESPN scoring: 10/20/40/80/160/320).
    """
    import json

    all_X, all_y, all_w = [], [], []

    for year in train_years:
        try:
            brp, pick_dist, seeds, context, first_round, games = _load_year_data(year, data_root)
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
            logger.debug("Skipping training year %d: %s", year, exc)
            continue

        vr1 = (load_vegas_r1(year, data_root) or {}) if use_vegas_r1 else None

        # Build a score lookup from games
        score_lookup: Dict[Tuple[str, str, str], Tuple[int, int]] = {}
        for g in games:
            t1 = g["team1_id"]
            t2 = g["team2_id"]
            rn = g.get("round_name", "")
            s1 = g.get("team1_score", 0)
            s2 = g.get("team2_score", 0)
            if s1 and s2:
                score_lookup[(t1, t2, rn)] = (s1, s2)
                score_lookup[(t2, t1, rn)] = (s2, s1)

        game_features_and_labels = _extract_game_features_and_labels(
            games, first_round, brp, pick_dist, seeds, context, vegas_r1=vr1
        )

        for feat_vec, label, round_name, team1, team2 in game_features_and_labels:
            pts = ESPN_SCORING.get(round_name, 10)

            if drop_chalk and round_name == "R64":
                s1 = seeds.get(team1, 8)
                s2 = seeds.get(team2, 8)
                if min(s1, s2) <= 2:
                    continue

            # Get margin (team1_score - team2_score)
            scores = score_lookup.get((team1, team2, round_name))
            if scores is None:
                # Fallback: infer from label
                margin = 5.0 if label == 1 else -5.0
            else:
                margin = float(scores[0] - scores[1])

            weight = float(pts)
            all_X.append(feat_vec)
            all_y.append(margin)
            all_w.append(weight)

            if augment:
                feat_swapped = _game_features(
                    team2,
                    team1,
                    round_name,
                    ROUND_NAMES.index(round_name),
                    brp,
                    pick_dist,
                    seeds,
                    context,
                    vegas_r1=vr1,
                )
                all_X.append(feat_swapped)
                all_y.append(-margin)
                all_w.append(weight)

    X = np.array(all_X, dtype=np.float64)
    y = np.array(all_y, dtype=np.float64)
    w = np.array(all_w, dtype=np.float64)
    return X, y, w


def train_meta_selector_margin(
    X: np.ndarray,
    y_margin: np.ndarray,
    weights: np.ndarray,
    max_depth: int = 3,
    n_estimators: int = 100,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> _MarginToClassifier:
    """Train XGBoost regressor on point differential, return classifier wrapper.

    Kaggle 10th place found regression on margin produces richer signal
    than binary classification on the same ~650 rows. The wrapper's
    .predict() returns 0/1 (compatible with build_trained_bracket).

    Args:
        y_margin: Point differential (team1_score - team2_score).
    """
    import xgboost as xgb

    reg = xgb.XGBRegressor(
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        min_child_weight=20,
        subsample=0.8,
        random_state=random_state,
        verbosity=0,
    )
    reg.fit(X, y_margin, sample_weight=weights)
    return _MarginToClassifier(reg)


class _MultiSeedEnsemble:
    """Ensemble of N models trained with different random seeds.

    Kaggle 10th place ran 6 seeds, each with its own feature subset and model.
    Here we keep features fixed but vary the random seed, providing diversity
    in the stochastic aspects of boosting (subsampling, split selection).

    .predict() returns majority vote across all models.
    """

    def __init__(self, models: List[Any]):
        self.models = models

    def predict(self, X) -> np.ndarray:
        """Majority vote across all models."""
        preds = np.array([m.predict(X) for m in self.models])
        # Sum predictions (each is 0 or 1), threshold at N/2
        return (preds.sum(axis=0) > len(self.models) / 2).astype(int)


def train_multi_seed_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    n_seeds: int = 6,
    trainer_fn: Optional[Callable] = None,
) -> _MultiSeedEnsemble:
    """Train N models with different random seeds, return ensemble.

    Args:
        trainer_fn: Model training function with signature (X, y, w, random_state=int).
            Defaults to train_meta_selector (LightGBM).
        n_seeds: Number of seeds to use (default 6, matching Kaggle 10th place).
    """
    if trainer_fn is None:
        trainer_fn = train_meta_selector

    models = []
    for i in range(n_seeds):
        model = trainer_fn(X, y, weights, random_state=42 + i)
        models.append(model)

    return _MultiSeedEnsemble(models)


class _FeatureMaskedModel:
    """Wraps a model that was trained with some features zeroed.

    At predict time, zeroes the same features so train/inference match.
    """

    def __init__(self, model: Any, keep_mask: np.ndarray):
        self.model = model
        self.keep_mask = keep_mask  # boolean, True = feature is active

    def predict(self, X) -> np.ndarray:
        import pandas as pd

        if isinstance(X, pd.DataFrame):
            X_masked = X.copy()
            drop_cols = [c for i, c in enumerate(X.columns) if not self.keep_mask[i]]
            X_masked[drop_cols] = 0.0
        else:
            X_masked = np.array(X, copy=True)
            X_masked[:, ~self.keep_mask] = 0.0
        return self.model.predict(X_masked)


def train_meta_selector_backward_elim(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    random_state: int = 42,
    keep_ratio: float = 0.5,
) -> _FeatureMaskedModel:
    """Importance-based feature elimination: train, drop low-importance, retrain.

    Kaggle 3rd place found aggressive pruning was the single biggest Brier gain.
    This uses LightGBM feature importance to identify and drop the bottom half
    of features, then retrains on the reduced set. Fast (~2 model trains).

    Returns a _FeatureMaskedModel that zeroes dropped features at inference.
    """
    names = feature_names()
    n_feat = X.shape[1]

    # Train initial model to get importance
    scout = train_meta_selector(X, y, weights, random_state=random_state)
    importances = scout.feature_importances_

    # Keep top features by importance
    n_keep = max(3, int(n_feat * keep_ratio))
    threshold = np.sort(importances)[-n_keep]
    keep = importances >= threshold
    # Tie-break: if too many at threshold, keep exactly n_keep
    if keep.sum() > n_keep:
        at_thresh = np.where(importances == threshold)[0]
        excess = int(keep.sum()) - n_keep
        for idx in at_thresh[:excess]:
            keep[idx] = False

    # Retrain on selected features only
    X_final = X.copy()
    X_final[:, ~keep] = 0.0
    final_model = train_meta_selector(X_final, y, weights, random_state=random_state)
    return _FeatureMaskedModel(final_model, keep)


def train_meta_selector_minimal(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    random_state: int = 42,
) -> _FeatureMaskedModel:
    """3-feature minimalism: keep only seed prob, SRS prob, and seed values.

    Kaggle 1st place won with seed_diff + quality_wins_diff + custom_rating.
    We approximate with: p_seed (seed win prob), p_srs (SRS rating prob),
    seed_t1, seed_t2 (raw seeds for diff signal), and round_index.
    """
    names = feature_names()
    keep_names = {"p_seed", "p_srs", "seed_t1", "seed_t2", "round_index"}
    keep = np.array([n in keep_names for n in names], dtype=bool)

    X_min = X.copy()
    X_min[:, ~keep] = 0.0
    model = train_meta_selector(X_min, y, weights, max_depth=2, n_estimators=30, random_state=random_state)
    return _FeatureMaskedModel(model, keep)


def build_trained_bracket(
    first_round_matchups: List[str],
    base_round_probs: Dict[str, Dict[str, Dict[str, float]]],
    pick_distribution: Optional[Dict[str, Dict[str, float]]],
    seeds: Dict[str, int],
    context: Optional[Dict[str, Dict[str, float]]],
    model: Any,
    vegas_r1: Optional[Dict[Tuple[str, str], float]] = None,
) -> np.ndarray:
    """Build one deterministic bracket using the trained meta-selector.

    For each game: assemble feature vector → model.predict() → pick winner.
    Path-consistent (winners advance to next round).
    Returns (63,) boolean array (True = team1 picked).
    """
    import pandas as pd

    _feat_names = feature_names(include_vegas_r1=vegas_r1 is not None)
    bracket = np.zeros(63, dtype=bool)
    current_teams = list(first_round_matchups)
    game_idx = 0

    for round_idx in range(6):
        round_name = ROUND_NAMES[round_idx]
        next_round = []

        for g in range(0, len(current_teams), 2):
            if g + 1 >= len(current_teams):
                next_round.append(current_teams[g])
                continue

            t1, t2 = current_teams[g], current_teams[g + 1]
            feat = _game_features(
                t1,
                t2,
                round_name,
                round_idx,
                base_round_probs,
                pick_distribution,
                seeds,
                context,
                vegas_r1=vegas_r1,
            )
            feat_df = pd.DataFrame(feat.reshape(1, -1), columns=_feat_names)
            pred = model.predict(feat_df)[0]
            pick_t1 = pred == 1
            bracket[game_idx] = pick_t1
            winner = t1 if pick_t1 else t2
            next_round.append(winner)
            game_idx += 1

        current_teams = next_round

    return bracket


def build_gbm_round_probs(
    first_round_matchups: List[str],
    base_round_probs: Dict[str, Dict[str, Dict[str, float]]],
    pick_distribution: Optional[Dict[str, Dict[str, float]]],
    seeds: Dict[str, int],
    context: Optional[Dict[str, Dict[str, float]]],
    model: Any,
    n_sims: int = 2000,
    rng_seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Monte Carlo simulation using GBM predict_proba → round probabilities.

    Simulates the tournament n_sims times. For each game, uses the model's
    predicted P(team1 wins) to sample the winner. Counts how often each
    team reaches each round. Returns {team_id: {round_name: probability}}
    in the same format as torvik round probs, compatible with construct_bracket.
    """
    import pandas as pd

    rng = np.random.default_rng(rng_seed)
    _feat_names = feature_names()
    reach_counts: Dict[str, Dict[str, int]] = {}

    for team in first_round_matchups:
        reach_counts[team] = {rn: 0 for rn in ROUND_NAMES}

    for _ in range(n_sims):
        current_teams = list(first_round_matchups)
        for round_idx in range(6):
            round_name = ROUND_NAMES[round_idx]
            next_round = []
            for g in range(0, len(current_teams), 2):
                if g + 1 >= len(current_teams):
                    next_round.append(current_teams[g])
                    continue
                t1, t2 = current_teams[g], current_teams[g + 1]
                feat = _game_features(
                    t1,
                    t2,
                    round_name,
                    round_idx,
                    base_round_probs,
                    pick_distribution,
                    seeds,
                    context,
                )
                feat_df = pd.DataFrame(feat.reshape(1, -1), columns=_feat_names)
                try:
                    proba = model.predict_proba(feat_df)[0]
                    p_t1 = proba[1] if len(proba) > 1 else 0.5
                except (AttributeError, IndexError):
                    pred = model.predict(feat_df)[0]
                    p_t1 = 0.7 if pred == 1 else 0.3
                winner = t1 if rng.random() < p_t1 else t2
                reach_counts[winner][round_name] = reach_counts.get(winner, {}).get(round_name, 0) + 1
                next_round.append(winner)
            current_teams = next_round

    # Normalize to probabilities
    round_probs: Dict[str, Dict[str, float]] = {}
    for team in first_round_matchups:
        round_probs[team] = {}
        for rn in ROUND_NAMES:
            round_probs[team][rn] = reach_counts[team][rn] / n_sims

    return round_probs


# ---------------------------------------------------------------------------
# S1: Champion-First Two-Phase Construction
# ---------------------------------------------------------------------------

GAMES_PER_ROUND = (32, 16, 8, 4, 2, 1)  # R64 through CHAMP


def _compute_team_path(
    team_id: str,
    first_round_matchups: List[str],
) -> List[Tuple[int, int]]:
    """Compute the game indices a team must win to reach the championship.

    Returns list of (game_index, round_index) pairs, length 6.
    Raises ValueError if team_id is not in first_round_matchups.
    """
    if team_id not in first_round_matchups:
        raise ValueError(f"{team_id} not in first_round_matchups")

    pos = first_round_matchups.index(team_id)
    slot = pos
    path = []
    offset = 0

    for r in range(6):
        game_in_round = slot // 2
        game_idx = offset + game_in_round
        path.append((game_idx, r))
        offset += GAMES_PER_ROUND[r]
        slot = game_in_round  # advance to next round's slot

    return path


def _rank_champion_candidates(
    base_round_probs: Dict[str, Dict[str, Dict[str, float]]],
    first_round_matchups: List[str],
    top_n: int = 6,
) -> List[Tuple[str, float]]:
    """Return top-N teams by mean CHAMP probability across available bases."""
    team_set = set(first_round_matchups)
    scores: Dict[str, List[float]] = {}

    for base, team_rounds in base_round_probs.items():
        for team_id, rounds in team_rounds.items():
            if team_id not in team_set:
                continue
            champ_prob = rounds.get("CHAMP", 0.0)
            if champ_prob > 0:
                scores.setdefault(team_id, []).append(champ_prob)

    ranked = [(tid, sum(probs) / len(probs)) for tid, probs in scores.items() if probs]
    ranked.sort(key=lambda x: -x[1])
    return ranked[:top_n]


def build_champion_locked_bracket(
    champion: str,
    first_round_matchups: List[str],
    base_round_probs: Dict[str, Dict[str, Dict[str, float]]],
    pick_distribution: Optional[Dict[str, Dict[str, float]]],
    seeds: Dict[str, int],
    context: Optional[Dict[str, Dict[str, float]]],
    model: Any,
) -> np.ndarray:
    """Build one deterministic bracket with a specific champion locked.

    The champion wins every game on their R64-to-CHAMP path.
    All other games decided by the GBM model.
    Path-consistent by construction.
    Returns (63,) boolean array.
    """
    import pandas as pd

    path = _compute_team_path(champion, first_round_matchups)
    locked_game_indices = {gi for gi, _ in path}

    _feat_names = feature_names()
    bracket = np.zeros(63, dtype=bool)
    current_teams = list(first_round_matchups)
    game_idx = 0

    for round_idx in range(6):
        round_name = ROUND_NAMES[round_idx]
        next_round = []

        for g in range(0, len(current_teams), 2):
            if g + 1 >= len(current_teams):
                next_round.append(current_teams[g])
                continue

            t1, t2 = current_teams[g], current_teams[g + 1]

            if game_idx in locked_game_indices:
                pick_t1 = t1 == champion
            else:
                feat = _game_features(
                    t1,
                    t2,
                    round_name,
                    round_idx,
                    base_round_probs,
                    pick_distribution,
                    seeds,
                    context,
                )
                feat_df = pd.DataFrame(feat.reshape(1, -1), columns=_feat_names)
                pred = model.predict(feat_df)[0]
                pick_t1 = pred == 1

            bracket[game_idx] = pick_t1
            winner = t1 if pick_t1 else t2
            next_round.append(winner)
            game_idx += 1

        current_teams = next_round

    return bracket


def build_champion_first_brackets(
    first_round_matchups: List[str],
    base_round_probs: Dict[str, Dict[str, Dict[str, float]]],
    pick_distribution: Optional[Dict[str, Dict[str, float]]],
    seeds: Dict[str, int],
    context: Optional[Dict[str, Dict[str, float]]],
    model: Any,
    top_n: int = 6,
) -> Tuple[np.ndarray, List[str]]:
    """Build champion-first brackets for top-N champion candidates.

    Returns:
        brackets: (N, 63) boolean array, one bracket per candidate.
        champions: list of team_ids (the locked champion per bracket).
    """
    candidates = _rank_champion_candidates(base_round_probs, first_round_matchups, top_n)

    brackets = []
    champion_ids = []
    for team_id, _prob in candidates:
        b = build_champion_locked_bracket(
            team_id,
            first_round_matchups,
            base_round_probs,
            pick_distribution,
            seeds,
            context,
            model,
        )
        brackets.append(b)
        champion_ids.append(team_id)

    return np.array(brackets, dtype=bool), champion_ids
