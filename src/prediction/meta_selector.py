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
)

# Context features appended after base probability features.
CONTEXT_KEYS = ("coach_experience", "momentum", "talent", "volatility")


def _pairwise_prob(
    team1: str,
    team2: str,
    round_name: str,
    round_probs: Dict[str, Dict[str, float]],
) -> float:
    """Normalize marginal advancement probs to head-to-head P(team1 wins)."""
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

    Total: len(BASE_FEATURE_ORDER) + 2 + 2 + 1 + 1 + 1 + 1 + 1 + 1 + len(CONTEXT_KEYS)
         = 12 + 2 + 2 + 1 + 1 + 1 + 1 + 1 + 1 + 4 = 26
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

    return np.array(feats, dtype=np.float64)


def feature_names() -> List[str]:
    """Return ordered list of feature names matching _game_features output."""
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
    return names


def n_features() -> int:
    """Number of features per game."""
    return len(feature_names())


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y, weights) for meta-selector training.

    For each tournament game in each training year:
        X:       feature vector (all base probs + ESPN picks + context)
        y:       1 if team1 actually won, 0 if team2 won
        weight:  round_pts (ESPN scoring, no contrarian adjustment)

    Args:
        augment: If True, also include the swapped (team2, team1) view of
            each game with flipped label. Doubles training data and removes
            positional bias (team1 is always the higher seed in R64).
        drop_chalk: If True, exclude R64 games where min seed <= 2
            (1v16 and 2v15 matchups). These are ~96% chalk and add noise.

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

        # Walk the actual tournament to get real matchups per round
        game_features_and_labels = _extract_game_features_and_labels(games, first_round, brp, pick_dist, seeds, context)

        for feat_vec, label, round_name, team1, team2 in game_features_and_labels:
            pts = ESPN_SCORING.get(round_name, 10)

            # Drop chalk: skip 1v16 and 2v15 R64 games
            if drop_chalk and round_name == "R64":
                s1 = seeds.get(team1, 8)
                s2 = seeds.get(team2, 8)
                if min(s1, s2) <= 2:
                    continue

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

    # Seeds
    with open(hist_dir / f"tournament_seeds_{year}.json") as f:
        seeds_file = json.load(f)
    seeds_raw = seeds_file["teams"] if isinstance(seeds_file, dict) and "teams" in seeds_file else seeds_file
    seeds = {t["team_id"]: t["seed"] for t in seeds_raw}

    # Tournament results (ground truth)
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

    return brp


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

    return context


def _extract_game_features_and_labels(
    games: List[dict],
    first_round_matchups: List[str],
    base_round_probs: Dict[str, Dict[str, Dict[str, float]]],
    pick_distribution: Optional[Dict[str, Dict[str, float]]],
    seeds: Dict[str, int],
    context: Dict[str, Dict[str, float]],
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
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        min_child_samples=20,
        subsample=0.8,
        random_state=random_state,
        verbosity=-1,
    )
    names = feature_names()
    model.fit(X, y, sample_weight=weights, feature_name=names)
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
    import lightgbm as lgb
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
            model = lgb.LGBMClassifier(
                max_depth=params["max_depth"],
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"],
                min_child_samples=20,
                subsample=0.8,
                random_state=random_state,
                verbosity=-1,
            )
            model.fit(
                X[train_idx],
                y[train_idx],
                sample_weight=weights[train_idx],
                feature_name=names,
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


def build_trained_bracket(
    first_round_matchups: List[str],
    base_round_probs: Dict[str, Dict[str, Dict[str, float]]],
    pick_distribution: Optional[Dict[str, Dict[str, float]]],
    seeds: Dict[str, int],
    context: Optional[Dict[str, Dict[str, float]]],
    model: Any,
) -> np.ndarray:
    """Build one deterministic bracket using the trained meta-selector.

    For each game: assemble feature vector → model.predict() → pick winner.
    Path-consistent (winners advance to next round).
    Returns (63,) boolean array (True = team1 picked).
    """
    import pandas as pd

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
