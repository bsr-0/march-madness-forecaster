"""Custom team rating systems computed from regular-season game results.

Implements three rating systems used by Kaggle 3rd and 10th place solutions:

1. **Colley Matrix** — Solves (2I + C)r = 1 + (w-l)/2 for bias-free rankings
   from W/L record weighted by opponent strength. Closed-form via linear solve.

2. **SRS (Simple Rating System)** — rating = avg_margin + mean(opponent_ratings),
   iterated to convergence. Captures both scoring dominance and schedule strength.

3. **GLM Quality (Raddar method)** — MLE team strength from game graph treating
   each outcome as Bernoulli trial. Logistic regression with team fixed effects.

All three take the same input: a season's regular-season game results from Kaggle
MRegularSeasonCompactResults.csv (or DetailedResults). Output is a dict of
{kaggle_team_id: rating} that can be converted to canonical IDs for pipeline use.

Point-in-time safe: only uses games with DayNum < cutoff (default 133 = Selection
Sunday). No tournament game leakage.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Selection Sunday is typically DayNum 133 in the Kaggle calendar.
DEFAULT_CUTOFF_DAY = 133


def _load_season_games(
    season: int,
    data_root: Path = Path("data"),
    cutoff_day: int = DEFAULT_CUTOFF_DAY,
) -> List[Tuple[int, int, int, int]]:
    """Load regular-season games as (winner_id, loser_id, w_score, l_score).

    Only includes games with DayNum < cutoff_day to prevent tournament leakage.
    Uses CompactResults (smaller, has scores) with DetailedResults as fallback.
    """
    for filename in ("MRegularSeasonCompactResults.csv", "MRegularSeasonDetailedResults.csv"):
        path = data_root / "kaggle" / filename
        if not path.exists():
            continue
        games = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["Season"]) != season:
                    continue
                if int(row["DayNum"]) >= cutoff_day:
                    continue
                games.append(
                    (
                        int(row["WTeamID"]),
                        int(row["LTeamID"]),
                        int(row["WScore"]),
                        int(row["LScore"]),
                    )
                )
        if games:
            return games
    return []


def _get_all_teams(games: List[Tuple[int, int, int, int]]) -> List[int]:
    """Extract sorted unique team IDs from games."""
    teams = set()
    for w, los, _, _ in games:
        teams.add(w)
        teams.add(los)
    return sorted(teams)


# ---------------------------------------------------------------------------
# Colley Matrix
# ---------------------------------------------------------------------------


def compute_colley_ratings(
    season: int,
    data_root: Path = Path("data"),
    cutoff_day: int = DEFAULT_CUTOFF_DAY,
) -> Dict[int, float]:
    """Compute Colley Matrix rankings for a season.

    Solves (2I + C)r = 1 + (w - l) / 2 where C is the connectivity matrix.

    Returns {kaggle_team_id: colley_rating}. Higher = better.
    """
    games = _load_season_games(season, data_root, cutoff_day)
    if not games:
        return {}

    teams = _get_all_teams(games)
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    # Build the Colley system: (2I + C)r = b
    # C[i][j] = number of games between team i and team j
    # b[i] = 1 + (wins_i - losses_i) / 2
    C = np.zeros((n, n))
    wins = np.zeros(n)
    losses = np.zeros(n)

    for w, los, _, _ in games:
        wi, li = idx[w], idx[los]
        C[wi][li] += 1
        C[li][wi] += 1
        wins[wi] += 1
        losses[li] += 1

    # Colley uses a negative off-diagonal: A[i][j] = -n_ij, with the diagonal
    # carrying 2 + t_i where t_i is team i's total games.
    A = np.diag(2.0 + wins + losses) - C

    b = 1.0 + (wins - losses) / 2.0

    try:
        ratings = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        logger.warning("Colley matrix singular for season %d, using least squares", season)
        ratings, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    return {teams[i]: float(ratings[i]) for i in range(n)}


# ---------------------------------------------------------------------------
# SRS (Simple Rating System)
# ---------------------------------------------------------------------------


def compute_srs_ratings(
    season: int,
    data_root: Path = Path("data"),
    cutoff_day: int = DEFAULT_CUTOFF_DAY,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Dict[int, float]:
    """Compute Simple Rating System rankings for a season.

    SRS: rating_i = avg_margin_i + mean(opponent_ratings)
    Iterated until convergence. Mean-centered each iteration.

    Returns {kaggle_team_id: srs_rating}. Higher = better.
    """
    games = _load_season_games(season, data_root, cutoff_day)
    if not games:
        return {}

    teams = _get_all_teams(games)
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    # Build per-team stats
    total_margin = np.zeros(n)
    game_count = np.zeros(n)
    opponents: List[List[int]] = [[] for _ in range(n)]

    for w, los, ws, ls in games:
        wi, loi = idx[w], idx[los]
        margin = ws - ls
        total_margin[wi] += margin
        total_margin[loi] -= margin
        game_count[wi] += 1
        game_count[loi] += 1
        opponents[wi].append(loi)
        opponents[loi].append(wi)

    avg_margin = np.where(game_count > 0, total_margin / game_count, 0.0)
    ratings = avg_margin.copy()

    for iteration in range(max_iter):
        new_ratings = np.zeros(n)
        for i in range(n):
            if not opponents[i]:
                new_ratings[i] = avg_margin[i]
                continue
            opp_avg = np.mean([ratings[j] for j in opponents[i]])
            new_ratings[i] = avg_margin[i] + opp_avg

        # Mean-center
        new_ratings -= np.mean(new_ratings)

        if np.max(np.abs(new_ratings - ratings)) < tol:
            ratings = new_ratings
            break
        ratings = new_ratings

    return {teams[i]: float(ratings[i]) for i in range(n)}


# ---------------------------------------------------------------------------
# GLM Quality (Raddar method)
# ---------------------------------------------------------------------------


def compute_glm_quality(
    season: int,
    data_root: Path = Path("data"),
    cutoff_day: int = DEFAULT_CUTOFF_DAY,
) -> Dict[int, float]:
    """Compute GLM team quality via logistic regression on game outcomes.

    Fits: P(team_i beats team_j) = sigmoid(strength_i - strength_j)
    using team fixed effects. MLE via iterative reweighted least squares
    (sklearn LogisticRegression under the hood).

    Returns {kaggle_team_id: glm_quality}. Higher = better.
    """
    games = _load_season_games(season, data_root, cutoff_day)
    if not games:
        return {}

    teams = _get_all_teams(games)
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    # Build the design matrix with BOTH orientations of every game.
    #
    # This previously wrote one row per game with y = 1 throughout, because the
    # winner is always encoded as +1. A single-class target cannot be fit:
    # LogisticRegression raised "needs samples of at least 2 classes" on every
    # season, the except branch below caught it, and this function silently
    # returned average margin under a GLM name -- for every caller, every
    # season, since it was written. It failed loudly into a log nobody read and
    # quietly everywhere else.
    #
    # Mirroring each game to (-x, 1 - y) supplies the second class and is the
    # same symmetric augmentation build_training_matrix documents for the
    # browser fit: if A beating B implies B losing to A, both statements belong
    # in the design. It also forces the fit through the origin, so no team
    # gains strength from the arbitrary choice of which side is written first.
    m = len(games)
    X = np.zeros((2 * m, n))
    y = np.concatenate([np.ones(m), np.zeros(m)])

    for k, (w, los, _, _) in enumerate(games):
        X[k, idx[w]] = 1.0
        X[k, idx[los]] = -1.0
        X[m + k, idx[w]] = -1.0
        X[m + k, idx[los]] = 1.0

    # Fit logistic regression with no intercept (team effects only)
    try:
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(
            fit_intercept=False,
            C=1.0,
            solver="lbfgs",
            max_iter=500,
            penalty="l2",
        )
        lr.fit(X, y)
        strengths = lr.coef_[0]
    except Exception as exc:
        logger.warning("GLM quality failed for season %d: %s, falling back to margin", season, exc)
        # Fallback: use average margin as proxy
        margin_sum = np.zeros(n)
        margin_count = np.zeros(n)
        for w, los, ws, ls in games:
            margin_sum[idx[w]] += ws - ls
            margin_sum[idx[los]] -= ws - ls
            margin_count[idx[w]] += 1
            margin_count[idx[los]] += 1
        strengths = np.where(margin_count > 0, margin_sum / margin_count, 0.0)

    return {teams[i]: float(strengths[i]) for i in range(n)}


# ---------------------------------------------------------------------------
# Non-Conference SOS (ncsos)
# ---------------------------------------------------------------------------


def compute_ncsos(
    season: int,
    data_root: Path = Path("data"),
    cutoff_day: int = DEFAULT_CUTOFF_DAY,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> Dict[int, float]:
    """Compute non-conference strength of schedule.

    Like SRS but only using games where opponent is in a different conference.
    Captures whether a team chose to play tough out-of-conference games.

    Returns {kaggle_team_id: ncsos}. Higher = tougher non-conf schedule.
    """
    games = _load_season_games(season, data_root, cutoff_day)
    if not games:
        return {}

    # Load conference memberships
    conf_path = data_root / "kaggle" / "MTeamConferences.csv"
    if not conf_path.exists():
        return {}

    team_conf: Dict[int, str] = {}
    with open(conf_path) as f:
        for row in csv.DictReader(f):
            if int(row["Season"]) == season:
                team_conf[int(row["TeamID"])] = row["ConfAbbrev"]

    # Filter to non-conference games only
    nc_games = []
    for w, los, ws, ls in games:
        if team_conf.get(w) != team_conf.get(los):
            nc_games.append((w, los, ws, ls))

    if not nc_games:
        return {}

    # Compute SRS on non-conference games
    teams = _get_all_teams(nc_games)
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    total_margin = np.zeros(n)
    game_count = np.zeros(n)
    opponents: List[List[int]] = [[] for _ in range(n)]

    for w, los, ws, ls in nc_games:
        if w not in idx or los not in idx:
            continue
        wi, loi = idx[w], idx[los]
        margin = ws - ls
        total_margin[wi] += margin
        total_margin[loi] -= margin
        game_count[wi] += 1
        game_count[loi] += 1
        opponents[wi].append(loi)
        opponents[loi].append(wi)

    avg_margin = np.where(game_count > 0, total_margin / game_count, 0.0)
    ratings = avg_margin.copy()

    for _ in range(max_iter):
        new_ratings = np.zeros(n)
        for i in range(n):
            if not opponents[i]:
                new_ratings[i] = avg_margin[i]
                continue
            opp_avg = np.mean([ratings[j] for j in opponents[i]])
            new_ratings[i] = avg_margin[i] + opp_avg
        new_ratings -= np.mean(new_ratings)
        if np.max(np.abs(new_ratings - ratings)) < tol:
            ratings = new_ratings
            break
        ratings = new_ratings

    return {teams[i]: float(ratings[i]) for i in range(n)}


# ---------------------------------------------------------------------------
# Close-Game Win Percentage
# ---------------------------------------------------------------------------


def compute_close_game_pct(
    season: int,
    data_root: Path = Path("data"),
    cutoff_day: int = DEFAULT_CUTOFF_DAY,
    margin_threshold: int = 7,
) -> Dict[int, float]:
    """Compute win percentage in close games (margin <= threshold).

    Returns {kaggle_team_id: close_game_win_pct}. Range [0, 1].
    Teams with no close games get 0.5 (neutral).
    """
    games = _load_season_games(season, data_root, cutoff_day)
    if not games:
        return {}

    close_wins: Dict[int, int] = {}
    close_games: Dict[int, int] = {}

    for w, los, ws, ls in games:
        margin = ws - ls
        if margin <= margin_threshold:
            close_wins[w] = close_wins.get(w, 0) + 1
            close_games[w] = close_games.get(w, 0) + 1
            close_games[los] = close_games.get(los, 0) + 1

    result = {}
    all_teams = _get_all_teams(games)
    for t in all_teams:
        n_close = close_games.get(t, 0)
        if n_close > 0:
            result[t] = close_wins.get(t, 0) / n_close
        else:
            result[t] = 0.5
    return result


# ---------------------------------------------------------------------------
# Elo Slope (within-season trend)
# ---------------------------------------------------------------------------


def compute_elo_slope(
    season: int,
    data_root: Path = Path("data"),
    cutoff_day: int = DEFAULT_CUTOFF_DAY,
    k_factor: int = 32,
    home_advantage: float = 100.0,
    last_n: int = 10,
) -> Dict[int, float]:
    """Compute Elo rating trend (slope) over the last N games of the season.

    Runs a full Elo system for the season, then fits a linear regression
    to each team's last N Elo snapshots. Positive slope = peaking,
    negative = declining.

    Returns {kaggle_team_id: elo_slope}. Units: Elo points per game.
    """
    games = _load_season_games(season, data_root, cutoff_day)
    if not games:
        return {}

    # Sort games by DayNum (they come from CSV in order, but be safe)
    # Games tuple is (w, l, ws, ls) — we need DayNum too. Re-load with DayNum.
    games_with_day = _load_season_games_with_day(season, data_root, cutoff_day)

    # Initialize Elo at 1500 for all teams
    elo: Dict[int, float] = {}
    elo_history: Dict[int, List[float]] = {}

    for day, w, los, ws, ls in games_with_day:
        if w not in elo:
            elo[w] = 1500.0
            elo_history[w] = []
        if los not in elo:
            elo[los] = 1500.0
            elo_history[los] = []

        # Expected scores
        exp_w = 1.0 / (1.0 + 10.0 ** ((elo[los] - elo[w]) / 400.0))
        exp_l = 1.0 - exp_w

        # Margin-of-victory multiplier (diminishing returns)
        margin = abs(ws - ls)
        mov_mult = np.log1p(margin) * 2.2 / (0.001 * margin + 2.2)

        # Update
        elo[w] += k_factor * mov_mult * (1.0 - exp_w)
        elo[los] += k_factor * mov_mult * (0.0 - exp_l)

        elo_history[w].append(elo[w])
        elo_history[los].append(elo[los])

    # Compute slope from last N games
    result = {}
    for t, history in elo_history.items():
        if len(history) < 3:
            result[t] = 0.0
            continue
        recent = history[-last_n:] if len(history) >= last_n else history
        x = np.arange(len(recent), dtype=np.float64)
        y = np.array(recent, dtype=np.float64)
        # Simple linear regression: slope = cov(x,y) / var(x)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        slope = np.sum((x - x_mean) * (y - y_mean)) / max(np.sum((x - x_mean) ** 2), 1e-10)
        result[t] = float(slope)

    return result


def _load_season_games_with_day(
    season: int,
    data_root: Path = Path("data"),
    cutoff_day: int = DEFAULT_CUTOFF_DAY,
) -> List[Tuple[int, int, int, int, int]]:
    """Load games as (day_num, winner_id, loser_id, w_score, l_score)."""
    for filename in ("MRegularSeasonCompactResults.csv", "MRegularSeasonDetailedResults.csv"):
        path = data_root / "kaggle" / filename
        if not path.exists():
            continue
        games = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["Season"]) != season:
                    continue
                if int(row["DayNum"]) >= cutoff_day:
                    continue
                games.append(
                    (
                        int(row["DayNum"]),
                        int(row["WTeamID"]),
                        int(row["LTeamID"]),
                        int(row["WScore"]),
                        int(row["LScore"]),
                    )
                )
        games.sort(key=lambda x: x[0])
        return games
    return []


# ---------------------------------------------------------------------------
# Conference tournament depth
# ---------------------------------------------------------------------------


def compute_conf_tourney_depth(
    season: int,
    data_root: Path = Path("data"),
) -> Dict[int, float]:
    """Compute conference tournament depth (number of wins) per team.

    0 = didn't play or lost first game, 1 = won one game, 3+ = champion.
    Teams not in conference tournament data get 0.

    Returns {kaggle_team_id: n_conf_tourney_wins}.
    """
    csv_path = data_root / "kaggle" / "MConferenceTourneyGames.csv"
    if not csv_path.exists():
        return {}

    wins: Dict[int, int] = {}
    losses: Dict[int, int] = {}

    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if int(row["Season"]) != season:
                continue
            w = int(row["WTeamID"])
            los = int(row["LTeamID"])
            wins[w] = wins.get(w, 0) + 1
            losses[los] = losses.get(los, 0) + 1

    # Return wins count (depth); teams that lost immediately have 0 wins
    all_teams = set(wins.keys()) | set(losses.keys())
    return {t: float(wins.get(t, 0)) for t in all_teams}


# ---------------------------------------------------------------------------
# Field chalk signal (adaptive strategy)
# ---------------------------------------------------------------------------


def compute_field_chalk_signal(
    season: int,
    seeds: Dict[str, int],
    data_root: Path = Path("data"),
) -> float:
    """Compute a [0, 1] signal measuring how chalk-dominant the field is.

    1.0 = all 1-seeds are historically dominant (play chalk)
    0.0 = weak/uneven 1-seeds, upset-prone field (play contrarian)

    Components:
    - min_barthag: weakest 1-seed's pre-tournament strength
    - barthag_spread: gap between strongest and weakest 1-seed
    - conf_depth_signal: did 1-seeds win their conference tournaments?

    The signal is calibrated against 2012-2026 historical data:
    - 2025/2026: signal ~0.9 (play chalk)
    - 2016/2023: signal ~0.2 (play contrarian)
    """
    import json

    one_seeds = [tid for tid, s in seeds.items() if s == 1]
    if not one_seeds:
        return 0.5

    # Component 1: 1-seed barthag from Torvik
    barthags = []
    try:
        torvik_path = data_root / "raw" / "historical" / f"torvik_{season}.json"
        with open(torvik_path) as f:
            torvik = json.load(f)
        teams = torvik.get("teams", torvik)
        barthag_map = {(t.get("team_id") or t.get("canonical_id")): t.get("barthag", 0.0) for t in teams}
        barthags = [barthag_map.get(t, 0.0) for t in one_seeds if barthag_map.get(t)]
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    if not barthags:
        return 0.5  # no data, neutral

    min_barthag = min(barthags)
    barthag_spread = max(barthags) - min(barthags)

    # Calibrated from historical: min_barthag ranges from ~0.90 to ~0.98
    # Map: 0.90 → 0.0, 0.98 → 1.0
    strength_signal = max(0.0, min(1.0, (min_barthag - 0.90) / 0.08))

    # Spread: low spread (< 0.02) = all strong = chalk; high (> 0.05) = uneven = upset risk
    spread_signal = max(0.0, min(1.0, 1.0 - (barthag_spread - 0.01) / 0.05))

    # Component 2: conf tourney depth of 1-seeds
    conf_signal = 0.5  # neutral default
    try:
        from src.data.normalize import normalize_team_id

        conf_depth = compute_conf_tourney_depth(season, data_root)
        canonical_depth = ratings_to_canonical(conf_depth, data_root)
        one_seed_depths = [canonical_depth.get(t, 0.0) for t in one_seeds]
        if one_seed_depths:
            # 3+ wins = champion; map avg depth: 0→0.0, 3→1.0
            avg_depth = sum(one_seed_depths) / len(one_seed_depths)
            conf_signal = max(0.0, min(1.0, avg_depth / 3.0))
    except Exception:
        pass

    # Weighted composite: strength matters most
    chalk_signal = 0.50 * strength_signal + 0.30 * spread_signal + 0.20 * conf_signal
    return round(max(0.0, min(1.0, chalk_signal)), 3)


def compute_field_volatility_signal(
    season: int,
    seeds: Dict[str, int],
    data_root: Path = Path("data"),
) -> float:
    """Improved chalk/upset signal incorporating full-field volatility.

    The original chalk signal only looks at 1-seed strength. This version
    adds three components that capture mid-field chaos potential:

    1. Seed-gap compression: how close are 2-4 seeds to 1-seeds?
       Small gap = deep field = more upsets.
    2. Mid-seed close-game %: teams seeded 7-12 with high close-game
       win rates are Cinderella candidates.
    3. Field volatility: game-margin variance of tournament teams.
       High-variance teams create unpredictable outcomes.

    Returns [0, 1]: 1.0 = chalk-dominated, 0.0 = upset-prone.
    """
    import json

    # Load Torvik barthag for all tournament teams
    barthag_map: Dict[str, float] = {}
    try:
        torvik_path = data_root / "raw" / "historical" / f"torvik_{season}.json"
        with open(torvik_path) as f:
            torvik = json.load(f)
        teams = torvik.get("teams", torvik)
        barthag_map = {(t.get("team_id") or t.get("canonical_id")): t.get("barthag", 0.0) for t in teams}
    except (FileNotFoundError, json.JSONDecodeError):
        return 0.5

    # Partition tournament teams by seed tier
    one_seeds = [tid for tid, s in seeds.items() if s == 1]
    two_four_seeds = [tid for tid, s in seeds.items() if 2 <= s <= 4]
    mid_seeds = [tid for tid, s in seeds.items() if 7 <= s <= 12]

    one_barthags = [barthag_map.get(t, 0.0) for t in one_seeds if barthag_map.get(t)]
    two_four_barthags = [barthag_map.get(t, 0.0) for t in two_four_seeds if barthag_map.get(t)]

    if not one_barthags:
        return 0.5

    # --- Component 1: Original 1-seed strength (30% weight) ---
    min_barthag = min(one_barthags)
    strength_signal = max(0.0, min(1.0, (min_barthag - 0.90) / 0.08))

    # --- Component 2: 1-vs-2 seed gap (30% weight) ---
    # The single most discriminating pre-tournament signal between chalk
    # and chaotic years. 2025 gap=0.029 (chalk, 9 upsets), 2026 gap=0.013
    # (chaotic, 16 upsets). When 2-seeds are near 1-seed quality, deep
    # runs and upsets follow (UConn 2-seed in 2026 championship).
    two_seeds = [tid for tid, s in seeds.items() if s == 2]
    two_barthags = [barthag_map.get(t, 0.0) for t in two_seeds if barthag_map.get(t)]
    gap_1v2_signal = 0.5
    if two_barthags and one_barthags:
        avg_one = sum(one_barthags) / len(one_barthags)
        avg_two = sum(two_barthags) / len(two_barthags)
        gap_1v2 = avg_one - avg_two
        # Empirical: 0.010 (compressed) to 0.035 (separated).
        gap_1v2_signal = max(0.0, min(1.0, (gap_1v2 - 0.010) / 0.025))

    # --- Component 3: 1-vs-(2-4) seed gap (20% weight) ---
    gap_signal = 0.5
    if two_four_barthags and one_barthags:
        avg_one = sum(one_barthags) / len(one_barthags)
        avg_two_four = sum(two_four_barthags) / len(two_four_barthags)
        gap = avg_one - avg_two_four
        gap_signal = max(0.0, min(1.0, (gap - 0.02) / 0.03))

    # --- Component 4: 1-seed barthag spread (20% weight) ---
    barthag_spread = max(one_barthags) - min(one_barthags)
    spread_signal = max(0.0, min(1.0, 1.0 - (barthag_spread - 0.005) / 0.04))

    # Weighted composite
    vol_signal = 0.30 * strength_signal + 0.30 * gap_1v2_signal + 0.20 * gap_signal + 0.20 * spread_signal
    return round(max(0.0, min(1.0, vol_signal)), 3)


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------


def compute_all_custom_ratings(
    season: int,
    data_root: Path = Path("data"),
    cutoff_day: int = DEFAULT_CUTOFF_DAY,
) -> Dict[str, Dict[int, float]]:
    """Compute all three rating systems for a season.

    Returns {"colley": {id: rating}, "srs": {id: rating}, "glm": {id: rating}}.
    """
    return {
        "colley": compute_colley_ratings(season, data_root, cutoff_day),
        "srs": compute_srs_ratings(season, data_root, cutoff_day),
        "glm": compute_glm_quality(season, data_root, cutoff_day),
    }


def ratings_to_canonical(
    ratings: Dict[int, float],
    data_root: Path = Path("data"),
) -> Dict[str, float]:
    """Convert Kaggle numeric team IDs to canonical string IDs.

    Uses MTeams.csv for the ID→name mapping, then normalize_team_id.
    """
    from src.data.normalize import normalize_team_id

    teams_path = data_root / "kaggle" / "MTeams.csv"
    if not teams_path.exists():
        return {str(k): v for k, v in ratings.items()}

    id_to_name = {}
    with open(teams_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            id_to_name[int(row["TeamID"])] = row["TeamName"]

    result = {}
    for tid, rating in ratings.items():
        name = id_to_name.get(tid)
        if name:
            result[normalize_team_id(name)] = rating
        else:
            result[str(tid)] = rating
    return result
