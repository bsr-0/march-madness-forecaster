"""Upset-tuned adjustment (catalog D2).

Calibrates round_probs against historical seed-by-round advancement rates.
If history says 11-seeds reach S16 at ~16% but this year's model predicts
5% for the 11-seeds in the field, the adjustment scales each 11-seed's
S16 probability up by ~3×. Applied as a multiplicative per-(seed, round)
calibration and re-normalized per round so the tournament tree still
respects team-count constraints (32 teams to R32, 16 to S16, …).

Walk-forward contract: historical reach rates are computed from tournament
years strictly before the test year. Uses `tournament_results_{year}.json`
(already team-ID-normalized to the Torvik / seeds ID scheme), so this
adjustment sidesteps the cbbpy ID-mismatch blocker that holds back Elo
(A4), roster-adj (C2), and volatile (D1).

Implementation note (deliberate deviation from catalog spec): the catalog
describes D2 as a pairwise Log5 calibration at sample time. Adjustments
in this architecture operate on round_probs before sampling, not on
pairwise matchups inside the sampler. We apply the calibration at the
seed-aggregate level, which captures the same signal (historical
over/under-performance by seed) while remaining composable with every
source and construction mode. If a pairwise version is ever needed it
would live inside the sampler, not here.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

ROUND_NAMES = ("R64", "R32", "S16", "E8", "F4", "CHAMP")

# tournament_results schema uses "NCG" for the championship game; our
# round_probs schema uses "CHAMP". Map at ingestion time.
_RESULTS_TO_PROB_ROUND = {
    "R64": "R64",
    "R32": "R32",
    "S16": "S16",
    "E8": "E8",
    "F4": "F4",
    "NCG": "CHAMP",
}

# round_probs[team][rnd] = P(team WINS its rnd game) — see the identical
# fix + comment in src/prediction/roster_adj_probabilities.py, same bug.
_TEAMS_PER_ROUND = {"R64": 32, "R32": 16, "S16": 8, "E8": 4, "F4": 2, "CHAMP": 1}

# Bound the calibration multiplier so small-N historical cells don't
# produce wildly distorted probabilities (e.g., a 16-seed with a 0%
# historical F4 rate shouldn't zero out the model's already-tiny F4 prob).
_CAL_FACTOR_MIN = 0.5
_CAL_FACTOR_MAX = 2.0


def compute_historical_seed_reach_rates(
    max_year_exclusive: int,
    data_root: Path,
    min_year: int = 2005,
) -> Dict[int, Dict[str, float]]:
    """Empirical seed-by-round advancement rates from tournaments < max_year_exclusive.

    For each seed 1-16 and each round R64..CHAMP, returns the fraction of
    teams seeded ``s`` that appeared in a round-``r`` game across the
    historical window ``[min_year, max_year_exclusive)``.

    Skips 2020 (COVID — no tournament). Interpretation:
        rates[3]["S16"] = 0.52 means 52% of 3-seeds reached the Sweet 16.
    """
    seed_reached: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # Each tournament has exactly 4 teams per seed (1-16); total = 4 * n_tournaments.
    seed_totals: Dict[int, int] = defaultdict(int)

    data_root = Path(data_root)
    hist_dir = data_root / "raw" / "historical"
    for year in range(min_year, max_year_exclusive):
        if year == 2020:
            continue
        ctx_path = hist_dir / f"tournament_context_{year}.json"
        raw = None
        if ctx_path.exists():
            with open(ctx_path) as f:
                ctx = json.load(f)
            raw = ctx.get("results")
        if raw is None:
            path = hist_dir / f"tournament_results_{year}.json"
            if not path.exists():
                continue
            with open(path) as f:
                raw = json.load(f)
        games = raw["games"] if isinstance(raw, dict) and "games" in raw else raw

        # Deduplicate (team_id, seed) within a year per round — one team can
        # appear in at most one game per round.
        teams_in_round: Dict[str, set] = defaultdict(set)
        for g in games:
            r = g.get("round_name") or g.get("round")
            if r not in _RESULTS_TO_PROB_ROUND:
                continue
            prob_round = _RESULTS_TO_PROB_ROUND[r]
            if g.get("team1_seed") is not None:
                teams_in_round[prob_round].add((g["team1_id"], int(g["team1_seed"])))
            if g.get("team2_seed") is not None:
                teams_in_round[prob_round].add((g["team2_id"], int(g["team2_seed"])))

        for prob_round, teams in teams_in_round.items():
            for _tid, seed in teams:
                seed_reached[seed][prob_round] += 1

        for seed in range(1, 17):
            seed_totals[seed] += 4  # 4 regions × 1 team per seed-line per region

    rates: Dict[int, Dict[str, float]] = {}
    for seed in range(1, 17):
        total = seed_totals[seed]
        rates[seed] = {}
        for rnd in ROUND_NAMES:
            if total > 0:
                rates[seed][rnd] = seed_reached[seed][rnd] / total
            else:
                rates[seed][rnd] = 0.0
    return rates


def build_upset_tuned_round_probs(
    model_rp: Dict[str, Dict[str, float]],
    seeds: Dict[str, int],
    historical_reach_rates: Dict[int, Dict[str, float]],
    *,
    factor_min: float = _CAL_FACTOR_MIN,
    factor_max: float = _CAL_FACTOR_MAX,
) -> Dict[str, Dict[str, float]]:
    """Apply per-(seed, round) historical calibration to round_probs.

    For each team ``t`` with seed ``s(t)`` and round ``r``:
        model_mean_s_r = mean(model_rp[t'][r] for t' in this year's field with seed s(t))
        factor        = clip(historical_reach_rates[s(t)][r] / model_mean_s_r, 0.5, 2.0)
        adjusted[t][r] = model_rp[t][r] * factor

    Re-normalizes per round so the total advancement mass matches the
    tree's team-count constraint (``_TEAMS_PER_ROUND``). Teams with no
    seed information pass through unchanged.
    """
    # Compute per-seed model mean reach rate for this year.
    model_mean: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    model_count: Dict[int, int] = defaultdict(int)
    for tid, s in seeds.items():
        if tid not in model_rp:
            continue
        model_count[s] += 1
        for rnd in ROUND_NAMES:
            model_mean[s][rnd] += model_rp[tid].get(rnd, 0.0)
    for s in model_mean:
        if model_count[s] > 0:
            for rnd in ROUND_NAMES:
                model_mean[s][rnd] /= model_count[s]

    # Apply factor per team per round.
    result: Dict[str, Dict[str, float]] = {}
    for tid, rounds in model_rp.items():
        result[tid] = {}
        s = seeds.get(tid)
        for rnd in ROUND_NAMES:
            model_p = rounds.get(rnd, 0.001)
            if s is None or s not in historical_reach_rates:
                result[tid][rnd] = max(model_p, 0.001)
                continue
            hist_rate = historical_reach_rates[s].get(rnd, 0.0)
            mean_model_rate = model_mean[s][rnd]
            if mean_model_rate < 1e-6 or hist_rate < 1e-6:
                # Either history or model has zero mass here — don't
                # manufacture an adjustment from a cell with no signal.
                result[tid][rnd] = max(model_p, 0.001)
                continue
            factor = hist_rate / mean_model_rate
            factor = max(factor_min, min(factor_max, factor))
            result[tid][rnd] = max(model_p * factor, 0.001)

    # Re-normalize per round (same pattern as build_contrarian_round_probs).
    for rnd in ROUND_NAMES:
        total = sum(result[tid].get(rnd, 0.0) for tid in result)
        target = _TEAMS_PER_ROUND.get(rnd, 32)
        if total > 0:
            scale = target / total
            for tid in result:
                if rnd in result[tid]:
                    result[tid][rnd] *= scale

    return result


def load_upset_tuned_context(
    test_year: int,
    data_root: Optional[Path] = None,
) -> Dict[int, Dict[str, float]]:
    """Convenience: walk-forward historical rates for a specific test year.

    Args:
        test_year: The year being backtested. History is taken from
            ``[2005, test_year)``.
        data_root: Repo data root. Defaults to <project>/data.
    """
    if data_root is None:
        data_root = Path(__file__).resolve().parent.parent.parent / "data"
    return compute_historical_seed_reach_rates(
        max_year_exclusive=test_year,
        data_root=data_root,
    )
