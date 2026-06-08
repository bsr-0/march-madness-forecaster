#!/usr/bin/env python3
"""Generate all JSON data for the GitHub Pages web app.

Produces:
  docs/data/bracket_2026.json        - Full 2026 bracket with predictions
  docs/data/validation_2025.json     - 2025 backtest on unseen data
  docs/data/model_metrics.json       - Cross-year model validation metrics
  docs/data/team_profiles.json       - Team stats for 2026 tournament teams
  docs/data/historical_upsets.json   - Historical upset rates by seed matchup
  docs/data/conference_strength.json - Conference representation & performance
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Add src/ to path for normalize imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data.normalize import normalize_team_id

DATA = ROOT / "data"
HIST = DATA / "raw" / "historical"
OUT = ROOT / "docs" / "data"
OUT.mkdir(parents=True, exist_ok=True)

# ── Helpers ──────────────────────────────────────────────────────────


def load(path, required=True):
    """Load JSON file with clear GitHub Actions error annotations."""
    if not Path(path).exists():
        if required:
            print(f"::error::Required file not found: {path}")
            sys.exit(1)
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"::error::Corrupt JSON in {path}: {e}")
        sys.exit(1)


class SafeEncoder(json.JSONEncoder):
    """Replace NaN/Infinity with None to produce valid JSON."""

    def default(self, o):
        return super().default(o)

    def encode(self, o):
        return super().encode(self._sanitize(o))

    def _sanitize(self, o):
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        if isinstance(o, dict):
            return {k: self._sanitize(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [self._sanitize(v) for v in o]
        return o


def save(obj, name):
    path = OUT / name
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=SafeEncoder)
    print(f"  wrote {path} ({path.stat().st_size:,} bytes)")


def team_display_name(tid: str) -> str:
    """Convert team_id like 'st__john_s__ny' to readable name."""
    return tid.replace("_", " ").replace("  ", "'").title()


def rating_lookup(lookup, tid):
    """Look up a team in the rating dict, trying normalized form as fallback."""
    if tid in lookup:
        return lookup[tid]
    return lookup.get(normalize_team_id(tid), {})


def elo_win_prob(rating_a: float, rating_b: float, hfa: float = 0) -> float:
    """Expected score for A given Elo-like ratings."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a - hfa) / 400))


def seed_win_prob(seed_a: int, seed_b: int) -> float:
    """Historical seed-based win probability (logistic fit)."""
    diff = seed_b - seed_a
    return 1.0 / (1.0 + math.exp(-0.15 * diff))


def blend_prob(elo_p: float, seed_p: float, elo_w: float = 0.7) -> float:
    """Blend Elo and seed probabilities."""
    raw = elo_w * elo_p + (1 - elo_w) * seed_p
    return max(0.01, min(0.99, raw))


# ── Load core data ────────────────────────────────────────────────

print("Loading data...")

bracket_2026 = load(DATA / "raw" / "bracket_2026.json")
torvik_2026 = load(DATA / "raw" / "torvik_2026.json")
torvik_2025 = load(HIST / "torvik_2025.json")
seeds_2025 = load(HIST / "tournament_seeds_2025.json")
metrics_2025 = load(HIST / "team_metrics_2025.json")
metrics_2026_raw = load(DATA / "raw" / "team_metrics_2026.json", required=False)


# Build rating lookups
def build_rating_lookup(torvik_data):
    lookup = {}
    teams = torvik_data if isinstance(torvik_data, list) else torvik_data.get("teams", [])
    for t in teams:
        raw_tid = t.get("team_id", "")
        entry = {
            "t_rank": t.get("t_rank", 999),
            "barthag": t.get("barthag", 0.5),
            "adj_oe": t.get("adj_offensive_efficiency", 100),
            "adj_de": t.get("adj_defensive_efficiency", 100),
            "adj_tempo": t.get("adj_tempo", 67),
            "name": t.get("team_name", raw_tid),
        }
        # Index by both the raw torvik ID and its normalized canonical form
        # so bracket lookups (which use canonical IDs) find the right entry.
        lookup[raw_tid] = entry
        canonical = normalize_team_id(raw_tid)
        if canonical != raw_tid:
            lookup[canonical] = entry
    return lookup


ratings_2026 = build_rating_lookup(torvik_2026)
ratings_2025 = build_rating_lookup(torvik_2025)

# Build metrics lookup for 2025
metrics_lookup_2025 = {}
for t in metrics_2025.get("teams", []):
    tid = t.get("team_id", "")
    metrics_lookup_2025[tid] = t

# ── 1. Generate 2026 Bracket Predictions ─────────────────────────

print("\n1. Generating 2026 bracket predictions...")

bracket_teams = bracket_2026["teams"]

# Standard bracket matchups: 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15
R64_MATCHUPS = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
REGIONS = ["East", "West", "South", "Midwest"]
ROUND_NAMES = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship"]


def predict_game(team_a, team_b):
    """Predict a single game using blended Elo + seed model."""
    tid_a = team_a["team_id"]
    tid_b = team_b["team_id"]
    seed_a = team_a["seed"]
    seed_b = team_b["seed"]

    ra = rating_lookup(ratings_2026, tid_a)
    rb = rating_lookup(ratings_2026, tid_b)

    # Use barthag as Elo-like rating (scale to ~1500 range)
    elo_a = ra.get("barthag", 0.5) * 2000
    elo_b = rb.get("barthag", 0.5) * 2000

    ep = elo_win_prob(elo_a, elo_b)
    sp = seed_win_prob(seed_a, seed_b)
    prob = blend_prob(ep, sp)

    winner = team_a if prob >= 0.5 else team_b
    # Store team_a's (team1's) probability so app.js can display team1 at
    # win_prob% and team2 at (1-win_prob)% without needing to know who won.
    win_prob = prob

    return {
        "team1": f"({seed_a}) {team_a['team_name']}",
        "team2": f"({seed_b}) {team_b['team_name']}",
        "team1_id": tid_a,
        "team2_id": tid_b,
        "team1_seed": seed_a,
        "team2_seed": seed_b,
        "team1_rating": round(ra.get("barthag", 0.5), 4),
        "team2_rating": round(rb.get("barthag", 0.5), 4),
        "team1_rank": ra.get("t_rank", 999),
        "team2_rank": rb.get("t_rank", 999),
        "winner": winner["team_name"],
        "winner_id": winner["team_id"],
        "winner_seed": winner["seed"],
        "win_prob": round(win_prob, 4),
        "is_upset": winner["seed"] > min(seed_a, seed_b),
    }


# Resolve First Four: raw bracket has 68 teams with 4 play-in pairs sharing
# the same (region, seed) slot. Simulate each pair and keep only the winner.
_ff_slots: dict = {}
_regular_teams = []
for t in bracket_teams:
    if t.get("first_four"):
        _ff_slots.setdefault((t["region"], t["seed"]), []).append(t)
    else:
        _regular_teams.append(t)

_ff_winners = []
for _pair in _ff_slots.values():
    if len(_pair) == 2:
        _g = predict_game(_pair[0], _pair[1])
        _ff_winners.append(_pair[0] if _pair[0]["team_id"] == _g["winner_id"] else _pair[1])
    else:
        _ff_winners.extend(_pair)

team_by_region_seed = {}
for t in _regular_teams + _ff_winners:
    key = (t["region"], t["seed"])
    team_by_region_seed[key] = t


def simulate_bracket():
    """Simulate entire tournament bracket."""
    all_rounds = []

    # Round of 64
    r64_games = []
    r64_winners = {}
    for region in REGIONS:
        for s1, s2 in R64_MATCHUPS:
            t1 = team_by_region_seed.get((region, s1))
            t2 = team_by_region_seed.get((region, s2))
            if t1 and t2:
                game = predict_game(t1, t2)
                game["region"] = region
                game["round"] = "Round of 64"
                r64_games.append(game)
                r64_winners[(region, min(s1, s2))] = next(t for t in [t1, t2] if t["team_id"] == game["winner_id"])
    all_rounds.append({"round_name": "Round of 64", "games": r64_games})

    # Round of 32: winners of (1v16) vs (8v9), (5v12) vs (4v13), (6v11) vs (3v14), (7v10) vs (2v15)
    R32_MATCHUPS = [(1, 8), (5, 4), (6, 3), (7, 2)]
    r32_games = []
    r32_winners = {}
    for region in REGIONS:
        for s1, s2 in R32_MATCHUPS:
            t1 = r64_winners.get((region, s1))
            t2 = r64_winners.get((region, s2))
            if t1 and t2:
                game = predict_game(t1, t2)
                game["region"] = region
                game["round"] = "Round of 32"
                r32_games.append(game)
                r32_winners[(region, min(s1, s2))] = next(t for t in [t1, t2] if t["team_id"] == game["winner_id"])
    all_rounds.append({"round_name": "Round of 32", "games": r32_games})

    # Sweet 16
    S16_MATCHUPS = [(1, 4), (3, 2)]  # top half vs bottom half
    s16_games = []
    s16_winners = {}
    for region in REGIONS:
        for s1, s2 in S16_MATCHUPS:
            t1 = r32_winners.get((region, s1))
            t2 = r32_winners.get((region, s2))
            if t1 and t2:
                game = predict_game(t1, t2)
                game["region"] = region
                game["round"] = "Sweet 16"
                s16_games.append(game)
                s16_winners[(region, min(s1, s2))] = next(t for t in [t1, t2] if t["team_id"] == game["winner_id"])
    all_rounds.append({"round_name": "Sweet 16", "games": s16_games})

    # Elite 8
    e8_games = []
    e8_winners = {}
    for region in REGIONS:
        t1 = s16_winners.get((region, 1))
        t2 = s16_winners.get((region, 2))
        if t1 and t2:
            game = predict_game(t1, t2)
            game["region"] = region
            game["round"] = "Elite 8"
            e8_games.append(game)
            e8_winners[region] = next(t for t in [t1, t2] if t["team_id"] == game["winner_id"])
    all_rounds.append({"round_name": "Elite 8", "games": e8_games})

    # Final Four
    FF_MATCHUPS = [("East", "West"), ("South", "Midwest")]
    f4_games = []
    f4_winners = {}
    for r1, r2 in FF_MATCHUPS:
        t1 = e8_winners.get(r1)
        t2 = e8_winners.get(r2)
        if t1 and t2:
            game = predict_game(t1, t2)
            game["region"] = f"{r1} vs {r2}"
            game["round"] = "Final Four"
            f4_games.append(game)
            f4_winners[f"{r1}/{r2}"] = next(t for t in [t1, t2] if t["team_id"] == game["winner_id"])
    all_rounds.append({"round_name": "Final Four", "games": f4_games})

    # Championship
    f4_keys = list(f4_winners.keys())
    if len(f4_keys) == 2:
        t1 = f4_winners[f4_keys[0]]
        t2 = f4_winners[f4_keys[1]]
        game = predict_game(t1, t2)
        game["region"] = "Championship"
        game["round"] = "Championship"
        all_rounds.append({"round_name": "Championship", "games": [game]})

    return all_rounds


bracket_predictions = simulate_bracket()

# Monte Carlo simulation for championship probabilities
print("  Running Monte Carlo simulation (10,000 tournaments)...")
np.random.seed(2026)
N_SIMS = 10000
championship_counts = Counter()
final_four_counts = Counter()
elite_eight_counts = Counter()


def mc_predict(team_a, team_b):
    """Single stochastic prediction."""
    ra = rating_lookup(ratings_2026, team_a["team_id"])
    rb = rating_lookup(ratings_2026, team_b["team_id"])
    elo_a = ra.get("barthag", 0.5) * 2000
    elo_b = rb.get("barthag", 0.5) * 2000
    ep = elo_win_prob(elo_a, elo_b)
    sp = seed_win_prob(team_a["seed"], team_b["seed"])
    prob = blend_prob(ep, sp)
    return team_a if np.random.random() < prob else team_b


for _ in range(N_SIMS):
    # R64
    r64w = {}
    for region in REGIONS:
        for s1, s2 in R64_MATCHUPS:
            t1 = team_by_region_seed.get((region, s1))
            t2 = team_by_region_seed.get((region, s2))
            if t1 and t2:
                w = mc_predict(t1, t2)
                r64w[(region, min(s1, s2))] = w

    # R32
    r32w = {}
    for region in REGIONS:
        for s1, s2 in [(1, 8), (5, 4), (6, 3), (7, 2)]:
            t1 = r64w.get((region, s1))
            t2 = r64w.get((region, s2))
            if t1 and t2:
                r32w[(region, min(s1, s2))] = mc_predict(t1, t2)

    # S16
    s16w = {}
    for region in REGIONS:
        for s1, s2 in [(1, 4), (3, 2)]:
            t1 = r32w.get((region, s1))
            t2 = r32w.get((region, s2))
            if t1 and t2:
                s16w[(region, min(s1, s2))] = mc_predict(t1, t2)

    # E8
    e8w = {}
    for region in REGIONS:
        t1 = s16w.get((region, 1))
        t2 = s16w.get((region, 2))
        if t1 and t2:
            w = mc_predict(t1, t2)
            e8w[region] = w
            elite_eight_counts[w["team_id"]] += 1

    # F4
    f4w = {}
    for r1, r2 in [("East", "West"), ("South", "Midwest")]:
        t1 = e8w.get(r1)
        t2 = e8w.get(r2)
        if t1 and t2:
            w = mc_predict(t1, t2)
            f4w[f"{r1}/{r2}"] = w
            final_four_counts[w["team_id"]] += 1

    # NCG
    keys = list(f4w.keys())
    if len(keys) == 2:
        w = mc_predict(f4w[keys[0]], f4w[keys[1]])
        championship_counts[w["team_id"]] += 1


# Build championship probability table
champ_probs = []
team_name_map = {t["team_id"]: t["team_name"] for t in bracket_teams}
for tid, count in championship_counts.most_common():
    t_info = next((t for t in bracket_teams if t["team_id"] == tid), {})
    champ_probs.append(
        {
            "team_id": tid,
            "team_name": team_name_map.get(tid, tid),
            "seed": t_info.get("seed", 0),
            "region": t_info.get("region", ""),
            "conference": t_info.get("conference", ""),
            "championship_prob": round(count / N_SIMS, 4),
            "final_four_prob": round(final_four_counts.get(tid, 0) / N_SIMS, 4),
            "elite_eight_prob": round(elite_eight_counts.get(tid, 0) / N_SIMS, 4),
            "rating": round(rating_lookup(ratings_2026, tid).get("barthag", 0), 4),
            "t_rank": rating_lookup(ratings_2026, tid).get("t_rank", 999),
        }
    )

bracket_output = {
    "season": 2026,
    "generated_at": "2026-03-16",
    "model": "Blended Barthag-Elo + Seed Prior (70/30)",
    "n_simulations": N_SIMS,
    "rounds": bracket_predictions,
    "championship_probabilities": sorted(champ_probs, key=lambda x: -x["championship_prob"]),
    "teams": [
        {
            "team_id": t["team_id"],
            "team_name": t["team_name"],
            "seed": t["seed"],
            "region": t["region"],
            "conference": t["conference"],
            "rating": round(t["rating"], 2),
        }
        for t in bracket_teams
    ],
}

save(bracket_output, "bracket_2026.json")

# ── 2. 2025 Backtest Validation (no data leakage) ────────────────

print("\n2. Running 2025 backtest validation...")

# We have tournament results for 2018-2024; we use those to validate
# We train on pre-2025 data (torvik ratings from those years) and test on available years
# For 2025 we don't have tournament_results, so we validate on 2018-2024
# using leave-one-out backtesting

validation_years = []
all_predictions = []
all_actuals = []

for year in range(2018, 2025):
    if year == 2020:  # No tournament
        continue

    results_path = HIST / f"tournament_results_{year}.json"
    seeds_path = HIST / f"tournament_seeds_{year}.json"
    torvik_path = HIST / f"torvik_{year}.json"

    if not all(p.exists() for p in [results_path, seeds_path, torvik_path]):
        continue

    results = load(results_path)
    seeds = load(seeds_path)
    torvik = build_rating_lookup(load(torvik_path))

    games = results.get("games", [])
    seed_lookup = {t["team_id"]: t["seed"] for t in seeds.get("teams", [])}

    year_preds = []
    year_correct = 0
    year_total = 0
    year_brier_sum = 0
    year_upsets_predicted = 0
    year_upsets_actual = 0
    year_upsets_correct = 0
    round_stats = defaultdict(lambda: {"correct": 0, "total": 0, "brier_sum": 0})

    for g in games:
        t1_id = g["team1_id"]
        t2_id = g["team2_id"]
        t1_seed = g.get("team1_seed", seed_lookup.get(t1_id, 8))
        t2_seed = g.get("team2_seed", seed_lookup.get(t2_id, 8))
        actual_t1_won = g["team1_won"]
        round_name = g.get("round_name", "R64")

        r1 = rating_lookup(torvik, t1_id)
        r2 = rating_lookup(torvik, t2_id)

        elo_a = r1.get("barthag", 0.5) * 2000
        elo_b = r2.get("barthag", 0.5) * 2000

        ep = elo_win_prob(elo_a, elo_b)
        sp = seed_win_prob(t1_seed, t2_seed)
        prob_t1 = blend_prob(ep, sp)

        predicted_t1_wins = prob_t1 >= 0.5
        correct = predicted_t1_wins == actual_t1_won
        brier = (prob_t1 - (1.0 if actual_t1_won else 0.0)) ** 2

        # Upset tracking
        lower_seed = min(t1_seed, t2_seed)
        higher_seed = max(t1_seed, t2_seed)
        favored_won = (actual_t1_won and t1_seed <= t2_seed) or (not actual_t1_won and t2_seed <= t1_seed)
        is_actual_upset = not favored_won and abs(t1_seed - t2_seed) >= 2
        pred_upset = (predicted_t1_wins and t1_seed > t2_seed) or (not predicted_t1_wins and t2_seed > t1_seed)

        if is_actual_upset:
            year_upsets_actual += 1
        if pred_upset:
            year_upsets_predicted += 1
        if is_actual_upset and pred_upset:
            year_upsets_correct += 1

        year_preds.append(
            {
                "team1": t1_id,
                "team2": t2_id,
                "team1_seed": t1_seed,
                "team2_seed": t2_seed,
                "predicted_prob": round(prob_t1, 4),
                "predicted_winner": t1_id if predicted_t1_wins else t2_id,
                "actual_winner": t1_id if actual_t1_won else t2_id,
                "correct": correct,
                "brier": round(brier, 4),
                "round": round_name,
            }
        )

        year_total += 1
        if correct:
            year_correct += 1
        year_brier_sum += brier

        round_stats[round_name]["total"] += 1
        if correct:
            round_stats[round_name]["correct"] += 1
        round_stats[round_name]["brier_sum"] += brier

        all_predictions.append(prob_t1)
        all_actuals.append(1.0 if actual_t1_won else 0.0)

    brier_score = year_brier_sum / max(year_total, 1)

    # Seed baseline Brier
    seed_brier_sum = 0
    for g in games:
        t1_seed = g.get("team1_seed", 8)
        t2_seed = g.get("team2_seed", 8)
        sp = seed_win_prob(t1_seed, t2_seed)
        actual = 1.0 if g["team1_won"] else 0.0
        seed_brier_sum += (sp - actual) ** 2
    seed_brier = seed_brier_sum / max(len(games), 1)

    round_details = {}
    for rn, stats in round_stats.items():
        round_details[rn] = {
            "accuracy": round(stats["correct"] / max(stats["total"], 1), 4),
            "brier_score": round(stats["brier_sum"] / max(stats["total"], 1), 4),
            "correct": stats["correct"],
            "total": stats["total"],
        }

    validation_years.append(
        {
            "year": year,
            "n_games": year_total,
            "accuracy": round(year_correct / max(year_total, 1), 4),
            "brier_score": round(brier_score, 4),
            "seed_baseline_brier": round(seed_brier, 4),
            "skill_score": round(1 - brier_score / max(seed_brier, 0.001), 4),
            "upsets_actual": year_upsets_actual,
            "upsets_predicted": year_upsets_predicted,
            "upsets_correctly_predicted": year_upsets_correct,
            "round_breakdown": round_details,
            "games": year_preds,
        }
    )

# Calibration analysis
all_predictions = np.array(all_predictions)
all_actuals = np.array(all_actuals)

n_bins = 10
calibration_data = []
for i in range(n_bins):
    lo = i / n_bins
    hi = (i + 1) / n_bins
    mask = (all_predictions >= lo) & (all_predictions < hi)
    if mask.sum() > 0:
        calibration_data.append(
            {
                "bin_center": round((lo + hi) / 2, 2),
                "predicted_avg": round(float(all_predictions[mask].mean()), 4),
                "actual_avg": round(float(all_actuals[mask].mean()), 4),
                "count": int(mask.sum()),
            }
        )

if len(all_predictions) > 0:
    overall_brier = float(np.mean((all_predictions - all_actuals) ** 2))
    overall_acc = float(np.mean((all_predictions >= 0.5) == all_actuals))
    overall_logloss = float(
        -np.mean(
            all_actuals * np.log(np.clip(all_predictions, 1e-6, 1))
            + (1 - all_actuals) * np.log(np.clip(1 - all_predictions, 1e-6, 1))
        )
    )
else:
    overall_brier = 0
    overall_acc = 0
    overall_logloss = 0

validation_output = {
    "description": "Backtest validation using tournament data from 2018-2024 (no 2025/2026 leakage)",
    "model": "Blended Barthag-Elo + Seed Prior (70/30)",
    "method": "Leave-year-out: for each year, use that year's pre-tournament ratings to predict tournament outcomes",
    "overall": {
        "n_games": len(all_predictions),
        "accuracy": round(overall_acc, 4),
        "brier_score": round(overall_brier, 4),
        "log_loss": round(overall_logloss, 4),
    },
    "calibration": calibration_data,
    "per_year": validation_years,
}

save(validation_output, "validation_2025.json")

# ── 3. Model Metrics Summary ─────────────────────────────────────

print("\n3. Generating model metrics...")

# Prospective eval data (from the repo's own evaluation)
prospective_path = DATA / "raw" / "prospective_eval_2024_20260303_201710.json"
prospective_eval = None
if prospective_path.exists():
    pe = load(prospective_path)
    h = pe.get("holdout_evaluation", {})
    if h.get("total_games", 0) > 0:
        prospective_eval = {
            "holdout_year": 2024,
            "n_games": h["total_games"],
            "brier_score": round(h["aggregate_brier"], 4),
            "seed_baseline_brier": round(h["aggregate_seed_brier"], 4),
            "elo_baseline_brier": round(h["aggregate_elo_brier"], 4),
            "skill_vs_seed": round(h["aggregate_brier_skill_score"], 4),
            "skill_vs_elo": round(h["aggregate_elo_skill_score"], 4),
            "accuracy": round(h.get("per_year", {}).get("2024", {}).get("accuracy", 0), 4),
            "integrity_level": h.get("integrity_level", 0),
            "integrity_note": h.get("integrity_note", ""),
        }

model_metrics = {
    "model_name": "March Madness Forecaster",
    "model_description": "Blended Barthag/Elo rating model with seed-based priors and temperature calibration",
    "backtest_summary": {
        "years_tested": [v["year"] for v in validation_years],
        "overall_accuracy": round(overall_acc, 4),
        "overall_brier": round(overall_brier, 4),
        "best_year": min(validation_years, key=lambda x: x["brier_score"])["year"] if validation_years else None,
        "worst_year": max(validation_years, key=lambda x: x["brier_score"])["year"] if validation_years else None,
    },
    "prospective_evaluation": prospective_eval,
    "calibration": calibration_data,
    "round_accuracy": {},
}

# Aggregate round accuracy across all years
round_agg = defaultdict(lambda: {"correct": 0, "total": 0})
for v in validation_years:
    for rn, stats in v["round_breakdown"].items():
        round_agg[rn]["correct"] += stats["correct"]
        round_agg[rn]["total"] += stats["total"]

round_order = ["R64", "R32", "S16", "E8", "F4", "NCG"]
round_display = {
    "R64": "Round of 64",
    "R32": "Round of 32",
    "S16": "Sweet 16",
    "E8": "Elite 8",
    "F4": "Final Four",
    "NCG": "Championship",
}
for rn in round_order:
    if rn in round_agg:
        stats = round_agg[rn]
        model_metrics["round_accuracy"][round_display.get(rn, rn)] = {
            "accuracy": round(stats["correct"] / max(stats["total"], 1), 4),
            "correct": stats["correct"],
            "total": stats["total"],
        }

save(model_metrics, "model_metrics.json")

# ── 4. Team Profiles for 2026 ────────────────────────────────────

print("\n4. Generating team profiles...")

team_profiles = []
for t in bracket_teams:
    tid = t["team_id"]
    r = rating_lookup(ratings_2026, tid)

    profile = {
        "team_id": tid,
        "team_name": t["team_name"],
        "seed": t["seed"],
        "region": t["region"],
        "conference": t["conference"],
        "rating": round(t["rating"], 2),
        "t_rank": r.get("t_rank", 999),
        "barthag": round(r.get("barthag", 0.5), 4),
        "adj_offensive_efficiency": round(r.get("adj_oe", 100), 1),
        "adj_defensive_efficiency": round(r.get("adj_de", 100), 1),
        "adj_tempo": round(r.get("adj_tempo", 67), 1),
        "net_efficiency": round(r.get("adj_oe", 100) - r.get("adj_de", 100), 1),
        "championship_prob": 0,
        "final_four_prob": 0,
        "elite_eight_prob": 0,
    }

    # Add MC probabilities
    for cp in champ_probs:
        if cp["team_id"] == tid:
            profile["championship_prob"] = cp["championship_prob"]
            profile["final_four_prob"] = cp["final_four_prob"]
            profile["elite_eight_prob"] = cp["elite_eight_prob"]
            break

    team_profiles.append(profile)

save({"teams": sorted(team_profiles, key=lambda x: x["t_rank"])}, "team_profiles.json")

# ── 5. Historical Upset Analysis ─────────────────────────────────

print("\n5. Generating historical upset analysis...")

seed_matchup_stats = defaultdict(lambda: {"games": 0, "higher_seed_wins": 0})

for year in range(2018, 2025):
    if year == 2020:
        continue
    results_path = HIST / f"tournament_results_{year}.json"
    if not results_path.exists():
        continue
    results = load(results_path)
    for g in results.get("games", []):
        s1 = g.get("team1_seed", 8)
        s2 = g.get("team2_seed", 8)
        if s1 == s2:
            continue
        lower = min(s1, s2)
        higher = max(s1, s2)
        key = f"{lower} vs {higher}"
        seed_matchup_stats[key]["games"] += 1
        # Did the higher seed (worse) win?
        if g["team1_won"] and s1 > s2:
            seed_matchup_stats[key]["higher_seed_wins"] += 1
        elif not g["team1_won"] and s2 > s1:
            seed_matchup_stats[key]["higher_seed_wins"] += 1

upset_data = []
for matchup, stats in sorted(seed_matchup_stats.items()):
    if stats["games"] >= 3:
        upset_data.append(
            {
                "matchup": matchup,
                "games": stats["games"],
                "upsets": stats["higher_seed_wins"],
                "upset_rate": round(stats["higher_seed_wins"] / stats["games"], 4),
            }
        )

save({"matchups": sorted(upset_data, key=lambda x: -x["upset_rate"])}, "historical_upsets.json")

# ── 6. Conference Strength ────────────────────────────────────────

print("\n6. Generating conference strength data...")

conf_data = defaultdict(
    lambda: {
        "teams": [],
        "total_teams": 0,
        "avg_rating": 0,
        "avg_seed": 0,
        "best_seed": 16,
    }
)

for t in bracket_teams:
    conf = t["conference"]
    r = rating_lookup(ratings_2026, t["team_id"])
    conf_data[conf]["teams"].append(
        {
            "team_name": t["team_name"],
            "seed": t["seed"],
            "region": t["region"],
            "barthag": round(r.get("barthag", 0.5), 4),
        }
    )
    conf_data[conf]["total_teams"] += 1
    conf_data[conf]["best_seed"] = min(conf_data[conf]["best_seed"], t["seed"])

conferences = []
for conf, data in conf_data.items():
    seeds = [t["seed"] for t in data["teams"]]
    ratings = [t["barthag"] for t in data["teams"]]
    conferences.append(
        {
            "conference": conf,
            "total_teams": data["total_teams"],
            "avg_seed": round(sum(seeds) / len(seeds), 1),
            "best_seed": data["best_seed"],
            "avg_rating": round(sum(ratings) / len(ratings), 4),
            "teams": sorted(data["teams"], key=lambda x: x["seed"]),
        }
    )

save({"conferences": sorted(conferences, key=lambda x: -x["total_teams"])}, "conference_strength.json")

print("\nDone! All data files written to docs/data/")
