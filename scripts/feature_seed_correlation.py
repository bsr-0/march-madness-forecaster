#!/usr/bin/env python3
"""Correlation analysis: production features vs tournament seed.

Council recommended first action: determine whether the 7 (actually 10)
production features are just proxies for seed strength. If r > 0.7 for
all features, BSS ≈ 0 vs seed baseline is tautological.
"""

import json
import os
import sys
from pathlib import Path
from scripts._common import load_seeds  # noqa: F401

import numpy as np

DATA_DIR = Path("data/raw")
HIST_DIR = DATA_DIR / "historical"

# Years matching production training window
YEARS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]


def load_torvik(year):
    """Load Torvik metrics. Returns {team_id: {metric: value}}."""
    path = DATA_DIR / f"torvik_{year}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    teams = data.get("teams", data if isinstance(data, list) else [])
    return {t["team_id"]: t for t in teams}


def load_four_factors(year):
    """Load four factors data. Returns {team_id: {metric: value}}."""
    path = DATA_DIR / f"torvik_four_factors_{year}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    # Can be team-keyed dict or list
    if isinstance(data, dict) and not data.get("teams"):
        return data
    teams = data.get("teams", data if isinstance(data, list) else [])
    return {t.get("team_id", ""): t for t in teams}


def load_advanced_metrics(year):
    """Load advanced metrics. Returns {team_id: {metric: value}}."""
    path = DATA_DIR / f"advanced_metrics_{year}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    teams = data.get("teams", data if isinstance(data, list) else [])
    return {t["team_id"]: t for t in teams}


def load_win_pct(year):
    """Compute win% from historical game records. Returns {team_id: win_pct}."""
    path = HIST_DIR / f"historical_games_{year}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    games = data.get("games", [])
    wins = {}
    total = {}
    for g in games:
        t1, t2 = g.get("team1_id"), g.get("team2_id")
        s1, s2 = g.get("team1_score", 0), g.get("team2_score", 0)
        if not t1 or not t2 or s1 == s2:
            continue
        for tid in [t1, t2]:
            wins.setdefault(tid, 0)
            total.setdefault(tid, 0)
        total[t1] += 1
        total[t2] += 1
        if s1 > s2:
            wins[t1] += 1
        else:
            wins[t2] += 1
    return {tid: wins[tid] / total[tid] for tid in total if total[tid] > 0}


def load_external_ratings(year):
    """Load external rating composite. Returns {team_id: avg_normalized_rating}."""
    path = HIST_DIR / f"external_ratings_{year}.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    systems = data.get("systems", {})
    team_ratings = {}
    team_counts = {}
    for items in systems.values():
        if not isinstance(items, list):
            continue
        for item in items:
            tid = item.get("team_id")
            rating = item.get("normalized") or item.get("rating")
            if tid and rating is not None:
                try:
                    r = float(rating)
                    if np.isfinite(r):
                        team_ratings[tid] = team_ratings.get(tid, 0.0) + r
                        team_counts[tid] = team_counts.get(tid, 0) + 1
                except (TypeError, ValueError):
                    pass
    return {tid: team_ratings[tid] / team_counts[tid] for tid in team_counts}


def safe_float(val, default=np.nan):
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def collect_features():
    """Collect team-level features + seed across all years."""
    rows = []

    for year in YEARS:
        seeds = load_seeds(year)
        torvik = load_torvik(year)
        four_factors = load_four_factors(year)
        adv = load_advanced_metrics(year)
        win_pcts = load_win_pct(year)
        ext_ratings = load_external_ratings(year)

        if not seeds:
            print(f"  {year}: no seed data, skipping")
            continue

        matched = 0
        for team_id, seed in seeds.items():
            tv = torvik.get(team_id, {})
            ff = four_factors.get(team_id, {})
            am = adv.get(team_id, {})

            # Try to get features from best available source
            adj_off = safe_float(tv.get("adj_offensive_efficiency") or am.get("adj_offensive_efficiency"))
            adj_def = safe_float(tv.get("adj_defensive_efficiency") or am.get("adj_defensive_efficiency"))
            adj_tempo = safe_float(tv.get("adj_tempo") or am.get("adj_tempo"))
            sos_adj_em = safe_float(am.get("sos_adj_em") or tv.get("sos_adj_em", tv.get("sos")))
            barthag = safe_float(tv.get("barthag") or am.get("barthag"))
            t_rank = safe_float(tv.get("t_rank") or am.get("t_rank") or am.get("overall_rank"))

            # Win pct from game records
            win_pct = safe_float(win_pcts.get(team_id))

            # External rating composite (proxy for Elo)
            ext_rating = safe_float(ext_ratings.get(team_id))

            # Four factors
            efg = safe_float(ff.get("effective_fg_pct") or tv.get("effective_fg_pct"))
            to_rate = safe_float(ff.get("turnover_rate") or tv.get("turnover_rate"))
            orb_rate = safe_float(ff.get("offensive_reb_rate") or tv.get("offensive_reb_rate"))
            ft_rate = safe_float(ff.get("free_throw_rate") or tv.get("free_throw_rate"))
            opp_efg = safe_float(ff.get("opp_effective_fg_pct") or tv.get("opp_effective_fg_pct"))
            opp_to_rate = safe_float(ff.get("opp_turnover_rate") or tv.get("opp_turnover_rate"))

            # Adjusted efficiency margin (proxy for overall quality)
            adj_em = adj_off - adj_def if np.isfinite(adj_off) and np.isfinite(adj_def) else np.nan

            rows.append(
                {
                    "year": year,
                    "team_id": team_id,
                    "seed": seed,
                    # Features available in Torvik/advanced data
                    "adj_off_eff": adj_off,
                    "adj_def_eff": adj_def,
                    "adj_tempo": adj_tempo,
                    "sos_adj_em": sos_adj_em,
                    "win_pct": win_pct,
                    "efg_pct": efg,
                    "orb_rate": orb_rate,
                    "opp_to_rate": opp_to_rate,
                    "ft_rate": ft_rate,
                    "opp_efg_pct": opp_efg,
                    # Composite metrics
                    "adj_em": adj_em,
                    "barthag": barthag,
                    "t_rank": t_rank,
                    "ext_rating": ext_rating,
                }
            )
            matched += 1

        print(f"  {year}: {matched}/{len(seeds)} teams matched")

    return rows


def pearson_r(x, y):
    """Compute Pearson r for finite values only."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return np.nan, 0
    x_clean, y_clean = x[mask], y[mask]
    n = len(x_clean)
    r = np.corrcoef(x_clean, y_clean)[0, 1]
    return r, n


# Features that map to real production SIMPLE_FEATURE_SET entries and whose
# raw values are available in the offline JSON snapshots. Used by both the
# main() printer and the O8 regression test.
PRODUCTION_FEATURE_MAP = {
    "adj_off_eff": "diff_adj_off_eff",
    "adj_def_eff": "diff_adj_def_eff",
    "sos_adj_em": "diff_sos_adj_em",
    "win_pct": "diff_win_pct",
    "adj_tempo": "diff_adj_tempo",
    "orb_rate": "diff_orb_rate",
    "opp_to_rate": "diff_opp_to_rate",
}


def compute_production_feature_correlations():
    """Return {production_feature_name: (r, n)} correlating each production
    feature against tournament seed (1=best, 16=worst).

    Reads historical tournament seeds + Torvik JSON snapshots already
    committed to data/. Fully offline.

    Used by:
      - scripts/feature_seed_correlation.py main() to print the diagnostic.
      - tests/test_feature_seed_collinearity.py to enforce the anti-tautology
        gate (no single production feature has |r| >= 0.85 with seed).

    Closes COUNCIL_LESSONS.md §2 O8.
    """
    rows = collect_features()
    if not rows:
        return {}
    seeds = np.array([row["seed"] for row in rows], dtype=float)
    results = {}
    for raw_name, prod_name in PRODUCTION_FEATURE_MAP.items():
        values = np.array([row.get(raw_name, np.nan) for row in rows], dtype=float)
        r, n = pearson_r(values, seeds)
        results[prod_name] = (float(r) if np.isfinite(r) else float("nan"), int(n))
    return results


def main():
    print("=" * 70)
    print("FEATURE-SEED CORRELATION ANALYSIS")
    print("Council Directive: Determine if production features are seed proxies")
    print("=" * 70)
    print(f"\nYears: {YEARS} (production training window)")
    print(f"Collecting team features for tournament teams...\n")

    rows = collect_features()
    if not rows:
        print("ERROR: No data collected. Check data files.")
        sys.exit(1)

    print(f"\nTotal team-year observations: {len(rows)}")

    # Convert to arrays
    seeds = np.array([r["seed"] for r in rows], dtype=float)

    # Features to correlate with seed
    # Note: seed=1 is best, seed=16 is worst, so "good" features
    # (high off eff, high win%) should correlate NEGATIVELY with seed
    feature_names = [
        # === AVAILABLE FROM RAW DATA (direct match to production features) ===
        "adj_off_eff",  # → diff_adj_off_eff in production
        "adj_def_eff",  # → diff_adj_def_eff in production (lower=better, so positive corr)
        "sos_adj_em",  # → diff_sos_adj_em in production
        "win_pct",  # → diff_win_pct in production
        "adj_tempo",  # → diff_adj_tempo in SIMPLE_FEATURE_SET
        "orb_rate",  # → diff_orb_rate in SIMPLE_FEATURE_SET
        "opp_to_rate",  # → diff_opp_to_rate in SIMPLE_FEATURE_SET
        # === COMPOSITE METRICS (strong seed proxies by construction) ===
        "adj_em",  # Efficiency margin (off - def)
        "barthag",  # Torvik win probability
        "t_rank",  # Torvik ranking
        "ext_rating",  # External rating composite (Elo proxy)
        # === FOUR FACTORS (additional) ===
        "efg_pct",  # Effective FG%
        "ft_rate",  # Free throw rate (proxy for free_throw_pct)
        "opp_efg_pct",  # Opponent eFG% (lower=better defense)
    ]

    print("\n" + "=" * 70)
    print("PEARSON CORRELATIONS: Team Feature vs Tournament Seed (1=best)")
    print("=" * 70)
    print(f"{'Feature':<25} {'r':>8} {'|r|':>8} {'N':>6}  {'Interpretation'}")
    print("-" * 70)

    results = []
    for fname in feature_names:
        vals = np.array([r.get(fname, np.nan) for r in rows], dtype=float)
        r, n = pearson_r(vals, seeds)
        results.append((fname, r, abs(r), n))

    # Sort by |r| descending
    results.sort(key=lambda x: -abs(x[1]) if np.isfinite(x[1]) else 0)

    for fname, r, abs_r, n in results:
        if not np.isfinite(r):
            interp = "INSUFFICIENT DATA"
        elif abs_r >= 0.7:
            interp = "STRONG SEED PROXY"
        elif abs_r >= 0.5:
            interp = "MODERATE SEED PROXY"
        elif abs_r >= 0.3:
            interp = "WEAK-MODERATE"
        elif abs_r >= 0.15:
            interp = "WEAK (some independence)"
        else:
            interp = "INDEPENDENT of seed"
        print(f"  {fname:<23} {r:>+8.3f} {abs_r:>8.3f} {n:>6}  {interp}")

    # === NOT AVAILABLE IN RAW DATA ===
    print("\n" + "=" * 70)
    print("FEATURES NOT IN RAW DATA (computed by proprietary engine)")
    print("=" * 70)
    missing = [
        ("elo_rating", "diff_elo_rating", "Elo rating (full-season trajectory) — SIMPLE_FEATURE_SET #1"),
        ("total_warp", "diff_total_warp", "Player WARP (wins above replacement) — SIMPLE_FEATURE_SET #2"),
        ("momentum", "diff_momentum", "Last-10-game momentum — SIMPLE_FEATURE_SET #4"),
        ("top5_rapm", "diff_top5_rapm", "Star player RAPM — SIMPLE_FEATURE_SET #8"),
        ("preseason_ap_rank", "diff_preseason_ap_rank", "Preseason AP rank — SIMPLE_FEATURE_SET #9"),
        ("free_throw_pct", "diff_free_throw_pct", "Free throw % (we have ft_rate as proxy)"),
    ]
    for raw_name, prod_name, desc in missing:
        print(f"  {raw_name:<23} — {desc}")

    # === SEED SELF-CORRELATION CHECK ===
    print("\n" + "=" * 70)
    print("SEED-ADJACENT FEATURES IN PRODUCTION MODEL")
    print("=" * 70)
    print("""
The production model uses 'diff_' features (team1 - team2 differentials).
If a team-level feature correlates at |r| > 0.7 with seed, then the
corresponding diff_ feature will correlate strongly with seed_diff,
making it a PROXY for seed, not an independent predictor.

Key threshold: |r| > 0.7 → seed proxy (BSS ≈ 0 is expected)
               |r| < 0.3 → genuinely independent signal
               |r| 0.3-0.7 → partial independence (may add marginal value)
""")

    # Summary statistics
    available_r = [(fname, r) for fname, r, _, _ in results if np.isfinite(r)]
    prod_features_available = [
        "adj_off_eff",
        "adj_def_eff",
        "sos_adj_em",
        "win_pct",
        "adj_tempo",
        "orb_rate",
        "opp_to_rate",
    ]
    print("PRODUCTION FEATURES SUMMARY (available in raw data):")
    print("-" * 50)
    strong_proxies = []
    independent = []
    for fname, r, abs_r, n in results:
        if fname in prod_features_available and np.isfinite(r):
            category = "PROXY" if abs_r >= 0.5 else "PARTIAL" if abs_r >= 0.3 else "INDEPENDENT"
            if abs_r >= 0.5:
                strong_proxies.append(fname)
            elif abs_r < 0.3:
                independent.append(fname)
            print(f"  {fname:<23} |r|={abs_r:.3f}  → {category}")

    print(f"\n  Strong seed proxies (|r| >= 0.5): {len(strong_proxies)}/{len(prod_features_available)}")
    print(f"  Independent features (|r| < 0.3): {len(independent)}/{len(prod_features_available)}")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    all_proxy = all(abs(r) >= 0.5 for fname, r in available_r if fname in prod_features_available)
    some_independent = any(abs(r) < 0.3 for fname, r in available_r if fname in prod_features_available)

    if all_proxy:
        print("""
ALL available production features are moderate-to-strong seed proxies.
BSS ≈ 0 is likely TAUTOLOGICAL — the model is regressing seed proxies
against a seed baseline. The fix is DIFFERENT FEATURES, not more complex models.
""")
    elif some_independent:
        print("""
MIXED RESULT: Some features are seed-correlated, but others show genuine
independence. BSS ≈ 0 is NOT fully explained by feature collinearity.
The information-theoretic ceiling may be real for these features.
Next step: investigate whether the independent features individually
beat the seed baseline in LOYO evaluation.
""")
    else:
        print("""
Most features are moderate seed proxies (|r| 0.3-0.7). BSS ≈ 0 is
PARTIALLY explained by collinearity. There may be marginal independent
signal, but it's small. Consider features from genuinely different
information sources (player tracking, injury timing, betting markets).
""")


if __name__ == "__main__":
    main()
