#!/usr/bin/env python3
"""Phase 0: Chalk/upset year signal correlation audit.

For each tournament year with real outcome data (2016+), compute six Tier 1
year-level signals derivable from the pre-game artifact, then correlate each
against that year's mean residual (actual_outcome - torvik_prob).

All signals use only pre-game observations (torvik, seeds, odds) — they are
LOYO-safe because they describe the tournament field at Selection Sunday,
before any games are played.

Output: artifacts/chalk_upset_signal_audit.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

PERGAME_ARTIFACT = REPO_ROOT / "artifacts" / "loyo_pergame_predictions.json"
OUTPUT_PATH = REPO_ROOT / "artifacts" / "chalk_upset_signal_audit.json"

# Only years with real outcome data (pre-2016 all have outcome=1, which is an
# artifact of the data encoding — team1 is always the winner in those years).
REAL_OUTCOME_START = 2016


def _compute_year_signals(games: list[dict]) -> dict[str, Optional[float]]:
    """Compute all Tier 1 pre-game chalk signals for a single year's 63 games."""
    n = len(games)
    if n == 0:
        return {}

    torvik = np.array([g["torvik"] for g in games], dtype=float)

    # Seed signals
    seed1 = np.array([g["seed1"] for g in games], dtype=float)
    seed2 = np.array([g["seed2"] for g in games], dtype=float)

    # Market signal — only games with real odds data
    odds_games = [g for g in games if g.get("odds") and float(g["odds"]) > 0]
    odds_arr = np.array([float(g["odds"]) for g in odds_games], dtype=float) if odds_games else None

    # Massey signal
    massey_games = [g for g in games if g.get("massey_avg") and float(g["massey_avg"]) > 0]
    massey_arr = np.array([float(g["massey_avg"]) for g in massey_games], dtype=float) if massey_games else None

    # ---- Signal 1: torvik confidence mean ----
    # mean(|torvik - 0.5| * 2) — how confident is torvik about favorites?
    # Range [0, 1]. Higher = more confident = stronger favorites expected.
    torvik_conf_mean = float(np.mean(np.abs(torvik - 0.5) * 2.0))

    # ---- Signal 2: fraction of games where torvik > 0.70 ----
    # High fraction = many blowout-expected matchups.
    pct_torvik_gt70 = float(np.mean(torvik > 0.70))

    # ---- Signal 3: mean absolute seed gap ----
    # mean(|seed2 - seed1| / 15). High = lopsided seeding = chalk expected.
    mean_seed_gap = float(np.mean(np.abs(seed2 - seed1) / 15.0))

    # ---- Signal 4: market confidence mean ----
    # mean(|odds - 0.5| * 2). Only games with real odds. None when no coverage.
    market_conf_mean: Optional[float] = None
    if odds_arr is not None and len(odds_arr) > 0:
        market_conf_mean = float(np.mean(np.abs(odds_arr - 0.5) * 2.0))

    # ---- Signal 5: torvik-market agreement rate ----
    # Fraction of games where |odds - torvik| < 0.05. High = lock-step confidence.
    torvik_mkt_agree_rate: Optional[float] = None
    if odds_arr is not None and len(odds_arr) > 0:
        odds_torvik = np.array([float(g["torvik"]) for g in odds_games], dtype=float)
        torvik_mkt_agree_rate = float(np.mean(np.abs(odds_arr - odds_torvik) < 0.05))

    # ---- Signal 6: massey confidence mean ----
    # mean(|massey_avg - 0.5| * 2). Second rating system as confirmation.
    massey_conf_mean: Optional[float] = None
    if massey_arr is not None and len(massey_arr) > 0:
        massey_conf_mean = float(np.mean(np.abs(massey_arr - 0.5) * 2.0))

    # ---- Signal 7: max torvik confidence ----
    # The maximum single-game confidence in a year. Captures whether there's
    # at least one "sure thing" matchup driving the average up.
    torvik_conf_max = float(np.max(np.abs(torvik - 0.5) * 2.0))

    # ---- Signal 8: torvik confidence variance ----
    # High variance = mixed field (some blowouts, some toss-ups). Low variance =
    # uniformly confident or uniformly uncertain.
    torvik_conf_std = float(np.std(np.abs(torvik - 0.5) * 2.0))

    # ---- Signal 9: mean torvik prob (raw) ----
    # Simple mean of torvik probs. High = forecasters think favorites dominate.
    mean_torvik_prob = float(np.mean(torvik))

    return {
        "torvik_conf_mean": torvik_conf_mean,
        "pct_torvik_gt70": pct_torvik_gt70,
        "mean_seed_gap": mean_seed_gap,
        "market_conf_mean": market_conf_mean,
        "torvik_mkt_agree_rate": torvik_mkt_agree_rate,
        "massey_conf_mean": massey_conf_mean,
        "torvik_conf_max": torvik_conf_max,
        "torvik_conf_std": torvik_conf_std,
        "mean_torvik_prob": mean_torvik_prob,
        "n_games": n,
        "n_with_odds": len(odds_games),
        "n_with_massey": len(massey_games) if massey_games else 0,
    }


def _compute_year_outcome(games: list[dict]) -> dict[str, float]:
    """Compute actual outcome stats for a year."""
    n = len(games)
    outcomes = np.array([g["outcome"] for g in games], dtype=float)
    torvik = np.array([g["torvik"] for g in games], dtype=float)
    residuals = outcomes - torvik
    return {
        "mean_residual": float(np.mean(residuals)),
        "std_residual": float(np.std(residuals)),
        "team1_win_rate": float(np.mean(outcomes)),
        "mean_torvik": float(np.mean(torvik)),
    }


def spearman_r(x: list[float], y: list[float]) -> tuple[float, float]:
    """Compute Spearman rank correlation and approximate p-value (two-tailed)."""
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    from scipy.stats import spearmanr

    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def _pearson_r(x: list[float], y: list[float]) -> float:
    """Pearson correlation as a secondary check."""
    n = len(x)
    if n < 3:
        return float("nan")
    xarr = np.array(x, dtype=float)
    yarr = np.array(y, dtype=float)
    xd = xarr - xarr.mean()
    yd = yarr - yarr.mean()
    denom = np.sqrt((xd**2).sum() * (yd**2).sum())
    if denom < 1e-12:
        return float("nan")
    return float((xd * yd).sum() / denom)


def main() -> None:
    data = json.loads(PERGAME_ARTIFACT.read_text())
    data = {int(year): rows for year, rows in data.items()}

    real_years = sorted(y for y in data if y >= REAL_OUTCOME_START)
    print(f"Years with real outcome data: {real_years}")
    print()

    year_records: list[dict] = []
    for year in real_years:
        games = data[year]
        signals = _compute_year_signals(games)
        outcomes = _compute_year_outcome(games)
        year_records.append({"year": year, **signals, **outcomes})

    # ---- Correlation table ----
    signal_names = [
        "torvik_conf_mean",
        "pct_torvik_gt70",
        "mean_seed_gap",
        "market_conf_mean",
        "torvik_mkt_agree_rate",
        "massey_conf_mean",
        "torvik_conf_max",
        "torvik_conf_std",
        "mean_torvik_prob",
    ]

    mean_residuals = [r["mean_residual"] for r in year_records]

    correlations: list[dict] = []
    for sig in signal_names:
        values = [r[sig] for r in year_records if r.get(sig) is not None]
        resids = [r["mean_residual"] for r in year_records if r.get(sig) is not None]
        if len(values) < 4:
            print(f"  {sig:<30}  SKIPPED (only {len(values)} data points with coverage)")
            correlations.append(
                {"signal": sig, "n": len(values), "spearman_r": None, "pearson_r": None, "p_value": None}
            )
            continue
        sp_r, p_val = spearman_r(values, resids)
        pe_r = _pearson_r(values, resids)
        correlations.append(
            {
                "signal": sig,
                "n": len(values),
                "spearman_r": sp_r,
                "pearson_r": pe_r,
                "p_value": p_val,
            }
        )

    # Sort by absolute Spearman r descending
    def sort_key(c: dict) -> float:
        r = c.get("spearman_r")
        return -abs(r) if r is not None else 0.0

    correlations.sort(key=sort_key)

    print("Signal correlation with mean_residual (Spearman):")
    print(f"  {'Signal':<30}  {'n':>3}  {'Spearman r':>11}  {'Pearson r':>10}  {'p-value':>8}  Direction")
    print("  " + "-" * 85)
    for c in correlations:
        if c["spearman_r"] is None:
            continue
        r = c["spearman_r"]
        direction = "← more chalk" if r < 0 else "→ more chalk"
        p_str = f"{c['p_value']:.3f}" if c["p_value"] is not None else "  n/a"
        print(f"  {c['signal']:<30}  {c['n']:>3}  {r:>+.4f}      {c['pearson_r']:>+.4f}    {p_str}  {direction}")

    print()

    # ---- Year-by-year breakdown ----
    print("Year-by-year signal values and residuals:")
    print(
        f"  {'Year':>4}  {'mean_resid':>11}  {'tv_conf':>8}  {'pct>0.7':>8}  {'seed_gap':>9}  {'mkt_conf':>9}  {'massey_c':>9}"
    )
    print("  " + "-" * 75)
    for r in year_records:
        mkt = f"{r['market_conf_mean']:.3f}" if r.get("market_conf_mean") is not None else "  n/a"
        msy = f"{r['massey_conf_mean']:.3f}" if r.get("massey_conf_mean") is not None else "  n/a"
        print(
            f"  {r['year']:>4}  {r['mean_residual']:>+.4f}      "
            f"{r['torvik_conf_mean']:>.3f}    "
            f"{r['pct_torvik_gt70']:>.3f}    "
            f"{r['mean_seed_gap']:>.4f}    "
            f"{mkt:>6}    "
            f"{msy:>6}"
        )

    print()

    # ---- Decision guidance ----
    passing_signals = [c for c in correlations if c.get("spearman_r") is not None and c["spearman_r"] < -0.25]
    if passing_signals:
        best = passing_signals[0]
        print(f"PHASE 0 RESULT: {len(passing_signals)} signal(s) meet the r < -0.25 threshold.")
        print(f"  Best: {best['signal']}  r={best['spearman_r']:+.4f}  p={best['p_value']:.3f}")
        print("  → Proceed to Phase 1 (architecture) and Phase 2 (implementation).")
    else:
        print("PHASE 0 RESULT: No signal meets the r < -0.25 threshold.")
        print("  → Signals are noise. BSS +0.133 is the ceiling with available data.")
        print("  → Consider Tier 2 signals (KenPom AdjEM, conf tourney upsets) before closing.")

    print()

    # ---- Write artifact ----
    output = {
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "real_outcome_years": real_years,
        "n_years": len(real_years),
        "signal_correlations": correlations,
        "year_records": year_records,
        "decision": {
            "threshold_r": -0.25,
            "passing_signals": [c["signal"] for c in passing_signals],
            "proceed_to_phase_1": len(passing_signals) > 0,
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(f"Artifact written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
