"""Does the ML pipeline beat the site's fitted model? Same games, same metric.

    python -m scripts.compare_pipeline_vs_pit --harness artifacts/backtest_wf.json

Inputs
  A  the TournamentPipeline's per-game tournament predictions, from a
     `backtest-harness --walk-forward` result JSON (per_year_games[year][i]
     = {team1, team2, seed1, seed2, round, outcome, pipeline}).
  B  the browser "Fitted model" (11-feature ridge on margin, Student-t link),
     via its Python mirror pit_production_model.pairwise_for_year, which fits
     strictly on seasons before the one being predicted.
  S  the seed baseline both must beat (loyo_protocol.compute_seed_baseline_probs).

Protocol
  * Only seasons where the harness ran the actual pipeline (per_year_source ==
    "pipeline"); seed-fallback seasons are refused, not scored.
  * Only main-draw games (R64..NCG). First Four games are dropped: B is not
    defined on them (its training matrix excludes them) and A predicts 0.5.
  * Same rows for all three models. A game is skipped if any model cannot
    score it, and the count is reported.
  * Per game: log loss and Brier. Pooled over all games, and per season.
  * Uncertainty: paired bootstrap over games on the A-B difference (5,000
    resamples), and a season-level count of which model won each season.
    Seasons are the honest unit of independence; games within a season
    share a bracket.

Decision rule (fixed before looking): "A beats B" only if the paired 95% CI
on mean per-game log-loss difference excludes zero AND A wins a majority of
seasons. Anything else is "indistinguishable at this sample size".
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.evaluation.loyo_protocol import compute_seed_baseline_probs  # noqa: E402
from src.prediction.pit_production_model import pairwise_for_year  # noqa: E402

MAIN_DRAW = {"R64", "R32", "S16", "E8", "F4", "NCG", "CHAMP", "F2"}
EPS = 1e-6


def _ll(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _brier(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (p - y) ** 2


def load_rows(harness_path: Path) -> Tuple[List[Dict], Dict[str, int]]:
    data = json.loads(harness_path.read_text())
    sources = data.get("per_year_source", {})
    if not data.get("walk_forward", False):
        raise SystemExit(
            "harness result was not produced with --walk-forward; the default LOYO trains on later "
            "seasons and is not comparable to B, which fits strictly on earlier ones"
        )
    rows: List[Dict] = []
    dropped: Dict[str, int] = defaultdict(int)
    for year_s, games in data.get("per_year_games", {}).items():
        year = int(year_s)
        if sources.get(year_s, "pipeline") != "pipeline":
            dropped[f"season {year} was seed_fallback"] += len(games)
            continue
        for g in games:
            rnd = str(g.get("round", "")).upper()
            if rnd not in MAIN_DRAW:
                dropped[f"round={rnd or 'unknown'}"] += 1
                continue
            rows.append({**g, "year": year})
    return rows, dict(dropped)


def _b_team_ids(year: int) -> set:
    """Team ids B has stats for in `year` -- the tournament field as the site
    knows it. Read once per year; membership decides scoreability up front
    rather than parsing it back out of a KeyError."""
    from src.prediction.pit_production_model import STATS

    rows = json.loads(STATS.read_text())["stats_by_year"].get(str(year)) or []
    return {r["team_id"] for r in rows if r.get("team_id")}


def score_b(rows: List[Dict]) -> Tuple[np.ndarray, Dict[str, int]]:
    """B's P(team1 wins) per row; NaN where B cannot score the game."""
    out = np.full(len(rows), np.nan)
    dropped: Dict[str, int] = defaultdict(int)
    by_year: Dict[int, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_year[r["year"]].append(i)
    for year, idxs in by_year.items():
        known = _b_team_ids(year)
        scoreable = [i for i in idxs if rows[i]["team1"] in known and rows[i]["team2"] in known]
        for i in idxs:
            if i not in scoreable:
                unknown = [t for t in (rows[i]["team1"], rows[i]["team2"]) if t not in known]
                dropped[f"{year}: B has no stats for {','.join(unknown)}"] += 1
        if not scoreable:
            continue
        teams = sorted({rows[i]["team1"] for i in scoreable} | {rows[i]["team2"] for i in scoreable})
        pw = pairwise_for_year(year, teams)
        for i in scoreable:
            out[i] = pw[(rows[i]["team1"], rows[i]["team2"])]
    return out, dict(dropped)


def paired_bootstrap(diff: np.ndarray, n: int = 5000, seed: int = 42) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for k in range(n):
        idx = rng.integers(0, len(diff), len(diff))
        means[k] = diff[idx].mean()
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", required=True, help="backtest-harness --walk-forward result JSON")
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    args = ap.parse_args()

    rows, dropped_a = load_rows(Path(args.harness))
    if not rows:
        raise SystemExit("no scoreable pipeline rows in harness result")
    pb, dropped_b = score_b(rows)
    keep = ~np.isnan(pb)
    rows = [r for r, k in zip(rows, keep) if k]
    pb = pb[keep]

    y = np.array([float(r["outcome"]) for r in rows])
    pa = np.array([float(r["pipeline"]) for r in rows])
    ps = compute_seed_baseline_probs(
        np.array([int(r["seed1"]) for r in rows]), np.array([int(r["seed2"]) for r in rows])
    )
    years = np.array([r["year"] for r in rows])

    print("=" * 96)
    print("ML PIPELINE (A) vs SITE FITTED MODEL (B) vs SEED (S) -- identical games, walk-forward, main draw only")
    print("=" * 96)
    print(f"games scored: {len(rows)}   seasons: {sorted(set(years.tolist()))}")
    if dropped_a or dropped_b:
        print("dropped:", {**dropped_a, **dropped_b})
    print()

    metrics = {}
    for name, p in (("A pipeline", pa), ("B fitted", pb), ("S seed", ps)):
        ll, br = _ll(p, y), _brier(p, y)
        metrics[name] = (ll, br)
    ll_s = metrics["S seed"][1].mean()

    print(f"  {'model':<12}{'log loss':>10}{'Brier':>9}{'BSS vs seed':>13}{'accuracy':>10}")
    print("  " + "-" * 54)
    for name, p in (("A pipeline", pa), ("B fitted", pb), ("S seed", ps)):
        ll, br = metrics[name]
        acc = ((p > 0.5) == (y > 0.5)).mean()
        bss = 1 - br.mean() / ll_s
        print(f"  {name:<12}{ll.mean():>10.4f}{br.mean():>9.4f}{bss:>13.3f}{acc:>10.1%}")
    print()

    # Paired differences A - B (negative = A better)
    d_ll = metrics["A pipeline"][0] - metrics["B fitted"][0]
    d_br = metrics["A pipeline"][1] - metrics["B fitted"][1]
    m_ll, lo_ll, hi_ll = paired_bootstrap(d_ll, args.n_bootstrap)
    m_br, lo_br, hi_br = paired_bootstrap(d_br, args.n_bootstrap)
    print("  Paired per-game difference, A minus B (negative favours A), 95% bootstrap CI over games:")
    print(f"    log loss  {m_ll:+.4f}  [{lo_ll:+.4f}, {hi_ll:+.4f}]")
    print(f"    Brier     {m_br:+.4f}  [{lo_br:+.4f}, {hi_br:+.4f}]")
    print()

    # Per season
    print(f"  {'season':<8}{'n':>4}{'A LL':>9}{'B LL':>9}{'S LL':>9}{'A Brier':>9}{'B Brier':>9}{'winner':>9}")
    print("  " + "-" * 66)
    a_wins = b_wins = 0
    season_d = []
    for yr in sorted(set(years.tolist())):
        m = years == yr
        a_ll, b_ll, s_ll = (metrics[k][0][m].mean() for k in ("A pipeline", "B fitted", "S seed"))
        a_br, b_br = metrics["A pipeline"][1][m].mean(), metrics["B fitted"][1][m].mean()
        winner = "A" if a_ll < b_ll else "B"
        a_wins += winner == "A"
        b_wins += winner == "B"
        season_d.append(a_ll - b_ll)
        print(f"  {yr:<8}{int(m.sum()):>4}{a_ll:>9.4f}{b_ll:>9.4f}{s_ll:>9.4f}{a_br:>9.4f}{b_br:>9.4f}{winner:>9}")
    n_seasons = len(season_d)
    sd = np.array(season_d)
    t_stat = sd.mean() / (sd.std(ddof=1) / np.sqrt(n_seasons)) if n_seasons > 1 and sd.std(ddof=1) > 0 else float("nan")
    print(f"\n  seasons won: A {a_wins}, B {b_wins} of {n_seasons}   paired t on season log loss (A-B): t={t_stat:+.2f}")
    print()

    a_beats_b = hi_ll < 0 and a_wins > n_seasons / 2
    b_beats_a = lo_ll > 0 and b_wins > n_seasons / 2
    if a_beats_b:
        verdict = "A (ML pipeline) beats B on both the paired game CI and the season count."
    elif b_beats_a:
        verdict = "B (site fitted model) beats A on both the paired game CI and the season count."
    else:
        verdict = "INDISTINGUISHABLE at this sample size: the paired CI includes zero and/or the season count is split."
    print("  VERDICT:", verdict)
    print("  Rule fixed in advance: a winner needs the paired 95% CI on log loss to exclude zero AND a season majority.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
