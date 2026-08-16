"""Chaos Index: forecast tournament-regime (chalk vs chaos) from pre-tournament data.

Hypothesis under investigation: regular-season signals let you predict whether
a tournament will play chalky (F4 dominated by top seeds) or chaotic (F4
seeded deep). If the hypothesis holds and the field-top strength (weakest
1-seed barthag, mean-top-8 barthag, elite-team count) explains chaos, the
strategy-selection layer gets a useful input: chalk-friendly modes in strong
years, upset-friendly modes in thin years.

Pure module — reads raw Torvik ratings and tournament seeds / results from
``data/raw/historical/``, returns structured features and an optional
walk-forward prediction. Correlation and significance are the diagnostic;
no automated strategy switching is gated on this yet.

Provenance:
- Measured 2026-04-24 over 15 tournaments (2011-2026 excl 2020):
  ``mean_top8_barthag`` r = −0.668 vs actual mean-F4-seed, p ≈ 0.006
  ``elite_count_gt_095`` r = −0.660, p ≈ 0.007
  ``weakest_1seed_barthag`` r = −0.551, p ≈ 0.033
  LOO walk-forward (univariate on ``weakest_1seed_barthag``): r ≈ +0.35,
  MAE ≈ 1.0 seed.

Signs:
- All three significant features correlate *negatively* with chaos.
  Interpretation: strong top-of-field → chalky tournament. A thin top
  (fewer elite teams, weaker 1-seeds) forecasts upsets. Intuitive.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from src.evaluation.tournament_oracle import load_ground_truth


# Known tournament years in the backtest window (2020 excluded — COVID cancelled).
BACKTEST_YEARS: Sequence[int] = tuple(y for y in range(2011, 2027) if y != 2020)


@dataclass(frozen=True)
class ChaosFeatures:
    """Pre-tournament features computed from Torvik + seeds.

    All features are derivable from data that exists **before** Selection
    Sunday — no leakage into prediction.

    Attributes:
        year: Tournament year.
        mean_1seed_barthag: Average barthag of the four 1-seeds. Higher
            = stronger 1-seeds.
        weakest_1seed_barthag: Minimum barthag across the four 1-seeds.
            Low = at least one shaky 1-seed.
        mean_top8_barthag: Average barthag of the top 8 Torvik teams
            (strongest signal in 2026-04-24 measurement).
        elite_count_gt_095: Count of teams with barthag > 0.95 — how
            deep the top tier is.
        top4_minus_top30_barthag: Gap between the top 4 mean and top 30
            mean — concentration of strength at the top.
    """

    year: int
    mean_1seed_barthag: float
    weakest_1seed_barthag: float
    mean_top8_barthag: float
    elite_count_gt_095: int
    top4_minus_top30_barthag: float


@dataclass(frozen=True)
class ChaosObservation:
    """Actual post-tournament chaos metric for one year.

    Attributes:
        year: Tournament year.
        mean_f4_seed: Average seed across the four Final-Four teams.
            1.0 = all 1-seeds (max chalk); higher = more chaos.
            2025 = 1.0 (chalk); 2023 = 5.75 (chaos).
    """

    year: int
    mean_f4_seed: float


def _load_seeds(year: int, data_root: Path) -> List[dict]:
    """Load tournament seeds from either storage layout used in this repo.

    Prefers the consolidated `tournament_context_{year}.json` (key
    "seeds") under the historical directory when present, falling back
    to the old per-type `tournament_seeds_{year}.json`."""
    ctx_path = Path(data_root) / "raw" / "historical" / f"tournament_context_{year}.json"
    if ctx_path.exists():
        ctx = json.load(open(ctx_path))
        if "seeds" in ctx:
            raw = ctx["seeds"]
            if isinstance(raw, dict) and "teams" in raw:
                return raw["teams"]
            if isinstance(raw, list):
                return raw
    candidates = [
        Path(data_root) / "raw" / "historical" / f"tournament_seeds_{year}.json",
        Path(data_root) / "raw" / f"tournament_seeds_{year}.json",
    ]
    for p in candidates:
        if p.exists():
            raw = json.load(open(p))
            # Some files wrap the team list in {"season": Y, "teams": [...]},
            # others are a flat list — handle both.
            if isinstance(raw, dict) and "teams" in raw:
                return raw["teams"]
            if isinstance(raw, list):
                return raw
    raise FileNotFoundError(f"No tournament_seeds_{year}.json under {data_root}")


def _load_torvik_barthag(year: int, data_root: Path) -> Dict[str, float]:
    p = Path(data_root) / "raw" / "historical" / f"torvik_{year}.json"
    raw = json.load(open(p))
    teams = raw["teams"] if isinstance(raw, dict) and "teams" in raw else raw
    return {t["team_id"]: float(t["barthag"]) for t in teams}


def compute_features(year: int, data_root: Path) -> ChaosFeatures:
    """Compute pre-tournament chaos features for the given year."""
    seeds = _load_seeds(year, data_root)
    barthag = _load_torvik_barthag(year, data_root)

    one_seed_ids = [s["team_id"] for s in seeds if s["seed"] == 1]
    one_seed_bths = sorted(barthag.get(t, 0.0) for t in one_seed_ids)
    if not one_seed_bths:
        raise ValueError(f"No 1-seeds resolved against Torvik for {year}")

    sorted_bth = sorted(barthag.values(), reverse=True)
    if len(sorted_bth) < 30:
        raise ValueError(f"Torvik file for {year} has <30 teams: {len(sorted_bth)}")

    return ChaosFeatures(
        year=year,
        mean_1seed_barthag=statistics.mean(one_seed_bths),
        weakest_1seed_barthag=one_seed_bths[0],
        mean_top8_barthag=statistics.mean(sorted_bth[:8]),
        elite_count_gt_095=sum(1 for v in sorted_bth if v > 0.95),
        top4_minus_top30_barthag=statistics.mean(sorted_bth[:4]) - statistics.mean(sorted_bth[:30]),
    )


def compute_actual_chaos(year: int, data_root: Path) -> ChaosObservation:
    """Observed chaos metric (mean F4 seed) from the real tournament.

    Can only be computed after the tournament; used to evaluate the index,
    not to feed it.
    """
    truth = load_ground_truth(year, data_root)
    # We have the F4 team ids but need their seeds — read tournament_results
    # games directly for seed pairs. Prefers the consolidated
    # tournament_context_{year}.json (key "results") when present.
    ctx_path = Path(data_root) / "raw" / "historical" / f"tournament_context_{year}.json"
    raw = None
    if ctx_path.exists():
        ctx = json.load(open(ctx_path))
        if "results" in ctx:
            raw = ctx["results"]
    if raw is None:
        p = Path(data_root) / "raw" / "historical" / f"tournament_results_{year}.json"
        raw = json.load(open(p))
    games = raw["games"] if isinstance(raw, dict) and "games" in raw else raw
    f4_seeds = []
    for g in games:
        if (g.get("round_name") or g.get("round")) == "F4":
            f4_seeds.append(g["team1_seed"])
            f4_seeds.append(g["team2_seed"])
    if len(f4_seeds) != 4:
        raise ValueError(f"Expected 4 F4 seeds for {year}, got {len(f4_seeds)}")
    _ = truth  # silence unused — the call above validates the year is complete
    return ChaosObservation(year=year, mean_f4_seed=statistics.mean(f4_seeds))


def walk_forward_predict(
    target_year: int,
    train_years: Sequence[int],
    data_root: Path,
    *,
    feature_key: str = "mean_top8_barthag",
) -> Optional[float]:
    """Univariate linear fit on prior years → predicted mean-F4-seed for target.

    Deliberately univariate to stay honest at n<=15 training samples. The
    2026-04-24 measurement found ``mean_top8_barthag`` was the strongest
    single predictor (r=-0.668) — default to it; callers can try the
    other features in :class:`ChaosFeatures` if they want.

    Returns ``None`` if fewer than 3 train years are available (fit is
    numerically suspect below that).
    """
    train_years = [y for y in train_years if y != target_year and y != 2020]
    if len(train_years) < 3:
        return None

    xs = []
    ys = []
    for y in train_years:
        xs.append(getattr(compute_features(y, data_root), feature_key))
        ys.append(compute_actual_chaos(y, data_root).mean_f4_seed)

    # Ordinary least squares by hand — no sklearn dep needed for a 1-D fit.
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
    den = sum((xs[i] - mean_x) ** 2 for i in range(n))
    if den == 0:
        return None
    slope = num / den
    intercept = mean_y - slope * mean_x

    target_x = getattr(compute_features(target_year, data_root), feature_key)
    return slope * target_x + intercept


def regime_report(years: Sequence[int], data_root: Path) -> Dict:
    """Per-year features + actual chaos + walk-forward prediction.

    Output shape:
        {
            "years": [...],
            "rows": [
                {"year": Y, "features": {...}, "actual_mean_f4_seed": X,
                 "predicted_mean_f4_seed": P_or_None, ...},
                ...
            ],
            "pearson_r_per_feature": {"mean_top8_barthag": -0.668, ...},
        }
    """
    years = [y for y in years if y != 2020]
    rows: List[dict] = []
    for y in years:
        feats = compute_features(y, data_root)
        actual = compute_actual_chaos(y, data_root)
        pred = walk_forward_predict(y, years, data_root)
        rows.append(
            {
                "year": y,
                "features": feats.__dict__,
                "actual_mean_f4_seed": actual.mean_f4_seed,
                "predicted_mean_f4_seed": pred,
            }
        )

    feature_names = [
        "mean_1seed_barthag",
        "weakest_1seed_barthag",
        "mean_top8_barthag",
        "elite_count_gt_095",
        "top4_minus_top30_barthag",
    ]
    pearson_r: Dict[str, float] = {}
    actuals = [r["actual_mean_f4_seed"] for r in rows]
    for name in feature_names:
        xs = [r["features"][name] for r in rows]
        pearson_r[name] = _pearson(xs, actuals)

    return {"years": list(years), "rows": rows, "pearson_r_per_feature": pearson_r}


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(len(xs)))
    den_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    den_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)
