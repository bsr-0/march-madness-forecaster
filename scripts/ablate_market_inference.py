"""Ablate: does providing market odds at inference time improve the Kaggle correction model?

The admitted torvik_corrected_recent5_conservative model is trained with 8 features
(seed_gap, torvik_confidence, market_prob, market_disagree, elo_disagree, round_num).
But the LOYO backtest in kaggle_torvik_submission.py calls predict_one(torvik, s1, s2)
WITHOUT market_prob — market_prob defaults to torvik (market_disagree=0).

Since loyo_pergame_predictions.json has 100% market odds coverage (2009-2026),
we can evaluate whether passing actual market odds at inference improves BSS.

Run:
    python scripts/ablate_market_inference.py

Compares per-year Brier for 3 variants:
    baseline   — market_prob=None at inference (current backtest behavior)
    mkt        — market_prob=odds field from records
    mkt+elo    — market_prob=odds, elo_prob=elo, round_num from records
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.prediction.torvik_correction import (
    fit_torvik_correction_from_year_records,
    TorvikCorrectionConfig,
    ROUND_ORDER,
)

RECORDS_PATH = REPO_ROOT / "artifacts" / "loyo_pergame_predictions.json"

# Admitted baseline params (torvik_corrected_recent5_conservative)
RIDGE = 20.0
MAX_CORRECTION = 0.06
RECENT_YEAR_COUNT = 5
RECENT_TOTAL_RATIO = 1.0
CLIP_LO, CLIP_HI = 0.01, 0.99

# Evaluation window matching RUNBOOK priority-1 (2023-2025) and full recent-5
EVAL_YEARS = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]


def seed_brier(s1: int, s2: int) -> float:
    """Seed-implied probability (logistic of seed gap)."""
    gap = (s2 - s1) / 15.0
    return 1.0 / (1.0 + np.exp(-4.0 * gap))


def main() -> None:
    with open(RECORDS_PATH) as f:
        all_records: dict[int, list[dict]] = {int(k): v for k, v in json.load(f).items()}

    config = TorvikCorrectionConfig(
        clip_lo=CLIP_LO,
        clip_hi=CLIP_HI,
        ridge=RIDGE,
        max_correction=MAX_CORRECTION,
    )

    print(f"\n{'=' * 90}")
    print("MARKET-AT-INFERENCE ABLATION — torvik_corrected_recent5_conservative params")
    print(f"ridge={RIDGE}, max_correction={MAX_CORRECTION}, recent_year_count={RECENT_YEAR_COUNT}")
    print(f"{'=' * 90}")
    print(f"\n  {'Year':<6} {'Baseline':>10} {'Mkt':>10} {'Mkt+Elo':>10} {'Seed':>10} {'ΔMkt':>8} {'ΔMkt+Elo':>10}")
    print(f"  {'-' * 76}")

    per_year: list[dict] = []

    for year in EVAL_YEARS:
        if year not in all_records:
            continue

        train_recs = {y: recs for y, recs in all_records.items() if y < year and recs}
        if len(train_recs) < 2:
            continue

        try:
            model = fit_torvik_correction_from_year_records(
                train_recs,
                config=config,
                recent_year_count=RECENT_YEAR_COUNT,
                recent_total_ratio=RECENT_TOTAL_RATIO,
            )
        except Exception as e:
            print(f"  {year}: SKIP (train failed: {e})")
            continue

        test_recs = all_records[year]

        base_errs, mkt_errs, mktelo_errs, seed_errs = [], [], [], []
        for rec in test_recs:
            torvik = float(rec["torvik"])
            outcome = float(rec["outcome"])
            s1 = int(rec.get("seed1", 8))
            s2 = int(rec.get("seed2", 8))
            rnd = rec.get("round", None)
            rnd_num = ROUND_ORDER.get(str(rnd), 0) / 5.0 if rnd else 0.0
            odds = rec.get("odds")
            elo = rec.get("elo")
            mkt_prob = float(odds) if odds is not None and float(odds) > 0 else None
            elo_prob = float(elo) if elo is not None and float(elo) > 0 else None

            # Baseline: no market/elo/round at inference (current backtest behavior)
            pred_base = np.clip(model.predict_one(torvik, s1, s2), CLIP_LO, CLIP_HI)
            # Mkt: add market odds at inference
            pred_mkt = np.clip(model.predict_one(torvik, s1, s2, market_prob=mkt_prob), CLIP_LO, CLIP_HI)
            # Mkt+Elo: add market + elo + round
            pred_mktelo = np.clip(
                model.predict_one(torvik, s1, s2, market_prob=mkt_prob, elo_prob=elo_prob, round_num=rnd_num),
                CLIP_LO,
                CLIP_HI,
            )

            base_errs.append((pred_base - outcome) ** 2)
            mkt_errs.append((pred_mkt - outcome) ** 2)
            mktelo_errs.append((pred_mktelo - outcome) ** 2)
            seed_errs.append((seed_brier(s1, s2) - outcome) ** 2)

        b_base = float(np.mean(base_errs))
        b_mkt = float(np.mean(mkt_errs))
        b_mktelo = float(np.mean(mktelo_errs))
        b_seed = float(np.mean(seed_errs))
        bss_base = 1.0 - b_base / b_seed
        bss_mkt = 1.0 - b_mkt / b_seed
        bss_mktelo = 1.0 - b_mktelo / b_seed

        per_year.append(
            {
                "year": year,
                "brier_baseline": b_base,
                "brier_mkt": b_mkt,
                "brier_mktelo": b_mktelo,
                "brier_seed": b_seed,
                "bss_baseline": bss_base,
                "bss_mkt": bss_mkt,
                "bss_mktelo": bss_mktelo,
            }
        )

        print(
            f"  {year:<6} {b_base:>10.4f} {b_mkt:>10.4f} {b_mktelo:>10.4f} {b_seed:>10.4f}"
            f" {b_mkt - b_base:>+8.4f} {b_mktelo - b_base:>+10.4f}"
        )

    if not per_year:
        print("  No results.")
        return

    print(f"  {'-' * 76}")
    mb = np.mean([r["brier_baseline"] for r in per_year])
    mm = np.mean([r["brier_mkt"] for r in per_year])
    me = np.mean([r["brier_mktelo"] for r in per_year])
    ms = np.mean([r["brier_seed"] for r in per_year])
    print(f"  {'Mean':<6} {mb:>10.4f} {mm:>10.4f} {me:>10.4f} {ms:>10.4f} {mm - mb:>+8.4f} {me - mb:>+10.4f}")
    print(f"\n  BSS (vs seed):")
    print(f"    Baseline (no market at inference): {1.0 - mb / ms:>+.4f}")
    print(f"    With market at inference:          {1.0 - mm / ms:>+.4f}  Δ={mm - mb:+.4f}")
    print(f"    With market+elo+round:             {1.0 - me / ms:>+.4f}  Δ={me - mb:+.4f}")

    n = len(per_year)
    mkt_better = sum(1 for r in per_year if r["brier_mkt"] < r["brier_baseline"])
    mktelo_better = sum(1 for r in per_year if r["brier_mktelo"] < r["brier_baseline"])
    print(f"\n  Mkt better than baseline:     {mkt_better}/{n} years")
    print(f"  Mkt+Elo better than baseline: {mktelo_better}/{n} years")

    out_path = REPO_ROOT / "artifacts" / "market_inference_ablation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "params": {"ridge": RIDGE, "max_correction": MAX_CORRECTION, "recent_year_count": RECENT_YEAR_COUNT},
                "per_year": per_year,
            },
            f,
            indent=2,
        )
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
