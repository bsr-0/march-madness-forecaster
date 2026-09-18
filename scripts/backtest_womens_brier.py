#!/usr/bin/env python3
"""Walk-forward Brier for the women's Kaggle model against a seed-only baseline.

    python scripts/backtest_womens_brier.py
    python scripts/backtest_womens_brier.py --min-year 2016 --out artifacts/womens_kaggle_walk_forward.json

Season Y is scored with beta, sigma and the link fit on seasons < Y (the
leakage rule in src/prediction/womens_kaggle_model.py). The baseline is a
one-parameter logistic in seed difference, also fit on seasons < Y. The
number to watch is pooled bss_vs_seed: if it is not positive, the women's
half of the CSV should be seeds, not this model.

Restored in name from the 2026-05 Kaggle track's LOYO script; the model,
the split (walk-forward, not leave-one-year-out) and the baseline are new,
so the numbers are not comparable to artifacts/womens_loyo_brier_baseline.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.prediction.womens_kaggle_model import DEFAULT_KAGGLE_DIR, walk_forward_report  # noqa: E402

DEFAULT_OUT = REPO / "artifacts" / "womens_kaggle_walk_forward.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kaggle-dir", type=Path, default=DEFAULT_KAGGLE_DIR)
    parser.add_argument("--min-year", type=int, default=2014, help="first season scored (earlier ones only train)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    report = walk_forward_report(args.kaggle_dir, min_year=args.min_year)

    print(f"{'year':>6} {'n':>4} {'model':>8} {'seed':>8} {'bss':>8} {'acc':>6} {'a':>6}")
    for r in report["per_year"]:
        print(
            f"{r['year']:>6} {r['n_games']:>4} {r['model_brier']:>8.4f} {r['seed_brier']:>8.4f} "
            f"{r['bss_vs_seed']:>+8.4f} {r['accuracy']:>6.3f} {r['link_a']:>6.3f}"
        )
    p = report["pooled"]
    print(
        f"{'pooled':>6} {p['n_games']:>4} {p['model_brier']:>8.4f} {p['seed_brier']:>8.4f} "
        f"{p['bss_vs_seed']:>+8.4f}   beats seed {p['seasons_model_beats_seed']}/{p['n_seasons']} seasons"
    )
    if report["skipped_games"]:
        print(f"skipped (no box scores for a team): {report['skipped_games']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0 if p["bss_vs_seed"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
