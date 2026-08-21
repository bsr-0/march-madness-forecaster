"""Generate docs/data/ml_backtest.json — the ML backtest tab's payload.

Everything here is computed from artifacts/loyo_pergame_predictions.json,
which holds one row per tournament game with the actual outcome alongside
each candidate model's pre-tournament probability *and* the seed baseline's.
That co-location is what makes honest skill reporting possible: model and
baseline are scored on exactly the same games, so BSS is a like-for-like
comparison rather than two numbers from different runs.

Deliberately reports the unflattering metrics next to the flattering ones:
raw accuracy (where the model barely separates from the seed baseline), the
losing model families, the closing-market comparison on the subset where
odds exist, and the methodology caveats FINDINGS.md raises about this
project's own evaluation discipline. Nothing here is hand-entered.

**Orientation.** The artifact is now written already-oriented (better seed
first) because ``load_tournament_games`` applies
``src.data.game_orientation`` at the load boundary. Before that fix it was
largely winner-first — 100% of 2005-2015 games had ``outcome == 1``, 90.2%
overall — which scored every model against a ~90% base rate instead of the
true ~72%. The orientation call below is kept anyway: it is idempotent, and
it keeps this generator correct against an older or hand-edited artifact.
Do not remove it.

Usage:
    python3 scripts/generate_ml_backtest_data.py
"""

import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.game_orientation import (  # noqa: E402
    favorite_won,
    orient_prediction_record,
)

SRC = REPO / "artifacts" / "loyo_pergame_predictions.json"
OUT = REPO / "docs" / "data" / "ml_backtest.json"

# The season used as an in-sample integration fixture. Spec 2027.v2 trains
# through it, so it may never appear in a headline accuracy figure.
REPLAY_YEAR = 2026

# Model key -> display label. "seed" is the baseline everything is scored
# against; "torvik" is the production probability source the bracket UI uses.
MODELS = [
    ("torvik", "Torvik ratings (production)"),
    ("seed", "Seed baseline"),
    ("massey_avg", "Massey composite"),
    ("knn", "k-nearest neighbours"),
    ("elo", "Elo"),
]

ROUND_ORDER = ["R64", "R32", "S16", "E8", "F4", "NCG"]
ROUND_LABELS = {
    "R64": "Round of 64",
    "R32": "Round of 32",
    "S16": "Sweet 16",
    "E8": "Elite 8",
    "F4": "Final Four",
    "NCG": "Championship",
}

BOOTSTRAP_N = 5000
SEED = 20260818


def brier(pairs):
    return sum((p - o) ** 2 for o, p in pairs) / len(pairs)


def accuracy(pairs):
    return sum(1 for o, p in pairs if (p >= 0.5) == (o == 1)) / len(pairs)


def log_loss(pairs):
    return -sum(o * math.log(max(p, 1e-15)) + (1 - o) * math.log(max(1 - p, 1e-15)) for o, p in pairs) / len(pairs)


def bootstrap_ci(pairs, stat_fn, n=BOOTSTRAP_N, level=0.95):
    """Percentile bootstrap CI over games."""
    rng = random.Random(SEED)
    k = len(pairs)
    if k < 2:
        return None, None
    stats = []
    for _ in range(n):
        sample = [pairs[rng.randrange(k)] for _ in range(k)]
        stats.append(stat_fn(sample))
    stats.sort()
    lo = stats[int((1 - level) / 2 * n)]
    hi = stats[int((1 + level) / 2 * n) - 1]
    return lo, hi


def pairs_for(rows, key):
    return [(g["outcome"], g[key]) for g in rows if g.get(key) is not None]


def summarize(rows, key, baseline_brier=None, with_ci=False):
    pairs = pairs_for(rows, key)
    if not pairs:
        return None
    b = brier(pairs)
    out = {
        "n_games": len(pairs),
        "brier": round(b, 5),
        "accuracy": round(accuracy(pairs), 5),
        "log_loss": round(log_loss(pairs), 5),
    }
    if baseline_brier:
        out["bss"] = round(1 - b / baseline_brier, 5)
    if with_ci:
        lo, hi = bootstrap_ci(pairs, brier)
        out["brier_ci"] = [round(lo, 5), round(hi, 5)]
        lo, hi = bootstrap_ci(pairs, accuracy)
        out["accuracy_ci"] = [round(lo, 5), round(hi, 5)]
    return out


def calibration_bins(rows, key, n_bins=10):
    bins = []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        sel = [
            (g["outcome"], g[key])
            for g in rows
            if g.get(key) is not None and (lo <= g[key] < hi or (i == n_bins - 1 and g[key] == 1.0))
        ]
        bins.append(
            {
                "lower": round(lo, 2),
                "upper": round(hi, 2),
                "count": len(sel),
                "mean_predicted": round(sum(p for _, p in sel) / len(sel), 5) if sel else None,
                "mean_actual": round(sum(o for o, _ in sel) / len(sel), 5) if sel else None,
            }
        )
    # Expected calibration error, weighted by bin population.
    total = sum(b["count"] for b in bins)
    ece = (
        sum(b["count"] * abs(b["mean_predicted"] - b["mean_actual"]) for b in bins if b["count"]) / total
        if total
        else None
    )
    return bins, (round(ece, 5) if ece is not None else None)


def main():
    if not SRC.exists():
        print(f"ERROR: {SRC} not found", file=sys.stderr)
        return 1

    with open(SRC) as f:
        raw = json.load(f)

    rows = []
    stored_outcome_1 = 0
    for year, games in raw.items():
        for g in games:
            stored_outcome_1 += 1 if g["outcome"] == 1 else 0
            # Idempotent: the artifact is written oriented as of the
            # game_orientation fix, but older copies may not be.
            r = orient_prediction_record(g)
            r["year"] = int(year)
            r["favorite_won"] = favorite_won(r)
            rows.append(r)

    years = sorted({g["year"] for g in rows})

    # ── The replay year is excluded from every headline figure ───────
    #
    # 2026 is an in-sample integration fixture, not a prospective evaluation
    # season: spec 2027.v2 trains through it. Leaving it in the headline would
    # make a performance claim out of a season the model has already seen -- and
    # a flattering one, because 2026 happens to be the model's best year
    # (accuracy .746 vs .721 across the honest window; Brier .145 vs .181).
    #
    # It is still reported, separately and labelled, in `replay_year`. The
    # separation is computed HERE rather than in the browser so the site cannot
    # reconstruct a contaminated headline by re-aggregating per-year rows.
    scored_rows = [g for g in rows if g["year"] != REPLAY_YEAR]
    scored_years = sorted({g["year"] for g in scored_rows})
    seed_brier_all = brier(pairs_for(scored_rows, "seed"))

    # ── Headline + model table (excluding the replay year) ───────────
    models = []
    for key, label in MODELS:
        s = summarize(scored_rows, key, baseline_brier=seed_brier_all, with_ci=True)
        if s:
            models.append({"key": key, "label": label, **s})

    # ── Per-year (production model vs seed baseline) ─────────────────
    per_year = []
    for y in years:
        yr_rows = [g for g in rows if g["year"] == y]
        sb = brier(pairs_for(yr_rows, "seed"))
        m = summarize(yr_rows, "torvik", baseline_brier=sb)
        s = summarize(yr_rows, "seed")
        per_year.append(
            {
                "year": y,
                "n_games": m["n_games"],
                "brier_model": m["brier"],
                "brier_seed": s["brier"],
                "bss": m["bss"],
                "accuracy_model": m["accuracy"],
                "accuracy_seed": s["accuracy"],
            }
        )

    # ── The replay year, reported apart from the headline ────────────
    replay_rows = [g for g in rows if g["year"] == REPLAY_YEAR]
    replay_year = None
    if replay_rows:
        rb = brier(pairs_for(replay_rows, "seed"))
        rm = summarize(replay_rows, "torvik", baseline_brier=rb)
        rs = summarize(replay_rows, "seed")
        replay_year = {
            "year": REPLAY_YEAR,
            "n_games": rm["n_games"],
            "brier_model": rm["brier"],
            "brier_seed": rs["brier"],
            "accuracy_model": rm["accuracy"],
            "accuracy_seed": rs["accuracy"],
            "is_out_of_sample": False,
            "label": f"{REPLAY_YEAR} replay (in-sample)",
            "disclaimer": (
                f"{REPLAY_YEAR} is an integration fixture. The model was trained on it, "
                "so these numbers are not evidence of predictive accuracy and are "
                "excluded from every headline figure on this page."
            ),
        }

    # ── Per-round ────────────────────────────────────────────────────
    per_round = []
    for rnd in ROUND_ORDER:
        rr = [g for g in scored_rows if g["round"] == rnd]
        if not rr:
            continue
        sb = brier(pairs_for(rr, "seed"))
        m = summarize(rr, "torvik", baseline_brier=sb)
        s = summarize(rr, "seed")
        upsets = sum(
            1
            for g in rr
            if (g["seed1"] > g["seed2"] and g["outcome"] == 1) or (g["seed2"] > g["seed1"] and g["outcome"] == 0)
        )
        per_round.append(
            {
                "round": rnd,
                "label": ROUND_LABELS[rnd],
                "n_games": m["n_games"],
                "brier_model": m["brier"],
                "brier_seed": s["brier"],
                "bss": m["bss"],
                "accuracy_model": m["accuracy"],
                "accuracy_seed": s["accuracy"],
                "upset_rate": round(upsets / len(rr), 5),
            }
        )

    # ── Calibration ──────────────────────────────────────────────────
    cal_bins, ece = calibration_bins(scored_rows, "torvik")

    # ── Closing-market subset ────────────────────────────────────────
    # Only ~1/4 of games have odds, so this is scored on its own subset
    # rather than compared against the full-sample numbers above.
    mkt_rows = [g for g in scored_rows if g.get("closing_market") is not None]
    market = None
    if mkt_rows:
        msb = brier(pairs_for(mkt_rows, "seed"))
        market = {
            "n_games": len(mkt_rows),
            "years": sorted({g["year"] for g in mkt_rows}),
            "model": summarize(mkt_rows, "torvik", baseline_brier=msb),
            "market": summarize(mkt_rows, "closing_market", baseline_brier=msb),
            "seed": summarize(mkt_rows, "seed"),
        }

    seeded = [g for g in scored_rows if g["favorite_won"] is not None]

    payload = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "source_file": "artifacts/loyo_pergame_predictions.json",
        "n_games": len(scored_rows),
        "years": scored_years,
        "all_years_including_replay": years,
        "replay_year": replay_year,
        "scoring_window_note": (
            f"Every headline figure excludes {REPLAY_YEAR}, which is an in-sample "
            "integration fixture rather than a prospective evaluation season."
        ),
        "baseline_key": "seed",
        "production_key": "torvik",
        # Excludes same-seed matchups (1-vs-1 Final Fours etc.), where there
        # is no favourite to be right or wrong about.
        "favorite_win_rate": round(sum(1 for g in seeded if g["favorite_won"]) / len(seeded), 5),
        "favorite_win_rate_n": len(seeded),
        "source_orientation_note": {
            "stored_outcome_1_rate": round(stored_outcome_1 / len(rows), 5),
            "detail": (
                "Source rows are stored winner-first — every 2005-2015 game has "
                "outcome=1. All metrics here are computed after re-orienting each "
                "game to better-seed-first, an outcome-independent convention."
            ),
        },
        "models": models,
        "per_year": per_year,
        "per_round": per_round,
        "calibration": {"bins": cal_bins, "ece": ece},
        "market_subset": market,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {OUT}")
    print(f"  {len(scored_rows)} scored games, {len(scored_years)} years "
          f"({scored_years[0]}-{scored_years[-1]}); {REPLAY_YEAR} held out of headline")
    for m in models:
        print(f"  {m['label']:32} Brier {m['brier']:.4f}  acc {m['accuracy']:.4f}  BSS {m.get('bss', 0):+.4f}")
    print(f"  ECE (production): {ece}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
