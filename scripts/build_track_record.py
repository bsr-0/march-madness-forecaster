#!/usr/bin/env python3
"""How each displayed bracket actually did: realised points and pool finish.

WHAT THIS ANSWERS
For a season that has been played, the site shows each strategy's bracket
graded red and green against what happened. It did not say what that came to:
how many ESPN points the bracket scored, and where it would have finished in
the pool the P(1st) figure is about. This script computes both.

SAME FIELD, SAME SCORER, REAL OUTCOME. P(1st) is the expected share of first
place against 29 ESPN-behaving opponents drawn by draw_selection_trials() and
scored by score_brackets_team_identity() under ESPN rules, averaged over 2,000
simulated tournaments. The track record is the identical construction with one
substitution: the actual tournament outcome in place of each trial's simulated
one. The opponent draws are the same objects (same seed), the scoring function
is the same object, the first-place-share definition is the same object; and,
as in scripts/evaluate_fitted_bracket.py, before scoring anything the replayed
tables are required to reproduce the candidate artifact's shipped P(1st)/EV
exactly. "Would have won X% of pools" therefore means: across the 2,000
opponent fields P(1st) was measured against, this bracket took (a share of)
first place in X% of them, given what really happened. Median finish is the
median competition rank in those fields.

WHY 2026 IS INCLUDED. build_ui_payload.actual_winners() said, in 2026-08, that
no total is derived from the actual results because the model had been trained
on 2026. That was true then and is not now: every input to every displayed
bracket is walk-forward for its season -- the noseed model is trained on
max_year=year and asserted so, the seed referee is built as_of=year, Torvik is
the pre-tournament snapshot, the public picks are the archived pre-tournament
file, the fitted model trains and calibrates on strictly earlier seasons
(2026-09 audit). 2026 is one of the 15 out-of-sample seasons the shipped
backtest claim rests on, and the payload says so.

NOT A LEADERBOARD. One season is one draw. A bracket that finished 25th of 30
in 2026 is not thereby worse than one that finished 3rd; the P(1st) figure is
the expectation, the track record is the realisation, and the page labels them
as such. This artifact exists so the page can show the realisation honestly,
not so anyone can pick a strategy by it.

USAGE (after build_ui_payload.py and evaluate_fitted_bracket.py):
    python scripts/build_track_record.py --year 2026
    python scripts/build_ui_payload.py        # picks it up
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts._common import load_tournament_results  # noqa: E402
from scripts.evaluate_fitted_bracket import (  # noqa: E402
    DEFAULT_SEED,
    KIND as FITTED_KIND,
    parity_check,
    rebuild_referee,
    sha256_file,
)
from scripts.experiments.build_candidate_artifact import DEFAULT_POOL_SIZE, _encode_rows  # noqa: E402
from scripts.mc_pool_backtest import ESPN_SCORING  # noqa: E402
from src.optimization.payout import first_place_shares  # noqa: E402
from src.simulation.pool_competition import ROUND_NAMES, actual_winners_by_round, score_brackets_team_identity  # noqa: E402

CANDIDATES_DIR = REPO / "artifacts" / "candidates"
FITTED_DIR = REPO / "artifacts" / "fitted_eval"
OUT_DIR = REPO / "artifacts" / "track_record"
DOCS = REPO / "docs"

KIND = "track_record"

# Payload strategy id -> candidate-artifact named strategy. The payload copies
# these verbatim (build_ui_payload.py); asserted below rather than assumed.
NAMED = {"p1": "blend_region_35", "ev": "ev_optimal"}


def outcome_hash(actual: Dict[str, set]) -> str:
    canon = {r: sorted(actual[r]) for r in ROUND_NAMES}
    return hashlib.sha256(json.dumps(canon, sort_keys=True).encode()).hexdigest()


def realised(winners: List[List[str]], actual: Dict[str, set], ref) -> Dict[str, Any]:
    """Points and finish against the real outcome, over the P(1st) trials."""
    row = _encode_rows(winners, ref["first_round"])
    pts = float(score_brackets_team_identity(row, actual, ref["first_round"], ESPN_SCORING)[0])
    shares, ranks, pool_median, pool_best = [], [], [], []
    for opp, _sim in ref["p1_trials"]:
        opp_scores = score_brackets_team_identity(opp, actual, ref["first_round"], ESPN_SCORING)
        shares.append(float(first_place_shares(np.array([pts]), opp_scores)[0]))
        ranks.append(1 + int(np.count_nonzero(opp_scores > pts)))
        pool_median.append(float(np.median(opp_scores)))
        pool_best.append(float(opp_scores.max()))
    return {
        "points": int(pts),
        "won_share": round(float(np.mean(shares)), 4),
        "median_rank": int(np.median(ranks)),
        "mean_rank": round(float(np.mean(ranks)), 2),
        "pool_median_points": int(np.median(pool_median)),
        "pool_best_points": int(np.median(pool_best)),
        "n_trials": len(ref["p1_trials"]),
    }


def build(year: int, seed: int) -> Dict[str, Any]:
    art = json.loads((CANDIDATES_DIR / f"candidates_{year}.json").read_text())
    season = json.loads((DOCS / "data" / f"season_{year}.json").read_text())
    team_ids = [t["id"] for t in art["teams"]]
    if [t["id"] for t in season["teams"]] != team_ids:
        raise RuntimeError("season payload teams differ from the candidate artifact's teams")

    # The outcome, from the same results file the backtest scores against --
    # and checked against the payload's `actual`, which is what the board
    # grades with, so the tally and the red/green cannot describe different
    # tournaments.
    actual = actual_winners_by_round(load_tournament_results(year))
    if not all(actual.get(r) for r in ROUND_NAMES) or len(actual["CHAMP"]) != 1:
        raise RuntimeError(f"{year}: tournament outcome incomplete; no track record to build")
    payload_actual = season.get("actual")
    if not payload_actual:
        raise RuntimeError(f"{year}: payload carries no actual results")
    for r, idxs in zip(ROUND_NAMES, payload_actual):
        if {team_ids[i] for i in idxs} != set(actual[r]):
            raise RuntimeError(f"{year}: payload `actual` disagrees with tournament results for {r}")

    ref = rebuild_referee(year, n_sims=art["meta"]["n_sims"], trials=art["meta"]["p1_trials"], seed=seed)
    print("[3/3] parity against the shipped artifact, then the track record ...")
    parity = parity_check(art, team_ids, ref)

    strategies: Dict[str, Any] = {}
    for sid, named in NAMED.items():
        w_idx = art["named_strategies"][named]["w"]
        payload_w = next(s for s in season["strategies"] if s["id"] == sid)["picks"]
        if payload_w != w_idx:
            raise RuntimeError(f"{year}: payload strategy {sid} picks differ from artifact {named}")
        winners = [[team_ids[i] for i in r] for r in w_idx]
        strategies[sid] = {"w": w_idx, "source": f"named_strategies.{named}", **realised(winners, actual, ref)}

    fitted_path = FITTED_DIR / f"fitted_eval_{year}.json"
    if fitted_path.exists():
        fe = json.loads(fitted_path.read_text())
        if fe.get("kind") == FITTED_KIND:
            winners = [[team_ids[i] for i in r] for r in fe["w"]]
            strategies["model"] = {
                "w": fe["w"],
                "source": str(fitted_path.relative_to(REPO)),
                "fitted_eval_generated_at": fe["generated_at"],
                **realised(winners, actual, ref),
            }

    return {
        "schema": 1,
        "kind": KIND,
        "year": year,
        "strategies": strategies,
        "meaning": (
            "points: ESPN points this bracket scored against the real outcome "
            "(10/20/40/80/160/320). won_share: across the same simulated 30-entry "
            "opponent fields P(1st) is measured against, the fraction in which this "
            "bracket took (a share of) first place given the real outcome. "
            "median_rank: median competition rank in those fields (1 = won). One "
            "season is one draw; this is the realisation, P(1st) is the expectation."
        ),
        "scorer": {
            "field": "draw_selection_trials, same seed and count as the candidate artifact's P(1st)",
            "scoring": "score_brackets_team_identity under ESPN scoring; first_place_shares for ties",
            "pool_size": DEFAULT_POOL_SIZE,
            "n_opponents": DEFAULT_POOL_SIZE - 1,
            "p1_trials": art["meta"]["p1_trials"],
            "seed": seed,
            "trials_seed": seed + 7,
        },
        "parity": parity,
        "inputs": {
            "outcome_sha256": outcome_hash(actual),
            "candidates_artifact": f"artifacts/candidates/candidates_{year}.json",
            "candidates_generated_at": art["meta"].get("generated_at"),
            "training_sha256": sha256_file(DOCS / "data" / "training.json"),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    a = ap.parse_args()
    out = build(a.year, a.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"track_record_{a.year}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\n  parity: {len(out['parity'])} shipped brackets reproduced exactly")
    for sid, r in out["strategies"].items():
        print(f"  {sid:6s} {r['points']:5d} pts  won {r['won_share']*100:5.1f}% of pools  median finish {r['median_rank']:2d}/{DEFAULT_POOL_SIZE}")
    print(f"  -> {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
