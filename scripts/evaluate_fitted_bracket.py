#!/usr/bin/env python3
"""Score the browser's fitted-model bracket with the production referee.

WHAT THIS IS, AND IS NOT
The site shows three brackets side by side. Two of them -- the win-maximiser
and the expected-points optimum -- carry a P(1st) and an expected-points figure
computed by the candidate-artifact pipeline (scripts/experiments/
build_candidate_artifact.py). The third, the model fitted live in the browser,
showed only an accuracy, because its bracket never passed through that
pipeline. This script passes it through. It is an EVALUATION artifact and
nothing else:

  * It does not add the fitted bracket to the candidate bank.
  * It does not touch named_strategies, selection, or the frozen candidate
    artifact (PROSPECTIVE_2027 checkpoint 2 forbids rewriting it, and this
    script never opens it for writing).
  * It does not change how the fitted model is trained, calibrated, or picks.

WHY THE NUMBERS ARE COMPARABLE, AND HOW THAT IS PROVED RATHER THAN ASSERTED
The candidate pipeline scores a bracket with two primitives -- expected_scores
(EV, under Torvik log5 round marginals) and pool_p_first (P(1st), against 29
ESPN-behaving opponents under the seed-rate referee, 2,000 shared trials) --
fed by a `marg` table and a `p1_trials` object it builds from seeded RNGs and
never persists. This script imports THE SAME FUNCTIONS from THE SAME MODULES
and rebuilds `marg` and `p1_trials` by replaying build()'s construction with
the same seeds, sizes and source order. Then, before it scores anything of its
own, it re-scores the artifact's shipped named strategies and requires their
EV and P(1st) to come back EXACTLY equal to what the artifact shipped. If
they do not, the replay is not the production scorer and nothing is written
-- there is no fallback formula, because a second EV/P(1st) implementation is
precisely what this must not become.

KNOWN CHARACTERISTIC, STATED PLAINLY. The displayed EV and P(1st) are
reproduced using the existing production definitions, which currently use
DIFFERENT probability tables: EV is expected ESPN points under Torvik log5
round marginals; P(1st) is the expected share of first place against
ESPN-crowd opponents under the historical seed-vs-seed referee. That is how
every shipped number is defined, so it is reproduced here unchanged and
recorded in each output's `scorer.note`. It is not an assertion that EV and
P(1st) come from one unified probability model, and whether they should is a
research question outside this script -- not a bug this evaluation is
entitled to "fix".

THE BRACKET IS THE BROWSER'S, NOT A PORT OF IT. scripts/fitted_bracket_js.js
executes docs/fit.js and docs/app.js under Node with the same payload files
the page fetches, so the picks scored here are the picks on screen, tie rule
and all. The inputs that determine those picks (training.json, and the
season payload's teams/first_round/z) are hashed into the output; the payload
builder embeds this evaluation only when those hashes still match, and the
page itself refuses to show the numbers unless the bracket it just solved is
identical to the one scored here.

USAGE (after build_ui_payload.py has written docs/data/season_YYYY.json):
    python scripts/evaluate_fitted_bracket.py --year 2026
    python scripts/build_ui_payload.py        # picks up the evaluation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# THE SAME NAMES THE ARTIFACT BUILDER IMPORTS, FROM THE SAME MODULES. Nothing
# below is a scoring formula; tests/test_fitted_eval.py asserts these are the
# identical function objects build_candidate_artifact.py binds.
from scripts._common import load_seeds_and_regions, load_seeds_block, load_tournament_results  # noqa: E402
from scripts.experiments.build_candidate_artifact import (  # noqa: E402
    DEFAULT_POOL_SIZE,
    _encode_rows,
    _rating_sources,
    assert_pretournament_inputs,
    resolve_field,
)
from scripts.experiments.conditional_bracket_engine import expected_scores, round_marginals  # noqa: E402
from scripts.experiments.objective_diversity_matrix import pool_p_first  # noqa: E402
from scripts.mc_pool_backtest import (  # noqa: E402
    ESPN_SCORING,
    build_bracket_order,
    build_espn_pick_distribution,
    draw_selection_trials,
)
from src.prediction.pairwise import PairwiseProbabilities, simulate_bracket_outcomes  # noqa: E402
from src.prediction.seed_probabilities import build_seed_probabilities  # noqa: E402
from src.simulation import bracket_topology as _bt  # noqa: E402

CANDIDATES_DIR = REPO / "artifacts" / "candidates"
OUT_DIR = REPO / "artifacts" / "fitted_eval"
DOCS = REPO / "docs"

# The seed build_candidate_artifact.py's CLI defaults to. It is not recorded in
# the artifact; the parity check below is what pins it -- a wrong seed cannot
# reproduce the shipped numbers, and a replay that cannot reproduce them does
# not get to score anything.
DEFAULT_SEED = 20260820

KIND = "fitted_model_evaluated"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fit_inputs_hash(season: Dict[str, Any]) -> str:
    """Hash of exactly the season-payload fields the browser's fit consumes.

    Shared with build_ui_payload.py, which recomputes it over the payload it is
    about to write and embeds this evaluation only on a match. teams carry id
    and seed (seed is the tie-break in solveByFit), first_round is the tree, z
    is the feature matrix. Nothing else in the payload can change the picks.
    """
    sub = {
        "teams": [{"id": t["id"], "seed": t["seed"]} for t in season["teams"]],
        "first_round": season["first_round"],
        "z": season["z"],
    }
    return hashlib.sha256(json.dumps(sub, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fitted_bracket_from_browser(year: int) -> Dict[str, Any]:
    """Run docs/fit.js + docs/app.js under Node; see scripts/fitted_bracket_js.js."""
    proc = subprocess.run(
        ["node", str(REPO / "scripts" / "fitted_bracket_js.js"), str(year), str(DOCS)],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"fitted_bracket_js.js failed ({proc.returncode}): {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def rebuild_referee(year: int, n_sims: int, trials: int, seed: int):
    """Replay build_candidate_artifact.build() up to its scoring tables.

    Line for line the same calls, in the same order, with the same RNG
    construction: one default_rng(seed) for the simulations, of which Torvik is
    the FIRST source drawn (so drawing only Torvik from a fresh rng yields the
    same tournaments), and default_rng(seed + 7) for the pool trials.
    """
    prov = assert_pretournament_inputs(year)
    seeds, regions = load_seeds_and_regions(year)
    prov["field"] = resolve_field(year, seeds, regions)
    region_order = _bt.resolve_region_order(
        year, games=load_tournament_results(year), regions=regions, seeds_block=load_seeds_block(year)
    )
    first_round = build_bracket_order(seeds, regions, region_order=region_order)
    sources = _rating_sources(year, seeds)
    if sources[0][0] != "torvik":
        raise RuntimeError("torvik is not the first rating source; the replay would not match build()")
    barthag = sources[0][1]
    per = max(1, n_sims // len(sources))

    rng = np.random.default_rng(seed)
    spw = PairwiseProbabilities.from_ratings(barthag, source=f"log5(torvik_{year})")
    print(f"[1/3] replaying {per:,} Torvik tournaments (seed {seed}, {len(sources)} sources) ...")
    _bank, torvik_rounds = simulate_bracket_outcomes(spw, first_round, per, rng, noise_std=0.0)
    marg = round_marginals(torvik_rounds)

    print(f"[2/3] replaying {trials:,} pool trials (seed {seed + 7}, {DEFAULT_POOL_SIZE - 1} opponents) ...")
    seed_pw = build_seed_probabilities(seeds, as_of=year)
    pick_dist = build_espn_pick_distribution(year, seeds)
    p1_trials = draw_selection_trials(
        trials,
        n_opponents=DEFAULT_POOL_SIZE - 1,
        first_round=first_round,
        pick_dist=pick_dist,
        matchup_probs=seed_pw,
        seeds=seeds,
        rng=np.random.default_rng(seed + 7),
    )
    return {
        "seeds": seeds, "regions": regions, "first_round": first_round, "marg": marg,
        "p1_trials": p1_trials, "per": per, "n_sources": len(sources), "prov": prov,
    }


def score(winners: List[List[str]], ref) -> Dict[str, float]:
    """The two lines _champion_equity_strategy() uses, with the same rounding."""
    row = _encode_rows(winners, ref["first_round"])
    return {
        "ev": round(float(expected_scores([winners], ref["marg"], ESPN_SCORING)[0]), 1),
        "p1": round(float(pool_p_first(row, ref["p1_trials"], ref["first_round"])[0]), 4),
    }


def parity_check(art: Dict[str, Any], team_ids: List[str], ref) -> Dict[str, Any]:
    """Re-score what the artifact shipped; demand exact equality.

    Every named strategy, plus the first few candidate rows: the named ones
    are the two cards this evaluation will sit beside, the candidates prove
    the same tables reproduce sampled brackets too, not only constructed ones.
    """
    checked = {}
    for name, v in art["named_strategies"].items():
        w = [[team_ids[i] for i in r] for r in v["w"]]
        got = score(w, ref)
        checked[f"named:{name}"] = {"shipped": {"ev": v["ev"], "p1": v["p1"]}, "replayed": got}
    for j, c in enumerate(art["candidates"][:5]):
        w = [[team_ids[i] for i in r] for r in c["w"]]
        got = score(w, ref)
        checked[f"candidate:{j}"] = {"shipped": {"ev": c["ev"], "p1": c["p1"]}, "replayed": got}
    mism = {k: v for k, v in checked.items() if v["shipped"] != v["replayed"]}
    if mism:
        raise RuntimeError(
            "REPLAY DOES NOT REPRODUCE THE SHIPPED NUMBERS -- refusing to score the "
            "fitted bracket with a scorer that is not the production one:\n"
            + json.dumps(mism, indent=2)
        )
    return checked


def evaluate(year: int, seed: int) -> Dict[str, Any]:
    art_path = CANDIDATES_DIR / f"candidates_{year}.json"
    art = json.loads(art_path.read_text())
    season = json.loads((DOCS / "data" / f"season_{year}.json").read_text())
    training_path = DOCS / "data" / "training.json"

    # The bracket the visitor sees, from the code the visitor runs.
    js = fitted_bracket_from_browser(year)

    # Identity: the payload's team order IS the artifact's team order (both are
    # sorted(seeds)), and the trees agree. Checked, not assumed.
    team_ids = [t["id"] for t in art["teams"]]
    if [t["id"] for t in season["teams"]] != team_ids:
        raise RuntimeError("season payload teams differ from the candidate artifact's teams")
    if season["first_round"] != art["first_round"]:
        raise RuntimeError("season payload first_round differs from the candidate artifact's")

    ref = rebuild_referee(year, n_sims=art["meta"]["n_sims"], trials=art["meta"]["p1_trials"], seed=seed)
    if [ref["first_round"].index(t) for t in [team_ids[i] for i in art["first_round"]]] != list(range(64)):
        raise RuntimeError("replayed first_round differs from the artifact's")

    print("[3/3] parity against the shipped artifact, then the fitted bracket ...")
    parity = parity_check(art, team_ids, ref)

    winners = [[team_ids[i] for i in r] for r in js["w"]]
    result = score(winners, ref)

    return {
        "schema": 1,
        "kind": KIND,
        "year": year,
        "w": js["w"],
        "ev": result["ev"],
        "p1": result["p1"],
        "champion": season["teams"][js["w"][5][0]]["name"],
        "not_a_candidate": (
            "This bracket was SCORED by the production referee, not SELECTED by "
            "it. It is not in the candidate bank, was never eligible for "
            "selection, and no selection behaviour reads this file."
        ),
        "p1_meaning": (
            "P(1st) here is the P(1st) of this bracket when evaluated in the common "
            "pool/referee framework the other displayed brackets are scored in -- "
            "not the fitted model's own belief about its chances."
        ),
        "scorer": {
            "ev": "scripts.experiments.conditional_bracket_engine.expected_scores under Torvik log5 round marginals",
            "p1": "scripts.experiments.objective_diversity_matrix.pool_p_first under seed-rate referee, ESPN-crowd opponents",
            "note": (
                "The displayed EV and P(1st) are reproduced using the existing production "
                "definitions, which currently use different probability tables: EV is "
                "expected ESPN points under Torvik log5 round marginals; P(1st) is the "
                "expected share of first place against ESPN-crowd opponents under the "
                "seed-rate referee. This is a known methodological characteristic of the "
                "framework, reproduced unchanged so these numbers sit on the same scale as "
                "the cards beside them. It is NOT an assertion that EV and P(1st) are "
                "derived from one unified probability model."
            ),
            "scoring_rules": "ESPN " + "/".join(str(v) for v in ESPN_SCORING),
            "pool_size": DEFAULT_POOL_SIZE,
            "n_opponents": DEFAULT_POOL_SIZE - 1,
            "n_sims_total": art["meta"]["n_sims"],
            "torvik_sims_replayed": ref["per"],
            "n_rating_sources": ref["n_sources"],
            "p1_trials": art["meta"]["p1_trials"],
            "seed": seed,
            "trials_seed": seed + 7,
        },
        "parity": parity,
        "fit": {
            "keys": js["keys"], "dropped": js["dropped"], "beta": js["beta"], "sigma": js["sigma"],
            "n_training_games": js["n"], "calibration": js["calibration"], "oos": js["oos"],
        },
        "inputs": {
            "fit_inputs_hash": fit_inputs_hash(season),
            "training_sha256": sha256_file(training_path),
            "fit_js_sha256": sha256_file(DOCS / "fit.js"),
            "app_js_sha256": sha256_file(DOCS / "app.js"),
            "candidates_artifact": str(art_path.relative_to(REPO)),
            "candidates_generated_at": art["meta"].get("generated_at"),
        },
        "p1_assumption": art["meta"]["p1_assumption"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="must match the candidate artifact build")
    a = ap.parse_args()

    out = evaluate(a.year, a.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"fitted_eval_{a.year}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\n  parity: {len(out['parity'])} shipped brackets reproduced exactly")
    print(f"  fitted bracket ({out['champion']}): ev {out['ev']}  p1 {out['p1']}")
    print(f"  -> {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
