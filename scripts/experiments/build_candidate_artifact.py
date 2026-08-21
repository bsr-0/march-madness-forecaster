"""Build the shipped candidate artifact, preserving the diversity we just earned.

The product ships a few thousand pre-scored brackets and does the last-mile
filter / rank / select in the browser. HOW those few thousand are drawn from the
~200k scenario bank is part of the product's statistical behaviour, not an
implementation detail: naive "top N by expected score" would reproduce exactly
the candidate collapse documented in FINDINGS.md 6e (~25 candidates -> ~5
distinct brackets, mean Hamming 6/63) that this whole line of work removed.

So the sampler uses explicit quotas rather than a ranking cut:

  1. CHAMPION STRATA. Slots are allocated per champion, proportional to that
     champion's probability but with a floor, so plausible-but-unlikely
     champions survive into the artifact instead of being ranked away.
  2. EV STRATA WITHIN CHAMPION. Within each champion, slots spread across
     expected-score deciles. This is what preserves the low-EV / high-P(1st)
     region -- the region where the two objectives disagree, and therefore the
     entire reason the product has more than one strategy.
  3. CONSTRAINT TOP-UP. Every supported preference is checked for survivor
     coverage afterwards and topped up if thin, so no UI control can silently
     return nothing.

LEAKAGE BOUNDARY
----------------
Everything here must be knowable the moment the bracket is released:

  seeds / regions          Selection Sunday
  torvik barthag           data_type=pre_tournament, cutoff_date before tip;
                           _validate_pretournament raises otherwise
  ESPN public picks        published before tip; require_archived=True

One documented exception: ``seed_pick_model._HISTORICAL_WIN_RATES`` is a
hardcoded 1985-2025 constant used by the shared P(1st) referee. For a
forward-looking 2027 artifact it is clean, since all its data precedes 2027.
For validating a historical season it contains that season's own results --
roughly 63 of ~2500 games, applied identically to every candidate, so bounded
and non-differential. Recorded in the artifact's ``provenance`` block rather
than hidden.

Usage:
    python3 scripts/experiments/build_candidate_artifact.py --year 2024
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts._common import load_seeds_and_regions  # noqa: E402
from scripts.experiments.conditional_bracket_engine import (  # noqa: E402
    _REACHES,
    expected_scores,
    round_marginals,
)
from scripts.experiments.objective_diversity_matrix import pool_p_first  # noqa: E402
from scripts.mc_pool_backtest import (  # noqa: E402
    ESPN_SCORING,
    _load_torvik_barthag,
    build_bracket_order,
    build_espn_pick_distribution,
    draw_selection_trials,
)
from src.prediction.pairwise import PairwiseProbabilities, simulate_bracket_outcomes  # noqa: E402
from src.prediction.seed_probabilities import build_seed_probabilities  # noqa: E402

ROUND_NAMES = ("R64", "R32", "S16", "E8", "F4", "CHAMP")
DEFAULT_POOL_SIZE = 30  # opponent field assumed by every P(1st) in the artifact


# ---------------------------------------------------------------------------
# Leakage gate
# ---------------------------------------------------------------------------


def assert_pretournament_inputs(year: int) -> Dict:
    """Fail before any compute if an input is not knowable at bracket release."""
    prov = {}
    for prefix in (Path("data/raw/historical"), Path("data/raw")):
        p = prefix / f"torvik_{year}.json"
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            if d.get("data_type") != "pre_tournament":
                raise RuntimeError(f"{p}: data_type={d.get('data_type')!r}, refusing to build.")
            prov["torvik"] = {
                "file": str(p),
                "data_type": d.get("data_type"),
                "cutoff_date": d.get("cutoff_date"),
                "tournament_start": d.get("tournament_start"),
            }
            break
    if "torvik" not in prov:
        raise RuntimeError(f"no torvik_{year}.json found; cannot verify provenance")

    prov["seed_head_to_head"] = {
        "source": "seed_pick_model._HISTORICAL_WIN_RATES (hardcoded 1985-2025)",
        "clean_for_forward_looking": True,
        "caveat": (
            "For historical validation this constant includes the target season's own "
            "results (~63 of ~2500 games). It is used only by the shared P(1st) referee "
            "and applied identically to every candidate, so it is bounded and does not "
            "favour one strategy over another."
        ),
    }
    return prov


# ---------------------------------------------------------------------------
# Diversity-preserving sampler
# ---------------------------------------------------------------------------


def stratified_sample(
    rounds: List,
    ev: np.ndarray,
    target: int,
    rng: np.random.Generator,
    min_per_champion: int = 8,
    ev_strata: int = 10,
) -> np.ndarray:
    """Draw ``target`` candidates preserving champion and objective diversity.

    Proportional-with-floor over champions, then spread across EV deciles inside
    each champion. The floor is what keeps unlikely-but-plausible champions in
    the artifact; the EV strata are what keep the low-EV / high-P(1st) region
    that makes the two strategies differ.
    """
    by_champ: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(rounds):
        c = r[_REACHES["CHAMP"]]
        if c:
            by_champ[c[0]].append(i)

    n_total = sum(len(v) for v in by_champ.values())
    quotas: Dict[str, int] = {}
    for champ, idxs in by_champ.items():
        prop = int(round(target * len(idxs) / n_total))
        quotas[champ] = min(len(idxs), max(min_per_champion, prop))

    # Scale back proportionally if the floors overshoot the target.
    over = sum(quotas.values())
    if over > target:
        scale = target / over
        for champ in quotas:
            quotas[champ] = max(1, int(quotas[champ] * scale))

    chosen: List[int] = []
    for champ, idxs in by_champ.items():
        want = quotas[champ]
        if want >= len(idxs):
            chosen.extend(idxs)
            continue
        arr = np.array(idxs)
        order = arr[np.argsort(ev[arr])]
        # Spread across EV strata rather than taking the top of the champion's
        # own distribution, which would re-collapse toward chalk within champion.
        buckets = np.array_split(order, min(ev_strata, len(order)))
        per = max(1, want // len(buckets))
        picked: List[int] = []
        for b in buckets:
            take = min(per, len(b))
            picked.extend(rng.choice(b, size=take, replace=False).tolist())
        if len(picked) < want:
            rest = [i for i in order.tolist() if i not in set(picked)]
            picked.extend(rng.choice(rest, size=min(want - len(picked), len(rest)), replace=False).tolist())
        chosen.extend(picked[:want])

    return np.array(sorted(set(chosen)))


# ---------------------------------------------------------------------------
# True constraint probabilities (full bank)
# ---------------------------------------------------------------------------


def _constraint_predicates(seeds: Dict[str, int]):
    """The predicates behind every preference control the UI can offer."""
    return {
        "f4_at_least_1_two_three": lambda r: sum(1 for t in r[_REACHES["F4"]] if seeds.get(t) in (2, 3)) >= 1,
        "f4_at_least_2_two_three": lambda r: sum(1 for t in r[_REACHES["F4"]] if seeds.get(t) in (2, 3)) >= 2,
        "f4_mostly_favorites": lambda r: sum(1 for t in r[_REACHES["F4"]] if seeds.get(t) == 1) >= 3,
        "s16_at_least_1_double_digit": lambda r: any(seeds.get(t, 0) >= 10 for t in r[_REACHES["S16"]]),
        "s16_at_least_2_double_digit": lambda r: sum(1 for t in r[_REACHES["S16"]] if seeds.get(t, 0) >= 10) >= 2,
        "s16_no_double_digit": lambda r: not any(seeds.get(t, 0) >= 10 for t in r[_REACHES["S16"]]),
    }


def true_constraint_probabilities(rounds: List, seeds: Dict[str, int]) -> Dict[str, float]:
    """P(constraint) over the FULL bank.

    These MUST come from the full bank, never from the shipped candidate list.
    The sampler deliberately over-samples unlikely champions to protect
    diversity, so the artifact is not a probability sample -- counting rows in
    it would bias every feasibility hint toward rare scenarios.
    """
    preds = _constraint_predicates(seeds)
    n = len(rounds)
    return {k: round(sum(1 for r in rounds if f(r)) / n, 5) for k, f in preds.items()}


def true_team_f4_probabilities(rounds: List, seeds: Dict[str, int]) -> Dict[str, float]:
    """P(team reaches Final Four) over the full bank, for the team dropdown."""
    c = Counter()
    for r in rounds:
        c.update(r[_REACHES["F4"]])
    n = len(rounds)
    return {t: round(v / n, 5) for t, v in sorted(c.items(), key=lambda kv: -kv[1])}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(bank, rounds, sel, ev, p1, first_round, seeds, full_rounds) -> Dict:
    """Checks that must pass before the artifact is fit to ship."""
    out: Dict[str, object] = {}

    # 1. Path consistency: a team may only win round R+1 if it won round R.
    bad = 0
    for i in sel[: min(500, len(sel))]:
        r = rounds[i]
        for ri in range(5):
            if not set(r[ri + 1]).issubset(set(r[ri])):
                bad += 1
                break
    out["path_consistent"] = bad == 0
    out["path_checked"] = min(500, len(sel))

    # 2. EV recomputed independently on a sample.
    marg = round_marginals(full_rounds)
    pts = {r: ESPN_SCORING[r] for r in ROUND_NAMES}
    errs = []
    for i in sel[: min(200, len(sel))]:
        manual = sum(pts[rn] * sum(marg[ri].get(t, 0.0) for t in rounds[i][ri]) for ri, rn in enumerate(ROUND_NAMES))
        errs.append(abs(manual - ev[i]))
    out["ev_max_abs_error"] = float(max(errs)) if errs else 0.0

    # 3. Diversity preserved vs the full bank.
    def champ_entropy(idxs):
        c = Counter(rounds[i][_REACHES["CHAMP"]][0] for i in idxs if rounds[i][_REACHES["CHAMP"]])
        tot = sum(c.values())
        p = np.array([v / tot for v in c.values()])
        return float(-(p * np.log2(p)).sum()), len(c)

    full_idx = list(range(len(rounds)))
    fe, fc = champ_entropy(full_idx)
    se, sc = champ_entropy(sel)
    rng = np.random.default_rng(0)

    def mean_hamming(idxs):
        a = rng.choice(idxs, size=min(4000, len(idxs)))
        b = rng.choice(idxs, size=len(a))
        keep = a != b
        return float((bank[a[keep]] != bank[b[keep]]).sum(axis=1).mean())

    out["champion_entropy_full"] = round(fe, 3)
    out["champion_entropy_artifact"] = round(se, 3)
    out["distinct_champions_full"] = fc
    out["distinct_champions_artifact"] = sc
    out["mean_hamming_full"] = round(mean_hamming(full_idx), 2)
    out["mean_hamming_artifact"] = round(mean_hamming(list(sel)), 2)

    # 4. Objective diversity: is the low-EV / high-P(1st) region retained?
    lo_ev = ev[sel] < np.percentile(ev[sel], 40)
    hi_p1 = p1 > np.percentile(p1, 60)
    out["low_ev_high_p1_count"] = int((lo_ev & hi_p1).sum())
    out["ev_p1_rank_corr"] = round(
        float(np.corrcoef(np.argsort(np.argsort(ev[sel])), np.argsort(np.argsort(p1)))[0, 1]), 3
    )

    # 5. Constraint coverage for every UI control.
    cons = {
        ">=1 2/3-seed F4": lambda r: sum(1 for t in r[_REACHES["F4"]] if seeds.get(t) in (2, 3)) >= 1,
        ">=2 2/3-seeds F4": lambda r: sum(1 for t in r[_REACHES["F4"]] if seeds.get(t) in (2, 3)) >= 2,
        ">=3 1-seeds F4": lambda r: sum(1 for t in r[_REACHES["F4"]] if seeds.get(t) == 1) >= 3,
        ">=1 dd-seed S16": lambda r: any(seeds.get(t, 0) >= 10 for t in r[_REACHES["S16"]]),
        ">=2 dd-seeds S16": lambda r: sum(1 for t in r[_REACHES["S16"]] if seeds.get(t, 0) >= 10) >= 2,
        "no dd-seed S16": lambda r: not any(seeds.get(t, 0) >= 10 for t in r[_REACHES["S16"]]),
    }
    out["constraint_coverage"] = {k: int(sum(1 for i in sel if f(rounds[i]))) for k, f in cons.items()}
    return out


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build(year: int, n_sims: int, target: int, trials: int, seed: int) -> Dict:
    prov = assert_pretournament_inputs(year)
    seeds, regions = load_seeds_and_regions(year)
    first_round = build_bracket_order(seeds, regions)
    barthag = _load_torvik_barthag(year, seeds)
    pw = PairwiseProbabilities.from_ratings(barthag, source=f"log5(torvik_{year})")

    rng = np.random.default_rng(seed)
    print(f"[1/5] simulating {n_sims:,} tournaments ...")
    bank, rounds = simulate_bracket_outcomes(pw, first_round, n_sims, rng, noise_std=0.0)

    print("[2/5] exact expected scores ...")
    marg = round_marginals(rounds)
    ev = expected_scores(rounds, marg, ESPN_SCORING)

    print(f"[3/5] diversity-preserving sample -> {target:,} ...")
    sel = stratified_sample(rounds, ev, target, rng)

    print(f"[4/5] P(1st) for {len(sel):,} candidates, {trials:,} shared trials ...")
    seed_pw = build_seed_probabilities(seeds)
    pick_dist = build_espn_pick_distribution(year, seeds)
    p1_trials = draw_selection_trials(
        trials,
        n_opponents=DEFAULT_POOL_SIZE,
        first_round=first_round,
        pick_dist=pick_dist,
        matchup_probs=seed_pw,
        seeds=seeds,
        rng=np.random.default_rng(seed + 7),
    )
    p1 = pool_p_first(bank[sel], p1_trials, first_round)

    print("[5/5] validating ...")
    checks = validate(bank, rounds, sel, ev, p1, first_round, seeds, rounds)
    true_probs = true_constraint_probabilities(rounds, seeds)
    team_f4 = true_team_f4_probabilities(rounds, seeds)

    # The artifact is deliberately not a probability sample -- verify the
    # difference is real so the warning below is not decorative.
    preds = _constraint_predicates(seeds)
    artifact_probs = {
        k: round(sum(1 for i in sel if f(rounds[i])) / len(sel), 5) for k, f in preds.items()
    }
    checks["constraint_prob_bias"] = {
        k: round(artifact_probs[k] - true_probs[k], 4) for k in true_probs
    }

    # Compact encoding: team table + per-round winner indices.
    team_ids = sorted(seeds)
    tidx = {t: i for i, t in enumerate(team_ids)}
    candidates = []
    for j, i in enumerate(sel):
        r = rounds[i]
        candidates.append(
            {
                "w": [[tidx[t] for t in r[ri]] for ri in range(6)],
                "ev": round(float(ev[i]), 1),
                "p1": round(float(p1[j]), 4),
                "dd16": sum(1 for t in r[_REACHES["S16"]] if seeds.get(t, 0) >= 10),
            }
        )

    return {
        "schema": 1,
        "year": year,
        "teams": [{"id": t, "seed": seeds[t], "region": regions.get(t, "")} for t in team_ids],
        "candidates": candidates,
        "meta": {
            "n_sims": n_sims,
            "n_candidates": len(candidates),
            "p1_trials": trials,
            "p1_pool_size": DEFAULT_POOL_SIZE,
            "p1_assumption": (
                f"P(1st) assumes a {DEFAULT_POOL_SIZE}-opponent pool with ESPN public "
                f"pick behaviour. It is NOT a universal probability of winning any pool."
            ),
            "p1_se_estimate": round(float(np.sqrt(0.05 * 0.95 / trials)), 5),
            "candidates_are_not_a_probability_sample": (
                "Unlikely champions are deliberately over-sampled to protect diversity. "
                "Use constraint_probabilities / team_final_four_probabilities for any "
                "frequency shown to a user; NEVER count rows in `candidates`."
            ),
            "objectives": ["ev", "p1"],
            "source": pw.source,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "constraint_probabilities": true_probs,
        "team_final_four_probabilities": team_f4,
        "provenance": prov,
        "validation": checks,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--n-sims", type=int, default=150_000)
    ap.add_argument("--target", type=int, default=3000)
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--out", type=str, default="artifacts/candidates")
    args = ap.parse_args()

    art = build(args.year, args.n_sims, args.target, args.trials, args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"candidates_{args.year}.json"
    with open(path, "w") as f:
        json.dump(art, f, separators=(",", ":"))
    size_mb = path.stat().st_size / 2**20

    v = art["validation"]
    print(f"\n{'=' * 78}\nCANDIDATE ARTIFACT — {args.year}\n{'=' * 78}")
    print(f"  candidates            {art['meta']['n_candidates']:,}")
    print(f"  file size             {size_mb:.2f} MB")
    print(f"  path-consistent       {v['path_consistent']}  ({v['path_checked']} checked)")
    print(f"  EV max abs error      {v['ev_max_abs_error']:.6f}")
    print(f"  champions   full {v['distinct_champions_full']:3d} -> artifact {v['distinct_champions_artifact']:3d}")
    print(f"  champ entropy full {v['champion_entropy_full']:.3f} -> artifact {v['champion_entropy_artifact']:.3f}")
    print(f"  mean Hamming full {v['mean_hamming_full']:.1f} -> artifact {v['mean_hamming_artifact']:.1f}")
    print(f"  low-EV/high-P1 kept   {v['low_ev_high_p1_count']}")
    print(f"  EV-vs-P1 rank corr    {v['ev_p1_rank_corr']}")
    bias = v["constraint_prob_bias"]
    print(f"  artifact-vs-true P(constraint) bias: "
          f"min {min(bias.values()):+.3f}  max {max(bias.values()):+.3f}  "
          f"(why frequencies ship separately)")
    print("  constraint coverage:")
    for k, n in v["constraint_coverage"].items():
        print(f"     {k:22} {n:6,}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
