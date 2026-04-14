"""
O21 Marginal-Blend Backtest — COUNCIL_LESSONS.md §2 O21.

Does blending pool-history marginals into the opponent model improve
bracket-ranking calibration against 2023-2026 actual pool outcomes?

Method (per year × per blend weight):
  1. Build opponent model with given pool_history_weight.
  2. MC-estimate P(1st) for each of the K actual pool brackets:
     - draw 30 synthetic opponents from the opponent model
     - score against actual tournament outcomes
     - count fraction of trials where bracket's real score beats max opponent
  3. Compute Spearman ρ between predicted P(1st) rank and actual-score rank
     across the K brackets. Higher ρ = better calibrated opponent model.

Closes / updates: §2 O21 gate "verify switch changes bracket rankings."
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.mc_pool_backtest import (  # noqa: E402
    build_actual_outcome,
    build_first_round_matchups,
    load_seeds_and_regions,
    load_tournament_results,
)
from src.simulation.pool_competition import (  # noqa: E402
    generate_opponent_brackets,
    score_brackets_against_outcome,
)
from src.simulation.ratings_opponent_model import build_opponent_model  # noqa: E402

POOL_HIST = ROOT / "pool_hist_results.json"
# NOTE: 2026 intentionally omitted. `data/raw/historical/tournament_results_2026.json`
# is malformed (49 NCG games, no R64 winners; round_name='R68' used for First Four).
# Re-including requires upstream fix to that file. Tracked as a separate data issue.
YEARS = [2023, 2024, 2025]
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
N_SIMS = 400
# N_OPPONENTS matches the actual pool size per year (K-1 other competitors).
# Using a fixed 30 for all years unfairly inflates the synthetic max in small
# pools (2023 K=18). Per-year n_opp=K-1 preserves the "would this bracket have
# won the same-sized pool if the K-1 others were drawn from the opponent model"
# interpretation.
RNG_SEED = 2026

# ESPN scoring: R64=10, R32=20, S16=40, E8=80, F4=160, CHAMP=320
SCORING_VECTOR = np.array(
    [10] * 32 + [20] * 16 + [40] * 8 + [80] * 4 + [160] * 2 + [320] * 1,
    dtype=np.float64,
)

# Pool JSON abbrevs → dataset team_ids. Reusing the 2026 mapping from the
# existing correlation analysis; extended with any teams only seen in 2023-2025.
# Keeping this local (not imported) because pool_correlation_analysis has
# 2026-only mappings. For O21's purposes we need a best-effort coverage
# across 4 years — missing teams fall through to fuzzy match.
from scripts.pool_correlation_analysis import ABBREV_TO_ID as BASE_ABBREV_MAP  # noqa: E402

EXTRA_ABBREVS = {
    # 2023-2025 teams not in 2026 mapping
    "BAY": "baylor",
    "MARQ": "marquette",
    "NW": "northwestern",
    "XAV": "xavier",
    "CREI": "creighton",
    "MISS": "mississippi",
    "AUB": "auburn",
    "ORE": "oregon",
    "MD": "maryland",
    "KENN": "kennesaw_state",
    "ASU": "arizona_state",
    "MEM": "memphis",
    "UNLV": "unlv",
    "COLO": "colorado",
    "MIN": "minnesota",
    "PITT": "pittsburgh",
    "WVU": "west_virginia",
    "MSST": "mississippi_state",
    "UNM": "new_mexico",
    "SDSU": "san_diego_state",
    "USC": "southern_california",
    "NCST": "nc_state",
    "SDST": "south_dakota_state",
    "FDU": "fairleigh_dickinson",
    "GRAM": "grambling",
    "COLG": "colgate",
    "CHAR": "charleston",
    "FAU": "florida_atlantic",
    "KENT": "kent_state",
    "UAB": "alabama_birmingham",
    "PRIN": "princeton",
    "TAMCC": "texas_a_m_corpus_christi",
    "SEMO": "southeast_missouri_state",
    "NCCU": "north_carolina_central",
    "MTST": "montana_state",
    "ASUN": "asun",
    "DRAKE": "drake",
    "DREX": "drexel",
    "LIB": "liberty",
    "ORAL": "oral_roberts",
    "FURM": "furman",
    "UNCA": "north_carolina_asheville",
    "WAG": "wagner",
    "MONT": "montana",
    "CFU": "central_florida",
}
ABBREV_TO_ID = {**BASE_ABBREV_MAP, **EXTRA_ABBREVS}


def fallback_team_id(abbrev: str) -> str:
    """Best-effort mapping; unknown abbrevs → lowercased abbrev."""
    return ABBREV_TO_ID.get(abbrev, abbrev.lower())


_REGION_ALIASES = {
    "East": "east",
    "West": "west",
    "South": "south",
    "Midwest": "midwest",
    "MidWest": "midwest",
}


def load_seeds_any(year: int) -> tuple[dict, dict]:
    """Load seeds+regions from historical dir, with fallback to data/raw/.

    Handles two schemas: ``{"teams": [...]}`` (historical) and bare-list
    ``[{...}, ...]`` (2026 in data/raw/).
    """
    seeds, regions = load_seeds_and_regions(year)
    if seeds:
        return seeds, regions
    alt = ROOT / "data" / "raw" / f"tournament_seeds_{year}.json"
    if alt.exists():
        with open(alt) as f:
            data = json.load(f)
        teams = data.get("teams", data) if isinstance(data, dict) else data
        s, r = {}, {}
        for t in teams:
            s[t["team_id"]] = t["seed"]
            raw = t.get("region", "")
            r[t["team_id"]] = _REGION_ALIASES.get(raw, raw)
        return s, r
    return {}, {}


def build_actual_round_winners(games: list[dict]) -> dict[str, set[str]]:
    """For each round, return the set of team_ids that WON that round (i.e.
    advanced past it). Keys map to pool-bracket round field names:
    r64, r32, s16, e8, f4, champ.
    """
    # Tournament results use round_name: R64, R32, S16, E8, F4, NCG.
    # 'FF' is First Four play-ins, which are NOT scored in the pool.
    round_map = {"R64": "r64", "R32": "r32", "S16": "s16", "E8": "e8", "F4": "f4", "NCG": "champ"}
    winners: dict[str, set[str]] = {k: set() for k in round_map.values()}
    for g in games:
        rn = g.get("round_name")
        if rn not in round_map:
            continue
        w = g["team1_id"] if g.get("team1_won") else g["team2_id"]
        winners[round_map[rn]].add(w)
    return winners


def score_pool_bracket(bracket: dict, round_winners: dict[str, set[str]]) -> float:
    """Order-independent scoring: count correct picks per round against the
    actual round winners set. Points per round: 10/20/40/80/160/320.
    """
    pts_per_round = {"r64": 10, "r32": 20, "s16": 40, "e8": 80, "f4": 160, "champ": 320}
    total = 0
    for rd, pts in pts_per_round.items():
        picks = bracket.get(rd, [])
        if isinstance(picks, str):
            picks = [picks]
        ids = [fallback_team_id(p) for p in picks]
        total += pts * sum(1 for tid in ids if tid in round_winners.get(rd, set()))
    return float(total)


def load_pool_brackets(year: int) -> list[dict]:
    with open(POOL_HIST) as f:
        data = json.load(f)
    return data["years"].get(str(year), {}).get("brackets", [])


def run_year(year: int, weight: float, rng: np.random.Generator) -> dict:
    seeds, regions = load_seeds_any(year)
    if not seeds:
        return {"year": year, "weight": weight, "error": "no seeds"}

    games = load_tournament_results(year)
    if not games:
        return {"year": year, "weight": weight, "error": "no results"}

    try:
        first_round = build_first_round_matchups(seeds, regions)
    except Exception as e:
        return {"year": year, "weight": weight, "error": f"fr_matchups: {e}"}

    try:
        actual_outcome = build_actual_outcome(first_round, games)
    except Exception as e:
        return {"year": year, "weight": weight, "error": f"actual_outcome: {e}"}

    pool = load_pool_brackets(year)
    if not pool:
        return {"year": year, "weight": weight, "error": "no pool brackets"}

    # Score each pool bracket via round-set-membership (order-independent).
    # The pool JSON's region order differs from build_first_round_matchups,
    # so the bool-array encoding can't be used directly for pool brackets.
    # Opponents still use bool-array since generate_opponent_brackets aligns
    # them to first_round.
    round_winners = build_actual_round_winners(games)
    pool_scores = np.array([score_pool_bracket(b, round_winners) for b in pool], dtype=np.float64)

    # Build opponent model
    opp_model = build_opponent_model(
        year=year,
        seeds=seeds,
        pool_history_path=str(POOL_HIST) if weight > 0 else None,
        pool_history_weight=weight,
        require_espn_picks=False,
    )

    # MC estimate P(1st) for each pool bracket. n_opp = K-1 matches the
    # actual pool size so P(1st) means "would this bracket have won if the
    # K-1 other competitors were drawn from the opponent model."
    K = len(pool)
    n_opp = max(1, K - 1)
    wins = np.zeros(K)
    for _ in range(N_SIMS):
        opp_b = generate_opponent_brackets(n_opp, first_round, {}, opp_model, seeds, rng)
        opp_scores = score_brackets_against_outcome(opp_b, actual_outcome, SCORING_VECTOR)
        max_opp = opp_scores.max()
        wins += (pool_scores > max_opp).astype(float)
        wins += 0.5 * (pool_scores == max_opp).astype(float)
    p1st = wins / N_SIMS

    # Spearman rho between predicted P(1st) rank and actual-score rank
    if len(set(pool_scores.tolist())) < 2 or len(set(p1st.tolist())) < 2:
        rho, pval = float("nan"), float("nan")
    else:
        res = spearmanr(p1st, pool_scores)
        rho, pval = float(res.statistic), float(res.pvalue)

    return {
        "year": year,
        "weight": weight,
        "K": K,
        "mean_p1st": float(p1st.mean()),
        "rho": rho,
        "pvalue": pval,
        "actual_score_range": [float(pool_scores.min()), float(pool_scores.max())],
        "max_p1st": float(p1st.max()),
    }


def main() -> int:
    rng = np.random.default_rng(RNG_SEED)
    results = []
    print(f"\nO21 blend backtest — years={YEARS} weights={WEIGHTS} n_sims={N_SIMS}\n")
    print(f"{'Year':>5} {'Weight':>7} {'K':>3} {'rho':>8} {'p':>7} {'mean P(1st)':>12}")
    print("-" * 55)
    for year in YEARS:
        for w in WEIGHTS:
            r = run_year(year, w, rng)
            results.append(r)
            if "error" in r:
                print(f"{year:>5} {w:>7.2f} {'—':>3} ERROR: {r['error']}")
            else:
                print(f"{year:>5} {w:>7.2f} {r['K']:>3d} {r['rho']:>+8.3f} {r['pvalue']:>7.4f} {r['mean_p1st']:>12.4f}")
        print()

    # Aggregate: mean rho per weight (across years where rho is finite)
    print("Aggregate (mean Spearman rho across 4 years):")
    print(f"{'Weight':>7} {'mean rho':>10} {'n_yrs':>6}")
    print("-" * 28)
    for w in WEIGHTS:
        rhos = [r["rho"] for r in results if r["weight"] == w and "error" not in r and not np.isnan(r["rho"])]
        if rhos:
            print(f"{w:>7.2f} {np.mean(rhos):>+10.3f} {len(rhos):>6d}")
        else:
            print(f"{w:>7.2f} {'—':>10} {'0':>6}")

    out = ROOT / "artifacts" / "o21_blend_backtest.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(
            {
                "config": {
                    "years": YEARS,
                    "weights": WEIGHTS,
                    "n_sims": N_SIMS,
                    "n_opponents": "K-1 per year",
                    "rng_seed": RNG_SEED,
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
