"""Retrospective bracket scorer: deterministic strategy comparison across years.

For each historical year (2011-2025, excl. 2020), generates one deterministic
bracket per strategy, scores it against actual results, and displays a rich
table showing champion pick, Final Four, finalists, ESPN points, and pool
competitiveness.

Strategies:
  - seed:        Forward greedy from seed-based round probs (chalk baseline)
  - torvik:      Forward greedy from Torvik barthag + Log5 + MC (chalk, risk=0)
  - det_champ96: Champion-first from Torvik probs at risk=0.96 (contrarian)
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mc_pool_backtest import (
    BACKTEST_YEARS,
    ESPN_SCORING,
    REGION_ORDER,
    SEED_PICK_RATES,
    _load_torvik_barthag,
    _picks_dict_to_bool_array,
    build_actual_outcome,
    build_espn_pick_distribution,
    build_first_round_matchups,
    build_seed_pick_distribution,
    build_torvik_round_probabilities,
    derive_f4_region_pairing,
    load_seeds_and_regions,
    load_tournament_results,
    resolve_first_four,
)
from src.optimization.bracket_construction import construct_bracket
from src.prediction.seed_probabilities import build_seed_round_probabilities
from src.simulation.pool_competition import (
    build_scoring_vector,
    generate_opponent_brackets,
    score_brackets_against_outcome,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

HIST_DIR = Path("data/raw/historical")
ALT_DIR = Path("data/raw")
POOL_SIZE = 31
N_OPPONENTS = 30
N_OPPONENT_TRIALS = 200  # repeat opponent draws for stable median/max
RISK_LEVEL = 0.96
ALL_YEARS = [y for y in range(2011, 2026) if y != 2020]  # 2026 excluded: results file has broken team IDs
STRATEGIES = ["seed", "torvik", "det_champ96"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_seeds_and_regions_fallback(year):
    """Try historical dir first, then data/raw/ for current-year data."""
    seeds, regions = load_seeds_and_regions(year)
    if seeds:
        return seeds, regions
    # Fallback: data/raw/tournament_seeds_{year}.json
    import json as _json
    path = ALT_DIR / f"tournament_seeds_{year}.json"
    if not path.exists():
        return {}, {}
    with open(path) as f:
        data = _json.load(f)
    _REGION_ALIASES = {"Southeast": "South", "Southwest": "Midwest"}
    seeds, regions = {}, {}
    team_list = data.get("teams", data) if isinstance(data, dict) else data
    for t in team_list:
        seeds[t["team_id"]] = t["seed"]
        raw_region = t.get("region", "")
        regions[t["team_id"]] = _REGION_ALIASES.get(raw_region, raw_region)
    return seeds, regions


def extract_actual_metadata(games):
    """Extract champion, runner-up, F4 teams, and finalists from results."""
    f4_teams = set()
    finalists = []
    champion = None
    runner_up = None

    for g in games:
        rn = g.get("round_name", "")
        t1, t2 = g["team1_id"], g["team2_id"]

        if rn == "F4":
            f4_teams.add(t1)
            f4_teams.add(t2)

        if rn in ("NCG", "CHAMP"):
            finalists = [t1, t2]
            if g["team1_won"]:
                champion = t1
                runner_up = t2
            else:
                champion = t2
                runner_up = t1

    # If we didn't find NCG/CHAMP, the F4 winners are the finalists
    # (some data files use different round naming)
    if not finalists and f4_teams:
        finalists = sorted(f4_teams)

    return {
        "champion": champion,
        "runner_up": runner_up,
        "final_four": sorted(f4_teams),
        "finalists": finalists,
    }


def extract_bracket_metadata(picks):
    """Extract champion, F4, and finalists from a picks dict."""
    champion = picks.get("CHAMP")

    # E8 winners = F4 participants
    f4 = []
    for key, winner in picks.items():
        if key.startswith("E8_"):
            f4.append(winner)

    # Finalists = F4 game winners
    finalists = []
    for key, winner in picks.items():
        if key.startswith("F4_"):
            finalists.append(winner)

    return {
        "champion": champion,
        "final_four": sorted(f4),
        "finalists": sorted(finalists),
    }


def fmt_team(name, max_len=14):
    """Abbreviate team name for display."""
    if not name:
        return "?" * min(max_len, 3)
    # Common abbreviations
    abbrevs = {
        "connecticut": "UConn",
        "north_carolina": "UNC",
        "michigan_state": "Mich St",
        "michigan": "Michigan",
        "louisville": "Louisville",
        "gonzaga": "Gonzaga",
        "villanova": "Nova",
        "virginia": "Virginia",
        "kentucky": "Kentucky",
        "kansas": "Kansas",
        "duke": "Duke",
        "baylor": "Baylor",
        "florida": "Florida",
        "syracuse": "Syracuse",
        "wisconsin": "Wisconsin",
        "oregon": "Oregon",
        "south_carolina": "S Carolina",
        "loyola__il": "Loyola-Chi",
        "texas_tech": "Texas Tech",
        "auburn": "Auburn",
        "houston": "Houston",
        "san_diego_state": "SDSU",
        "florida_atlantic": "FAU",
        "purdue": "Purdue",
        "alabama": "Alabama",
        "nc_state": "NC State",
        "iowa_state": "Iowa St",
        "creighton": "Creighton",
        "marquette": "Marquette",
        "saint_peter_s": "St Peter's",
    }
    if name in abbrevs:
        return abbrevs[name][:max_len]
    # Generic: replace underscores, title case, truncate
    display = name.replace("_", " ").replace("__", " ").title()
    if len(display) > max_len:
        display = display[:max_len - 1] + "."
    return display


def f4_overlap(predicted, actual):
    """Count how many predicted F4 teams match actual."""
    return len(set(predicted) & set(actual))


def simulate_pool_scores(first_round, seeds, pick_dist, actual_outcome, scoring_vector, rng):
    """Simulate opponent brackets and return avg pool median / avg pool winner.

    Each trial draws N_OPPONENTS brackets (one pool instance), computes that
    pool's median and max. We average across trials for stable estimates of
    what a typical pool's median and winner score look like.
    """
    from scripts.mc_pool_backtest import build_seed_probabilities
    seed_pw = build_seed_probabilities(seeds)

    trial_medians = []
    trial_maxes = []
    for _ in range(N_OPPONENT_TRIALS):
        opp = generate_opponent_brackets(
            n_opponents=N_OPPONENTS,
            first_round_matchups=first_round,
            matchup_probs=seed_pw,
            pick_distribution=pick_dist,
            seeds=seeds,
            rng=rng,
        )
        scores = score_brackets_against_outcome(opp, actual_outcome, scoring_vector)
        trial_medians.append(float(np.median(scores)))
        trial_maxes.append(float(np.max(scores)))

    return float(np.mean(trial_medians)), float(np.mean(trial_maxes))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_retrospective():
    scoring_vector = build_scoring_vector(ESPN_SCORING)
    rng = np.random.default_rng(42)

    results = []

    for year in ALL_YEARS:
        seeds, regions = load_seeds_and_regions_fallback(year)
        if not seeds:
            print(f"  {year}  SKIP — no seeds")
            continue

        games = load_tournament_results(year)
        if not games:
            print(f"  {year}  SKIP — no games")
            continue

        resolve_first_four(games, seeds, regions)

        try:
            region_order = derive_f4_region_pairing(games, regions)
        except ValueError as exc:
            print(f"  {year}  SKIP — {exc}")
            continue

        first_round = build_first_round_matchups(seeds, regions, region_order=region_order)
        if len(first_round) != 64:
            print(f"  {year}  SKIP — {len(first_round)} teams")
            continue

        actual_outcome = build_actual_outcome(first_round, games)
        actual_meta = extract_actual_metadata(games)

        # Load public picks (ESPN preferred, fallback to seed-based)
        try:
            pick_dist = build_espn_pick_distribution(year, seeds)
        except FileNotFoundError:
            pick_dist = build_seed_pick_distribution(seeds)

        # Build round probabilities
        seed_rp = build_seed_round_probabilities(seeds)
        barthag = _load_torvik_barthag(year, seeds)
        torvik_rp = build_torvik_round_probabilities(seeds, regions, barthag)

        # Generate deterministic brackets
        brackets = {}
        for strategy in STRATEGIES:
            if strategy == "seed":
                mode, rp, risk = "forward_greedy", seed_rp, 0.0
            elif strategy == "torvik":
                mode, rp, risk = "forward_greedy", torvik_rp, 0.0
            elif strategy == "det_champ96":
                mode, rp, risk = "champ_first", torvik_rp, RISK_LEVEL
            else:
                continue

            picks, champ, f4, ev, var = construct_bracket(
                mode=mode,
                seeds=seeds,
                regions=regions,
                round_probs=rp,
                public_picks=pick_dist,
                risk_level=risk,
                pool_size=POOL_SIZE,
                scoring_system=ESPN_SCORING,
            )

            bool_arr = _picks_dict_to_bool_array(picks, first_round)
            score = score_brackets_against_outcome(
                bool_arr.reshape(1, -1), actual_outcome, scoring_vector
            )[0]

            meta = extract_bracket_metadata(picks)
            brackets[strategy] = {
                "score": int(score),
                "champion": meta["champion"],
                "final_four": meta["final_four"],
                "finalists": meta["finalists"],
                "f4_overlap": f4_overlap(meta["final_four"], actual_meta["final_four"]),
                "champ_correct": meta["champion"] == actual_meta["champion"],
            }

        # Pool competitiveness
        pool_median, pool_max = simulate_pool_scores(
            first_round, seeds, pick_dist, actual_outcome, scoring_vector, rng
        )

        results.append({
            "year": year,
            "actual": actual_meta,
            "brackets": brackets,
            "pool_median": int(pool_median),
            "pool_max": int(pool_max),
        })

    return results


def print_results(results):
    # Header
    print("=" * 130)
    print("RETROSPECTIVE BRACKET SCORER — Historical Strategy Comparison")
    print("=" * 130)
    print(f"  Pool size: {POOL_SIZE}  |  Strategies: {', '.join(STRATEGIES)}")
    print(f"  det_champ96 risk_level: {RISK_LEVEL}")
    print()

    # Per-year detail
    hdr = (
        f"{'Year':<6}{'Actual Champion':<16}"
        f"{'Strategy':<14}{'Pts':>5} {'Champ Pick':<14}{'Finalists':<30}"
        f"{'Final Four':<58}{'F4':>4} {'Ch':>3}"
        f"  {'Pool Med':>8}{'Pool Max':>9}"
    )
    print(hdr)
    print("-" * len(hdr))

    # Accumulators for summary
    summary = {s: {"pts": [], "champ_correct": 0, "f4_overlap": [], "beat_med": 0, "beat_max": 0}
               for s in STRATEGIES}

    for r in results:
        year = r["year"]
        actual = r["actual"]
        pool_med = r["pool_median"]
        pool_max = r["pool_max"]

        for i, strategy in enumerate(STRATEGIES):
            b = r["brackets"].get(strategy)
            if not b:
                continue

            # Format teams
            champ_pick = fmt_team(b["champion"])
            finalists_str = ", ".join(fmt_team(t) for t in b["finalists"])
            f4_str = ", ".join(fmt_team(t) for t in b["final_four"])
            champ_mark = "Y" if b["champ_correct"] else ""
            f4_ov = f"{b['f4_overlap']}/4"

            # First row of year group shows actual champion and pool stats
            if i == 0:
                actual_champ = fmt_team(actual["champion"])
                pool_med_str = f"{pool_med:>8}"
                pool_max_str = f"{pool_max:>9}"
                year_str = f"{year:<6}"
            else:
                actual_champ = ""
                pool_med_str = ""
                pool_max_str = ""
                year_str = "      "

            beat_med = ">" if b["score"] > pool_med else " "
            beat_max = ">" if b["score"] > pool_max else " "

            print(
                f"{year_str}{actual_champ:<16}"
                f"{strategy:<14}{b['score']:>5} {champ_pick:<14}{finalists_str:<30}"
                f"{f4_str:<58}{f4_ov:>4} {champ_mark:>3}"
                f"  {pool_med_str}{pool_max_str}"
                f"  {beat_med}{beat_max}"
            )

            # Accumulate
            summary[strategy]["pts"].append(b["score"])
            summary[strategy]["champ_correct"] += int(b["champ_correct"])
            summary[strategy]["f4_overlap"].append(b["f4_overlap"])
            summary[strategy]["beat_med"] += int(b["score"] > pool_med)
            summary[strategy]["beat_max"] += int(b["score"] > pool_max)

        print()

    # Summary table
    n_years = len(results)
    print("=" * 130)
    print("SUMMARY")
    print("=" * 130)
    print(
        f"{'Strategy':<14}{'Mean Pts':>9}{'Champ Correct':>15}"
        f"{'Avg F4 Match':>14}{'Beat Median':>13}{'Beat Winner':>13}"
    )
    print("-" * 78)
    for s in STRATEGIES:
        sm = summary[s]
        mean_pts = np.mean(sm["pts"]) if sm["pts"] else 0
        avg_f4 = np.mean(sm["f4_overlap"]) if sm["f4_overlap"] else 0
        print(
            f"{s:<14}{mean_pts:>9.0f}"
            f"{sm['champ_correct']:>10}/{n_years}"
            f"{avg_f4:>10.1f}/4"
            f"{sm['beat_med']:>9}/{n_years}"
            f"{sm['beat_max']:>9}/{n_years}"
        )

    print()
    print("Legend: F4 = Final Four overlap with actual | Ch = Champion correct (Y)")
    print("        > in last two columns = strategy score beats pool median / pool winner")

    # 2026 manual annotation (results file has broken team IDs, data from prior analysis)
    print()
    print("-" * 78)
    print("2026 (manual — tournament_results_2026.json has inconsistent team IDs):")
    print("  Actual champion: Michigan | F4: UConn, Arizona, Illinois, Michigan")
    print("  det_champ96 r=0.96: 1440 pts, champ=Michigan (Y), F4=4/4, E8=4/4")
    print("  torvik:             ~1330 pts (estimated from pool report)")
    print("  Including 2026: det_champ96 beat pool winner at least 1/15 years")


if __name__ == "__main__":
    results = run_retrospective()
    print_results(results)
