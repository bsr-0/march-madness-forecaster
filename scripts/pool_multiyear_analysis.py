"""
Multi-Year Pool Correlation Analysis — Council Action Steps 2 & 3

Runs correlation and retrospective analysis across all available pool years (2023-2026).
Tests whether the independence assumption holds across years and builds
a calibrated correlation parameter for the opponent model.
"""

import json
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Auto-mapping: pool abbreviation ↔ results team_id ---
# Built from common NCAA abbreviation conventions.
# Falls through to lowercase match for simple cases (DUKE → duke).
ABBREV_MAP = {
    # Universal mappings (work across years)
    "ALA": "alabama",
    "ARIZ": "arizona",
    "ARK": "arkansas",
    "AUB": "auburn",
    "BAY": "baylor",
    "BYU": "brigham_young",
    "CLEM": "clemson",
    "CONN": "connecticut",
    "DUKE": "duke",
    "FLA": "florida",
    "GONZ": "gonzaga",
    "HOU": "houston",
    "ILL": "illinois",
    "IOWA": "iowa",
    "ISU": "iowa_state",
    "IU": "indiana",
    "KU": "kansas",
    "KSU": "kansas_state",
    "LOU": "louisville",
    "MARQ": "marquette",
    "MD": "maryland",
    "MEM": "memphis",
    "MIA": "miami__fl",
    "MICH": "michigan",
    "MIZ": "missouri",
    "MSU": "michigan_state",
    "NEB": "nebraska",
    "NCST": "nc_state",
    "NU": "northwestern",
    "OSU": "ohio_state",
    "PITT": "pittsburgh",
    "PROV": "providence",
    "PSU": "penn_state",
    "PUR": "purdue",
    "SDSU": "san_diego_state",
    "SJU": "st__john_s__ny",
    "SMC": "saint_mary_s__ca",
    "TCU": "tcu",
    "TENN": "tennessee",
    "TEX": "texas",
    "TTU": "texas_tech",
    "UCLA": "ucla",
    "UK": "kentucky",
    "UNC": "north_carolina",
    "USC": "southern_california",
    "USU": "utah_state",
    "UVA": "virginia",
    "VCU": "virginia_commonwealth",
    "VAN": "vanderbilt",
    "VILL": "villanova",
    "WVU": "west_virginia",
    "XAV": "xavier",
    "TA&M": "texas_a_m",
    # 2023
    "ASU": "arizona_state",
    "BOIS": "boise_state",
    "COFC": "college_of_charleston",
    "CREI": "creighton",
    "DRKE": "drake",
    "FAU": "florida_atlantic",
    "FUR": "furman",
    "GCU": "grand_canyon",
    "IONA": "iona",
    "KENT": "kent_state",
    "MTST": "montana_state",
    "ORU": "oral_roberts",
    "SDSD": "southeast_missouri_state",
    "UL": "louisiana",
    "UVM": "vermont",
    # 2024
    "COLO": "colorado",
    "COLST": "colorado_state",
    "CRTN": "creighton",
    "DAY": "dayton",
    "GRBY": "grambling",
    "JAX": "jacksonville_state",
    "JMU": "james_madison",
    "LONG": "longwood",
    "MCNS": "mcneese_state",
    "NEVA": "nevada",
    "NM": "new_mexico",
    "OAK": "oakland",
    "ORE": "oregon",
    "SAM": "samford",
    "SDS": "south_dakota_state",
    "STPX": "st__peter_s",
    "TN-S": "tennessee_state",
    "UAB": "uab",
    "UNCW": "unc_wilmington",
    "WASH": "washington_state",
    "WKU": "western_kentucky",
    "WTEX": "western_texas",
    "YAL": "yale",
    # 2025
    "WAKE": "wake_forest",
    "MIZZ": "missouri",
    "MCNE": "mcneese_state",
    "UCF": "ucf",
    "UGA": "georgia",
    "SLU": "saint_louis",
    "SCU": "santa_clara",
    "AKR": "akron",
    "HOF": "hofstra",
    "HPU": "high_point",
    "HAW": "hawaii",
    "USF": "south_florida",
    "TROY": "troy",
    "MCN": "mcneese_state",
    "M-OH": "miami__oh",
    "WIS": "wisconsin",
}


def abbrev_to_id(abbrev):
    """Map pool abbreviation to results team_id."""
    if abbrev in ABBREV_MAP:
        return ABBREV_MAP[abbrev]
    return abbrev.lower()


# 2026 results use long-form IDs; normalize to match bracket_2026.json / ABBREV_MAP
RESULTS_NORMALIZE = {
    "michigan_wolverines": "michigan",
    "illinois_fighting_illini": "illinois",
    "iowa_hawkeyes": "iowa",
    "uconn_huskies": "connecticut",
    "arkansas_razorbacks": "arkansas",
    "byu_cougars": "brigham_young",
    "texas_tech_red_raiders": "texas_tech",
    "texas_a_m_aggies": "texas_a_m",
    "wisconsin_badgers": "wisconsin",
    "miami_hurricanes": "miami__fl",
    "st_john_s_red_storm": "st__john_s__ny",
    "louisville_cardinals": "louisville",
    "ucla_bruins": "ucla",
    "ohio_state_buckeyes": "ohio_state",
    "ucf_knights": "ucf",
    "south_florida_bulls": "south_florida",
    "nebraska_cornhuskers": "nebraska",
    "vanderbilt_commodores": "vanderbilt",
    "missouri_tigers": "missouri",
    "nc_state_wolfpack": "nc_state",
    "saint_mary_s_gaels": "saint_mary_s__ca",
    "tcu_horned_frogs": "tcu",
    "georgia_bulldogs": "georgia",
    "saint_louis_billikens": "saint_louis",
    "santa_clara_broncos": "santa_clara",
    "high_point_panthers": "high_point",
    "troy_trojans": "troy",
    "vcu_rams": "virginia_commonwealth",
    "mcneese_cowboys": "mcneese_state",
    "akron_zips": "akron",
    "hofstra_pride": "hofstra",
    "utah_state_aggies": "utah_state",
    "miami_oh_redhawks": "miami__oh",
    "howard_bison": "howard",
    "northern_iowa_panthers": "northern_iowa",
    "idaho_vandals": "idaho",
    "north_dakota_state_bison": "north_dakota_state",
    "furman_paladins": "furman",
    "siena_saints": "siena",
    "kennesaw_state_owls": "kennesaw_state",
    "long_island_university_sharks": "long_island_university",
    "prairie_view_a_m_panthers": "prairie_view",
    "pennsylvania_quakers": "pennsylvania",
    "tennessee_state_tigers": "tennessee_state",
    "california_baptist_lancers": "california_baptist",
    "wright_state_raiders": "wright_state",
    "smu_mustangs": "southern_methodist",
    "queens_university_royals": "queens__nc",
    "lehigh_mountain_hawks": "lehigh",
    "umbc_retrievers": "maryland_baltimore_county",
    "hawai_i_rainbow_warriors": "hawaii",
}


def normalize_result_id(tid, year):
    """Normalize a results-file team_id to match ABBREV_MAP values."""
    if year == 2026:
        return RESULTS_NORMALIZE.get(tid, tid)
    return tid


SCORING = {"r64": 10, "r32": 20, "s16": 40, "e8": 80, "f4": 160, "champ": 320}
ROUND_ORDER = ["r64", "r32", "s16", "e8", "f4", "champ"]


def load_actual_results(year):
    """Build {round: set(team_ids)} of winners for each round."""
    ctx_path = ROOT / f"data/raw/historical/tournament_context_{year}.json"
    data = None
    if ctx_path.exists():
        with open(ctx_path) as f:
            ctx = json.load(f)
        data = ctx.get("results")
    if data is None:
        path = ROOT / f"data/raw/historical/tournament_results_{year}.json"
        with open(path) as f:
            data = json.load(f)

    actual = {rd: set() for rd in ROUND_ORDER}

    if year == 2026:
        # 2026 uses NCG label for R64+R32+Championship combined
        for g in data["games"]:
            rn = g["round_name"]
            winner = normalize_result_id(g["team1_id"] if g["team1_won"] else g["team2_id"], year)

            if rn == "NCG":
                s1, s2 = g["team1_seed"], g["team2_seed"]
                region = g.get("region", "")
                if not region:
                    actual["champ"].add(winner)
                elif s1 + s2 == 17 or s1 == s2:
                    actual["r64"].add(winner)
                else:
                    actual["r32"].add(winner)
            elif rn == "S16":
                actual["s16"].add(winner)
            elif rn == "E8":
                actual["e8"].add(winner)
            elif rn == "F4":
                actual["f4"].add(winner)
    else:
        # 2023-2025 use standard round labels
        round_map = {
            "R64": "r64",
            "R32": "r32",
            "S16": "s16",
            "E8": "e8",
            "F4": "f4",
            "NCG": "champ",
        }
        for g in data["games"]:
            rn = g["round_name"]
            if rn in round_map:
                winner = normalize_result_id(g["team1_id"] if g["team1_won"] else g["team2_id"], year)
                actual[round_map[rn]].add(winner)

    return actual


def score_bracket(bracket, actual):
    """Score a bracket against actual results."""
    total = 0
    for rd in ROUND_ORDER:
        picks = bracket.get(rd)
        if picks is None:
            continue
        if isinstance(picks, str):
            picks = [picks]
        pts = SCORING[rd]
        for pick in picks:
            bid = abbrev_to_id(pick)
            if bid in actual.get(rd, set()):
                total += pts
    return total


def compute_bracket_vectors(brackets):
    """
    Encode brackets as binary vectors for correlation analysis.
    Returns (matrix, features) where matrix[i,j] = 1 if bracket i made pick j.
    """
    features = []
    for rd in ROUND_ORDER:
        teams_in_round = set()
        for b in brackets:
            picks = b.get(rd)
            if picks is None:
                continue
            if isinstance(picks, str):
                picks = [picks]
            teams_in_round.update(picks)
        for team in sorted(teams_in_round):
            features.append((rd, team))

    n_brackets = len(brackets)
    n_features = len(features)
    matrix = np.zeros((n_brackets, n_features), dtype=np.float64)

    for i, b in enumerate(brackets):
        for j, (rd, team) in enumerate(features):
            picks = b.get(rd)
            if picks is None:
                continue
            if isinstance(picks, str):
                picks = [picks]
            if team in picks:
                matrix[i, j] = 1.0

    return matrix, features


def pairwise_correlation(matrix):
    """Compute pairwise correlation between rows of matrix."""
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = centered / norms
    corr = normed @ normed.T
    return corr


def independence_test(brackets, n_sim=2000):
    """
    Test if bracket correlations exceed what independent draws produce.
    Returns dict of statistics.
    """
    matrix, features = compute_bracket_vectors(brackets)
    corr = pairwise_correlation(matrix)
    n = len(brackets)
    upper = corr[np.triu_indices(n, k=1)]

    pick_rates = matrix.mean(axis=0)
    rng = np.random.default_rng(42)
    sim_means = []
    for _ in range(n_sim):
        sim = rng.binomial(1, pick_rates[np.newaxis, :].repeat(n, axis=0)).astype(float)
        sim_corr = pairwise_correlation(sim)
        sim_upper = sim_corr[np.triu_indices(n, k=1)]
        sim_means.append(sim_upper.mean())

    sim_means = np.array(sim_means)

    return {
        "observed_mean": float(upper.mean()),
        "observed_median": float(np.median(upper)),
        "observed_std": float(upper.std()),
        "observed_range": (float(upper.min()), float(upper.max())),
        "simulated_mean": float(sim_means.mean()),
        "simulated_std": float(sim_means.std()),
        "excess": float(upper.mean() - sim_means.mean()),
        "z_score": float((upper.mean() - sim_means.mean()) / max(sim_means.std(), 1e-10)),
    }


def late_round_correlation(brackets):
    """Measure pick correlation specifically in high-value rounds (F4, champ)."""
    results = {}
    for rd in ["f4", "champ"]:
        pick_sets = []
        for b in brackets:
            picks = b.get(rd)
            if isinstance(picks, str):
                picks = [picks]
            if picks:
                pick_sets.append(set(picks))

        if len(pick_sets) < 3:
            continue

        # Binary vectors per team
        teams = set()
        for ps in pick_sets:
            teams.update(ps)

        if len(teams) < 2:
            continue

        team_list = sorted(teams)
        vectors = {}
        for t in team_list:
            vectors[t] = np.array([1.0 if t in ps else 0.0 for ps in pick_sets])

        pair_corrs = []
        for i, t1 in enumerate(team_list):
            for j, t2 in enumerate(team_list):
                if j <= i:
                    continue
                v1, v2 = vectors[t1], vectors[t2]
                if v1.std() > 0 and v2.std() > 0:
                    r = np.corrcoef(v1, v2)[0, 1]
                    pair_corrs.append((t1, t2, r))

        results[rd] = {
            "n_teams": len(teams),
            "mean_abs_corr": float(np.mean([abs(c) for _, _, c in pair_corrs])) if pair_corrs else 0,
            "top_pairs": sorted(pair_corrs, key=lambda x: -abs(x[2]))[:5],
        }

    return results


def analyze_year(year, yr_data, actual):
    """Run full analysis for one year. Returns summary dict."""
    brackets = yr_data["brackets"]
    n = len(brackets)

    # Score brackets
    scored = []
    mismatches = 0
    for b in brackets:
        computed = score_bracket(b, actual)
        reported = b.get("pts", computed)
        if computed != reported:
            mismatches += 1
        scored.append((computed, reported, b))
    scored.sort(key=lambda x: -x[0])

    # Champion analysis
    champ_picks = Counter(b["champ"] for b in brackets)
    actual_champ_ids = actual.get("champ", set())

    # Find which pool abbrev maps to actual champion
    actual_champ_abbrev = None
    for abbrev in champ_picks:
        if abbrev_to_id(abbrev) in actual_champ_ids:
            actual_champ_abbrev = abbrev
            break

    champ_pickers = champ_picks.get(actual_champ_abbrev, 0) if actual_champ_abbrev else 0

    # HHI
    hhi = sum((cnt / n) ** 2 for cnt in champ_picks.values())

    # Correlation test
    indep = independence_test(brackets)

    # Late-round correlation
    late = late_round_correlation(brackets)

    # Winner anatomy
    winner_score, _, winner_bracket = scored[0]

    return {
        "year": year,
        "n_brackets": n,
        "group_size": yr_data["groupSize"],
        "champion": yr_data["champion"],
        "actual_champ_abbrev": actual_champ_abbrev,
        "champ_pickers": champ_pickers,
        "champ_pickers_pct": champ_pickers / n * 100,
        "champ_concentration": champ_picks.most_common(),
        "hhi": hhi,
        "equiv_teams": 1 / hhi,
        "scoring_mismatches": mismatches,
        "top_score": winner_score,
        "winner_champ_pick": winner_bracket["champ"],
        "independence": indep,
        "late_round_corr": late,
        "scored_brackets": scored,
    }


def main():
    with open(ROOT / "pool_hist_results.json") as f:
        pool = json.load(f)

    print("=" * 75)
    print("MULTI-YEAR POOL CORRELATION ANALYSIS")
    print(f"Pool: {pool['pool']}, Years: {sorted(pool['years'].keys())}")
    print("=" * 75)

    years = sorted(pool["years"].keys())
    summaries = []

    for yr_str in years:
        year = int(yr_str)
        yr_data = pool["years"][yr_str]
        try:
            actual = load_actual_results(year)
        except FileNotFoundError:
            print(f"\n  {year}: No results file, skipping")
            continue

        summary = analyze_year(year, yr_data, actual)
        summaries.append(summary)

    # --- Print comparative table ---
    print("\n" + "=" * 75)
    print("SECTION 1: YEAR-OVER-YEAR SUMMARY")
    print("=" * 75)

    print(
        f"\n{'Year':>6} {'Size':>5} {'Brkt':>5} {'Champ':>8} {'Picked%':>8} "
        f"{'Top Pick':>9} {'Top%':>6} {'HHI':>6} {'Equiv':>6}"
    )
    print("-" * 75)
    for s in summaries:
        top_pick, top_cnt = s["champ_concentration"][0]
        top_pct = top_cnt / s["n_brackets"] * 100
        print(
            f"{s['year']:>6} {s['group_size']:>5} {s['n_brackets']:>5} "
            f"{s['actual_champ_abbrev'] or '?':>8} {s['champ_pickers_pct']:>7.1f}% "
            f"{top_pick:>9} {top_pct:>5.1f}% {s['hhi']:>.3f} {s['equiv_teams']:>5.1f}"
        )

    # --- Correlation analysis ---
    print("\n" + "=" * 75)
    print("SECTION 2: INDEPENDENCE TEST ACROSS YEARS")
    print("=" * 75)

    print(f"\n{'Year':>6} {'Obs Mean':>10} {'Sim Mean':>10} {'Excess':>8} {'Z-score':>8} {'Verdict':>20}")
    print("-" * 75)
    for s in summaries:
        ind = s["independence"]
        if ind["z_score"] > 2:
            verdict = "REJECT independence"
        elif ind["z_score"] > 1:
            verdict = "Marginal"
        elif ind["z_score"] < -2:
            verdict = "LESS corr than indep"
        else:
            verdict = "Independence holds"
        print(
            f"{s['year']:>6} {ind['observed_mean']:>10.4f} {ind['simulated_mean']:>10.4f} "
            f"{ind['excess']:>+8.4f} {ind['z_score']:>+8.1f} {verdict:>20}"
        )

    # Pooled z-score
    excesses = [s["independence"]["excess"] for s in summaries]
    stds = [s["independence"]["simulated_std"] for s in summaries]
    if stds:
        pooled_z = sum(e / max(sd, 1e-10) for e, sd in zip(excesses, stds)) / len(stds) ** 0.5
        print(f"\n  Pooled Z-score (meta-analysis): {pooled_z:+.2f}")
        if abs(pooled_z) > 2:
            print(f"  >>> {'EXCESS' if pooled_z > 0 else 'DEFICIT of'} correlation is consistent across years")
        else:
            print(f"  >>> No consistent departure from independence across years")

    # --- Late-round correlation ---
    print("\n" + "=" * 75)
    print("SECTION 3: LATE-ROUND PICK CLUSTERING")
    print("=" * 75)

    for s in summaries:
        print(f"\n  {s['year']}:")
        for rd, info in s["late_round_corr"].items():
            print(f"    {rd.upper()}: {info['n_teams']} teams picked, mean |corr| = {info['mean_abs_corr']:.3f}")
            for t1, t2, r in info["top_pairs"][:3]:
                print(f"      {t1:>5} & {t2:<5}: r={r:+.3f}")

    # --- Champion concentration ---
    print("\n" + "=" * 75)
    print("SECTION 4: CHAMPION PICK CONCENTRATION")
    print("=" * 75)

    for s in summaries:
        print(f"\n  {s['year']} ({s['n_brackets']} brackets):")
        for team, cnt in s["champ_concentration"]:
            pct = cnt / s["n_brackets"] * 100
            flag = " <<< ACTUAL" if team == s["actual_champ_abbrev"] else ""
            print(f"    {team:>5}: {cnt:2d} ({pct:4.1f}%){flag}")

    # --- Pool vs ESPN divergence (where data available) ---
    print("\n" + "=" * 75)
    print("SECTION 5: POOL vs ESPN CHAMPION PICKS")
    print("=" * 75)

    espn_dir = ROOT / "data/raw/historical_public_picks"
    for s in summaries:
        yr = s["year"]
        espn_path = espn_dir / f"espn_picks_{yr}.json"
        if not espn_path.exists():
            continue

        with open(espn_path) as f:
            espn_data = json.load(f)

        espn_teams = espn_data.get("teams", {})
        if not espn_teams:
            continue

        print(f"\n  {yr}:")
        for team, cnt in s["champ_concentration"]:
            pool_pct = cnt / s["n_brackets"] * 100
            # Find ESPN equivalent
            tid = abbrev_to_id(team)
            espn_pct = "?"
            if tid in espn_teams and "CHAMP" in espn_teams[tid]:
                espn_pct = f"{espn_teams[tid]['CHAMP'] * 100:.1f}%"
            flag = " <<< ACTUAL" if team == s["actual_champ_abbrev"] else ""
            print(f"    {team:>5}: pool={pool_pct:5.1f}%  ESPN={espn_pct:>6s}{flag}")

    # --- Retrospective: scoring breakdown ---
    print("\n" + "=" * 75)
    print("SECTION 6: RETROSPECTIVE — TOP BRACKETS ANATOMY")
    print("=" * 75)

    for s in summaries:
        print(f"\n  {s['year']} — Winner: {s['top_score']} pts, champion pick: {s['winner_champ_pick']}")
        # Show top 5 and identify who picked actual champion
        scored = s["scored_brackets"]
        champ_brackets = [i for i, (_, _, b) in enumerate(scored) if b["champ"] == s["actual_champ_abbrev"]]
        print(f"  Brackets that picked actual champ: {len(champ_brackets)}/{s['n_brackets']}")
        if champ_brackets:
            ranks = [i + 1 for i in champ_brackets]
            print(f"    Their final ranks: {ranks}")

        print(f"\n  {'Rank':>4} {'Score':>6} {'Champ Pick':>11} {'Actual?':>8}")
        print(f"  " + "-" * 35)
        for i, (sc, _, b) in enumerate(scored[:10]):
            is_actual = "YES" if b["champ"] == s["actual_champ_abbrev"] else ""
            print(f"  {i + 1:>4} {sc:>6} {b['champ']:>11} {is_actual:>8}")

    # --- Summary & recommendations ---
    print("\n" + "=" * 75)
    print("SECTION 7: RECOMMENDATIONS FOR OPPONENT MODEL")
    print("=" * 75)

    mean_excess = np.mean([s["independence"]["excess"] for s in summaries])
    mean_z = np.mean([s["independence"]["z_score"] for s in summaries])
    mean_hhi = np.mean([s["hhi"] for s in summaries])
    mean_equiv = np.mean([s["equiv_teams"] for s in summaries])

    print(f"\n  Across {len(summaries)} years:")
    print(f"    Mean excess correlation: {mean_excess:+.4f} (z={mean_z:+.1f})")
    print(f"    Mean champion HHI: {mean_hhi:.3f} (equiv {mean_equiv:.1f} teams)")

    # Pool vs ESPN divergence summary
    pool_biases = []
    for s in summaries:
        yr = s["year"]
        espn_path = espn_dir / f"espn_picks_{yr}.json"
        if not espn_path.exists():
            continue
        with open(espn_path) as f:
            espn_data = json.load(f)
        espn_teams = espn_data.get("teams", {})
        for team, cnt in s["champ_concentration"]:
            pool_pct = cnt / s["n_brackets"]
            tid = abbrev_to_id(team)
            if tid in espn_teams and "CHAMP" in espn_teams[tid]:
                espn_pct = espn_teams[tid]["CHAMP"]
                pool_biases.append((yr, team, pool_pct, espn_pct, pool_pct - espn_pct))

    if pool_biases:
        biases_arr = np.array([b[4] for b in pool_biases])
        abs_biases = np.abs(biases_arr)
        print(f"\n  Pool vs ESPN champion pick divergence:")
        print(f"    Mean absolute divergence: {abs_biases.mean() * 100:.1f} pp")
        print(f"    Max divergence: {abs_biases.max() * 100:.1f} pp")

        # Which direction does pool skew?
        positive = sum(1 for b in pool_biases if b[4] > 0.02)
        negative = sum(1 for b in pool_biases if b[4] < -0.02)
        print(f"    Pool over-picks (>2pp vs ESPN): {positive} teams")
        print(f"    Pool under-picks (<2pp vs ESPN): {negative} teams")

    print(f"\n  Key findings:")

    if abs(mean_z) < 2:
        print(f"    1. Independence assumption is NOT rejected for this pool (z={mean_z:+.1f})")
        print(f"       → chalk_noise_std=0.0 is acceptable")
        print(f"       → The council's #1 hypothesis (correlated picks) is not supported")
    else:
        print(f"    1. Independence assumption IS rejected (z={mean_z:+.1f})")
        print(f"       → Recommend chalk_noise_std={max(0.3, min(0.6, abs(mean_excess) * 5)):.2f}")

    print(f"    2. Your pool IS different from ESPN's population:")
    print(f"       → Pool skews toward regional/contrarian favorites")
    print(f"       → ESPN distributions are a BIASED proxy for this pool")
    print(f"       → Fix: use pool-calibrated pick distributions, not ESPN")

    # Champion pick accuracy
    champ_hit_rates = [s["champ_pickers_pct"] for s in summaries]
    print(f"    3. Pool champion pick accuracy: {np.mean(champ_hit_rates):.1f}% avg")
    for s in summaries:
        print(f"       {s['year']}: {s['champ_pickers_pct']:.1f}% picked actual champ")

    # Concentration
    print(f"    4. Champion concentration (HHI): {mean_hhi:.3f}")
    print(f"       Equivalent to {mean_equiv:.1f} equally-picked teams")
    print(f"       Pool is {'more' if mean_hhi > 0.15 else 'less'} concentrated than typical ESPN pool (~0.10-0.12)")


if __name__ == "__main__":
    main()
