"""
Pool Correlation Analysis — Council Action Step 2

Analyzes the 2026 pool bracket data to:
1. Compute pick frequencies per team per round (pool vs ESPN)
2. Measure inter-bracket correlation (independence assumption test)
3. Score all brackets against actual 2026 results
4. Build retrospective: what did top-10 vs #11 look like?
"""

import json
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Team abbreviation mapping ---
# Pool data uses short abbrevs; results use full IDs
ABBREV_TO_ID = {
    "ARIZ": "arizona",
    "PUR": "purdue",
    "GONZ": "gonzaga",
    "ARK": "arkansas",
    "WIS": "wisconsin",
    "BYU": "brigham_young",
    "MIA": "miami__fl",
    "VILL": "villanova",
    "USU": "utah_state",
    "MIZ": "missouri",
    "TEX": "texas",
    "HPU": "high_point",
    "HAW": "hawaii",
    "DUKE": "duke",
    "CONN": "connecticut",
    "MSU": "michigan_state",
    "KU": "kansas",
    "SJU": "st__john_s__ny",
    "LOU": "louisville",
    "UCLA": "ucla",
    "OSU": "ohio_state",
    "TCU": "tcu",
    "UCF": "ucf",
    "USF": "south_florida",
    "FLA": "florida",
    "HOU": "houston",
    "ILL": "illinois",
    "NEB": "nebraska",
    "VAN": "vanderbilt",
    "UNC": "north_carolina",
    "SMC": "saint_mary_s__ca",
    "CLEM": "clemson",
    "IOWA": "iowa",
    "TA&M": "texas_a_m",
    "VCU": "virginia_commonwealth",
    "MCN": "mcneese_state",
    "TROY": "troy",
    "MICH": "michigan",
    "ISU": "iowa_state",
    "UVA": "virginia",
    "ALA": "alabama",
    "TTU": "texas_tech",
    "TENN": "tennessee",
    "UK": "kentucky",
    "UGA": "georgia",
    "SLU": "saint_louis",
    "SCU": "santa_clara",
    "AKR": "akron",
    "HOF": "hofstra",
    "M-OH": "miami__oh",
}

# Reverse for results -> abbrev
ID_TO_ABBREV = {v: k for k, v in ABBREV_TO_ID.items()}

# Results data uses slightly different IDs than bracket data
RESULTS_ID_TO_BRACKET_ID = {
    "michigan_wolverines": "michigan",
    "illinois_fighting_illini": "illinois",
    "iowa_hawkeyes": "iowa",
    "uconn_huskies": "connecticut",
    "arkansas_razorbacks": "arkansas",
    "byu_cougars": "brigham_young",
    "florida": "florida",
    "houston": "houston",
    "michigan_state": "michigan_state",
    "duke": "duke",
    "arizona": "arizona",
    "purdue": "purdue",
    "gonzaga": "gonzaga",
    "clemson": "clemson",
    "tennessee": "tennessee",
    "alabama": "alabama",
    "iowa_state": "iowa_state",
    "kentucky": "kentucky",
    "texas": "texas",
    "texas_tech_red_raiders": "texas_tech",
    "texas_a_m_aggies": "texas_a_m",
    "wisconsin_badgers": "wisconsin",
    "miami_hurricanes": "miami__fl",
    "villanova": "villanova",
    "kansas": "kansas",
    "st_john_s_red_storm": "st__john_s__ny",
    "louisville_cardinals": "louisville",
    "ucla_bruins": "ucla",
    "ohio_state_buckeyes": "ohio_state",
    "ucf_knights": "ucf",
    "south_florida_bulls": "south_florida",
    "north_carolina": "north_carolina",
    "nebraska_cornhuskers": "nebraska",
    "vanderbilt_commodores": "vanderbilt",
    "missouri_tigers": "missouri",
    "nc_state_wolfpack": "nc_state",
    "saint_mary_s_gaels": "saint_mary_s__ca",
    "tcu_horned_frogs": "tcu",
    "virginia": "virginia",
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


def normalize_results_id(rid):
    return RESULTS_ID_TO_BRACKET_ID.get(rid, rid)


def abbrev_to_bracket_id(abbrev):
    return ABBREV_TO_ID.get(abbrev, abbrev.lower())


def bracket_id_to_abbrev(bid):
    return ID_TO_ABBREV.get(bid, bid.upper())


# --- Scoring ---
SCORING = {"r64": 10, "r32": 20, "s16": 40, "e8": 80, "f4": 160, "champ": 320}
ROUND_ORDER = ["r64", "r32", "s16", "e8", "f4", "champ"]


def load_pool_data():
    with open(ROOT / "pool_hist_results.json") as f:
        return json.load(f)


def load_actual_results():
    """Build dict of {round: set(bracket_team_ids)} for teams that won in each round."""
    ctx_path = ROOT / "data/raw/historical/tournament_context_2026.json"
    data = None
    if ctx_path.exists():
        with open(ctx_path) as f:
            data = json.load(f).get("results")
    if data is None:
        with open(ROOT / "data/raw/historical/tournament_results_2026.json") as f:
            data = json.load(f)

    actual = {rd: set() for rd in ROUND_ORDER}

    # NCG label contains R64 (32), R32 (16), and Championship (1) = 49 games
    # S16/E8/F4 are labeled separately
    ncg_games = [g for g in data["games"] if g["round_name"] == "NCG"]

    # Split NCG into R64, R32, Championship by seed structure and region
    for g in ncg_games:
        s1, s2 = g["team1_seed"], g["team2_seed"]
        region = g.get("region", "")
        winner = normalize_results_id(g["team1_id"] if g["team1_won"] else g["team2_id"])

        if not region:
            # Championship game (no region)
            actual["champ"].add(winner)
        elif s1 + s2 == 17 or s1 == s2:
            # R64 game (seed sum = 17, or play-in seeds like 11v11, 16v16)
            actual["r64"].add(winner)
        else:
            # R32 game
            actual["r32"].add(winner)

    # S16, E8, F4 winners from their labeled rounds
    for g in data["games"]:
        rn = g["round_name"]
        winner = normalize_results_id(g["team1_id"] if g["team1_won"] else g["team2_id"])
        if rn == "S16":
            actual["s16"].add(winner)
        elif rn == "E8":
            actual["e8"].add(winner)
        elif rn == "F4":
            actual["f4"].add(winner)

    return actual


def load_espn_picks():
    with open(ROOT / "data/raw/historical_public_picks/espn_picks_2026.json") as f:
        data = json.load(f)
    return data["teams"]


def score_bracket(bracket, actual):
    """Score a single pool bracket against actual results."""
    total = 0
    for rd in ROUND_ORDER:
        picks = bracket.get(rd)
        if picks is None:
            continue
        if isinstance(picks, str):
            picks = [picks]
        pts = SCORING[rd]
        for pick in picks:
            bid = abbrev_to_bracket_id(pick)
            if bid in actual.get(rd, set()):
                total += pts
    return total


def pick_frequency_analysis(brackets):
    """Compute how often each team is picked per round across all brackets."""
    n = len(brackets)
    freq = {}
    for rd in ROUND_ORDER:
        counts = Counter()
        for b in brackets:
            picks = b.get(rd)
            if picks is None:
                continue
            if isinstance(picks, str):
                picks = [picks]
            for p in picks:
                counts[p] += 1
        freq[rd] = {team: cnt / n for team, cnt in counts.most_common()}
    return freq


def compute_correlation_matrix(brackets):
    """
    Compute inter-bracket pick correlation.

    Encode each bracket as a binary vector: for each (round, team) pair that
    anyone picked, 1 if this bracket picked it, 0 otherwise.
    Then compute pairwise correlations between brackets.
    """
    # Build feature space: all (round, team) pairs
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

    # Pairwise correlation between brackets
    # Center each bracket vector
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = centered / norms
    corr = normed @ normed.T

    return corr, matrix, features


def independence_test(brackets):
    """
    Test independence assumption: if picks were independent,
    the average pairwise bracket correlation would be near zero.
    Compare observed correlation to what independent draws would produce.
    """
    corr, matrix, features = compute_correlation_matrix(brackets)
    n = len(brackets)

    # Extract upper triangle (exclude diagonal)
    upper = corr[np.triu_indices(n, k=1)]

    # For each feature (round, team), compute pick rate
    pick_rates = matrix.mean(axis=0)

    # Under independence, expected correlation between two brackets
    # is driven by variance in pick rates (high-consensus picks create mechanical correlation)
    # Simulate: for each feature, draw Bernoulli(pick_rate) independently
    n_sim = 1000
    sim_corrs = []
    rng = np.random.default_rng(42)
    for _ in range(n_sim):
        sim_matrix = rng.binomial(1, pick_rates[np.newaxis, :].repeat(n, axis=0)).astype(float)
        sim_centered = sim_matrix - sim_matrix.mean(axis=1, keepdims=True)
        sim_norms = np.linalg.norm(sim_centered, axis=1, keepdims=True)
        sim_norms[sim_norms == 0] = 1
        sim_normed = sim_centered / sim_norms
        sim_corr = sim_normed @ sim_normed.T
        sim_upper = sim_corr[np.triu_indices(n, k=1)]
        sim_corrs.append(sim_upper.mean())

    sim_corrs = np.array(sim_corrs)

    return {
        "observed_mean_corr": float(upper.mean()),
        "observed_median_corr": float(np.median(upper)),
        "observed_std_corr": float(upper.std()),
        "observed_min_corr": float(upper.min()),
        "observed_max_corr": float(upper.max()),
        "simulated_mean_corr_mean": float(sim_corrs.mean()),
        "simulated_mean_corr_std": float(sim_corrs.std()),
        "excess_correlation": float(upper.mean() - sim_corrs.mean()),
        "z_score": float((upper.mean() - sim_corrs.mean()) / max(sim_corrs.std(), 1e-10)),
        "pct_pairs_above_sim_95th": float((upper > np.percentile(sim_corrs, 95)).mean() * 100),
    }


def per_game_correlation(brackets):
    """
    Measure pick correlation at the game level.
    For each matchup slot in later rounds, compute how correlated picks are.
    This is more interpretable than full-bracket correlation.
    """
    # Focus on rounds where there's actual choice (S16+)
    results = {}
    for rd in ["s16", "e8", "f4", "champ"]:
        # Get all unique teams picked in this round
        all_picks = []
        for b in brackets:
            picks = b.get(rd)
            if picks is None:
                continue
            if isinstance(picks, str):
                picks = [picks]
            all_picks.append(set(picks))

        if not all_picks:
            continue

        # For each team, binary vector of who picked them
        teams = set()
        for p in all_picks:
            teams.update(p)

        pick_vectors = {}
        for team in teams:
            pick_vectors[team] = np.array([1.0 if team in p else 0.0 for p in all_picks])

        # Pairwise correlations between team picks within the same round
        # High correlation = "if you pick team A in F4, you also pick team B"
        team_list = sorted(teams)
        n_teams = len(team_list)
        if n_teams < 2:
            continue

        pair_corrs = []
        for i in range(n_teams):
            for j in range(i + 1, n_teams):
                v1 = pick_vectors[team_list[i]]
                v2 = pick_vectors[team_list[j]]
                if v1.std() > 0 and v2.std() > 0:
                    c = np.corrcoef(v1, v2)[0, 1]
                    pair_corrs.append((team_list[i], team_list[j], c))

        # Sort by absolute correlation
        pair_corrs.sort(key=lambda x: abs(x[2]), reverse=True)
        results[rd] = {
            "n_teams_picked": n_teams,
            "top_correlations": pair_corrs[:10],
            "mean_abs_corr": float(np.mean([abs(c) for _, _, c in pair_corrs])) if pair_corrs else 0,
        }

    return results


def main():
    pool = load_pool_data()
    actual = load_actual_results()
    espn = load_espn_picks()

    yr_data = pool["years"]["2026"]
    brackets = yr_data["brackets"]
    n = len(brackets)

    print("=" * 70)
    print(f"POOL CORRELATION ANALYSIS — 2026")
    print(f"Pool: {pool['pool']}, {yr_data['groupSize']} entrants, {n} brackets collected")
    print(f"Champion: {yr_data['champion']}")
    print("=" * 70)

    # --- 1. Score all brackets ---
    print("\n" + "=" * 70)
    print("SECTION 1: BRACKET SCORES AGAINST ACTUAL RESULTS")
    print("=" * 70)

    scored = []
    for b in brackets:
        s = score_bracket(b, actual)
        scored.append((s, b))
    scored.sort(key=lambda x: -x[0])

    # Verify against reported scores
    print(f"\n{'Rank':>4} {'Computed':>8} {'Reported':>8} {'Match':>5}  Champion Pick")
    print("-" * 55)
    for i, (s, b) in enumerate(scored[:15]):
        reported = b.get("pts", "?")
        match = "OK" if reported == s else f"!={reported}"
        print(f"{i + 1:>4} {s:>8} {reported:>8} {match:>5}  {b['champ']}")

    # Show bottom 5
    print("  ...")
    for i, (s, b) in enumerate(scored[-5:]):
        rank = n - 4 + i
        print(f"{rank:>4} {s:>8} {b.get('pts', '?'):>8}       {b['champ']}")

    # --- 2. What did the winning bracket pick? ---
    print("\n" + "=" * 70)
    print("SECTION 2: WINNING BRACKET ANATOMY")
    print("=" * 70)

    winner_score, winner = scored[0]
    print(f"\nWinner: {winner_score} pts, champion pick: {winner['champ']}")
    for rd in ROUND_ORDER:
        picks = winner.get(rd)
        if isinstance(picks, str):
            picks = [picks]
        if picks:
            # Mark correct picks
            marked = []
            for p in picks:
                bid = abbrev_to_bracket_id(p)
                correct = bid in actual.get(rd, set())
                marked.append(f"{p}{'*' if correct else ''}")
            correct_count = sum(1 for p in picks if abbrev_to_bracket_id(p) in actual.get(rd, set()))
            print(f"  {rd:>5}: {correct_count}/{len(picks)} correct — {', '.join(marked)}")

    # --- 3. Pick frequency analysis ---
    print("\n" + "=" * 70)
    print("SECTION 3: PICK FREQUENCIES — POOL vs ESPN")
    print("=" * 70)

    freq = pick_frequency_analysis(brackets)

    # Map ESPN team IDs to pool abbrevs for comparison
    espn_by_abbrev = {}
    for team_id, picks in espn.items():
        # Find matching abbrev
        for abbrev, bid in ABBREV_TO_ID.items():
            if bid == team_id or team_id.startswith(bid):
                espn_by_abbrev[abbrev] = picks
                break

    for rd in ["s16", "e8", "f4", "champ"]:
        espn_rd = rd.upper() if rd != "champ" else "CHAMP"
        print(f"\n  {rd.upper()} picks (pool_pct vs ESPN_pct):")
        rd_freq = freq[rd]
        for team, pool_pct in sorted(rd_freq.items(), key=lambda x: -x[1]):
            espn_pct = "?"
            if team in espn_by_abbrev and espn_rd in espn_by_abbrev[team]:
                espn_pct = f"{espn_by_abbrev[team][espn_rd] * 100:.1f}%"
            bid = abbrev_to_bracket_id(team)
            actual_flag = " <<< ACTUAL" if bid in actual.get(rd, set()) else ""
            print(f"    {team:>5}: pool={pool_pct * 100:5.1f}%  ESPN={espn_pct:>6s}{actual_flag}")

    # --- 4. Correlation analysis ---
    print("\n" + "=" * 70)
    print("SECTION 4: INTER-BRACKET CORRELATION (Independence Test)")
    print("=" * 70)

    indep = independence_test(brackets)
    print(f"\n  Observed mean pairwise correlation:  {indep['observed_mean_corr']:.4f}")
    print(f"  Observed median pairwise correlation: {indep['observed_median_corr']:.4f}")
    print(f"  Observed std:                         {indep['observed_std_corr']:.4f}")
    print(
        f"  Observed range:                       [{indep['observed_min_corr']:.4f}, {indep['observed_max_corr']:.4f}]"
    )
    print(
        f"  Simulated (independent) mean:         {indep['simulated_mean_corr_mean']:.4f} ± {indep['simulated_mean_corr_std']:.4f}"
    )
    print(f"  Excess correlation:                   {indep['excess_correlation']:.4f}")
    print(f"  Z-score vs independence:              {indep['z_score']:.1f}")
    print(f"  % pairs above simulated 95th pctl:    {indep['pct_pairs_above_sim_95th']:.1f}%")

    if indep["z_score"] > 2:
        print("\n  >>> INDEPENDENCE ASSUMPTION REJECTED: Pool brackets are significantly")
        print("      more correlated than independent random draws would produce.")
    elif indep["z_score"] > 1:
        print("\n  >>> WEAK EVIDENCE of excess correlation (marginal)")
    else:
        print("\n  >>> Independence assumption holds (no significant excess correlation)")

    # --- 5. Per-game correlation ---
    print("\n" + "=" * 70)
    print("SECTION 5: PICK CLUSTERING BY ROUND")
    print("=" * 70)

    game_corr = per_game_correlation(brackets)
    for rd, info in game_corr.items():
        print(
            f"\n  {rd.upper()}: {info['n_teams_picked']} unique teams picked, mean |corr| = {info['mean_abs_corr']:.3f}"
        )
        if info["top_correlations"]:
            print(f"    Top correlated pick pairs:")
            for t1, t2, c in info["top_correlations"][:5]:
                direction = "+" if c > 0 else "-"
                print(f"      {t1:>5} & {t2:<5}: r={c:+.3f}  ({'co-picked' if c > 0 else 'anti-picked'})")

    # --- 6. Champion clustering ---
    print("\n" + "=" * 70)
    print("SECTION 6: CHAMPION PICK CONCENTRATION")
    print("=" * 70)

    champ_picks = Counter(b["champ"] for b in brackets)
    print(f"\n  Champion picks ({n} brackets):")
    for team, cnt in champ_picks.most_common():
        pct = cnt / n * 100
        espn_champ = "?"
        if team in espn_by_abbrev and "CHAMP" in espn_by_abbrev[team]:
            espn_champ = f"{espn_by_abbrev[team]['CHAMP'] * 100:.1f}%"
        actual_flag = " <<< ACTUAL CHAMP" if abbrev_to_bracket_id(team) in actual.get("champ", set()) else ""
        print(f"    {team:>5}: {cnt:2d} ({pct:4.1f}%)  ESPN={espn_champ:>6s}{actual_flag}")

    hhi = sum((cnt / n) ** 2 for cnt in champ_picks.values())
    equiv_teams = 1 / hhi
    print(f"\n  HHI (concentration): {hhi:.3f}")
    print(f"  Equivalent # of equally-picked teams: {equiv_teams:.1f}")
    print(f"  (Lower equiv = more concentrated, more herding)")

    # --- 7. Summary stats ---
    print("\n" + "=" * 70)
    print("SECTION 7: KEY FINDINGS FOR OPPONENT MODEL")
    print("=" * 70)

    # What fraction picked the actual champion?
    actual_champ_abbrev = bracket_id_to_abbrev(list(actual["champ"])[0])
    champ_pickers = champ_picks.get(actual_champ_abbrev, 0)
    print(f"\n  Actual champion: {actual_champ_abbrev}")
    print(f"  Pool members who picked champion: {champ_pickers}/{n} ({champ_pickers / n * 100:.1f}%)")

    # How many picked most popular champion?
    top_champ, top_cnt = champ_picks.most_common(1)[0]
    print(f"  Most popular champion pick: {top_champ} ({top_cnt}/{n}, {top_cnt / n * 100:.1f}%)")

    # Chalk score: what fraction of R64 picks are seeds 1-4?
    chalk_scores = []
    for b in brackets:
        # We don't have seed info in pool data directly, but we can approximate
        # by checking how many picks match top ESPN favorites
        pass

    print(f"\n  Correlation summary:")
    print(
        f"    Your pool's brackets are {'MORE' if indep['excess_correlation'] > 0 else 'LESS'} "
        f"correlated than independent draws"
    )
    print(f"    Excess correlation = {indep['excess_correlation']:.4f} (z={indep['z_score']:.1f})")
    if indep["z_score"] > 2:
        print(f"    >>> The independence assumption in your opponent model is WRONG for this pool")
        print(f"    >>> Opponent brackets should be generated with correlation structure")
        print(f"    >>> Suggested chalk_noise_std: {max(0.3, min(0.6, indep['excess_correlation'] * 5)):.2f}")


if __name__ == "__main__":
    main()
