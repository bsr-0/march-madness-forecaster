#!/usr/bin/env python3
"""
Conference Tournament Prediction Pipeline — March 2026
=======================================================
Scrapes fresh Torvik data, runs predictions + Monte Carlo sims,
and outputs JSON results + interactive HTML dashboard.
"""

import json
import logging
import math
import shutil
import sys
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_RAW = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# 1.  DATA ACQUISITION — scrape fresh 2026 Four Factors & Shooting
# ──────────────────────────────────────────────────────────────────────

def scrape_fresh_data():
    """Scrape fresh 2026 Torvik Four Factors + Shooting via BartTorvikScraper.

    The scraper tries HTML first, then auto-falls back to the CSV
    player-stats endpoint (``getadvstats.php``) when the JS wall blocks
    the HTML page.  We always re-scrape to ensure we have the latest
    data — no "file exists" short-circuit.
    """
    from src.data.scrapers.torvik import BartTorvikScraper

    # Do NOT pass cache_dir so the scraper always hits the live endpoint.
    scraper = BartTorvikScraper(cache_dir=None)

    # --- Four Factors 2026 ---
    logger.info("Scraping 2026 Four Factors from barttorvik.com ...")
    try:
        ff = scraper.fetch_four_factors(year=2026)
        if ff and len(ff) > 50:
            ff_path = DATA_RAW / "torvik_four_factors_2026.json"
            with open(ff_path, "w") as f:
                json.dump(ff, f, indent=2)
            logger.info("  Saved %d teams to %s", len(ff), ff_path.name)
        else:
            logger.warning("  Scrape returned %d teams — pipeline will use fallback files",
                           len(ff) if ff else 0)
    except Exception as e:
        logger.warning("  Four Factors scrape failed (%s) — pipeline will use fallback files", e)

    # --- Shooting 2026 ---
    logger.info("Scraping 2026 Shooting stats from barttorvik.com ...")
    try:
        sh = scraper.fetch_shooting_stats(year=2026)
        if sh and len(sh) > 50:
            sh_path = DATA_RAW / "torvik_shooting_2026.json"
            with open(sh_path, "w") as f:
                json.dump(sh, f, indent=2)
            logger.info("  Saved %d teams to %s", len(sh), sh_path.name)
        else:
            logger.warning("  Scrape returned %d teams — pipeline will use fallback files",
                           len(sh) if sh else 0)
    except Exception as e:
        logger.warning("  Shooting scrape failed (%s) — pipeline will use fallback files", e)


# ──────────────────────────────────────────────────────────────────────
# 2.  ENHANCED LOGISTIC MODEL (better than base standalone)
# ──────────────────────────────────────────────────────────────────────

class EnhancedPredictor:
    """
    Conference tournament predictor that augments the base logistic model
    with home-court neutralization, conference-familiarity adjustments,
    and momentum features.
    """

    def __init__(self, torvik_path: str, data_dir: str = None, year: int = 2026):
        with open(torvik_path) as f:
            self.raw_data = json.load(f)

        self.data_dir = data_dir or str(Path(torvik_path).parent)
        self.year = year
        self.teams_by_id = {}  # team_id -> team_dict (enriched)
        self.teams_by_conf = defaultdict(list)

        # Load supplementary data
        self.four_factors = self._load_json("torvik_four_factors", year)
        self.shooting = self._load_json("torvik_shooting", year)
        self.conf_records = {}  # team_id -> (conf_wins, conf_losses)

        # Build team index
        self._build_teams()

    def _load_json(self, prefix: str, year: int) -> Optional[dict]:
        for y in range(year, max(year - 3, 2004), -1):
            p = Path(self.data_dir) / f"{prefix}_{y}.json"
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                if y != year:
                    logger.info("Using %s_%d.json as fallback for %d", prefix, y, year)
                return data
        return None

    def _build_teams(self):
        try:
            from src.data.normalize import normalize_team_id as _norm
        except ImportError:
            _norm = None

        for t in self.raw_data.get("teams", []):
            tid = t.get("team_id", "")
            if _norm:
                tid = _norm(tid)
            conf = t.get("conference", "")
            if not tid or not conf:
                continue

            team = {
                "team_id": tid,
                "name": t.get("team_name", "") or t.get("name", ""),
                "conference": conf,
                "t_rank": t.get("t_rank", 999),
                "adj_o": t.get("adj_offensive_efficiency", 100.0),
                "adj_d": t.get("adj_defensive_efficiency", 100.0),
                "adj_em": t.get("adj_offensive_efficiency", 100.0) - t.get("adj_defensive_efficiency", 100.0),
                "tempo": t.get("adj_tempo", 68.0),
                "barthag": t.get("barthag", 0.5),
            }

            # Enrich with Four Factors
            if self.four_factors and tid in self.four_factors:
                ff = self.four_factors[tid]
                team.update({
                    "efg": ff.get("effective_fg_pct", 0.0),
                    "to_rate": ff.get("turnover_rate", 0.0),
                    "orb_rate": ff.get("offensive_reb_rate", 0.0),
                    "ft_rate": ff.get("free_throw_rate", 0.0),
                    "opp_efg": ff.get("opp_effective_fg_pct", 0.0),
                    "opp_to_rate": ff.get("opp_turnover_rate", 0.0),
                    "drb_rate": ff.get("defensive_reb_rate", 0.0),
                    "opp_ft_rate": ff.get("opp_free_throw_rate", 0.0),
                })
            else:
                team.update({
                    "efg": 0.0, "to_rate": 0.0, "orb_rate": 0.0, "ft_rate": 0.0,
                    "opp_efg": 0.0, "opp_to_rate": 0.0, "drb_rate": 0.0, "opp_ft_rate": 0.0,
                })

            # Enrich with Shooting
            if self.shooting and tid in self.shooting:
                sh = self.shooting[tid]
                team["three_pt_pct"] = sh.get("three_pt_pct", 0.0)
                team["ft_pct"] = sh.get("ft_pct", 0.0)
            else:
                team["three_pt_pct"] = 0.0
                team["ft_pct"] = 0.0

            self.teams_by_id[tid] = team
            self.teams_by_conf[conf].append(team)

        # Load seed overrides if available
        seed_overrides = {}
        seed_path = Path(self.data_dir) / f"seed_overrides_{self.year}.json"
        if seed_path.exists():
            try:
                with open(seed_path) as f:
                    raw_overrides = json.load(f)
                from src.data.normalize import normalize_team_id
                for conf_key, seeds in raw_overrides.items():
                    seed_overrides[conf_key] = {
                        normalize_team_id(k): v for k, v in seeds.items()
                    }
                logger.info("Loaded seed overrides from %s", seed_path)
            except Exception as e:
                logger.warning("Failed to load seed overrides: %s", e)

        # Sort each conference by seed override (if available) or AdjEM (desc)
        for conf in self.teams_by_conf:
            if conf in seed_overrides:
                overrides = seed_overrides[conf]
                # Apply override seeds, fall back to AdjEM for unmatched teams
                assigned = set()
                for t in self.teams_by_conf[conf]:
                    if t["team_id"] in overrides:
                        t["conf_seed"] = overrides[t["team_id"]]
                        assigned.add(t["conf_seed"])
                unassigned = [t for t in self.teams_by_conf[conf] if t["team_id"] not in overrides]
                unassigned.sort(key=lambda t: -t["adj_em"])
                next_seed = 1
                for t in unassigned:
                    while next_seed in assigned:
                        next_seed += 1
                    t["conf_seed"] = next_seed
                    assigned.add(next_seed)
                    next_seed += 1
                self.teams_by_conf[conf].sort(key=lambda t: t["conf_seed"])
            else:
                self.teams_by_conf[conf].sort(key=lambda t: -t["adj_em"])
                for i, t in enumerate(self.teams_by_conf[conf]):
                    t["conf_seed"] = i + 1

    def predict_matchup(self, t1: dict, t2: dict) -> float:
        """P(team1 wins) using multi-feature logistic model."""
        # Core: AdjEM difference is the main signal
        em_diff = t1["adj_em"] - t2["adj_em"]
        logit = 0.075 * (t1["adj_o"] - t2["adj_o"]) + (-0.075) * (t1["adj_d"] - t2["adj_d"])

        # Four Factors (only when data is available)
        if t1.get("efg", 0) > 0 and t2.get("efg", 0) > 0:
            logit += 1.5 * (t1["efg"] - t2["efg"])
            logit += -1.0 * (t1["to_rate"] - t2["to_rate"])
            logit += 0.8 * (t1["orb_rate"] - t2["orb_rate"])
            logit += 0.5 * (t1["ft_rate"] - t2["ft_rate"])

            # Defensive Four Factors
            logit += -1.2 * (t1["opp_efg"] - t2["opp_efg"])
            logit += 0.7 * (t1["opp_to_rate"] - t2["opp_to_rate"])
            logit += 0.4 * (t1["drb_rate"] - t2["drb_rate"])
            logit += -0.3 * (t1["opp_ft_rate"] - t2["opp_ft_rate"])

        # Shooting (3PT variance matters in single-elim)
        if t1.get("three_pt_pct", 0) > 0 and t2.get("three_pt_pct", 0) > 0:
            logit += 0.6 * (t1["three_pt_pct"] - t2["three_pt_pct"])

        if t1.get("ft_pct", 0) > 0 and t2.get("ft_pct", 0) > 0:
            logit += 0.4 * (t1["ft_pct"] - t2["ft_pct"])

        # Conference tournament seed advantage (slight, for tiebreakers/familiarity)
        seed_diff = t2["conf_seed"] - t1["conf_seed"]
        logit += 0.02 * seed_diff  # very small — mostly cosmetic

        prob = 1.0 / (1.0 + math.exp(-logit))
        return max(0.02, min(0.98, prob))


# ──────────────────────────────────────────────────────────────────────
# 3.  BRACKET SIMULATION
# ──────────────────────────────────────────────────────────────────────

# Conference tournament sizes (2025-26 formats)
CONF_TOURNAMENT_SIZES = {
    "ACC": 15, "B10": 18, "B12": 16, "SEC": 16, "BE": 11,
    "Amer": 10, "MWC": 12, "WCC": 12, "A10": 14, "MVC": 11,
    "CAA": 13, "AE": 8, "ASun": 12, "BSky": 10, "BSth": 9,
    "BW": 8, "CUSA": 10, "Horz": 11, "Ivy": 4, "MAAC": 10,
    "MAC": 8, "MEAC": 8, "NEC": 8, "OVC": 8, "Pat": 10,
    "SB": 14, "SC": 10, "SWAC": 12, "Slnd": 8, "Sum": 8, "WAC": 7,
}

CONF_FULL_NAMES = {
    "A10": "Atlantic 10", "ACC": "ACC", "AE": "America East",
    "ASun": "ASUN", "Amer": "American Athletic", "B10": "Big Ten",
    "B12": "Big 12", "BE": "Big East", "BSky": "Big Sky",
    "BSth": "Big South", "BW": "Big West", "CAA": "CAA",
    "CUSA": "Conference USA", "Horz": "Horizon League",
    "Ivy": "Ivy League", "MAAC": "MAAC", "MAC": "MAC",
    "MEAC": "MEAC", "MVC": "Missouri Valley", "MWC": "Mountain West",
    "NEC": "NEC", "OVC": "Ohio Valley", "Pat": "Patriot League",
    "SB": "Sun Belt", "SC": "Southern", "SEC": "SEC",
    "SWAC": "SWAC", "Slnd": "Southland", "Sum": "Summit League",
    "WAC": "WAC", "WCC": "WCC",
}


def simulate_bracket(predictor: EnhancedPredictor, conf: str,
                     num_sims: int = 10000, noise_std: float = 0.16,
                     seed: int = 2026) -> dict:
    """
    Simulate a conference tournament bracket via Monte Carlo.

    Returns dict with:
      - deterministic_bracket: single best-guess bracket
      - championship_probs: team_id -> P(champion)
      - semifinal_probs: team_id -> P(reach semis)
      - expected_upsets: notable upset probabilities
      - all_rounds: round-by-round matchups (deterministic)
    """
    rng = np.random.default_rng(seed)

    all_teams = predictor.teams_by_conf.get(conf, [])
    if not all_teams:
        return None

    # Determine tournament size
    n_teams = min(CONF_TOURNAMENT_SIZES.get(conf, len(all_teams)), len(all_teams))
    teams = all_teams[:n_teams]

    # --- Single deterministic bracket ---
    det_bracket = _run_bracket(predictor, teams, deterministic=True)

    # --- Monte Carlo simulations ---
    champ_counts = Counter()
    semi_counts = Counter()

    # Pre-compute base probs for all pairs
    base_probs = {}
    for i, t1 in enumerate(teams):
        for j, t2 in enumerate(teams):
            if i != j:
                key = (t1["team_id"], t2["team_id"])
                if key not in base_probs:
                    p = predictor.predict_matchup(t1, t2)
                    base_probs[key] = p
                    base_probs[(t2["team_id"], t1["team_id"])] = 1.0 - p

    # Track convergence: snapshot championship probs at intervals
    # Use ~40 sample points (logarithmically spaced early, linear later)
    snapshot_points = sorted(set(
        [int(x) for x in np.geomspace(10, min(500, num_sims), 15)] +
        list(range(500, num_sims + 1, max(1, num_sims // 25)))
    ))
    # Ensure final point is included
    if num_sims not in snapshot_points:
        snapshot_points.append(num_sims)
    snapshot_set = set(snapshot_points)

    convergence = {t["team_id"]: [] for t in teams}
    convergence_x = []  # simulation counts where snapshots taken

    for sim_i in range(1, num_sims + 1):
        result = _run_bracket(predictor, teams, deterministic=False,
                              rng=rng, noise_std=noise_std, base_probs=base_probs)
        if result["champion"]:
            champ_counts[result["champion"]["team_id"]] += 1
        for t in result.get("semifinalists", []):
            semi_counts[t["team_id"]] += 1

        # Record convergence snapshot
        if sim_i in snapshot_set:
            convergence_x.append(sim_i)
            for t in teams:
                convergence[t["team_id"]].append(
                    round(champ_counts.get(t["team_id"], 0) / sim_i, 4)
                )

    # Compute probabilities
    champ_probs = {t["team_id"]: champ_counts.get(t["team_id"], 0) / num_sims for t in teams}
    semi_probs = {t["team_id"]: semi_counts.get(t["team_id"], 0) / num_sims for t in teams}

    # Sort champ probs descending
    sorted_champ = sorted(champ_probs.items(), key=lambda x: -x[1])

    # Only keep convergence data for top 6 teams (to keep JSON small)
    top_team_ids = [tid for tid, _ in sorted_champ[:6]]
    convergence_data = {
        "x": convergence_x,
        "teams": {tid: convergence[tid] for tid in top_team_ids},
        "team_names": {tid: predictor.teams_by_id[tid]["name"] for tid in top_team_ids},
    }

    return {
        "conference": conf,
        "conference_name": CONF_FULL_NAMES.get(conf, conf),
        "num_teams": n_teams,
        "deterministic_bracket": det_bracket,
        "championship_probs": dict(sorted_champ),
        "semifinal_probs": semi_probs,
        "predicted_champion": det_bracket["champion"],
        "mc_favorite": {
            "team_id": sorted_champ[0][0],
            "name": predictor.teams_by_id[sorted_champ[0][0]]["name"],
            "prob": sorted_champ[0][1],
        },
        "convergence": convergence_data,
    }


def _run_bracket(predictor, teams, deterministic=True,
                 rng=None, noise_std=0.16, base_probs=None,
                 bracket_format=None):
    """Run a single bracket simulation using BracketFormat definitions.

    Uses the BracketFormat to determine which seeds enter at each round,
    supporting multi-level byes (e.g. ACC 15-team format with double byes
    for top 4 seeds).
    """
    from src.conference_tournament.bracket_formats import get_bracket_format

    n = len(teams)

    # Look up the correct bracket format
    conf = teams[0]["conference"] if teams else ""
    fmt = bracket_format or get_bracket_format(conf, n)

    total_rounds = fmt.total_rounds
    seed_to_team = {t["conf_seed"]: t for t in teams}

    all_rounds = []
    round_results = []

    # ── Round 1: pair first-round teams ──────────────────────────────
    r1_seeds = list(fmt.round_entry.get(1, ()))
    r1_teams = [seed_to_team[s] for s in r1_seeds if s in seed_to_team]
    r1_teams.sort(key=lambda t: t["conf_seed"])

    r1_games = []

    if fmt.first_round_matchups:
        # Explicit matchups
        ordered_r1 = []
        for seed_a, seed_b in fmt.first_round_matchups:
            if seed_a in seed_to_team and seed_b in seed_to_team:
                ordered_r1.append(seed_to_team[seed_a])
                ordered_r1.append(seed_to_team[seed_b])
    else:
        n_games = len(r1_teams) // 2
        has_byes = any(rnd > 1 and seeds for rnd, seeds in fmt.round_entry.items())

        if not has_byes and len(r1_teams) >= 4:
            from src.models.conference_tournament import bracket_seed_order
            positions = bracket_seed_order(len(r1_teams))
            ordered_r1 = [r1_teams[p] for p in positions]
        else:
            ordered_r1 = []
            for i in range(n_games):
                ordered_r1.append(r1_teams[i])
                ordered_r1.append(r1_teams[len(r1_teams) - 1 - i])

    n_games = len(ordered_r1) // 2
    for i in range(n_games):
        t1 = ordered_r1[2 * i]
        t2 = ordered_r1[2 * i + 1]
        winner, prob = _play_game(predictor, t1, t2, deterministic, rng, noise_std, base_probs)
        r1_games.append({
            "team1": t1, "team2": t2, "winner": winner,
            "win_prob": prob,
            "is_upset": winner["conf_seed"] > min(t1["conf_seed"], t2["conf_seed"]),
        })
        round_results.append(winner)

    all_rounds.append({"round_name": _round_name(1, total_rounds), "games": r1_games})

    # ── Rounds 2+ ────────────────────────────────────────────────────
    semifinalists = []
    for round_num in range(2, total_rounds + 1):
        # Check if new seeds enter at this round
        new_seeds = fmt.round_entry.get(round_num, ())
        new_teams = [seed_to_team[s] for s in new_seeds if s in seed_to_team]

        if new_teams:
            entrants = list(new_teams) + list(round_results)
            entrants.sort(key=lambda t: t["conf_seed"])
            from src.models.conference_tournament import bracket_seed_order
            positions = bracket_seed_order(len(entrants))
            entrants = [entrants[p] for p in positions]
        else:
            entrants = list(round_results)

        # Track semifinalists
        if round_num == total_rounds - 1 or (total_rounds <= 2 and round_num == total_rounds):
            semifinalists = list(entrants)

        round_games = []
        round_results = []
        for i in range(len(entrants) // 2):
            t1 = entrants[2 * i]
            t2 = entrants[2 * i + 1]
            winner, prob = _play_game(predictor, t1, t2, deterministic, rng, noise_std, base_probs)
            round_games.append({
                "team1": t1, "team2": t2, "winner": winner,
                "win_prob": prob,
                "is_upset": winner["conf_seed"] > min(t1["conf_seed"], t2["conf_seed"]),
            })
            round_results.append(winner)

        if len(entrants) % 2 == 1:
            round_results.append(entrants[-1])

        all_rounds.append({"round_name": _round_name(round_num, total_rounds), "games": round_games})

    champion = round_results[0] if round_results else None

    return {
        "champion": champion,
        "semifinalists": semifinalists,
        "rounds": all_rounds,
    }


def _play_game(predictor, t1, t2, deterministic, rng, noise_std, base_probs):
    """Simulate a single game."""
    if base_probs and (t1["team_id"], t2["team_id"]) in base_probs:
        base_p = base_probs[(t1["team_id"], t2["team_id"])]
    else:
        base_p = predictor.predict_matchup(t1, t2)

    if deterministic:
        prob = base_p
    else:
        logit = math.log(max(base_p, 1e-6) / max(1.0 - base_p, 1e-6))
        noisy_logit = logit + rng.normal(0, noise_std)
        prob = 1.0 / (1.0 + math.exp(-noisy_logit))
        prob = max(0.01, min(0.99, prob))

    if deterministic:
        winner = t1 if prob >= 0.5 else t2
        return winner, prob if winner == t1 else 1.0 - prob
    else:
        if rng.random() < prob:
            return t1, prob
        else:
            return t2, 1.0 - prob


def _round_name(round_num, total_rounds):
    names = {
        1: {1: "Championship"},
        2: {1: "Semifinals", 2: "Championship"},
        3: {1: "Quarterfinals", 2: "Semifinals", 3: "Championship"},
        4: {1: "First Round", 2: "Quarterfinals", 3: "Semifinals", 4: "Championship"},
        5: {1: "Play-in", 2: "First Round", 3: "Quarterfinals", 4: "Semifinals", 5: "Championship"},
    }
    return names.get(total_rounds, {}).get(round_num, f"Round {round_num}")


# ──────────────────────────────────────────────────────────────────────
# 3b.  OUTPUT VALIDATION
# ──────────────────────────────────────────────────────────────────────

def _validate_output(json_path: Path) -> List[str]:
    """Validate generated JSON output against bracket format definitions.

    Reads back the output file and checks every conference for:
    - Correct number of rounds per BracketFormat
    - R1 matchup seeds match first_round_matchups definition
    - Total games == num_teams - 1

    Returns list of error strings (empty = all valid).
    """
    from src.conference_tournament.bracket_formats import get_bracket_format

    with open(json_path) as f:
        output = json.load(f)

    errors = []
    for conf_key, conf_data in output.items():
        rounds = conf_data["rounds"]
        num_teams = conf_data["num_teams"]
        fmt = get_bracket_format(conf_key, num_teams)

        # Check round count matches format
        if len(rounds) != fmt.total_rounds:
            errors.append(
                f"{conf_key}: expected {fmt.total_rounds} rounds, got {len(rounds)}"
            )

        # Check R1 matchups match format definition
        if fmt.first_round_matchups and rounds:
            r1_games = rounds[0]["games"]
            expected_matchups = set()
            for a, b in fmt.first_round_matchups:
                expected_matchups.add((min(a, b), max(a, b)))

            actual_matchups = set()
            for g in r1_games:
                # Extract seeds from "(10) Stanford" format
                s1 = int(g["team1"].split(")")[0].strip("( "))
                s2 = int(g["team2"].split(")")[0].strip("( "))
                actual_matchups.add((min(s1, s2), max(s1, s2)))

            if actual_matchups != expected_matchups:
                errors.append(
                    f"{conf_key}: R1 matchups {sorted(actual_matchups)} "
                    f"!= expected {sorted(expected_matchups)}"
                )

        # Check total games == num_teams - 1
        total_games = sum(len(r["games"]) for r in rounds)
        expected_games = num_teams - 1
        if total_games != expected_games:
            errors.append(
                f"{conf_key}: {total_games} total games != expected {expected_games}"
            )

    return errors


# ──────────────────────────────────────────────────────────────────────
# 4.  GENERATE HTML DASHBOARD
# ──────────────────────────────────────────────────────────────────────

def generate_dashboard(all_results: dict, output_path: str):
    """Generate interactive HTML dashboard."""

    # Build conference cards data
    power_confs = ["ACC", "B10", "B12", "SEC", "BE"]
    mid_major_confs = ["Amer", "MWC", "WCC", "A10", "MVC", "CAA"]
    small_confs = [c for c in sorted(all_results.keys()) if c not in power_confs and c not in mid_major_confs]

    # Build JSON data for the dashboard
    dashboard_data = {}
    for conf, result in all_results.items():
        if result is None:
            continue

        det = result["deterministic_bracket"]
        champ = det["champion"]

        # Top 5 championship contenders
        top_contenders = []
        for tid, prob in list(result["championship_probs"].items())[:8]:
            t = result["deterministic_bracket"]["rounds"][0]["games"][0]["team1"]  # fallback
            for conf_teams in [g["team1"] for rnd in det["rounds"] for g in rnd["games"]] + \
                              [g["team2"] for rnd in det["rounds"] for g in rnd["games"]]:
                if conf_teams["team_id"] == tid:
                    t = conf_teams
                    break
            top_contenders.append({
                "team_id": tid,
                "name": t.get("name", tid),
                "seed": t.get("conf_seed", 99),
                "t_rank": t.get("t_rank", 999),
                "adj_em": round(t.get("adj_em", 0), 1),
                "champ_prob": round(prob * 100, 1),
            })

        # Count upsets in deterministic bracket
        upset_count = sum(
            1 for rnd in det["rounds"] for g in rnd["games"] if g.get("is_upset")
        )

        # Build round-by-round for detail view
        rounds_detail = []
        for rnd in det["rounds"]:
            games_detail = []
            for g in rnd["games"]:
                games_detail.append({
                    "team1_name": g["team1"]["name"],
                    "team1_seed": g["team1"]["conf_seed"],
                    "team2_name": g["team2"]["name"],
                    "team2_seed": g["team2"]["conf_seed"],
                    "winner_name": g["winner"]["name"],
                    "winner_seed": g["winner"]["conf_seed"],
                    "win_prob": round(g["win_prob"] * 100, 1),
                    "is_upset": g.get("is_upset", False),
                })
            rounds_detail.append({
                "round_name": rnd["round_name"],
                "games": games_detail,
            })

        dashboard_data[conf] = {
            "conference": conf,
            "conference_name": CONF_FULL_NAMES.get(conf, conf),
            "num_teams": result["num_teams"],
            "champion_name": champ["name"] if champ else "TBD",
            "champion_seed": champ["conf_seed"] if champ else 0,
            "champion_rank": champ["t_rank"] if champ else 999,
            "mc_favorite_name": result["mc_favorite"]["name"],
            "mc_favorite_prob": round(result["mc_favorite"]["prob"] * 100, 1),
            "upset_count": upset_count,
            "top_contenders": top_contenders,
            "rounds": rounds_detail,
        }

    # Generate HTML
    html = _build_html(dashboard_data, power_confs, mid_major_confs, small_confs)

    with open(output_path, "w") as f:
        f.write(html)
    logger.info("Dashboard saved to %s", output_path)


def _build_html(data, power_confs, mid_major_confs, small_confs):
    """Build the complete HTML dashboard."""

    # Build conference cards
    def conf_card(conf):
        d = data.get(conf)
        if not d:
            return ""

        contender_rows = ""
        for c in d["top_contenders"][:6]:
            bar_width = min(c["champ_prob"] * 3, 100)
            contender_rows += f"""
            <div class="contender-row">
                <span class="seed-badge">#{c['seed']}</span>
                <span class="team-name">{c['name']}</span>
                <div class="prob-bar-container">
                    <div class="prob-bar" style="width: {bar_width}%"></div>
                </div>
                <span class="prob-pct">{c['champ_prob']}%</span>
            </div>"""

        # Build bracket detail (hidden by default)
        bracket_html = ""
        for rnd in d["rounds"]:
            games_html = ""
            for g in rnd["games"]:
                upset_class = " upset" if g["is_upset"] else ""
                winner_indicator = lambda name, is_winner: f'<strong class="winner">{name}</strong>' if is_winner else name
                t1_display = winner_indicator(f"({g['team1_seed']}) {g['team1_name']}", g['winner_name'] == g['team1_name'])
                t2_display = winner_indicator(f"({g['team2_seed']}) {g['team2_name']}", g['winner_name'] == g['team2_name'])

                games_html += f"""
                <div class="bracket-game{upset_class}">
                    <div class="matchup-line">{t1_display}</div>
                    <div class="matchup-line">{t2_display}</div>
                    <div class="game-prob">{g['win_prob']}%</div>
                </div>"""

            bracket_html += f"""
            <div class="bracket-round">
                <div class="round-header">{rnd['round_name']}</div>
                {games_html}
            </div>"""

        upset_badge = f'<span class="upset-badge">{d["upset_count"]} upset{"s" if d["upset_count"] != 1 else ""}</span>' if d["upset_count"] > 0 else ''

        return f"""
        <div class="conf-card" data-conf="{conf}">
            <div class="card-header" onclick="toggleBracket('{conf}')">
                <div class="conf-title">
                    <h3>{d['conference_name']}</h3>
                    <span class="team-count">{d['num_teams']} teams</span>
                    {upset_badge}
                </div>
                <div class="champion-prediction">
                    <div class="champ-label">Predicted Champion</div>
                    <div class="champ-name">{d['champion_name']}</div>
                    <div class="champ-detail">Seed #{d['champion_seed']} | T-Rank #{d['champion_rank']}</div>
                    <div class="mc-prob">MC: {d['mc_favorite_name']} ({d['mc_favorite_prob']}%)</div>
                </div>
            </div>
            <div class="contenders-section">
                <div class="section-label">Championship Probability</div>
                {contender_rows}
            </div>
            <div class="bracket-detail" id="bracket-{conf}" style="display:none;">
                <div class="bracket-container">
                    {bracket_html}
                </div>
            </div>
        </div>"""

    # Build all sections
    power_html = "".join(conf_card(c) for c in power_confs if c in data)
    mid_html = "".join(conf_card(c) for c in mid_major_confs if c in data)
    small_html = "".join(conf_card(c) for c in small_confs if c in data)

    # Summary stats
    total_upsets = sum(d["upset_count"] for d in data.values())
    total_confs = len(data)
    one_seeds_winning = sum(1 for d in data.values() if d["champion_seed"] == 1)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>2026 Conference Tournament Predictions</title>
<style>
:root {{
    --bg: #0f1117;
    --card-bg: #1a1d27;
    --card-border: #2a2d3a;
    --accent: #4f8cff;
    --accent-light: #6ba3ff;
    --upset-red: #ff4757;
    --winner-green: #2ed573;
    --text: #e8eaed;
    --text-muted: #8b8fa3;
    --text-dim: #5a5e72;
    --power-gradient: linear-gradient(135deg, #4f8cff 0%, #7c5cfc 100%);
    --mid-gradient: linear-gradient(135deg, #2ed573 0%, #1abc9c 100%);
    --small-gradient: linear-gradient(135deg, #ffa502 0%, #ff6348 100%);
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 20px; }}
.dashboard-header {{ text-align: center; padding: 30px 0 20px; }}
.dashboard-header h1 {{ font-size: 2.2em; background: var(--power-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
.dashboard-header .subtitle {{ color: var(--text-muted); font-size: 1.1em; margin-top: 5px; }}
.summary-bar {{ display: flex; justify-content: center; gap: 40px; padding: 20px; margin: 15px auto; max-width: 800px; background: var(--card-bg); border-radius: 12px; border: 1px solid var(--card-border); }}
.summary-stat {{ text-align: center; }}
.summary-stat .stat-value {{ font-size: 2em; font-weight: 700; color: var(--accent); }}
.summary-stat .stat-label {{ font-size: 0.85em; color: var(--text-muted); }}
.section-title {{ font-size: 1.4em; font-weight: 700; margin: 30px 0 15px; padding-left: 10px; border-left: 4px solid var(--accent); }}
.section-title.mid {{ border-color: #2ed573; }}
.section-title.small {{ border-color: #ffa502; }}
.conf-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(450px, 1fr)); gap: 16px; margin-bottom: 30px; }}
.conf-card {{ background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px; overflow: hidden; transition: transform 0.2s; }}
.conf-card:hover {{ transform: translateY(-2px); border-color: var(--accent); }}
.card-header {{ padding: 16px 20px; cursor: pointer; display: flex; justify-content: space-between; align-items: flex-start; }}
.conf-title h3 {{ font-size: 1.15em; margin-bottom: 4px; }}
.team-count {{ font-size: 0.8em; color: var(--text-muted); }}
.upset-badge {{ background: var(--upset-red); color: white; font-size: 0.7em; padding: 2px 8px; border-radius: 10px; margin-left: 8px; }}
.champion-prediction {{ text-align: right; }}
.champ-label {{ font-size: 0.7em; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; }}
.champ-name {{ font-size: 1.2em; font-weight: 700; color: var(--winner-green); }}
.champ-detail {{ font-size: 0.75em; color: var(--text-muted); }}
.mc-prob {{ font-size: 0.75em; color: var(--accent-light); margin-top: 2px; }}
.contenders-section {{ padding: 0 20px 16px; }}
.section-label {{ font-size: 0.7em; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }}
.contender-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 5px; font-size: 0.85em; }}
.seed-badge {{ background: var(--card-border); color: var(--text-muted); font-size: 0.75em; padding: 1px 6px; border-radius: 4px; min-width: 28px; text-align: center; }}
.team-name {{ min-width: 120px; }}
.prob-bar-container {{ flex: 1; height: 8px; background: #252836; border-radius: 4px; overflow: hidden; }}
.prob-bar {{ height: 100%; background: var(--power-gradient); border-radius: 4px; transition: width 0.5s; }}
.prob-pct {{ font-weight: 600; min-width: 45px; text-align: right; font-size: 0.85em; }}
.bracket-detail {{ padding: 0 20px 20px; }}
.bracket-container {{ display: flex; gap: 12px; overflow-x: auto; padding: 10px 0; }}
.bracket-round {{ min-width: 200px; }}
.round-header {{ font-size: 0.8em; font-weight: 600; color: var(--accent); margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }}
.bracket-game {{ background: #252836; border-radius: 8px; padding: 8px 12px; margin-bottom: 8px; font-size: 0.82em; position: relative; }}
.bracket-game.upset {{ border-left: 3px solid var(--upset-red); }}
.matchup-line {{ padding: 2px 0; color: var(--text-muted); }}
.matchup-line .winner {{ color: var(--winner-green); }}
.game-prob {{ position: absolute; top: 8px; right: 10px; font-size: 0.75em; color: var(--text-dim); }}
.filter-bar {{ display: flex; gap: 8px; margin: 15px 0; flex-wrap: wrap; }}
.filter-btn {{ background: var(--card-bg); border: 1px solid var(--card-border); color: var(--text-muted); padding: 6px 14px; border-radius: 20px; cursor: pointer; font-size: 0.85em; transition: all 0.2s; }}
.filter-btn:hover, .filter-btn.active {{ background: var(--accent); color: white; border-color: var(--accent); }}
@media (max-width: 600px) {{ .conf-grid {{ grid-template-columns: 1fr; }} .summary-bar {{ flex-wrap: wrap; gap: 20px; }} }}
</style>
</head>
<body>
<div class="dashboard-header">
    <h1>2026 Conference Tournament Predictions</h1>
    <div class="subtitle">Multi-feature logistic model + 10,000 Monte Carlo simulations per conference</div>
    <div class="subtitle" style="font-size:0.85em; margin-top:4px; color:#5a5e72;">Generated March 9, 2026 | Tournaments begin March 10</div>
</div>

<div class="summary-bar">
    <div class="summary-stat">
        <div class="stat-value">{total_confs}</div>
        <div class="stat-label">Conferences</div>
    </div>
    <div class="summary-stat">
        <div class="stat-value">{one_seeds_winning}</div>
        <div class="stat-label">#1 Seeds Win</div>
    </div>
    <div class="summary-stat">
        <div class="stat-value">{total_confs - one_seeds_winning}</div>
        <div class="stat-label">Non-#1 Champs</div>
    </div>
    <div class="summary-stat">
        <div class="stat-value">{total_upsets}</div>
        <div class="stat-label">Predicted Upsets</div>
    </div>
</div>

<div class="filter-bar">
    <button class="filter-btn active" onclick="filterConfs('all')">All</button>
    <button class="filter-btn" onclick="filterConfs('power')">Power Conferences</button>
    <button class="filter-btn" onclick="filterConfs('mid')">Mid-Majors</button>
    <button class="filter-btn" onclick="filterConfs('small')">Small Conferences</button>
    <button class="filter-btn" onclick="filterConfs('upsets')">Upset Picks</button>
</div>

<div class="section-title" id="section-power">Power Conferences</div>
<div class="conf-grid" id="grid-power">
    {power_html}
</div>

<div class="section-title mid" id="section-mid">Mid-Major Conferences</div>
<div class="conf-grid" id="grid-mid">
    {mid_html}
</div>

<div class="section-title small" id="section-small">Small Conferences</div>
<div class="conf-grid" id="grid-small">
    {small_html}
</div>

<script>
function toggleBracket(conf) {{
    const el = document.getElementById('bracket-' + conf);
    el.style.display = el.style.display === 'none' ? 'block' : 'none';
}}

const POWER = {json.dumps(power_confs)};
const MID = {json.dumps(mid_major_confs)};

function filterConfs(filter) {{
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    event.target.classList.add('active');

    document.querySelectorAll('.conf-card').forEach(card => {{
        const conf = card.dataset.conf;
        let show = true;
        if (filter === 'power') show = POWER.includes(conf);
        else if (filter === 'mid') show = MID.includes(conf);
        else if (filter === 'small') show = !POWER.includes(conf) && !MID.includes(conf);
        else if (filter === 'upsets') show = card.querySelector('.upset-badge') !== null;
        card.style.display = show ? '' : 'none';
    }});

    // Show/hide section headers
    ['power', 'mid', 'small'].forEach(s => {{
        const section = document.getElementById('section-' + s);
        const grid = document.getElementById('grid-' + s);
        if (filter === 'all' || filter === s || filter === 'upsets') {{
            section.style.display = '';
            grid.style.display = '';
        }} else {{
            section.style.display = 'none';
            grid.style.display = 'none';
        }}
    }});
    if (filter === 'upsets') {{
        ['power', 'mid', 'small'].forEach(s => {{
            document.getElementById('section-' + s).style.display = '';
            document.getElementById('grid-' + s).style.display = '';
        }});
    }}
}}
</script>
</body>
</html>"""


# ──────────────────────────────────────────────────────────────────────
# 5.  MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 70)
    logger.info("  CONFERENCE TOURNAMENT PREDICTION PIPELINE  —  March 2026")
    logger.info("=" * 70)

    # Step 1: Scrape fresh data
    logger.info("\n[STEP 1] Data Acquisition")
    scrape_fresh_data()

    # Step 2: Build enhanced predictor
    logger.info("\n[STEP 2] Building Enhanced Predictor")
    torvik_path = str(DATA_RAW / "torvik_2026.json")
    predictor = EnhancedPredictor(torvik_path, data_dir=str(DATA_RAW), year=2026)

    n_teams = len(predictor.teams_by_id)
    n_confs = len(predictor.teams_by_conf)
    n_ff = sum(1 for t in predictor.teams_by_id.values() if t.get("efg", 0) > 0)
    logger.info("  Loaded %d teams across %d conferences", n_teams, n_confs)
    logger.info("  Four Factors enrichment: %d/%d teams (%.0f%%)", n_ff, n_teams, n_ff/n_teams*100 if n_teams else 0)

    # Step 3: Run simulations for all conferences
    logger.info("\n[STEP 3] Running Monte Carlo Simulations (10,000 per conference)")
    all_results = {}
    for conf in sorted(predictor.teams_by_conf.keys()):
        result = simulate_bracket(predictor, conf, num_sims=10000)
        if result:
            all_results[conf] = result
            mc = result["mc_favorite"]
            det_champ = result["predicted_champion"]
            logger.info("  %s: Champion=%s (Seed #%d) | MC Favorite=%s (%.1f%%)",
                       CONF_FULL_NAMES.get(conf, conf),
                       det_champ["name"] if det_champ else "?",
                       det_champ["conf_seed"] if det_champ else 0,
                       mc["name"], mc["prob"] * 100)

    # Step 4: Save JSON results
    logger.info("\n[STEP 4] Saving Results")

    json_output = {}
    for conf, result in all_results.items():
        det = result["deterministic_bracket"]
        json_output[conf] = {
            "conference": conf,
            "conference_name": CONF_FULL_NAMES.get(conf, conf),
            "num_teams": result["num_teams"],
            "predicted_champion": {
                "name": det["champion"]["name"] if det["champion"] else None,
                "team_id": det["champion"]["team_id"] if det["champion"] else None,
                "seed": det["champion"]["conf_seed"] if det["champion"] else None,
                "t_rank": det["champion"]["t_rank"] if det["champion"] else None,
            },
            "mc_favorite": result["mc_favorite"],
            "championship_probabilities": {
                tid: round(p, 4) for tid, p in result["championship_probs"].items() if p >= 0.005
            },
            "rounds": [
                {
                    "round_name": rnd["round_name"],
                    "games": [
                        {
                            "team1": f"({g['team1']['conf_seed']}) {g['team1']['name']}",
                            "team2": f"({g['team2']['conf_seed']}) {g['team2']['name']}",
                            "winner": g["winner"]["name"],
                            "win_prob": round(g["win_prob"], 3),
                            "is_upset": g["is_upset"],
                        }
                        for g in rnd["games"]
                    ]
                }
                for rnd in det["rounds"]
            ],
            "convergence": result.get("convergence"),
        }

    json_path = OUTPUT_DIR / "conference_tournament_predictions_2026.json"
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)
    logger.info("  JSON: %s", json_path)

    # Sync to docs/data for the web app (GitHub Pages)
    docs_data_dir = PROJECT_ROOT / "docs" / "data"
    docs_data_dir.mkdir(parents=True, exist_ok=True)
    docs_json_path = docs_data_dir / "conference_predictions_2026.json"
    shutil.copy2(json_path, docs_json_path)
    logger.info("  Synced to %s", docs_json_path)

    # Step 4b: Validate output against bracket format definitions
    logger.info("\n[STEP 4b] Validating Output")
    validation_errors = _validate_output(json_path)
    if validation_errors:
        for err in validation_errors:
            logger.critical("  VALIDATION FAIL: %s", err)
        raise RuntimeError(
            f"Output validation failed with {len(validation_errors)} error(s). "
            "See log above for details."
        )
    logger.info("  All %d conferences passed output validation", len(json_output))

    # Step 5: Generate dashboard
    dashboard_path = str(PROJECT_ROOT / "outputs" / "conference_tournament_predictions_2026.html")
    generate_dashboard(all_results, dashboard_path)
    logger.info("  Dashboard: %s", dashboard_path)

    # Step 6: Print summary
    logger.info("\n" + "=" * 70)
    logger.info("  PREDICTION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Conference':<25} {'Champion':<25} {'Seed':>4}  {'T-Rank':>6}  {'MC Prob':>7}")
    logger.info("-" * 75)

    for conf in sorted(all_results.keys()):
        r = all_results[conf]
        det = r["deterministic_bracket"]
        champ = det["champion"]
        mc = r["mc_favorite"]
        if champ:
            logger.info(f"  {CONF_FULL_NAMES.get(conf, conf):<23} {champ['name']:<25} {champ['conf_seed']:>4}  {champ['t_rank']:>6}  {mc['prob']*100:>6.1f}%")

    # Upset summary
    total_upsets = sum(
        sum(1 for rnd in r["deterministic_bracket"]["rounds"] for g in rnd["games"] if g["is_upset"])
        for r in all_results.values()
    )
    one_seeds = sum(1 for r in all_results.values() if r["deterministic_bracket"]["champion"] and r["deterministic_bracket"]["champion"]["conf_seed"] == 1)

    logger.info("-" * 75)
    logger.info(f"  #1 seeds winning: {one_seeds}/{len(all_results)}")
    logger.info(f"  Total predicted upsets: {total_upsets}")
    logger.info("=" * 70)

    return all_results


if __name__ == "__main__":
    results = main()
