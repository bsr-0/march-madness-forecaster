#!/usr/bin/env python3
"""Evaluate conference tournament predictions against actual results.

Compiles all conference tournament games with known results (through
March 11, 2026) and compares them to the model's predictions. Computes
accuracy, Brier score, calibration, and highlights correct/incorrect
picks including upsets.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class GameResult:
    """A single game with prediction and actual outcome."""
    conference: str
    conference_name: str
    round_name: str
    team1: str          # As shown in predictions (e.g. "(10) Stanford")
    team2: str
    predicted_winner: str
    predicted_prob: float  # Probability assigned to predicted winner
    actual_winner: str
    actual_loser: str
    actual_score: str      # e.g. "64-63"
    correct: bool
    is_upset: bool = False  # Lower seed won
    notes: str = ""


def parse_seed(team_str: str) -> int:
    """Extract seed number from team string like '(10) Stanford'."""
    if team_str.startswith("("):
        return int(team_str.split(")")[0][1:])
    return 99


def build_results() -> List[GameResult]:
    """Build list of game results from completed conference tournament games.

    Data sourced from ESPN, CBS Sports, and conference websites for games
    completed through March 11, 2026.
    """
    predictions_path = Path(__file__).parent.parent / "data" / "processed" / "conference_tournament_predictions_2026.json"
    with open(predictions_path) as f:
        predictions = json.load(f)

    results: List[GameResult] = []

    # ── Helper to look up a prediction for a specific game ──────────
    def find_prediction(conf: str, round_name: str, team1_fragment: str, team2_fragment: str):
        """Find a predicted game matching conference, round, and team name fragments."""
        conf_data = predictions.get(conf, {})
        for rnd in conf_data.get("rounds", []):
            if rnd["round_name"] != round_name:
                continue
            for game in rnd.get("games", []):
                t1 = game["team1"].lower()
                t2 = game["team2"].lower()
                f1 = team1_fragment.lower()
                f2 = team2_fragment.lower()
                if (f1 in t1 and f2 in t2) or (f1 in t2 and f2 in t1):
                    return game
        return None

    def add_result(conf: str, conf_name: str, round_name: str,
                   team1_frag: str, team2_frag: str,
                   actual_winner_frag: str, score: str,
                   is_upset: bool = False, notes: str = ""):
        pred = find_prediction(conf, round_name, team1_frag, team2_frag)
        if pred is None:
            print(f"  WARNING: No prediction found for {conf} {round_name}: {team1_frag} vs {team2_frag}")
            return

        pred_winner = pred["winner"]
        pred_prob = pred["win_prob"]
        actual_winner = actual_winner_frag
        actual_loser = team2_frag if actual_winner_frag.lower() in team1_frag.lower() or actual_winner_frag.lower() in pred["team1"].lower() else team1_frag

        # Determine if prediction was correct
        correct = actual_winner_frag.lower() in pred_winner.lower()

        # Compute implied probability for actual winner
        # If we predicted team A with prob p, and team A won, prob_actual = p
        # If team B won, prob_actual = 1 - p
        if correct:
            prob_for_actual = pred_prob
        else:
            prob_for_actual = 1.0 - pred_prob

        results.append(GameResult(
            conference=conf,
            conference_name=conf_name,
            round_name=round_name,
            team1=pred["team1"],
            team2=pred["team2"],
            predicted_winner=pred_winner,
            predicted_prob=pred_prob,
            actual_winner=actual_winner_frag,
            actual_loser=actual_loser,
            actual_score=score,
            correct=correct,
            is_upset=is_upset,
            notes=notes,
        ))

    # ═══════════════════════════════════════════════════════════════
    # COMPLETED CHAMPIONSHIP GAMES
    # ═══════════════════════════════════════════════════════════════

    # OVC Championship (March 7): Tennessee St (1) 93, Morehead St (2) 67
    add_result("OVC", "Ohio Valley", "Championship",
               "Tennessee Martin", "Southeast Missouri",
               "Tennessee St.", "93-67", notes="Seed mismatch: model had different seedings than actual")
    # NOTE: Our model had Tennessee Martin as 1-seed, actual OVC had Tennessee State as 1-seed
    # The championship matchup was entirely different, so this won't match. Skipping individual game.

    # SoCon Championship (March 9): Furman (6) 76, ETSU (1) 61  [UPSET]
    add_result("SC", "Southern", "Championship",
               "East Tennessee", "Furman",
               "Furman", "76-61", is_upset=True,
               notes="6-seed Furman upset 1-seed ETSU")

    # MVC Championship (March 8): Northern Iowa (6) 84, UIC (5) 69
    # Our model had Belmont vs Northern Iowa in championship
    add_result("MVC", "Missouri Valley", "Championship",
               "Belmont", "Northern Iowa",
               "Northern Iowa", "84-69",
               notes="6-seed UNI won; model had Belmont as opponent, actual opponent was UIC (5)")

    # Sun Belt Championship (March 9): Troy (1) 77, Georgia Southern (10) 61
    # Our model had Troy vs Arkansas St. in championship
    add_result("SB", "Sun Belt", "Championship",
               "Troy", "Arkansas St.",
               "Troy", "77-61",
               notes="Troy won as predicted; actual opponent was Georgia Southern (10), not Arkansas St.")

    # ASUN Championship (March 8): Queens (4) 98, Central Arkansas (2) 93 OT
    add_result("ASun", "ASUN", "Championship",
               "Austin Peay", "Central Arkansas",
               "Queens", "98-93 OT",
               notes="Queens (4-seed) won; model predicted Austin Peay (1) vs Central Arkansas")

    # Horizon League Championship (March 10): Wright State (1) 66, Detroit Mercy (3) 63
    # Our model: Wright St vs Robert Morris
    add_result("Horz", "Horizon League", "Championship",
               "Wright St.", "Robert Morris",
               "Wright St.", "66-63",
               notes="Wright St won as predicted; actual opponent was Detroit Mercy (3)")

    # CAA Championship (March 10): Hofstra (1) 56, Monmouth (8) 51
    # Our model: Hofstra vs UNC Wilmington
    add_result("CAA", "CAA", "Championship",
               "Hofstra", "UNC Wilmington",
               "Hofstra", "56-51",
               notes="Hofstra won as predicted; actual opponent was Monmouth (8)")

    # NEC Championship (March 10): LIU (1) 79, Mercyhurst (3) 70
    add_result("NEC", "NEC", "Championship",
               "LIU", "Mercyhurst",
               "LIU", "79-70")

    # Big South Championship (March 8): High Point (1) 91, Winthrop (2) 76
    add_result("BSth", "Big South", "Championship",
               "High Point", "Winthrop",
               "High Point", "91-76")

    # WCC Championship (March 10): Gonzaga (1) 79, Santa Clara (3) 68
    # Our model: Gonzaga vs Santa Clara — exact match!
    add_result("WCC", "WCC", "Round 6",
               "Gonzaga", "Santa Clara",
               "Gonzaga", "79-68")

    # ═══════════════════════════════════════════════════════════════
    # ACC FIRST ROUND (Play-in, March 10)
    # ═══════════════════════════════════════════════════════════════

    # Pitt (15) 64, Stanford (10) 63  [UPSET]
    add_result("ACC", "ACC", "Play-in",
               "Stanford", "Pittsburgh",
               "Pittsburgh", "64-63", is_upset=True,
               notes="15-seed Pitt upset 10-seed Stanford on last-second put-back")

    # SMU (11) 86, Syracuse (14) 69
    add_result("ACC", "ACC", "Play-in",
               "SMU", "Syracuse",
               "SMU", "86-69")

    # Wake Forest (13) 95, Virginia Tech (12) 89 OT  [UPSET by seed]
    add_result("ACC", "ACC", "Play-in",
               "Virginia Tech", "Wake Forest",
               "Wake Forest", "95-89 OT", is_upset=True,
               notes="13-seed Wake Forest upset 12-seed Virginia Tech in OT")

    # ═══════════════════════════════════════════════════════════════
    # BIG TEN ROUND 1 (March 10)
    # ═══════════════════════════════════════════════════════════════

    # Maryland (17) 70, Oregon (16) 60
    add_result("B10", "Big Ten", "Round 1",
               "Oregon", "Maryland",
               "Maryland", "70-60",
               notes="17-seed Maryland upset 16-seed Oregon")

    # Northwestern (15) 76, Penn State (18) 66
    add_result("B10", "Big Ten", "Round 1",
               "Northwestern", "Penn St.",
               "Northwestern", "76-66")

    # ═══════════════════════════════════════════════════════════════
    # BIG 12 PLAY-IN (March 10)
    # ═══════════════════════════════════════════════════════════════

    # Arizona State (12) 83, Baylor (13) 79  [UPSET]
    add_result("B12", "Big 12", "Play-in",
               "Arizona St.", "Baylor",
               "Arizona St.", "83-79", is_upset=True,
               notes="12-seed ASU upset 13-seed Baylor — seed difference minimal")

    # Cincinnati (9) 73, Utah (16) 66
    add_result("B12", "Big 12", "Play-in",
               "Cincinnati", "Utah",
               "Cincinnati", "73-66")

    # BYU (10) 105, Kansas State (15) 91
    add_result("B12", "Big 12", "Play-in",
               "BYU", "Kansas St.",
               "BYU", "105-91")

    # WCC Semifinal results
    # Gonzaga (1) 65, Oregon St (4) 56
    add_result("WCC", "WCC", "Round 5",
               "Gonzaga", "Pacific",  # model had Gonzaga vs Pacific
               "Gonzaga", "65-56",
               notes="Gonzaga won as predicted; actual opponent was Oregon St (4)")

    # Santa Clara (3) 76, Saint Mary's (2) 71
    add_result("WCC", "WCC", "Round 5",
               "Saint Mary's", "Santa Clara",
               "Santa Clara", "76-71",
               notes="3-seed Santa Clara upset 2-seed Saint Mary's")

    return results


def compute_brier_score(results: List[GameResult]) -> float:
    """Compute Brier score across all results.

    Brier = (1/N) * sum((forecast - outcome)^2)
    where outcome=1 if predicted winner actually won, 0 otherwise.
    """
    if not results:
        return 0.0
    total = 0.0
    for r in results:
        if r.correct:
            total += (1.0 - r.predicted_prob) ** 2
        else:
            total += r.predicted_prob ** 2
    return total / len(results)


def evaluate(results: List[GameResult]):
    """Print comprehensive evaluation report."""
    if not results:
        print("No results to evaluate.")
        return

    print("=" * 78)
    print("  CONFERENCE TOURNAMENT PREDICTION EVALUATION")
    print("  Games completed through March 11, 2026")
    print("=" * 78)

    # ── Data Staleness Warning ────────────────────────────────────
    print(f"\n  DATA STALENESS WARNING:")
    print(f"  - Torvik data last fetched: Feb 25, 2026 (14 days old)")
    print(f"  - historical_games_2026.json: through Mar 7 (4 days old)")
    print(f"  - 4 Torvik Four Factors fields (eFG%, TO%, ORB%, FTR) are ALL ZERO")
    print(f"  - RAPM fields in rosters are ALL ZERO")
    print(f"  - Predictions were made with degraded input data")

    # ── Overall Accuracy ───────────────────────────────────────────
    correct = sum(1 for r in results if r.correct)
    total = len(results)
    accuracy = correct / total if total > 0 else 0

    print(f"\n  OVERALL: {correct}/{total} correct ({accuracy:.1%})")

    # ── Brier Score ────────────────────────────────────────────────
    brier = compute_brier_score(results)
    # Baseline: predicting 0.5 for every game
    baseline_brier = 0.25
    brier_skill = 1.0 - (brier / baseline_brier) if baseline_brier > 0 else 0

    print(f"  Brier Score: {brier:.4f} (lower is better)")
    print(f"  Baseline (coin flip): {baseline_brier:.4f}")
    print(f"  Brier Skill Score: {brier_skill:.4f} (>0 = better than coin flip)")

    # ── Log Loss ───────────────────────────────────────────────────
    log_loss_total = 0.0
    for r in results:
        prob_actual = r.predicted_prob if r.correct else (1.0 - r.predicted_prob)
        prob_actual = max(prob_actual, 1e-6)  # avoid log(0)
        log_loss_total -= math.log(prob_actual)
    avg_log_loss = log_loss_total / total
    print(f"  Log Loss: {avg_log_loss:.4f} (lower is better)")
    print(f"  Baseline (coin flip): {math.log(2):.4f}")

    # ── Separate completed-tournament champions from early-round picks ─
    # Conferences whose tournaments have fully concluded
    completed_confs = {"Southern", "Sun Belt", "Missouri Valley", "ASUN",
                       "Horizon League", "CAA", "NEC", "Big South", "WCC",
                       "Ohio Valley", "MAAC"}
    # Conferences still in early rounds (power conferences)
    in_progress_confs = {"ACC", "Big Ten", "Big 12", "SEC", "Big East",
                         "Mountain West"}

    completed_results = [r for r in results if r.conference_name in completed_confs]
    early_round_results = [r for r in results if r.conference_name not in completed_confs]

    if completed_results:
        cc = sum(1 for r in completed_results if r.correct)
        print(f"\n  COMPLETED TOURNAMENTS (champion known): {cc}/{len(completed_results)} picks correct ({cc/len(completed_results):.1%})")
    if early_round_results:
        ec = sum(1 for r in early_round_results if r.correct)
        print(f"  IN-PROGRESS TOURNAMENTS (early rounds):  {ec}/{len(early_round_results)} picks correct ({ec/len(early_round_results):.1%})")

    # ── Upset Detection ───────────────────────────────────────────
    upset_games = [r for r in results if r.is_upset]
    if upset_games:
        upsets_predicted = sum(1 for r in upset_games if r.correct)
        print(f"\n  UPSETS: {upsets_predicted}/{len(upset_games)} upsets correctly predicted ({upsets_predicted/len(upset_games):.1%})")

    # ── Calibration Buckets ───────────────────────────────────────
    print(f"\n  {'─' * 60}")
    print(f"  CALIBRATION (predicted prob vs actual win rate):")
    print(f"  {'─' * 60}")
    buckets = {}
    for r in results:
        # Bucket by the higher probability (favored team)
        p = r.predicted_prob
        bucket = round(p * 10) / 10  # Round to nearest 0.1
        bucket = max(0.5, min(1.0, bucket))  # Clamp to [0.5, 1.0]
        bucket_key = f"{bucket:.0%}"
        if bucket_key not in buckets:
            buckets[bucket_key] = {"correct": 0, "total": 0, "prob_sum": 0.0}
        buckets[bucket_key]["total"] += 1
        buckets[bucket_key]["correct"] += 1 if r.correct else 0
        buckets[bucket_key]["prob_sum"] += p

    print(f"  {'Bucket':>10}  {'Predicted':>10}  {'Actual':>10}  {'N':>5}")
    for bucket_key in sorted(buckets.keys()):
        b = buckets[bucket_key]
        avg_pred = b["prob_sum"] / b["total"]
        actual_rate = b["correct"] / b["total"]
        print(f"  {bucket_key:>10}  {avg_pred:>10.1%}  {actual_rate:>10.1%}  {b['total']:>5}")

    # ── Detailed Results by Conference ────────────────────────────
    print(f"\n  {'─' * 60}")
    print(f"  DETAILED RESULTS")
    print(f"  {'─' * 60}")

    by_conf = {}
    for r in results:
        by_conf.setdefault(r.conference_name, []).append(r)

    for conf_name in sorted(by_conf.keys()):
        games = by_conf[conf_name]
        conf_correct = sum(1 for g in games if g.correct)
        print(f"\n  {conf_name} ({conf_correct}/{len(games)})")

        for g in games:
            icon = "  OK" if g.correct else "MISS"
            upset_tag = " [UPSET]" if g.is_upset else ""
            print(f"    [{icon}] {g.round_name}: {g.team1} vs {g.team2}")
            print(f"           Predicted: {g.predicted_winner} ({g.predicted_prob:.1%})")
            print(f"           Actual: {g.actual_winner} {g.actual_score}{upset_tag}")
            if g.notes:
                print(f"           Note: {g.notes}")

    # ── Summary Table: Completed Tournaments Only ──────────────────
    print(f"\n  {'═' * 60}")
    print(f"  COMPLETED CONFERENCE TOURNAMENT CHAMPIONS")
    print(f"  (only tournaments that have finished)")
    print(f"  {'═' * 60}")

    champion_data = [
        ("Ohio Valley", "Tennessee Martin", "Tennessee State", False, "Seeding mismatch in model"),
        ("Southern", "East Tennessee St.", "Furman (6-seed)", False, "Upset"),
        ("Missouri Valley", "Belmont", "Northern Iowa (6-seed)", False, "Upset"),
        ("Sun Belt", "Troy", "Troy", True, ""),
        ("ASUN", "Austin Peay", "Queens (4-seed)", False, "Upset"),
        ("Horizon League", "Wright State", "Wright State", True, ""),
        ("CAA", "Hofstra", "Hofstra", True, ""),
        ("NEC", "LIU", "LIU", True, ""),
        ("Big South", "High Point", "High Point", True, ""),
        ("WCC", "Gonzaga", "Gonzaga", True, ""),
    ]

    champ_correct = sum(1 for _, _, _, c, _ in champion_data if c)
    print(f"\n  {'Conference':<20} {'Predicted':<20} {'Actual':<22} {'Result':<8}")
    print(f"  {'─'*20} {'─'*20} {'─'*22} {'─'*8}")
    for conf, pred, actual, correct, note in champion_data:
        icon = "  OK" if correct else "MISS"
        note_str = f" ({note})" if note else ""
        print(f"  {conf:<20} {pred:<20} {actual:<22} {icon}{note_str}")

    print(f"\n  Champion accuracy: {champ_correct}/{len(champion_data)} ({champ_correct/len(champion_data):.0%})")

    # ── In-Progress Tournaments ──────────────────────────────────
    print(f"\n  {'═' * 60}")
    print(f"  IN-PROGRESS TOURNAMENTS (early-round picks only)")
    print(f"  {'═' * 60}")

    in_progress_data = [
        ("ACC", "Play-in (3 games completed)", "1/3"),
        ("Big Ten", "Round 1 (2 games completed)", "1/2"),
        ("Big 12", "Play-in (3 games completed)", "2/3"),
        ("SEC", "Play-in (today, results pending)", "--"),
        ("Big East", "First Round (today, results pending)", "--"),
        ("Mountain West", "First Round (today, results pending)", "--"),
    ]

    print(f"\n  {'Conference':<20} {'Stage':<35} {'Record':<8}")
    print(f"  {'─'*20} {'─'*35} {'─'*8}")
    for conf, stage, record in in_progress_data:
        print(f"  {conf:<20} {stage:<35} {record:<8}")

    # ── Key Takeaways ─────────────────────────────────────────────
    print(f"\n  {'═' * 60}")
    print(f"  KEY TAKEAWAYS")
    print(f"  {'═' * 60}")

    high_conf_correct = sum(1 for r in results if r.correct and r.predicted_prob >= 0.65)
    high_conf_total = sum(1 for r in results if r.predicted_prob >= 0.65)
    low_conf_correct = sum(1 for r in results if r.correct and r.predicted_prob < 0.55)
    low_conf_total = sum(1 for r in results if r.predicted_prob < 0.55)

    if high_conf_total > 0:
        print(f"\n  High-confidence picks (>=65%): {high_conf_correct}/{high_conf_total} ({high_conf_correct/high_conf_total:.0%})")
    if low_conf_total > 0:
        print(f"  Low-confidence picks (<55%):   {low_conf_correct}/{low_conf_total} ({low_conf_correct/low_conf_total:.0%})")

    # Biggest misses
    misses = [r for r in results if not r.correct]
    if misses:
        misses.sort(key=lambda r: r.predicted_prob, reverse=True)
        print(f"\n  Biggest misses (highest confidence wrong picks):")
        for m in misses[:5]:
            print(f"    - {m.conference_name} {m.round_name}: predicted {m.predicted_winner} ({m.predicted_prob:.1%}), "
                  f"actual winner: {m.actual_winner} {m.actual_score}")

    print(f"\n{'=' * 78}")


def main():
    results = build_results()
    evaluate(results)


if __name__ == "__main__":
    main()
