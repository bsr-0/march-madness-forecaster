"""Generate the pairwise tournament matchup table for the web app.

Produces:
  docs/data/matchups_by_year.json - one row per tournament game, 2010-2026.

Matchup features are inherently pairwise — "this offense against that
defense" is a property of a PAIR of teams, not of either team alone — so
they cannot live as columns in the per-team stats table. This is that
missing half: every tournament game, with each side's pre-tournament
profile expressed as the crossings that actually decide games (my offense
vs your defense, my ball security vs your pressure, my offensive glass vs
your defensive glass), plus what actually happened.

Team stats are read straight from docs/data/team_stats_by_year.json rather
than re-derived, so this table inherits that file's leakage guarantees and
ID fixes by construction — every stat here is pre-tournament, and the
result columns are namespaced `result_*` exactly as they are there.

Orientation: team1 is always the better seed (the favourite), team2 the
worse seed. Differentials are favourite-minus-underdog, so a positive
number always means "the favourite has the edge". Seed ties are broken by
Torvik rank.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._common import load_tournament_results

OUT = PROJECT_ROOT / "docs" / "data"
STATS_PATH = OUT / "team_stats_by_year.json"

ROUND_LABEL = {
    "FF": "First Four",
    "R64": "Round of 64",
    "R32": "Round of 32",
    "S16": "Sweet 16",
    "E8": "Elite 8",
    "F4": "Final Four",
    "NCG": "Championship",
}
ROUND_ORDER = ["FF", "R64", "R32", "S16", "E8", "F4", "NCG"]


def _diff(a, b):
    return None if a is None or b is None else round(a - b, 3)


def build_matchups(year: int, team_rows: list[dict]) -> list[dict]:
    stats = {r["team_id"]: r for r in team_rows}
    games = load_tournament_results(year)
    out = []
    for g in games:
        t1, t2 = g.get("team1_id"), g.get("team2_id")
        s1, s2 = stats.get(t1), stats.get(t2)
        if not s1 or not s2:
            continue  # a team without a Torvik profile — nothing to compare

        # Orient favourite first: better (lower) seed, Torvik rank breaks ties.
        def rank_key(s):
            return (s["seed"], s["t_rank"] if s["t_rank"] is not None else 999)

        fav, dog = (s1, s2) if rank_key(s1) <= rank_key(s2) else (s2, s1)
        fav_score = g["team1_score"] if fav is s1 else g["team2_score"]
        dog_score = g["team2_score"] if fav is s1 else g["team1_score"]

        row = {
            "round": ROUND_LABEL.get(g.get("round_name"), g.get("round_name")),
            "round_key": g.get("round_name"),
            "region": g.get("region"),
            "fav": fav["team_name"],
            "fav_seed": fav["seed"],
            "dog": dog["team_name"],
            "dog_seed": dog["seed"],
            # --- pairwise crossings, favourite minus underdog -----------------
            "barthag_diff": _diff(fav["barthag"], dog["barthag"]),
            "seed_diff": dog["seed"] - fav["seed"],
            # My offense against your defense, both directions. Positive on
            # `fav_off_vs_dog_def` means the favourite scores more efficiently
            # than the underdog typically concedes.
            "fav_off_vs_dog_def": _diff(fav["adj_offensive_efficiency"], dog["adj_defensive_efficiency"]),
            "dog_off_vs_fav_def": _diff(dog["adj_offensive_efficiency"], fav["adj_defensive_efficiency"]),
            # Shooting: my eFG% against what you normally allow.
            "fav_efg_vs_dog_def": _diff(fav["effective_fg_pct"], dog["opp_effective_fg_pct"]),
            "dog_efg_vs_fav_def": _diff(dog["effective_fg_pct"], fav["opp_effective_fg_pct"]),
            # Ball security against pressure: my turnover rate vs the rate you force.
            "fav_to_vs_dog_press": _diff(fav["turnover_rate"], dog["opp_turnover_rate"]),
            "dog_to_vs_fav_press": _diff(dog["turnover_rate"], fav["opp_turnover_rate"]),
            # The glass: my offensive rebounding against your defensive rebounding.
            "fav_oreb_vs_dog_dreb": _diff(fav["offensive_reb_rate"], dog["defensive_reb_rate"]),
            "dog_oreb_vs_fav_dreb": _diff(dog["offensive_reb_rate"], fav["defensive_reb_rate"]),
            "tempo_diff": _diff(fav["adj_tempo"], dog["adj_tempo"]),
            # Style/fragility context carried over from the team table.
            "fav_margin_sd": fav["reg_season_margin_std"],
            "dog_margin_sd": dog["reg_season_margin_std"],
            "fav_close_win_pct": fav["close_game_win_rate"],
            "dog_close_win_pct": dog["close_game_win_rate"],
        }

        # --- what actually happened (post-hoc) --------------------------------
        if fav_score is not None and dog_score is not None:
            fav_won = fav_score > dog_score
            row.update(
                {
                    "result_winner": fav["team_name"] if fav_won else dog["team_name"],
                    "result_score": f"{max(fav_score, dog_score)}-{min(fav_score, dog_score)}",
                    "result_margin": abs(fav_score - dog_score),
                    "result_upset": not fav_won,
                }
            )
        else:
            row.update({"result_winner": None, "result_score": None, "result_margin": None, "result_upset": None})
        out.append(row)

    order = {r: i for i, r in enumerate(ROUND_ORDER)}
    out.sort(key=lambda r: (order.get(r["round_key"], 99), r["fav_seed"]))
    return out


def main() -> None:
    if not STATS_PATH.exists():
        raise SystemExit(f"{STATS_PATH} not found — run scripts/generate_team_stats_table.py first")
    with open(STATS_PATH) as f:
        stats = json.load(f)

    matchups_by_year = {}
    for year_str, rows in stats["stats_by_year"].items():
        games = build_matchups(int(year_str), rows)
        if games:
            matchups_by_year[year_str] = games
            upsets = sum(1 for g in games if g["result_upset"])
            print(f"  {year_str}: {len(games)} games, {upsets} won by the lower seed")

    payload = {
        "years": sorted(int(y) for y in matchups_by_year),
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "matchups_by_year": matchups_by_year,
    }
    out_path = OUT / "matchups_by_year.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)  # NaN/Infinity are not valid JSON
        f.write("\n")
    total = sum(len(v) for v in matchups_by_year.values())
    print(f"\nWrote {out_path} ({len(payload['years'])} years, {total} games)")


if __name__ == "__main__":
    main()
