"""Generate a multi-year pre-tournament team stats table for the web app.

Produces:
  docs/data/team_stats_by_year.json - one row per tournament-qualified team
    per year, 2010-2026, combining three source families:

  1. Torvik pre-tournament ratings (data/raw/historical/torvik_{year}.json) —
     barthag, adjusted efficiencies, tempo, and the eight four-factor fields.
  2. Regular-season volatility, computed here from the full game log
     (data/raw/historical/historical_games_{year}.json) — scoring margin
     mean/spread, close-game rate and record, and rate of losses to
     lower-ranked opponents.
  3. Post-hoc tournament outcome (tournament_context_{year}.json) — how far
     the team actually got. This is NOT pre-tournament information; it is
     kept in clearly-labelled `outcome_*` fields and rendered as a visually
     separate block in the UI so it can never be mistaken for a feature that
     was knowable before the tournament tipped off.

Leakage guard: family (2) filters the game log to games played strictly
BEFORE that year's tournament_start. Most years' logs run through the
national championship, so an unfiltered read would silently fold tournament
results into a "pre-tournament" column.

Deliberately excluded: data/raw/torvik_shooting_{year}.json (3PT%/FT%).
docs/data-provenance-and-leakage-audit.md classifies the 2008-2025 shooting
files as post-tournament player stats with "no date filtering possible" —
only 2026 is verified pre-tournament. A column clean in 1 of 16 years and
contaminated in the rest is worse than no column.
"""

import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._common import HIST_DIR, load_seeds_and_regions, load_torvik_and_ff, load_tournament_results
from src.data.normalize import bridge_cbbpy_id

OUT = PROJECT_ROOT / "docs" / "data"
OUT.mkdir(parents=True, exist_ok=True)

YEARS = range(2010, 2027)
VALID_PRETOURNAMENT_TYPES = {"pre_tournament", "pre_tournament_computed"}

CLOSE_GAME_MARGIN = 6  # points; a "close game" is decided by <= this

# Extra cbbpy->canonical aliases, applied on top of src.data.normalize's shared
# bridge. These are the ~10 programs whose cbbpy IDs share no prefix with their
# Torvik ID ("unlv_rebels" vs "nevada_las_vegas"), so longest-prefix matching
# can't reach them; without these they silently lose their game-log stats.
#
# Kept LOCAL rather than added to normalize._CBBPY_EDGE_CASES on purpose: that
# shared dict feeds three production probability bases (Elo A4, roster_adj C2,
# volatile), so widening it would change what those bases see for these teams
# and could move backtest results without going through the acceptance gate.
# This table is display-only, so the blast radius stays here.
_SEEDS_TO_TORVIK_ALIASES = {
    # A handful of seeds-file IDs that Torvik spells differently. Without
    # these the team is dropped from the table entirely (it has no Torvik row
    # to join to), which also punches a hole in that year's bracket.
    "sam_houston": "sam_houston_state",
    "southern_miss": "southern_mississippi",
    "umass": "massachusetts",
}

_EXTRA_CBBPY_ALIASES = {
    "unlv_rebels": "nevada_las_vegas",
    "usc_trojans": "southern_california",
    "lsu_tigers": "louisiana_state",
    "ole_miss_rebels": "mississippi",
    "loyola_chicago_ramblers": "loyola__il",
    "loyola_maryland_greyhounds": "loyola_md",
    "ualbany_great_danes": "albany__ny",
    "charleston_cougars": "college_of_charleston",
    "app_state_mountaineers": "appalachian_state",
    "mount_st_mary_s_mountaineers": "mount_st__mary_s",
    "sam_houston_bearkats": "sam_houston_state",
    "southern_miss_golden_eagles": "southern_mississippi",
}

STAT_FIELDS = (
    "conference",
    "t_rank",
    "barthag",
    "adj_offensive_efficiency",
    "adj_defensive_efficiency",
    "adj_tempo",
    "effective_fg_pct",
    "turnover_rate",
    "offensive_reb_rate",
    "free_throw_rate",
    "opp_effective_fg_pct",
    "opp_turnover_rate",
    "defensive_reb_rate",
    "opp_free_throw_rate",
)

# Bracket rounds in order. FF (First Four play-in) is excluded on purpose:
# the rest of this project scores a 63-game R64-onward bracket, so a play-in
# win is not a "round won" here.
BRACKET_ROUNDS = ("R64", "R32", "S16", "E8", "F4", "NCG")
FINISH_ON_LOSS = {
    "FF": "First Four",
    "R64": "Round of 64",
    "R32": "Round of 32",
    "S16": "Sweet 16",
    "E8": "Elite 8",
    "F4": "Final Four",
    "NCG": "Runner-up",
}


def _torvik_meta(year: int) -> dict:
    path = HIST_DIR / f"torvik_{year}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {"data_type": data.get("data_type"), "tournament_start": data.get("tournament_start")}


def build_regular_season_stats(year: int, canonical_ids, t_ranks, tournament_start):
    """Per-team regular-season volatility from the pre-tournament game log.

    Returns `{canonical_team_id: {...}}`. Only games strictly before
    `tournament_start` are counted — the raw log runs through the national
    championship in most years.
    """
    path = HIST_DIR / f"historical_games_{year}.json"
    if not path.exists() or not tournament_start:
        return {}
    with open(path) as f:
        games = json.load(f).get("games", [])

    # --- Build the cbbpy -> canonical map, then de-duplicate it. ---------
    # bridge_cbbpy_id falls back to longest-prefix matching, which happily
    # maps non-D1 schools onto a D1 team whose name they start with:
    # "virginia_union_panthers" -> "virginia", "arkansas_tech_wonder_boys" ->
    # "arkansas". Left alone that folds a D2 team's blowout losses into a
    # tournament team's volatility numbers.
    #
    # Disambiguate by schedule length: in a D1 game log the real team plays a
    # full ~30-game season while an impostor appears once or twice. Measured
    # on 2026 the gap is 28-33 games vs 1-4, so "most games wins" separates
    # them with a wide margin.
    appearances: dict[str, int] = {}
    for g in games:
        for key in ("team1_id", "team2_id"):
            raw = g.get(key)
            if raw:
                appearances[raw] = appearances.get(raw, 0) + 1

    candidates: dict[str, list[str]] = {}
    for raw in appearances:
        alias = _EXTRA_CBBPY_ALIASES.get(raw)
        canonical = alias if (alias is not None and alias in canonical_ids) else bridge_cbbpy_id(raw, canonical_ids)
        if canonical is not None:
            candidates.setdefault(canonical, []).append(raw)

    bridge_map: dict[str, str] = {}
    for canonical, raws in candidates.items():
        winner = max(raws, key=lambda r: appearances[r])
        bridge_map[winner] = canonical

    def bridge(raw_id):
        return bridge_map.get(raw_id)

    margins: dict[str, list[int]] = {}
    bad_losses: dict[str, int] = {}
    for g in games:
        date = g.get("date")
        if not date or date >= tournament_start:
            continue  # tournament (or post-tournament) game — not pre-tournament
        s1, s2 = g.get("team1_score"), g.get("team2_score")
        if s1 is None or s2 is None:
            continue
        t1, t2 = bridge(g.get("team1_id", "")), bridge(g.get("team2_id", ""))
        for team, opp, own_score, opp_score in ((t1, t2, s1, s2), (t2, t1, s2, s1)):
            if team is None:
                continue
            margins.setdefault(team, []).append(own_score - opp_score)
            # A "bad loss" needs both sides ranked, so non-D1 opponents
            # (which never bridge) are excluded from the numerator.
            if own_score < opp_score and opp is not None:
                own_rank, opp_rank = t_ranks.get(team), t_ranks.get(opp)
                if own_rank is not None and opp_rank is not None and opp_rank > own_rank:
                    bad_losses[team] = bad_losses.get(team, 0) + 1

    out = {}
    for team, vals in margins.items():
        n = len(vals)
        close = [m for m in vals if abs(m) <= CLOSE_GAME_MARGIN]
        out[team] = {
            "games_played": n,
            "reg_season_margin_avg": round(statistics.fmean(vals), 2),
            "reg_season_margin_std": round(statistics.pstdev(vals), 2) if n > 1 else None,
            "close_game_rate": round(len(close) / n, 4),
            "close_game_win_rate": (round(sum(1 for m in close if m > 0) / len(close), 4) if close else None),
            "losses_to_weaker_rate": round(bad_losses.get(team, 0) / n, 4),
        }
    return out


def build_tournament_outcomes(year: int):
    """Post-hoc: how far each team actually got. NOT pre-tournament data.

    Derived from the deepest round a team APPEARS in, not from per-game
    `team1_won` flags. Appearance is the more reliable signal: a bracket is
    single-elimination, so playing in the Sweet 16 proves you won two games
    no matter what the R32 record claims. This matters because the source
    data has at least one transposed game — `tournament_context_2018.json`
    records Cincinnati beating Nevada in the R32 while also showing Nevada
    (correctly) in the Sweet 16. Reality: Nevada won 75-73. A win-counting
    approach silently mislabels both teams there; this one self-corrects and
    prints a warning so the upstream bug stays visible.
    """
    games = load_tournament_results(year)
    if not games:
        return {}

    depth = {rnd: i for i, rnd in enumerate(BRACKET_ROUNDS)}  # R64=0 ... NCG=5
    deepest: dict[str, int] = {}
    losses: dict[str, list[str]] = {}
    ncg_winner = None
    for g in games:
        rnd, t1, t2 = g.get("round_name"), g.get("team1_id"), g.get("team2_id")
        if not t1 or not t2:
            continue
        d = depth.get(rnd, -1)  # FF (play-in) sits below R64 at -1
        for team in (t1, t2):
            deepest[team] = max(deepest.get(team, -1), d)
        if g.get("team1_won") is not None:
            loser = t2 if g["team1_won"] else t1
            losses.setdefault(loser, []).append(rnd)
            if rnd == "NCG":
                ncg_winner = t1 if g["team1_won"] else t2

    # Single-elimination: nobody can lose twice. More than one recorded loss
    # means the source file has a game whose result contradicts the bracket.
    for team, rounds in sorted(losses.items()):
        if len(rounds) > 1:
            print(
                f"    {year}: SOURCE DATA BUG — {team!r} recorded as losing "
                f"{len(rounds)} games ({', '.join(rounds)}); single-elimination "
                f"allows one. Using round-appearance instead."
            )

    out = {}
    for team, d in deepest.items():
        if d < 0:
            out[team] = {"outcome_finish": "First Four", "outcome_rounds_won": 0}
        elif d == depth["NCG"]:
            won = team == ncg_winner
            out[team] = {
                "outcome_finish": "Champion" if won else "Runner-up",
                "outcome_rounds_won": 6 if won else 5,
            }
        else:
            out[team] = {"outcome_finish": FINISH_ON_LOSS[BRACKET_ROUNDS[d]], "outcome_rounds_won": d}
    return out


def build_year_rows(year: int) -> list[dict]:
    seeds, regions = load_seeds_and_regions(year)
    if not seeds:
        return []  # no tournament that year (2020)

    meta = _torvik_meta(year)
    if meta.get("data_type") not in VALID_PRETOURNAMENT_TYPES:
        print(f"  {year}: SKIP — data_type={meta.get('data_type')!r}, not pre-tournament")
        return []

    torvik, _ff = load_torvik_and_ff(year)
    t_ranks = {tid: t.get("t_rank") for tid, t in torvik.items()}
    reg = build_regular_season_stats(year, set(torvik), t_ranks, meta.get("tournament_start"))
    outcomes = build_tournament_outcomes(year)

    rows = []
    for team_id, seed in seeds.items():
        # The seeds file and Torvik disagree on a few IDs; alias across so the
        # team keeps its stats instead of vanishing from the bracket.
        tv_id = _SEEDS_TO_TORVIK_ALIASES.get(team_id, team_id)
        t = torvik.get(tv_id)
        if not t:
            print(f"    {year}: no Torvik row for {team_id!r} — dropped")
            continue
        row = {
            "team_id": team_id,
            "team_name": t.get("team_name"),
            "seed": seed,
            "region": regions.get(team_id),
        }
        for field in STAT_FIELDS:
            row[field] = t.get(field)
        row.update(
            reg.get(
                tv_id,
                {
                    "games_played": None,
                    "reg_season_margin_avg": None,
                    "reg_season_margin_std": None,
                    "close_game_rate": None,
                    "close_game_win_rate": None,
                    "losses_to_weaker_rate": None,
                },
            )
        )
        row.update(outcomes.get(team_id, {"outcome_finish": None, "outcome_rounds_won": None}))
        rows.append(row)

    rows.sort(key=lambda r: r["t_rank"] if r["t_rank"] is not None else 999)
    return rows


def attach_seed_deltas(stats_by_year: dict) -> dict:
    """Add `outcome_vs_seed_delta` = rounds won minus the average for that seed.

    The baseline is computed across every year in this dataset, so it is a
    descriptive "how did this team do relative to a typical team on its seed
    line" — not a forecast. Like the other `outcome_*` fields it is post-hoc.
    """
    by_seed: dict[int, list[int]] = {}
    for rows in stats_by_year.values():
        for r in rows:
            if r.get("outcome_rounds_won") is not None:
                by_seed.setdefault(r["seed"], []).append(r["outcome_rounds_won"])
    expected = {seed: statistics.fmean(vals) for seed, vals in by_seed.items()}
    for rows in stats_by_year.values():
        for r in rows:
            exp = expected.get(r["seed"])
            if r.get("outcome_rounds_won") is None or exp is None:
                r["outcome_vs_seed_delta"] = None
            else:
                r["outcome_vs_seed_delta"] = round(r["outcome_rounds_won"] - exp, 2)
    return {seed: round(v, 3) for seed, v in sorted(expected.items())}


def attach_historical_residual(stats_by_year: dict) -> None:
    """Add `hist_residual` — a program's tournament over/under-performance to date.

    For team T in year Y: the mean of T's `outcome_vs_seed_delta` across its
    tournament appearances in years STRICTLY BEFORE Y. Because it only ever
    looks backwards, this is genuinely pre-tournament information — unlike
    the `outcome_*` block it sits alongside the other stat columns.

    This is the "does a program's March history tell you anything beyond its
    current seed and strength?" question, expressed as a residual rather than
    a raw count of Final Fours (which would just re-encode program strength).

    `hist_appearances` carries the sample size, which matters a lot here: the
    dataset starts in 2010, so early years have little or no prior history
    and a single appearance makes for a very noisy residual.

    The per-seed baseline inside `outcome_vs_seed_delta` is pooled across all
    years rather than walked forward. It is a structural constant (a 1-seed
    averages ~3.3 wins, a fact established over decades and not by this
    16-year window) and carries no team-specific information, so it leaks
    nothing about the team whose residual is being computed — while a
    walk-forward baseline would make 2011-2013 unusable.
    """
    history: dict[str, list[float]] = {}
    for year in sorted(stats_by_year, key=int):
        rows = stats_by_year[year]
        # Read history BEFORE folding this year in, so a team never sees itself.
        for r in rows:
            past = history.get(r["team_id"], [])
            r["hist_appearances"] = len(past)
            r["hist_residual"] = round(statistics.fmean(past), 2) if past else None
        for r in rows:
            if r.get("outcome_vs_seed_delta") is not None:
                history.setdefault(r["team_id"], []).append(r["outcome_vs_seed_delta"])


def main() -> None:
    stats_by_year = {}
    for year in YEARS:
        rows = build_year_rows(year)
        if rows:
            stats_by_year[str(year)] = rows
            with_reg = sum(1 for r in rows if r["games_played"])
            print(f"  {year}: {len(rows)} teams ({with_reg} with game-log stats)")

    expected_by_seed = attach_seed_deltas(stats_by_year)
    attach_historical_residual(stats_by_year)

    payload = {
        "years": sorted(int(y) for y in stats_by_year),
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "expected_rounds_won_by_seed": expected_by_seed,
        "stats_by_year": stats_by_year,
    }

    out_path = OUT / "team_stats_by_year.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"\nWrote {out_path} ({len(payload['years'])} years)")


if __name__ == "__main__":
    main()
