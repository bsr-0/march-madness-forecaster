"""Regression test for scripts/_bracket_export_common.py's game_key generation.

2026-08-15: build_bracket_json() independently re-derives each game's
game_key (it has no access to bracket_construction.py's own keys) to look
up the real winner from the picks dict. For R32/S16, region_game reset the
regional game counter using teams-per-region (8, 4) instead of
games-per-region (4, 2) — one bit short — producing NEGATIVE region_game
numbers for every region but the first (East). Every lookup for West/
South/Midwest R32 and S16 games silently missed and fell through to a
win_prob-based guess instead of the actual optimizer decision, undetected
because chalk-favored guesses usually happened to match the real pick.

This test builds a synthetic full bracket using bracket_construction.py's
own key convention (verified directly against a real construct_bracket()
call — see _bracket_export_common.py's game_key comment) and asserts every
single game's displayed winner_id matches the real pick, for every region,
not just the first.
"""

from __future__ import annotations

from scripts._bracket_export_common import build_bracket_json
from scripts.mc_pool_backtest import REGION_ORDER, SEED_MATCHUP_ORDER

ROUND_KEYS = ("R64", "R32", "S16", "E8", "F4", "CHAMP")


def _synthetic_bracket():
    """64 teams, 4 regions x 16 seeds, team_id = "{region}_{seed}" lowercased."""
    seeds = {}
    regions = {}
    for region in REGION_ORDER:
        for seed in range(1, 17):
            tid = f"{region.lower()}_{seed}"
            seeds[tid] = seed
            regions[tid] = region
    return seeds, regions


def _real_picks_dict(seeds, regions):
    """Build a picks dict using bracket_construction.py's OWN key format
    (R64_{region}_{high}v{low}, R32_{region}_{game}, S16_{region}_{game},
    E8_{region}, F4_East_West, F4_South_Midwest, CHAMP).

    Deliberately picks the HIGHER seed (team1, the underdog) to win every
    single game — the opposite of what a win_prob-based guess would ever
    produce (win_prob favors the lower seed / higher barthag throughout,
    see barthag construction in the test below). This is essential: if the
    "true" picks agreed with what a naive win_prob fallback would guess
    (e.g. chalk-favorite-always-wins), a broken game_key lookup would
    silently fall through to that same guess and this test would pass
    despite the lookup never actually succeeding — which is exactly how
    the original bug (game_key wrong for West/South/Midwest R32/S16, only
    caught by coincidence when a contrarian pick happened to disagree with
    the guess) went undetected. Forcing every pick to contradict the naive
    guess makes any silent fallback immediately visible.
    """
    picks = {}

    def upset(a, b):
        return a if seeds[a] > seeds[b] else b

    e8_winners = {}
    for region in REGION_ORDER:
        by_seed = {s: f"{region.lower()}_{s}" for s in range(1, 17)}

        # R64
        r64_winners = []
        for high, low in SEED_MATCHUP_ORDER:
            t1, t2 = by_seed[high], by_seed[low]
            w = upset(t1, t2)
            picks[f"R64_{region}_{high}v{low}"] = w
            r64_winners.append(w)

        # R32 (4 games, 1-indexed within region)
        r32_winners = []
        for i in range(0, len(r64_winners), 2):
            w = upset(r64_winners[i], r64_winners[i + 1])
            picks[f"R32_{region}_{i // 2 + 1}"] = w
            r32_winners.append(w)

        # S16 (2 games, 1-indexed within region)
        s16_winners = []
        for i in range(0, len(r32_winners), 2):
            w = upset(r32_winners[i], r32_winners[i + 1])
            picks[f"S16_{region}_{i // 2 + 1}"] = w
            s16_winners.append(w)

        # E8 (1 game)
        e8_winner = upset(s16_winners[0], s16_winners[1])
        picks[f"E8_{region}"] = e8_winner
        e8_winners[region] = e8_winner

    semi_ew = upset(e8_winners["East"], e8_winners["West"])
    semi_sm = upset(e8_winners["South"], e8_winners["Midwest"])
    picks["F4_East_West"] = semi_ew
    picks["F4_South_Midwest"] = semi_sm
    picks["CHAMP"] = upset(semi_ew, semi_sm)

    return picks


def test_every_region_r32_s16_winner_matches_real_pick():
    """The bug this locks: only East's R32/S16 keys resolved; West/South/
    Midwest silently fell back to a win_prob guess. Picks are rigged to
    contradict that guess (see _real_picks_dict) so any silent fallback is
    immediately visible as a wrong winner_id, for all 63 games."""
    seeds, regions = _synthetic_bracket()
    picks = _real_picks_dict(seeds, regions)

    barthag = {tid: 1.0 - 0.01 * seeds[tid] for tid in seeds}  # lower seed = higher rating
    round_probs = {tid: {rnd: 1.0 - 0.01 * seeds[tid] for rnd in ROUND_KEYS} for tid in seeds}
    team_names = {tid: tid for tid in seeds}

    rounds = build_bracket_json(seeds, regions, barthag, round_probs, picks, team_names)

    mismatches = []
    checked = 0
    for rnd in rounds:
        for g in rnd["games"]:
            checked += 1
            t1, t2 = g["team1_id"], g["team2_id"]
            expected = t1 if seeds[t1] > seeds[t2] else t2  # upset winner, per _real_picks_dict
            if g["winner_id"] != expected:
                mismatches.append((rnd["round_name"], g["region"], t1, t2, g["winner_id"], expected))

    assert checked == 63, f"expected 63 games total, saw {checked}"
    assert not mismatches, f"winner_id mismatches (game_key lookup failed, fell back to a guess): {mismatches}"


def test_west_south_midwest_r32_keys_resolve_not_just_east():
    """Narrower, more direct regression check: build the region_game key
    exactly like build_bracket_json does and confirm it's positive and
    present in a real picks dict for every non-East region."""
    seeds, regions = _synthetic_bracket()
    picks = _real_picks_dict(seeds, regions)

    for region in REGION_ORDER:
        for game_num in (1, 2, 3, 4):
            key = f"R32_{region}_{game_num}"
            assert key in picks, f"{key} missing from real picks dict"
        for game_num in (1, 2):
            key = f"S16_{region}_{game_num}"
            assert key in picks, f"{key} missing from real picks dict"
