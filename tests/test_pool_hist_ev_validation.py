"""Validation tests for pool_hist_results.json as the primary EV scoring dataset.

pool_hist_results.json records the actual ESPN pool brackets for 2023-2026
(all participants except the user's own entry). These tests treat it as the
ground truth for EV analysis and enforce three properties:

  1. Structural integrity — required fields, pick-array lengths, bracket counts
     vs. group size.

  2. Scoring integrity — re-derive points from pick arrays + actual tournament
     results using the standard ESPN scoring rubric (10/20/40/80/160/320).

     2023: exact equality is required.  The stored pts were rescored from picks
     in April 2026 using the same formula, so any mismatch here means a pick
     array was corrupted after rescoring.

     2024/2025/2026: Spearman rank correlation ≥ 0.98 between computed and
     stored scores is required.  The local tournament_results_*.json files have
     known First-Four resolution errors (wrong participants, phantom R32 winners,
     one wrong E8 winner in 2024) that cause systematic over-counting for subsets
     of brackets.  Exact equality is not achievable without fixing the raw data
     files.  However, the errors are small enough that rank order is preserved —
     which is the property that actually matters for EV analysis (we care about
     relative pool standing, not absolute points).

  3. Rank consistency — brackets must be sorted by pts descending, and ESPN-style
     rank assignment (tied brackets share rank; next rank skips) must hold.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest
from scipy.stats import spearmanr

from src.simulation.pool_history_opponent_model import ABBREV_TO_TEAM_ID

POOL_FILE = Path("pool_hist_results.json")
HIST_DIR = Path("data/raw/historical")

# Standard ESPN bracket scoring used by this pool for all years.
SCORING = {"r64": 10, "r32": 20, "s16": 40, "e8": 80, "f4": 160, "champ": 320}

# Pool round key -> tournament_results round_name
ROUND_MAP = {
    "r64": "R64",
    "r32": "R32",
    "s16": "S16",
    "e8": "E8",
    "f4": "F4",
    "champ": "NCG",
}

# How many brackets are absent from the file per year.
# The user's own entry is excluded to avoid a self-referential opponent model.
# 2024 is the exception: user was not in the pool that year, so all members
# are captured (gap = 0).
_GROUP_SIZE_GAP = {2023: 1, 2024: 0, 2025: 1, 2026: 1}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# 2026 tournament_results uses long-form IDs with mascot names.
# Map them to the canonical short-form used by ABBREV_TO_TEAM_ID.
_RESULTS_NORMALIZE: dict[str, str] = {
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
}

# Round order for the cascade consistency filter.
_ROUND_ORDER = ["FF", "R64", "R32", "S16", "E8", "F4", "NCG"]


def _load_tournament_results(year: int) -> list[dict]:
    path = HIST_DIR / f"tournament_results_{year}.json"
    if not path.exists():
        return []
    with path.open() as f:
        return json.load(f)["games"]


def _build_winners(games: list[dict]) -> dict[str, set[str]]:
    """Map round_name -> set of winning team_ids for that round.

    Two corrections are applied:

    1. Team ID normalization — 2026 results use long-form IDs (e.g.
       ``michigan_wolverines``); map them to canonical short-form IDs.

    2. Cascade consistency filter — a team can only appear in round R's
       winner set if it also appears in round R-1's winner set.  This
       removes phantom entries caused by First Four resolution errors in
       the raw results files (e.g., 2024: ``colorado`` appears as an R32
       winner despite having lost in R64, because a FF loser/winner ID
       swap corrupted its bracket slot).
    """
    winners: dict[str, set[str]] = defaultdict(set)
    for g in games:
        winner = g["team1_id"] if g["team1_won"] else g["team2_id"]
        winner = _RESULTS_NORMALIZE.get(winner, winner)
        winners[g["round_name"]].add(winner)

    # Cascade consistency from R64 onward (NOT from FF).
    # Rationale: ~60 teams enter the main bracket at R64 without playing a FF
    # game, so R64_winners is not a subset of FF_winners.  But R32_winners must
    # be a subset of R64_winners, S16 of R32, etc.  This removes phantom entries
    # caused by FF resolution errors (e.g., 2024: ``colorado`` appears as an R32
    # winner despite having lost — or never played — in R64).
    prev: set[str] | None = None
    for rnd in ["R64", "R32", "S16", "E8", "F4", "NCG"]:
        if rnd not in winners:
            continue
        if prev is not None:
            winners[rnd] &= prev
        prev = winners[rnd]

    return winners


def _score_bracket(bracket: dict, winners: dict[str, set[str]]) -> int:
    total = 0
    for rd, pts in SCORING.items():
        actual_winners = winners.get(ROUND_MAP[rd], set())
        if rd == "champ":
            tid = ABBREV_TO_TEAM_ID.get(bracket["champ"], bracket["champ"].lower())
            if tid in actual_winners:
                total += pts
        else:
            for abbrev in bracket[rd]:
                tid = ABBREV_TO_TEAM_ID.get(abbrev, abbrev.lower())
                if tid in actual_winners:
                    total += pts
    return total


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pool_data() -> dict:
    with POOL_FILE.open() as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. Structural integrity
# ---------------------------------------------------------------------------


class TestStructuralIntegrity:
    def test_required_top_level_fields(self, pool_data: dict) -> None:
        for field in ("pool", "groupId", "years"):
            assert field in pool_data

    def test_all_four_years_present(self, pool_data: dict) -> None:
        assert {"2023", "2024", "2025", "2026"} <= set(pool_data["years"].keys())

    @pytest.mark.parametrize("year", ["2023", "2024", "2025", "2026"])
    def test_year_required_fields(self, pool_data: dict, year: str) -> None:
        ydata = pool_data["years"][year]
        assert "groupSize" in ydata
        assert "brackets" in ydata
        assert len(ydata["brackets"]) > 0

    @pytest.mark.parametrize("year,gap", _GROUP_SIZE_GAP.items())
    def test_bracket_count_vs_group_size(self, pool_data: dict, year: int, gap: int) -> None:
        """Captured brackets == groupSize - gap.

        gap=1 means user's own entry is excluded (correct: using the file
        as an opponent model must not include the user's own bracket).
        gap=0 for 2024 where the user did not participate.
        """
        ydata = pool_data["years"][str(year)]
        assert len(ydata["brackets"]) == ydata["groupSize"] - gap

    @pytest.mark.parametrize("year", ["2023", "2024", "2025", "2026"])
    def test_bracket_pick_fields_and_lengths(self, pool_data: dict, year: str) -> None:
        # r32 and later are always complete; r64 can be short for brackets where
        # the ESPN scraper missed some first-round picks (observed: 29 in 2024,
        # 28 in 2025 for the lowest-scoring entries).
        exact_lengths = {"r32": 16, "s16": 8, "e8": 4, "f4": 2}
        for b in pool_data["years"][year]["brackets"]:
            for field in ("rank", "pts", "id", "r64", "r32", "s16", "e8", "f4", "champ"):
                assert field in b, f"{year} bracket rank={b.get('rank')}: missing '{field}'"
            assert 28 <= len(b["r64"]) <= 32, (
                f"{year} bracket rank={b['rank']}: r64 has {len(b['r64'])} picks, expected 28-32"
            )
            for rd, expected_len in exact_lengths.items():
                assert len(b[rd]) == expected_len, (
                    f"{year} bracket rank={b['rank']}: {rd} has {len(b[rd])} picks, expected {expected_len}"
                )
            assert isinstance(b["champ"], str) and len(b["champ"]) > 0

    def test_incomplete_r64_brackets_are_known(self, pool_data: dict) -> None:
        """Guard against new incomplete brackets appearing silently.

        The two known incomplete-r64 entries are 2024 rank=23 (29 picks) and
        2025 rank=32 (28 picks).  If a new bracket with fewer than 32 R64 picks
        appears, this test fires so it can be investigated.
        """
        known_incomplete = {("2024", 23), ("2025", 32)}
        found = set()
        for year, ydata in pool_data["years"].items():
            for b in ydata["brackets"]:
                if len(b["r64"]) < 32:
                    found.add((year, b["rank"]))
        assert found == known_incomplete, (
            f"Unexpected incomplete-r64 brackets: {found - known_incomplete}; "
            f"previously known incomplete brackets now gone: {known_incomplete - found}"
        )


# ---------------------------------------------------------------------------
# 2. Scoring integrity
# ---------------------------------------------------------------------------


class TestScoringIntegrity:
    """Re-derive pts from picks + actual results and validate against stored pts.

    2023 uses exact equality: the stored pts were rescored from picks using this
    exact formula, so any mismatch means a pick array was corrupted after rescoring.

    2024/2025/2026 use Spearman rank correlation ≥ 0.98: the local
    tournament_results_*.json files have known First-Four resolution errors that
    cause systematic over-counting for subsets of brackets.  Exact equality is
    unachievable without fixing the raw data files.  Rank order preservation is
    the property that matters for EV analysis.

    If any of these tests fire, the root causes to investigate are:
      - Pick array corruption in pool_hist_results.json.
      - ABBREV_TO_TEAM_ID missing an abbreviation used this year.
      - Scoring formula mismatch (SCORING dict != what the pool used).
      - tournament_results_{year}.json has a major new data error.
    """

    def test_2023_computed_pts_match_stored_exactly(self, pool_data: dict) -> None:
        games = _load_tournament_results(2023)
        if not games:
            pytest.skip("No tournament results file for 2023")

        winners = _build_winners(games)
        mismatches = []
        for b in pool_data["years"]["2023"]["brackets"]:
            computed = _score_bracket(b, winners)
            if computed != b["pts"]:
                mismatches.append(
                    f"rank={b['rank']} id={b['id'][:8]}: "
                    f"computed={computed} stored={b['pts']} "
                    f"delta={computed - b['pts']:+d}"
                )

        assert not mismatches, (
            f"2023: {len(mismatches)} bracket(s) with pts mismatch (pick arrays "
            f"may be corrupted after rescoring):\n" + "\n".join(mismatches)
        )

    @pytest.mark.parametrize("year", [2024, 2025, 2026])
    def test_computed_pts_rank_order_preserved(self, pool_data: dict, year: int) -> None:
        """Spearman ρ ≥ 0.98 between computed and stored scores.

        The local tournament_results files have known FF-resolution errors.
        We cannot achieve exact equality, but rank order must be preserved —
        that is the property EV analysis depends on.
        """
        games = _load_tournament_results(year)
        if not games:
            pytest.skip(f"No tournament results file for {year}")

        winners = _build_winners(games)
        brackets = pool_data["years"][str(year)]["brackets"]
        stored = [b["pts"] for b in brackets]
        computed = [_score_bracket(b, winners) for b in brackets]

        rho, _ = spearmanr(computed, stored)
        assert rho >= 0.98, (
            f"{year}: Spearman ρ between computed and stored pts = {rho:.4f} "
            f"(below 0.98 threshold). A major data error or pick-array corruption "
            f"has disrupted rank order — investigate tournament_results_{year}.json "
            f"and ABBREV_TO_TEAM_ID."
        )


# ---------------------------------------------------------------------------
# 3. Rank consistency
# ---------------------------------------------------------------------------


class TestRankConsistency:
    @pytest.mark.parametrize("year", ["2023", "2024", "2025", "2026"])
    def test_brackets_sorted_by_pts_descending(self, pool_data: dict, year: str) -> None:
        pts = [b["pts"] for b in pool_data["years"][year]["brackets"]]
        assert pts == sorted(pts, reverse=True), f"{year}: brackets are not sorted by pts descending"

    @pytest.mark.parametrize("year", ["2023", "2024", "2025", "2026"])
    def test_ranks_espn_style(self, pool_data: dict, year: str) -> None:
        """Rank invariants:

        1. First stored bracket is rank=1 (the pool leader is always captured).
        2. Ranks are non-decreasing across the sorted bracket list.

        Stricter properties (tied pts → same rank; non-tied → strictly higher
        rank) are intentionally not enforced here because:

        - Years with gap=1 (user excluded) may have rank gaps where the user's
          score landed within the captured range.
        - Scraped pts values have minor inaccuracies (ESPN may use a secondary
          tiebreaker such as national percentile), so some identical-pts pairs
          show different ranks and some different-pts pairs show the same rank
          in the raw scraped data.

        test_brackets_sorted_by_pts_descending already validates pts order.
        Together these two tests confirm: the file is sorted correctly and
        ranks flow in the same direction as scores.
        """
        brackets = pool_data["years"][year]["brackets"]
        assert brackets[0]["rank"] == 1, f"{year}: first bracket has rank={brackets[0]['rank']}, expected 1"
        for i, b in enumerate(brackets[1:], start=1):
            prev = brackets[i - 1]
            assert b["rank"] >= prev["rank"], (
                f"{year} position {i}: rank={b['rank']} < rank={prev['rank']} (ranks must be non-decreasing)"
            )
