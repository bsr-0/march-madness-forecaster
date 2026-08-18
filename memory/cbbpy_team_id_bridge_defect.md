# cbbpy team-ID bridge: prefix-collision defect

**Found and fixed 2026-08-18.** Surfaced while building the team-stats table, where
Virginia 2026 read a −7.1 scoring margin with a 37.5-point standard deviation.

## What was wrong

`bridge_cbbpy_id` maps a cbbpy ID (`illinois_fighting_illini`) onto a canonical Torvik /
seeds ID (`illinois`) by longest-prefix match. That is only correct if the set it matches
against contains every school that could out-match. Callers were passing the **68
tournament teams**, so two classes of ID landed on the wrong team:

1. **Other D1 schools.** `alabama_state_hornets` prefix-matches `alabama` because
   `alabama_state` is not among the 68. On 2026, **19 of 68** seeded teams collected at
   least one such ID. Not fixable by weighting — Alabama State played 32 games to
   Alabama's 31, and Florida Gulf Coast played 34 to Florida's 31.
2. **Non-D1 schools.** `virginia_lynchburg_dragons`, `arkansas_baptist_buffaloes` — real
   in the game log, in no canonical set. **13 of 68** on 2026.

Each consumer then folded the impostor's data in differently: `volatile` appended its
margins, `roster_adj` and the stats table silently **overwrote** the real team's entry.

## The fix

`resolve_cbbpy_bridge(weighted_ids, canonical_ids, universe=...)` in `src/data/normalize.py`
resolves a whole collection at once, applying both halves:

- **Universe** — match against the full D1 field (`load_d1_team_ids(year, data_root)`), so
  `alabama_state_hornets` finds `alabama_state` and drops out of the 68-team target. Kills
  class 1; nothing else does.
- **Weight** — whatever survives onto one canonical ID is decided by whichever candidate
  carries the most data (games played, roster player-games, Elo). Kills class 2: across all
  13 collisions in 2026 the real team led by ≥2.9x. Ties break on the ID string, so the
  result never depends on dict ordering.

Also swept every seeded team across 2011–2026 for ones bridging to nothing, and added the
12 they turned up to `_CBBPY_EDGE_CASES` (`ole_miss_rebels`, `lsu_tigers`, `usc_trojans`,
`unlv_rebels`, `ualbany_great_danes`, `charleston_cougars`, `loyola_chicago_ramblers`,
`loyola_maryland_greyhounds`, `app_state_mountaineers`, `mount_st_mary_s_mountaineers`,
`sam_houston_bearkats`, `southern_miss_golden_eagles`). Before this, `mississippi`
prefix-matched `mississippi_state_bulldogs` — **Ole Miss was running on Mississippi State's
roster and game log.**

## Measured impact (2011–2026, 1020 team-years)

| Source | Values changed | Notes |
|---|---|---|
| `volatile` (D1) | **713 / 1020 (70%)** | Kentucky 2023 went 0.985 → 0.176; it was never the noisiest team in the country |
| `roster_adj` (C2) | 156 / 1020 (15%) | 28 teams newly resolved, 0 lost |
| `elo` (A4) | 43 / 1020 (4%) | All 43 from the new edge cases. The universe/weight fix alone was a **verified no-op** — its pre-existing highest-Elo tiebreak already landed on the right team in all 1020 |
| stats table roster cols | 101 team-years | Texas 2010 went from an implausible 100% freshman minutes to 32.9% |
| stats table volatility cols | **0** | That script already bridged against the full Torvik list, so it only ever had class 2, and its "most games wins" rule already handled it |

## What this does NOT affect

**The production strategy `meta_region_poolaware` does not use any of these sources.** It
runs on `tv` / `mass_avg` / `mass_best` / `blend` / `tv_mass80`, none of which touch the
bridge. The 11.2% P(1st) baseline re-established 2026-08-18 against repaired ground truth
is unaffected and did not need re-running.

## Guardrail

`tests/test_team_id_bridge.py` pins all of it: the D1-vs-D1 case asserts both the correct
result *and* that dropping the universe reproduces the bug, so the universe argument
cannot be quietly removed. The edge-case test pins Ole Miss beating Mississippi State for
`mississippi` even when the state school carries more data.
