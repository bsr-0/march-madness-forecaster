# Ground-truth defect: 8 transposed games in tournament_context_{year}.json

**Investigated and FIXED 2026-08-18.** The repair moved P(1st) only
11.3%→11.2% (MeanRank 11.6→11.4, significance strengthened t 9.219→11.002) —
a provenance correction, not a performance revision. CLAUDE.md's current
baseline table reflects the repaired ground truth. The analysis below is kept
for the record of what was wrong and why the fix was low-risk.

Run `python3 scripts/audit_tournament_results.py` to reproduce.

## What's wrong

`tournament_context_{year}.json` → `results.games[]` is the backtest's ground
truth. `actual_winners_by_round` and `build_actual_outcome` read `team1_won`
straight out of it, so a transposed game silently mis-scores **every** bracket
in that year's pool — the model's and all 30 opponents'.

Eight games record the **wrong winner**. In each, the losing team's score is
also corrupted, so this is not a clean field swap; it looks like a join that
kept the winning score but attached it to the wrong team.

## The eight, each confirmed by an independent scrape

Three checks agree on all eight: (A) the team is recorded as losing twice, which
single-elimination forbids; (B) the recorded winner never appears in the next
round; (C) `historical_games_{year}.json` — a completely separate cbbpy scrape —
gives the opposite winner.

| Year | Round | File says | **Truth** (game log) | Pts |
|---|---|---|---|---|
| 2018 | R32 | Cincinnati (2) won 75-73 | **Nevada (7) won 75-73** | 20 |
| 2019 | S16 | Houston (3) won 62-58 | **Kentucky (2) won 62-58** | 40 |
| 2019 | E8 | Duke (1) won 68-67 | **Michigan St (2) won 68-67** | 80 |
| 2021 | S16 | Alabama (2) won 88-51 | **UCLA (11) won 88-78** | 40 |
| 2022 | R32 | Texas (6) won 81-69 | **Purdue (3) won 81-71** | 20 |
| 2022 | S16 | Gonzaga (1) won 82-68 | **Arkansas (4) won 74-68** | 40 |
| 2023 | R32 | Duke (5) won 66-65 | **Tennessee (4) won 65-52** | 20 |
| 2024 | R32 | Colorado (10) won 81-78 | **Marquette (2) won 81-77** | 20 |

Every one of these is a well-known result (2018 is Loyola-Chicago's Final Four
run; 2022 R32/S16 is Saint Peter's beating Purdue in the *next* round; 2021 is
UCLA's 11-seed run to the Gonzaga buzzer-beater).

**Affected: 6 of the 15 backtest years** (2018, 2019×2, 2021, 2022×2, 2023, 2024).
2010-2017, 2025 and 2026 are clean.

## Scope is exactly eight — verified, not assumed

Invariants A and B are independent, and they flag the **same eight games with no
residue**. Round sizes (C) are correct in all 16 years. So there is no
additional silent corruption of the kind where a wrong winner would vanish
without a trace.

## Separately: 2025 R64 names the wrong loser (harmless)

`mississippi 72-65 san_diego_state` is recorded, but SDSU lost the First Four to
North Carolina 95-68 and never reached the R64 — the game log shows **UNC**
playing that R64 game and losing 64-71. So the file has the wrong loser *and*
slightly wrong scores.

**No backtest impact.** Only winners are scored, and `mississippi` is recorded
correctly as the winner; neither SDSU nor UNC appears in any winner set. This is
cosmetic — it swaps the two teams' `outcome_finish` labels in the stats table
(both have `outcome_rounds_won = 0` either way).

## Backtest impact

280 points of ESPN scoring are attached to the wrong team across 6 years. Not a
uniform shift — it changes *which* brackets score well, since every bracket in
the pool is scored against the same wrong truth.

Direction is mixed but not balanced by value:

- **Corrupted data favours chalk — 180 pts:** 2018 R32 (2-seed credited over a
  7), 2019 E8 (1 over 2), 2021 S16 (2 over 11), 2022 S16 (1 over 4).
- **Corrupted data favours the underdog — 100 pts:** 2019 S16, 2022 R32,
  2023 R32, 2024 R32.

The chalk-favouring errors sit in the higher-value rounds (E8=80, S16=40) while
the upset-favouring ones are mostly R32=20. Since opponent brackets are
chalk-heavy and this strategy's edge comes from correct early-round picks
against the field, the recorded numbers **may understate** true P(1st). That is
a hypothesis about direction, not a measurement — quantifying it requires a
backtest re-run.

## Resolution (was: "not fixed yet — decision needed")

This section originally flagged the fix as a three-part job requiring
approval before running: (1) correct the eight games, (2) re-run the
canonical 15-year backtest, (3) update every published figure that moves.
All three happened 2026-08-18 — CLAUDE.md's baseline table, `docs/app.js`,
and this repo's other published figures now reflect the repaired ground
truth (11.2% P(1st), 15/15 years, p<0.0001).

## Guardrail

`scripts/audit_tournament_results.py` exits non-zero while any violation stands.
Wire it into CI **after** the repair, so it prevents regression rather than
failing the build on a known defect.
