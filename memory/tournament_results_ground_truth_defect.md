# Ground-truth defect: 8 transposed games in tournament_context_{year}.json

**Investigated 2026-08-18. Confirmed. NOT yet fixed — fixing changes backtest
ground truth and therefore every recorded baseline, so it needs a decision.**

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

## Why this is not fixed yet — decision needed

Correcting the data is a two-line-per-game edit, but it **invalidates every
recorded baseline**: the 11.3% P(1st), MeanRank 11.6, "15/15 years beating
seed", the p<0.0001 significance claim (`MEMORY.md` §3), the numbers baked into
`docs/app.js`, and the strategy comparison table in `CLAUDE.md`. All were
measured against the corrupted truth for 6 of 15 years.

So a fix is really a three-part job:
1. Correct the eight games (and optionally 2025's loser identity).
2. Re-run the canonical 15-year backtest — which needs explicit approval per
   `memory/run_policy.md`.
3. Update every published figure that moves.

**Do not** do (1) without (2) and (3), or the repo will carry numbers that match
neither the old truth nor the new one.

## Guardrail

`scripts/audit_tournament_results.py` exits non-zero while any violation stands.
Wire it into CI **after** the repair, so it prevents regression rather than
failing the build on a known defect.
