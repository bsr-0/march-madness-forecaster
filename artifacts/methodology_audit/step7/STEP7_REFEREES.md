# Step 7 — Referee construction and leakage

Evidence: `step7_math.py` output (referee tables, topology, coupling on the
real 2026 field), calibration recomputation on 2013/2019/2025,
`tests/test_referee_validity.py`, the referee inventory (this file §1–§3),
`artifacts/referee_audit/` (the published runs, NOT regenerated here — see §12).

## 1. Referee inventory

| referee | source → inputs | transformation → formula | fitting window for season Y | role |
|---|---|---|---|---|
| seed | Kaggle seeds + compact results, R64+ (DayNum ≥ 136), 2010+ | cell win rate shrunk toward logistic(0.175·Δseed) with weight n/(n+8); cells < 8 games → logistic | **walk-forward** 2010..Y−1 (`as_of=year`, Step 2 P3-1); 2011 is 100% logistic, 2013 45%, 2016+ ≤ 23% (D7-3 diagnostics) | selection referee, criterion referee, incumbent, opponent fallback |
| torvik | `torvik_{Y}.json`, `data_type` guarded `pre_tournament` | barthag → Log5 | none (pre-tournament snapshot of Y) | criterion referee AND candidate base |
| blend | seed (above) + noseed model (12 Torvik features, LR + GBM, trained on prior tournaments) | 0.5·seed + 0.5·noseed pairwise | walk-forward both halves (Step 3 F3-4) | criterion referee AND candidate base; the shipped rule |
| pit | `training.json` (Torvik z-features, tournament rows) | ridge margin → Student-t, causal calibration | walk-forward rows y<Y | criterion referee; never a base; independent referee in the published run |
| market (v1) | `unified_odds_{Y}.json` before Mar 15 | BT on implied home prob, no ID resolution, spread-sign guard | season Y regular season only | DISQUALIFIED (construction); kept for reproduction only |
| market_v2 | same file | curated ID resolution, HCA −0.875 logit for non-neutral, BT, r/(r+1) → Log5 | same | PROVISIONAL in the published run |
| odds_api | multi-book closing consensus, 2021–2025 | BT, no venue handling | season Y pre-tournament | referee-only, partial coverage |
| fte | 538 power ratings (Kaggle bundle) | Φ((r_A−r_B)/11) | none | supplementary; **provenance unverified** (D7-2) |
| actual | real winners | — | — | supplementary |

Massey referees are candidate bases only, not referees.

## 2. Dependency graph (what is shared)

- blend ⊃ seed by construction (0.5 weight) and uses Torvik features.
- torvik, blend, pit all consume the same Torvik snapshots; pit and blend's
  noseed half train on the same historical tournament rows.
- torvik and blend are also candidate CONSTRUCTION bases; seed generates
  every opponent field.
- market ≡ market_v2 data; odds_api is a different file of the same market.
- fte is a distinct source.

Logit-scale correlation of the pairwise tables over the 2026 field:

| | seed | torvik | blend | pit | market | market_v2 | odds_api |
|---|---|---|---|---|---|---|---|
| seed | 1 | .957 | .983 | .871 | .768 | .749 | .835 |
| torvik | | 1 | .988 | .919 | .800 | .764 | .882 |
| blend | | | 1 | .913 | .787 | .759 | .865 |
| pit | | | | 1 | .744 | .722 | .769 |
| market | | | | | 1 | .970 | .706 |
| market_v2 | | | | | | 1 | .729 |

Classification: seed / torvik / blend / pit — **derived or partially
independent** of one another (one data family, blend contains seed); market
family — **independent source**; fte — independent source, unverified
timing.

## 3. Source / coverage / fallback

| referee | missing-team behaviour | recorded? | measured fallback |
|---|---|---|---|
| seed | logistic curve for thin cells | now yes (D7-3) | 63/63 games in 2011, 28/63 in 2013, ≤ 9 from 2019 |
| torvik | seed proxy max(0.10, 1−0.04·seed) | log only | 0–1 teams/season |
| blend | per-feature defaults inside noseed | no | — |
| pit | referee dropped for the season | yes | 14/14 present |
| market v1 | seed proxy | log only | 27–33 of 68 teams/season through 2022 |
| market_v2 | seed proxy | yes | 0–2 teams/season |
| odds_api | seed proxy | log only | 0–2 teams/season, 5 seasons |
| fte | referee dropped | yes | 7 seasons |

No referee falls back to the production model. 2025 has 62 scored games
in every referee (one results row missing).

## 4. Fitting windows / chronology (items 3, 4, 14)

All fitted referees are structurally walk-forward after Steps 2–3 (seed
`as_of`, noseed `as_of`, pit `y<Y` with causal link). Market referees use
only games dated before March 15 of season Y. Torvik snapshots are
provenance-gated. No referee reads season Y tournament games. **The
published artifacts (`FINDINGS*.md`, `qualification.json`, matrices) were
produced before these corrections**: their seed and blend referees contained
season Y. Recomputed calibration on three seasons:

| season | seed old → now | blend old → now | torvik | pit | market_v2 |
|---|---|---|---|---|---|
| 2013 | .6097 → .6100 | .5712 → .5727 | = | = | = |
| 2019 | .4949 → .5225 | .4888 → .5022 | = | = | = |
| 2025 | .4216 → .4357 | .4229 → .4311 | +.002 | +.006 | = |

The in-sample incumbent was up to 0.028 log loss better than its honest
walk-forward self. Because gate G2 compares challengers to the incumbent,
every published QUALIFIED/PROVISIONAL status is stale in the direction that
favoured the incumbent and penalised the only independent-source referee.

## 5–7. Probability mathematics

Verified on the real 2026 field for all seven available referees:
max |p(a,b)+p(b,a)−1| ≤ 1e-16, all probabilities strictly inside (0, 1).
FTE link reproduced independently (Φ of the rating difference over 11);
Bradley–Terry rating r → barthag r/(r+1) → Log5 equals r_A/(r_A+r_B)
exactly (test, 200 random pairs), so no double transformation. Market
home-court: logit(p_home) − 0.875 for non-neutral games, the correct
direction; 0.875 = 3.5 pts / 4 is an assumption carried from `spread_power`.
Spread sign: v2 and odds_api use implied probability only; v1's sign guard
inverted the SBRO convention (29–45% agreement) and is retired from runtime
(D7-1). torvik and pit tables exceed 0.99 for 5 and 43 ordered pairs and are
capped by the simulator (Step 3 F3-5, referee design).

## 8. Independence classification

- Independent: market_v2, odds_api (market source); fte (unverified timing).
- Partially independent: pit (different model, same feature family, same
  historical rows as blend's noseed half; never a base).
- Derived: torvik (a candidate base), blend (contains seed; a candidate base).
- Incumbent: seed (selection referee and opponent generator).

The published robustness claim ("robust to the qualified referees, including
held-out selection") rests on {seed, torvik, blend, pit}: three of the four
are one data family and two are inside candidate construction, so LORO
cannot hold them out of construction; only pit is clean of both selection and
construction, and pit correlates 0.87–0.92 with the others.

## 9–10. Referee selection and the qualification gate

Rule (`PREREGISTRATION_QUALIFICATION.md`, 2026-09-14, pinned in code):
G1 beats a coin flip (paired season-bootstrap CI < 0); G2 mean log loss and
mean Brier not worse than seed; independent referee = first qualified
full-coverage referee in the fixed order market_v2, pit, torvik, blend. The
rule is calibration-only and never looks at P(1st). Two process caveats,
both stated in the document: it was written after audit 1 showed which
referee it would remove (the market v1, whose construction defects are
demonstrable from inputs alone), and G2 is a point-estimate test. And one
defect not stated there: the incumbent it compares against had seen the
evaluation games (§4). Classification: **INDETERMINATE** until the same
frozen rule is recomputed on the corrected referees (Steps 8–9). No rule is
changed here.

## 11–12. Coverage before performance

§3 above; seed-table coverage now recorded per season (D7-3);
market_v2 fallback teams recorded (0–2/season); 2025 62 games.

## 13. Market referee

v1 defects confirmed from inputs (ID resolution, sign guard, no HCA) and
v1 retired from every runtime base (D7-1); v2 has curated resolution, no
fuzzy matching, no sign guard, HCA on non-neutral games, 0–2 seed fallbacks
per season, and reports them. odds_api: 5 seasons, no venue handling.

## 15. Topology

`build_season_context` derives the real pairing and every referee is
simulated by the one simulator on that tree; asserted equal to the
canonical bracket order for 2026 (`step7_math.py`). No referee-specific
simulator exists.

## 17. Selection vs evaluation referees

Selection referee = seed (production). Evaluation = each referee in turn.
The matrix reports production-selected P(1st) under every referee and
LORO-selected under held-out referees. The "self" column is an in-sample
ceiling and is labelled so.

## 18. LORO

`loro_choices` reselects on the mean P(1st) of the training referees'
selection trials and evaluates under the held-out one; the held-out
referee's outcomes do not enter selection or ranking. It does NOT hold the
referee out of candidate construction (torvik/blend marginals build the
candidates) or out of the opponent fallback. Classification: genuine for
pit and the market referees; partial for torvik and blend.

## 19. Unit of inference

Every CI is a paired bootstrap over seasons (n = 14; 5,000 resamples,
seed 42), never over trials. Correct.

## 10–12. Defects, fixes, invalidation

| id | defect | class | fix | test |
|---|---|---|---|---|
| D7-1 | defective v1 market loader still a runtime base (`odds`) in the backtest, meta_selector, stacked | MATERIAL-LOCAL (research bases) | runtime callers → `load_market_ratings_v2`; v1 documented defective | `test_runtime_bases_do_not_use_the_defective_v1_market_loader` |
| D7-2 | FTE point-in-time unverified, presented as pre-tournament | INDETERMINATE (data) | flagged `FTE_PROVENANCE`; supplementary only | `test_fte_is_supplementary_and_flagged` |
| D7-3 | seed referee's logistic-fallback share not recorded | MINOR (coverage transparency) | `seed_table_diagnostics` in `calibration_rows` | `test_seed_table_diagnostics_are_recorded` |
| — | published qualification and matrices computed with in-sample seed/blend, wrong topology, old P(1st), inert candidate family | stale results | none here | Steps 8–9 rerun |

Historical results invalidated: every number in `artifacts/referee_audit/`
(qualification statuses, +7.4/+7.3/+6.1/+4.9 pp referee deltas, self-referee
premium, LORO retention 60–101%, market_v2 +2.9 pp) and the README lines
quoting them. The originals are preserved unchanged; nothing is erased.

## Gate

| requirement | status |
|---|---|
| valid probability mathematics | PASS |
| correct topology | PASS |
| correct chronology | PASS (structurally walk-forward after Steps 2–3) |
| documented source independence | PASS (documented; weak for the qualified set) |
| valid coverage | PASS (recorded; v1 retired) |
| explicit fallback behaviour | PASS |
| no target-season leakage | PASS in current code; published runs leaked via the incumbent |
| reproducible construction | PASS |
| defensible qualification criteria | INDETERMINATE (rule sound in form; comparator was in-sample; timing reactive) |

**Step 7 verdict: INDETERMINATE**, not FAIL: no referee in current code
leaks, mis-signs or falls back to production, and the mathematics and
topology are verified; but the qualification result that decides which
referees support the robustness claim was computed against an in-sample
incumbent and the qualified set is largely one data family. Nothing about
the referees is presented as stronger than that.

**Downstream steps requiring rerun:** Step 8 — apply the already-frozen
gate to the corrected referees (no rule change; report every referee);
Step 9 — rerun the full matrix and LORO under the unified P(1st), real
topology, walk-forward seed/noseed, and the current candidate set; then
update README §"Selected and scored against the same referee".
