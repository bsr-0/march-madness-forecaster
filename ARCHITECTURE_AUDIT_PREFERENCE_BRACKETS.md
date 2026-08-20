# Architecture Audit — Preference-Driven Bracket Generation

**Date:** 2026-08-19
**Scope:** read-only audit. No production or UI code changed.
**Objective shift:** from "find the single best bracket" to "generate high-quality brackets
subject to user-specified structural preferences."

---

## 1. Current architecture

There are **two independent simulation stacks** in this repo. They do not share code and
they do not agree.

```
                        ┌─────────────────────────────────────────┐
                        │  STACK A — pipeline (pairwise, correct) │
                        └─────────────────────────────────────────┘

  team ratings ──> predict_fn(t1,t2) ──> MonteCarloEngine.simulate_tournament()
  (barthag/ELO/    PAIRWISE            src/simulation/monte_carlo.py:354
   massey/market)  Dict[(t1,t2)]->p             │
                                                │  _run_batch() keeps full per-sim
                                                │  round_results  (monte_carlo.py:143)
                                                ▼
                                    _aggregate_raw_results()          ◄── ⚠ PER-SIM
                                    monte_carlo.py:461                    OUTCOMES
                                                │                         DISCARDED
                                                ▼
                                    AggregatedResults (MARGINALS ONLY)
                                    championship_odds / final_four_odds / ...
                                                │
                                                ▼
                                    to_round_probabilities()
                                    src/pipeline/stages/simulation.py:296
                                                │
                                                ▼
                                    round_probs: team -> {R64..CHAMP -> p}
                                                │
                          ┌─────────────────────┴─────────────────────┐
                          ▼                                           ▼
                  PoolOptimizer                               LeverageCalculator
                  src/optimization/pool_optimizer.py           / ParetoOptimizer
                                                               src/optimization/leverage.py


                        ┌─────────────────────────────────────────┐
                        │  STACK B — pool backtest (PRODUCTION)   │
                        └─────────────────────────────────────────┘

  barthag ──_log5()──> pairwise ──> build_torvik_round_probabilities()
            mc_pool_backtest.py:496      mc_pool_backtest.py:505
                                                │
                                    ⚠ PAIRWISE PROBS THROWN AWAY HERE
                                                │
                                                ▼
                                    torvik_rp: team -> {round -> p}
                                                │
                     ┌──────────────────────────┴───────────────────────┐
                     ▼                                                  ▼
        CANDIDATE GENERATION                                  CANDIDATE SELECTION
        construct_bracket(mode="region_top_n", ...)           simulate_tournament_outcomes(
        src/optimization/bracket_construction.py:1131             matchup_probs = seed_pw )
                     │                                        src/simulation/pool_competition.py:390
                     │  ⚠ p1/(p1+p2) reconstruction                     │
                     │    _enumerate_region_outcomes:945                │  ⚠ scores candidates under a
                     ▼                                                  │    SEED-ONLY world model,
        ~25 candidate brackets                                          │    not the model's own probs
        (champ × risk × prob-base × mode)                               │
                     └──────────────────────────┬───────────────────────┘
                                                ▼
                                    argmax P(1st) over 25 candidates
                                    mc_pool_backtest.py:3322-3362
                                    ⚠ independent RNG draws per candidate
                                                │
                                                ▼
                                    meta_region_poolaware bracket
                                    (11.2% P(1st) — the production number)
```

**Key structural fact:** the pairwise game-level probabilities that everything should be
built on are computed *twice* and destroyed *twice*. Both stacks compress to per-team
marginal round probabilities, then downstream consumers try to reconstruct pairwise
probabilities from those marginals. That reconstruction is invalid (§3).

---

## 2. Every marginal → pairwise conversion site

`round_probs[team][R]` means **P(team wins its round-R game)** — the joint probability of
reaching *and* winning round R. Confirmed at `mc_pool_backtest.py:548`
(`advance_counts[winner][rnd] += 1`) and documented at `bracket_construction.py:445-449`.

| # | File:line | Function | Formula | Used by |
|---|---|---|---|---|
| 1 | `scripts/mc_pool_backtest.py:757` | `sample_model_brackets` | `p1/(p1+p2)` | all 4 base modes (seed/noseed/blend/torvik) |
| 2 | `scripts/mc_pool_backtest.py:859` | anchor-and-lock sampler | `p1/(p1+p2)` | construction-mode stochastic samplers |
| 3 | `scripts/mc_pool_backtest.py:1342` | confidence-routed sampler | `p1/(p1+p2)` then ±1.5× upset boost | M6 portfolio mode |
| 4 | `scripts/mc_pool_backtest.py:1526` | tiered-lock sampler | `p1/(p1+p2)` | champ/F4/E8-locked modes |
| 5 | `src/optimization/bracket_construction.py:945` | `_enumerate_region_outcomes` | `p1/(p1+p2)` → `log_prob` | **`region_top_n` — the production generator** |
| 6 | `src/optimization/bracket_construction.py:857-864` | `_run_sa_construction.predict_fn` | `max(probs_t1.values())/(...)` | `meta_sa`, `meta_sa_chalk` |
| 7 | `src/optimization/leverage.py:919` | `_approx_predict_fn` | `mean(probs)` ratio | `_filter_brackets_by_path_protection` |
| 8 | `src/optimization/bracket_construction.py:426` | `_compute_expected_points` | uses raw marginal as per-game P | EV/variance reported for every constructed bracket |
| 9 | `src/optimization/bracket_construction.py:178` | `_make_ev_scorer` | uses raw marginal as pick score | **every construction mode's objective** |

Sites 6 and 7 are the most degenerate: both collapse the round dimension entirely
(`max` / `mean` across rounds), producing a single round-agnostic "strength" per team.
Since marginals decrease monotonically with round, `max(probs_t1.values())` is always the
R64 value — so SA scores an F4 game using R64 win rates.

### CORRECTION (2026-08-19, during implementation)

Re-examining each site before changing it revised this table in three ways. The audit's
central finding (§3) is unaffected; the blast radius is not.

**Sites 8 and 9 are NOT defects. Marginals are the mathematically correct quantity for
expected score, and I was wrong to list them.** Under team-identity scoring you earn
`pts_R` for each team you named at round R that actually won a round-R game. By linearity
of expectation:

```
E[points] = sum_R pts_R * sum_{t in picked_R} marginal[t][R]
```

exactly — no independence assumption needed for the mean. `_make_ev_scorer` comparing two
marginals to choose a pick, and `_compute_expected_points` summing them, are both correct.
(The *variance* in site 8 does assume independence, which its own docstring already
admits.) These were left unchanged.

**Site 5 is real but dead.** `_enumerate_region_outcomes` accumulates the fabricated
probability into a `log_prob` that no caller reads — ranking is by `total_ev` at every
stage (lines 979, 985, 994) and the single caller discards it. It had zero effect on any
bracket. Removed rather than repaired.

**Three sites were missed and are now known:**

| # | File:line | Function | Status |
|---|---|---|---|
| 10 | `scripts/_bracket_export_common.py:144` | `build_bracket_json` | **LIVE, feeds the web UI's displayed `win_prob`.** Left unchanged — out of scope — but now documented in-place and allowlisted. |
| 11 | `scripts/divergence_diagnostic.py:80,85` | `compute_divergence` | Fixed. The script's entire purpose is comparing head-to-head probabilities across bases; it was comparing marginal ratios. Any divergence figure it produced before 2026-08-19 is unreliable. |
| 12 | `src/simulation/pool_competition.py:370` | `_get_pick_prob` | **Not a defect.** Normalizes two public *ownership* shares, which is a genuine ratio of two shares of the same population. Allowlisted with justification. |
| 13 | `src/prediction/meta_selector.py:69` | `_pairwise_prob` | **LIVE, feeds every `meta_gbm*` mode's features.** Left unchanged — see below. |

Site 13 was found only after strengthening the static scanner. The original check looked
for `p1 / (p1 + p2)` as a single expression; `_pairwise_prob` writes it across two
statements (`total = p1 + p2` then `p1 / total`), which slipped through. The scanner now
follows one level of indirection, and site 13 is the only remaining hit repo-wide.

It is retained deliberately. The value is consumed as a **GBM feature**, not as a
probability — `_game_features` computes it identically across every probability base and
feeds the model the *disagreement between bases*. A consistent monotone transform is a
legitimate feature even when it is not a calibrated probability, and the learner can
recalibrate it. Replacing it would change every `meta_gbm*` mode's feature matrix and
require retraining plus a fresh backtest, which is model work rather than a contract fix.
Worth revisiting later: the deeper rounds, where this transform is worst, are also where
the highest-value picks live.

**Consequence for expectations:** the production `meta_region_poolaware` path is driven
only by `_make_ev_scorer` (correct), so correcting the conversions does **not** change the
production bracket. This was verified empirically — see §9. The defect's real blast radius
is the stochastic samplers, SA, and the leverage path-protection filter.

---

## 3. Are the conversions valid?

**No, except in R64.** I measured it directly using the repo's own primitives: generate
pairwise probabilities with `_log5`, run 200k tournament sims, aggregate to marginal
`round_probs`, then apply `p1/(p1+p2)` and compare against the log5 values that *generated*
the data. A valid conversion would recover the inputs exactly up to Monte Carlo noise.

```
 round  n_pairs  mean|err|  max|err|  mean signed   (signed > 0 = favorite inflated)
   R64       32     0.0008    0.0028       0.0001
   R32       64     0.0708    0.1246       0.0708
   S16      104     0.1118    0.1959       0.1118
    E8       84     0.1169    0.1956       0.1169
    F4       65     0.1283    0.2220       0.1278
 CHAMP       48     0.1358    0.1823       0.1350
```

Worst individual cases:

```
   F4   6-seed vs 3-seed    true=0.302  approx=0.080  err=-0.222
   F4   2-seed vs 5-seed    true=0.742  approx=0.946  err=+0.204
  S16  13-seed vs 8-seed    true=0.303  approx=0.107  err=-0.196
   E8  10-seed vs 4-seed    true=0.222  approx=0.027  err=-0.196
```

Reproduce with `/tmp/audit_norm_bias.py` (self-contained, no repo imports).

### Why R64 is exact and everything after is not

Write `p_i = r_i · w̄_i`, where `r_i` = P(team *i* reaches this slot) and `w̄_i` = its average
win probability against the opponent distribution at that slot.

- **R64:** both teams reach with certainty (`r₁ = r₂ = 1`) and exactly one wins, so
  `p₁ + p₂ = 1` and `p₁/(p₁+p₂) = p₁` — correct by construction.
- **R32 onward:** four or more teams contest each slot, so `p₁ + p₂ < 1`. The ratio computes
  `P(t₁ wins this slot | one of {t₁,t₂} wins it)` — **not** `P(t₁ beats t₂ head-to-head)`.
  The reach probabilities `r₁, r₂` never cancel.

Since the favorite has both a higher `r` and a higher `w̄`, its advantage is **multiplied
rather than isolated**. The bias is one-directional (`mean signed ≈ mean |err|` at every
round — every single pair is distorted toward chalk) and grows monotonically with round
depth, because `r₁/r₂` diverges as rounds accumulate.

### Consequences

1. **The bias is exactly antagonistic to the new objective.** Every constraint on the target
   list — double-digit seed in the S16, non-1-seeds in the F4, aggressive upset profiles —
   lives in the tail this conversion suppresses. A 5-seed's genuine 35% F4 chance is
   reported as 14%. Preference-driven generation built on this substrate will systematically
   under-serve every upset-flavored preference a user asks for.
2. **"SA construction is fundamentally broken" is probably not a fact about SA.**
   `CLAUDE.md` §"Key Lessons" #4 records SA as killed at 1-2% P(1st) "regardless of signal
   quality." Site #6 shows SA's `predict_fn` is round-agnostic — it scores every game in
   the tournament with R64 win rates. That is sufficient on its own to explain the result.
   The conclusion should be treated as **unproven**, not as a closed question.
3. **Site #9 taints every construction mode's objective**, including the production one.
   `_ev_score = model_prob × pts × blended_diff` uses the raw marginal as the per-game win
   probability. For a bracket that has already advanced a team to the E8, the relevant
   quantity is `P(win the E8 game | reached the E8)`, but the marginal `P(reach and win E8)`
   is used instead — double-counting survival that the bracket path already asserts. Chalk
   picks are systematically over-scored relative to upset picks, at a rate that grows with
   round value (E8=80, F4=160, CHAMP=320 — precisely where the distortion is largest).

### Two further defects found in the same path

4. **Generator and evaluator use different world models.** Candidates are generated from
   `torvik_rp` / `massey_rp` / blends (`mc_pool_backtest.py:3217-3260`), but scored via
   `simulate_tournament_outcomes(matchup_probs=seed_pw)` (`:3336`) — historical seed-based
   pairwise probabilities. Selection therefore rewards brackets that do well *in a
   seed-only world*. Any edge the ratings model has over seed is invisible to the selector.
   This may be a deliberate anti-self-confirmation choice; it is not documented as one.

5. **Candidate selection uses independent noise per candidate.** The loop at
   `mc_pool_backtest.py:3322-3362` calls `generate_opponent_brackets` and
   `simulate_tournament_outcomes` *inside* the per-candidate loop, advancing a shared
   `_pa_rng`. Every candidate faces a different opponent field and a different tournament.
   With `pa_trials=500` and ~25 candidates, the per-candidate standard error on P(1st) is
   roughly `sqrt(0.11·0.89/500) ≈ 1.4pp` against a true spread between candidates that is
   likely of the same order — so the argmax is substantially selecting noise. **Common
   random numbers across candidates would remove most of this variance at zero cost.**
   This matters far more under the new objective, where the candidate count grows from ~25
   into the thousands and max-of-N noise scales with N.

---

## 4. Components that support conditional generation as-is

**Ready to use, no changes needed:**

| Component | Location | Why it fits |
|---|---|---|
| `simulate_tournament_outcomes` | `pool_competition.py:390` | **The single most important asset.** Already takes *pairwise* `matchup_probs` — the correct interface. Returns `(n_sims, 63)` bool array **and** `outcomes_by_round[sim][round][winners]`. This is a scenario bank; conditioning is row filtering. |
| `_log5` | `mc_pool_backtest.py:496` | Correct pairwise primitive, already present. |
| `score_brackets_team_identity` | `pool_competition.py:637` | Correct identity-based scoring (vs the shape-encoded variant at `:517`, which its own docstring warns against). Vectorized over brackets. |
| `generate_opponent_brackets` | `pool_competition.py:232` | Opponent field for P(1st). Takes pairwise `matchup_probs`. |
| `build_scoring_vector` | `pool_competition.py:469` | 63-slot points vector. |
| `MonteCarloEngine` | `monte_carlo.py:331` | Correct pairwise interface, richer noise model (regional correlation, injuries). Only needs per-sim outcome retention. |
| `wilson_score_interval` | `pool_competition.py:693` | Feasibility CIs for rare constraints. |
| `strategy_cache` | `src/data/strategy_cache.py` | Manifest + versioned cache for `(n, 63)` bracket arrays — scenario banks slot straight in. |

**The single blocking gap:** both stacks discard per-simulation outcomes at aggregation
(`monte_carlo.py:461`, and the `_pa_br`/`_4c_br` locals in `mc_pool_backtest.py` which are
consumed and dropped). Conditional scenarios require **retaining** the `(n_sims, 63)` matrix.
`simulate_tournament_outcomes` already returns it — no rewrite required, just don't throw
it away.

---

## 5. Existing diversity / leverage / constrained-optimization / portfolio code

Substantially more exists than the current single-bracket objective uses.

**Constrained optimization (proto):**
- `construct_bracket(forced_champion=..., max_one_seeds_f4=...)` — `bracket_construction.py:1131`.
  Two working constraints already. `max_one_seeds_f4` maps almost directly onto
  *"exactly two 1-seeds reach the Final Four."*
- `_pick_f4_teams` — `bracket_construction.py:489`. Enforcement is a **greedy repair**
  (unconstrained argmax per region, then demote 1-seeds until the cap holds) applied only
  in `f4_first`/`champ_first` modes. It is `max`-only, not `exactly`/`at least`, and does
  not generalize. Useful as a template, not as a foundation.
- `QuadrantCorrelationConstraint` — `path_protection.py:403`. A real constraint object with
  `evaluate()` / `compute_penalty()`. The closest thing to a constraint interface here.
- `PathProtectionScorer` — `path_protection.py:130`. Enforces path consistency, which is a
  hard feasibility requirement for any constrained bracket.

**Portfolio / diversity:**
- `BracketPortfolioGenerator` — `bracket_portfolio.py:115`. `generate_portfolio`,
  `_enforce_diversification` (:425), `_select_diverse_brackets` (:659), `_bracket_correlation`
  (:527), plus chalk / contrarian / targeted generators (:701, :766, :886).
- `portfolio_diversification.py` — `compute_portfolio_entropy`, `compute_bracket_correlation`,
  `compute_portfolio_anti_correlation`, `compute_champion_diversity`,
  `compute_top_k_coverage_estimate`. Small, dependency-light, directly reusable as
  preference-portfolio metrics.
- `ParetoOptimizer.generate_pareto_brackets` — `leverage.py:1172`. Risk-level sweep +
  dedup + path-protection filter. Already the right *shape* for "aggressive vs conservative
  upset profile" — it sweeps `risk_level ∈ [0,1]`. Its scoring is contaminated by sites #7/#9.
- `RiskProfile` / `ESPNPoolPortfolio` — `bracket_portfolio.py:963, 1021`, with
  `recommend_profiles(pool_size)`.

**Leverage:**
- `LeverageCalculator.find_leverage_picks` / `find_fade_picks` — `leverage.py:744, 806`.
- `AggregatedResults.get_leverage_picks` — `monte_carlo.py:116`.
- `compute_ev_edge` — `leverage.py:31`; `evaluate_pool_roi` — `leverage.py:338`;
  `compute_kelly_fraction` — `leverage.py:311`.

**Assessment:** the *portfolio and diversity* layer is in good shape and mostly reusable.
The *constraint* layer does not exist as a layer — it is two ad-hoc keyword arguments and a
greedy repair inside one construction mode. The *leverage* layer is conceptually sound but
built on the contaminated marginal probabilities.

**Also relevant:** `CLAUDE.md` §"Hard-Won Rule: ESPN Has NO Contrarian Bonus" constrains the
design. Uniqueness/leverage must stay a property evaluated over the *full bracket vs the
field* — it must not be pushed into per-game weights. The scenario-bank design below
respects this: leverage is computed by scoring complete brackets against an opponent field,
never by reweighting individual games.

---

## 6. Proposed experiment — smallest thing that answers the question

**Question:** does conditioning on a user preference produce a bracket that is *structurally
different* from the unconditional optimum while remaining *high quality*? If conditioning
just returns the chalk bracket with one pick nudged, the whole direction is not worth building.

**Non-goal:** fixing the conversion defects in production. This experiment routes around
them by using pairwise probabilities end-to-end, which incidentally establishes the
reference implementation for a later fix.

### Design: scenario bank + sample-average approximation

One standalone script, `scripts/experiments/conditional_bracket_poc.py`. **No changes to
`src/`, `docs/`, or any production path.** One season (2025 and 2026, both already ingested),
no backtest sweep — so this does **not** trip the `--tier` run-policy gate.

```
 STEP 1  pairwise probabilities        barthag ──_log5()──> matchup_probs[(t1,t2)]
         (reuse mc_pool_backtest.py:496 — no new math)

 STEP 2  scenario bank                 simulate_tournament_outcomes(n=200_000, matchup_probs)
         (reuse pool_competition.py:390)   -> outcomes (200k, 63) bool   ≈ 12.6 MB
                                           -> outcomes_by_round[sim][round] -> winner ids

 STEP 3  constraint predicate          C: outcomes_by_round[sim] -> bool
         (~40 lines, one function per constraint)
                                       mask = [C(s) for s in bank];  p_C = mask.mean()
                                       report p_C with wilson_score_interval()

 STEP 4  candidate brackets            THE KEY SIMPLIFICATION:
                                       use the simulated tournaments themselves as
                                       candidate brackets. Every row of the bank is a
                                       path-consistent, feasible 63-game bracket by
                                       construction. Deduplicate; take top-K by frequency
                                       plus a random tail.
                                         - UNCONDITIONAL arm: candidates from full bank
                                         - CONDITIONAL   arm: candidates from bank[mask]
                                       No new construction algorithm, no path-consistency
                                       repair, no constraint-satisfaction solver.

 STEP 5  evaluation (COMMON RANDOM NUMBERS — fixes defect #5)
         Draw ONE evaluation set up front and reuse it for every candidate in both arms:
           - eval bank:      a held-out slice of the UNCONDITIONAL bank
                             (the user's preference does not change reality —
                              conditional brackets must be judged against the real world)
           - opponent field: generate_opponent_brackets(), fixed draw, shared
         Score with score_brackets_team_identity (identity, not shape).
         Objectives:  E[ESPN points]   and   P(1st) = P(score >= max(opponent scores))

 STEP 6  report                        per constraint: p_C, Hamming distance between
                                       conditional and unconditional winners, champion /
                                       F4 deltas, EV retention, P(1st) retention,
                                       seed-profile of the differing picks
```

**Constraints to test** (all six from the project brief, plus a null control):

| # | Constraint | Expected `p_C` |
|---|---|---|
| 0 | *(null control — always true)* | 1.00 |
| 1 | ≥1 double-digit seed reaches the S16 | ~0.55 |
| 2 | ≥1 of a 2/3-seed reaches the F4 | ~0.60 |
| 3 | exactly two 1-seeds reach the F4 | ~0.20 |
| 4 | a named team (e.g. UConn) reaches the F4 | ~0.15 |
| 5 | ≥2 double-digit seeds reach the S16 | ~0.20 |
| 6 | 1-seed champion **and** ≥1 double-digit S16 (compound) | ~0.35 |

The null control is essential: with constraint 0 the conditional arm must reproduce the
unconditional arm exactly under the same seed. If it does not, the harness is broken and
every other number is meaningless.

**Estimated size:** ~250 lines, ~90% of which is reporting. Everything computational is an
existing function. Runtime: dominated by Step 2 (200k sims in the pure-Python loop at
`pool_competition.py:390`, roughly 15-25 min single-core — parallelize by seed if needed).

### What this deliberately does not do

- Does not fix sites #1-#9. Those are a separate, larger change gated on this result.
- Does not touch `construct_bracket` or any production construction mode.
- Does not run `run_experiment --tier` or `mc_pool_backtest` — no run-policy approval needed.
- Does not attempt "maximize uniqueness/leverage" as an objective yet. Step 5 measures
  P(1st) against a field, which is the correct leverage-aware objective; an explicit
  uniqueness knob is a follow-on once the conditional machinery is shown to work.

---

## 7. Acceptance criteria

The experiment **succeeds** — meaning the preference-driven direction is worth building —
only if all of the following hold. Each is measured and reported by the script; none is a
judgment call.

**Harness correctness (must pass, else results are void):**

- **A1.** Null control (constraint 0) selects a bracket **identical in all 63 games** to the
  unconditional arm under the same master seed.
- **A2.** **100.0%** of brackets emitted by the conditional arm satisfy their constraint, and
  100% are path-consistent. Zero tolerance.
- **A3.** `p_C` is reported with a Wilson 95% CI for every constraint. Any constraint whose
  filtered bank has fewer than 2,000 surviving scenarios is reported as **infeasible at this
  bank size** and excluded from A4-A6 rather than reported with a wide CI.

**The substantive question — is conditioning meaningful?**

- **A4. Structural difference.** Across the ≥5 feasible non-null constraints, the median
  Hamming distance between the conditional and unconditional selected brackets is **≥ 4 of
  63 games**, and **≥ 3 constraints** produce at least one differing pick in the S16 or
  later. *Rationale: a conditional bracket that differs only in a first-round 5/12 game is
  a cosmetic result — the user's preference must propagate into the high-value rounds where
  ESPN scoring concentrates (E8=80, F4=160, CHAMP=320).*
- **A5. Quality retention.** For every constraint with `p_C ≥ 0.10`, the conditional
  bracket retains **≥ 92% of unconditional E[ESPN points]** and **≥ 60% of unconditional
  P(1st)**, both measured against the *unconditional* eval bank and the *shared* opponent
  field. *Rationale: a preference should cost something, but a preference that halves your
  expected score is a losing product. The P(1st) bar is deliberately looser than the EV bar
  because P(1st) is a tail statistic with far higher variance.*
- **A6. Cost monotonicity.** Spearman ρ between `log p_C` and EV retention across the
  feasible constraints is **≥ 0.6**. *Rationale: this is the strongest single evidence that
  real conditioning is happening. Rarer preferences must cost more. If cost is uncorrelated
  with rarity, the machinery is returning noise regardless of how good A4 and A5 look.*

**Selection stability (tests defect #5 directly):**

- **A7.** Re-running the full experiment under **3 different master seeds** selects the same
  bracket for **≥ 80%** of (arm × constraint) cells. If this fails, the evaluation trial
  count is too low and the argmax is selecting noise — report the trial count needed to
  reach 80% rather than reporting the brackets. *This criterion doubles as a direct
  measurement of how much of the production 11.2% P(1st) figure is selection noise.*

**Hygiene:**

- **A8.** `git diff` touches only `scripts/experiments/` and `artifacts/`. Zero lines
  changed in `src/`, `docs/`, or `scripts/mc_pool_backtest.py`.
- **A9.** The conversion-bias measurement (§3) is committed as a regression test asserting
  mean |err| ≤ 0.01 per round for the pairwise path — so that the eventual production fix
  has a pass/fail gate, and so the current failure is pinned in the test suite.

**Interpretation.** A4 + A6 passing while A5 fails means the direction is real but the
objective needs work (likely: constrain the *scenario bank* rather than the *candidate set*,
i.e. importance-weight instead of hard-filter). A4 failing means preferences are not
propagating and the candidate-generation step needs a real constrained optimizer rather
than the SAA shortcut. A7 failing invalidates nothing about the direction but blocks any
quantitative claim.

---

## 8. Recommended sequencing

1. **This experiment** (standalone, no production risk).
2. **If it passes:** replace the marginal→pairwise reconstruction with a pairwise-native
   path. Sites #5, #8, #9 first — they are on the production `region_top_n` path. Gate on
   the existing P(1st) acceptance criterion (≥8/15 backtest years) and on A9's regression
   test. This is the change that requires run-policy approval.
3. **Then:** promote the constraint predicate into a real interface, using
   `QuadrantCorrelationConstraint` (`path_protection.py:403`) as the shape.
4. **Then:** UI.

**Separately, and independent of all the above:** re-open the "SA is fundamentally broken"
finding (§3 consequence 2) once site #6 is fixed. The recorded conclusion rests on a
`predict_fn` that scored F4 games with R64 win rates.

---

## Appendix — repo state notes

- `CLAUDE.md` and the entire `memory/` directory are **deleted in the working tree**
  (staged `D`). `CLAUDE.md` references `memory/run_policy.md`, `memory/project_north_star_metric.md`,
  and others that no longer exist on disk. The content quoted in this audit came from
  `git show HEAD:CLAUDE.md`. Worth resolving before the next session — the run-policy gate
  is currently undiscoverable by anyone reading the working tree.
- Untracked feature artifacts present: `data/raw/historical/{clutch,shooting}_features_{2024,2025}.json`,
  `player_minutes_2025.json`.

---

## 9. Post-correction measurements (2026-08-19)

The pairwise contract was implemented (see `src/prediction/pairwise.py`) and the invalid
conversions replaced. Two questions were answered empirically by running identical probes
against a clean `HEAD` worktree and the corrected tree.

### 9.0 Method, and a validity check on it

All before/after numbers below come from running the same probe script against a detached
`git worktree` at `HEAD` and against the corrected tree.

**A flaw in that setup was caught and checked.** The intent was to point the worktree at the
main tree's `data/` via a symlink, but `ln -sfn` against an existing directory creates the
link *inside* it — so `/tmp/mmf_head/data/data` was created and the worktree actually used
its own checkout. Since `git-lfs` is not installed on this machine, that raised the
possibility that the two trees read different input data, which would invalidate every
comparison.

It did not. Verified directly: `_load_torvik_barthag` returns **bit-identical** values on
both trees for 2023-2026 (same SHA-256 over the sorted rating vector), 68 teams each, and
**zero** teams falling back to the seed ladder — i.e. both trees loaded the same real Torvik
ratings. No LFS pointer stubs exist in either tree's `data/` or `artifacts/`. The
comparisons stand.

One residual difference: the worktree has 146 files in `data/raw/historical` versus the main
tree's 158, because the main tree carries untracked feature artifacts. This is why
`test_meta_selector` reported 70 passed on the corrected tree but 69 passed + 1 skipped on
`HEAD` — a data-availability skip, not a behavioural difference. The two runs' wall times
(21:19 vs 8:45) overlapped and contended for CPU, so they are not comparable; a micro-
benchmark put `ProbabilityBase`'s Mapping overhead at ~0.1 microseconds per access, far too
small to account for the gap.

### 9.1 Do the existing deterministic strategies diverge? No.

`torvik` round_probs and the brackets produced by `region_top_n`, `exhaustive_champion`,
`champ_first`, `f4_first`, `e8_first` and `forward_greedy` are **bit-identical** before and
after, across 2023-2026 (probe compared SHA-256 of the sorted picks dict, the champion, the
Final Four, and expected points). This is the predicted consequence of the §2 correction:
those modes never consumed a fabricated pairwise probability.

**The production `meta_region_poolaware` bracket and the 11.2% P(1st) baseline are
therefore unchanged.** No re-validation of the North Star metric is required.

### 9.2 Does the corrected architecture produce more variation? Substantially.

`sample_model_brackets` on torvik probabilities, 20,000 brackets per year:

| Year | metric | before | after | delta |
|------|--------|-------:|------:|------:|
| 2023 | mean R64 upsets | 8.647 | 8.662 | **+0.015** |
| 2023 | P(1-seed champion) | 0.619 | 0.424 | **-0.194** |
| 2023 | P(>=1 double-digit seed in S16) | 0.852 | 0.913 | +0.061 |
| 2024 | mean R64 upsets | 7.832 | 7.838 | **+0.006** |
| 2024 | P(1-seed champion) | 0.752 | 0.596 | **-0.156** |
| 2024 | P(>=1 double-digit seed in S16) | 0.718 | 0.832 | +0.114 |
| 2025 | mean R64 upsets | 7.250 | 7.249 | **-0.001** |
| 2025 | P(1-seed champion) | 0.882 | 0.703 | **-0.179** |
| 2025 | P(>=1 double-digit seed in S16) | 0.692 | 0.815 | +0.124 |
| 2026 | mean R64 upsets | 6.303 | 6.314 | **+0.011** |
| 2026 | P(1-seed champion) | 0.761 | 0.594 | **-0.168** |
| 2026 | P(>=1 double-digit seed in S16) | 0.429 | 0.611 | +0.182 |

Champion-seed tail mass, `P(champion seed >= 5)`:

| Year | before | after | ratio |
|------|-------:|------:|------:|
| 2023 | 0.0230 | 0.1437 | 6.2x |
| 2024 | 0.0034 | 0.0601 | 17.7x |
| 2025 | 0.0022 | 0.0581 | 26.4x |
| 2026 | 0.0055 | 0.0613 | 11.1x |

**The R64 rows are the control.** R64 is the one round where `p1/(p1+p2)` is exact, and the
R64 upset rate moves by at most 0.015 games out of 32 — i.e. not at all. Everything that
moved, moved in later rounds. That is exactly the signature the mechanism predicts, and it
is strong evidence the change is a correction rather than a perturbation.

**Answer to the question the pivot hinged on:** the model was not short of Cinderella
signal. The pipeline was destroying it between the ratings and the brackets. A 2025
bracket sample that gave a non-1-seed champion 12% of the time now gives one 30% of the
time, and 5-seeds-or-worse went from 0.2% to 5.8%.

### 9.3 Simulated annealing

SA's bracket changed in all four years, and its pre-fix output was visibly pathological:

| Year | F4 seeds before | F4 seeds after |
|------|-----------------|----------------|
| 2023 | 1, 1, 1, 1 | 1, 1, 2, 3 |
| 2024 | 1, 1, 1, 1 | 1, 1, 1, 6 |
| 2025 | **1, 1, 13, 16** | 1, 2, 4, 6 |
| 2026 | 1, 1, 2, 4 | 1, 2, 5, 6 |

A 16-seed in the 2025 Final Four is the round-agnostic `predict_fn` in plain sight: SA was
evaluating F4 games with R64 win rates, where a 16-seed's number is not absurd. Alongside
that, all-1-seed Final Fours in 2023 and 2024.

**The recorded conclusion that "SA construction is fundamentally broken regardless of
signal quality" (CLAUDE.md, 14-technique bakeoff) was measured against this. It should be
treated as unproven, not as a closed question.** Re-testing SA requires a backtest run and
is deliberately left for a separate, explicitly-approved experiment.

### 9.4 Paired 15-year backtest: the correction COSTS pool performance

Canonical contract (`--team-identity --opponent pool --n-opponents 30 --n-repeats 100`,
2011-2026 excl. 2020, n=15) run on both trees. The `HEAD` run reproduces the documented
baselines exactly — seed 4.92% against CLAUDE.md's 4.9%, SA 1.0-1.3% against its recorded
"1-2%, KILLED" — which validates the harness before any comparison is drawn.

| Mode | P(1st) before | P(1st) after | delta | MeanScore before | after |
|------|-------------:|------------:|------:|----------------:|------:|
| seed | 4.92% | 4.05% | **-0.87** | 695 | 612 |
| noseed | 3.93% | 0.58% | **-3.35** | 637 | 338 |
| blend | 4.78% | 1.95% | **-2.83** | 661 | 464 |
| torvik | 4.23% | 3.40% | **-0.83** | 728 | 662 |
| meta_sa | 1.33% | 0.73% | -0.60 | 199 | 215 |
| meta_sa_chalk | 1.00% | 1.67% | +0.67 | 195 | 316 |
| meta_sa_vol | 1.00% | 0.93% | -0.07 | 253 | 281 |

Approximate SE on P(1st) at these rates over 15 years x 100 repeats is 0.3-0.5pp, so the
noseed and blend collapses are ~6-7 SE and unambiguous; the meta_sa_chalk gain is ~2.5 SE
across correlated years and should not be treated as a real improvement.

**Mathematically correct probabilities produce worse pool results.** The invalid transform
was load-bearing. Two distinct effects are mixed together here and should not be conflated:

**Effect A — the chalk bias was accidentally beneficial (all modes).** `p1/(p1+p2)`
sharpened every later-round matchup toward the favorite. Favorites usually win, so
chalk-biased brackets score more points, and in a 31-entry winner-take-all pool P(1st) is
driven substantially by expected score. MeanScore fell for every base mode. This is the
project's own "ESPN has NO contrarian bonus" rule reasserting itself: the recovered upset
variation (§9.2) is real, but variation that does not hit simply costs points. It is also
consistent with `meta_region_poolaware` — near-chalk construction plus selection — sitting
at 11.2% while every stochastic mode sits at 4-5%.

**Effect B — noseed and blend have a much weaker pairwise source than marginal source.**
Their collapse is far larger than seed's or torvik's, and the reason is structural.
`build_noseed_round_probabilities` starts from *seed-based* historical advancement rates
and adjusts them, so it inherits seed's strong chalk structure. `build_noseed_probabilities`
is the raw LR+GBM head-to-head model. Sampling used to run on the former and now runs on
the latter. MeanScore 338 is close to random-bracket territory. `torvik` — the one base
whose marginals are derived from its own pairwise by simulation, and therefore internally
consistent — moved least.

**SUPERSEDED 2026-08-19 (later the same day): Effect B is a bug, not a calibration
weakness.** Follow-up investigation root-caused it to train/serve feature skew — the
backtest serves `_build_feature_vector` a four-factors-only stats dict, so `adj_offensive_
efficiency`, `adj_defensive_efficiency`, `adj_tempo` and `barthag` are identically 0.0 on
both sides of every differential. The model returns ~0.5 for everything, ranks the seed
favorite in only 17 of 32 R64 games (chance), and puts 1-seeds over 16-seeds at 0.474-0.540.
The no-seed model is not weak; it is being fed a crippled vector. Full diagnosis, evidence
table and proposed fix in `FINDINGS.md` §6c. The consequence for this section is that the
`noseed` and `blend` rows above should be read as "the bug became visible", not as a
measurement of those models.

**What this does NOT mean.** It does not mean the old transform should be restored. It was
invalid (§3, pinned by test), the production path never used it (§9.1), and correct
conditional probabilities are a prerequisite for the conditional-generation work. It means
the historical figures for the stochastic comparison modes were flattered by a bug, and
their corrected figures are the honest ones.

**A note on what these deltas do and do not measure.** It is tempting to read the torvik
row (-0.83pp) as "the price of allowing more upsets". It is not, and should not be quoted
that way. It is the cost of *this particular probability correction* under *this particular
15-year pool evaluation*. The correction changed several things at once — the shape of the
later-round distribution, the source of each base's sampling probabilities, and (for
seed/noseed/blend) the consistency between a base's marginals and its pairwise table. No
single one of those is isolated by this comparison. The upset/EV tradeoff is suggestive
here but has not been measured.

**Recommended follow-up, deliberately NOT done here (see §10).** A temperature `tau` on the
pairwise logit, `p' = sigmoid(logit(p) / tau)`, where `tau < 1` sharpens toward chalk,
`tau > 1` flattens toward upsets, and `tau = 1` is the calibrated probability. Fitting
`tau` walk-forward against P(1st) and MeanScore would isolate the sharpening axis and give
a clean estimate of the upset/EV tradeoff that this section cannot.

**`tau` must not be chosen to reproduce the pre-correction numbers.** That would be
reverse-engineering the bug and handing it a privileged status it has not earned. The old
transform is not a baseline to hit; it is a defect that happened to sharpen. The question
worth asking is scientific and independent of it: *can controlled probability sharpening
improve winner-take-all pool performance without destroying calibration?* If `tau = 1` wins
out of sample, that is a real answer and the correction stands unmodified.

### 9.5 Simulated annealing: question reopened, then closed

§3 argued the recorded "SA is fundamentally broken regardless of signal quality" conclusion
was unproven because SA's `predict_fn` was round-agnostic. That was a fair challenge, and
it has now been tested rather than argued.

With a correct pairwise `predict_fn`, SA lands at 0.73% / 1.67% / 0.93% P(1st) across its
three variants, against a seed baseline of 4.05%. Its pathological brackets are gone (no
more 16-seeds in the Final Four, §9.3) and its MeanScore improved in all three variants,
but it remains far below baseline and is not competitive.

**The original conclusion stands. SA is ineffective on the merits, not merely because its
predict_fn was broken.** The `CLAUDE.md` entry can be left as-is, with the note that it has
now been re-validated under a corrected matchup model.
