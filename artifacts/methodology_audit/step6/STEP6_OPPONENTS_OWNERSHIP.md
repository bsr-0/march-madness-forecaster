# Step 6 — Opponent and ownership assumptions

Evidence: `opponent_audit.{py,txt}` (2026 data + synthetic pools),
`tests/test_opponent_model_contract.py` (7 checks). Base: Step 1–5 working tree.

## 1. The estimand, in plain language

- **P(1st)** of a bracket b = E over trials of the first-place share, where a
  trial draws (i) one field of n opponent brackets from the pick model and
  (ii) one tournament from the seed-vs-seed referee (walk-forward table,
  0.16 logit noise), scores b and the n opponents on that same tournament
  with ESPN points, and returns 1 if b is the strict maximum, 1/(1+k) if tied
  with k opponents at the maximum, else 0. n = the real pool's size − 1 for
  2023–2026 (17, 24, 31, 29), 29 otherwise. This is "probability of being the
  highest-scoring bracket against a specified synthetic pool, ties split" —
  exactly the expected winner-take-all prize. It is NOT the expected score.
- **EV** = expected ESPN points of b under the Torvik Log5 marginals,
  absolute, no opponent term.
- **Expected rank / P(top5) / P(top25)** = reported only (mean of
  better + 1 + tied/2), never optimised.
- **Ownership** enters twice: (a) as the opponent pick model above; (b) in
  candidate CONSTRUCTION, where `region_top_n`'s pick score is
  `model_prob × pts × ((1−risk) + risk × (1−public_prob))`. There is no
  ownership-adjusted objective at selection; selection is P(1st) alone.

Gate 1: PASS — unambiguous and matches the implementation (Step 4 verified
the share arithmetic; this step verifies the opponent side).

## 2. Opponent pipeline (from code)

`draw_selection_trials` → per trial `generate_opponent_brackets(n, first_round,
seed_pw, pick_dist, seeds, rng, chalk_noise_std)` then
`simulate_tournament_outcomes(1, …, noise 0.16)`. Each opponent is walked
game by game on the season's tree; P(pick t1 | t1, t2 in the slot) =
share_R(t1) / (share_R(t1) + share_R(t2)) from per-team round pick shares
(ESPN national shares, or the real pool's Laplace-smoothed shares for
2023–2026), falling back to the seed referee probability if both shares are
≈0. Opponents are independent of each other (chalk noise 0 in every canonical
run), redrawn every trial, drawn before any candidate is scored, and the same
field and tournament score every candidate (CRN). No candidate information
reaches the generator (its arguments are the field data only).

Gate 2: PASS on structure (matches the stated model, no candidate-dependent
behaviour). See F6-1 for fidelity to the stated inputs.

## 3. Pool size

Shipped 2026 bracket, 1,000 trials: P(1st) share 0.678 / 0.261 / 0.178 /
0.102 / 0.047 / 0.037 / 0.015 at n = 1 / 5 / 10 / 20 / 29 / 50 / 100,
against 1/(n+1) = 0.500 / 0.167 / 0.091 / 0.048 / 0.033 / 0.020 / 0.010.
Monotone, above the no-skill line throughout, converging toward it. The
regression test pins non-increase on a shared field. n = 29 is the documented
30-entry pool assumption (real pools 18–33). PASS.

## 4. Opponent selection bias

Opponents come from the field's pick shares and the seed table only; never
from the candidate bank, never from the candidate's strategy, never from the
candidate. PASS.

## 5. Common tournament outcomes

Candidate and opponents are scored on the same realisation. Controlled
comparison on the shipped 2026 bracket: shared 0.072 vs independent
realisations 0.201. These are different quantities — with independent draws
the candidate's good tournaments are compared against opponents' typical
ones, which destroys the common shock every pool entrant shares. The shared
form is the pool's estimand; the independent form is not a legitimate
estimator of it. PASS (and the directive's expectation that they agree is
not applicable here).

## 6. Ownership data semantics

| seasons | source | quantity |
|---|---|---|
| ≤2022 (except 2012), 2026 | `data/raw/historical_public_picks/espn_picks_{year}.json` (Kaggle/nishaanamin, "original source unverified" per the file's own caveat) | fraction of the national field advancing each team to each round; sums 31.4/15.8/7.9/3.95/2/1 (First Four losers absent) |
| 2023–2025 (2026 also has pool data) | the pool's own entries, `build_pool_pick_distribution` | same quantity for this pool, Laplace α = 0.5 |
| 2012 | static seed pick rates (Step 4 F4-6) | generic prior |

Semantics match what the objective assumes (marginal pick shares of the
field the bracket competes in), with the caveat that the national field is a
proxy for a 30-person pool. INDETERMINATE on provenance for the ESPN files
(documented in-file as unverified); not resolvable from the repository.

## 7. Independence assumptions (inventoried)

- Opponents mutually independent: explicit (chalk noise 0); the real 2023–26
  fields measured LESS correlated than independent draws (README, H3), so
  this is conservative.
- Within an opponent, per-game picks are conditionally independent given the
  slot; path consistency is enforced by the tree walk. Respects the tree.
- Team-level marginal shares are converted to per-game conditionals by the
  ratio rule: an approximation, not an identity (F6-1).
- Game outcomes: independent Bernoulli per game on the tree (Step 3).

## 8. Opponent legality

2,000 generated opponents decoded: 0 illegal; 32/16/8/4/2/1 winners, every
later winner won the earlier round, on the real 2026 tree. PASS.

## F6-1 — INDETERMINATE (fidelity to the stated inputs) — the ratio rule does not reproduce the later-round shares

20,000 opponents drawn from the 2026 ESPN shares, realised vs input share:

| round | max abs error | corr | worst cell |
|---|---|---|---|
| R64 | 0.023 | 0.9999 | — |
| R32 | 0.076 | 0.9986 | |
| S16 | 0.093 | 0.9984 | |
| E8 | 0.125 | 0.9959 | Michigan: input 0.602, realised 0.727 |
| F4 | 0.105 | 0.9925 | |
| CHAMP | 0.096 | 0.9909 | |

Independent per-game ratios compound favourites along the path, so the
simulated field is chalkier at deep rounds than the shares it is built from.
The marginals alone do not determine a joint bracket distribution, so any
sampler needs an assumption beyond the data; the ratio rule is one such
assumption and it is stated in code, but the README described the field as
"draws from ESPN pick rates", which the realised shares are not. For
2023–2026 the real entries are available and are reduced to marginals before
resampling, discarding their joint structure. No unambiguous correction
exists without a modelling choice (e.g. resampling real entries where they
exist, or calibrating conditionals to hit the marginals), so per the gate's
definition this is INDETERMINATE and recorded as **R-3 (optimization
item)**. README corrected to describe what the sampler does.

## 9/10. Ownership → P(1st) and synthetic pools

Synthetic pools of 29 against 1,500 referee draws, production `pool_p_first`
vs an independent share calculation:

| pool | production | independent |
|---|---|---|
| A all chalk | 0.4458 | 0.4458 |
| B identical to candidate | 0.0333 (= 1/30 exactly) | 0.0333 |
| C uniform random legal | 0.7477 | 0.7477 |
| D 10 chalk / 5 candidate / 14 random | 0.0740 | 0.0740 |

Exact agreement; ties at pool level behave as expected first-place share
(candidate tied with every opponent → 1/30). PASS (items 9–11).

## 12. Does ownership change the estimand?

Both pathways are real and measured (2026, blend base, risk 0.35):
- Construction: the bracket differs in 6 of 63 games with `public_picks`
  on vs off.
- Opponents: the ownership-built bracket scores P(1st) 0.060 against
  ESPN-share opponents but 0.092 against seed-model opponents; the
  ownership-free bracket 0.021 vs 0.099. The opponent model moves P(1st)
  by several pp and changes the ordering of brackets, i.e. the estimand
  depends materially on which field is assumed. That dependence is inherent
  to a pool objective, not a defect; it is why the referee/opponent
  assumptions are pre-registered rather than tuned.

## Gate

| Component | Finding | Severity | Evidence | Fix required? | Historical impact |
|---|---|---|---|---|---|
| Estimand definition | unambiguous, matches code | — | §1 | no | — |
| Opponent pipeline structure | matches stated model; no candidate dependence | — | §2, §4 | no | — |
| Pool-size behaviour | monotone, above 1/(n+1) | — | §3, test | no | — |
| Shared tournament realisation | correct estimand; independent draws are a different quantity | — | §5 | no | — |
| Ownership data semantics | marginal field shares; ESPN provenance unverified | INDETERMINATE (data) | §6 | no (document) | none |
| Opponent legality | 0/2,000 illegal | — | §8, test | no | — |
| Ratio-rule fidelity to input shares (F6-1) | deep-round shares off by up to 12 pp; joint structure of real entries discarded | INDETERMINATE (empirical assumption) → R-3 | §F6-1 | README corrected; modelling change deferred | none invalidated; the field is as stated in code |
| Ownership → P(1st) mathematics | exact on synthetic pools | — | §9/10 | no | — |
| Pool-level ties | expected share; 1/30 when tied with all | — | §10 | no | — |
| Ownership pathways | both construction and opponents; estimand depends on the assumed field | documented | §12 | no | — |

**Step 6 verdict: PASS on coherence, with two INDETERMINATE items recorded
(ESPN share provenance; the marginal-to-joint sampling assumption, R-3).**
No foundational defect; no result invalidated; no rebuild required.
