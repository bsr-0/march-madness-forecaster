# Council Report — 2026-04-19 How to Fix Bracket Selection

## Question

The MC ranker's #1 pick beats the pool winner in only 1 of 12 mode×year combinations (≈ random). Generation works (6.5% of brackets win). How should we improve bracket selection for the 2027 pool?

## Metadata

- **Bucket:** E (genuinely novel — prior councils assumed the ranker worked; §3 row 69 invalidated that)
- **Panel:** 5 advisors (Contrarian, First Principles, Statistician, Outsider, Executor)
- **Peer review:** ran (3 reviewers) — 2+ advisors materially diverged on whether selection is fixable
- **Timestamp:** 2026-04-19

## Suggested Actions

> 1. **Check rank stability first (Statistician's prerequisite).** Run the ranker twice with independent RNG seeds on the same 50 brackets. If the top-5 don't overlap across runs, the ranker is selecting from noise and no downstream fix will help — the problem is simulation variance, not objective function.
>
> 2. **If stable: reframe the selection objective from "most correct bracket" to "bracket that wins given what opponents pick" (First Principles + all 3 reviewers).** The current ranker optimizes P(1st) across simulated outcomes — circular, since brackets were built from the same distribution. The fix is to condition selection on the actual opponent pick distribution (observable via ESPN before lock), computing per-bracket conditional win probability against the specific field.
>
> 3. **If unstable: raise n_tournaments from 5,000 to 20,000 and regenerate opponents per bracket (Executor's variance-reduction fixes).** At 5,000 sims with P(1st) ≈ 6.5%, SE ≈ 1.1% — brackets are 1-3 SE apart, unresolvable. 20K cuts SE to 0.5%.

## Per-Advisor Bottom Line

- **Contrarian:** Selection may be irreducible noise. The P(1st) spread across 50 brackets is probably 0.04%-0.08% — within simulation noise. The 14-year aggregate ρ = +0.61 is a mirage from cross-year structural variation. Random top-quartile pick likely equals the ranker's pick.
- **First Principles:** The ranker is circular — it predicts its own model's tournament simulations, not reality. Two orthogonal signals worth investigating: leverage concentration (differentiation from public field) and robustness across noise seeds. Reframe from "most correct" to "wins given opponent picks."
- **Statistician:** 1/12 vs 5% base rate gives zero statistical power at n=4 years. Need 60-80 pool-years to validate any improvement. The ρ trajectory (near-zero 2023, strong 2024-2025, reversal 2026) suggests the ranker tracks chalk-heaviness, not true quality. First prerequisite: check rank stability across independent simulation runs.
- **Outsider:** Don't select — enter multiple brackets if pool rules allow (ESPN allows up to 25 per account). If not allowed, rank by how unpopular each bracket's picks are, not how accurate. The 17 failures all assumed "most correct = best." Wrong frame for winner-take-all.
- **Executor:** Concrete fixes: (1) raise n_tournaments 5K→20K to resolve brackets that are 1-3 SE apart, (2) regenerate opponent field per bracket instead of once globally, (3) add diversity penalty for overlapping picks, (4) validate with ρ > 0.3 threshold.

## Where Advisors Agree Most

- **The current P(1st) ranker optimizes the wrong thing.** It asks "which bracket is most likely correct?" when it should ask "which bracket wins given what opponents pick?" (First Principles, Outsider, confirmed by all 3 reviewers).
- **Statistical validation is impossible at n=4.** No selection improvement can be proven with 4 years of data (Contrarian, Statistician).

## Where Advisors Clash

- **Contrarian vs everyone on whether selection is fixable.** Contrarian says P(1st) spread is too narrow for any ranker to resolve — dead end. Executor says it's a variance problem fixable with more compute. First Principles says it's an objective-function problem fixable with reframing. Reviewers side with First Principles (3/3 pick B as strongest, A as biggest blind spot).
- **Outsider vs field on multi-entry.** Outsider proposes entering multiple brackets. Others don't engage with this. Reviewers flag that private pool rules likely prohibit multi-entry — the 25-bracket ESPN limit is a platform feature, not a pool feature.

## Blind Spots from Peer Review

- **The opponent field is observable (flagged by 3/3 reviewers).** In a ~30-person ESPN pool, competitors' brackets are visible via the ESPN group page before lock. None of the 5 advisors mentioned using actual opponent bracket data to compute true differential EV. This is the highest-leverage signal available and is practically achievable. The system already has pool_hist_results.json with 4 years of actual opponent brackets and a Tavily MCP server registered for opponent-intelligence workflows.
- **The distribution of losses is informative (flagged by 1/3 reviewers).** Did losing brackets lose by being too chalky or by picking the wrong upset? That distinction drives the fix but no advisor examined it.

## Load-Bearing Assumptions

- **Scoring encoding:** team-identity (real ESPN) — fixed via O26/O27
- **Opponent field:** ESPN-national 60/30/10 for simulation; actual ~30-person pool for retroactive comparison
- **RNG / sample count:** rng_seed=42, n_tournaments=5000 (current), 50 brackets × 50 repeats
- **Year scope:** 2023-2026 (4 years for pool retroactive); 2011-2026 ex 2020 (14 years for backtest)
- **Baseline anchor:** ranker's #1 pick vs pool winner (1/12 current success rate)
- **Data sources:** `artifacts/backtest_brackets/backtest_brackets_{2023-2026}.json`, `pool_hist_results.json`

## Framed Question + Prior Art

**Core question:** The MC ranker can't identify pool-winning brackets pre-tournament. How to fix selection?

**Prior art:**
- §3 row 69: ρ = −1.000 fix — revealed the ranker was being evaluated against the answer key
- §3 row 53: Council designed the bracket selection optimizer — proposed P(b_i > max(opponents)) as objective. Became O25, which FAILED (ρ ≈ 0).
- §3 row 58 (superseded): "opponent-model > selection" — reversed by row 65/69
- §2 O25: Empirical supremum dead-end (D17)
- §2 O24: Score threshold FAIL
- §2 D16: Chalk-fade FAIL
- §1: BSS ≈ 0 ceiling locked; f4_first_tv mode locked; 17 dead-ends documented

## Advisor Responses

### The Contrarian
The ranking correlation data is the tell: ρ swings from +0.07 to −0.81 across years, with 2 of 4 near-zero. That's not "selection is broken" — that's the signal doesn't exist at n=50 brackets. You're trying to rank 50 objects using a noisy tournament simulator, and the variance of "who wins one 64-team bracket" dwarfs any structural advantage your brackets have over each other. The fatal flaw: the 14-year aggregate ρ = +0.61 is a mirage. Aggregating across years collapses structural variation between bracket cohorts. Within a single year's 50 brackets, the spread in true P(rank=1) is probably 0.04% to 0.08% — indistinguishable from simulation noise even at 10,000 simulations per bracket. The generation claim ("13/200 contain the pool winner") is doing suspicious work here. The real question nobody's asking: does submitting the bracket with the highest expected score against ESPN field distributions materially outperform submitting one random bracket from the top quartile? Probably not, given the variance. If selection is irreducible noise, optimizing it is a dead end.

### The First Principles Thinker
The surface question is "how do we rank brackets better." The real question is: what are we actually selecting for? The ranker optimizes P(1st place) across simulated tournaments. But simulated tournaments are drawn from the same distribution the brackets were built against. You're not predicting the real tournament — you're predicting your own model's hallucinations of it. Circular. Strip back further: why do we have 50 brackets at all? Because we can't predict the winner. So we're hedging across a possibility space. The ranker then collapses that hedge by picking one bracket. You've built a portfolio, then asked a flawed oracle to pick the single best asset. Two candidates worth examining: (1) leverage concentration — does the bracket take maximally differentiated positions from the expected public field? (2) Robustness across simulation variance — not P(1st), but P(top-3) or win-rate stability across noise seeds. The deeper reframe: you don't need the best bracket. You need the bracket that wins given what your opponents are likely to pick.

### The Statistician
The 1/12 win rate (8.3%) against a 5% base rate gives you exactly zero statistical power to distinguish "broken ranker" from "lucky noise" — the confidence interval on a 4-observation binomial spans essentially the full [0,1] range. You need roughly 60-80 pool-years to detect a true win-rate lift of 5-8 percentage points at 80% power with α=0.05. You have 12. The ρ trajectory is the more damning signal. A monotone decline from near-zero in 2023 to −0.78 in 2025, then a snap back to +0.04 in 2026, is not consistent with a stable negative relationship. That pattern suggests the ranker is tracking a latent variable — possibly chalk-heaviness — that correlates with pool performance in chalk-friendly years and anti-correlates in upset-heavy years. On selection improvement: the distributional assumption needs empirical validation before any resampling strategy will help. First question: do the top-5 brackets by P(rank=1) consistently overlap across independent simulation runs? If not, you are selecting from noise, and the fix is variance reduction, not a different selection rule.

### The Outsider
The ranking problem is unsolvable with 4 data points. Stop trying. But here's what I notice: you already solved the hard part. You reliably generate a bracket that wins. The failure is in selection, not generation. So don't select. Enter multiple brackets if the pool rules allow it. If multi-entry isn't allowed, the selection problem isn't really about predicting which bracket is best. It's about picking the bracket most different from what other entrants will submit. In a winner-take-all pool, the winning bracket is almost never the most accurate one — it's the one that got lucky on the right upsets that other people didn't have. So don't rank by predicted accuracy. Rank by how unpopular each bracket's picks are. The 17 failed approaches all seem to share the same frame: "which bracket is most likely correct?" That's the wrong question for this pool format. Ask instead: "which bracket wins only if something unusual happens — and does it win big when it does?"

### The Executor
The root cause is clear. The ranker does 50 sims of a randomly sampled tournament outcome and scores all 50 brackets against each. With noise_std=0.16, each sim outcome is nearly independent noise. The ranker has no signal because the opponent field is also built from that same distribution. Concrete Monday morning plan: (1) Raise n_tournaments to 20,000 — at 5000 sims with P(1st)≈6.5%, SE≈1.1%. Your 50 brackets likely span 5-8% P(1st) — 1-3 SE apart, unresolvable. 20K cuts SE to 0.5%. (2) Re-generate the opponent field per bracket, not once globally — eliminates the fixed-field artifact. (3) Add a diversity penalty — if brackets #1 and #2 share 60+ picks, they're not different bets. (4) Validate: run backtest with --team-identity, measure ρ. If ρ > 0.3 you've fixed it; if ρ stays near 0, the brackets themselves are too homogeneous.

## Peer Review

### Reviewer 1
**Strongest: B.** Correctly identifies circular dependency and reframes to "wins given opponent picks." **Biggest blind spot: A.** Declares dead end without engaging the opponent-field problem. **All missed:** Opponent field is observable — ESPN group page shows competitors' brackets before lock. None mentioned using actual opponent data for differential EV.

### Reviewer 2
**Strongest: B.** Distinguishes "best forecast" from "best pool entry" — the core conceptual error. **Biggest blind spot: D.** Multi-entry loophole almost certainly violates private pool rules; dismisses 17 data points without examining loss distribution. **All missed:** Opponent pick frequencies are observable. Selection is a game-theory problem against a known field, not a forecasting problem.

### Reviewer 3
**Strongest: B.** Leverage concentration and noise-seed robustness directly address pool-specific differentiation. **Biggest blind spot: A.** Conflates simulation resolution with whether a better criterion exists. 6.5% win rate proves signal exists in the bracket set. **All missed:** Actual opponent brackets are knowable. Tavily MCP server is already registered for opponent-intelligence workflows. Ranker should score against modeled opponent field, not simulation truth.

---

_Council report generated 2026-04-19. What was counciled: how to fix bracket selection given the ranker can't identify winners pre-tournament (1/12 success rate)._
