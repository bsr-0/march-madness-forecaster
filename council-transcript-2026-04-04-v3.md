# LLM Council Transcript — Session #8
**Date:** April 4, 2026
**Topic:** Next Step After Fixing Team-Name Matching

---

## Original Question

> Council this: next step

---

## Framed Question

**Decision: What should be the next step for this March Madness bracket pool optimizer?**

We just fixed the critical team-name matching bug — ESPN public picks now flow cleanly (100% match for 2008-2025), silent fallback eliminated. The last stochastic backtest (with broken data) showed all strategies performing near-random (mean rank ~490-500/1000, P(1st)≈0.1%). The repo has pivoted from ML prediction (zero value, BSS=0) to game theory pool optimization. Pool optimizer is end-to-end functional. Opponent model uses arbitrary 60/30/10 blend weights.

**Candidate next steps from prior sessions:**
1. Re-run stochastic backtest with clean ESPN data
2. Validate 60/30/10 blend (try ESPN-only 100/0/0)
3. Analyze public pick drift over 19 years
4. Clean up giant files (139K, 81K lines)
5. Delete orphaned files
6. Fix stacking weight contamination
7. Build contrarian signals from historical ESPN mispricings
8. Optimize for P(rank=1) not EV

---

## Advisor Responses

### The Contrarian

Re-run the stochastic backtest immediately. Everything else is premature.

If results still show near-random performance, the entire game theory pivot is built on nothing. Options 2-8 all assume the fixed data creates signal. You don't know that yet. Validating blend weights is pointless if the data still doesn't differentiate strategies. Building contrarian signals from ESPN data that might be uninformative is waste. Optimizing for P(rank=1) is an objective function change on a system that hasn't proven it can beat random.

The uncomfortable question nobody's asking: what if the ESPN public picks, even perfectly resolved, simply don't contain enough exploitable information to beat a pool? What if the public isn't wrong enough, often enough, in ways that matter? Your backtest will answer this.

Don't clean files. Don't delete orphans. Don't tune weights. Run the one experiment that tells you whether anything downstream is worth building.

### The First Principles Thinker

The question isn't "what should I do next." The question is: do you have any evidence this optimizer can beat random?

You don't. Strip away all the complexity. A bracket pool optimizer needs exactly two things: (1) better-than-public probability estimates for game outcomes, and (2) a leverage strategy that exploits the gap. ML adds zero value (BSS=0). So where are better estimates coming from? The 56 rating systems? If they're not demonstrably better than ESPN consensus, your optimizer is just adding noise to noise.

Run the backtest with clean data. But set a kill criterion BEFORE running it. At what point do you accept that bracket pools may be too efficient (or too luck-dominated over 63 games) for systematic edge? You need that number before you run the test, not after, or you'll rationalize continuing indefinitely.

### The Expansionist

Option 1 first, but only because it unlocks the real prize: option 7 is where the alpha lives.

Re-run the backtest — yes, obviously, it takes minutes. But the reason it matters isn't just validation. You now have 19 years of real public pick data flowing cleanly, which means you can finally answer: where does the crowd systematically misprice teams, and how stable are those mispricings?

Think about what you're sitting on. 56 rating systems, 19 years of ESPN public picks, and a working optimizer. Most bracket tools use one year of data and static seed-based upset probabilities. You have infrastructure to build a historically-calibrated contrarian signal.

The sequence: 1 → 7 → 8. Run the clean backtest, build the contrarian signal layer from historical mispricings, then optimize for winning outright rather than expected value. Options 2-6 are housekeeping — they don't change the ceiling.

Stop tuning plumbing and start mining the dataset.

### The Outsider

I can read a decision tree, and this one is obvious. Do option 1 first. Nothing else matters until you do.

You just fixed what you called a "critical" data pipeline bug. Your last measurement showed everything performing identically to random. You have no idea if that result was caused by the broken data or by something fundamental.

Why is option 6 (fix stacking contamination) not bundled with option 1? If you know about another bug, running a backtest with it still active gives you a result you can't trust either.

Options 3, 7, and 8 sound like the kind of things smart people do to avoid looking at a number that might tell them the whole approach doesn't work. You built a complete system. You fixed the data. Now measure it.

### The Executor

Here's what you do Monday morning:

**Step 1:** Re-run the backtest. Right now. It takes under 30 minutes. You fixed the data bug. Every other decision is meaningless until you see what the numbers look like with clean data.

**Step 2:** If the baseline still stinks, run the 100/0/0 ESPN-only variant. This takes the same 30 minutes. If pure ESPN data outperforms the 60/30/10 blend, your blend weights are actively hurting you. That's the single highest-leverage diagnostic.

**Step 3:** If the baseline looks reasonable, then fix stacking contamination. Because if future data is leaking into your backtest, your numbers are lying.

Everything else waits. The sequence is: measure → diagnose → fix leaks → then build features.

---

## Peer Reviews

### Anonymization Mapping
- Response A = The Contrarian
- Response B = The First Principles Thinker
- Response C = The Expansionist
- Response D = The Outsider
- Response E = The Executor

### Review 1
**Strongest: D (Outsider)** — bundling stacking fix with backtest is the sharpest insight. **Blind spot: C (Expansionist)** — treats backtest as formality, leaps to mining signals. **All missed:** Nobody asked what the bug's blast radius was — does the fix actually affect the near-random result? Understanding impact determines how much hope to attach.

### Review 2
**Strongest: E (Executor)** — only one with concrete sequence and decision tree. **Blind spot: C (Expansionist)** — jumps to building before gating on results. **All missed:** Nobody defines what "success" looks like numerically. Without a pre-committed threshold, you'll rationalize whatever result you get.

### Review 3
**Strongest: E (Executor)** — correct sequencing, time-boxed, conditional branches. **Blind spot: A (Contrarian)** — "approach may be invalid" is thought-terminating; random output from contaminated pipeline tells you nothing about the approach. **All missed:** Need specific metric and threshold (e.g., KS test p<0.05 on rank distribution) before running.

### Review 4
**Strongest: E (Executor)** — decision tree with branch points. **B's kill criterion is the most important IDEA** but everyone else ignored it. **All missed:** Sample size reality. 19 years = 19 data points for portfolio-level outcomes. The signal-to-noise ratio for P(rank=1) may not permit a statistically meaningful backtest.

### Review 5
**Strongest: E (Executor)** — correct sequence. **Blind spot: A (Contrarian)** — preparing to quit before testing. **All missed:** ESPN data serves dual role as both calibration source (game probabilities) AND strategic input (opponent model). Using it as both without separation creates circularity — a potential second data contamination bug.

---

## Chairman Synthesis

### Where the Council Agrees

Everyone says the same thing: re-run the backtest with clean data. No one argues for skipping it. No one argues for building features first. The last measurement showed near-random performance, and the data pipeline was broken. The single highest-value action is to see if fixing the pipeline changes the number. This is unanimous.

### Where the Council Clashes

The real disagreement is about what comes after the number, and how much hope to carry into it.

- **The Contrarian and First Principles Thinker** want a pre-committed kill criterion. If the number is still bad, walk away. They treat this as a go/no-go gate.
- **The Expansionist** treats the backtest as a formality before mining 19 years of ESPN data for contrarian signals. This is the weakest position — it assumes the answer before asking the question.
- **The Executor and Outsider** land in the middle: run the test, but have a diagnostic branch ready (100/0/0 ESPN-only variant, stacking contamination fix) so a bad result leads to root-cause analysis, not paralysis.

The clash is between "prepare to quit" and "prepare to diagnose." The Executor wins this argument. A single bad number from a system with multiple known issues (arbitrary blend weights, possible stacking contamination) is not sufficient evidence to kill the approach. But it IS sufficient to redirect effort toward isolation testing.

### Blind Spots the Council Caught

1. **No success threshold defined** (Reviews 2, 3, 4). Without a pre-committed number, you will rationalize whatever comes out. "Better than random" is not a threshold. Need a specific metric and cutoff before running.

2. **Sample size reality** (Review 4). 19 years = 19 tournament-level observations. For portfolio-level outcomes like P(rank=1), statistical power is brutally low. May not be able to distinguish real edge from noise.

3. **ESPN data circularity** (Review 5). ESPN data is used to estimate game probabilities AND to model opponent behavior. Optimizing against yourself partially. Not a showstopper but needs examination if edge appears — the edge might be an artifact.

### The Recommendation

**Run the backtest. But do three things the council mostly missed:**

1. **Define the threshold now.** Before running: if the optimizer's median rank is above the 40th percentile (rank 400+/1000), the system has no detectable edge and further work on the current architecture is not justified. If median rank is between 20th-40th percentile, edge exists but is weak — proceed to diagnostic isolation. If below 20th percentile, the pipeline fix was the bottleneck and move to optimization.

2. **Run two variants, not one.** The Executor's 100/0/0 ESPN-only variant is the right diagnostic companion. If the full blend is near-random but ESPN-only shows signal, the blend weights are the problem. If both are near-random, the issue is deeper. This adds 30 minutes and doubles the information.

3. **Do not fix stacking contamination yet.** The Outsider suggests bundling it, but that muddies the measurement. Isolate the effect of the data pipeline fix. Fix one thing, measure, then fix the next thing and measure again.

### The One Thing to Do First

Write down the success threshold — median rank below 400/1000 to continue — then run the backtest with clean data. Read the number. Everything else follows from that number or is premature without it.
