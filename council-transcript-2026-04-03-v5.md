# LLM Council Transcript — Session 5
**Date:** April 3, 2026
**Question:** After stochastic backtest shows all modes near-random, what next?

---

## Framed Question

We're building a March Madness bracket forecasting system with three modes: seed (historical seed win rates), noseed (ML model on team features), and blend. After 4 council sessions, we fixed the argmax defect with stochastic bracket sampling.

Stochastic backtest results (17 years, 1000-person pools, 50 model brackets, 50 repeats):
- Seed: mean rank 489, best bracket rank 18, P(top5%)=6.1%, mean score 1097
- Noseed: mean rank 496, best bracket rank 20, P(top5%)=5.5%, mean score 1093
- Blend: mean rank 500, best bracket rank 27, P(top5%)=3.2%, mean score 1088
- Paired t-test (seed vs noseed mean rank): p=0.70 — completely indistinguishable
- Mean ranks ~490-500 are near random (500 = coin flip)

What we know:
1. Noseed has significantly better Brier scores (p=0.0006, BSS=+0.066)
2. That Brier advantage does NOT translate to pool performance
3. All three modes are statistically indistinguishable
4. Mean ranks near 500 suggest none meaningfully beat the opponent field
5. The opponent model uses SEED_PICK_RATES (seed-based pick frequencies with chalk bias)

The question: Is this the end of the road, or is there signal being left on the table?

---

## Advisor Responses

### The Contrarian

The opponent model is the fatal flaw nobody wants to name.

Your "random baseline" is not random. SEED_PICK_RATES with chalk bias means you're measuring performance against a field that systematically over-picks favorites. That's not a neutral benchmark — it's a specific exploitable bias you're failing to exploit.

Mean rank ~490-500 out of 1000 doesn't mean "coin flip performance." It means your strategies are not beating chalk-heavy humans. Those are different problems with different solutions.

The Brier score advantage is real (p=0.0006 is not noise) but you're converting probabilities into picks the same way everyone else does. Better calibration means nothing if your bracket construction doesn't differentiate from the field. You're using superior probability estimates to make the same chalk picks the crowd makes.

Here's what's actually being left on the table: contrarian pick selection. If you know the crowd will over-pick the 1-seeds in certain matchups, and your ML model agrees those teams are favorites but at lower probability than the crowd implies, the correct play is not to pick them. You need to model expected pool value (your probability minus the field's pick frequency), not just raw win probability.

The deeper problem: you never tested whether your probabilities actually diverge meaningfully from the crowd's implicit probabilities. If SEED_PICK_RATES and your noseed model agree on who wins most games, you will always perform near median regardless of calibration quality.

The months of ML work aren't wasted — they're being misapplied. The question isn't "who wins" but "who wins that the crowd isn't picking." You haven't built that yet.

### The First Principles Thinker

The framing of "months of ML work has no value" is the wrong conclusion to draw.

The actual problem being solved in a 1000-person pool is not "predict games accurately." It is "predict games differently than the field predicts them."

The Brier score improvement (BSS=+0.066, p=0.0006) is real. The model genuinely knows something. But the pool performance is near-random because the opponent model is wrong. You're measuring "how does noseed do against chalk-biased humans?" — and if your noseed model is also chalk-biased (picks similar teams to the crowd), better probability estimates buy you nothing in a differentiation game.

The actual question to ask: does noseed produce structurally different brackets than seed? If both modes are picking similar champion paths, similar Final Fours, you're swimming in the same lane as the field regardless of which probability distribution generated the picks.

What needs to be built next is not a better forecasting model. It is a contrarian sampling mechanism. The opponent model (SEED_PICK_RATES) is your only lever for pool edge. If you shift to a low-popularity-team variant — sampling from the noseed model while downweighting chalk picks — you accept lower mean score in exchange for higher variance brackets that occasionally land in the top 1%.

The Brier advantage is real signal. It's just being wasted in a symmetric sampling strategy. The ML model is the right foundation. The sampling layer needs to exploit crowd mispricings, not just maximize expected score.

This is not the end of the road. It's the beginning of the actual problem: using probability edge to construct positively differentiated brackets.

### The Expansionist

The Brier score gap (p=0.0006) is real signal — you're just measuring the wrong outcome to harvest it.

Pool rank against a chalk-biased field is not the right target. When your opponents all cluster toward favorites, the path to top-5% isn't "predict better" — it's "predict differently in the right spots." Your ML model has genuine probability signal that isn't being weaponized for contrarian leverage.

Here's what's being missed: the opponent model uses chalk-biased SEED_PICK_RATES. That means the field systematically overweights chalk. A better probability model should identify games where the chalk is most wrong — upsets that are likelier than the field believes — and deliberately over-index there. This is Kelly-style differentiation, not accuracy optimization.

The adjacent opportunity: conditional pool strategy. Don't use stochastic sampling uniformly. Use your ML probabilities to identify 3-5 "leverage picks" per bracket where your model diverges most from the field's chalk distribution, then bias sampling toward those divergences. You're currently sampling with better probabilities but not exploiting the gap between your probabilities and the field's probabilities.

Concretely: compute P_model(upset) - P_field(upset) for every first-round game. Bracket generation should over-weight upsets where this delta is largest. That's where your Brier edge becomes a pool edge.

The months of ML work aren't wasted — they produced real probability signal. The system just isn't using that signal against the specific opponent distribution it's competing against. That's a strategy layer problem, not a model quality problem. Build the contrarian sampling layer before declaring defeat.

### The Outsider

The opponent model is your blind spot.

You've spent months refining your probability estimates and measuring how well they predict game outcomes. But pool performance is not about predicting games — it's about predicting what other people pick, and then picking differently in the right places.

Your baseline compares against SEED_PICK_RATES with chalk bias. That's a 2004-era opponent model. Real 2026 pool players have ESPN analytics, KenPom, bracket builders, and four weeks of sports radio. If they've gotten smarter, your edge from any model shrinks because you're no longer finding under-picked upsets — everyone's found them.

The Brier score improvement is real but answering the wrong question. You have a better probability model, but a better probability model only wins pools if you translate it into contrarian picks at the right seed lines. A team at 40% to win a game is worth picking if everyone else ignores them. A team at 60% is worthless to pick if 80% of the field already has them.

The concrete test: compare your model's pick frequencies against actual pool pick rates (ESPN provides public bracket data each year). Where does your model diverge from the crowd? Are those divergences in the right direction — i.e., do those contrarian picks actually win more than the crowd expects?

If you don't have that data, you're flying blind on the variable that actually determines your pool rank.

The months of ML work isn't wasted — but it's currently solving for accuracy when it should be solving for calibrated contrarianism. Those are related but not the same problem.

### The Executor

The Brier score advantage is real and it's not showing up in pools because your opponent model is wrong. That's the Monday morning diagnosis.

Your "opponents" are modeled as seed-based chalk pickers. Real 1000-person pools have chalk pickers, upset hunters, star-player chasers, and bracket copiers. If your opponent distribution is wrong, your pool rank simulation is meaningless — you could have a massive edge and never see it.

Here's the action sequence:

**This week:** Get real bracket data. ESPN Tournament Challenge and Yahoo Sports publish aggregate pick percentages by game. Pull the actual 2024 or 2025 pick distributions and replace SEED_PICK_RATES with real human pick frequencies. Run the backtest again. If noseed's Brier advantage suddenly shows up as pool rank improvement, you found the bug.

**If that doesn't move the needle:** The issue is differentiation, not prediction quality. Winning pools requires being differently right than the field. A model that's 6% better at predicting outcomes but picks similar brackets to everyone else earns nothing. Audit how often noseed's top bracket differs from the field consensus. If the predicted winner appears in >60% of human brackets, noseed needs a contrarian sampling strategy, not better probabilities.

**If neither works:** The signal exists (Brier score proves it), but the pool format doesn't reward it. Consider whether the project goal should shift to a different competition format — head-to-head, or pools with scoring rules that reward calibration rather than just upset picks.

Do not abandon the ML model. Do fix the opponent model first. That's the fastest path to knowing whether you have something.

---

## Peer Reviews

### Anonymization Mapping
- Response A = The First Principles Thinker
- Response B = The Executor
- Response C = The Outsider
- Response D = The Expansionist
- Response E = The Contrarian

### Review 1
**Strongest: B** — concrete, falsifiable action sequence. The only response that gives an ordered plan with a verification gate.
**Biggest blind spot: D** — assumes model upset probabilities are correct. If the Brier improvement is in calibration on chalk games, upset deltas may be noise.
**All missed:** Scoring rules. Pool rank depends on point-weighting by round. Brier improvement in early rounds (low leverage) would be worthless for pool rank.

### Review 2
**Strongest: B** — tells you what to do Monday with a falsifiable test.
**Biggest blind spot: D** — prescribes P_model - P_field but P_field data doesn't exist yet. Assumes the broken proxy is usable.
**All missed:** Scoring structure. Contrarian sampling in early rounds is irrelevant noise. Leverage is in rounds 3-6 where points multiply.

### Review 3
**Strongest: B** — actionable, not just analytical.
**Biggest blind spot: D** — proposes fix using the broken SEED_PICK_RATES proxy.
**All missed:** Scoring function mismatch. Brier-calibrated model may be structurally irrelevant in pools that reward upsets with 10x points. Could explain Brier/rank disconnect independently of opponent model.

### Review 4
**Strongest: B** — distinguishes two failure modes with verification gate.
**Biggest blind spot: D** — skips diagnostic step, could amplify wrong signal.
**All missed:** Scoring function. If noseed's edge is front-loaded in low-stakes games, structurally incapable of moving pool rank.

### Review 5
**Strongest: B** — falsifiable condition.
**Biggest blind spot: D** — greedy per-game contrarianism ignores bracket cascade effects (round 1 upsets change who's available in rounds 2-5).
**All missed:** Scoring structure. Brier decomposition by round needed. If edge is in early rounds only, no amount of strategy fixes the problem.

---

## Chairman Synthesis

### Where the Council Agrees
The opponent model is broken. SEED_PICK_RATES is a stale proxy for actual human pick distributions. Every advisor agrees this is the structural defect explaining the Brier/rank disconnect. Better probability estimates are being neutralized by a miscalibrated picture of what the field is doing. The Brier advantage is real signal. The ML work is not wasted — it is being misapplied.

### Where the Council Clashes
The Expansionist says start computing P_model - P_field per game now. The Executor says fix the opponent model first, then see if the advantage appears automatically. The peer reviews sided with the Executor: you cannot optimize against a distribution you have not yet measured.

### Blind Spots the Council Caught
The peer reviews caught something every advisor missed — the most important structural point: Brier score improvement may be concentrated in early rounds with low scoring leverage. Pool scoring weights late rounds 4-32x more. If noseed's calibration edge is in R64/R32, it is structurally incapable of moving pool rank regardless of opponent model or contrarian strategy.

Note: Existing per-round Brier data shows noseed's largest advantages at F4 (BSS=+0.138) and E8 (BSS=+0.081), suggesting the scoring mismatch may NOT be the binding constraint. But it should be verified explicitly.

### The Recommendation
Two parallel diagnostics:
1. Replace SEED_PICK_RATES with real ESPN aggregate pick percentages and rerun the stochastic backtest unchanged. If rank improves, found the bug. If not, ruled out most likely culprit.
2. Decompose Brier improvement by round in pool-scoring-weighted context. Verify edge concentrates in high-leverage rounds (S16+).

Do not build contrarian sampling until both diagnostics complete.

### The One Thing to Do First
Swap SEED_PICK_RATES for real ESPN aggregate pick percentages and rerun the backtest. Fastest falsifiable test. Either outcome advances the diagnosis.
