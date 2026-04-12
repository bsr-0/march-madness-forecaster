# LLM Council Transcript — 2026-04-12

## Original Question
Given the retrospective scorer results across 14 years (2011-2025) plus manual 2026 data, what critical improvement steps should we take to target next year's (2027) pool strategy?

## Framed Question
Given 15 years of retrospective bracket scoring data and a winner-take-all N=31 pool, what are the most impactful improvements to make before March 2027?

Context:
- March Madness bracket optimizer. Pipeline: Data → Features → ML Model → Calibration → MC Simulation → Bracket Construction → Pool Optimization.
- ML model (LGB/XGB/LR ensemble) has BSS ≈ 0 vs seed baseline — zero predictive value.
- Three strategies tested (14 automated years + 1 manual):
  - seed (chalk): 1236 avg pts, 4/15 champ correct, beats median 11/14, never beats winner
  - torvik (Torvik barthag + Log5): 1201 avg pts, 2/15 champ correct, beats winner 2/14
  - det_champ96 (champion-first r=0.96): 953 avg pts, 2/15 champ correct, beats winner 1/15 (2026)
- 2026: det_champ96 scored 1440 (Michigan champ, 4/4 F4, 4/4 E8). Torvik ~1330.
- All modes pick 4 one-seeds for F4 (historical avg 1.5)
- N=14, power 9-16%
- System HAD winning bracket but ranked it lower by EV

## Advisor Responses

### The Contrarian
Fatal flaw: 14 data points at 9-16% power = drawing conclusions from noise. What's broken in order: 1. Ranking function is the real failure — optimizing EV when should optimize win probability. EV favors chalk, win probability favors correlated upsets. 2. F4 construction broken — all 1-seeds = consensus bracket, can't win winner-take-all. 3. Don't waste time improving ML model. 4. N=14 backtesting will actively mislead — need theoretical analysis of win probability instead. "Stop tuning the engine. The steering wheel is broken."

### The First Principles Thinker
Wrong objective function. Pipeline built to predict game outcomes but goal is P(rank 1 of 31). This is game theory, not forecasting. Key leverage: 1. Model the field, not the tournament — maximize P(winning | field distribution). 2. Stop picking all 4 one-seeds for F4. 3. Kill the ML ensemble. 4. Treat bracket construction as a portfolio problem — need 3.2% win rate, not best average bracket. "The question isn't how to predict games better. It's how to pick differently from 30 people and be right when they're wrong."

### The Statistician
Power analysis already told you: can't learn from 14 points. BSS≈0 is the most important number — pipeline is optimizing over noise. 2026 det_champ96 is textbook survivorship bias (N=1 from high-variance distribution, avg 953 vs seed's 1236). Simulated opponent methodology needs scrutiny — max-of-30 sensitive to tail shape. What matters: 1. Fix F4 calibration. 2. Abandon ML until BSS > 0. 3. Accept optimal strategy is high-variance contrarian — game theory problem.

### The Outsider
Built a prediction engine when needed a gambling strategy. Finishing second pays nothing, so need maximum variance not maximum EV. Three things: 1. Model the pool, not the tournament. 2. Stop picking all 4 #1 seeds. 3. Submit multiple bracket archetypes if allowed, otherwise pick highest-ceiling bracket. 2026 proves the concept — HAD the winning bracket but system buried it.

### The Executor
Monday-morning plan: 1. Fix 2026 scoring data (broken team IDs, half-day). 2. Kill the Final Four bottleneck — force at least 2 non-one-seeds into F4 using historical rates (one day). 3. Change ranking function from EV to P(finish first) via Monte Carlo (2-3 days). 4. Stop touching ML model. Three tasks, one week total.

## Peer Reviews

Anonymization mapping: A=Executor, B=First Principles, C=Outsider, D=Contrarian, E=Statistician

### Review 1
Strongest: D (Contrarian) — names epistemic problem first. Weakest: A (Executor) — punch list without engaging measurement problem. ALL MISSED: Nobody questions the simulated opponent pool. P(1st) optimization is only as good as the opponent model.

### Review 2
Strongest: D (Contrarian). Weakest: A (Executor) — treats 2026 as diagnostic evidence rather than single observation. ALL MISSED: None question whether opponent model is well-characterized. If field is diverse, contrarian strategy collapses.

### Review 3
Strongest: B (First Principles) — coherent game theory framing with 3.2% target. Weakest: D (Contrarian) — "astrology" dismissal is lazy; F4 bug is deterministic not statistical. ALL MISSED: Nobody engages with existing codebase for specific code changes.

### Review 4
Strongest: B (First Principles). Weakest: D (Contrarian) — nihilism not a strategy. ALL MISSED: Calibration-to-pool interaction — slightly miscalibrated model overestimating upsets might be MORE valuable.

### Review 5
Strongest: B (First Principles). Weakest: D (Contrarian). ALL MISSED: Nobody examined the scoring function — late-round points weighted heavily, optimal strategy depends on point structure.

## Chairman Synthesis

### Where the Council Agrees
Every advisor independently reached the same core diagnosis: the system is optimizing the wrong objective function. The pipeline maximizes expected bracket score (EV), but in a winner-take-all pool, EV-optimal brackets are losers. The 2026 result is the proof.

Four of five agree ML ensemble should be frozen or killed. BSS≈0 means not beating seed baselines.

All five agree F4 construction is broken. Four one-seeds = consensus bracket. Historical avg ~1.5.

### Where the Council Clashes
Can you learn from 14 years of backtesting? Contrarian/Statistician say no. Executor/First Principles say yes. Chairman sides with yes for binary/directional tests, no for parameter tuning.

Is ML worth keeping? Chairman sides with freeze, not kill. Not hurting if not tuned. Preserves infrastructure.

### Blind Spots the Council Caught
- Opponent model unexamined — P(1st) only as good as simulated opponents
- Scoring function structure ignored — championship points may dominate
- Calibration-to-pool interaction — deliberate upset overestimation could be a feature

### The Chairman's Take
Council nailed diagnosis, sloppy on prescription. "Optimize P(1st)" requires an opponent model that is itself unvalidated. Risk of replacing one unvalidated optimization with another.

2026 retrospective is more diagnostic than acknowledged: examine which picks the EV ranker penalized to design a non-statistical ranking fix.

The real unlock: separate game probability estimation from bracket construction strategy. Mediocre probabilities can still win pools if construction solves the game-theory problem.

### Critical Actions
1. Fix bracket ranking: replace EV with P(1st) using MC simulation against modeled opponent field. Gate: system ranks det_champ_first bracket in top 3 on 2026 re-run.
2. Fix F4 construction: enforce historical base rates (cap one-seeds at 2). Gate: average one-seed F4 count between 1-2 across generated brackets.
3. Validate opponent model against real pool results. Gate: simulated max-of-30 within 15% of observed, or discrepancy documented.
