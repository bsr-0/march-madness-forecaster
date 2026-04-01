**March Madness Pipeline**
Metric Optimization & Evaluation Strategy v2.0
System Prompt & Protocol for Complete Repository Refactor

> **STATUS (2026-04-01): SUPERSEDED.** This protocol describes the old ensemble architecture (LightGBM/XGBoost/stacking, 86-dim features, learned selection). The production pipeline has since been simplified to a 7-feature logistic regression with two-stage domain adaptation (regular-season training + multi-year tournament calibration). A baseline experiment confirmed the complex ensemble adds no value over the simple model. See README.md for current architecture. This document is retained for historical context only.

This repository builds a machine learning pipeline to predict the outcomes of NCAA March Madness tournament games by producing calibrated win probabilities for every possible matchup. Those probabilities serve two distinct downstream goals: submitting optimized predictions to the Kaggle March Machine Learning Mania competition, and constructing high-value brackets for ESPN and other bracket pool contests.
**This document serves as the authoritative system prompt and refactoring protocol. **Every architectural decision, metric choice, and implementation step is specified here. An AI assistant or developer should be able to refactor the entire repository by following this document section by section.

# 1. The Dual Optimization Framework

Using separate optimization strategies for Kaggle and ESPN bracket pools is not bad practice. It is the only correct approach. These are fundamentally different problems with different objective functions, and every serious practitioner in this space (FiveThirtyEight, PoolGenius/TeamRankings, Kaggle winners) uses separate optimization for probability quality vs. bracket construction.

## 1.1 Core Distinction


| Dimension | Kaggle Submission | ESPN Bracket Pool |
| --- | --- | --- |
| Objective | Minimize Brier score across all 63+ games | Maximize P(finish 1st) given pool size N |
| Output | Probability for every possible matchup | A single bracket: 63 binary picks |
| What matters | Probability calibration and accuracy | Being correct WHERE others are wrong |
| Scoring | Brier score (mean squared error) | Points per correct pick, escalating by round |
| Optimal strategy | Report most honest probabilities | Deliberately deviate on high-leverage picks |


**Key insight: **The Kaggle pathway is a probability estimation problem. The ESPN pathway is a game theory problem that uses probabilities as an input. Both need excellent probability estimates, but the ESPN pathway adds a second optimization layer on top.

## 1.2 Shared Foundation, Separate Endpoints

The two pathways share approximately 80% of the pipeline: data ingestion, feature engineering, base model training, LOYO cross-validation, and ensemble construction. They diverge at post-processing and output. This is analogous to quantitative finance where the same risk model serves both regulatory reporting (accuracy) and portfolio construction (relative positioning).


# 2. Data Integrity: Point-in-Time Feature Protocol

**CRITICAL REQUIREMENT: **Every feature used in LOYO cross-validation must be a Point-in-Time (PIT) snapshot. This is the single most common source of data leakage in sports modeling and must be enforced as a hard architectural constraint.

## 2.1 The Problem

KenPom ratings, NET rankings, Sagarin ratings, and similar team-strength metrics are updated continuously throughout the season. If your LOYO fold for 2019 uses end-of-season KenPom ratings to predict tournament games that happened in March 2019, those ratings contain information from games played during and after the tournament. This is data leakage, and it inflates LOYO performance in ways that will not generalize to live prediction.

## 2.2 The Temporal Triage Protocol

Implement a three-tier classification for every feature in the pipeline:

| Tier | Definition | Examples | Rule |
| --- | --- | --- | --- |
| Tier 1: Static | Does not change within a season | Seed, conference, preseason poll rank | No restriction |
| Tier 2: Cumulative | Computed from game-by-game data you control | Season win %, avg margin of victory, SOS | Must use only games played before tournament start date for that LOYO year |
| Tier 3: External Rating | Sourced from third-party systems that update continuously | KenPom AdjEM, NET ranking, Sagarin, BPI | Must use the snapshot from Selection Sunday morning (or latest pre-tournament date available) for that LOYO year |


## 2.3 Implementation

- Create a features/snapshots/ directory containing dated CSV files of all Tier 3 features (e.g., kenpom_2019-03-17.csv).
- Add a PIT enforcement check to the LOYO data loader: for each fold year, assert that no Tier 2 or Tier 3 feature contains data from after Selection Sunday of that year.
- Log a warning if any feature’s latest data point is within 48 hours of a tournament game in that fold. This catches edge cases where ratings were scraped too late.
- Document each feature’s tier classification in a features/MANIFEST.yaml file that serves as the single source of truth.


# 3. Kaggle Pathway: Probability Quality Optimization


## 3.1 The Multi-Metric Selection Gate

A model must pass all three thresholds to be considered for ensembling:

| Metric | Threshold | Purpose |
| --- | --- | --- |
| Brier Score | < 0.190 (LOYO) | Overall probability accuracy |
| Log Loss | < 0.560 (LOYO) | Penalizes overconfident errors exponentially |
| Brier-Log Divergence | < 0.015 | Flags metric gaming between scoring rules |


Among models passing all three gates, rank by Brier score (since that is the Kaggle evaluation metric). This preserves Brier as the final arbiter while preventing models that game Brier at the expense of calibration.

## 3.2 Bayesian Model Averaging for the Ensemble

**Replace the current grid-search ensemble weight optimization with Bayesian Model Averaging (BMA). **Rather than selecting a single “best” model or a fixed-weight ensemble, BMA produces a weighted average across all models that pass the gate, where the weights are posterior model probabilities that account for model uncertainty.

### Why BMA Over Fixed-Weight Ensembles

- Fixed-weight ensembles are fragile when models are close in performance. A model that barely clears the Brier gate at 0.189 gets treated identically to one at 0.170 in a winner-take-all selection.
- BMA naturally hedges against model uncertainty. If three models are within 0.005 Brier of each other, BMA assigns them similar weights rather than picking one arbitrarily.
- BMA has strong theoretical backing for probabilistic forecasting. Raftery et al. (2005) demonstrated that BMA-calibrated forecast ensembles produced sharper and better-calibrated predictions than both individual models and simple ensemble means in weather forecasting, a domain with direct analogies to tournament prediction.
- BMA produces proper posterior predictive distributions, meaning the resulting probabilities inherit the calibration properties of the component models rather than distorting them the way fixed-weight optimization can.

### Implementation

- After the multi-metric gate, collect all passing models and their LOYO probability predictions.
- Compute BMA weights using the EM algorithm: P(M_k | data) is proportional to P(data | M_k) * P(M_k), where P(data | M_k) is the likelihood under each model’s LOYO predictions, and P(M_k) is a uniform prior across passing models.
- The final Kaggle submission probability for each matchup is the BMA-weighted average: P(A beats B) = sum of w_k * P_k(A beats B) across all models k.
- Track the effective number of models (1 / sum of w_k^2) as a diagnostic. If this drops below 1.5, BMA is effectively selecting a single model and you should investigate why.

## 3.3 Core Directives

- Sharpening ban: The BrierOptimalSharpener is prohibited for Kaggle submissions. The power transform p_sharp = 0.5 + sign(p-0.5) * |2p-1|^α risks degrading calibration in exchange for artificial Brier improvement.
- Unweighted Brier for selection: Replace round-weighted Brier with unweighted Brier for model selection. Kaggle round weights (R64=1x through NCG=32x) create instability because late-round sample sizes are tiny (4–28 games across LOYO years).
- Training-selection alignment: LightGBM trains with binary log loss but you select on Brier. Monitor the Brier-Log Divergence metric to catch cases where these objectives disagree. Consider training with a custom Brier-based objective function as a long-term improvement.
- Temporal integrity: All LOYO evaluations must use PIT feature snapshots as specified in Section 2.

## 3.4 Diagnostic Metrics (Track, Do Not Select On)


| Metric | What It Tells You | Red Flag |
| --- | --- | --- |
| Brier Decomposition: Reliability | How well calibrated probabilities are | Reliability > 0.015 |
| Brier Decomposition: Resolution | How well model separates outcomes | Resolution decreasing while Brier improves |
| Classwise ECE (10 bins) | Per-range calibration error | Any bin > 0.10 absolute error |
| Reliability Diagram | Visual calibration check | S-curve or systematic deviation |
| LOYO Accuracy | Raw correct/incorrect rate | Below 68% |
| Brier Skill Score vs Seeds | Improvement over trivial model | BSS < 0.05 |
| BMA Effective Model Count | Ensemble diversity | Below 1.5 |


# 4. ESPN Bracket Pathway: Expected Value Optimization


## 4.1 The Optimization Target

Optimize for: **P(finish in money | pool size N, scoring system S, payout structure).**
This is a fundamentally different metric than Brier score. A bracket maximizing expected points is NOT the same as one maximizing probability of winning. This distinction separates serious pool strategists from everyone else.

## 4.2 Required Inputs

- Game probabilities from the shared model (calibrated probabilities before any sharpening).
- Public pick percentages from ESPN’s “Who Picked Whom,” Yahoo bracket data, or similar sources. This is the most important input that the Kaggle pathway does not need.
- Pool size (N). Strategy changes dramatically: 10-person pool rewards chalk; 1,000-person pool rewards contrarian picks.
- Scoring system (ESPN standard: 10-20-40-80-160-320). Flatter systems shift value to early rounds.
- Payout structure. Winner-take-all pools maximize upside. Top-half-pays pools reward safety.

## 4.3 Leverage Calculation

***Leverage = P(team advances) − Public Pick Rate***
Positive leverage means the team is undervalued by the public. If your model gives a team 22% to win the tournament but only 8% of ESPN brackets pick them, that +14% gap is where your edge lives.

## 4.4 Path-Dependent Leverage and Quadrant Correlation

**CRITICAL: **Leverage is not static. It is path-dependent through the bracket structure. A high-leverage pick is worthless if you don’t also pick the teams that get them to the high-value rounds.
**The Quadrant Correlation Rule: **If your champion or Final Four pick is a high-leverage Cinderella or contrarian selection, your picks in the preceding rounds of that same quadrant (region) must be chalk (favorites). You are protecting the path that delivers your high-value upset to the late rounds where it earns maximum points.
Example: If you pick a 3-seed as your champion because they have +12% leverage, but you also pick a 12-over-5 upset in their region’s first round, you have created an internally inconsistent bracket. The 12-seed upset could eliminate a team your champion needs to beat in the Sweet 16, reducing your champion’s probability of reaching the Final Four.

### Implementation

- For each candidate champion/F4 pick, compute the conditional path probability: P(champion reaches NCG) = product of P(win) for each game on the path through their region.
- When evaluating candidate upsets in earlier rounds of the same region, compute the path disruption cost: how much does this upset reduce P(champion reaches F4)?
- Only select early-round upsets in a region where the path disruption cost is below a threshold (e.g., < 3% reduction in champion path probability), OR where the upset is in the opposite half of that region from your champion’s path.
- Cross-region upsets (picks in regions where your F4 team is NOT from) can be more aggressive since they cannot disrupt your high-value path.

## 4.5 Strategic Tilt by Pool Size


| Pool Size | Strategy | Champion Pick | Upset Frequency |
| --- | --- | --- | --- |
| Small (N < 50) | Chalk-heavy. Let others make mistakes. | 1-seed or 2-seed with best win probability | 0–2 first-round upsets maximum |
| Medium (50–500) | Moderate contrarian. Find 1–2 leverage points. | Team with largest positive leverage gap | 2–4 upsets where leverage > +5% |
| Large (N > 500) | Aggressive contrarian. Differentiate or lose. | Undervalued 2/3-seed with high ceiling | Strategic upsets in all regions |


## 4.6 The Bracket Optimization Algorithm

- Simulate the tournament 10,000+ times using game probabilities.
- For each simulation, generate N–1 opponent brackets from public pick distribution.
- Score your candidate bracket and all opponents under the pool’s scoring system.
- Track P(finish 1st) and P(finish in money) for each candidate bracket.
- Enforce quadrant correlation constraints during bracket construction.
- Iterate over candidate brackets (varying champion, F4, key upsets) to maximize P(finish in money).

## 4.7 ESPN Evaluation Metrics


| Metric | Definition | Target |
| --- | --- | --- |
| Simulated Pool Rank %ile | Average percentile finish across LOYO simulations | > 80th |
| P(Top 10%) Rate | Fraction of simulations finishing in top decile | > 30% |
| Path Protection Score | P(champion reaches F4) given all picks in their region | > 85% of unconditional P |
| Leverage Accuracy | Win rate among picks with leverage > +5% | > 55% |


# 5. Shared Pipeline Architecture


## 5.1 What Is Shared

- Data ingestion, feature engineering, and preprocessing (with PIT enforcement)
- Base model training (LightGBM/XGBoost with binary log loss objective)
- LOYO cross-validation framework
- Multi-metric quality gate (Brier + log loss + divergence check)
- Bayesian Model Averaging ensemble construction
- Isotonic regression or Platt scaling for base calibration

## 5.2 Where They Diverge


| Stage | Kaggle Path | ESPN Path |
| --- | --- | --- |
| Post-calibration | No sharpening. Output honest probabilities. | Sharpening permissible (validated OOS). Bracket optimizer benefits from decisive probabilities. |
| Output format | CSV of P(A beats B) for all matchups | Single filled bracket (63 binary picks) |
| Optimization | None — BMA probabilities are the submission | Monte Carlo simulator maximizing P(finish in money) |
| Evaluation | Brier (primary) + decomposition diagnostics | Simulated pool rank percentile |
| Round weighting | Unweighted for selection | Implicitly handled by ESPN scoring (320 pts champion) |


# 6. Implementation Checklist


## 6.1 Phase 1: Immediate (High Impact, Low Risk)

- Add log loss computation to every model evaluation point alongside Brier.
- Add Brier decomposition (reliability, resolution, uncertainty) via Murphy 1973 decomposition to LOYO reporting.
- Add Brier Skill Score vs. seed baseline. If BSS ≈ 0, the ML pipeline adds minimal value.
- Gate the BrierOptimalSharpener behind a flag: OFF by default for Kaggle.
- Create features/MANIFEST.yaml classifying every feature into Tier 1/2/3 per the PIT protocol.
- Add PIT enforcement assertions to the LOYO data loader.

## 6.2 Phase 2: Kaggle Refactor

- Implement BMA ensemble (EM algorithm for posterior model weights) to replace grid-search weight optimization.
- Implement the multi-metric gate in _select_best_single_model.
- Switch model selection from round-weighted to unweighted Brier.
- Add Brier-Log Divergence computation and rejection logic.
- Add BMA effective model count diagnostic.

## 6.3 Phase 3: ESPN Path Build-Out

- Build public pick distribution scraper (ESPN Who Picked Whom, Yahoo bracket data).
- Implement Monte Carlo bracket simulator with opponent bracket generation.
- Build leverage calculation engine: P(advance) minus public pick rate per team per round.
- Implement quadrant correlation constraints and path protection scoring.
- Build bracket optimizer searching over candidate brackets to maximize P(finish in money).
- Build LOYO backtesting for ESPN path using historical public pick data.

## 6.4 Phase 4: Long-Term Research

- Explore point-spread prediction as alternative outcome variable. Models on point differential consistently outperform binary outcome models.
- Consider calibration-first training pipeline. Walsh & Joshi (2024) showed calibration-optimized models produce ~70% higher returns in sports contexts.
- Multi-bracket ESPN strategy: generate portfolio of 3–5 brackets with varying risk profiles.
- Investigate custom Brier-based training objective for LightGBM to eliminate training-selection metric mismatch.


# 7. Metric Quick Reference


| Metric | Use For | What It Measures | Pipeline Location |
| --- | --- | --- | --- |
| Brier Score | Kaggle ranking | Mean squared probability error | Model selection (after gates) |
| Log Loss | Kaggle gate | Exponential penalty for overconfidence | Model selection gate |
| Brier-Log Divergence | Kaggle gate | Detects metric gaming | Model selection gate |
| Brier Decomposition | Both (diagnostic) | Separates calibration from resolution | LOYO reporting |
| Brier Skill Score | Both (diagnostic) | Improvement over seed baseline | LOYO reporting |
| Classwise ECE | Both (diagnostic) | Per-bin calibration error | LOYO reporting |
| Reliability Diagram | Both (visual) | Calibration shape | LOYO reporting |
| BMA Effective Count | Kaggle (diagnostic) | Ensemble diversity | Ensemble stage |
| LOYO Accuracy | Both (sanity) | Raw win/loss rate | LOYO reporting |
| Simulated Pool Rank | ESPN primary | Average pool finish | ESPN backtesting |
| P(Top 10%) Rate | ESPN primary | Top-decile finish frequency | ESPN backtesting |
| Leverage | ESPN primary | P(advance) minus public pick rate | Bracket construction |
| Path Protection Score | ESPN primary | Champion path viability | Bracket construction |


# 8. System Prompt for Repository Refactoring

*Copy the text below as the system prompt when using an AI assistant to refactor the codebase.*

**BEGIN SYSTEM PROMPT**
You are refactoring a March Madness prediction pipeline. This codebase predicts NCAA tournament game outcomes by producing calibrated win probabilities. It serves two distinct goals: (1) Kaggle March Machine Learning Mania competition submissions evaluated on Brier score, and (2) ESPN bracket pool optimization evaluated on simulated pool rank percentile.
**ARCHITECTURAL PRINCIPLES:**
The pipeline has a shared foundation and two diverging pathways. Never optimize both pathways with the same metric. Kaggle = probability quality. ESPN = game theory.
**DATA INTEGRITY — POINT-IN-TIME (PIT) PROTOCOL:**
Every feature is classified as Tier 1 (static: seed, conference), Tier 2 (cumulative: season stats you compute), or Tier 3 (external ratings: KenPom, NET, Sagarin, BPI). For LOYO cross-validation, Tier 2 features must use only pre-tournament games. Tier 3 features must use the Selection Sunday snapshot for that year. A features/MANIFEST.yaml file documents each feature’s tier. The LOYO data loader must assert PIT compliance and reject any fold where a feature contains post-tournament data.
**KAGGLE PATHWAY:**
Multi-Metric Gate: Models must pass Brier < 0.190, Log Loss < 0.560, and Brier-Log Divergence < 0.015 before consideration. Among passing models, rank by Brier. Ensemble via Bayesian Model Averaging (BMA) using EM algorithm for posterior model weights, NOT fixed grid-search weights. BMA weights = P(data | model) * P(model), normalized. Track BMA effective model count (1 / sum of w_k^2); flag if < 1.5. BrierOptimalSharpener is PROHIBITED for Kaggle. Use unweighted Brier, not round-weighted. Track diagnostics: Brier decomposition (reliability, resolution), classwise ECE, reliability diagrams, BSS vs seed baseline, LOYO accuracy.
**ESPN PATHWAY:**
Optimize for P(finish in money | pool size N, scoring system S). Requires public pick percentages (ESPN Who Picked Whom). Core metric: Leverage = P(advance) minus public pick rate. Path-dependency rule: if champion pick has high leverage, preceding picks in that region must be chalk to protect the path. Implement quadrant correlation: compute path disruption cost for each upset and reject if it reduces champion path probability by > 3%. Monte Carlo simulator: 10,000+ tournament simulations, N-1 opponent brackets from public picks, score under pool system, track P(finish 1st) and P(in money). Strategic tilt: small pools favor chalk, large pools favor contrarian champion picks.
**KEY FILES TO MODIFY:**
src/pipeline/stages/baseline_training.py — _select_best_single_model: add log loss gate, BMA ensemble. src/ml/calibration/brier_optimal.py — gate sharpener behind flag, OFF for Kaggle. src/pipeline/config.py — add enable_brier_sharpening: false default, add log_loss_max and brier_log_divergence_max config values. Create features/MANIFEST.yaml. Create src/pipeline/stages/pit_validation.py. Create src/espn/leverage.py, src/espn/bracket_optimizer.py, src/espn/mc_simulator.py.
**METRICS ALWAYS REPORTED IN LOYO:**
Brier Score, Log Loss, Brier-Log Divergence, Brier Decomposition (reliability + resolution + uncertainty), Brier Skill Score vs seed baseline, Classwise ECE (10 bins), LOYO Accuracy, BMA Effective Model Count.
**WHAT NOT TO DO:**
Never optimize a single metric for both pathways. Never use end-of-season ratings in LOYO without PIT snapshots. Never apply BrierOptimalSharpener to Kaggle submissions. Never use round-weighted Brier for model selection. Never pick high-leverage upsets that disrupt your champion’s path through their region.
**END SYSTEM PROMPT**
**Sources: **Lopez (2018) on log loss limitations in NCAA tournaments; Walsh & Joshi (2024) on calibration vs accuracy for sports betting; Murphy (1973) Brier decomposition; Raftery et al. (2005) on BMA for ensemble calibration; Kaggle March Machine Learning Mania rules (2023–2026); PoolGenius/TeamRankings bracket optimization; FiveThirtyEight March Madness methodology; Gneiting & Raftery (2007) on proper scoring rules.