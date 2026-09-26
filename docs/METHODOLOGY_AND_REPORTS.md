# Methodology, validation, and reports

This is the canonical methodology and report-governance document. The audit protocol below defines how model and bracket claims are validated; the final sections define where current and historical project documentation belongs.

## Exact Execution & Failure Protocols

### A. Execution order for a complete foundational audit

When conducting a complete foundational audit, execute checklist items 1→18 sequentially. Do not skip, reorder, or combine items within that audit. For a season-specific release, [the project roadmap](./PROJECT_ROADMAP_2027.md) defines scope and identifies which protocol checks are required for each phase; unchecked items outside that scope are not automatic release blockers.

### B. Existing implementation is not ground truth

Treat current UI behavior, production code, cached artifacts, prior audits, and documentation as claims to verify. Do not preserve a component because it is already shipped. Statistical correctness takes precedence over backward compatibility.

### C. No optimization during foundational audit

Do not tune models, hyperparameters, referee choice, weighting, candidate generation, or bracket strategies while determining whether the existing methodology is valid. First establish correctness; optimization comes afterward.

### D. Evidence requirement

Every reported result or gate decision must have an evidence record. Include:

| Field | Required content |
|---|---|
| Status | PASS, FAIL, or INDETERMINATE; label descriptive/historical results as such |
| Question and decision | What was tested and what decision the result informs |
| Code and data identity | Commit/revision, data versions or source provenance, and point-in-time cutoff |
| Evaluation design | Target seasons, baseline/comparison set, metric, and scoring/settings |
| Uncertainty and run settings | Statistical method, random seeds, simulation/trial counts, and configuration, where applicable |
| Reproduction | Exact command/procedure and required versioned inputs |
| Result and impact | Estimate, uncertainty, limitations, affected conclusions/downstream gates |
| Evidence location and currency | Artifact/report path and whether current, superseded, or historical |

For fields that do not apply, state N/A rather than silently omitting them. A PASS/FAIL must cite the exact code/data path and include a quantitative check where feasible. "Looks correct" is not evidence.

### E. Bug protocol

If a defect is found:

1. stop the affected experiment;
2. document the defect and its downstream impact;
3. determine whether prior results are invalidated;
4. fix the defect only if the fix is unambiguous;
5. add a regression test;
6. rerun every downstream gate affected by the fix.

Never patch only the displayed result.

### F. Leakage protocol

Any discovered target leakage is an automatic FAIL for the affected methodology. Identify the leakage path, repair it, and invalidate/re-run all affected validation results.

### G. Ambiguous methodology protocol

If the correct interpretation cannot be established from code, data, or documented requirements, mark INDETERMINATE. Do not choose the interpretation that favors the existing model or strategy.

### H. Statistical uncertainty protocol

Do not call a difference meaningful merely because its point estimate is larger. Report uncertainty using the pre-registered method and distinguish:

* PASS;
* FAIL;
* INDETERMINATE.

"Not significant" is not automatically evidence of equality.

### I. Post-hoc discovery protocol

If an unexpected result suggests a new hypothesis, record it as a follow-up experiment. Do not modify the current gate, referee set, metric, sample window, or weighting scheme after seeing results.

### J. Failed gate protocol

A failed gate means:

* stop downstream optimization that depends on it;
* identify affected conclusions;
* repair/re-design if appropriate;
* rerun from the earliest affected checkpoint.

Do not average a failure away.

### K. UI-impact protocol

If the statistically correct methodology differs from the current UI:

* report the discrepancy;
* prioritize statistical correctness;
* update the UI only after the methodology is validated.
  The UI must conform to the validated methodology, not determine it.

### L. Reproducibility requirement

Every final result must be reproducible from a clean checkout using documented commands and versioned inputs. Record commit hash, data versions, random seeds, simulation counts, and configuration.

### M. Final decision rule

At completion, produce a gate table:

| # | Gate | PASS/FAIL/INDETERMINATE | Evidence | Downstream impact |
|---|------|--------------------------|----------|--------------------|

For a complete foundational audit, proceed to final optimization only when all foundational gates PASS or an explicit decision documents why an INDETERMINATE item is acceptable; any unresolved foundational FAIL blocks that audit's optimization decision. For the 2027 prospective release, the active gate and release fallback in [the roadmap](./PROJECT_ROADMAP_2027.md) control.

### N. Investigation record

Before a non-routine audit or roadmap investigation, record:

1. Question and decision it will inform.
2. Relevance to the active roadmap gate; if none, put it in the parking lot.
3. Predeclared method, comparison, data window, and metrics.
4. Timebox and exit criterion.

At completion, use the evidence fields above and record the decision. Routine edits and simple lookups do not require a separate investigation record.

## Final Methodology & Validation Audit

**Instruction:** Execute these steps in order. This is a foundational statistical audit, not a UI-preservation exercise. Treat existing UI behavior, production code, prior conclusions, and documented assumptions as hypotheses—not evidence. If a component is statistically unsound, contains leakage/bugs, or rests on an unjustified assumption, flag and fix/re-design it even if that changes the UI. Do not optimize advanced bracket strategies until foundational gates pass.

### 1. Trace the actual production path

Trace UI → artifacts → probabilities → construction → scoring.
PASS: shipped behavior exactly matches documented methodology.
FAIL: undocumented divergence, stale artifact, or incorrect implementation.

### 2. Audit pairwise probability mathematics

Verify Log5/margin formulas, symmetry, calibration, clipping, temperature scaling, and neutral-site handling.
PASS: mathematically correct and directionally consistent.
FAIL: any probability-generation/circularity defect.

### 3. Audit tournament propagation

Verify pairwise probabilities → simulated games → advancement probabilities. Compare analytical marginals against high-volume simulation.
PASS: agreement within Monte Carlo error.
FAIL: systematic discrepancy.

### 4. Audit scoring and objective mathematics

Verify expected score, actual scoring, payout/tie handling, and P(1st) definition.
PASS: objective corresponds to the stated pool objective.
FAIL: scoring/objective mismatch or unjustified independence assumption.

### 5. Audit candidate-space integrity

Determine whether candidate generation can represent materially different plausible brackets and whether search is constrained by candidate availability.
PASS: adequate coverage; no construction artifact drives results.
FAIL: optimizer performance depends materially on missing candidate regions.

### 6. Audit opponent/ownership assumptions

Verify ownership inputs are point-in-time, correctly mapped, and representative. Test reasonable ownership sensitivity.
PASS: results aren't dominated by arbitrary ownership assumptions.
FAIL: P(1st) changes materially under reasonable alternatives without explanation.

### 7. Audit all referees for construction/leakage

For validation season Y, referee inputs must exclude Y tournament outcomes. Audit IDs, coverage, signs, venue, fallbacks, and source timing.
PASS: clean pre-target referee.
FAIL: target leakage, silent fallback, sign/mechanics defect, or inadequate coverage.

### 8. Freeze referee qualification gate

Pre-register calibration qualification using Brier/log loss against actual tournament games. Do not select referees based on P(1st).
PASS: qualified set determined without strategy-performance information.
FAIL: post-hoc referee selection.

### 9. Re-run qualified referee matrix

Use identical frozen candidates and CRN across referees. Report P(1st), expected score, ranks, top-3/top-10, and season-level paired deltas.
PASS: strategy advantage remains positive across qualified referees.
FAIL: material reversal or referee dependence.

### 10. Define the recent-regime methodology

Do not discard older regular-season training data merely because only the last 3 tournaments are the optimization target. Use all strictly pre-Selection-Sunday regular-season data available at each historical target season to estimate model parameters, while using the most recent 3 tournament seasons as the primary regime-selection/optimization window. Retain the 14-season tournament history only as a stability/safety screen.
PASS: older data informs parameter estimation; recent 3 seasons determine configuration preference; no future information enters either.
FAIL: treating 3 seasons as the sole training sample, or using older tournament outcomes to override the pre-registered recent-regime objective.

### 11. Run recent-season leave-one-season-out validation

For each recent target season, fit only from information available before that season and select configurations using earlier recent seasons.
PASS: no target-season tuning; improvement is not isolated to one season.
FAIL: target leakage or performance dependent on one anomalous tournament.

### 12. Test upset-sensitive modeling

Define upsets independently of the realized tournament result where possible (e.g. model-implied favorite loses). Evaluate overall and upset-subset Brier/log loss versus seed baseline. Test a small pre-registered weighting grid.
PASS: upset improvement without unacceptable overall degradation.
FAIL: improvement exists only on tiny/noisy upset sample or materially damages overall calibration.

### 13. Audit regular-season training mechanics

Verify point-in-time features, venue effects, opponent adjustment, season boundaries, and training/test separation.
PASS: every training observation could have existed at prediction time.
FAIL: leakage, venue confounding, or incorrect temporal construction.

### 14. Audit calibration independently of bracket performance

Evaluate reliability by probability bucket, round, seed matchup, and era/recentness.
PASS: probabilities are reasonably calibrated or calibration deficiencies are explicitly modeled.
FAIL: bracket results are being used to conceal materially miscalibrated probabilities.

### 15. Audit Monte Carlo uncertainty

Quantify simulation SE and verify CRN/sample size are sufficient relative to reported strategy differences.
PASS: simulation noise is materially smaller than claimed effects.
FAIL: conclusions change materially with simulation seed/count.

### 16. Audit probability-noise assumptions

Reassess the 0.16 logit-noise mechanism: distinguish uncertainty in estimated probabilities from ordinary Bernoulli game variance.
PASS: interpretation is statistically defensible and sensitivity is documented.
FAIL: noise is double-counting outcome variance or materially drives conclusions without justification.

### 17. Apply final model-selection gate

A configuration may replace production only if it:

* passes historical safety;
* improves/preserves recent-3-season performance;
* does not materially worsen upset performance;
* remains calibrated;
* survives referee LORO;
* improves the actual production pathway.

PASS: all gates satisfied.
FAIL: retain current model or declare result indeterminate; do not cherry-pick.

### 18. Final prospective freeze

Only after all gates pass, freeze model, calibration, opponent model, construction, candidate generation, and scoring before the next Selection Sunday.
PASS: reproducible prospective artifact.
FAIL: any post-outcome tuning invalidates prospective status.

## Final reporting rule

Report separately:

1. predictive calibration/accuracy;
2. bracket-construction performance under qualified referees;
3. simulated P(1st);
4. prospective real-world performance.

Never convert simulated P(1st) into a claim of equivalent real-world improvement without evidence.

## Documentation governance

### Active documentation

- [../README.md](../README.md) — project overview and quick start
- [../CLAUDE.md](../CLAUDE.md) — repository instructions and guardrails
- [PROJECT_ROADMAP_2027.md](./PROJECT_ROADMAP_2027.md) — controlling release phases and decision log
- [METHODOLOGY_AND_REPORTS.md](./METHODOLOGY_AND_REPORTS.md) — this document
- [OPERATIONS.md](./OPERATIONS.md) — analytics, pool settings, storage, delivery, and site-review notes

### Archived documentation

Historical notes and superseded project documents are preserved under [archive/README.md](./archive/README.md). They provide context but are not current operational truth.

### Documentation rules

- Keep current operating guidance in the root docs and the three canonical files in `docs/`.
- Put dated research notes, superseded plans, and historical reports in `docs/archive/`.
- Keep generated evidence and audit outputs under `artifacts/`, with provenance and checksums where applicable.
- Do not duplicate the same policy or methodology claim across multiple active documents.
