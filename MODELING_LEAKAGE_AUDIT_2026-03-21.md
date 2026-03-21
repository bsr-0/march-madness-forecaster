# Modeling & Leakage Audit — First Pass

Date: 2026-03-21

## Scope

This document preserves the first-pass senior audit of the repository's
modeling stack, with a specific focus on:

- temporal integrity,
- contamination risk,
- calibration discipline,
- hyperparameter tuning rigor,
- promotion governance,
- and cache / artifact hygiene as it relates to leakage investigations.

## Executive Summary

The production path is fundamentally a disciplined tabular system: stacked
LightGBM/XGBoost/logistic-style modeling with temperature calibration,
Bayesian Bradley-Terry, spread modeling, and Monte Carlo simulation.

The strongest existing parts of the codebase are:

1. explicit year partitions,
2. tournament-only calibration defaults,
3. LOYO-aware evaluation tooling,
4. calibration leakage guards,
5. production freeze / governance checks.

The biggest remaining concern is not obvious label leakage in a single line of
code; it is **evaluation contamination risk from historical re-use**, plus
**runtime ambiguity introduced by repo-local caches and generated artifacts**.

## Root Cause Analysis — First Pass

### Root cause 1: Retrospective model search pressure

The repo already documents that some historical evaluation is still
retrospective / partially circular.  This is the main statistical leakage
concern and should continue to be addressed with strict dev/eval separation.

### Root cause 2: Cache markers were not bound to execution context

The DAG cache implementation previously skipped work when a marker existed,
without verifying that the marker belonged to the exact same execution
context.  That creates a stale-cache contamination surface when task keys are
too coarse or when contexts drift across seasons / splits / experiments.

### Root cause 3: Ephemeral runtime caches lived inside the repo tree

Local runtime directories such as `__pycache__` and `.pytest_cache` are not
model artifacts, but they clutter leakage investigations and make it easier to
confuse local residue with durable pipeline state.

## First-Pass Remediations Implemented

### 1. Context-bound DAG cache markers

Cache markers now include:

- `protocol_version`
- `context_fingerprint`
- task name
- output key

Stale / legacy markers that do not match the current context are now ignored
and invalidated instead of being silently reused.

### 2. Ephemeral repo cache purge during pre-run checks

Pre-run orchestration now removes transient repo-local runtime caches before
running leakage-sensitive validation steps.

### 3. Added cache hygiene unit tests

New tests cover:

- context fingerprint stability,
- context-sensitive cache invalidation,
- rejection of legacy markers,
- discovery and purge of ephemeral runtime caches.

## Recommended Next Steps

1. Make dev/eval split mandatory for all hyperparameter tuning and feature
   ablations.
2. Emit a frozen post-selection feature manifest for every production run.
3. Add artifact provenance metadata to all saved calibration / validation
   reports.
4. Distinguish durable artifacts from scratch outputs more aggressively in the
   repo layout.
5. Extend context fingerprints to selected high-risk scraper caches if they are
   later used in leakage-sensitive workflows.

## Original Audit Themes Preserved

- Promote on Brier with calibration constraints.
- Use temperature scaling as the default March Madness calibration method.
- Treat GNN / transformer modules as research-only until they show stable
  prospective gains.
- Require significance testing, robustness testing, and fold-level wins before
  production promotion.
- Treat very small retrospective deltas as non-actionable unless they survive
  confidence-interval and contamination scrutiny.
