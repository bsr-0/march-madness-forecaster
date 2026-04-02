---
name: systematic-debugging
description: Use when investigating bugs, test failures, production errors, or performance problems - especially after multiple failed fix attempts or when under time pressure
---

# Systematic Debugging Framework

## Overview

Rigorous four-phase approach to bug resolution that prioritizes understanding root causes before attempting fixes.

**Core principle:** NO FIXES WITHOUT ROOT CAUSE INVESTIGATION FIRST.

Random fixes waste time and create new bugs.

## When to Use

- Test failures
- Production bugs
- Performance problems
- Build failures
- Especially when under time pressure or after multiple failed attempts

## Phase 1: Root Cause Investigation

BEFORE proposing fixes, add diagnostic instrumentation:

- Carefully analyze error messages and stack traces
- Consistently reproduce the issue
- Review recent changes and dependencies
- Gather evidence across component boundaries
- Trace data flow through the call stack

```bash
# For this project, start with:
pytest -x -v --tb=long  # Get full traceback
ruff check src/ --output-format=full  # Check for lint issues
```

## Phase 2: Pattern Analysis

- Locate similar working implementations in the codebase
- Compare working versus broken code
- Identify ALL differences systematically
- Understand component dependencies
- Document findings before proposing solutions

## Phase 3: Hypothesis and Testing

Apply the scientific method:

1. State one clear hypothesis about the cause
2. Make a minimal, isolated change to test it
3. Verify results before proceeding
4. If wrong, return to Phase 1 with new evidence

## Phase 4: Implementation

1. Create a failing test case first (TDD)
2. Implement a single targeted fix
3. Verify the fix works without breaking other functionality
4. Run full test suite: `pytest`

### Architectural Checkpoint

If 3+ fixes have failed: **Question the pattern, don't fix again.** The underlying design may need to change.

## Red Flags - STOP

These indicate you're violating the process:

| Red Flag | Action |
|----------|--------|
| Proposing solutions without investigation | Return to Phase 1 |
| Multiple simultaneous changes | Revert, test one at a time |
| Skipping test creation | Write the failing test first |
| "It's simple, just try this" | That's rationalization - investigate |
| "We're in a hurry" | Rushing causes more bugs |
| 3+ failed fix attempts | Question the architecture |

## Verification

Before claiming a bug is fixed:
- [ ] Root cause identified and documented
- [ ] Failing test written that reproduces the bug
- [ ] Fix implemented and test passes
- [ ] Full test suite passes: `pytest`
- [ ] No regressions introduced
