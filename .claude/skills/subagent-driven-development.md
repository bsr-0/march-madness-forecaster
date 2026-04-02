---
name: subagent-driven-development
description: Use when executing implementation plans with independent tasks in the current session
---

# Subagent-Driven Development

## Overview

Execute implementation plans by dispatching fresh subagents for each task, with two-stage review after each completion.

**Core principle:** Fresh subagent per task + two-stage review (spec then quality) = high quality, fast iteration.

## When to Use

- You have an implementation plan with mostly independent tasks
- Tasks should remain in your current session
- You want structured review after each task

## Process Flow

For each task in the plan:

1. **Dispatch implementer subagent** with complete task specifications
2. Implementer implements, tests, and self-reviews
3. **Spec compliance reviewer** validates adherence to requirements
4. **Code quality reviewer** assesses implementation standards
5. If issues found: implementer fixes, reviewers re-evaluate
6. Both reviews pass → mark task complete → next task

## Model Efficiency

Assign capabilities strategically:
- Mechanical, isolated tasks → faster/cheaper model
- Multi-file coordination → standard model
- Architectural decisions → most capable model

## Critical Rules

- **Never** start implementation on main/master without explicit user consent
- **Never** skip reviews or accept "close enough"
- **Never** proceed with unfixed issues from reviews
- **Always** follow: spec compliance review BEFORE code quality review

## Integration

- **Requires:** Written plan (from `writing-plans`)
- **Uses:** `using-git-worktrees` for isolated workspace
- **Uses:** `code-reviewer` agent for reviews
- **Ends with:** `finishing-a-development-branch`

## Verification

After all tasks complete:
```bash
pytest -v          # All tests pass
ruff check src/    # Linter clean
```
