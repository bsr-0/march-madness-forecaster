---
name: writing-plans
description: Use when a task requires multiple steps, touches multiple files, or needs coordination - before any implementation begins
---

# Writing Plans

## Overview

Create comprehensive, bite-sized implementation plans for multi-step development tasks. Plans are guides for engineers with limited codebase context.

**Core principle:** Every step must be atomic (2-5 minutes), complete (actual code, exact paths), and verifiable (test commands with expected output).

## When to Use

- Multi-file changes
- New features requiring design
- Complex refactors
- Any task where "just start coding" would waste time

## Plan Header (Required)

Every plan starts with:
- **Goal:** What we're building and why
- **Architecture:** How components fit together
- **Tech Stack:** Python, pytest, ruff, pandas, scikit-learn, etc.

## File Structure First

Before tasks, map which files will be created or modified:
- Each file has one focused purpose
- Clear responsibility boundaries
- Note create vs modify

## Task Format

```markdown
### Task N: [Descriptive Name]

**Files:** `src/path/file.py` (modify), `tests/test_file.py` (create)

Steps:
- [ ] Write failing test in `tests/test_file.py`
- [ ] Run `pytest tests/test_file.py -v` — expect FAIL
- [ ] Implement in `src/path/file.py`
- [ ] Run `pytest tests/test_file.py -v` — expect PASS
- [ ] Run `pytest` — all tests pass
- [ ] `git commit -m "feat: descriptive message"`
```

## Critical Requirements

**No placeholder language:**
- NO "TBD", "add validation", "similar to Task N"
- Include actual code, exact file paths, complete commands
- Every step that modifies code includes the code

**Consistent typing:**
- Method names, signatures, properties must match across all tasks
- Inconsistencies are bugs — fix during self-review

## Self-Review

Before presenting the plan:
- [ ] Spec coverage: does every requirement have a task?
- [ ] Placeholder scan: any "TBD" or vague steps?
- [ ] Type consistency: do signatures match across tasks?

## Execution Handoff

After plan approval, offer:
1. **Subagent-driven:** Fresh agent per task with reviews (use `subagent-driven-development` skill)
2. **Inline execution:** Batched with checkpoints (use `executing-plans` skill)

## Plan Location

Save plans to: `docs/plans/YYYY-MM-DD-<feature-name>.md`
