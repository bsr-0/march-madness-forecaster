---
name: executing-plans
description: Use when you have a written implementation plan ready to execute, after planning is complete
---

# Executing Plans

## Overview

Structured three-step workflow for implementing pre-written plans with review checkpoints.

**Core principle:** Follow the plan exactly. Skip no verifications. Never start on main/master without explicit approval.

## Step 1: Load and Review

1. Read the plan file
2. Critically examine for gaps or concerns
3. Raise issues BEFORE starting — don't proceed with uncertainty

## Step 2: Execute Tasks

For each task in the plan:

1. Mark task as in-progress
2. Follow steps precisely as written
3. Run specified verifications after each step
4. Mark task complete only after all verifications pass

```bash
# Verification for this project:
pytest -v                    # Tests pass
ruff check src/             # Linter clean
```

### Critical Stopping Points

**STOP executing immediately when:**
- Hit a blocker (missing dependency, test fails unexpectedly, instruction unclear)
- Verification fails and you can't resolve it
- Plan step doesn't match current codebase state

Don't force through obstacles — request clarification.

## Step 3: Complete Development

After all tasks verified, use the `finishing-a-development-branch` skill to:
- Verify all tests pass
- Present integration options
- Clean up

## Integration

- **Requires:** A written plan (from `writing-plans` skill)
- **Uses:** `using-git-worktrees` for isolated workspace
- **Ends with:** `finishing-a-development-branch` for proper closure
