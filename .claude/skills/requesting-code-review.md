---
name: requesting-code-review
description: Use when completing tasks, implementing major features, or before merging to verify work meets requirements
---

# Requesting Code Review

Dispatch the code-reviewer agent to catch issues before they cascade. The reviewer gets precisely crafted context — never your session's history.

**Core principle:** Review early, review often.

## When to Request Review

**Mandatory:**
- After each task in subagent-driven development
- After completing a major feature
- Before merge to main

**Optional but valuable:**
- When stuck (fresh perspective)
- Before refactoring (baseline check)
- After fixing a complex bug

## How to Request

### 1. Get git SHAs

```bash
BASE_SHA=$(git rev-parse HEAD~1)  # or origin/main
HEAD_SHA=$(git rev-parse HEAD)
```

### 2. Dispatch code-reviewer agent

Use the Agent tool with these details:
- **WHAT_WAS_IMPLEMENTED:** What you just built
- **PLAN_OR_REQUIREMENTS:** What it should do
- **BASE_SHA / HEAD_SHA:** Commit range
- **DESCRIPTION:** Brief summary

### 3. Act on feedback

- Fix **Critical** issues immediately
- Fix **Important** issues before proceeding
- Note **Minor** issues for later
- Push back if reviewer is wrong (with reasoning)

## Integration with Workflows

**Subagent-Driven Development:** Review after EACH task
**Executing Plans:** Review after each batch (3 tasks)
**Ad-Hoc Development:** Review before merge

## Red Flags

- Never skip review because "it's simple"
- Never ignore Critical issues
- Never proceed with unfixed Important issues
