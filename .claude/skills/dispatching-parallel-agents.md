---
name: dispatching-parallel-agents
description: Use when facing 2+ independent tasks that can be worked on without shared state or sequential dependencies
---

# Dispatching Parallel Agents

## Overview

Delegate tasks to specialized agents with isolated context. Each agent gets precisely crafted instructions — never your session's context or history.

**Core principle:** Dispatch one agent per independent problem domain. Let them work concurrently.

## When to Use

- 3+ test files failing with different root causes
- Multiple subsystems broken independently
- Each problem can be understood without context from others
- No shared state between investigations

## When NOT to Use

- Failures are related (fix one might fix others)
- Need to understand full system state first
- Agents would interfere with each other (shared files)
- Exploratory debugging with unclear root causes

## The Pattern

### 1. Identify Independent Domains

Group failures by what's broken:
- File A tests: Data loading pipeline
- File B tests: Model training logic
- File C tests: Calibration system

### 2. Create Focused Agent Tasks

Each agent receives:
- **Specific scope:** One test file or subsystem
- **Clear goal:** Make these tests pass
- **Constraints:** Don't change other code
- **Context:** Error messages, test names, relevant file paths
- **Expected output:** Summary of findings and fixes

### 3. Dispatch in Parallel

Use the Agent tool to launch multiple agents concurrently.

### 4. Review and Integrate

When agents return:
- Read each summary
- Verify fixes don't conflict
- Run full test suite: `pytest`
- Integrate all changes

## Agent Prompt Structure

Effective prompts are:
1. **Focused** — One clear problem domain
2. **Self-contained** — All necessary context included
3. **Specific about output** — What should be returned

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Too broad scope | Give each agent a narrow, focused assignment |
| No context in prompt | Include error messages and test names |
| No constraints | Specify what shouldn't be changed |
| Vague deliverables | Clarify what changes should be documented |
