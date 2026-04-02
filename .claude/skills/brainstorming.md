---
name: brainstorming
description: Use when starting a new feature, project, or significant change that needs design before implementation
---

# Brainstorming

## Overview

Transform ideas into approved designs before any implementation begins.

**Core principle:** Do NOT invoke any implementation skill, write any code, scaffold any project, or take any implementation action until you have presented a design and the user has approved it.

## The Hard Gate

No implementation until design is approved. Even "simple" projects require design review — they often harbor unexamined assumptions.

## The Process

1. **Explore context** — Review existing files, docs, and code patterns
2. **Ask clarifying questions** — One at a time, prefer multiple-choice
3. **Propose 2-3 approaches** — With documented trade-offs for each
4. **Present design progressively** — Section by section, get approval after each
5. **Write the spec** — Save to a document for reference
6. **Self-review the spec** — Check for placeholders, contradictions, ambiguity
7. **User review** — Get explicit approval before proceeding
8. **Hand off to planning** — Invoke `writing-plans` skill

## Design Principles

- **Unit isolation:** Break systems into focused components with clear interfaces
- **YAGNI:** Remove unnecessary features ruthlessly
- **Avoid false simplicity:** "Simple" projects still need design review
- **Conversational refinement:** One question per message, multiple-choice preferred

## What NOT to Do

- Don't start coding during brainstorming
- Don't skip to implementation because "it's obvious"
- Don't combine brainstorming with execution
- Don't present the entire design at once — break it into digestible sections
