# LLM Council System

The llm-council tool implements a structured multi-perspective decision framework.

## Core Mechanism

The system deploys five independent advisors with distinct thinking lenses: **The Contrarian** (identifies fatal flaws), **First Principles Thinker** (questions assumptions), **Expansionist** (seeks hidden upside), **Outsider** (provides fresh perspective), and **Executor** (focuses on implementation feasibility).

## When to Activate

Council triggering occurs with phrases like "council this," "should I X or Y," or "pressure-test this"—but only for decisions with genuine stakes and multiple options. Simple factual questions or one-right-answer scenarios don't warrant council convening.

## Process Flow

**Step 1:** Frame the question using workspace context (relevant files, business details, past decisions) to enrich advisor inputs beyond surface-level queries.

**Step 2:** Spawn all five advisors simultaneously, each providing 150-300 word independent analyses without hedging or balance-seeking.

**Step 3:** Anonymize responses and have advisors peer-review each other, identifying strongest arguments, blind spots, and overlooked considerations.

**Step 4:** A chairman synthesizes findings into structured output: areas of convergence, genuine disagreements, emergent blind spots, actionable recommendation, and single next step.

**Step 5-6:** Generate visual HTML report and full markdown transcript.

## Key Design Principles

Parallel spawning prevents sequential bias. Anonymized peer review ensures merit-based evaluation. The chairman can override majority opinion if reasoning supports it. Trivial questions bypass the council entirely.
